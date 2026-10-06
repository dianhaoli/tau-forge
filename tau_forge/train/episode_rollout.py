"""Multi-turn episode rollouts by token APPEND: the torch-free core of
`grpo_episodes` (TRL `rollout_func`) and of the batched episode audit.

What it does. Given one `EpisodeTask` per generation, a tokenizer-like object
(`apply_chat_template`, `encode`, `decode`) and an engine callable

    engine(list[prompt_token_ids]) -> list[(token_ids, text, logprobs, finish_reason)]

it runs every episode to the end and returns, per episode, the token sequence
GRPO trains on:

  * `prompt_ids` = the chat template over system + tools + greeting + opening
    with the generation prompt (`<|im_start|>assistant\\n`), rendered ONCE;
  * `completion_ids` = for every assistant turn, the SAMPLED ids as the engine
    returned them (env_mask 1, the engine's logprobs), then the tokenized
    environment reply -- the newline the template puts after `<|im_end|>`,
    the tool results / user reply, and the next assistant header (env_mask 0,
    logprob 0.0). History is never re-rendered: re-rendering would re-tokenize
    the policy's own text (and re-serialize its tool calls), so the training
    forward pass would score different ids from the ones vLLM sampled and the
    importance-sampling ratio would be meaningless (wf1_train_eff.md, TRL
    gotcha 3);
  * each sequence ends right after the LAST SAMPLED token: the trailing user
    reply / tool result is dropped. A completed last turn therefore ends in
    `<|im_end|>` (= Qwen's eos), and an episode whose last turn hit the
    per-turn token cap ends mid-turn, so TRL's `mask_truncated_completions`
    drops it (`grpo_trainer.py:2506-2513` checks `ids[-1] in {eos, pad}`);
  * `episode_reward` = `score_episode(task, episode.result())`.

Scheduling. At most `max_inflight` episodes are active; every engine call
carries one request per active episode, and an episode that ends is replaced
from the queue before the next call (continuous batching across turns, at the
granularity of one engine call). The cap is what keeps the per-episode KV
history resident in vLLM's prefix cache between turns -- the 2,000-episode
lockstep audit re-prefilled ~17M tokens because 2,000 histories cannot fit in
a 75k-token cache (wf1_train_eff.md, bottleneck 1). With appended ids, every
request of an episode is a strict extension of its previous request, so the
cache hit is the whole previous turn.

Groups. Episodes with the same task id form a GRPO group (TRL repeats each
prompt `num_generations` times). A group whose rewards are all equal has zero
advantage under `scale_rewards="none"`; `mask_flat_groups=True` additionally
zeroes its env_mask so its tokens leave the DAPO token-count denominator (TRL
has no DAPO dynamic sampling).

The Qwen3-2507 delta renderer (`render_env_delta`) reproduces the chat
template's text for messages appended after an assistant turn: a `tool`
message is a `<tool_response>` block inside a USER turn, consecutive tool
messages share one user turn. `check_append_equals_rerender` is the GPU-box
check that appending the pieces tokenizes to exactly the ids of a full
re-render (`scripts/check_token_append.py`).
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import Episode
from tau_forge.episodes.task import EpisodeTask

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# (sampled token ids, decoded text, per-token logprobs or None, finish reason)
EngineResult = tuple[list[int], str, Optional[list[Optional[float]]], str]
Engine = Callable[[list[list[int]]], list[EngineResult]]
UserSeeds = Union[None, Sequence[int], Callable[[int, EpisodeTask], int]]


# --------------------------------------------------------------- rendering


def _content(message: dict[str, Any]) -> str:
    c = message.get("content")
    return c if isinstance(c, str) else ""


def render_tool_call(tool_call: dict[str, Any]) -> str:
    """One `<tool_call>` block exactly as the Qwen3-2507 template writes it:
    arguments verbatim when they are a string (the runner stores
    `json.dumps(arguments)`), else HF's `tojson` (json.dumps, ensure_ascii
    off, keys unsorted)."""
    fn = tool_call.get("function", tool_call)
    args = fn.get("arguments")
    args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return '<tool_call>\n{"name": "' + str(fn["name"]) + '", "arguments": ' + args_text + "}\n</tool_call>"


def render_assistant_body(message: dict[str, Any]) -> str:
    """The text the template puts between `<|im_start|>assistant\\n` and
    `<|im_end|>`: content, then each tool call (a newline before the first
    only when there is content). This is the canonical re-render -- what a
    tau2 eval feeds back as history -- not necessarily the raw sampled text."""
    content = _content(message)
    out = content
    for k, tc in enumerate(message.get("tool_calls") or []):
        if k > 0 or content:
            out += "\n"
        out += render_tool_call(tc)
    return out


def render_env_delta(messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool = True) -> str:
    """Template text for `messages` appended right after an assistant turn
    whose `<|im_end|>` is already in the token stream.

    Starts with the "\\n" the template emits after every assistant
    `<|im_end|>`. `tool` messages render as `<tool_response>` blocks inside a
    `<|im_start|>user` turn, consecutive ones sharing that turn; user (and
    non-first system) messages render as their own turn. Ends with the next
    assistant header when `add_generation_prompt`."""
    out = "\n"
    prev = "assistant"
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "tool":
            if prev != "tool":
                out += IM_START + "user"
            out += "\n<tool_response>\n" + _content(m) + "\n</tool_response>"
            nxt = messages[i + 1]["role"] if i + 1 < len(messages) else None
            if nxt != "tool":
                out += IM_END + "\n"
        elif role in ("user", "system"):
            out += IM_START + role + "\n" + _content(m) + IM_END + "\n"
        elif role == "assistant":
            out += IM_START + "assistant\n" + render_assistant_body(m) + IM_END + "\n"
        else:
            raise ValueError(f"cannot render role {role!r}")
        prev = role
    if add_generation_prompt:
        out += IM_START + "assistant\n"
    return out


def encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids of `text` with no BOS/EOS added. Special-token strings in
    `text` (`<|im_start|>`, `<|im_end|>`) map to their single ids, as HF
    tokenizers do by default."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def special_token_id(tokenizer: Any, token: str) -> int:
    conv = getattr(tokenizer, "convert_tokens_to_ids", None)
    if conv is not None:
        tid = conv(token)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
            return tid
    ids = encode(tokenizer, token)
    if len(ids) != 1:
        raise ValueError(f"{token!r} is not a single token for this tokenizer: {ids}")
    return ids[0]


def render_prompt(tokenizer: Any, messages: Sequence[dict[str, Any]], tools: Optional[list] = None) -> str:
    return tokenizer.apply_chat_template(
        [dict(m) for m in messages], tools=tools or None, tokenize=False, add_generation_prompt=True
    )


def _first_user_index(messages: Sequence[dict[str, Any]]) -> int:
    return next(i for i, m in enumerate(messages) if m["role"] == "user")


def check_append_equals_rerender(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Optional[list] = None,
    n_prompt_messages: Optional[int] = None,
) -> dict[str, Any]:
    """Does appending pieces give the ids of a full chat-template re-render?

    Pieces, in rollout order: the prompt (`messages[:n_prompt_messages]`,
    default through the first user message, with the generation prompt); per
    assistant message its canonical body (`render_assistant_body`) then a
    single `<|im_end|>` id; per run of environment messages the
    `render_env_delta` text. The reference is `apply_chat_template` over all
    of `messages` -- with the generation prompt when the conversation ends on
    an environment message, minus the template's trailing "\\n" when it ends
    on an assistant message (rollout sequences end at `<|im_end|>`).

    Why it matters: rollout trains on appended ids, while a tau2 eval
    re-renders the whole conversation and tokenizes it at once. A merge
    across a piece boundary would mean the policy is trained on token
    sequences it never sees at eval. (The assistant bodies at train time are
    the model's own samples, which this check approximates by the canonical
    body tokenized on its own.)"""
    n = _first_user_index(messages) + 1 if n_prompt_messages is None else n_prompt_messages
    im_end = special_token_id(tokenizer, IM_END)
    pieces: list[tuple[str, str]] = [("prompt", render_prompt(tokenizer, messages[:n], tools))]
    kinds: Counter = Counter()
    i = n
    while i < len(messages):
        m = messages[i]
        if m["role"] != "assistant":
            raise ValueError(f"message {i} should be an assistant turn, is {m['role']!r}")
        pieces.append(("assistant", render_assistant_body(m)))
        pieces.append(("im_end", IM_END))
        kinds["assistant_tool_call" if m.get("tool_calls") else "assistant_text"] += 1
        i += 1
        j = i
        while j < len(messages) and messages[j]["role"] != "assistant":
            j += 1
        env = list(messages[i:j])
        if env:
            pieces.append(("env", render_env_delta(env, add_generation_prompt=True)))
            n_tools = sum(1 for e in env if e["role"] == "tool")
            kinds["tool_multi" if n_tools > 1 else "tool" if n_tools else env[0]["role"]] += 1
        i = j
    ends_on_env = messages[-1]["role"] != "assistant" or len(messages) == n
    full = tokenizer.apply_chat_template(
        [dict(m) for m in messages], tools=tools or None, tokenize=False, add_generation_prompt=ends_on_env
    )
    if not ends_on_env:
        if not full.endswith(IM_END + "\n"):
            raise ValueError("template does not end an assistant turn with '<|im_end|>\\n'")
        full = full[:-1]
    appended_text = "".join(text for _, text in pieces)
    appended_ids: list[int] = []
    for kind, text in pieces:
        appended_ids += [im_end] if kind == "im_end" else encode(tokenizer, text)
    full_ids = encode(tokenizer, full)
    report: dict[str, Any] = {
        "ok": appended_text == full and appended_ids == full_ids,
        "text_equal": appended_text == full,
        "ids_equal": appended_ids == full_ids,
        "n_ids": len(full_ids),
        "n_pieces": len(pieces),
        "kinds": dict(kinds),
    }
    if not report["ids_equal"]:
        k = next((x for x, (a, b) in enumerate(zip(appended_ids, full_ids)) if a != b), min(len(appended_ids), len(full_ids)))
        report["first_mismatch"] = {
            "index": k,
            "appended": appended_ids[max(0, k - 4): k + 4],
            "rerender": full_ids[max(0, k - 4): k + 4],
            "context": tokenizer.decode(full_ids[max(0, k - 12): k + 12]),
        }
    if not report["text_equal"]:
        k = next((x for x, (a, b) in enumerate(zip(appended_text, full)) if a != b), min(len(appended_text), len(full)))
        report["first_text_mismatch"] = {"index": k, "appended": appended_text[k - 40: k + 40], "rerender": full[k - 40: k + 40]}
    return report


# ------------------------------------------------------------ token traces


@dataclass
class TokenTrace:
    """One episode's token stream as the policy sees it."""

    prompt_ids: list[int]
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[Optional[float]] = field(default_factory=list)
    env_mask: list[int] = field(default_factory=list)
    consumed: int = 0  # episode messages already in the stream
    sampled_end: int = 0  # len(completion_ids) right after the last sampled turn
    n_turns: int = 0
    n_sampled: int = 0
    n_env: int = 0
    n_appended_im_end: int = 0

    def request_ids(self) -> list[int]:
        return self.prompt_ids + self.completion_ids

    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)


class TokenAppender:
    """Builds and extends `TokenTrace`s from an `Episode`'s message list.

    `start` renders the opening prompt; after each `Episode.step`, `extend`
    appends the environment's reply (everything the step added after the
    assistant message) as a `render_env_delta`; `record` appends what the
    engine sampled. `fits` is the context budget: an episode whose next
    request plus `max_new_tokens` exceeds `max_model_len` must end with
    finish reason "context" (vLLM would reject the request)."""

    def __init__(self, tokenizer: Any, tools: Optional[list], *, max_new_tokens: int, max_model_len: int):
        self.tokenizer = tokenizer
        self.tools = tools
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.im_end_id = special_token_id(tokenizer, IM_END)

    def start(self, messages: Sequence[dict[str, Any]]) -> TokenTrace:
        ids = encode(self.tokenizer, render_prompt(self.tokenizer, messages, self.tools))
        return TokenTrace(prompt_ids=ids, consumed=len(messages))

    def extend(self, trace: TokenTrace, messages: Sequence[dict[str, Any]]) -> None:
        new = list(messages[trace.consumed:])
        if not new:
            return
        if trace.n_turns == 0:
            raise ValueError("messages were added before the first sampled turn")
        if new[0]["role"] != "assistant":
            raise ValueError(f"expected the sampled assistant message first, got {new[0]['role']!r}")
        env = new[1:]
        if any(m["role"] == "assistant" for m in env):
            raise ValueError("more than one assistant message per sampled turn")
        ids = encode(self.tokenizer, render_env_delta(env, add_generation_prompt=True))
        trace.completion_ids += ids
        trace.logprobs += [0.0] * len(ids)
        trace.env_mask += [0] * len(ids)
        trace.n_env += len(ids)
        trace.consumed = len(messages)

    def fits(self, trace: TokenTrace) -> bool:
        return trace.total_len() + self.max_new_tokens <= self.max_model_len

    def record(
        self,
        trace: TokenTrace,
        token_ids: Sequence[int],
        logprobs: Optional[Sequence[Optional[float]]],
        finish_reason: str,
        *,
        require_logprobs: bool = False,
    ) -> None:
        ids = list(token_ids)
        if logprobs is None:
            if require_logprobs:
                raise ValueError("engine returned no logprobs; set SamplingParams(logprobs=0)")
            lps: list[Optional[float]] = [0.0] * len(ids)
        else:
            lps = list(logprobs)
            if len(lps) != len(ids):
                raise ValueError(f"engine returned {len(lps)} logprobs for {len(ids)} tokens")
        trace.completion_ids += ids
        trace.logprobs += lps
        trace.env_mask += [1] * len(ids)
        trace.n_sampled += len(ids)
        if finish_reason == "stop" and (not ids or ids[-1] != self.im_end_id):
            # Stopped on another stop token (or the engine dropped the eos): close
            # the turn so the template's structure -- and TRL's eos check -- hold.
            trace.completion_ids.append(self.im_end_id)
            trace.logprobs.append(0.0)
            trace.env_mask.append(0)
            trace.n_appended_im_end += 1
        trace.sampled_end = len(trace.completion_ids)
        trace.n_turns += 1

    def finalize(self, trace: TokenTrace) -> tuple[list[int], list[Optional[float]], list[int]]:
        """(completion_ids, logprobs, env_mask) cut after the last sampled
        turn. An episode that never sampled (its first prompt overflowed) gets
        a lone `<|im_end|>` with env_mask 0: TRL indexes `ids[-1]`."""
        cut = trace.sampled_end
        if cut == 0:
            return [self.im_end_id], [0.0], [0]
        return trace.completion_ids[:cut], trace.logprobs[:cut], trace.env_mask[:cut]


# --------------------------------------------------------------- scheduler


def drive_episodes(
    n: int,
    make_episode: Callable[[int], Episode],
    generate_batch: Callable[[list[tuple[int, Episode]]], list[tuple[str, str]]],
    *,
    max_inflight: Optional[int] = None,
    on_done: Optional[Callable[[int, Episode], None]] = None,
    on_round: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[Episode], int]:
    """Run episodes 0..n-1 to the end with at most `max_inflight` active.

    Each round: admit queued episodes (in index order) up to the cap, ask
    `generate_batch` for one (text, finish_reason) per active episode, step
    each, then retire the finished ones (`on_done`, in index order). An
    exception escaping `Episode.step` -- a harness bug, never a policy error
    -- ends that episode alone (`Episode.abort`). Admission order and results
    are deterministic for a deterministic `generate_batch`, whatever the cap.
    Returns (episodes by index, number of rounds)."""
    if max_inflight is not None and max_inflight < 1:
        raise ValueError("max_inflight must be >= 1 (or None for no cap)")
    queue = deque(range(n))
    episodes: dict[int, Episode] = {}
    active: list[int] = []
    rounds = 0
    while queue or active:
        while queue and (max_inflight is None or len(active) < max_inflight):
            i = queue.popleft()
            episodes[i] = make_episode(i)
            active.append(i)
        batch = [(i, episodes[i]) for i in active]
        outs = generate_batch(batch)
        if len(outs) != len(batch):
            raise RuntimeError(f"generator returned {len(outs)} turns for {len(batch)} requests")
        for (i, ep), (text, finish_reason) in zip(batch, outs):
            if ep.done:
                continue
            try:
                ep.step(text, finish_reason)
            except Exception as e:  # noqa: BLE001 -- isolate the failure to this episode
                ep.abort(f"{type(e).__name__}: {e}")
        rounds += 1
        if on_round:
            on_round(rounds, len(active))
        still = []
        for i in active:
            if episodes[i].done:
                if on_done:
                    on_done(i, episodes[i])
            else:
                still.append(i)
        active = still
    return [episodes[i] for i in range(n)], rounds


def _seed_for(user_seeds: UserSeeds, i: int, task: EpisodeTask) -> int:
    if user_seeds is None:
        return 0
    if callable(user_seeds):
        return int(user_seeds(i, task))
    return int(user_seeds[i])


def _flat(rewards: list[float], tol: float) -> bool:
    return max(rewards) - min(rewards) <= tol


def rollout_episodes(
    tasks: Sequence[EpisodeTask],
    tokenizer: Any,
    engine: Engine,
    *,
    tools: Optional[list] = None,
    system_message: Optional[dict[str, str]] = None,
    max_inflight: Optional[int] = 64,
    max_turns: int = 20,
    max_calls: int = 20,
    max_new_tokens: int = 512,
    max_model_len: int = 16384,
    user_seeds: UserSeeds = None,
    mask_flat_groups: bool = False,
    group_size: Optional[int] = None,
    flat_tol: float = 1e-9,
    require_logprobs: bool = False,
    keep_episodes: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Run one episode per entry of `tasks` (repeat a task G times for a GRPO
    group) and return the TRL `rollout_func` payload.

    Returns a dict with per-episode lists `prompt_ids`, `completion_ids`,
    `logprobs`, `env_mask`, `episode_reward` (None for an episode the harness
    crashed on: TRL treats it as unscorable, and its env_mask is zeroed),
    `episode_success`, `end_reason`, `task_id`, plus `stats` (a dict) and,
    with `keep_episodes`, `episodes` (the finished `Episode` objects).

    `user_seeds`: None (every episode seed 0, as the audit), a sequence, or
    `fn(index, task)`. Give a whole group one seed so its spread is the
    policy's alone. `group_size`: groups of any other size (a group split
    across ranks) are never called flat. `require_logprobs`: raise if the
    engine returns none (training needs them for the IS correction)."""
    t0 = time.time()
    appender = TokenAppender(tokenizer, tools, max_new_tokens=max_new_tokens, max_model_len=max_model_len)
    traces: dict[int, TokenTrace] = {}
    engine_seconds = 0.0
    n_engine_calls = 0
    max_batch = 0

    def make_episode(i: int) -> Episode:
        ep = Episode(
            tasks[i],
            system_message=system_message,
            user_seed=_seed_for(user_seeds, i, tasks[i]),
            max_turns=max_turns,
            max_calls=max_calls,
        )
        traces[i] = appender.start(ep.messages)
        return ep

    def generate_batch(batch: list[tuple[int, Episode]]) -> list[tuple[str, str]]:
        nonlocal engine_seconds, n_engine_calls, max_batch
        out: list[Optional[tuple[str, str]]] = [None] * len(batch)
        send: list[int] = []
        for k, (i, ep) in enumerate(batch):
            appender.extend(traces[i], ep.messages)
            if appender.fits(traces[i]):
                send.append(k)
            else:
                out[k] = ("", "context")
        if send:
            t = time.time()
            results = engine([traces[batch[k][0]].request_ids() for k in send])
            engine_seconds += time.time() - t
            n_engine_calls += 1
            max_batch = max(max_batch, len(send))
            if len(results) != len(send):
                raise RuntimeError(f"engine returned {len(results)} results for {len(send)} prompts")
            for k, (ids, text, lps, finish) in zip(send, results):
                appender.record(traces[batch[k][0]], ids, lps, finish, require_logprobs=require_logprobs)
                out[k] = (text, finish if finish in ("stop", "length") else "length")
        return out  # type: ignore[return-value]

    episodes, rounds = drive_episodes(
        len(tasks), make_episode, generate_batch, max_inflight=max_inflight, on_round=progress
    )

    prompt_ids, completion_ids, logprobs, env_mask = [], [], [], []
    rewards: list[Optional[float]] = []
    successes: list[bool] = []
    for i, ep in enumerate(episodes):
        c, lp, m = appender.finalize(traces[i])
        prompt_ids.append(traces[i].prompt_ids)
        completion_ids.append(c)
        logprobs.append(lp)
        env_mask.append(m)
        scored = score_episode(tasks[i], ep.result())
        crashed = ep.log.runner_error is not None
        rewards.append(None if crashed else scored.reward)
        successes.append(bool(scored.success) and not crashed)
        if crashed:
            env_mask[i] = [0] * len(m)

    # Groups: same task id, in first-appearance order.
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(t.id, []).append(i)
    complete = [g for g in groups.values() if len(g) >= 2 and (group_size is None or len(g) == group_size)]
    flat_groups = []
    for g in complete:
        rs = [rewards[i] for i in g if rewards[i] is not None]
        if len(rs) == len(g) and _flat(rs, flat_tol):
            flat_groups.append(g)
    n_masked_flat = 0
    if mask_flat_groups:
        for g in flat_groups:
            for i in g:
                env_mask[i] = [0] * len(env_mask[i])
                n_masked_flat += 1

    end_reasons = Counter(ep.end_reason for ep in episodes)
    scored_rewards = [r for r in rewards if r is not None]
    n = max(1, len(episodes))
    stats: dict[str, Any] = {
        "n_episodes": len(episodes),
        "n_groups": len(groups),
        "n_complete_groups": len(complete),
        "n_incomplete_groups": len(groups) - len(complete),
        "n_flat_groups": len(flat_groups),
        "flat_group_share": round(len(flat_groups) / len(complete), 4) if complete else 0.0,
        "n_masked_flat_episodes": n_masked_flat,
        "mean_reward": round(sum(scored_rewards) / len(scored_rewards), 4) if scored_rewards else 0.0,
        "success_rate": round(sum(successes) / n, 4),
        "end_reasons": dict(end_reasons),
        "n_truncated": end_reasons.get("truncated", 0),
        "n_context_budget": end_reasons.get("context_budget", 0),
        "n_runner_errors": end_reasons.get("runner_error", 0),
        "n_rounds": rounds,
        "n_engine_calls": n_engine_calls,
        "max_batch": max_batch,
        "mean_turns": round(sum(traces[i].n_turns for i in traces) / n, 3),
        "mean_sampled_tokens": round(sum(traces[i].n_sampled for i in traces) / n, 1),
        "mean_env_tokens": round(sum(traces[i].n_env for i in traces) / n, 1),
        "mean_prompt_tokens": round(sum(len(p) for p in prompt_ids) / n, 1),
        "mean_completion_tokens": round(sum(len(c) for c in completion_ids) / n, 1),
        "max_sequence_tokens": max((len(p) + len(c) for p, c in zip(prompt_ids, completion_ids)), default=0),
        "n_appended_im_end": sum(traces[i].n_appended_im_end for i in traces),
        "seconds_engine": round(engine_seconds, 3),
        "seconds_total": round(time.time() - t0, 3),
    }
    out: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "episode_reward": rewards,
        "episode_success": successes,
        "end_reason": [ep.end_reason for ep in episodes],
        "task_id": [t.id for t in tasks],
        "stats": stats,
    }
    if keep_episodes:
        out["episodes"] = episodes
    return out


def score_tasks(
    tasks: Sequence[EpisodeTask], tokenizer: Any, engine: Engine, **rollout_kwargs: Any
) -> dict[str, Any]:
    """Synthetic-val scoring: one episode per task through the same rollout
    core, summarized as mean reward / success overall and per template."""
    res = rollout_episodes(tasks, tokenizer, engine, **rollout_kwargs)
    return summarize_scores(
        [(t.template, r, s) for t, r, s in zip(tasks, res["episode_reward"], res["episode_success"])]
    ) | {"stats": res["stats"]}


def summarize_scores(rows: Iterable[tuple[str, Optional[float], bool]]) -> dict[str, Any]:
    by_t: dict[str, list[tuple[float, bool]]] = {}
    allr: list[tuple[float, bool]] = []
    for template, r, s in rows:
        if r is None:
            continue
        by_t.setdefault(template, []).append((r, s))
        allr.append((r, s))

    def agg(xs: list[tuple[float, bool]]) -> dict[str, float]:
        if not xs:
            return {"n": 0, "mean_reward": math.nan, "success_rate": math.nan}
        return {
            "n": len(xs),
            "mean_reward": round(sum(r for r, _ in xs) / len(xs), 4),
            "success_rate": round(sum(1 for _, s in xs if s) / len(xs), 4),
        }

    return {"overall": agg(allr), "per_template": {t: agg(v) for t, v in sorted(by_t.items())}}


__all__ = [
    "Engine",
    "EngineResult",
    "IM_END",
    "IM_START",
    "TokenAppender",
    "TokenTrace",
    "check_append_equals_rerender",
    "drive_episodes",
    "encode",
    "render_assistant_body",
    "render_env_delta",
    "render_prompt",
    "render_tool_call",
    "rollout_episodes",
    "score_tasks",
    "special_token_id",
    "summarize_scores",
]
