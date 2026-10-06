"""Multi-turn GRPO rollout core (`tau_forge.train.episode_rollout`), its TRL
wiring (`tau_forge.train.grpo_episodes`), the task-pool prefilter, the
in-flight-capped audit and the GPU-box check scripts -- all on CPU, no torch.

The fake tokenizer is character-level but renders with the REAL
Qwen3-Instruct-2507 chat template (copied verbatim from TRL 1.12.0's
`trl/chat_templates/qwen3_instruct_2507.jinja`) through jinja2 with HF's
`tojson`, so the delta renderer is checked against the template itself. The
fake engine sees only token ids: it decodes the prompt, parses the rendered
conversation back into messages and answers with a `ReferenceAgent` -- so the
token stream provably carries everything the policy needs.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jinja2
import pytest

from tau_forge.decontam.real_tasks import RealTaskExclusions
from tau_forge.envs.retail import RetailEnv
from tau_forge.episodes.audit import Generation, TokenAppendPolicy, run_audit
from tau_forge.episodes.generate import generate_tasks, write_jsonl
from tau_forge.episodes.reference_agents import ReferenceAgent, call
from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import Episode, run_episode
from tau_forge.train import grpo_episodes
from tau_forge.train.episode_rollout import (
    IM_END,
    TokenAppender,
    check_append_equals_rerender,
    drive_episodes,
    render_assistant_body,
    render_env_delta,
    rollout_episodes,
    score_tasks,
)
from tau_forge.train.task_pool import classify_record, prefilter_tasks

REPO_ROOT = Path(__file__).resolve().parents[1]
EMPTY = RealTaskExclusions(frozenset(), frozenset())
SYS = {"role": "system", "content": "You are a retail agent. Follow the policy."}

QWEN3_INSTRUCT_2507 = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
MARK_BASE = 50  # ids 50.. are invisible "mode marker" tokens the fake engine samples
CHAR_BASE = 1000


class CharTokenizer:
    """Character-level tokenizer + the real Qwen3-2507 template. `merges`
    adds multi-character tokens (greedy longest match) to model a BPE whose
    merges can straddle a piece boundary."""

    def __init__(self, merges: tuple[str, ...] = ()):
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)
        self.template = env.from_string(QWEN3_INSTRUCT_2507)
        self.merges = sorted(merges, key=len, reverse=True)
        self.merge_ids = {m: 500 + k for k, m in enumerate(self.merges)}
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = None

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)

    def convert_tokens_to_ids(self, token):
        return SPECIALS.index(token)

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            sp = next((s for s in SPECIALS if text.startswith(s, i)), None)
            if sp:
                ids.append(SPECIALS.index(sp))
                i += len(sp)
                continue
            m = next((m for m in self.merges if text.startswith(m, i)), None)
            if m:
                ids.append(self.merge_ids[m])
                i += len(m)
                continue
            ids.append(CHAR_BASE + ord(text[i]))
            i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.merge_ids.items()}
        out = []
        for t in ids:
            if t < len(SPECIALS):
                out.append("" if skip_special_tokens else SPECIALS[t])
            elif MARK_BASE <= t < MARK_BASE + 50:
                continue
            elif t in inv:
                out.append(inv[t])
            else:
                out.append(chr(t - CHAR_BASE))
        return "".join(out)


_CALL_RE = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
_RESP_RE = re.compile(r"\n<tool_response>\n(.*?)\n</tool_response>", re.DOTALL)


def parse_rendered(text: str) -> list[dict[str, Any]]:
    """The rendered conversation back into runner-shaped messages (system
    block dropped)."""
    msgs: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    n_calls = 0
    for chunk in text.split("<|im_start|>")[1:]:
        if "<|im_end|>" not in chunk:  # the trailing generation prompt
            continue
        role, _, body = chunk.partition("\n")
        body = body.split("<|im_end|>")[0]
        if role == "system":
            continue
        if role == "assistant":
            calls = _CALL_RE.findall(body)
            m: dict[str, Any] = {"role": "assistant", "content": body.split("<tool_call>")[0].strip() if calls else body}
            if calls:
                m["tool_calls"] = []
                for c in calls:
                    n_calls += 1
                    d = json.loads(c)
                    tc = {"id": f"c{n_calls}", "type": "function",
                          "function": {"name": d["name"], "arguments": json.dumps(d["arguments"])}}
                    m["tool_calls"].append(tc)
                pending = list(m["tool_calls"])
            msgs.append(m)
        elif chunk.startswith("user\n<tool_response>"):
            for content in _RESP_RE.findall(chunk[4:].split("<|im_end|>")[0]):
                tc = pending.pop(0)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"], "content": content})
        else:
            msgs.append({"role": "user", "content": body})
    return msgs


class AgentEngine:
    """Fake vLLM over token ids. The k-th first-turn request for a given
    prompt gets mode `modes[k % len(modes)]`, recorded as an invisible marker
    token at the start of its first sample; later turns read the marker back.
    The agent's state is rebuilt by replaying every earlier prefix."""

    def __init__(self, tok: CharTokenizer, tasks, modes, *, length_at_turn=None, drop_eos=False, logprob=-0.25):
        self.tok, self.modes = tok, list(modes)
        self.by_opening = {t.opening: t for t in tasks}
        self.first_seen: Counter = Counter()
        self.batches: list[int] = []
        self.calls: list[list[list[int]]] = []
        self.length_at_turn = length_at_turn
        self.drop_eos = drop_eos
        self.logprob = logprob

    def __call__(self, prompts):
        self.batches.append(len(prompts))
        self.calls.append([list(p) for p in prompts])
        out = []
        for ids in prompts:
            marks = [t for t in ids if MARK_BASE <= t < MARK_BASE + 50]
            if marks:
                mode, prefix = self.modes[marks[0] - MARK_BASE], []
            else:
                k = self.first_seen[tuple(ids)]
                self.first_seen[tuple(ids)] += 1
                idx = k % len(self.modes)
                mode, prefix = self.modes[idx], [MARK_BASE + idx]
            msgs = parse_rendered(self.tok.decode(ids))
            task = self.by_opening[next(m["content"] for m in msgs if m["role"] == "user")]
            agent = ReferenceAgent(task, mode)
            for j, m in enumerate(msgs):
                if m["role"] == "assistant" and j > 0:
                    agent(msgs[:j])
            text = agent(msgs)
            n_turn = sum(1 for m in msgs if m["role"] == "assistant")  # greeting counts as 1
            if self.length_at_turn is not None and n_turn == self.length_at_turn:
                gen = prefix + self.tok.encode(text[:5])
                out.append((gen, text[:5], [self.logprob] * len(gen), "length"))
                continue
            gen = prefix + self.tok.encode(text) + ([] if self.drop_eos else [self.tok.eos_token_id])
            out.append((gen, text, [self.logprob] * len(gen), "stop"))
        return out


@pytest.fixture(scope="module")
def tasks():
    rep = generate_tasks(3, 0, exclusions=EMPTY, log=lambda _: None)
    return rep.tasks


@pytest.fixture(scope="module")
def tok():
    return CharTokenizer()


def _pick(tasks, templates=("cancel", "exchange", "foreign_order_refusal", "modify_payment", "return_fallback"), k=1):
    out = []
    for t in templates:
        out += [x for x in tasks if x.template == t][:k]
    return out


def _rollout(tasks_rep, tok, engine, **kw):
    kw.setdefault("tools", None)
    kw.setdefault("system_message", SYS)
    kw.setdefault("max_model_len", 10**7)
    kw.setdefault("keep_episodes", True)
    return rollout_episodes(tasks_rep, tok, engine, **kw)


# ----------------------------------------------------------- delta renderer


def test_env_delta_matches_the_qwen3_template_for_every_message_type(tok):
    base = [SYS, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "help"}]
    a_text = {"role": "assistant", "content": "Your email?"}
    a_call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}},
        {"id": "2", "type": "function", "function": {"name": "g", "arguments": {"y": "é"}}}]}
    a_both = {"role": "assistant", "content": "Checking.", "tool_calls": a_call["tool_calls"][:1]}
    t1 = {"role": "tool", "tool_call_id": "1", "name": "f", "content": "{\"ok\": true}"}
    t2 = {"role": "tool", "tool_call_id": "2", "name": "g", "content": "Error: nope"}
    u = {"role": "user", "content": "\n leading newline and trailing space "}
    for asst, env in [(a_text, [u]), (a_call, [t1, t2]), (a_both, [t1]), (a_call, [t1, t2, u]), (a_text, [])]:
        for tools in (None, RetailEnv().all_openai_schemas()[:2]):
            before = tok.apply_chat_template(base + [asst], tools=tools)
            after = tok.apply_chat_template(base + [asst] + env, tools=tools, add_generation_prompt=True)
            assert before.endswith(render_assistant_body(asst) + IM_END + "\n")
            assert after == before[:-1] + render_env_delta(env)
            rep = check_append_equals_rerender(tok, base + [asst] + env, tools=tools)
            assert rep["ok"], rep


def test_check_append_detects_a_merge_across_a_piece_boundary():
    merging = CharTokenizer(merges=("\n\n",))
    msgs = [SYS, {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "x"},
            {"role": "assistant", "content": "\nLeading newline"}]
    rep = check_append_equals_rerender(merging, msgs)
    assert rep["text_equal"] and not rep["ids_equal"] and not rep["ok"]
    assert "first_mismatch" in rep
    ok = check_append_equals_rerender(merging, msgs[:3] + [{"role": "assistant", "content": "fine"}])
    assert ok["ok"]


# ------------------------------------------------------------ rollout core


def test_masks_cover_exactly_the_sampled_tokens_and_sequences_end_at_im_end(tasks, tok):
    picked = _pick(tasks)
    reps = [t for t in picked for _ in range(3)]
    eng = AgentEngine(tok, picked, ["oracle", "no_confirm", "transfer"])
    res = _rollout(reps, tok, eng)
    assert len(res["completion_ids"]) == len(reps)
    for i, ep in enumerate(res["episodes"]):
        c, m, lp = res["completion_ids"][i], res["env_mask"][i], res["logprobs"][i]
        assert len(c) == len(m) == len(lp)
        assert c[-1] == tok.eos_token_id and m[-1] == 1  # the sampled <|im_end|>
        # sampled tokens carry the engine's logprob, environment tokens 0.0
        assert all(x == -0.25 for x, k in zip(lp, m) if k == 1)
        assert all(x == 0.0 for x, k in zip(lp, m) if k == 0)
        # runs of sampled tokens == the assistant turns' raw text + <|im_end|>, in order
        runs, cur = [], []
        for t, k in zip(c, m):
            if k:
                cur.append(t)
            elif cur:
                runs.append(cur)
                cur = []
        runs.append(cur)
        policy_turns = [x for x in ep.messages[3:] if x["role"] == "assistant"]
        assert len(runs) == len(policy_turns) == ep.log.n_assistant_turns
        for run, msg in zip(runs, policy_turns):
            assert tok.decode(run) == msg.get("raw", render_assistant_body(msg)) + IM_END
        # appended ids == a full re-render of the conversation (reference agents emit canonical text)
        full = tok.apply_chat_template(ep.messages, tools=None)
        last_asst = max(j for j, x in enumerate(ep.messages) if x["role"] == "assistant")
        expect = tok.apply_chat_template(ep.messages[: last_asst + 1], tools=None)[:-1]
        assert tok.decode(res["prompt_ids"][i] + c) == expect and full.startswith(expect)


def test_rewards_equal_run_episode_plus_score_episode_on_the_same_completions(tasks, tok):
    picked = _pick(tasks, k=2)
    modes = ["oracle", "no_confirm", "transfer", "oracle"]
    reps = [t for t in picked for _ in range(len(modes))]
    seeds = [7 * (i // len(modes)) + 1 for i in range(len(reps))]
    res = _rollout(reps, tok, AgentEngine(tok, picked, modes), user_seeds=seeds, max_turns=20, max_calls=20)
    for i, t in enumerate(reps):
        mode = modes[i % len(modes)]
        ref = run_episode(t, ReferenceAgent(t, mode), system_message=SYS, user_seed=seeds[i], max_turns=20, max_calls=20)
        assert res["episode_reward"][i] == score_episode(t, ref).reward
        assert res["episode_success"][i] == score_episode(t, ref).success
        assert res["end_reason"][i] == ref.end_reason
        assert res["episodes"][i].messages == ref.messages
    assert res["stats"]["mean_reward"] == pytest.approx(sum(res["episode_reward"]) / len(reps), abs=1e-4)


def test_flat_groups_are_detected_and_optionally_masked(tasks, tok):
    cancel = [t for t in tasks if t.template == "cancel"][:2]
    # group 0: all transfer (flat 0.0); group 1: oracle/no_confirm/transfer (live)
    eng_modes = {cancel[0].opening: ["transfer"], cancel[1].opening: ["oracle", "no_confirm", "transfer"]}

    class PerTask(AgentEngine):
        def __call__(self, prompts):
            out = []
            for p in prompts:
                msgs = parse_rendered(self.tok.decode(p))
                opening = next(m["content"] for m in msgs if m["role"] == "user")
                self.modes = eng_modes[opening]
                out += super().__call__([p])
            return out

    reps = [cancel[0]] * 3 + [cancel[1]] * 3
    plain = _rollout(reps, tok, PerTask(tok, cancel, ["oracle"]), group_size=3)
    assert plain["stats"]["n_flat_groups"] == 1 and plain["stats"]["flat_group_share"] == 0.5
    assert plain["episode_reward"][:3] == [0.0, 0.0, 0.0] and len(set(plain["episode_reward"][3:])) >= 2
    assert all(any(m) for m in plain["env_mask"])
    masked = _rollout(reps, tok, PerTask(tok, cancel, ["oracle"]), group_size=3, mask_flat_groups=True)
    assert all(not any(m) for m in masked["env_mask"][:3]) and all(any(m) for m in masked["env_mask"][3:])
    assert masked["completion_ids"] == plain["completion_ids"] and masked["stats"]["n_masked_flat_episodes"] == 3
    # a group of the wrong size (split across ranks) is never called flat
    split = _rollout(reps[:2] + reps[3:], tok, PerTask(tok, cancel, ["oracle"]), group_size=3, mask_flat_groups=True)
    assert split["stats"]["n_flat_groups"] == 0 and split["stats"]["n_incomplete_groups"] == 1
    assert all(any(m) for m in split["env_mask"])


def test_inflight_cap_refills_and_results_do_not_depend_on_it(tasks, tok):
    picked = _pick(tasks)
    reps = [t for t in picked for _ in range(2)]
    modes = ["oracle", "no_confirm"]
    runs = {}
    for cap in (None, 1, 3):
        eng = AgentEngine(tok, picked, modes)
        res = _rollout(reps, tok, eng, max_inflight=cap)
        assert max(eng.batches) <= (cap or len(reps))
        runs[cap] = (res, eng)
    base = runs[None][0]
    for cap in (1, 3):
        res = runs[cap][0]
        for k in ("prompt_ids", "completion_ids", "logprobs", "env_mask", "episode_reward", "end_reason"):
            assert res[k] == base[k], k
    assert runs[None][1].batches[0] == len(reps)
    assert runs[1][1].batches == [1] * len(runs[1][1].batches)
    # refill: with cap 3, some call mixes a just-admitted episode (no marker yet) with older ones
    mixed = [c for c in runs[3][1].calls if len(c) > 1 and any(not any(MARK_BASE <= t < MARK_BASE + 50 for t in p) for p in c)
             and any(any(MARK_BASE <= t < MARK_BASE + 50 for t in p) for p in c)]
    assert mixed
    # every request extends that episode's previous request (prefix-cache friendly)
    assert runs[3][0]["stats"]["max_batch"] == 3


def test_requests_are_strict_extensions_of_the_previous_turn(tasks, tok):
    t = _pick(tasks, templates=("exchange",))[0]
    eng = AgentEngine(tok, [t], ["oracle"])
    res = _rollout([t], tok, eng)
    reqs = [c[0] for c in eng.calls]
    assert len(reqs) == res["episodes"][0].log.n_assistant_turns
    for a, b in zip(reqs, reqs[1:]):
        assert b[: len(a)] == a
    assert reqs[0] == res["prompt_ids"][0]


def test_truncated_last_turn_ends_mid_turn_and_context_overflow_ends_cleanly(tasks, tok):
    t = _pick(tasks, templates=("cancel",))[0]
    res = _rollout([t], tok, AgentEngine(tok, [t], ["oracle"], length_at_turn=3))
    assert res["end_reason"] == ["truncated"]
    assert res["completion_ids"][0][-1] != tok.eos_token_id  # TRL's mask_truncated_completions drops it
    assert res["env_mask"][0][-1] == 1
    # a prompt that cannot fit: one masked <|im_end|>, context_budget
    tiny = _rollout([t], tok, AgentEngine(tok, [t], ["oracle"]), max_model_len=50)
    assert tiny["end_reason"] == ["context_budget"]
    assert tiny["completion_ids"] == [[tok.eos_token_id]] and tiny["env_mask"] == [[0]]
    # budget hit mid-episode: the sequence still ends at the last sampled <|im_end|>
    first = _rollout([t], tok, AgentEngine(tok, [t], ["oracle"]))
    budget = len(first["prompt_ids"][0]) + 200
    mid = _rollout([t], tok, AgentEngine(tok, [t], ["oracle"]), max_model_len=budget, max_new_tokens=100)
    assert mid["end_reason"] == ["context_budget"] and mid["completion_ids"][0][-1] == tok.eos_token_id
    assert len(mid["prompt_ids"][0]) + len(mid["completion_ids"][0]) <= budget


def test_a_stop_without_im_end_gets_a_masked_im_end(tasks, tok):
    t = _pick(tasks, templates=("cancel",))[0]
    res = _rollout([t], tok, AgentEngine(tok, [t], ["oracle"], drop_eos=True))
    ref = _rollout([t], tok, AgentEngine(tok, [t], ["oracle"]))
    assert res["completion_ids"] == ref["completion_ids"]
    assert res["stats"]["n_appended_im_end"] == res["episodes"][0].log.n_assistant_turns
    assert res["env_mask"][0][-1] == 0 and ref["env_mask"][0][-1] == 1
    assert res["episode_reward"] == ref["episode_reward"]


def test_missing_logprobs_raise_when_required(tasks, tok):
    t = _pick(tasks, templates=("cancel",))[0]

    def engine(prompts):
        return [(tok.encode("hello") + [1], "hello", None, "stop") for _ in prompts]

    with pytest.raises(ValueError, match="logprobs"):
        _rollout([t], tok, engine, require_logprobs=True)
    assert _rollout([t], tok, engine)["logprobs"][0][0] == 0.0


def test_a_harness_crash_is_unscorable_and_masked(tasks, tok, monkeypatch):
    picked = _pick(tasks, templates=("cancel",), k=2)
    real = Episode.step

    def flaky(self, completion, finish_reason="stop"):
        if self.task.id == picked[0].id:
            raise KeyError("boom")
        return real(self, completion, finish_reason)

    monkeypatch.setattr(Episode, "step", flaky)
    res = _rollout(picked, tok, AgentEngine(tok, picked, ["oracle"]))
    assert res["episode_reward"][0] is None and res["end_reason"][0] == "runner_error"
    assert not any(res["env_mask"][0]) and res["episode_reward"][1] == 1.0
    assert res["stats"]["n_runner_errors"] == 1


def test_group_bookkeeping_with_interleaved_groups(tasks, tok):
    a, b = _pick(tasks, templates=("cancel", "modify_payment"))
    reps = [a, b, a, b]
    res = _rollout(reps, tok, AgentEngine(tok, [a, b], ["oracle"]), group_size=2)
    s = res["stats"]
    assert s["n_groups"] == 2 and s["n_complete_groups"] == 2 and s["n_flat_groups"] == 2
    assert res["task_id"] == [a.id, b.id, a.id, b.id]
    assert s["n_episodes"] == 4 and s["end_reasons"] == {"user_stop": 4}


def test_drive_episodes_rejects_a_bad_cap():
    with pytest.raises(ValueError):
        drive_episodes(1, lambda i: None, lambda b: [], max_inflight=0)


def test_score_tasks_summarizes_per_template(tasks, tok):
    picked = _pick(tasks)
    s = score_tasks(picked, tok, AgentEngine(tok, picked, ["oracle"]), system_message=SYS, max_model_len=10**7)
    assert s["overall"] == {"n": len(picked), "mean_reward": 1.0, "success_rate": 1.0}
    assert set(s["per_template"]) == {t.template for t in picked}


def test_real_system_prompt_and_tools_render_once(tasks, tok):
    from tau_forge.train.grpo_episodes import system_message

    t = _pick(tasks, templates=("foreign_order_refusal",))[0]
    tools = RetailEnv().all_openai_schemas()
    res = rollout_episodes([t, t], tok, AgentEngine(tok, [t], ["oracle", "transfer"]), tools=tools,
                           system_message=system_message(), max_model_len=10**7, group_size=2)
    assert len(tools) == 16
    assert res["prompt_ids"][0] == res["prompt_ids"][1]
    assert tok.decode(res["prompt_ids"][0]).count('{"type": "function"') == 16
    assert res["episode_reward"][0] == 1.0 and res["episode_reward"][1] == 0.3


# ----------------------------------------------------------------- audit


def test_audit_inflight_cap_gives_identical_results(tasks):
    picked = _pick(tasks)

    def scripted(modes):
        agents = {}

        def gen(requests):
            out = []
            for r in requests:
                ag = agents.setdefault(r.key, ReferenceAgent(r.task, modes[r.key[1] % len(modes)]))
                out.append(Generation(ag(r.messages)))
            sizes.append(len(requests))
            return out
        return gen

    sizes: list[int] = []
    base = run_audit(picked, scripted(["oracle", "no_confirm", "transfer"]), n_samples=3)
    assert max(sizes) == 15
    sizes.clear()
    done: list[str] = []
    capped = run_audit(picked, scripted(["oracle", "no_confirm", "transfer"]), n_samples=3, max_inflight=4,
                       on_task_done=lambda rec: done.append(rec["id"]))
    assert max(sizes) <= 4 and capped["n_turns"] > base["n_turns"]
    assert capped["per_task"] == base["per_task"] and capped["summary"] == base["summary"]
    assert sorted(done) == sorted(t.id for t in picked)


def test_token_append_policy_in_the_audit(tasks, tok):
    picked = _pick(tasks, k=1)
    eng = AgentEngine(tok, picked, ["oracle", "no_confirm"])
    policy = TokenAppendPolicy(lambda ps: [(i, x, None, f) for i, x, _, f in eng(ps)], tok, [], 50, 10**7)
    res = run_audit(picked, policy, n_samples=2, system_message=SYS, max_inflight=3)
    for t, rec in zip(picked, res["per_task"]):
        expect = [score_episode(t, run_episode(t, ReferenceAgent(t, m), system_message=SYS)).reward
                  for m in ("oracle", "no_confirm")]
        assert sorted(rec["rewards"]) == sorted(expect)
    assert policy.traces == {}  # released as episodes ended
    assert policy.max_prompt_tokens > 0 and policy.n_context_overflows == 0


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_episode_audit_script_token_mode_with_injected_engine(tmp_path, tasks, tok):
    picked = _pick(tasks, templates=("cancel", "foreign_order_refusal"))
    path = tmp_path / "eps.jsonl"
    write_jsonl(picked, path)
    out = tmp_path / "audit.json"

    def engine(prompts):
        return [(tok.encode(call("transfer_to_human_agents", {"summary": "s"})) + [1],
                 call("transfer_to_human_agents", {"summary": "s"}), None, "stop") for _ in prompts]

    _load("episode_audit").main(["--tasks", str(path), "--samples-per-task", "2", "--output", str(out),
                                 "--prompt-mode", "tokens", "--max-inflight", "3", "--max-model-len", "1000000"], tokenizer=tok, engine=engine)
    data = json.loads(out.read_text())
    assert data["config"]["prompt_mode"] == "tokens" and data["config"]["max_inflight"] == 3
    assert data["summary"]["overall"]["end_reasons"] == {"transfer": 4}
    assert set(data) == {"n_turns", "per_task", "summary", "config"}


# ------------------------------------------------------------ task pool


def test_classify_and_prefilter(tasks):
    assert classify_record({"rewards": [1.0, 1.0], "successes": [True, True]}) == "solved"
    assert classify_record({"rewards": [0, 0.0], "successes": [False, False]}) == "dead"
    assert classify_record({"rewards": [0.1, 0.1], "successes": [False, False]}) == "dead"
    assert classify_record({"rewards": [0.7, 0.7], "successes": [True, True]}) == "flat_other"
    assert classify_record({"rewards": [0.7, 1.0], "successes": [True, True]}) == "live"
    assert classify_record({"rewards": [1.0], "successes": [True]}) == "unjudged"
    cancel = [t for t in tasks if t.template == "cancel"]
    recs = []
    for k, t in enumerate(cancel):
        rewards = [[1.0] * 4, [0.0] * 4, [0.0, 1.0, 0.7, 0.0]][k % 3]
        recs.append({"id": t.id, "template": t.template, "rewards": rewards, "successes": [r > 0.5 for r in rewards]})
    audit = {"per_task": recs}
    kept, rep = prefilter_tasks(cancel, audit, keep_dead_fraction=0.0)
    row = rep["per_template"]["cancel"]
    assert row["solved"] == 1 and row["dead"] == 1 and row["live"] == 1
    assert [t.id for t in kept] == [cancel[2].id]
    kept_all, _ = prefilter_tasks(cancel, audit, keep_dead_fraction=1.0, keep_solved_fraction=1.0)
    assert len(kept_all) == 3
    # not in the audit: kept as unjudged
    kept2, rep2 = prefilter_tasks(tasks, audit, keep_dead_fraction=0.0)
    assert len(kept2) == len(tasks) - 2 and rep2["overall"]["unjudged"] == len(tasks) - 3


def test_prefilter_keeps_a_seeded_fifth_of_dead_tasks_per_template(tasks):
    fake = [SimpleNamespace(id=f"t{i}", template="cancel" if i < 50 else "exchange") for i in range(60)]
    audit = {"per_task": [{"id": t.id, "rewards": [0.0] * 8, "successes": [False] * 8} for t in fake]}
    kept, rep = prefilter_tasks(fake, audit)
    assert rep["per_template"]["cancel"]["kept_dead"] == 10 and rep["per_template"]["exchange"]["kept_dead"] == 2
    again, _ = prefilter_tasks(fake, audit)
    assert [t.id for t in kept] == [t.id for t in again]
    other, _ = prefilter_tasks(fake, audit, seed=1)
    assert [t.id for t in other] != [t.id for t in kept]


def test_prefilter_script_on_a_real_audit_shape(tmp_path, tasks):
    pool = _pick(tasks, k=3)
    path = tmp_path / "pool.jsonl"
    write_jsonl(pool, path)
    recs = [{"id": t.id, "template": t.template, "rewards": [1.0] * 8 if k % 2 else [0.0, 1.0] * 4,
             "successes": [True] * 8 if k % 2 else [False, True] * 4} for k, t in enumerate(pool)]
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"per_task": recs, "config": {"tasks": str(path)}}))
    out = tmp_path / "kept.jsonl"
    rep = _load("prefilter_tasks").main(["--audit", str(audit), "--out", str(out)])
    kept = [json.loads(x)["id"] for x in out.read_text().splitlines()]
    assert kept == [t.id for k, t in enumerate(pool) if k % 2 == 0]
    assert rep["overall"]["solved"] == sum(1 for k in range(len(pool)) if k % 2)


# -------------------------------------------------------------- GRPO wiring


def _args(*extra):
    return grpo_episodes.parse_args(["--num-processes", "2", *extra])


def test_grpo_config_defaults_follow_the_plan():
    a = _args()
    kw = grpo_episodes.build_config_kwargs(a)
    assert kw["num_generations"] == 8 and kw["gradient_accumulation_steps"] == 64 and kw["per_device_train_batch_size"] == 1
    assert kw["loss_type"] == "dapo" and kw["beta"] == 0.0 and kw["scale_rewards"] == "none"
    assert kw["epsilon"] == 0.2 and kw["epsilon_high"] == 0.28 and kw["learning_rate"] == 1e-6
    assert kw["mask_truncated_completions"] and kw["vllm_importance_sampling_correction"]
    assert kw["use_vllm"] and kw["vllm_mode"] == "colocate" and kw["vllm_enable_sleep_mode"] is False
    assert kw["bf16"] and kw["gradient_checkpointing"] and kw["max_completion_length"] == 512
    assert kw["top_k"] == 0 and kw["max_steps"] == 200 and kw["model_init_kwargs"] == {"dtype": "float32"}
    assert kw["weight_decay"] == 0.1 and not kw["use_liger_kernel"]
    assert a.max_turns == 20 and a.max_inflight == 64 and a.mask_flat_groups
    assert grpo_episodes.batch_layout(a)["whole_groups_per_rank"] == 1
    lora = grpo_episodes.build_config_kwargs(_args("--lora", "--liger", "--smoke"))
    assert lora["learning_rate"] == 1e-5 and lora["weight_decay"] == 0.0 and lora["use_liger_kernel"]
    assert lora["max_steps"] == 30 and lora["model_init_kwargs"] == {"dtype": "bfloat16"}
    assert grpo_episodes.lora_config_kwargs(_args("--lora"))["target_modules"] == "all-linear"
    with pytest.raises(ValueError):
        grpo_episodes.batch_layout(grpo_episodes.parse_args(["--num-processes", "3"]))
    assert grpo_episodes.batch_layout(grpo_episodes.parse_args(["--num-processes", "4", "--prompts-per-step", "2"]))[
        "whole_groups_per_rank"] == 0


def test_task_split_holds_out_val_before_the_prefilter(tmp_path, tasks):
    path = tmp_path / "pool.jsonl"
    write_jsonl(tasks, path)
    audit = tmp_path / "a.json"
    audit.write_text(json.dumps({"per_task": [{"id": t.id, "rewards": [1.0] * 8, "successes": [True] * 8} for t in tasks[:4]]}))
    a = grpo_episodes.parse_args(["--tasks", str(path), "--val-holdout", "3", "--prefilter-audit", str(audit)])
    train, val, rep = grpo_episodes.load_task_split(a, log=lambda _: None)
    assert len(val) == 3 and not ({t.id for t in val} & {t.id for t in train})
    assert not ({t.id for t in tasks[:4]} - {t.id for t in val}) & {t.id for t in train}
    assert rep["pool"] == len(tasks) and rep["train"] == len(train)
    rows = grpo_episodes.task_rows(train)
    assert json.loads(rows[0]["prompt"])["id"] == train[0].id


def test_group_seed_is_shared_within_a_step_and_changes_across_steps():
    assert grpo_episodes.group_seed("t1", 3) == grpo_episodes.group_seed("t1", 3)
    assert grpo_episodes.group_seed("t1", 3) != grpo_episodes.group_seed("t1", 4)
    assert grpo_episodes.group_seed("t1", 3) != grpo_episodes.group_seed("t2", 3)


class _FakeLLM:
    """vLLM `LLM.generate` shape: RequestOutput.outputs[0].{token_ids, text, logprobs, finish_reason}."""

    def __init__(self, engine):
        self.engine = engine
        self.params = []

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        self.params.append(sampling_params)
        res = self.engine([p["prompt_token_ids"] for p in prompts])
        return [SimpleNamespace(outputs=[SimpleNamespace(
            token_ids=ids, text=text, finish_reason=fr,
            logprobs=[{t: SimpleNamespace(logprob=lp, rank=1)} for t, lp in zip(ids, lps)])]) for ids, text, lps, fr in res]


def test_rollout_func_with_a_fake_trainer(tasks, tok):
    picked = _pick(tasks, templates=("cancel", "foreign_order_refusal"))
    prompts = [json.dumps(t.to_dict()) for t in picked for _ in range(2)]
    args = grpo_episodes.parse_args(["--num-generations", "2", "--max-model-len", str(10**7)])
    llm = _FakeLLM(AgentEngine(tok, picked, ["oracle", "transfer"]))
    trainer = SimpleNamespace(vllm_generation=SimpleNamespace(llm=llm), state=SimpleNamespace(global_step=5),
                              model=SimpleNamespace(training=True), _metrics={"train": defaultdict(list)})
    fn = grpo_episodes.make_rollout_func(args, tok, [], SYS, sampling_params="SP")
    out = fn(prompts, trainer)
    assert set(out) == set(grpo_episodes.TRL_KEYS)
    assert all(len(out[k]) == len(prompts) for k in out)
    assert all(isinstance(x, float) for lp in out["logprobs"] for x in lp)
    assert all(p == "SP" for p in llm.params)
    m = trainer._metrics["train"]
    assert m["episodes/n_episodes"] == [4.0] and "episodes/flat_group_share" in m and "episodes/end/user_stop" in m
    rewards = grpo_episodes.episode_reward_func(prompts=prompts, completions=[], episode_reward=out["episode_reward"])
    assert rewards == out["episode_reward"] and rewards[0] == 1.0 and rewards[1] == 0.0
    # same step -> the same user seed for every episode of a group, as rollout_episodes saw it
    assert grpo_episodes.episode_reward_func(episode_reward=[None, 0.5]) == [None, 0.5]


def test_vllm_results_maps_nan_logprobs_and_missing_finish():
    o = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5, 6], text="x", finish_reason=None,
                                                 logprobs=[{5: SimpleNamespace(logprob=-1.0, rank=1)},
                                                           {7: SimpleNamespace(logprob=float("nan"), rank=1)}])])
    assert grpo_episodes.vllm_results([o]) == [([5, 6], "x", [-1.0, None], "stop")]


def test_validation_scoring_without_accelerate(tasks, tok):
    picked = _pick(tasks)
    args = grpo_episodes.parse_args(["--max-model-len", str(10**7)])
    s = grpo_episodes.run_validation(SimpleNamespace(), picked, tok, AgentEngine(tok, picked, ["oracle"]), args, [], SYS)
    assert s["overall"]["mean_reward"] == 1.0 and s["overall"]["n"] == len(picked)


def test_trim_padding_columns_keeps_only_attended_columns():
    np = pytest.importorskip("numpy")
    # 2 rows: prompts left-padded to 5 (lengths 3, 4), completions right-padded to 6 (lengths 2, 4)
    pm = np.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    cm = np.array([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]])
    inputs = {
        "prompt_ids": np.arange(10).reshape(2, 5), "prompt_mask": pm,
        "completion_ids": np.arange(12).reshape(2, 6), "completion_mask": cm,
        "tool_mask": np.ones((2, 6)), "old_per_token_logps": np.zeros((2, 6)),
        "importance_sampling_ratio": np.ones((2, 1)), "advantages": np.array([0.5, -0.5]),
        "num_items_in_batch": np.array(6),
    }
    out = grpo_episodes.trim_padding_columns(inputs)
    assert out["prompt_ids"].shape == (2, 4) and out["prompt_mask"].tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert out["completion_ids"].shape == (2, 4) and out["tool_mask"].shape == (2, 4)
    assert out["old_per_token_logps"].shape == (2, 4) and out["importance_sampling_ratio"].shape == (2, 1)
    assert out["advantages"] is inputs["advantages"] and out["num_items_in_batch"] is inputs["num_items_in_batch"]
    # one row (per_device_train_batch_size=1): exactly its own length
    one = {k: (v[1:] if getattr(v, "ndim", 0) >= 1 else v) for k, v in inputs.items()}
    out1 = grpo_episodes.trim_padding_columns(one)
    assert out1["prompt_ids"].shape == (1, 4) and out1["completion_ids"].shape == (1, 4)
    # a truncated (fully masked) row alone keeps one completion column, never zero (= "all logits")
    dead = dict(one, completion_mask=np.zeros((1, 6), dtype=int))
    assert grpo_episodes.trim_padding_columns(dead)["completion_ids"].shape == (1, 1)
    # an unknown 2-D width (e.g. a field spanning prompt+completion): untouched
    odd = dict(inputs, token_type_ids=np.zeros((2, 11)))
    assert grpo_episodes.trim_padding_columns(odd) is odd


def test_grpo_episodes_dry_run_and_no_gpu_imports(tmp_path, tasks):
    path = tmp_path / "pool.jsonl"
    write_jsonl(tasks, path)
    code = (
        "import sys; from tau_forge.train import grpo_episodes as g, episode_rollout, task_pool;"
        f"g.main(['--tasks', {str(path)!r}, '--dry-run', '--val-holdout', '2']);"
        "bad = [x for x in ('torch', 'vllm', 'transformers', 'trl') if x in sys.modules];"
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert '"loss_type": "dapo"' in r.stdout and "exiting before torch import" in r.stdout


# ------------------------------------------------------- check script


def test_check_token_append_script_with_the_fake_tokenizer(tmp_path, tasks, tok):
    path = tmp_path / "eps.jsonl"
    write_jsonl(_pick(tasks), path)
    mod = _load("check_token_append")
    summary = mod.main(["--tasks", str(path), "--per-template", "1", "--modes", "oracle,transfer"],
                       tokenizer=tok, tools=RetailEnv().all_openai_schemas()[:3], system_message=SYS)
    assert summary["failures"] == []
    assert {"assistant_text", "assistant_tool_call", "tool", "tool_multi", "user"} <= set(summary["kinds"])
    bad = mod.main(["--tasks", str(path), "--per-template", "1", "--modes", "oracle"],
                   tokenizer=CharTokenizer(merges=("\nC", "\nB")), tools=[], system_message=SYS)
    assert bad["failures"]
