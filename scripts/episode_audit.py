"""Zero-shot VARIANCE AUDIT for multi-step episode tasks, on a GPU box.

The episode counterpart of `zero_shot_baseline --use-vllm`: before any
episode GRPO run, measure for each generated task how often the base policy
succeeds over n samples, and what fraction of tasks would give a GRPO group a
usable spread (std >= 0.05). Episodes exist because single-decision prompts
sat near p=0 or p=1 (72.5% zero-variance at n=16); this is the measurement
that says whether the episode templates and knobs actually land in between,
and which cells (template x knob value) to weight.

Multi-turn, batched: each round every active episode contributes one request
-- the same system message `tau_forge.train.dataset._system_message` builds
for training and the 16 tau2 tool schemas, through the tokenizer's chat
template -- and all of them go through one vLLM `generate`. Tool calls
execute against each episode's private copy-on-write db; text turns go to
the scripted user. The loop itself is `tau_forge.episodes.audit.run_audit`
and is unit-tested with fake generators (tests/test_episodes.py,
tests/test_episode_rollout.py); nothing at module level imports torch,
transformers or vLLM.

Speed (wf1_train_eff.md, bottleneck 1). `--max-inflight` (default 64) caps the
active episodes and refills from the queue as episodes end, so each episode's
history stays in vLLM's prefix cache between its turns instead of being
evicted and re-prefilled by 2,000 lockstep histories. With the built-in vLLM
engine, prompts are token ids built by APPENDING each turn's sampled ids and
the tokenized environment reply (`--prompt-mode tokens`, the default there),
exactly as GRPO training builds them; `--prompt-mode text` re-renders every
turn's full conversation as before. `--max-inflight 0` restores lockstep. The
output JSON has the same format either way.

Usage (GPU box, `uv sync --extra train`):
    # 64 tasks per template generated on the fly (seed 0), 16 samples each
    python scripts/episode_audit.py --per-template 64 --samples-per-task 16

    # From a saved task file, sampler matched to grpo_train's
    python -m tau_forge.episodes.generate --per-template 200 --out data/episodes/episodes_s0.jsonl
    python scripts/episode_audit.py --tasks data/episodes/episodes_s0.jsonl \\
        --samples-per-task 16 --temperature 1.0 --top-p 1.0 --top-k 0

    # CPU dry run of the whole pipeline with a scripted policy (no model)
    python scripts/episode_audit.py --per-template 4 --samples-per-task 2 --fake-policy oracle
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tau_forge.episodes.audit import ChatTemplatePolicy, EpisodeRequest, Generation, TokenAppendPolicy, run_audit
from tau_forge.episodes.task import TEMPLATES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "trained" / "episode_audit.json"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tasks", default=None, help="JSONL from `python -m tau_forge.episodes.generate`. Default: generate now.")
    p.add_argument("--per-template", type=int, default=64, help="Tasks per template when generating (ignored with --tasks).")
    p.add_argument("--seed", type=int, default=0, help="Generation seed, and vLLM's sampling seed.")
    p.add_argument("--templates", default=None,
                   help="Comma-separated templates to keep (incl. 'composite'). Default: every task in --tasks, or all "
                        "single-request templates when generating.")
    p.add_argument("--samples-per-task", type=int, default=16, help="The GRPO group size this audit predicts for.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0, help="MUST match grpo_train's, or this measures a different sampler.")
    p.add_argument("--top-k", type=int, default=0, help="0 = disabled (passed to vLLM as -1, explicitly).")
    p.add_argument("--max-new-tokens", type=int, default=1024, help="Per assistant turn. A turn that hits it ends the episode as 'truncated'.")
    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="vLLM KV-cache ceiling. The system prompt + tool schemas are ~6k tokens and a long "
        "exchange episode adds ~4.4k of tool output plus turns; an episode whose next prompt + "
        "--max-new-tokens exceeds this ends as 'context_budget' instead of being sent.",
    )
    p.add_argument("--max-turns", type=int, default=30, help="Assistant messages per episode (tool calls included).")
    p.add_argument("--max-calls", type=int, default=30)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument(
        "--max-inflight",
        type=int,
        default=64,
        help="Episodes in flight at once; a finished one is replaced from the queue. Keep it within what the "
        "KV cache holds (~6k shared prefix + ~1.8k per episode: ~38 on an A10G at 0.85, ~64 on an H100 at 0.35). "
        "0 = every episode at once (the old lockstep).",
    )
    p.add_argument(
        "--prompt-mode",
        choices=("auto", "tokens", "text"),
        default="auto",
        help="tokens: append sampled ids + tokenized env replies (training's construction); text: re-render the "
        "chat template every turn. auto = tokens with the built-in vLLM engine, text with an injected engine.",
    )
    p.add_argument("--save-transcripts", action="store_true", help="Store every episode's messages and reward breakdown (large).")
    p.add_argument(
        "--fake-policy",
        default=None,
        help="Skip the model and drive every episode with a tau_forge.episodes.reference_agents mode "
        "(oracle, no_confirm, ...) -- a CPU check of generation, user, runner, reward and report.",
    )
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args(argv)


def load_tasks(args: argparse.Namespace) -> list:
    from tau_forge.episodes.generate import generate_tasks, read_jsonl

    if args.tasks:
        tasks = read_jsonl(args.tasks)
        if args.templates:
            keep = set(args.templates.split(","))
            tasks = [t for t in tasks if t.template in keep]
        return tasks
    templates = tuple(args.templates.split(",")) if args.templates else TEMPLATES
    return generate_tasks(args.per_template, args.seed, templates=templates).tasks


def make_vllm_engine(args: argparse.Namespace) -> Callable[[list[str]], list[tuple[str, str]]]:
    """The only GPU-bound piece. Prefix caching matters here: every request
    of every turn starts with the same ~6k-token system prompt."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_new_tokens,
    )

    def engine(prompts: list[str]) -> list[tuple[str, str]]:
        outputs = llm.generate(prompts, params, use_tqdm=False)
        return [(o.outputs[0].text, o.outputs[0].finish_reason or "stop") for o in outputs]

    return engine


def make_vllm_token_engine(args: argparse.Namespace) -> Callable[[list[list[int]]], list[tuple]]:
    """`make_vllm_engine` over token ids: (token_ids, text, None, finish)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_new_tokens,
    )

    def engine(prompt_ids: list[list[int]]) -> list[tuple]:
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], params, use_tqdm=False)
        return [
            (list(o.outputs[0].token_ids), o.outputs[0].text, None, o.outputs[0].finish_reason or "stop")
            for o in outputs
        ]

    return engine


def fake_policy(mode: str) -> Callable[[list[EpisodeRequest]], list[Generation]]:
    from tau_forge.episodes.reference_agents import ReferenceAgent

    agents: dict[tuple[int, int], Any] = {}

    def generate(requests: list[EpisodeRequest]) -> list[Generation]:
        out = []
        for r in requests:
            if r.key not in agents:
                if r.task.subs:
                    from tau_forge.episodes.composite import CompositeAgent

                    agents[r.key] = CompositeAgent(r.task, mode)
                else:
                    agents[r.key] = ReferenceAgent(r.task, mode)
            agent = agents[r.key]
            out.append(Generation(agent(r.messages)))
        return out

    return generate


def build_policy(args: argparse.Namespace, tokenizer=None, engine=None) -> tuple[Callable, dict[str, Any]]:
    """(batch generator, prompt-side facts for the report). `tokenizer` and
    `engine` are injectable so tests can run this exact path without
    transformers or vLLM."""
    if args.fake_policy:
        return fake_policy(args.fake_policy), {"fake_policy": args.fake_policy}
    from tau_forge.envs.retail import RetailEnv

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    tools = RetailEnv().all_openai_schemas()
    mode = args.prompt_mode if args.prompt_mode != "auto" else ("text" if engine is not None else "tokens")
    if mode == "tokens":
        token_policy = TokenAppendPolicy(
            engine=engine if engine is not None else make_vllm_token_engine(args),
            tokenizer=tokenizer,
            tools=tools,
            max_new_tokens=args.max_new_tokens,
            max_model_len=args.max_model_len,
        )
        return token_policy, {"n_tools": len(tools), "prompt_mode": "tokens"}
    policy = ChatTemplatePolicy(
        engine=engine if engine is not None else make_vllm_engine(args),
        apply_chat_template=functools.partial(tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True),
        tools=tools,
        count_tokens=lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"]),
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
    )
    return policy, {"n_tools": len(tools), "prompt_mode": "text"}


def system_message() -> dict[str, str]:
    from tau_forge.train.dataset import _default_policy_text, _system_message

    return _system_message(_default_policy_text())


def print_summary(summary: dict[str, Any]) -> None:
    o = summary["overall"]
    print(
        f"[episode_audit] {o['n_tasks']} tasks, {o['n_episodes']} episodes: success={o['success_rate']:.3f} "
        f"mean={o['mean_reward']:.3f} effective_variance={o['effective_variance_fraction']:.3f} "
        f"flat={o['flat_fraction']:.3f}"
    )
    for t, a in summary["per_template"].items():
        print(f"  {t:24s} n={a['n_tasks']:4d} success={a['success_rate']:.3f} eff_var={a['effective_variance_fraction']:.3f} ends={a['end_reasons']}")
    for t, knobs in summary["per_knob"].items():
        for k, vals in knobs.items():
            cells = "  ".join(f"{v}: p={a['success_rate']:.2f} ev={a['effective_variance_fraction']:.2f} (n={a['n_tasks']})" for v, a in vals.items())
            print(f"    {t}.{k}: {cells}")


def main(argv: Optional[list[str]] = None, *, tokenizer=None, engine=None) -> dict[str, Any]:
    args = parse_args(argv)
    tasks = load_tasks(args)
    if not tasks:
        raise ValueError("no tasks to audit -- check --tasks/--templates")
    policy, facts = build_policy(args, tokenizer=tokenizer, engine=engine)
    print(
        f"[episode_audit] {len(tasks)} tasks x {args.samples_per_task} samples, "
        f"max_inflight={args.max_inflight or 'all'}"
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-task checkpoint: each task's record is appended the moment its last
    # sample ends, so a crash or preemption hours in keeps what finished.
    partial = out.with_name(out.name + ".partial.jsonl")
    partial.write_text("")

    def checkpoint(rec: dict[str, Any]) -> None:
        with partial.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    t0 = time.time()
    result = run_audit(
        tasks,
        policy,
        n_samples=args.samples_per_task,
        system_message=system_message(),
        max_turns=args.max_turns,
        max_calls=args.max_calls,
        keep_transcripts=args.save_transcripts,
        progress=lambda turn, n: print(f"[episode_audit] turn {turn}: {n} active episodes", file=sys.stderr),
        on_task_done=checkpoint,
        max_inflight=args.max_inflight if args.max_inflight > 0 else None,
    )
    if isinstance(policy, (ChatTemplatePolicy, TokenAppendPolicy)):
        facts.update(max_prompt_tokens=policy.max_prompt_tokens, context_overflows=policy.n_context_overflows)
    result["config"] = {
        **vars(args),
        **facts,
        "seconds": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(result, indent=1, default=str))
    print_summary(result["summary"])
    print(f"[episode_audit] wrote {out}")
    return result


if __name__ == "__main__":
    main()
