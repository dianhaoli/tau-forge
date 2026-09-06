"""Phase 8 evaluation entrypoint: run the real tau2-bench retail benchmark
against a locally-served checkpoint, with the training-time system prompt.

Why this exists rather than a bare `tau2 run`
---------------------------------------------
Two things have to be true for the number this produces to mean "RLVR improved
the model", and neither is true of a plain `tau2 run`:

1. **Prompt parity.** tau2's stock agent prompt omits the
   `TOOL_CALL_FORMAT_INSTRUCTION` suffix every training prompt carried. This
   module calls `tau_forge.eval.prompt_parity.patch_agent_instruction()`
   before constructing anything, and asserts the two prompts are byte-identical
   before it will run. See that module for why the same regime must be used
   for the baseline.
2. **A fixed comparison.** Everything except the agent weights has to be held
   constant between the baseline run and the trained run: the user-simulator
   model, its temperature, the seed, the number of trials, max steps. Those are
   flags here with defaults that are recorded into the results directory name,
   so two runs that differ in any of them are visibly different runs.

Baseline first, always. `--label baseline` against the untouched base model is
the number every later checkpoint is measured from; without it "improvement"
has no referent.

Serving the policy (separate terminal, or a separate box):

    vllm serve <model-or-checkpoint-path> \\
        --served-model-name tau-forge-policy \\
        --enable-auto-tool-choice --tool-call-parser hermes \\
        --max-model-len 16384 --port 8000

`--tool-call-parser hermes` is the one that parses Qwen's
`<tool_call>{...}</tool_call>` convention -- the same convention
`tau_forge.train.completion_parsing` grades during training. Without it, vLLM
returns the tool call as plain assistant text, tau2 sees an agent that never
calls a tool, and every task fails for a reason that has nothing to do with the
policy.

Then:

    python -m tau_forge.eval.run_tau2 --label baseline \\
        --agent-llm hosted_vllm/tau-forge-policy \\
        --agent-api-base http://localhost:8000/v1 \\
        --task-split-name test --num-trials 4

Held-out data policy (README): the 114 real retail tasks are evaluation-only.
This module reads them; nothing here updates weights, and no output of it may
be fed back into training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

DEFAULT_SERVED_MODEL = "hosted_vllm/tau-forge-policy"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", default="retail", help="'retail' for the headline number; 'airline' for the zero-shot transfer check.")
    p.add_argument(
        "--label",
        required=True,
        help="Short name for this run, e.g. 'baseline' or 'step250'. Goes into the results "
        "directory name so runs are self-identifying.",
    )
    p.add_argument("--agent-llm", default=DEFAULT_SERVED_MODEL)
    p.add_argument(
        "--agent-api-base",
        default=os.environ.get("TAU_FORGE_AGENT_API_BASE", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL of the vLLM server holding the policy.",
    )
    p.add_argument("--agent-temperature", type=float, default=0.0)
    p.add_argument(
        "--user-llm",
        default="gpt-4.1-2025-04-14",
        help="The user simulator. tau2 routes this through litellm, so any provider works: "
        "'anthropic/claude-sonnet-5' or 'anthropic/claude-haiku-4-5' with ANTHROPIC_API_KEY set "
        "are drop-in alternatives to the OpenAI default. Two caveats. First, HOLD THIS FIXED "
        "across the baseline and every trained run -- it speaks half of every conversation, so "
        "changing it changes the benchmark rather than the policy, and a before/after across "
        "two different simulators measures nothing. Second, tau2's published leaderboard numbers "
        "use the gpt-4.1 default; a run on any other simulator is internally valid but not "
        "comparable to those, so keep the default if you ever want to line up against them.",
    )
    p.add_argument("--user-temperature", type=float, default=0.0)
    p.add_argument(
        "--task-split-name",
        default="test",
        help="'test' (40 tasks) for the headline number, 'base' for all 114. Both are "
        "evaluation-only under the held-out data policy.",
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=4,
        help="Independent runs of each task. 4 is what pass^k up to k=4 needs; 1 trial gives a "
        "pass^1 estimate too noisy to read a training delta off, since the user simulator is "
        "itself stochastic.",
    )
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--save-to", default=None, help="Defaults to a label/split/timestamp name.")
    p.add_argument(
        "--stock-prompt",
        action="store_true",
        help="Skip the prompt-parity patch and evaluate under tau2's unmodified agent prompt. "
        "Valid only if the training corpus was also built without TOOL_CALL_FORMAT_INSTRUCTION "
        "-- and, either way, only if the baseline used the same setting.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the resolved config and exit.")
    return p.parse_args(argv)


def build_run_config(args: argparse.Namespace):
    from tau2.data_model.simulation import TextRunConfig

    save_to = args.save_to or (
        f"tau_forge_{args.label}_{args.domain}_{args.task_split_name}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    return TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        llm_agent=args.agent_llm,
        # api_base rides in llm_args so litellm routes to the local vLLM server
        # instead of a hosted provider.
        llm_args_agent={"temperature": args.agent_temperature, "api_base": args.agent_api_base},
        user="user_simulator",
        llm_user=args.user_llm,
        llm_args_user={"temperature": args.user_temperature},
        task_split_name=args.task_split_name,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        save_to=save_to,
    )


def main() -> None:
    args = parse_args()

    from tau_forge.eval.prompt_parity import assert_prompts_match, patch_agent_instruction

    if args.stock_prompt:
        print("[run_tau2] --stock-prompt: evaluating under tau2's unmodified agent prompt.")
    else:
        patch_agent_instruction()
        assert_prompts_match()
        print("[run_tau2] prompt parity: training and eval system prompts are byte-identical.")

    config = build_run_config(args)
    print(f"[run_tau2] {config.model_dump_json(indent=2)}")
    if args.dry_run:
        print("[run_tau2] --dry-run: exiting before launching simulations.")
        return

    from tau2.run import run_domain

    results = run_domain(config)
    print(f"[run_tau2] Done. Results saved under data/simulations/{config.save_to}/")
    print("[run_tau2] Score them with: tau2 view  (or tau2's leaderboard tooling)")
    return results


if __name__ == "__main__":
    main()
