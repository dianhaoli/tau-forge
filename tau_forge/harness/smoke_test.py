"""Phase 6 smoke test: prove the harness (Phase 1 env + Phase 4 reward +
Phase 6 data prep) works end-to-end against the real, trusted 74 τ²-bench
retail train tasks -- before spending effort on synthetic data (Phase 2/3)
or GPU training (Phase 7).

**Scope note:** this environment has no GPU and no ML training stack
(no torch/trl/vllm -- confirmed absent here), so this script does not run
real GRPO steps. It validates the *plumbing* a GRPO trainer would sit on
top of, using two scripted stand-in policies instead of a live model:

- `gold`  -- replays each task's gold action sequence verbatim. Expected to
  score a perfect 1.0 on every turn of every task; any task that doesn't
  is a real bug in env/reward/data-prep alignment, not a training problem,
  and fails the smoke test.
- `noisy` -- gold actions with randomly injected corruption (wrong tool,
  bad argument, or a dropped call). Expected to score meaningfully lower
  than `gold`; if it doesn't, the reward function isn't discriminating
  between good and bad rollouts and is unsafe to train against.

Real GRPO smoke-test steps (Phase 7) need an actual GPU box -- see the
AWS EC2 setup notes in `README.md`.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH

from tau_forge.data_prep.trusted_tasks import DEFAULT_OUTPUT_PATH, build_and_save, load_saved
from tau_forge.envs.retail import RetailEnv
from tau_forge.harness.rollout import TrajectoryScore, gold_rollout, score_rollout
from tau_forge.reward.reward import Action

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "trusted" / "phase6_smoke_report.json"
GOLD_PASS_THRESHOLD = 0.999  # per-turn mean; allow for float rounding only
NOISE_RATE = 0.5  # fraction of turns corrupted in the `noisy` policy
SEED = 0


def _all_tool_names() -> list[str]:
    return RetailEnv().tool_names()


def noisy_rollout(task, rng: random.Random, all_tools: list[str]) -> list[Action]:
    actions = []
    for a in task.gold_actions:
        roll = rng.random()
        if roll < NOISE_RATE / 3:
            actions.append(Action(tool_name=None))  # dropped call
        elif roll < 2 * NOISE_RATE / 3:
            wrong_tool = rng.choice([t for t in all_tools if t != a.tool_name])
            actions.append(Action(wrong_tool, dict(a.arguments)))
        elif roll < NOISE_RATE:
            corrupted = dict(a.arguments)
            if corrupted:
                key = rng.choice(list(corrupted.keys()))
                corrupted[key] = "__corrupted__"
            actions.append(Action(a.tool_name, corrupted))
        else:
            actions.append(Action(a.tool_name, dict(a.arguments)))
    return actions


def _summarize(scores: list[TrajectoryScore]) -> dict[str, Any]:
    means = [s.mean for s in scores]
    return {
        "n_tasks": len(scores),
        "n_turns": sum(len(s.turn_scores) for s in scores),
        "mean_of_task_means": sum(means) / len(means) if means else 0.0,
        "min_task_mean": min(means) if means else 0.0,
        "per_task": {s.task_id: s.mean for s in scores},
    }


def run() -> dict[str, Any]:
    if DEFAULT_OUTPUT_PATH.exists():
        tasks = load_saved()
    else:
        tasks = build_and_save()
    assert tasks, "no trusted train tasks loaded"

    all_tools = _all_tool_names()
    rng = random.Random(SEED)
    initial_db = RetailDB.load(RETAIL_DB_PATH)

    gold_scores = [score_rollout(t, gold_rollout(t), initial_db=initial_db) for t in tasks]
    noisy_scores = [
        score_rollout(t, noisy_rollout(t, rng, all_tools), initial_db=initial_db) for t in tasks
    ]

    gold_summary = _summarize(gold_scores)
    noisy_summary = _summarize(noisy_scores)

    report = {
        "n_train_tasks": len(tasks),
        "gold_policy": gold_summary,
        "noisy_policy": noisy_summary,
        "gold_pass_threshold": GOLD_PASS_THRESHOLD,
        "passed": gold_summary["mean_of_task_means"] >= GOLD_PASS_THRESHOLD,
        "reward_discriminates": noisy_summary["mean_of_task_means"] < gold_summary["mean_of_task_means"],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    report = run()
    print(f"trusted train tasks: {report['n_train_tasks']}")
    print(f"gold policy  mean-of-task-means: {report['gold_policy']['mean_of_task_means']:.4f} "
          f"(min task mean: {report['gold_policy']['min_task_mean']:.4f})")
    print(f"noisy policy mean-of-task-means: {report['noisy_policy']['mean_of_task_means']:.4f}")
    print(f"report written to {REPORT_PATH}")
    if not report["passed"]:
        raise SystemExit(
            f"SMOKE TEST FAILED: gold policy mean {report['gold_policy']['mean_of_task_means']:.4f} "
            f"< threshold {GOLD_PASS_THRESHOLD} -- env/reward/data-prep are not aligned on trusted data."
        )
    if not report["reward_discriminates"]:
        raise SystemExit("SMOKE TEST FAILED: noisy policy scored >= gold policy -- reward is not discriminating.")
    print("PASSED")
