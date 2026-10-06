"""Prefilter an episode task pool with a rollout-only audit before GRPO.

Drops tasks every audit sample solved at reward 1.0 (8/8 at n=8) and keeps
only a seeded `--keep-dead-fraction` (default 0.2) per template of the dead
ones (flat rewards, no success: flat 0.0 or a flat shaping crumb). Live tasks,
tasks the audit did not cover, and flat-but-not-solved-nor-dead tasks (e.g.
every sample succeeded without confirmation, flat 0.7) are kept; pass
`--drop-flat-other` to drop the last. Classification:
`tau_forge/train/task_pool.py`. Prints the counts per template.

Usage:
    python scripts/episode_audit.py --tasks data/episodes/episodes_s1.jsonl --samples-per-task 8 \\
        --output data/trained/episode_audit_s1_n8.json            # GPU box
    python scripts/prefilter_tasks.py --audit data/trained/episode_audit_s1_n8.json \\
        --out data/episodes/pool_s1.jsonl
    # then: grpo_episodes --tasks data/episodes/pool_s1.jsonl   (or --prefilter-audit on the trainer)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tau_forge.train.task_pool import format_report, prefilter_tasks  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit", required=True, help="scripts/episode_audit.py output JSON.")
    p.add_argument("--tasks", default=None, help="Task pool JSONL. Default: the audit's own --tasks.")
    p.add_argument("--out", default=None, help="Kept tasks JSONL. Default: <tasks>_prefiltered.jsonl next to the pool.")
    p.add_argument("--ids-out", default=None, help="Also write the kept ids, one per line.")
    p.add_argument("--keep-dead-fraction", type=float, default=0.2)
    p.add_argument("--keep-solved-fraction", type=float, default=0.0)
    p.add_argument("--drop-flat-other", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.exists() or p.is_absolute() else REPO_ROOT / p


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    from tau_forge.episodes.generate import read_jsonl, write_jsonl

    args = parse_args(argv)
    audit = json.loads(Path(args.audit).read_text())
    tasks_path = args.tasks or (audit.get("config") or {}).get("tasks")
    if not tasks_path:
        raise ValueError("the audit was generated on the fly; pass --tasks with the pool it ran on")
    tasks_path = _resolve(tasks_path)
    tasks = read_jsonl(tasks_path)
    kept, report = prefilter_tasks(
        tasks, audit, keep_dead_fraction=args.keep_dead_fraction, keep_solved_fraction=args.keep_solved_fraction,
        drop_flat_other=args.drop_flat_other, seed=args.seed,
    )
    out = Path(args.out) if args.out else tasks_path.with_name(tasks_path.stem + "_prefiltered.jsonl")
    write_jsonl(kept, out)
    if args.ids_out:
        Path(args.ids_out).write_text("".join(t.id + "\n" for t in kept))
    n_samples = {len(r.get("rewards", [])) for r in audit["per_task"]}
    print(f"[prefilter_tasks] audit {args.audit}: {len(audit['per_task'])} task records, samples/task {sorted(n_samples)}")
    print(format_report(report))
    print(f"[prefilter_tasks] kept {len(kept)} of {len(tasks)} tasks -> {out}")
    report["out"] = str(out)
    return report


if __name__ == "__main__":
    main()
