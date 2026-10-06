"""Episode task-pool prefilter from a rollout-only audit (wf1_train_eff.md §2,
"Dynamic sampling" (a)).

TRL has no DAPO-style dynamic sampling, so groups that cannot produce a
gradient are removed before training instead. From an episode audit JSON
(`scripts/episode_audit.py` output; `per_task[*].rewards` / `.successes`, n
samples per task) every task is classified:

  * solved     -- every sample succeeded at reward 1.0 (8/8 at n=8): flat at
                  the top, zero advantage;
  * dead       -- flat (all rewards equal) and no sample succeeded: flat 0.0,
                  or a flat shaping-only crumb such as 0.1. Zero advantage, and
                  no sign the policy can reach the goal;
  * flat_other -- flat but neither of the above (e.g. every sample succeeded
                  without confirmation at 0.7). Kept unless `drop_flat_other`;
  * live       -- the rewards differ: the groups GRPO learns from;
  * unjudged   -- fewer than 2 samples in the audit, or not in the audit.
                  Kept.

Solved tasks are dropped (`keep_solved_fraction` 0.0); a deterministic
`keep_dead_fraction` (default 0.2) of the dead tasks of each template is kept,
so a policy that improves can still find signal there and the pool does not
lose whole cells. Torch-free; used by `scripts/prefilter_tasks.py` and
`grpo_episodes --prefilter-audit`.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence, Union

CATEGORIES = ("live", "solved", "dead", "flat_other", "unjudged")


def classify_record(rec: dict[str, Any], tol: float = 1e-9) -> str:
    rewards = [float(r) for r in rec.get("rewards", [])]
    successes = list(rec.get("successes", []))
    if len(rewards) < 2:
        return "unjudged"
    if max(rewards) - min(rewards) > tol:
        return "live"
    if successes and all(successes) and min(rewards) >= 1.0 - tol:
        return "solved"
    if not any(successes):
        return "dead"
    return "flat_other"


def classify_audit(audit: Union[dict[str, Any], str, Path], tol: float = 1e-9) -> dict[str, str]:
    """task id -> category, for every task record in an episode audit."""
    if not isinstance(audit, dict):
        audit = json.loads(Path(audit).read_text())
    return {rec["id"]: classify_record(rec, tol) for rec in audit["per_task"]}


def prefilter_tasks(
    tasks: Sequence[Any],
    audit: Union[dict[str, Any], str, Path],
    *,
    keep_dead_fraction: float = 0.2,
    keep_solved_fraction: float = 0.0,
    drop_flat_other: bool = False,
    seed: int = 0,
) -> tuple[list[Any], dict[str, Any]]:
    """(kept tasks in input order, report). `tasks` are `EpisodeTask`s (any
    object with `.id` and `.template`). The dead/solved subsets kept are a
    seeded sample of floor(fraction * count) per template."""
    cats = classify_audit(audit)
    by_cat: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in tasks:
        by_cat[t.template][cats.get(t.id, "unjudged")].append(t.id)

    keep_ids: set[str] = set()
    report: dict[str, Any] = {"per_template": {}, "settings": {
        "keep_dead_fraction": keep_dead_fraction, "keep_solved_fraction": keep_solved_fraction,
        "drop_flat_other": drop_flat_other, "seed": seed,
    }}
    for template in sorted(by_cat):
        groups = by_cat[template]
        row: dict[str, int] = {c: len(groups.get(c, [])) for c in CATEGORIES}
        row["n"] = sum(row[c] for c in CATEGORIES)
        rng = random.Random(f"{seed}:{template}")
        kept_dead = _sample(groups.get("dead", []), keep_dead_fraction, rng)
        kept_solved = _sample(groups.get("solved", []), keep_solved_fraction, rng)
        keep_ids.update(groups.get("live", []))
        keep_ids.update(groups.get("unjudged", []))
        if not drop_flat_other:
            keep_ids.update(groups.get("flat_other", []))
        keep_ids.update(kept_dead)
        keep_ids.update(kept_solved)
        row["kept_dead"] = len(kept_dead)
        row["kept_solved"] = len(kept_solved)
        row["kept"] = (
            row["live"] + row["unjudged"] + (0 if drop_flat_other else row["flat_other"]) + len(kept_dead) + len(kept_solved)
        )
        report["per_template"][template] = row
    kept = [t for t in tasks if t.id in keep_ids]
    tot = {k: sum(r[k] for r in report["per_template"].values()) for k in (*CATEGORIES, "n", "kept", "kept_dead", "kept_solved")}
    flat_kept = tot["kept_dead"] + tot["kept_solved"] + (0 if drop_flat_other else tot["flat_other"])
    judged_kept = tot["kept"] - tot["unjudged"]
    tot["flat_share_before"] = round((tot["solved"] + tot["dead"] + tot["flat_other"]) / max(1, tot["n"] - tot["unjudged"]), 4)
    tot["flat_share_after"] = round(flat_kept / max(1, judged_kept), 4)
    report["overall"] = tot
    return kept, report


def _sample(ids: list[str], fraction: float, rng: random.Random) -> list[str]:
    k = int(math.floor(max(0.0, min(1.0, fraction)) * len(ids) + 1e-9))
    return sorted(rng.sample(sorted(ids), k)) if k else []


def format_report(report: dict[str, Any]) -> str:
    cols = ("n", "live", "solved", "dead", "flat_other", "unjudged", "kept_dead", "kept_solved", "kept")
    lines = [f"{'template':24s} " + " ".join(f"{c:>10s}" for c in cols)]
    for t, row in report["per_template"].items():
        lines.append(f"{t:24s} " + " ".join(f"{row[c]:>10d}" for c in cols))
    o = report["overall"]
    lines.append(f"{'TOTAL':24s} " + " ".join(f"{o[c]:>10d}" for c in cols))
    lines.append(f"flat-group share (judged tasks): before {o['flat_share_before']:.3f} -> after {o['flat_share_after']:.3f}")
    return "\n".join(lines)
