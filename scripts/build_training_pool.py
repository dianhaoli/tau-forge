"""Build the episode training pool and a user-disjoint synthetic validation set.

    python scripts/build_training_pool.py --out-dir data/episodes/pool_v2

1. Generates single-request tasks for every template (decontaminated against
   the 114 real tasks via `tau_forge.decontam.real_tasks`, ids/hashes only).
2. Composes multi-request tasks from that pool (`composite.py`).
3. Splits by USER: ~15% of users (hash of the user id) are validation-only, so
   validation tasks never share a customer with training tasks.
4. Samples the training pool to a template mixture after tau2 retail's task
   shapes (aggregate train-split counts: ~half the tasks write more than once;
   exchange / return / modify-items most common; payment changes rare) and the
   post-fix variance audit (return_fallback and modify_payment mostly solved,
   so down-weighted).

Writes train.jsonl, val.jsonl and manifest.json (counts per template, the
mixture, seeds). Then audit the pool (`scripts/episode_audit.py --tasks
.../train.jsonl --samples-per-task 8`) and prefilter it
(`grpo_episodes --prefilter-audit`) before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Share of the training pool per template (normalised over what is available).
MIXTURE = {
    "composite": 0.36,
    "exchange": 0.13,
    "modify_items": 0.11,
    "info": 0.07,
    "status_refusal": 0.07,
    "cancel": 0.06,
    "foreign_order_refusal": 0.06,
    "return_fallback": 0.06,
    "modify_address": 0.05,
    "modify_user_address": 0.02,
    "modify_payment": 0.02,
}


def is_val_user(user_id: str, frac: float) -> bool:
    h = int(hashlib.sha256(f"val/{user_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def main(argv=None) -> None:
    from tau_forge.decontam.real_tasks import load_real_task_exclusions
    from tau_forge.episodes.composite import compose_tasks
    from tau_forge.episodes.generate import generate_tasks, write_jsonl
    from tau_forge.episodes.task import TEMPLATES

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "episodes" / "pool_v2"))
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--per-template", type=int, default=400)
    p.add_argument("--composites", type=int, default=600)
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--val-size", type=int, default=160)
    p.add_argument("--val-user-frac", type=float, default=0.15)
    args = p.parse_args(argv)

    exclusions = load_real_task_exclusions()
    singles = generate_tasks(args.per_template, args.seed, templates=TEMPLATES, exclusions=exclusions).tasks
    comps = compose_tasks(singles, args.composites, args.seed, exclusions=exclusions,
                          log=lambda m: print(m, file=sys.stderr))
    tasks = singles + comps

    def is_val(t) -> bool:
        return any(is_val_user(u, args.val_user_frac) for u in t.involved_users)

    def is_train(t) -> bool:
        return not any(is_val_user(u, args.val_user_frac) for u in t.involved_users)

    rng = random.Random(f"pool/{args.seed}")
    by_tpl: dict[str, list] = {}
    for t in tasks:
        if is_train(t):
            by_tpl.setdefault(t.template, []).append(t)
    for v in by_tpl.values():
        rng.shuffle(v)
    avail = {k: len(v) for k, v in by_tpl.items()}
    weights = {k: w for k, w in MIXTURE.items() if avail.get(k)}
    total_w = sum(weights.values())
    want = {k: round(args.train_size * w / total_w) for k, w in weights.items()}
    train = []
    for k, n in want.items():
        train += by_tpl[k][:n]
    rng.shuffle(train)

    val_by_tpl: dict[str, list] = {}
    for t in tasks:
        if is_val(t) and all(is_val_user(u, args.val_user_frac) for u in t.involved_users if u == t.user_id):
            val_by_tpl.setdefault(t.template, []).append(t)
    val = []
    for k, w in weights.items():
        cands = val_by_tpl.get(k, [])
        rng.shuffle(cands)
        val += cands[: max(4, round(args.val_size * w / total_w))]
    rng.shuffle(val)
    assert not ({u for t in train for u in t.involved_users} & {t.user_id for t in val}), "train/val share a user"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    manifest = {
        "seed": args.seed, "generator_commit": commit, "per_template": args.per_template,
        "mixture": MIXTURE, "available_train": avail,
        "train": dict(Counter(t.template for t in train)), "val": dict(Counter(t.template for t in val)),
        "train_composite_combos": dict(Counter("+".join(t.difficulty["combo"]) for t in train if t.subs)),
        "val_user_frac": args.val_user_frac,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
