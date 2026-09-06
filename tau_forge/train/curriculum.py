"""Training-set composition: category mixture, dead-scenario filtering, and a
synthetic validation split.

Three separate problems, all of which decide how much Phase 8 improvement the
GRPO run can actually produce, and none of which the raw 541-scenario corpus
solves on its own.

1. Mixture. The corpus is balanced by *generation taxonomy* (roughly 110 per
   category), not by how often each shape occurs in the thing being evaluated.
   Counted directly off `data/synthetic/raw/`:

       ambiguous                104   gold = no tool call        (all 104)
       policy_violation         112   gold = no tool call        ( 78 of 112)
       out_of_scope             107   gold = transfer_to_human_agents
       happy_path               110   gold = a real retail action
       requires_earlier_context 108   gold = a real retail action

   So 182 scenarios (33.6%) are graded on *withholding* a call and another 107
   (19.8%) on escalating out of the domain -- 289 of 541, **53.4% of the
   training signal, spent teaching the policy not to do retail work.** The
   benchmark it is graded on is the opposite shape: this project's own earlier
   finding is that only 4 of the 114 real retail tasks (3.5%) ever need
   `transfer_to_human_agents`, and Phase 6 found exactly 2 of the 74 train
   tasks (2.7%) whose correct behavior is purely conversational. Real tasks
   average 4.8 sequential tool calls; they are won by *acting* correctly.

   Training a 53% not-acting mixture is not a neutral choice. On an
   all-or-nothing multi-turn benchmark it pushes toward a policy that asks a
   clarifying question or escalates where it should have authenticated and
   mutated, and every such turn fails the task outright. `REAL_TASK_ALIGNED_MIX`
   is a target distribution shaped like the benchmark instead of like the
   taxonomy. It deliberately keeps a real minority of guardrail scenarios --
   refusing an out-of-policy mutation is worth genuine points too, and dropping
   them entirely would trade one lopsided policy for its mirror image.

2. Dead scenarios. A scenario whose group reward is constant contributes no
   gradient (see `zero_shot_baseline`). Filtering those out concentrates the
   compute budget on prompts that can actually move weights. Note the two
   flavors are not equally dead: constant-at-1.0 is *solved* (harmless to keep
   beyond the wasted rollouts, and worth re-checking later since a scenario can
   regress), while constant-at-0.0 is a cold start that needs a different
   intervention -- shaping (`tau_forge.train.shaping`), prompting, or an SFT
   warm-start -- and will stay dead no matter how long it is trained on.
   `load_zero_variance_ids` reads a `zero_shot_baseline` output directly so the
   exclusion list is measured, never guessed.

3. Validation. Checkpoint selection needs a held-out score, and the held-out
   data policy (README) puts all 114 real tasks off-limits to anything steering
   weight updates -- which selecting a checkpoint by them would be. So the
   validation set is carved out of the synthetic corpus instead, stratified by
   `category__theme` cell so all 30 cells are represented on both sides.

Torch/trl-free.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from tau_forge.train.dataset import TrainingExample

# Target *shares* of the final training set, not counts. Shaped after the real
# benchmark's distribution (see module docstring), not after the generation
# taxonomy. Applied by proportional downsampling: no category is ever upsampled
# past what the corpus actually holds, so these are ceilings in practice.
REAL_TASK_ALIGNED_MIX: dict[str, float] = {
    "happy_path": 0.36,
    "requires_earlier_context": 0.36,
    "policy_violation": 0.15,
    "ambiguous": 0.08,
    "out_of_scope": 0.05,
}

# The corpus as generated -- pass this to keep Phase 2's balance untouched.
UNIFORM_MIX: dict[str, float] = {k: 0.2 for k in REAL_TASK_ALIGNED_MIX}


def parse_mix(spec: str) -> dict[str, float]:
    """`"happy_path=0.4,ambiguous=0.1"` -> dict. Shares are renormalized by
    `apply_mixture`, so they need not sum to 1."""
    mix: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        mix[name.strip()] = float(value)
    return mix


def _cell(example: TrainingExample) -> str:
    return f"{example.category}__{example.theme}"


def load_zero_variance_ids(
    baseline_path: str | Path, include_solved: bool = False, tolerance: float = 1e-3
) -> set[str]:
    """Scenario ids that scored a constant reward across every sample of a
    `zero_shot_baseline` run.

    `include_solved=False` (the default) returns only the constant-at-0.0 cold
    starts -- the ones that stay dead. Constant-at-1.0 scenarios are kept in
    training by default because they are cheap insurance against regression on
    behavior the policy already has, and because a scenario that is solved for
    the *base* model can stop being solved a few hundred steps in."""
    data = json.loads(Path(baseline_path).read_text())
    per_scenario = data["per_scenario_scores"]
    dead: set[str] = set()
    for scenario_id, scores in per_scenario.items():
        if not scores:
            continue
        if max(scores) - min(scores) > tolerance:
            continue
        if include_solved or abs(scores[0]) <= tolerance:
            dead.add(scenario_id)
    return dead


def exclude_ids(examples: list[TrainingExample], ids: Iterable[str]) -> list[TrainingExample]:
    drop = set(ids)
    return [e for e in examples if e.id not in drop]


def apply_mixture(
    examples: list[TrainingExample],
    mix: dict[str, float],
    seed: int = 0,
) -> list[TrainingExample]:
    """Downsample toward `mix` without ever upsampling.

    The kept size is chosen so every category's target share is satisfiable
    from what the corpus actually has: `N = min over categories of
    (available_c / share_c)`. Within a category, scenarios are drawn evenly
    across that category's themes so downsampling never silently deletes a
    whole cell -- the theme axis is the one carrying the domain variety that
    makes these prompts differ from each other at all.
    """
    if not examples:
        return []

    total_share = sum(mix.get(e.category, 0.0) for e in {e.category: e for e in examples}.values())
    if total_share <= 0:
        raise ValueError(f"Mixture {mix} assigns no weight to any present category.")

    by_category: dict[str, list[TrainingExample]] = defaultdict(list)
    for e in examples:
        by_category[e.category].append(e)

    shares = {c: mix.get(c, 0.0) / total_share for c in by_category}
    budget = min(
        (len(items) / shares[c] for c, items in by_category.items() if shares[c] > 0),
        default=0.0,
    )

    rng = random.Random(seed)
    kept: list[TrainingExample] = []
    for category, items in sorted(by_category.items()):
        target = int(round(budget * shares[category]))
        if target <= 0:
            continue
        if target >= len(items):
            kept.extend(items)
            continue

        by_theme: dict[str, list[TrainingExample]] = defaultdict(list)
        for e in items:
            by_theme[e.theme].append(e)
        for pool in by_theme.values():
            rng.shuffle(pool)

        # Round-robin across themes until the category's target is met, so the
        # loss falls evenly on the cells rather than on whichever theme sorts last.
        themes = sorted(by_theme)
        picked: list[TrainingExample] = []
        index = 0
        while len(picked) < target and any(by_theme[t] for t in themes):
            pool = by_theme[themes[index % len(themes)]]
            if pool:
                picked.append(pool.pop())
            index += 1
        kept.extend(picked)

    kept.sort(key=lambda e: e.id)
    return kept


def train_val_split(
    examples: list[TrainingExample], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Stratified by `category__theme` cell. At least one validation example
    per cell whenever the cell has two or more, so a per-cell validation
    breakdown is always available."""
    if val_fraction <= 0:
        return list(examples), []

    by_cell: dict[str, list[TrainingExample]] = defaultdict(list)
    for e in examples:
        by_cell[_cell(e)].append(e)

    rng = random.Random(seed)
    train: list[TrainingExample] = []
    val: list[TrainingExample] = []
    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell], key=lambda e: e.id)
        rng.shuffle(pool)
        n_val = int(round(len(pool) * val_fraction))
        if len(pool) >= 2:
            n_val = max(1, min(n_val, len(pool) - 1))
        val.extend(pool[:n_val])
        train.extend(pool[n_val:])

    train.sort(key=lambda e: e.id)
    val.sort(key=lambda e: e.id)
    return train, val


def summarize(examples: list[TrainingExample]) -> dict[str, Any]:
    by_category: dict[str, int] = defaultdict(int)
    no_call = 0
    for e in examples:
        by_category[e.category] += 1
        if e.expected_tool_name is None:
            no_call += 1
    total = len(examples)
    return {
        "n": total,
        "by_category": dict(sorted(by_category.items())),
        "n_cells": len({_cell(e) for e in examples}),
        "no_call_fraction": (no_call / total) if total else 0.0,
        "non_acting_fraction": (
            sum(1 for e in examples if e.expected_tool_name in (None, "transfer_to_human_agents")) / total
            if total
            else 0.0
        ),
    }


def build_training_sets(
    examples: list[TrainingExample],
    mix: Optional[dict[str, float]] = None,
    exclude: Optional[Iterable[str]] = None,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Filter -> mix -> split, in that order. Filtering first keeps the mixture
    honest (a category gutted by exclusions shrinks the whole budget rather
    than quietly ending up over-represented); splitting last keeps train and
    validation drawn from the same post-mixture distribution."""
    pool = exclude_ids(examples, exclude) if exclude else list(examples)
    if mix:
        pool = apply_mixture(pool, mix, seed=seed)
    return train_val_split(pool, val_fraction=val_fraction, seed=seed)
