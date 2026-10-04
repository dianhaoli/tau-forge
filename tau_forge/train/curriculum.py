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
   exclusion list is measured, never guessed -- and it reads the *shaped*
   scores when the audit has them, because those are what GRPO sees with
   `--shaping` on (its default). Reading raw `reward()` instead dropped exactly
   the groups shaping exists to revive: shaping only adds credit to wrong-tool
   completions, so every revived group is flat 0.0 under raw scores. What
   counts as dead is "no effective variance and not solved"
   (`scorecard.lacks_gradient`), not "flat at 0.0": shaping can lift a cold
   start to a flat 0.13, which is just as gradient-free.

3. Label defects. `data/synthetic/label_audit.json` marks 222 of the 541
   scenarios whose gold label contradicts the retail policy (a write the user
   never confirmed, an id only a guess can produce, a transfer where policy.md
   says deny, ...). On those the policy-correct action scores 0, so training
   on them pushes *away* from the policy, and since every policy-following
   sample scores the same 0 they are also flat groups. `load_label_defect_ids`
   reads that file; grpo_train drops them by default. They are not spread
   evenly: dropping them takes the no-call gold share from 33.6% to 56.1%
   (no-call or transfer: 53.4% to 67.4%), which makes the mixture in point 1
   more necessary, not less, and empties three cells outright
   (`exclusion_shift`).

4. Validation. Checkpoint selection needs a held-out score, and the held-out
   data policy (README) puts all 114 real tasks off-limits to anything steering
   weight updates -- which selecting a checkpoint by them would be. So the
   validation set is carved out of the synthetic corpus instead, stratified by
   `category__theme` cell so every cell with two or more scenarios is
   represented on both sides -- 30 cells as generated, 27 once the label
   defects are dropped.

Torch/trl-free.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from tau_forge.train.dataset import TrainingExample
from tau_forge.train.scorecard import EFFECTIVE_MIN_STD, lacks_gradient

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_AUDIT = REPO_ROOT / "data" / "synthetic" / "label_audit.json"

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
    baseline_path: str | Path,
    include_solved: bool = False,
    min_std: float = EFFECTIVE_MIN_STD,
    prefer_shaped: bool = True,
) -> set[str]:
    """Scenario ids whose group carried no usable gradient in a
    `zero_shot_baseline` run: group std below `min_std` (the effective-variance
    threshold, `scorecard.EFFECTIVE_MIN_STD`), whatever value the group sits at.

    `include_solved=False` (the default) keeps groups that are flat at (or
    within `min_std` of) 1.0 -- cheap insurance against regression on
    behavior the policy already has, and a scenario that is solved for the
    *base* model can stop being solved a few hundred steps in. Everything else
    without effective variance is dropped: flat at 0.0, flat at a partial
    plateau, flat at a shaped value, or a one-outlier micro-variance group.

    That is wider than the original rule ("flat at exactly 0.0"), on purpose.
    Once flatness is judged on shaped scores, a raw cold start can be lifted
    to a *flat nonzero* value -- every sample makes the same right-record
    lookup and earns the same 0.13 -- which is std 0 and no gradient, but
    "flat at 0.0" kept it. On a 541-id synthetic audit with a fifth of the
    groups in that state, the narrower rule kept 109 such groups in training.

    `prefer_shaped=True` (the default) judges on `per_scenario_shaped_scores`
    when the audit was run `--with-shaping`, matching `scorecard.load_scores`
    and `data_scorecard --emit-dead-ids`. Pass False to force raw `reward()`
    -- right for a `--no-shaping` run, whose groups are exactly the raw ones.

    A plain id list (one per line, `#` comments allowed -- what
    `data_scorecard --emit-dead-ids` writes) is also accepted and returned
    as-is; the dead/solved split was already decided when it was emitted, so
    `include_solved`, `min_std` and `prefer_shaped` do not apply to it."""
    return zero_variance_ids_with_basis(baseline_path, include_solved, min_std, prefer_shaped)[0]


def zero_variance_ids_with_basis(
    baseline_path: str | Path,
    include_solved: bool = False,
    min_std: float = EFFECTIVE_MIN_STD,
    prefer_shaped: bool = True,
) -> tuple[set[str], str]:
    """`load_zero_variance_ids`, plus which scores it judged on -- so the
    trainer's log can say so rather than leave the reader to assume."""
    text = Path(baseline_path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        ids = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
        return ids, "plain id list"

    if prefer_shaped and data.get("per_scenario_shaped_scores"):
        per_scenario, basis = data["per_scenario_shaped_scores"], "reward() + shaping"
    else:
        per_scenario, basis = data["per_scenario_scores"], "reward() alone"
    dead = {
        scenario_id
        for scenario_id, scores in per_scenario.items()
        if lacks_gradient(scores, min_std, include_solved)
    }
    return dead, basis


def load_blocking_labels(path: str | Path = DEFAULT_LABEL_AUDIT) -> dict[str, list[str]]:
    """`{scenario_id: [blocking label, ...]}` for every scenario the label
    audit covers; an empty list means the gold is consistent with policy.md."""
    data = json.loads(Path(path).read_text())
    return {sid: list(entry.get("blocking") or []) for sid, entry in data["scenarios"].items()}


def load_label_defect_ids(path: str | Path = DEFAULT_LABEL_AUDIT) -> set[str]:
    """Ids whose gold contradicts the retail policy (a non-empty `blocking`
    list in the label audit). 222 of 541 as committed."""
    return {sid for sid, blocking in load_blocking_labels(path).items() if blocking}


def blocking_category(label: str) -> str:
    """`gold_ids_unreachable:item_ids,payment_method_id` -> `gold_ids_unreachable`:
    the suffix names which ids, the prefix is the defect kind worth counting."""
    return label.split(":", 1)[0]


@dataclass
class Exclusions:
    """What a run drops before mixing/splitting, and why. Built once by
    `resolve_exclusions` and used by both grpo_train and zero_shot_baseline
    `--split train/val`, so the two compute the same split by construction --
    before this, the baseline ignored exclusions entirely, and with a 25%
    exclusion list its val set and grpo_train's shared 0 of ~40 ids."""

    label_defects: set[str] = field(default_factory=set)
    zero_variance: set[str] = field(default_factory=set)
    zero_variance_basis: Optional[str] = None
    zero_variance_min_std: Optional[float] = None
    blocking: dict[str, list[str]] = field(default_factory=dict)
    report: list[str] = field(default_factory=list)

    @property
    def ids(self) -> set[str]:
        return self.label_defects | self.zero_variance

    def fingerprint(self) -> str:
        """Short hash of the excluded id set. Recorded by both entry points so
        a before/after can be checked for identical exclusions rather than
        assumed to have them."""
        return hashlib.sha256("\n".join(sorted(self.ids)).encode()).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_label_defects_excluded": len(self.label_defects),
            "n_zero_variance_excluded": len(self.zero_variance - self.label_defects),
            "zero_variance_basis": self.zero_variance_basis,
            "zero_variance_min_std": self.zero_variance_min_std,
            "n_excluded": len(self.ids),
            "excluded_ids_sha256_12": self.fingerprint(),
        }


def resolve_exclusions(
    examples: list[TrainingExample],
    label_audit: Optional[str | Path] = DEFAULT_LABEL_AUDIT,
    keep_label_defects: bool = False,
    zero_variance_from: Optional[str | Path] = None,
    include_solved: bool = False,
    prefer_shaped: bool = True,
    min_std: float = EFFECTIVE_MIN_STD,
) -> Exclusions:
    """Label defects (unless `keep_label_defects`) plus measured zero-variance
    scenarios, restricted to ids actually in `examples`, with a per-category
    report of what is dropped and what remains."""
    corpus_ids = {e.id for e in examples}
    out = Exclusions()

    if label_audit:
        out.blocking = load_blocking_labels(label_audit)
        uncovered = corpus_ids - set(out.blocking)
        if uncovered:
            out.report.append(
                f"label audit {label_audit} does not cover {len(uncovered)} corpus scenarios "
                f"(e.g. {sorted(uncovered)[:3]}); they are treated as clean. Re-run the label "
                "audit if the corpus changed."
            )
        if not keep_label_defects:
            out.label_defects = {sid for sid, b in out.blocking.items() if b} & corpus_ids

    if zero_variance_from:
        ids, out.zero_variance_basis = zero_variance_ids_with_basis(
            zero_variance_from, include_solved=include_solved, min_std=min_std, prefer_shaped=prefer_shaped
        )
        out.zero_variance_min_std = min_std
        out.zero_variance = ids & corpus_ids

    out.report.extend(exclusion_table(examples, out))
    return out


def exclusion_shift(examples: list[TrainingExample], ex: Exclusions) -> dict[str, Any]:
    """What the exclusions do to the corpus's *shape*, not just its size.

    The per-category counts undersell it. Measured on the committed corpus,
    dropping the 222 label defects alone takes the no-call gold share from
    33.6% (182/541) to 56.1% (179/319), and no-call-or-transfer from 53.4% to
    67.4%: the defects sit overwhelmingly on the acting golds (happy_path
    keeps 40 of 110), while ambiguous and refusal golds survive almost whole.
    It also empties three of the 30 cells outright (54 scenarios), which the
    stratified val split then cannot cover."""
    remaining = [e for e in examples if e.id not in ex.ids]
    before, after = summarize(examples), summarize(remaining)
    cells_before = Counter(_cell(e) for e in examples)
    cells_after = Counter(_cell(e) for e in remaining)
    return {
        "before": before,
        "after": after,
        "emptied_cells": {c: n for c, n in sorted(cells_before.items()) if not cells_after[c]},
    }


def _share(summary: dict[str, Any], key: str) -> str:
    n = summary["n"]
    return f"{round(summary[key] * n)}/{n} ({summary[key]:.1%})"


def exclusion_table(examples: list[TrainingExample], ex: Exclusions) -> list[str]:
    """Per-category counts of corpus, dropped (by reason) and remaining, the
    blocking-label kinds among the label-defect drops, then how the
    exclusions shift the no-call share and which cells they empty (see
    `exclusion_shift`). A scenario both defective and zero-variance is counted
    once, as a label defect."""
    if not ex.ids:
        return []
    total = Counter(e.category for e in examples)
    label = Counter(e.category for e in examples if e.id in ex.label_defects)
    zero = Counter(
        e.category for e in examples if e.id in ex.zero_variance and e.id not in ex.label_defects
    )
    lines = [f"{'category':26} {'corpus':>6} {'label_defect':>12} {'zero_var':>8} {'remaining':>9}"]
    for category in sorted(total):
        remaining = total[category] - label[category] - zero[category]
        lines.append(
            f"{category:26} {total[category]:6} {label[category]:12} {zero[category]:8} {remaining:9}"
        )
    n_total, n_label, n_zero = sum(total.values()), sum(label.values()), sum(zero.values())
    lines.append(f"{'TOTAL':26} {n_total:6} {n_label:12} {n_zero:8} {n_total - n_label - n_zero:9}")
    if ex.label_defects:
        kinds = Counter(
            kind
            for sid in ex.label_defects
            for kind in {blocking_category(b) for b in ex.blocking.get(sid, [])}
        )
        lines.append(
            "label-defect drops by blocking label (a scenario can carry several): "
            + ", ".join(f"{k}={v}" for k, v in kinds.most_common())
        )
    shift = exclusion_shift(examples, ex)
    before, after = shift["before"], shift["after"]
    lines.append(
        f"gold = no tool call: {_share(before, 'no_call_fraction')} -> {_share(after, 'no_call_fraction')}; "
        f"no call or transfer: {before['non_acting_fraction']:.1%} -> {after['non_acting_fraction']:.1%}"
    )
    if shift["emptied_cells"]:
        emptied = shift["emptied_cells"]
        lines.append(
            f"cells emptied entirely: {len(emptied)} of {before['n_cells']} ({sum(emptied.values())} "
            "scenarios) -- " + ", ".join(f"{c} ({n})" for c, n in emptied.items())
        )
    return lines


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
