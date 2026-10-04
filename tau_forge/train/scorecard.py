"""Per-cell data diagnosis: which scenarios are carrying the run, and what the
corpus mixture should be as a consequence.

`curriculum.REAL_TASK_ALIGNED_MIX` is a *prior* -- it is shaped like the
benchmark's task distribution, which is an argument about relevance, not a
measurement of whether a given cell actually teaches the policy anything. This
module supplies the missing half: read a `zero_shot_baseline` output and score
every `category__theme` cell on how much usable gradient it produced, then
combine the two into a mixture that is defensible on both axes.

The three numbers per cell
--------------------------
* **yield** -- fraction of the cell's scenarios whose group reward varies
  *enough to matter*: sample std >= `EFFECTIVE_MIN_STD` (0.05), the same std
  TRL logs as `reward_std`. A flat group contributes no gradient regardless of
  its mean, and a nearly-flat one contributes almost none (see "Effective
  variance" below), so this is the fraction of the cell that can move weights.
  The older "varied at all" fraction is still reported as `yield_any_variance`.
* **headroom** -- `1 - mean_score`. How much reward is still on the table. A
  cell at mean 0.95 has almost nothing left to win even where it varies, and
  spending rollouts there buys regression insurance, not improvement.
* **signal** -- `yield x headroom`. The composite, and the thing to rank by.
  High on both is where GRPO compute converts into Phase 8 points.

Effective variance
------------------
"Varied at all" (range > 1e-3) counts a group of fifteen 0.0s and one 0.02 as
alive. Under shaping that shape is common -- `tau_forge.train.shaping` pays at
most 0.15 inside the wrong-tool floor -- and with grpo_train's defaults
(Dr. GRPO, `scale_rewards` off) the advantage is `r - mean` with no division
by std, so such a group's largest advantage is ~0.15 against ~0.5 for a real
0/1 split: a gradient 5-50x weaker. A threshold on the group's std puts a
number on "enough". For a group of 16 with one outlier of size x the sample std
is exactly x/4, so the 0.05 default admits a single sample at reward()'s
smallest tier (0.2, schema-invalid right tool) and rejects a single
shaping-only outlier (<= 0.15 -> std <= 0.0375). Two shaping outliers of 0.15
(std 0.051) clear it, which is the intended reading: repeated partial credit is
signal, one stray micro-reward is not.

Why signal alone must not pick the mixture
------------------------------------------
A cell can score high on signal and still be worth little: `out_of_scope` is
hard for the base model and has plenty of headroom, but only 4 of 114 real
retail tasks ever escalate. Optimizing measured signal alone would rebuild the
lopsided corpus this project is trying to get away from. So `recommend_mix`
multiplies measured signal by a **relevance** prior over categories, and both
factors are reported separately so a surprising recommendation can be traced to
whichever one drove it.

The relevance prior here is derived from task shape, not from failure data.
The stronger version needs the Phase 8 baseline eval: classify which decisions
the base policy actually gets wrong on the real benchmark, and weight the cells
that rehearse those. Until that run exists, this is the best available proxy
and should be labeled as one.

Torch/trl-free.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

# Share of the benchmark each category's shape accounts for. Same reasoning as
# curriculum.REAL_TASK_ALIGNED_MIX (see that module for the counts): real retail
# tasks are won by authenticating and mutating correctly; 3.5% ever escalate and
# 2.7% are purely conversational. Guardrail categories keep a real minority
# because refusing an out-of-policy mutation scores points too.
BENCHMARK_RELEVANCE: dict[str, float] = {
    "happy_path": 0.36,
    "requires_earlier_context": 0.36,
    "policy_violation": 0.15,
    "ambiguous": 0.08,
    "out_of_scope": 0.05,
}

UNIFORM_RELEVANCE: dict[str, float] = {k: 1.0 for k in BENCHMARK_RELEVANCE}

# Below this, a cell is producing so little gradient that rollouts spent on it
# are close to wasted. Not a hard drop -- a flag for the report.
LOW_YIELD = 0.25

# Minimum group sample std for a group to count as *effectively* varying. See
# the module docstring, "Effective variance", for why 0.05.
EFFECTIVE_MIN_STD = 0.05

# Float slack on the std comparison, so a group sitting exactly on the
# threshold (one 0.2 outlier in 16 -> std 0.05 to the last bit) is not decided
# by rounding.
_STD_EPS = 1e-9


def group_std(scores: list[float]) -> float:
    """Sample (Bessel-corrected) std, as TRL's `nanstd` computes the
    `reward_std` it logs -- so a threshold here reads directly against the
    training log. 0.0 for a group of fewer than two samples."""
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    return (sum((s - mean) ** 2 for s in scores) / (n - 1)) ** 0.5


def has_effective_variance(scores: list[float], min_std: float = EFFECTIVE_MIN_STD) -> bool:
    return group_std(scores) >= min_std - _STD_EPS


def is_solved(scores: list[float], min_std: float = EFFECTIVE_MIN_STD) -> bool:
    """A group that is (near-)perfect: mean within one effective-std
    threshold of 1.0 (0.95 by default). Only meaningful for a group *without*
    effective variance, where it separates "already learned" (regression
    insurance, kept by default) from every other flat value. The bar sits
    above every flat value reward() produces short of a near-exact match --
    0.0, the 0.2/0.3 floors, 0.5 -- and above anything shaping can add
    (<= 0.15), so a flat 0.13 or 0.3 group never counts as solved. A flat
    0.97 (right call, slightly-off free text) does: 0.03 of headroom is not
    worth a slot."""
    return bool(scores) and sum(scores) / len(scores) >= 1.0 - max(min_std, 1e-3)


def lacks_gradient(
    scores: list[float], min_std: float = EFFECTIVE_MIN_STD, include_solved: bool = False
) -> bool:
    """The exclusion test `curriculum.load_zero_variance_ids` and
    `dead_scenario_ids` share: no effective variance, and not solved (unless
    `include_solved`). Judged on effective variance rather than on "flat at
    0.0" because the value a flat group sits at is irrelevant to its gradient:
    under shaping, a raw cold start whose sixteen samples all make the same
    right-record lookup sits flat at 0.13 (classify -> stuck_partial), and the
    old flat-at-0.0 rule kept it in training at std 0. Same for a raw plateau
    flat at 0.3, and for a one-outlier micro-variance group (std < 0.05)."""
    if not scores or has_effective_variance(scores, min_std):
        return False
    return include_solved or not is_solved(scores, min_std)


def has_any_variance(scores: list[float]) -> bool:
    """The pre-existing liveness test (zero_shot_baseline, bucket_analysis):
    more than one distinct score at 3 decimals. Kept so the old zero-variance
    fraction stays comparable across audits."""
    return len(set(round(s, 3) for s in scores)) > 1


def variance_summary(
    table: dict[str, list[float]],
    min_std: float = EFFECTIVE_MIN_STD,
    ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Zero-variance and effective-variance counts side by side, over `ids`
    (default: every scenario in `table`). The two fractions do not sum to 1:
    the gap between them is the micro-variance groups -- alive by the old test,
    carrying almost no gradient."""
    wanted = list(table) if ids is None else [i for i in ids if i in table]
    groups = [table[i] for i in wanted if table[i]]
    n = len(groups)
    zero = sum(1 for g in groups if not has_any_variance(g))
    effective = sum(1 for g in groups if has_effective_variance(g, min_std))
    return {
        "n": n,
        "zero_variance_count": zero,
        "zero_variance_fraction": zero / n if n else 0.0,
        "effective_variance_count": effective,
        "effective_variance_fraction": effective / n if n else 0.0,
        "micro_variance_count": n - zero - effective,
        "effective_variance_min_std": min_std,
    }


def variance_by_label_status(
    table: dict[str, list[float]],
    blocking: dict[str, list[str]],
    min_std: float = EFFECTIVE_MIN_STD,
) -> dict[str, dict[str, Any]]:
    """`variance_summary` split into scenarios with no blocking label-audit
    entry ("clean") and scenarios whose gold contradicts policy.md
    ("label_defect"). A defective gold makes every policy-following sample
    score the same 0, so the two populations are expected to differ, and a
    variance figure over their union says little about either. Scenarios the
    label audit does not cover count as clean."""
    defective = {sid for sid in table if blocking.get(sid)}
    return {
        "clean": variance_summary(table, min_std, [sid for sid in table if sid not in defective]),
        "label_defect": variance_summary(table, min_std, sorted(defective)),
    }


@dataclass
class CellStats:
    cell: str
    category: str
    theme: str
    n: int
    mean_score: float
    n_usable: int  # group reward varied at all (range > 1e-3)
    n_effective: int  # group std >= min_std -> produces a gradient worth the rollouts
    n_cold: int  # flat at 0.0 -> needs prompting/SFT, not more sampling
    n_solved: int  # flat at 1.0 -> already learned, harmless to keep
    n_stuck: int  # flat at an intermediate value -> reward plateau, see below

    @property
    def yield_(self) -> float:
        """Effective yield: what `signal` and `recommend_mix` rank by."""
        return self.n_effective / self.n if self.n else 0.0

    @property
    def yield_any_variance(self) -> float:
        """The pre-effective-variance yield, kept visible: the gap between
        this and `yield_` is the cell's micro-variance groups."""
        return self.n_usable / self.n if self.n else 0.0

    @property
    def headroom(self) -> float:
        return max(0.0, 1.0 - self.mean_score)

    @property
    def signal(self) -> float:
        return self.yield_ * self.headroom

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "category": self.category,
            "theme": self.theme,
            "n": self.n,
            "mean_score": round(self.mean_score, 4),
            "yield": round(self.yield_, 4),
            "yield_any_variance": round(self.yield_any_variance, 4),
            "headroom": round(self.headroom, 4),
            "signal": round(self.signal, 4),
            "usable": self.n_usable,
            "effective": self.n_effective,
            "cold_start": self.n_cold,
            "already_solved": self.n_solved,
            "stuck_partial": self.n_stuck,
            "low_yield": self.yield_ < LOW_YIELD,
        }


def classify(scores: list[float], tolerance: float = 1e-3) -> str:
    """One scenario's group -> which bucket it falls in. `stuck_partial` is its
    own bucket rather than lumped with cold starts because it means something
    different: the policy reliably reaches the same *partial* outcome, which is
    usually a reward plateau (e.g. reward.py floors both 'wrong record' and
    'padded free-text summary' at 0.300) rather than an inability to improve."""
    if not scores:
        return "empty"
    if max(scores) - min(scores) > tolerance:
        return "usable"
    value = scores[0]
    if value <= tolerance:
        return "cold_start"
    if value >= 1.0 - tolerance:
        return "already_solved"
    return "stuck_partial"


def _split_id(scenario_id: str) -> tuple[str, str]:
    """Scenario ids are `<category>__<theme>__<n>`; the cell is the first two."""
    parts = scenario_id.split("__")
    category = parts[0]
    theme = parts[1] if len(parts) > 2 else "unknown"
    return category, theme


def score_cells(
    per_scenario_scores: dict[str, list[float]], min_std: float = EFFECTIVE_MIN_STD
) -> list[CellStats]:
    grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for scenario_id, scores in per_scenario_scores.items():
        grouped[_split_id(scenario_id)].append(scores)

    stats: list[CellStats] = []
    for (category, theme), groups in sorted(grouped.items()):
        counts = defaultdict(int)
        effective = 0
        flat_scores: list[float] = []
        for scores in groups:
            counts[classify(scores)] += 1
            effective += has_effective_variance(scores, min_std)
            flat_scores.extend(scores)
        stats.append(
            CellStats(
                cell=f"{category}__{theme}",
                category=category,
                theme=theme,
                n=len(groups),
                mean_score=sum(flat_scores) / len(flat_scores) if flat_scores else 0.0,
                n_usable=counts["usable"],
                n_effective=effective,
                n_cold=counts["cold_start"],
                n_solved=counts["already_solved"],
                n_stuck=counts["stuck_partial"],
            )
        )
    return stats


def category_signal(stats: Iterable[CellStats]) -> dict[str, float]:
    """Scenario-count-weighted mean signal per category, so a category is not
    flattered by one tiny high-signal cell."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for cell in stats:
        totals[cell.category] += cell.signal * cell.n
        counts[cell.category] += cell.n
    return {c: (totals[c] / counts[c] if counts[c] else 0.0) for c in totals}


def recommend_mix(
    stats: Iterable[CellStats],
    relevance: Optional[dict[str, float]] = None,
    floor: float = 0.03,
) -> dict[str, float]:
    """Measured signal x benchmark relevance, normalized to shares.

    `floor` keeps every present category at a small non-zero share. Zeroing a
    category outright is a bet that its behavior never matters, and the
    guardrail categories in particular are cheap insurance against training a
    policy that mutates when it should refuse."""
    relevance = BENCHMARK_RELEVANCE if relevance is None else relevance
    signal = category_signal(stats)
    raw = {c: signal.get(c, 0.0) * relevance.get(c, 0.0) for c in signal}

    total = sum(raw.values())
    if total <= 0:
        # Nothing measured any signal at all -- fall back to relevance alone
        # rather than emitting an all-zero mixture.
        present = {c: relevance.get(c, 0.0) for c in signal}
        subtotal = sum(present.values()) or 1.0
        return {c: v / subtotal for c, v in present.items()}

    mix = {c: v / total for c, v in raw.items()}
    lifted = {c: max(v, floor) for c, v in mix.items()}
    subtotal = sum(lifted.values())
    return {c: round(v / subtotal, 4) for c, v in lifted.items()}


def dead_scenario_ids(
    per_scenario_scores: dict[str, list[float]],
    include_solved: bool = False,
    min_std: float = EFFECTIVE_MIN_STD,
) -> list[str]:
    """Ids `lacks_gradient` flags -- the same rule grpo_train
    `--exclude-zero-variance-from` applies to an audit JSON, so the emitted
    list and the audit it came from exclude the same scenarios. It used to be
    `classify() == cold_start`, which missed flat-but-nonzero shaped groups."""
    return sorted(
        sid
        for sid, scores in per_scenario_scores.items()
        if lacks_gradient(scores, min_std, include_solved)
    )


def load_scores(path: str | Path, prefer_shaped: bool = True) -> tuple[dict[str, list[float]], str]:
    """Reads a `zero_shot_baseline` output. Prefers the shaped scores when the
    audit was run with `--with-shaping`, because that is what GRPO will
    actually see; returns which one it used so the report can say so."""
    data = json.loads(Path(path).read_text())
    if prefer_shaped and data.get("per_scenario_shaped_scores"):
        return data["per_scenario_shaped_scores"], "reward() + shaping"
    return data["per_scenario_scores"], "reward() alone"


def load_blocking_labels(path: str | Path) -> Optional[dict[str, list[str]]]:
    """The per-scenario blocking label-audit entries a `zero_shot_baseline`
    run recorded (`per_scenario_blocking_labels`), or None for an audit taken
    before they were recorded -- callers then fall back to
    `curriculum.load_blocking_labels` on the label audit itself."""
    data = json.loads(Path(path).read_text())
    return data.get("per_scenario_blocking_labels")
