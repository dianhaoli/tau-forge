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
* **yield** -- fraction of the cell's scenarios whose group reward varied at
  all. A flat group contributes no gradient regardless of its mean, so this is
  the fraction of the cell that can move weights.
* **headroom** -- `1 - mean_score`. How much reward is still on the table. A
  cell at mean 0.95 has almost nothing left to win even where it varies, and
  spending rollouts there buys regression insurance, not improvement.
* **signal** -- `yield x headroom`. The composite, and the thing to rank by.
  High on both is where GRPO compute converts into Phase 8 points.

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


@dataclass
class CellStats:
    cell: str
    category: str
    theme: str
    n: int
    mean_score: float
    n_usable: int  # group reward varied -> produces a gradient
    n_cold: int  # flat at 0.0 -> needs prompting/SFT, not more sampling
    n_solved: int  # flat at 1.0 -> already learned, harmless to keep
    n_stuck: int  # flat at an intermediate value -> reward plateau, see below

    @property
    def yield_(self) -> float:
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
            "headroom": round(self.headroom, 4),
            "signal": round(self.signal, 4),
            "usable": self.n_usable,
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


def score_cells(per_scenario_scores: dict[str, list[float]]) -> list[CellStats]:
    grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for scenario_id, scores in per_scenario_scores.items():
        grouped[_split_id(scenario_id)].append(scores)

    stats: list[CellStats] = []
    for (category, theme), groups in sorted(grouped.items()):
        counts = defaultdict(int)
        flat_scores: list[float] = []
        for scores in groups:
            counts[classify(scores)] += 1
            flat_scores.extend(scores)
        stats.append(
            CellStats(
                cell=f"{category}__{theme}",
                category=category,
                theme=theme,
                n=len(groups),
                mean_score=sum(flat_scores) / len(flat_scores) if flat_scores else 0.0,
                n_usable=counts["usable"],
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
    per_scenario_scores: dict[str, list[float]], include_solved: bool = False
) -> list[str]:
    wanted = {"cold_start"} | ({"already_solved", "stuck_partial"} if include_solved else set())
    return sorted(sid for sid, scores in per_scenario_scores.items() if classify(scores) in wanted)


def load_scores(path: str | Path, prefer_shaped: bool = True) -> tuple[dict[str, list[float]], str]:
    """Reads a `zero_shot_baseline` output. Prefers the shaped scores when the
    audit was run with `--with-shaping`, because that is what GRPO will
    actually see; returns which one it used so the report can say so."""
    data = json.loads(Path(path).read_text())
    if prefer_shaped and data.get("per_scenario_shaped_scores"):
        return data["per_scenario_shaped_scores"], "reward() + shaping"
    return data["per_scenario_scores"], "reward() alone"
