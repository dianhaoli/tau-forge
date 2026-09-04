"""Phase 3, stage 3: human review sampling + rendering.

Stage 3 is deliberately NOT automated -- deciding whether a scenario is
actually good training data is a human call, not something this session
should try to replace with more machine judgment (stage 2 already did the
machine-judgment pass). What IS automatable, and what this module does, is
making a *sample* review efficient rather than demanding the reviewer read
all 541 scenarios:

  - `build_sample()`: a stratified random sample with a fixed, documented
    seed -- `BASE_PER_CELL` scenarios per cell (every one of the 30
    category x theme cells represented), plus every stage-2-flagged
    scenario in that cell (up to `MAX_FLAGGED_EXTRA` additional ones), so
    the sample deliberately over-represents whatever the model checker
    already found questionable.
  - `render_markdown()`: renders the sample into one readable file --
    prior_turns/user_message/expected_tool_calls/distractor shown side by
    side per scenario, not raw JSON -- so a human can read through it
    without editor-diffing dozens of separate scenario files.
  - `save_stub_results()` / `load_results()` / `record_verdict()`: the
    verdict capture side -- a small, auditable JSON file
    (`data/synthetic/human_review/sample_results.json`) recording what the
    reviewer actually said about each sampled scenario, not just a
    conversation that happened and left no trace.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tau_forge.gen.taxonomy import all_cells
from tau_forge.validate.model_checker import MODEL_CHECK_DIR, load_reports

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"
HUMAN_REVIEW_DIR = REPO_ROOT / "data" / "synthetic" / "human_review"

# Fixed, documented seed -- reruns of build_sample() reproduce the exact same
# sample as long as the underlying raw scenario files haven't changed.
SAMPLE_SEED = 42
BASE_PER_CELL = 4  # 4 * 30 cells = 120 baseline, within the 90-150 target range
MAX_FLAGGED_EXTRA = 3  # cap on how many extra flagged scenarios one cell can add

VERDICTS = ("confirmed_fine", "flagged", "pending")


def _load_cell_scenarios(cell: str, raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    path = raw_dir / f"{cell}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _flagged_ids_by_cell(model_check_dir: Path = MODEL_CHECK_DIR) -> dict[str, set[str]]:
    flagged: dict[str, set[str]] = {}
    if not model_check_dir.exists():
        return flagged
    for report in load_reports(model_check_dir):
        flagged[report.cell] = {f.scenario_id for f in report.flagged}
    return flagged


def sample_cell(
    cell: str,
    scenarios: list[dict[str, Any]],
    flagged_ids: set[str],
    rng: random.Random,
    base_n: int = BASE_PER_CELL,
    max_flagged_extra: int = MAX_FLAGGED_EXTRA,
) -> list[dict[str, Any]]:
    """Sample `base_n` scenarios at random (seeded), then add up to
    `max_flagged_extra` stage-2-flagged scenarios not already picked --
    oversampling what the model checker already found questionable rather
    than trusting a plain random sample to happen to cover it."""
    if not scenarios:
        return []
    pool = list(scenarios)
    rng.shuffle(pool)
    picked = pool[: min(base_n, len(pool))]
    picked_ids = {s["id"] for s in picked}

    if flagged_ids:
        flagged_not_picked = [s for s in scenarios if s["id"] in flagged_ids and s["id"] not in picked_ids]
        rng.shuffle(flagged_not_picked)
        picked = picked + flagged_not_picked[:max_flagged_extra]

    return picked


def build_sample(
    raw_dir: Path = RAW_DIR,
    model_check_dir: Path = MODEL_CHECK_DIR,
    seed: int = SAMPLE_SEED,
    base_n: int = BASE_PER_CELL,
) -> dict[str, list[dict[str, Any]]]:
    flagged_by_cell = _flagged_ids_by_cell(model_check_dir)
    sample: dict[str, list[dict[str, Any]]] = {}
    for category, theme in all_cells():
        cell = f"{category}__{theme}"
        scenarios = _load_cell_scenarios(cell, raw_dir)
        rng = random.Random(f"{seed}:{cell}")  # per-cell deterministic sub-seed, stable regardless of dict order
        sample[cell] = sample_cell(cell, scenarios, flagged_by_cell.get(cell, set()), rng, base_n)
    return sample


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_turns(prior_turns: list[dict[str, str]]) -> str:
    if not prior_turns:
        return "*(no prior turns)*"
    lines = []
    for turn in prior_turns:
        role = turn.get("role", "?")
        lines.append(f"- **{role}**: {turn.get('content', '')}")
    return "\n".join(lines)


def _render_calls(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "*(none -- correct answer is not calling a tool)*"
    lines = []
    for call in calls:
        args = json.dumps(call.get("arguments", {}), indent=None)
        lines.append(f"`{call.get('name')}({args})`")
    return "\n".join(lines)


def render_markdown(
    sample: dict[str, list[dict[str, Any]]], flagged_by_cell: Optional[dict[str, set[str]]] = None
) -> str:
    flagged_by_cell = flagged_by_cell or {}
    total = sum(len(v) for v in sample.values())
    lines = [
        "# Phase 3 stage 3 -- human review sample",
        "",
        f"Stratified sample of {total} scenarios across {len(sample)} cells "
        f"(seed {SAMPLE_SEED}, {BASE_PER_CELL} random + up to {MAX_FLAGGED_EXTRA} "
        "stage-2-flagged per cell). For each scenario: read prior_turns + user_message, "
        "check expected_tool_calls is actually the right next action, check the "
        "distractor is a plausible wrong answer for the stated reason, and (for "
        "ambiguous/policy_violation) check ambiguity_note is specific, not generic.",
        "",
        "Record your verdict in `data/synthetic/human_review/sample_results.json` "
        '(or note it in conversation and have it captured there) -- `"confirmed_fine"` '
        'or `"flagged"` with a note.',
        "",
        "---",
        "",
    ]
    for cell, scenarios in sample.items():
        cell_flagged = flagged_by_cell.get(cell, set())
        lines.append(f"## {cell}")
        lines.append("")
        for s in scenarios:
            flag_marker = " ⚠️ *(stage-2 flagged)*" if s["id"] in cell_flagged else ""
            lines.append(f"### `{s['id']}`{flag_marker}")
            lines.append("")
            lines.append(f"**Prior turns:**\n{_render_turns(s.get('prior_turns', []))}")
            lines.append("")
            lines.append(f"**User message:** {s.get('user_message', '')}")
            lines.append("")
            lines.append(f"**Expected tool call(s):** {_render_calls(s.get('expected_tool_calls', []))}")
            lines.append("")
            lines.append(
                f"**Distractor:** `{s.get('distractor_tool', '')}` -- "
                f"{s.get('distractor_rationale', '')}"
            )
            if s.get("ambiguity_note"):
                lines.append("")
                lines.append(f"**Ambiguity note:** {s['ambiguity_note']}")
            lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


def sample_manifest(sample: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "seed": SAMPLE_SEED,
        "base_per_cell": BASE_PER_CELL,
        "max_flagged_extra": MAX_FLAGGED_EXTRA,
        "total_sampled": sum(len(v) for v in sample.values()),
        "cells": {cell: [s["id"] for s in scenarios] for cell, scenarios in sample.items()},
    }


# ---------------------------------------------------------------------------
# Verdict capture
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    scenario_id: str
    cell: str
    verdict: str  # "confirmed_fine" | "flagged" | "pending"
    note: str = ""
    reviewer: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_stub_results(
    sample: dict[str, list[dict[str, Any]]], out_path: Path, overwrite: bool = False
) -> list[Verdict]:
    """Write (or return, if already present and overwrite=False) a
    'pending' verdict entry for every sampled scenario -- a checklist the
    reviewer's actual verdicts get filled into, so there's a durable record
    of exactly which scenarios were and weren't reviewed."""
    if out_path.exists() and not overwrite:
        existing = load_results(out_path)
        existing_ids = {v.scenario_id for v in existing}
        for cell, scenarios in sample.items():
            for s in scenarios:
                if s["id"] not in existing_ids:
                    existing.append(Verdict(scenario_id=s["id"], cell=cell, verdict="pending"))
        save_results(existing, out_path)
        return existing

    verdicts = [
        Verdict(scenario_id=s["id"], cell=cell, verdict="pending")
        for cell, scenarios in sample.items()
        for s in scenarios
    ]
    save_results(verdicts, out_path)
    return verdicts


def save_results(verdicts: list[Verdict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([v.to_dict() for v in verdicts], indent=2) + "\n")


def load_results(path: Path) -> list[Verdict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Verdict(**d) for d in data]


def record_verdict(
    path: Path,
    scenario_id: str,
    verdict: str,
    note: str = "",
    reviewer: str = "",
) -> list[Verdict]:
    """Update one scenario's verdict in place (by scenario_id) and re-save."""
    if verdict not in ("confirmed_fine", "flagged"):
        raise ValueError(f"verdict must be 'confirmed_fine' or 'flagged', got {verdict!r}")
    verdicts = load_results(path)
    for v in verdicts:
        if v.scenario_id == scenario_id:
            v.verdict = verdict
            v.note = note
            v.reviewer = reviewer
            v.timestamp = datetime.now(timezone.utc).isoformat()
            break
    else:
        raise KeyError(f"scenario_id {scenario_id!r} not found in {path} -- not part of the sample?")
    save_results(verdicts, path)
    return verdicts


def summarize_results(verdicts: list[Verdict]) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(v.verdict for v in verdicts))


def main() -> int:
    sample = build_sample()
    HUMAN_REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = HUMAN_REVIEW_DIR / "sample_manifest.json"
    manifest_path.write_text(json.dumps(sample_manifest(sample), indent=2) + "\n")

    flagged_by_cell = _flagged_ids_by_cell()
    md_path = HUMAN_REVIEW_DIR / "sample.md"
    md_path.write_text(render_markdown(sample, flagged_by_cell))

    results_path = HUMAN_REVIEW_DIR / "sample_results.json"
    verdicts = save_stub_results(sample, results_path)

    total = sum(len(v) for v in sample.values())
    print(f"Sampled {total} scenarios across {len(sample)} cells (seed {SAMPLE_SEED}).")
    print(f"Manifest: {manifest_path}")
    print(f"Readable review file: {md_path}")
    print(f"Verdict checklist ({len(verdicts)} entries, 'pending' until reviewed): {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
