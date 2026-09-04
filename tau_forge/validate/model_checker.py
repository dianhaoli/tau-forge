"""Phase 3, stage 2: LLM-based quality judge for Phase 2 synthetic scenarios.

Stage 1 (`rule_checker.py`) is mechanical -- it can confirm a scenario's
`expected_tool_calls` re-executes cleanly, but it cannot judge whether the
scenario is any *good*: whether the narrative actually motivates the gold
answer, whether the distractor is a plausible wrong answer for the right
reason, or whether an ambiguity_note is specific rather than generic
boilerplate. That's a judgment call, so it's delegated to an LLM judge --
following this repo's established pattern (see README's "Synthetic
generation" sections) of using Task-tool subagents to do judgment-heavy work
in parallel batches, one subagent per group of cells.

This module does NOT call an LLM itself -- there is no such API available to
plain Python in this repo. It provides:

  - `render_model_check_prompt()`: the self-contained brief a subagent judge
    needs (real tool schemas/policy.md, same inputs `prompt_template.py`
    already assembles, plus the cell's scenarios) -- reused across however
    many cells one subagent is assigned, the same way `prompt_template.py` is
    reused across generation subagents.
  - `ScenarioFinding` / `CellModelCheckReport`: the report shape a judge
    subagent's output is expected to match, deliberately following the
    rule checker's `Finding`/`CellReport` shape philosophy (one dataclass
    per per-scenario finding, one per per-cell report) so this composes with
    the same kind of tooling instead of inventing a new report format.
  - `load_reports()` / `validate_report_shape()` / `print_report()`: loads
    the JSON files subagent judges write to `data/synthetic/model_check/`,
    validates they actually match the expected shape (every scenario in the
    cell accounted for exactly once, severities from a fixed vocabulary),
    and aggregates/prints a summary -- the same role `rule_checker.print_report`
    plays for stage 1.

Stage 2 does NOT "fix" flagged scenarios -- it only reports issues. Deciding
what (if anything) to regenerate is a human call, stage 3.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau_forge.gen.taxonomy import CATEGORIES, THEMES

REPO_ROOT = Path(__file__).resolve().parents[2]
_TAU2 = REPO_ROOT / "third_party" / "tau2-bench"
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"
MODEL_CHECK_DIR = REPO_ROOT / "data" / "synthetic" / "model_check"

SEVERITIES = ("none", "minor", "major")


def _read(relpath: str) -> str:
    return (_TAU2 / relpath).read_text()


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


@dataclass
class ScenarioFinding:
    scenario_id: str
    issues: list[str] = field(default_factory=list)
    severity: str = "none"  # "none" | "minor" | "major"

    def to_dict(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario_id, "issues": self.issues, "severity": self.severity}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenarioFinding":
        return cls(
            scenario_id=d["scenario_id"],
            issues=list(d.get("issues", [])),
            severity=d.get("severity", "none"),
        )


@dataclass
class CellModelCheckReport:
    cell: str
    findings: list[ScenarioFinding] = field(default_factory=list)

    @property
    def major(self) -> list[ScenarioFinding]:
        return [f for f in self.findings if f.severity == "major"]

    @property
    def minor(self) -> list[ScenarioFinding]:
        return [f for f in self.findings if f.severity == "minor"]

    @property
    def flagged(self) -> list[ScenarioFinding]:
        """Anything with severity != 'none' (has issues worth a human look)."""
        return [f for f in self.findings if f.severity != "none"]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_model_check_prompt(cells: list[tuple[str, str]], n_scenarios_hint: int | None = None) -> str:
    """Render the brief for a subagent judging one or more (category, theme)
    cells. Batching multiple cells into one subagent call is fine (and
    encouraged for cells sharing a category, since the judgment axes below
    are the same for all themes of a category) -- this is reused verbatim
    for a single cell or a batch, mirroring `prompt_template.render_cell_prompt`.
    """
    for category, theme in cells:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Known: {sorted(CATEGORIES)}")
        if theme not in THEMES:
            raise ValueError(f"Unknown theme: {theme}. Known: {sorted(THEMES)}")

    tools_source = _read("src/tau2/domains/retail/tools.py")
    policy_text = _read("data/tau2/domains/retail/policy.md")
    data_model_source = _read("src/tau2/domains/retail/data_model.py")

    cells_block = "\n".join(
        f"- `{category}__{theme}`: read scenarios from "
        f"`data/synthetic/raw/{category}__{theme}.json`, write findings to "
        f"`data/synthetic/model_check/{category}__{theme}.json`"
        for category, theme in cells
    )
    categories_here = sorted({c for c, _ in cells})
    category_defs = "\n".join(f"- **{c}**: {CATEGORIES[c]}" for c in categories_here)

    return f"""You are the quality judge for synthetic tool-calling training scenarios in
the tau2-bench retail domain (Phase 3, stage 2 of this project -- "the model
checker"). Stage 1 (a deterministic rule checker) already confirmed every
scenario below is mechanically valid: schema-correct, and its
`expected_tool_calls` actually re-executes cleanly against the real
`db.json`. Your job is different and cannot be done mechanically: judge
whether each scenario is actually a *good* piece of training data.

## Working environment

Repo root: {REPO_ROOT} (already fully set up -- a `uv`-managed venv exists,
don't reinstall or re-sync anything).

## Tools available (full source, verbatim -- the exact interface the agent sees)

```python
{tools_source}
```

## Agent policy (governs what counts as a policy violation, ambiguity, etc.)

{policy_text}

## Database schema

```python
{data_model_source}
```

## Category definitions for the cell(s) you're judging

{category_defs}

## Your assignment

For each cell below, read its scenario file, and for EVERY scenario in it,
judge these axes (skip an axis where it doesn't apply to that scenario, as
noted):

(a) **Narrative-answer fit**: do `prior_turns` + `user_message` actually
    motivate `expected_tool_calls` as the correct next action? A scenario
    can be mechanically valid (the call executes) while still being a bad
    training example if nothing in the narrative actually calls for that
    specific action, or if the narrative is confusing/contradictory.
(b) **Distractor quality**: is `distractor_tool` a plausible wrong answer
    someone might actually pick given this narrative (not a random unrelated
    tool), and does `distractor_rationale` correctly and specifically explain
    why it's wrong for *this* scenario (not a generic boilerplate reason)?
(c) **Ambiguity/policy specificity** (ambiguous and policy_violation
    scenarios only, skip for other categories): is `ambiguity_note` correct
    and specific to this scenario's actual underspecification/violation, not
    a generic restatement like "this is ambiguous" or "this violates
    policy"?

For each scenario, decide an overall `severity`:
- `"none"` -- no real issues, this is good training data.
- `"minor"` -- something worth a human glancing at, but plausibly fine
  (mildly generic distractor_rationale, slightly thin motivation) --
  wouldn't block using this scenario, just a lower-confidence one.
- `"major"` -- a real problem that would teach the wrong thing or waste a
  training example (narrative doesn't actually support the gold answer,
  distractor isn't actually a plausible wrong answer, ambiguity_note is
  wrong or is boilerplate that doesn't explain the actual ambiguity).

List concrete `issues` (short strings, one per distinct problem found) for
any scenario with severity `"minor"` or `"major"` -- empty list for
`"none"`. Be specific and cite what's actually wrong, not vague hedging.

**Do not "fix" anything.** You are a judge, not an editor -- report issues,
don't rewrite scenarios or regenerate data. Deciding what to regenerate is a
separate, human-driven stage.

## Cells to judge

{cells_block}

## Output format

For EACH cell, write a JSON array to its output path (create parent
directories if needed) -- one object per scenario in that cell's file, same
order, with EVERY scenario id from the input file accounted for exactly
once:

```json
[
  {{"scenario_id": "{cells[0][0]}__{cells[0][1]}__001", "issues": [], "severity": "none"}},
  {{"scenario_id": "{cells[0][0]}__{cells[0][1]}__002", "issues": ["distractor_rationale is generic boilerplate, doesn't reference this order's actual state"], "severity": "minor"}}
]
```

Before reporting done, verify your own output for each cell you were
assigned: same scenario_id set as the input file (no extras, none missing,
no duplicates), `severity` is one of `"none"`/`"minor"`/`"major"`, valid
JSON. You can sanity check this yourself with:
`uv run python3 -c "from tau_forge.validate.model_checker import load_reports, validate_report_shape; [print(r.cell, validate_report_shape(r)) for r in load_reports()]"`

When done, report back: total scenarios judged, how many flagged
minor/major per cell, and a couple of representative examples of real
issues found (not just a count) so a human reviewer knows what to expect.
"""


# ---------------------------------------------------------------------------
# Loading + validation of subagent output
# ---------------------------------------------------------------------------


def _raw_scenario_ids(cell: str) -> list[str]:
    path = RAW_DIR / f"{cell}.json"
    if not path.exists():
        return []
    scenarios = json.loads(path.read_text())
    return [s["id"] for s in scenarios if isinstance(s, dict) and "id" in s]


def validate_report_shape(report: CellModelCheckReport) -> list[str]:
    """Structural checks on a subagent's output for one cell: every raw
    scenario id present exactly once, no unknown ids, severities from the
    fixed vocabulary. Returns a list of problem descriptions (empty = clean).
    """
    problems: list[str] = []
    expected_ids = _raw_scenario_ids(report.cell)
    if not expected_ids:
        problems.append(f"no raw scenario file found for cell '{report.cell}' (or it's empty)")
        return problems

    expected_set = set(expected_ids)
    seen: dict[str, int] = {}
    for f in report.findings:
        seen[f.scenario_id] = seen.get(f.scenario_id, 0) + 1
        if f.severity not in SEVERITIES:
            problems.append(f"{f.scenario_id}: unknown severity '{f.severity}'")
        if f.severity == "none" and f.issues:
            problems.append(f"{f.scenario_id}: severity 'none' but issues list is non-empty")
        if f.severity != "none" and not f.issues:
            problems.append(f"{f.scenario_id}: severity '{f.severity}' but issues list is empty")

    for sid, count in seen.items():
        if count > 1:
            problems.append(f"duplicate finding for scenario_id '{sid}' ({count} times)")
        if sid not in expected_set:
            problems.append(f"finding for unknown scenario_id '{sid}' (not in {report.cell}.json)")

    missing = expected_set - seen.keys()
    if missing:
        problems.append(f"missing findings for {len(missing)} scenario id(s): {sorted(missing)[:5]}...")

    return problems


def load_reports(model_check_dir: Path = MODEL_CHECK_DIR) -> list[CellModelCheckReport]:
    reports = []
    for path in sorted(model_check_dir.glob("*.json")):
        cell = path.stem
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            reports.append(CellModelCheckReport(cell=cell, findings=[]))
            print(f"WARNING: {path} is not valid JSON: {e}", file=sys.stderr)
            continue
        findings = [ScenarioFinding.from_dict(d) for d in data]
        reports.append(CellModelCheckReport(cell=cell, findings=findings))
    return reports


def print_report(reports: list[CellModelCheckReport]) -> bool:
    total = sum(len(r.findings) for r in reports)
    total_major = sum(len(r.major) for r in reports)
    total_minor = sum(len(r.minor) for r in reports)
    all_ok = True

    for r in reports:
        problems = validate_report_shape(r)
        status = "OK" if not problems else "SHAPE-INVALID"
        if problems:
            all_ok = False
        print(
            f"[{status}] {r.cell}: {len(r.findings)} scenarios judged, "
            f"{len(r.major)} major, {len(r.minor)} minor"
        )
        for p in problems:
            print(f"       shape problem: {p}")
        for f in r.major:
            print(f"       MAJOR   {f.scenario_id}: {'; '.join(f.issues)}")

    print()
    print(
        f"TOTAL: {total} scenarios judged across {len(reports)} cells "
        f"({total_major} major, {total_minor} minor, "
        f"{total - total_major - total_minor} clean)"
    )
    return all_ok


def main() -> int:
    reports = load_reports()
    if not reports:
        print(f"No model-check reports found under {MODEL_CHECK_DIR} -- stage 2 hasn't been run yet.")
        return 1
    ok = print_report(reports)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
