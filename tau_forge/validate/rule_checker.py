"""Phase 3, stage 1: deterministic rule checker for Phase 2 synthetic scenarios.

Cheap, mechanical, no-LLM checks that catch the class of error interactive
gold-answer authoring can still slip past: a scenario JSON that looks right but
whose `expected_tool_calls` doesn't actually re-execute clean against the real
`db.json`, or that violates one of the structural rules the taxonomy/prompt
template promise (category-specific `expected_tool_calls` shape, the one-call
cap, fabricated ids). Run this after every cell (or small batch of cells), not
just at the end -- see README's "Rule checker -- Phase 3, stage 1" section for
the rationale and the two prompt-template fixes it drove.

This is stage 1 only: mechanical correctness. It says nothing about scenario
quality, realism, or difficulty calibration -- that's stage 2 (model checker),
stage 3 (human review), stage 4 (difficulty calibration), none of which are
built yet.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau_forge.envs.retail import RetailEnv
from tau_forge.gen.taxonomy import CATEGORIES, THEMES

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"

READ_TOOLS = {
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "get_user_details",
    "list_all_product_types",
}

REQUIRED_KEYS = {
    "id",
    "category",
    "theme",
    "prior_turns",
    "user_message",
    "expected_tool_calls",
    "expected_tool_calls_verified",
    "distractor_tool",
    "distractor_rationale",
    "ambiguity_note",
}


@dataclass
class Finding:
    scenario_id: str
    severity: str  # "error" | "warning"
    message: str


@dataclass
class CellReport:
    cell: str
    n_scenarios: int = 0
    findings: list[Finding] = field(default_factory=list)
    lookup_only_count: int = 0
    non_empty_calls_count: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.errors


def _err(findings: list[Finding], sid: str, msg: str) -> None:
    findings.append(Finding(sid, "error", msg))


def _warn(findings: list[Finding], sid: str, msg: str) -> None:
    findings.append(Finding(sid, "warning", msg))


def check_scenario(
    scenario: dict[str, Any], expected_category: str, expected_theme: str, env: RetailEnv
) -> list[Finding]:
    findings: list[Finding] = []
    sid = scenario.get("id", "<missing id>")

    missing = REQUIRED_KEYS - scenario.keys()
    if missing:
        _err(findings, sid, f"missing required keys: {sorted(missing)}")
        return findings  # nothing else is safe to check

    if scenario["category"] != expected_category:
        _err(
            findings,
            sid,
            f"category field '{scenario['category']}' != file's cell category '{expected_category}'",
        )
    if scenario["theme"] != expected_theme:
        _err(
            findings,
            sid,
            f"theme field '{scenario['theme']}' != file's cell theme '{expected_theme}'",
        )
    if not sid.startswith(f"{expected_category}__{expected_theme}__"):
        _err(findings, sid, f"id '{sid}' doesn't match cell prefix '{expected_category}__{expected_theme}__'")

    calls = scenario["expected_tool_calls"]
    if not isinstance(calls, list):
        _err(findings, sid, "expected_tool_calls is not a list")
        return findings

    # Global hard cap: at most one tool call. This mirrors policy.md itself --
    # the real agent may only emit one tool call per turn -- so a scenario
    # (one decision point) can never legitimately need more than one. A
    # multi-step need must be represented via prior_turns showing turn 1's
    # call+result already resolved, with expected_tool_calls capturing only
    # the next single call.
    if len(calls) > 1:
        _err(
            findings,
            sid,
            f"expected_tool_calls has {len(calls)} calls; hard cap is 1 "
            "(represent earlier steps via prior_turns, not a chain here)",
        )

    if expected_category == "ambiguous":
        if calls != []:
            _err(findings, sid, "ambiguous scenario must have expected_tool_calls == []")
        if not scenario.get("ambiguity_note"):
            _err(findings, sid, "ambiguous scenario missing non-empty ambiguity_note")

    elif expected_category == "policy_violation":
        if len(calls) not in (0, 1):
            pass  # already caught by hard cap above
        if len(calls) == 1:
            name = calls[0].get("name")
            if name not in READ_TOOLS:
                _err(
                    findings,
                    sid,
                    f"policy_violation's single allowed call must be a READ tool "
                    f"(establishing state for the refusal); got '{name}'",
                )

    elif expected_category == "out_of_scope":
        if len(calls) != 1 or calls[0].get("name") != "transfer_to_human_agents":
            _err(
                findings,
                sid,
                "out_of_scope must have exactly one call to transfer_to_human_agents",
            )
        elif not calls[0].get("arguments", {}).get("summary"):
            _err(findings, sid, "transfer_to_human_agents call missing non-empty 'summary' argument")

    elif expected_category in ("happy_path", "requires_earlier_context"):
        if len(calls) != 1:
            _err(
                findings,
                sid,
                f"{expected_category} scenario must have exactly 1 expected_tool_calls, got {len(calls)}",
            )

    # Distractor sanity.
    distractor = scenario.get("distractor_tool")
    if distractor and not env.has_tool(distractor):
        _err(findings, sid, f"distractor_tool '{distractor}' is not a real tool name")
    if calls and distractor == calls[0].get("name"):
        _err(findings, sid, "distractor_tool is identical to the expected tool call")

    # Re-execution: every call must actually be issuable against the real DB
    # from a fresh env, in order, each succeeding. This is the check that
    # catches fabricated ids, stale state assumptions, and wrong arguments --
    # the interactive-authoring step is supposed to have already ruled these
    # out, this just re-confirms it mechanically and independently.
    if calls:
        fresh_env = RetailEnv()
        for i, call in enumerate(calls):
            name = call.get("name")
            args = call.get("arguments", {})
            if not env.has_tool(name):
                _err(findings, sid, f"call {i} references unknown tool '{name}'")
                break
            extra = fresh_env.extra_arguments(name, args)
            if extra:
                _err(findings, sid, f"call {i} ('{name}') has arguments not in its schema: {extra}")
            result = fresh_env.execute(name, args)
            if not result.ok:
                _err(
                    findings,
                    sid,
                    f"call {i} ('{name}') failed re-execution against real db.json: "
                    f"{result.error_type}: {result.error}",
                )
                break
        if scenario.get("expected_tool_calls_verified") is not True:
            _warn(
                findings,
                sid,
                "expected_tool_calls is non-empty but expected_tool_calls_verified is not true",
            )
    else:
        if scenario.get("expected_tool_calls_verified") not in (False, None):
            _warn(
                findings,
                sid,
                "expected_tool_calls is empty; expected_tool_calls_verified should be false",
            )

    return findings


def check_cell_file(path: Path, env: RetailEnv) -> CellReport:
    stem = path.stem  # "{category}__{theme}"
    parts = stem.split("__", 1)
    category = parts[0]
    theme = parts[1] if len(parts) > 1 else ""
    report = CellReport(cell=stem)

    if category not in CATEGORIES or theme not in THEMES:
        _err(report.findings, stem, f"filename '{stem}' doesn't match a known category__theme cell")
        return report

    try:
        scenarios = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _err(report.findings, stem, f"invalid JSON: {e}")
        return report

    if not isinstance(scenarios, list):
        _err(report.findings, stem, "top-level JSON must be a list")
        return report

    report.n_scenarios = len(scenarios)
    seen_ids: Counter[str] = Counter()

    for scenario in scenarios:
        sid = scenario.get("id", "<missing id>") if isinstance(scenario, dict) else "<not an object>"
        if not isinstance(scenario, dict):
            _err(report.findings, sid, "scenario is not a JSON object")
            continue
        seen_ids[sid] += 1
        report.findings.extend(check_scenario(scenario, category, theme, env))

        calls = scenario.get("expected_tool_calls") or []
        if category in ("happy_path", "requires_earlier_context") and len(calls) == 1:
            report.non_empty_calls_count += 1
            if calls[0].get("name") in READ_TOOLS:
                report.lookup_only_count += 1

    for sid, count in seen_ids.items():
        if count > 1:
            _err(report.findings, sid, f"duplicate id appears {count} times within this cell")

    return report


def run(raw_dir: Path = RAW_DIR) -> list[CellReport]:
    env = RetailEnv()
    files = sorted(raw_dir.glob("*.json"))
    return [check_cell_file(f, env) for f in files]


def print_report(reports: list[CellReport]) -> bool:
    total_scenarios = sum(r.n_scenarios for r in reports)
    total_errors = sum(len(r.errors) for r in reports)
    total_warnings = sum(len(r.warnings) for r in reports)
    passed_scenarios = 0

    for r in reports:
        error_ids = {f.scenario_id for f in r.errors}
        cell_pass = r.n_scenarios - len(error_ids)
        passed_scenarios += cell_pass
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.cell}: {r.n_scenarios} scenarios, "
              f"{len(r.errors)} errors, {len(r.warnings)} warnings")
        if r.non_empty_calls_count:
            share = r.lookup_only_count / r.non_empty_calls_count
            flag = "" if 0.15 <= share <= 0.55 else "  <-- far from ~1/3 target"
            print(f"       lookup-only share: {r.lookup_only_count}/{r.non_empty_calls_count} "
                  f"({share:.0%}){flag}")
        for f in r.errors:
            print(f"       ERROR   {f.scenario_id}: {f.message}")
        for f in r.warnings:
            print(f"       warning {f.scenario_id}: {f.message}")

    print()
    print(f"TOTAL: {passed_scenarios}/{total_scenarios} scenarios clean across {len(reports)} cells "
          f"({total_errors} errors, {total_warnings} warnings)")
    return total_errors == 0


def main() -> int:
    reports = run()
    if not reports:
        print(f"No scenario files found under {RAW_DIR}")
        return 1
    ok = print_report(reports)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
