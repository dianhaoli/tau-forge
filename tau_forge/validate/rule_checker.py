"""Phase 3 rule checker (stage 1 of the validation pipeline): re-executes every
scenario's `expected_tool_calls` against the real Phase 1 `RetailEnv` and flags
anything that doesn't hold up mechanically or structurally. No LLM judge, no
human review -- pure re-execution against the real DB and the domain's own
stated rules, so a failure here is unambiguous, not a matter of taste.

Checks per scenario:
  - at most one tool call in `expected_tool_calls` -- policy.md is explicit
    ("You should at most make one tool call at a time"), and the Phase 4
    reward function only grades a single `Action`, so a chained multi-call
    gold answer is both domain-inconsistent and ungradable as written.
  - category-shape rules: `ambiguous` must have zero calls; `policy_violation`
    must have zero calls or exactly one READ-type call; `out_of_scope` must be
    exactly one `transfer_to_human_agents` call.
  - the call actually executes cleanly against a fresh copy of the real
    db.json: known tool, schema-valid arguments, no unrecognized
    (hallucinated) arguments, no exception raised.
  - `expected_tool_calls_verified` is internally consistent with whether
    there's actually anything to verify.

Run as a script: `uv run python3 -m tau_forge.validate.rule_checker`
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau2.domains.retail.data_model import RetailDB
from tau2.environment.toolkit import ToolType

from tau_forge.envs.retail import RetailEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"

ZERO_CALL_CATEGORIES = {"ambiguous"}
ZERO_OR_ONE_READ_CATEGORIES = {"policy_violation"}
EXACT_TRANSFER_CATEGORIES = {"out_of_scope"}


@dataclass
class CheckResult:
    scenario_id: str
    category: str
    ok: bool
    issues: list[str] = field(default_factory=list)


def check_scenario(scenario: dict[str, Any], base_db: RetailDB) -> CheckResult:
    issues: list[str] = []
    calls: list[dict[str, Any]] = scenario.get("expected_tool_calls") or []
    category = scenario.get("category", "?")
    scenario_id = scenario.get("id", "?")

    if len(calls) > 1:
        issues.append(
            f"{len(calls)} chained tool calls in expected_tool_calls -- violates "
            "policy.md's one-tool-call-per-turn rule and isn't gradable by the "
            "single-Action reward function; only the first call would be checked"
        )

    live_env = RetailEnv(db=base_db.model_copy(deep=True))

    if category in ZERO_CALL_CATEGORIES and calls:
        issues.append(f"category={category} requires zero calls, got {len(calls)}")
    elif category in ZERO_OR_ONE_READ_CATEGORIES and len(calls) == 1:
        name = calls[0].get("name")
        if not live_env.has_tool(name) or live_env.tool_type(name) != ToolType.READ:
            issues.append(f"category={category}'s single call {name!r} is not a READ tool")
    elif category in EXACT_TRANSFER_CATEGORIES:
        if len(calls) != 1 or calls[0].get("name") != "transfer_to_human_agents":
            issues.append("category=out_of_scope must be exactly one transfer_to_human_agents call")

    for i, call in enumerate(calls):
        name = call.get("name")
        args = call.get("arguments") or {}
        if not live_env.has_tool(name):
            issues.append(f"call[{i}] ({name}): unknown tool")
            break
        schema_valid, schema_err = live_env.validate_arguments(name, args)
        extra = live_env.extra_arguments(name, args)
        if not schema_valid:
            issues.append(f"call[{i}] ({name}): schema-invalid arguments -- {schema_err}")
        if extra:
            issues.append(f"call[{i}] ({name}): unrecognized/hallucinated arguments {extra}")
        result = live_env.execute(name, args)
        if not result.ok:
            issues.append(f"call[{i}] ({name}): execution failed -- {result.error}")
            break

    verified_flag = scenario.get("expected_tool_calls_verified")
    if calls and verified_flag is False:
        issues.append("expected_tool_calls_verified=false but expected_tool_calls is non-empty")
    if not calls and verified_flag is True:
        issues.append("expected_tool_calls_verified=true but expected_tool_calls is empty")

    return CheckResult(scenario_id, category, ok=not issues, issues=issues)


def check_file(path: Path, base_db: RetailDB) -> list[CheckResult]:
    scenarios = json.loads(path.read_text())
    return [check_scenario(s, base_db) for s in scenarios]


def main() -> None:
    base_db = RetailEnv().snapshot()
    all_results: list[CheckResult] = []
    per_file: dict[str, list[CheckResult]] = {}

    for path in sorted(RAW_DIR.glob("*.json")):
        results = check_file(path, base_db)
        per_file[path.name] = results
        all_results.extend(results)

    total = len(all_results)
    passed = [r for r in all_results if r.ok]
    failed = [r for r in all_results if not r.ok]

    print(f"=== Phase 3 rule checker: {total} scenarios across {len(per_file)} files ===\n")
    for fname, results in per_file.items():
        n_pass = sum(r.ok for r in results)
        print(f"{fname}: {n_pass}/{len(results)} passed")

    print(f"\nTOTAL: {len(passed)}/{total} passed ({len(failed)} flagged)\n")

    if failed:
        print("--- Flagged scenarios ---")
        for r in failed:
            print(f"[{r.category}] {r.scenario_id}")
            for issue in r.issues:
                print(f"    - {issue}")


if __name__ == "__main__":
    main()
