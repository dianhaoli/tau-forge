"""Phase 3, stage 4: difficulty calibration for Phase 2 synthetic scenarios.

Every scenario here is a static single-decision-point snapshot, not a live
multi-turn conversation (see README's "Data prep + harness" section for why
that split exists) -- so there's no notion of "how many turns did it take"
to infer difficulty from. Instead this combines four *scenario features*,
each with a documented reason to correlate with how hard the decision
actually is:

1. **Category** -- `ambiguous`/`policy_violation` require the harder judgment
   call of recognizing when NOT to act (no gold tool call to lean on, no
   partial credit for "close enough"); `requires_earlier_context` requires
   tracking state across turns; `happy_path` is the mechanical
   lookup-then-execute case; `out_of_scope` requires only recognizing a
   scope boundary before a fixed-shape call.
2. **Action type** -- a mutating (WRITE) call has more ways to get subtly
   wrong than a read-only lookup, per the Phase 4 reward function's tiered
   scoring (schema-valid-but-wrong-record, right-record-wrong-field, etc.
   all apply to WRITE calls and don't to READ calls); withholding a call
   entirely (the `[]` case) requires the same judgment-call difficulty as
   category (1) already captures, scored here too since it's a fact about
   *this* scenario's answer shape, not just its category label.
3. **Distractor closeness** -- a distractor in the same tool "family" as the
   correct answer (e.g. `exchange_delivered_order_items` vs.
   `return_delivered_order_items`, both order-mutating) is harder to rule
   out than an unrelated one (e.g. `calculate`). For `[]`-answer scenarios
   there's no "correct tool" to compare against, so a WRITE-tool distractor
   is scored harder than a READ/GENERIC one -- calling it at all would be
   the specific mistake the scenario is designed to tempt.
4. **Stage 2 flags** -- a scenario the model checker had to flag (minor or
   major) is treated as genuinely harder: if a careful LLM judge found
   something to question, a policy model is more likely to stumble on it
   too.

Output is a **sidecar file per cell** (`data/synthetic/difficulty/<cell>.json`,
list of `{id, difficulty_score, difficulty_label, components}`) rather than a
mutated field on the raw scenario JSON -- the raw files already passed
stage-1 review and are the audit trail for the generation sweep; keeping
difficulty as a derived, regenerable-from-scratch sidecar avoids touching
that record and makes it trivial to recompute if the scoring formula
changes (e.g. after stage 3 human review surfaces something the formula
should weight differently). `difficulty_score` is a float in [0, 1] (higher
= harder) for Phase 7 curriculum ordering / stratified batch sampling;
`difficulty_label` is a fixed easy/medium/hard tag derived from it via the
thresholds below, for anyone who wants a coarser bucket.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tau2.environment.toolkit import ToolType

from tau_forge.envs.retail import RetailEnv
from tau_forge.gen.taxonomy import CATEGORIES, THEMES
from tau_forge.validate.model_checker import MODEL_CHECK_DIR, load_reports

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"
DIFFICULTY_DIR = REPO_ROOT / "data" / "synthetic" / "difficulty"

# ---------------------------------------------------------------------------
# Tool families -- grouped by what a distractor sharing the family actually
# shares with the correct action (same target-record type + same rough
# "kind" of mutation), since that's what makes ruling it out hard.
# ---------------------------------------------------------------------------

TOOL_FAMILIES: dict[str, str] = {
    # Order-mutating: all WRITE tools that act on an existing order. The
    # richest source of same-family distractors in this domain (see
    # README's order_state_confusion findings) -- cancel vs. modify-items
    # vs. modify-address vs. modify-payment vs. exchange vs. return all
    # look superficially similar and only one is valid for a given order's
    # actual current status.
    "cancel_pending_order": "order_mutation",
    "modify_pending_order_address": "order_mutation",
    "modify_pending_order_items": "order_mutation",
    "modify_pending_order_payment": "order_mutation",
    "exchange_delivered_order_items": "order_mutation",
    "return_delivered_order_items": "order_mutation",
    # User-mutating: different target-record type from the order family
    # above, even though it's also a WRITE.
    "modify_user_address": "user_mutation",
    # Identity/authentication lookups.
    "find_user_id_by_name_zip": "identity_lookup",
    "find_user_id_by_email": "identity_lookup",
    # General READ lookups (order/product/item/user details, catalog list).
    "get_order_details": "read_lookup",
    "get_product_details": "read_lookup",
    "get_item_details": "read_lookup",
    "get_user_details": "read_lookup",
    "list_all_product_types": "read_lookup",
    # Generic, not record-mutating.
    "calculate": "generic",
    "transfer_to_human_agents": "generic",
}


def tool_family(name: Optional[str]) -> str:
    if name is None:
        return "none"
    return TOOL_FAMILIES.get(name, "unknown")


# ---------------------------------------------------------------------------
# Component scores, each in [0, 1]
# ---------------------------------------------------------------------------

# Base difficulty by category: ambiguous/policy_violation require recognizing
# when NOT to act (no gold call to fall back on, no partial credit for
# "close enough" the way a mutating call's tiered scoring gives); happy_path
# is the mechanical baseline; out_of_scope only requires spotting a scope
# boundary before a fixed-shape transfer call.
CATEGORY_BASE: dict[str, float] = {
    "happy_path": 0.20,
    "out_of_scope": 0.30,
    "requires_earlier_context": 0.55,
    "policy_violation": 0.70,
    "ambiguous": 0.75,
}

DIFFICULTY_THRESHOLDS = (0.40, 0.65)  # score < low -> easy, < high -> medium, else hard

MODEL_CHECK_BUMP = {"none": 0.0, "minor": 0.07, "major": 0.15}


def _action_type_score(env: RetailEnv, calls: list[dict[str, Any]]) -> tuple[float, str]:
    """How hard is *this scenario's particular answer shape* to get right."""
    if not calls:
        # The empty-answer case: correct behavior is refusing/asking, not a
        # tool call -- inherently the hardest answer shape, since there's no
        # partial credit for a near-miss the way a WRITE call's tiered
        # scoring provides (Phase 4: message-only-correct scenarios score
        # the rollout 0 for ANY tool call, right or wrong).
        return 0.75, "no_call"
    name = calls[0].get("name")
    if name == "transfer_to_human_agents":
        # Fixed-shape call (order_id-free, one free-text `summary` arg) --
        # easy to get the *call* right once scope is correctly recognized.
        return 0.30, "transfer"
    if env.tool_mutates_state(name):
        # Mutating: more ways to get subtly wrong per the Phase 4 reward
        # function's tiered scoring (wrong record, right record wrong
        # critical field, right record wrong graded field, ...).
        return 0.80, "mutating"
    return 0.30, "read_only"


def _distractor_score(env: RetailEnv, expected_name: Optional[str], distractor: Optional[str]) -> tuple[float, str]:
    if not distractor:
        return 0.0, "no_distractor"
    if expected_name is not None:
        same_family = tool_family(expected_name) == tool_family(distractor)
        if same_family:
            return 1.0, "same_family_as_answer"
        if env.has_tool(distractor) and env.tool_mutates_state(distractor):
            # Different family, but still a mutating tool -- calling it
            # wrongly is a worse mistake than a read-only distractor would
            # be, so it's not as easy to dismiss as a totally unrelated
            # generic tool.
            return 0.5, "different_family_but_mutating"
        return 0.25, "different_family_read_or_generic"
    # No expected tool call to compare against (ambiguous / policy_violation
    # with empty expected_tool_calls) -- score by how dangerous the
    # distractor itself is: a WRITE-tool distractor is the specific mistake
    # ("acted when it should have asked/refused") these scenarios are
    # designed to tempt.
    if env.has_tool(distractor) and env.tool_mutates_state(distractor):
        return 0.9, "mutating_distractor_no_answer_call"
    if env.has_tool(distractor) and env.tool_type(distractor) == ToolType.READ:
        return 0.4, "read_distractor_no_answer_call"
    return 0.3, "generic_distractor_no_answer_call"


def difficulty_label(score: float) -> str:
    low, high = DIFFICULTY_THRESHOLDS
    if score < low:
        return "easy"
    if score < high:
        return "medium"
    return "hard"


@dataclass
class ScenarioDifficulty:
    scenario_id: str
    difficulty_score: float
    difficulty_label: str
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.scenario_id,
            "difficulty_score": round(self.difficulty_score, 4),
            "difficulty_label": self.difficulty_label,
            "components": self.components,
        }


def compute_difficulty(
    scenario: dict[str, Any],
    env: RetailEnv,
    model_check_severity: str = "none",
) -> ScenarioDifficulty:
    category = scenario["category"]
    calls = scenario.get("expected_tool_calls") or []
    expected_name = calls[0].get("name") if calls else None
    distractor = scenario.get("distractor_tool")

    category_score = CATEGORY_BASE.get(category, 0.5)
    action_score, action_kind = _action_type_score(env, calls)
    distractor_score, distractor_kind = _distractor_score(env, expected_name, distractor)
    model_check_bump = MODEL_CHECK_BUMP.get(model_check_severity, 0.0)

    raw = 0.40 * category_score + 0.30 * action_score + 0.20 * distractor_score + model_check_bump
    score = max(0.0, min(1.0, raw))

    return ScenarioDifficulty(
        scenario_id=scenario["id"],
        difficulty_score=score,
        difficulty_label=difficulty_label(score),
        components={
            "category": category,
            "category_score": category_score,
            "action_kind": action_kind,
            "action_score": action_score,
            "distractor_kind": distractor_kind,
            "distractor_score": distractor_score,
            "model_check_severity": model_check_severity,
            "model_check_bump": model_check_bump,
        },
    )


# ---------------------------------------------------------------------------
# Batch computation over all cells
# ---------------------------------------------------------------------------


def _model_check_severity_by_id(model_check_dir: Path = MODEL_CHECK_DIR) -> dict[str, str]:
    """scenario_id -> severity, from every stage-2 report found. Scenarios
    with no stage-2 report at all (stage 2 not run yet, or this scenario's
    cell wasn't judged) default to "none" downstream -- difficulty is still
    computable from the other three signals alone."""
    severities: dict[str, str] = {}
    if not model_check_dir.exists():
        return severities
    for report in load_reports(model_check_dir):
        for finding in report.findings:
            severities[finding.scenario_id] = finding.severity
    return severities


def compute_cell_difficulty(
    path: Path, env: RetailEnv, severity_by_id: dict[str, str]
) -> list[ScenarioDifficulty]:
    scenarios = json.loads(path.read_text())
    return [
        compute_difficulty(s, env, model_check_severity=severity_by_id.get(s["id"], "none"))
        for s in scenarios
    ]


def run(raw_dir: Path = RAW_DIR, model_check_dir: Path = MODEL_CHECK_DIR) -> dict[str, list[ScenarioDifficulty]]:
    env = RetailEnv()
    severity_by_id = _model_check_severity_by_id(model_check_dir)
    results: dict[str, list[ScenarioDifficulty]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        cell = path.stem
        parts = cell.split("__", 1)
        if len(parts) != 2 or parts[0] not in CATEGORIES or parts[1] not in THEMES:
            continue
        results[cell] = compute_cell_difficulty(path, env, severity_by_id)
    return results


def write_results(results: dict[str, list[ScenarioDifficulty]], out_dir: Path = DIFFICULTY_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell, scored in results.items():
        (out_dir / f"{cell}.json").write_text(
            json.dumps([s.to_dict() for s in scored], indent=2) + "\n"
        )


def print_summary(results: dict[str, list[ScenarioDifficulty]]) -> None:
    from collections import Counter

    all_scored = [s for scored in results.values() for s in scored]
    label_counts = Counter(s.difficulty_label for s in all_scored)
    print(f"TOTAL: {len(all_scored)} scenarios scored across {len(results)} cells")
    print(f"  easy:   {label_counts['easy']}")
    print(f"  medium: {label_counts['medium']}")
    print(f"  hard:   {label_counts['hard']}")

    print()
    print("By category:")
    by_category: dict[str, list[float]] = {}
    for cell, scored in results.items():
        category = cell.split("__", 1)[0]
        by_category.setdefault(category, []).extend(s.difficulty_score for s in scored)
    for category, scores in sorted(by_category.items()):
        mean = sum(scores) / len(scores)
        print(f"  {category:26s} n={len(scores):3d}  mean={mean:.3f}  "
              f"min={min(scores):.3f}  max={max(scores):.3f}")


def main() -> int:
    results = run()
    if not results:
        print(f"No scenario files found under {RAW_DIR}")
        return 1
    write_results(results)
    print_summary(results)
    print()
    print(f"Wrote sidecar difficulty files to {DIFFICULTY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
