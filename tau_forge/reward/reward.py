"""Phase 4 reward function.

Grades a rollout action against a scenario's gold (`expected`) action by comparing
*outcomes*, not literal arguments -- matching how tau2-bench itself grades (DB-end-
state equivalence, not trajectory match). Built on top of `tau_forge.envs.retail`;
does not reimplement any tool logic.

Score tiers (see module docstring in the Phase 4 plan for the full rationale):
    0.0        wrong tool / missing or unexpected call
    0.2        schema-invalid args, hallucinated (unknown) args, or the call
               raised at execution time despite valid schema
    0.3 - 1.0  schema-valid, right tool, right target record: graded on how close
               the outcome (resulting DB state, or output/args when there's no DB
               side effect) is to gold. 1.0 only for an exact outcome match.

`reward()` is stateless: it never mutates the `db_state` passed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

from tau2.domains.retail.data_model import RetailDB
from tau2.environment.toolkit import ToolType

from tau_forge.envs.retail import RetailEnv, execute_against

# ---------------------------------------------------------------------------
# Domain knowledge tables. These encode facts about the *retail* tool schemas
# and data model, not generic assumptions -- kept here, next to the reward
# logic that depends on them, rather than inferred structurally.
# ---------------------------------------------------------------------------

# Which argument identifies the record a WRITE tool acts on, for tools whose
# outcome is graded by diffing DB state. Every mutating retail tool except
# `modify_user_address` targets an order; that one targets a user.
TARGET_ID_ARG: dict[str, str] = {
    "cancel_pending_order": "order_id",
    "exchange_delivered_order_items": "order_id",
    "modify_pending_order_address": "order_id",
    "modify_pending_order_items": "order_id",
    "modify_pending_order_payment": "order_id",
    "return_delivered_order_items": "order_id",
    "modify_user_address": "user_id",
}

# Order fields where *which* value ended up there is the substance of the
# action -- a mismatch here means the outcome is materially wrong, not just
# imprecisely described. Near-exhaustive over Order's mutable fields.
CRITICAL_ORDER_FIELDS: set[str] = {
    "status",
    "items",
    "exchange_items",
    "exchange_new_items",
    "exchange_payment_method_id",
    "return_items",
    "return_payment_method_id",
    "payment_history",
    "address",
}

# Order fields that are correct-outcome-adjacent but not identity-defining --
# a mismatch here is a real miss but deserves partial credit, not zero.
# "numeric" fields get tolerance; others get an exact-vs-partial split.
GRADED_ORDER_FIELDS: dict[str, str] = {
    "cancel_reason": "categorical",
    "exchange_price_difference": "numeric",
}

# READ tools (plus `calculate`) return a value that is *itself* evidence the
# call was right or wrong -- two different-but-valid inputs producing the same
# output means the rollout is correct. `transfer_to_human_agents` is GENERIC
# but its return value ("Transfer successful") is constant regardless of
# `summary`, so output-equality would silently ignore summary quality. Do not
# add a tool here unless its return value actually varies with its arguments.
OUTPUT_DETERMINES_CORRECTNESS = {
    "calculate",
    "find_user_id_by_name_zip",
    "find_user_id_by_email",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "get_user_details",
    "list_all_product_types",
}

ID_LIKE_KEYS = {
    "order_id",
    "user_id",
    "item_id",
    "product_id",
    "payment_method_id",
    "new_item_id",
    "zip",
    "email",
}

NUMERIC_TOLERANCE = 0.05  # relative
TEXT_SIMILARITY_FLOOR = 0.3  # below this, a free-text field scores 0
SHORT_STRING_LEN = 40  # at/under this length, treat strings as exact-match categorical, not free text


@dataclass
class Action:
    """One action to grade: either a tool call, or `tool_name=None` for a
    message-only turn (used for `ambiguous`/`policy_violation` scenarios where
    the correct move is not calling a tool at all)."""

    tool_name: Optional[str]
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    score: float
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


def _numeric_tolerance(pv: Any, gv: Any) -> float:
    if pv is None or gv is None:
        return 1.0 if pv == gv else 0.0
    try:
        pv, gv = float(pv), float(gv)
    except (TypeError, ValueError):
        return 0.0
    if gv == 0:
        return 1.0 if abs(pv) < 1e-6 else 0.0
    rel_err = abs(pv - gv) / abs(gv)
    return max(0.0, 1.0 - rel_err / NUMERIC_TOLERANCE)


def _text_similarity_score(pv: str, gv: str) -> float:
    sim = SequenceMatcher(None, str(pv).strip().lower(), str(gv).strip().lower()).ratio()
    return max(0.0, (sim - TEXT_SIMILARITY_FLOOR) / (1 - TEXT_SIMILARITY_FLOOR))


def _field_score(key: str, pv: Any, gv: Any) -> float:
    if pv is None and gv is None:
        return 1.0
    if pv is None or gv is None:
        return 0.0
    if isinstance(gv, list):
        if list(pv) == list(gv):
            return 1.0
        try:
            return 0.5 if sorted(map(str, pv)) == sorted(map(str, gv)) else 0.0
        except TypeError:
            return 0.0
    if key in ID_LIKE_KEYS:
        return 1.0 if str(pv).lower() == str(gv).lower() else 0.0
    if isinstance(gv, bool):
        return 1.0 if pv == gv else 0.0
    if isinstance(gv, (int, float)):
        return _numeric_tolerance(pv, gv) if isinstance(pv, (int, float)) else 0.0
    if isinstance(gv, str):
        if len(gv) <= SHORT_STRING_LEN:
            return 1.0 if str(pv).strip().lower() == gv.strip().lower() else 0.0
        return _text_similarity_score(pv, gv)
    return 1.0 if pv == gv else 0.0


def arg_match_score(predicted_args: dict[str, Any], expected_args: dict[str, Any]) -> float:
    """Fallback outcome proxy for actions with no DB side effect to diff (or
    where output-equality didn't already resolve it): how well the rollout's
    arguments match gold's, field by field. Exact for IDs/enums/short strings,
    tolerant for numbers, similarity-with-a-floor for free text. Filling an
    extra optional field gold didn't specify costs a mild penalty (distinct
    from the harder hallucination cap for args outside the tool's schema
    entirely, which is handled upstream of this function)."""
    if not expected_args:
        return 1.0
    scores = [_field_score(k, predicted_args.get(k), v) for k, v in expected_args.items()]
    base = sum(scores) / len(scores)
    unexpected = [
        k for k in predicted_args if k not in expected_args and predicted_args.get(k) not in (None, "")
    ]
    return max(0.0, base - 0.15 * len(unexpected))


def state_match_score(
    tool_name: str,
    predicted_args: dict[str, Any],
    expected_args: dict[str, Any],
    predicted_db: RetailDB,
    gold_db: RetailDB,
) -> tuple[float, dict[str, Any]]:
    """How close a mutating rollout's resulting DB state is to gold's, given
    they already share the same tool and started from the same snapshot.
    Touching a different record than gold (wrong order/user id) scores 0 --
    superficially similar tool calls that acted on the wrong entity are a
    real failure, not a near miss. Touching the right record but landing on
    a different critical field (wrong items, wrong status, wrong payment
    method) also scores 0. Only a mismatch confined to graded fields (e.g.
    `cancel_reason`) earns partial credit."""
    id_key = TARGET_ID_ARG.get(tool_name)
    if id_key is None:
        return (1.0 if predicted_db == gold_db else 0.0), {"reason": "no_target_id_mapping"}

    predicted_target = predicted_args.get(id_key)
    gold_target = expected_args.get(id_key)
    if predicted_target != gold_target:
        return 0.0, {
            "reason": "wrong_record",
            "predicted_target": predicted_target,
            "gold_target": gold_target,
        }

    collection = predicted_db.orders if id_key == "order_id" else predicted_db.users
    gold_collection = gold_db.orders if id_key == "order_id" else gold_db.users
    predicted_record = collection.get(predicted_target)
    gold_record = gold_collection.get(gold_target)
    if predicted_record is None or gold_record is None:
        return 0.0, {"reason": "record_missing"}

    pred_dump = predicted_record.model_dump(mode="json")
    gold_dump = gold_record.model_dump(mode="json")

    critical_fields = CRITICAL_ORDER_FIELDS if id_key == "order_id" else {"address"}
    for f in critical_fields:
        if pred_dump.get(f) != gold_dump.get(f):
            return 0.0, {"reason": "critical_field_mismatch", "field": f}

    graded_fields = GRADED_ORDER_FIELDS if id_key == "order_id" else {}
    if not graded_fields:
        return 1.0, {"reason": "all_critical_fields_match"}

    detail: dict[str, Any] = {}
    scores = []
    for f, kind in graded_fields.items():
        pv, gv = pred_dump.get(f), gold_dump.get(f)
        s = _numeric_tolerance(pv, gv) if kind == "numeric" else (1.0 if pv == gv else 0.5)
        scores.append(s)
        detail[f] = s
    return 0.7 + 0.3 * (sum(scores) / len(scores)), detail


def reward(rollout: Action, expected: Action, db_state: RetailDB) -> RewardBreakdown:
    """Grade `rollout` against `expected` starting from `db_state`. Never
    mutates `db_state`."""
    if expected.tool_name is None:
        if rollout.tool_name is None:
            return RewardBreakdown(1.0, "correct_no_call")
        return RewardBreakdown(0.0, "unexpected_call", {"rollout_tool": rollout.tool_name})

    if rollout.tool_name is None:
        return RewardBreakdown(0.0, "missing_call", {"expected_tool": expected.tool_name})

    if rollout.tool_name != expected.tool_name:
        return RewardBreakdown(
            0.0, "wrong_tool", {"rollout_tool": rollout.tool_name, "expected_tool": expected.tool_name}
        )

    probe = RetailEnv(db=db_state.model_copy(deep=True))
    if not probe.has_tool(rollout.tool_name):
        return RewardBreakdown(0.0, "unknown_tool", {"tool": rollout.tool_name})

    schema_valid, schema_err = probe.validate_arguments(rollout.tool_name, rollout.tool_input)
    extra_args = probe.extra_arguments(rollout.tool_name, rollout.tool_input)
    if not schema_valid or extra_args:
        return RewardBreakdown(
            0.2,
            "schema_invalid_or_hallucinated_args",
            {"schema_error": schema_err, "extra_args": extra_args},
        )

    predicted_result, predicted_db = execute_against(db_state, rollout.tool_name, rollout.tool_input)
    gold_result, gold_db = execute_against(db_state, expected.tool_name, expected.tool_input)

    if not gold_result.ok:
        raise ValueError(
            f"Gold action failed to execute against db_state -- invalid gold label for "
            f"{expected.tool_name}({expected.tool_input}): {gold_result.error}"
        )

    if not predicted_result.ok:
        return RewardBreakdown(0.2, "execution_failed", {"error": predicted_result.error})

    mutates = probe.tool_mutates_state(rollout.tool_name)
    output_determines_correctness = rollout.tool_name in OUTPUT_DETERMINES_CORRECTNESS

    if not mutates:
        if output_determines_correctness and predicted_result.value == gold_result.value:
            return RewardBreakdown(1.0, "output_match")
        a = arg_match_score(rollout.tool_input, expected.tool_input)
        return RewardBreakdown(0.3 + 0.7 * a, "arg_match", {"arg_match_score": a})

    if predicted_db == gold_db:
        return RewardBreakdown(1.0, "state_match_exact")

    s, detail = state_match_score(
        rollout.tool_name, rollout.tool_input, expected.tool_input, predicted_db, gold_db
    )
    return RewardBreakdown(0.3 + 0.7 * s, "state_match_partial", detail)
