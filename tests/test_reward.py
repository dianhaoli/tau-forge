"""Phase 4 adversarial tests for `tau_forge.reward.reward`.

Each test targets one of the failure modes the plan spec calls out explicitly:
a reward function that can't be gamed by padding, near-miss hallucination, or a
plausible-looking-but-wrong action, and that gives real partial credit for
outcomes that are actually close (equivalent valid args, a merely-descriptive
mismatch) while still separating those from a superficially-similar-but-wrong
action (a different record touched entirely).

All scenarios execute against the real shipped `db.json` via `RetailEnv`/
`execute_against` -- concrete order/user/item ids below were pulled from that
fixture (see `data/synthetic/` generation notes for the same discipline), not
invented.
"""

from __future__ import annotations

import pytest
from tau2.domains.retail.data_model import RetailDB

from tau_forge.envs.retail import RetailEnv
from tau_forge.reward.reward import Action, reward

# Fixture facts, pulled directly from the shipped db.json (see module docstring):
PENDING_ORDER = "#W5918442"  # user sofia_rossi_8776, credit_card_5051208, 4 items
PENDING_ORDER_OTHER = "#W2974929"  # a different, unrelated pending order
PENDING_USER_PAYMENT = "credit_card_5051208"
DELIVERED_ORDER = "#W4817420"  # user ava_moore_2033, gift_card_8168843 (balance 69.0)
DELIVERED_USER_PAYMENT = "gift_card_8168843"


@pytest.fixture(scope="module")
def base_db() -> RetailDB:
    return RetailEnv().snapshot()


def test_exact_match_mutating_tool_scores_full_credit(base_db: RetailDB) -> None:
    gold = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"})
    result = reward(gold, gold, base_db)
    assert result.score == 1.0
    assert result.reason == "state_match_exact"


def test_exact_match_read_tool_scores_full_credit(base_db: RetailDB) -> None:
    gold = Action("get_order_details", {"order_id": PENDING_ORDER})
    result = reward(gold, gold, base_db)
    assert result.score == 1.0


def test_correctly_withholding_a_call_scores_full_credit(base_db: RetailDB) -> None:
    expected = Action(tool_name=None)
    rollout = Action(tool_name=None)
    assert reward(rollout, expected, base_db).score == 1.0


def test_wrong_tool_scores_zero(base_db: RetailDB) -> None:
    expected = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"})
    rollout = Action("get_order_details", {"order_id": PENDING_ORDER})
    result = reward(rollout, expected, base_db)
    assert result.score == 0.0
    assert result.reason == "wrong_tool"


def test_calling_a_tool_when_none_was_expected_scores_zero(base_db: RetailDB) -> None:
    expected = Action(tool_name=None)  # e.g. an `ambiguous`-category scenario
    rollout = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"})
    result = reward(rollout, expected, base_db)
    assert result.score == 0.0
    assert result.reason == "unexpected_call"


# ---------------------------------------------------------------------------
# 1. Padded, generic free-text field -- must not score high just because the
#    tool and record are right.
# ---------------------------------------------------------------------------


def test_padded_generic_free_text_scores_low(base_db: RetailDB) -> None:
    gold = Action(
        "transfer_to_human_agents",
        {"summary": "Customer disputes a duplicate charge of $84.20 on order #W1234567; needs billing escalation."},
    )
    padded = Action(
        "transfer_to_human_agents",
        {
            "summary": (
                "I am unable to fully assist with this particular request at this time, so I will "
                "go ahead and connect you with a human agent representative who can help further."
            )
        },
    )
    result = reward(padded, gold, base_db)
    assert result.score < 0.5, f"padded/generic summary should score low, got {result.score}"
    # Sanity: an exact match on the same tool must still score 1.0, proving the
    # low score above is about content quality, not a broken comparison.
    assert reward(gold, gold, base_db).score == 1.0


# ---------------------------------------------------------------------------
# 2. Schema-valid but subtly wrong (wrong variant chosen) -- mid-range score,
#    not near 1.0.
# ---------------------------------------------------------------------------


def test_subtly_wrong_variant_scores_mid_range(base_db: RetailDB) -> None:
    gold = Action(
        "exchange_delivered_order_items",
        {
            "order_id": DELIVERED_ORDER,
            "item_ids": ["6700049080"],
            "new_item_ids": ["6117189161"],  # 4K, silver, waterproof -- available
            "payment_method_id": DELIVERED_USER_PAYMENT,
        },
    )
    subtly_wrong = Action(
        "exchange_delivered_order_items",
        {
            "order_id": DELIVERED_ORDER,
            "item_ids": ["6700049080"],
            "new_item_ids": ["9391733462"],  # a different available variant of the same product
            "payment_method_id": DELIVERED_USER_PAYMENT,
        },
    )
    result = reward(subtly_wrong, gold, base_db)
    assert 0.2 <= result.score <= 0.5, f"expected mid-range score, got {result.score} ({result.reason})"


# ---------------------------------------------------------------------------
# 3. Right tool/args plus an unrequested extra field -- hallucination penalty
#    must fire even though everything gold asked for is otherwise correct.
# ---------------------------------------------------------------------------


def test_hallucinated_extra_field_is_penalized(base_db: RetailDB) -> None:
    gold = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"})
    hallucinated = Action(
        "cancel_pending_order",
        {"order_id": PENDING_ORDER, "reason": "no longer needed", "priority": "high"},
    )
    result = reward(hallucinated, gold, base_db)
    assert result.score < 1.0, "an extra hallucinated argument must not score a perfect match"
    assert result.reason == "schema_invalid_or_hallucinated_args"
    assert result.score == 0.2


# ---------------------------------------------------------------------------
# 4. Different-but-valid arguments reaching the identical resulting DB state
#    -- must score 1.0 via the outcome-match path, not be penalized for
#    literal argument mismatch.
# ---------------------------------------------------------------------------


def test_equivalent_reordered_args_score_full_credit(base_db: RetailDB) -> None:
    # NB: `modify_pending_order_items` turned out to be the wrong tool for this
    # case -- tau2's own implementation has a latent bug where `item.price`/
    # `item.options` get set from the *last* variant processed in an earlier
    # loop, not each item's own new variant, so reordering the pairs actually
    # does change the resulting state there (a real upstream quirk, not ours).
    # `exchange_delivered_order_items` sorts `exchange_items`/`exchange_new_items`
    # independently and never touches `order.items` directly, so pair order is
    # genuinely irrelevant to its outcome -- verified empirically before writing
    # this assertion, not assumed.
    gold = Action(
        "exchange_delivered_order_items",
        {
            "order_id": DELIVERED_ORDER,
            "item_ids": ["6700049080", "9624127908"],
            "new_item_ids": ["6117189161", "4064702754"],
            "payment_method_id": DELIVERED_USER_PAYMENT,
        },
    )
    reordered = Action(
        "exchange_delivered_order_items",
        {
            "order_id": DELIVERED_ORDER,
            "item_ids": ["9624127908", "6700049080"],  # same two pairs, listed in the other order
            "new_item_ids": ["4064702754", "6117189161"],
            "payment_method_id": DELIVERED_USER_PAYMENT,
        },
    )
    result = reward(reordered, gold, base_db)
    assert result.score == 1.0, f"equivalent reordered args should reach the same DB state, got {result.detail}"
    assert result.reason == "state_match_exact"


def test_equivalent_calculate_expression_scores_full_credit(base_db: RetailDB) -> None:
    gold = Action("calculate", {"expression": "2 + 2"})
    equivalent = Action("calculate", {"expression": "4 - 0"})
    result = reward(equivalent, gold, base_db)
    assert result.score == 1.0, "different expressions with the same numeric result must score full credit"
    assert result.reason == "output_match"


# ---------------------------------------------------------------------------
# 5. Near miss: right record, one *descriptive* field off -- meaningfully high
#    partial credit, not close to 0.
# ---------------------------------------------------------------------------


def test_near_miss_descriptive_field_scores_high_partial_credit(base_db: RetailDB) -> None:
    gold = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"})
    wrong_reason = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "ordered by mistake"})
    result = reward(wrong_reason, gold, base_db)
    assert result.score >= 0.75, f"a same-order, reason-only miss should score high, got {result.score}"
    assert result.score < 1.0


# ---------------------------------------------------------------------------
# 6. Wrong-record miss: same tool, superficially similar arguments, but the
#    wrong order entirely -- must score much lower than the near miss above,
#    despite "looking close" (same tool, same argument shape).
# ---------------------------------------------------------------------------


def test_wrong_record_touched_scores_much_lower_than_near_miss(base_db: RetailDB) -> None:
    gold = Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"})
    wrong_order = Action("cancel_pending_order", {"order_id": PENDING_ORDER_OTHER, "reason": "no longer needed"})
    result = reward(wrong_order, gold, base_db)
    near_miss_score = reward(
        Action("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "ordered by mistake"}), gold, base_db
    ).score
    assert result.reason == "state_match_partial"
    assert result.detail.get("reason") == "wrong_record"
    assert result.score <= 0.35
    assert result.score < near_miss_score - 0.3, (
        f"wrong-record ({result.score}) should score much lower than the descriptive-field "
        f"near miss ({near_miss_score})"
    )
