"""Tests for the Phase 6 multi-turn scoring harness
(`tau_forge.harness.rollout`), the core of the smoke test's pass/fail logic."""

from __future__ import annotations

from tau_forge.data_prep.trusted_tasks import GoldAction, TrustedTask
from tau_forge.harness.rollout import gold_rollout, replay_prefix_state, score_rollout
from tau_forge.reward.reward import Action

REAL_USER_TASK = TrustedTask(
    task_id="t-real-user",
    prompt="irrelevant for these tests",
    gold_actions=[
        GoldAction("find_user_id_by_name_zip", {"first_name": "Sara", "last_name": "Martin", "zip": "60654"}),
    ],
    reward_basis=["DB"],
    communicate_info=[],
)

# A gold trajectory that deliberately probes a nonexistent product first --
# mirrors the real τ2-bench train task "2" pattern (Phase 6 finding).
EXPECTED_FAILURE_TASK = TrustedTask(
    task_id="t-expected-failure",
    prompt="irrelevant for these tests",
    gold_actions=[GoldAction("get_product_details", {"product_id": "does-not-exist"})],
    reward_basis=["DB"],
    communicate_info=[],
)


def _find_a_real_user():
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    db = RetailDB.load(RETAIL_DB_PATH)
    user = next(iter(db.users.values()))
    return user


def test_empty_gold_and_empty_rollout_scores_1_0_not_0_0():
    """Regression test for the Phase 6 finding on real tasks "24"/"57":
    a task whose correct behavior is zero tool calls, matched by a rollout
    that also makes zero tool calls, must score 1.0 (perfect match), not the
    0.0 an empty turn-score list previously averaged to."""
    task = TrustedTask(
        task_id="t-no-actions-needed", prompt="x", gold_actions=[], reward_basis=["DB"], communicate_info=[]
    )
    result = score_rollout(task, [])
    assert result.mean == 1.0
    assert result.turn_scores[0].reason == "correct_no_call"


def test_gold_rollout_scores_perfect_1_0_on_every_turn():
    user = _find_a_real_user()
    task = TrustedTask(
        task_id="t-gold",
        prompt="x",
        gold_actions=[GoldAction("get_user_details", {"user_id": user.user_id})],
        reward_basis=["DB"],
        communicate_info=[],
    )
    result = score_rollout(task, gold_rollout(task))
    assert result.mean == 1.0
    assert result.min == 1.0


def test_expected_failure_gold_action_scores_1_0_when_reproduced():
    """A gold action that itself raises (real train-task pattern) should
    still grade 1.0 when the rollout reproduces the same failure -- not
    crash the harness, which is what happened before this was fixed."""
    result = score_rollout(EXPECTED_FAILURE_TASK, gold_rollout(EXPECTED_FAILURE_TASK))
    assert result.mean == 1.0


def test_missing_call_scores_0():
    task = REAL_USER_TASK
    result = score_rollout(task, [Action(tool_name=None)])
    assert result.turn_scores[0].score == 0.0
    assert result.turn_scores[0].reason == "missing_call"


def test_extra_rollout_turn_beyond_gold_scores_0():
    user = _find_a_real_user()
    task = TrustedTask(
        task_id="t-extra",
        prompt="x",
        gold_actions=[GoldAction("get_user_details", {"user_id": user.user_id})],
        reward_basis=["DB"],
        communicate_info=[],
    )
    rollout = gold_rollout(task) + [Action("get_user_details", {"user_id": user.user_id})]
    result = score_rollout(task, rollout)
    assert len(result.turn_scores) == 2
    assert result.turn_scores[1].score == 0.0
    assert result.turn_scores[1].reason == "extra_unrequested_turn"


def test_replay_prefix_state_does_not_mutate_across_calls():
    state_at_0 = replay_prefix_state(REAL_USER_TASK.gold_actions, 0)
    state_at_1 = replay_prefix_state(REAL_USER_TASK.gold_actions, 1)
    # find_user_id_by_name_zip is read-only -- both prefixes are the initial db.
    assert state_at_0 == state_at_1
