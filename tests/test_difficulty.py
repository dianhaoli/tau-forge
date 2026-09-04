"""Tests for Phase 3, stage 4's difficulty calibration."""

from __future__ import annotations

import pytest

from tau_forge.envs.retail import RetailEnv
from tau_forge.validate.difficulty import (
    compute_difficulty,
    difficulty_label,
    tool_family,
)


@pytest.fixture(scope="module")
def env() -> RetailEnv:
    return RetailEnv()


def _base_scenario(**overrides):
    scenario = {
        "id": "happy_path__order_state_confusion__001",
        "category": "happy_path",
        "expected_tool_calls": [{"name": "cancel_pending_order", "arguments": {}}],
        "distractor_tool": "modify_pending_order_items",
    }
    scenario.update(overrides)
    return scenario


def test_tool_family_groups_order_mutation_tools_together():
    assert tool_family("cancel_pending_order") == tool_family("exchange_delivered_order_items")
    assert tool_family("cancel_pending_order") != tool_family("modify_user_address")
    assert tool_family("get_order_details") == tool_family("get_product_details")


def test_tool_family_unknown_for_bad_name():
    assert tool_family("not_a_real_tool") == "unknown"
    assert tool_family(None) == "none"


def test_score_in_unit_interval(env: RetailEnv):
    d = compute_difficulty(_base_scenario(), env)
    assert 0.0 <= d.difficulty_score <= 1.0
    assert d.difficulty_label in ("easy", "medium", "hard")


def test_happy_path_lookup_only_is_easier_than_happy_path_mutating(env: RetailEnv):
    mutating = compute_difficulty(
        _base_scenario(
            expected_tool_calls=[{"name": "cancel_pending_order", "arguments": {}}],
            distractor_tool="get_order_details",
        ),
        env,
    )
    lookup = compute_difficulty(
        _base_scenario(
            expected_tool_calls=[{"name": "get_order_details", "arguments": {}}],
            distractor_tool="calculate",
        ),
        env,
    )
    assert lookup.difficulty_score < mutating.difficulty_score


def test_ambiguous_empty_calls_harder_than_happy_path(env: RetailEnv):
    happy = compute_difficulty(
        _base_scenario(category="happy_path", expected_tool_calls=[
            {"name": "get_order_details", "arguments": {}}
        ], distractor_tool="calculate"),
        env,
    )
    ambiguous = compute_difficulty(
        _base_scenario(
            category="ambiguous",
            expected_tool_calls=[],
            distractor_tool="modify_pending_order_address",
        ),
        env,
    )
    assert ambiguous.difficulty_score > happy.difficulty_score


def test_same_family_distractor_harder_than_unrelated_distractor(env: RetailEnv):
    same_family = compute_difficulty(
        _base_scenario(
            expected_tool_calls=[{"name": "cancel_pending_order", "arguments": {}}],
            distractor_tool="modify_pending_order_items",  # also order_mutation family
        ),
        env,
    )
    unrelated = compute_difficulty(
        _base_scenario(
            expected_tool_calls=[{"name": "cancel_pending_order", "arguments": {}}],
            distractor_tool="calculate",  # generic family
        ),
        env,
    )
    assert same_family.difficulty_score > unrelated.difficulty_score
    assert same_family.components["distractor_kind"] == "same_family_as_answer"


def test_model_check_severity_increases_score(env: RetailEnv):
    none_flagged = compute_difficulty(_base_scenario(), env, model_check_severity="none")
    minor_flagged = compute_difficulty(_base_scenario(), env, model_check_severity="minor")
    major_flagged = compute_difficulty(_base_scenario(), env, model_check_severity="major")
    assert none_flagged.difficulty_score < minor_flagged.difficulty_score < major_flagged.difficulty_score


def test_out_of_scope_transfer_call_is_relatively_easy(env: RetailEnv):
    out_of_scope = compute_difficulty(
        _base_scenario(
            category="out_of_scope",
            expected_tool_calls=[{"name": "transfer_to_human_agents", "arguments": {"summary": "x"}}],
            distractor_tool="get_order_details",
        ),
        env,
    )
    policy_violation = compute_difficulty(
        _base_scenario(
            category="policy_violation",
            expected_tool_calls=[],
            distractor_tool="cancel_pending_order",
        ),
        env,
    )
    assert out_of_scope.difficulty_score < policy_violation.difficulty_score


def test_difficulty_label_thresholds():
    assert difficulty_label(0.0) == "easy"
    assert difficulty_label(0.39) == "easy"
    assert difficulty_label(0.40) == "medium"
    assert difficulty_label(0.64) == "medium"
    assert difficulty_label(0.65) == "hard"
    assert difficulty_label(1.0) == "hard"
