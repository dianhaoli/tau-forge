"""Tests for the Phase 5 decontamination check.

Uses small fixture data for the similarity/flagging logic itself (the real
114x541 comparison is what `python -m tau_forge.decontam.check` runs, not
what pytest exercises) -- but `check_split` is cheap and is tested directly
against the real pinned tau2-bench data, same as `test_trusted_tasks.py`
does for the train split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tau_forge.decontam.check import (
    build_tfidf_matrix,
    check_split,
    render_real_narrative,
    render_synthetic_narrative,
    run_decontam_check,
    tokenize,
    tool_shape_score,
)


# --- lightweight stubs mirroring the tau2 Task attribute shape check.py duck-types against ---


@dataclass
class _StubAction:
    requestor: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubEvalCriteria:
    actions: list[_StubAction]


@dataclass
class _StubInstructions:
    reason_for_call: str


@dataclass
class _StubUserScenario:
    instructions: _StubInstructions


@dataclass
class _StubTask:
    id: str
    reason_for_call: str
    actions: list[_StubAction] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.user_scenario = _StubUserScenario(_StubInstructions(self.reason_for_call))
        self.evaluation_criteria = _StubEvalCriteria(self.actions)


def _synthetic(id_: str, user_message: str, calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": id_,
        "prior_turns": [],
        "user_message": user_message,
        "expected_tool_calls": calls or [],
    }


# --- tokenize / tfidf ---


def test_tokenize_lowercases_and_adds_bigrams():
    toks = tokenize("Return the Hose")
    assert "return" in toks
    assert "hose" in toks
    assert "return_the" in toks


def test_identical_documents_have_similarity_one():
    docs = ["cancel my pending order please", "cancel my pending order please"]
    mat = build_tfidf_matrix([tokenize(d) for d in docs])
    sim = float(mat[0] @ mat[1])
    assert sim > 0.999


def test_disjoint_vocabulary_has_similarity_zero():
    docs = ["cancel my pending order", "the weather is sunny today"]
    mat = build_tfidf_matrix([tokenize(d) for d in docs])
    sim = float(mat[0] @ mat[1])
    assert sim == 0.0


# --- narrative rendering ---


def test_render_synthetic_narrative_joins_prior_turns_and_message():
    scenario = {
        "prior_turns": [{"role": "user", "content": "Hi it's Alex."}, {"role": "assistant", "content": "Hi Alex."}],
        "user_message": "I want to return my hose.",
    }
    text = render_synthetic_narrative(scenario)
    assert "Alex" in text
    assert "return my hose" in text


def test_render_real_narrative_pulls_reason_for_call_only():
    task = _StubTask(id="1", reason_for_call="You want to cancel your pending order.")
    assert render_real_narrative(task) == "You want to cancel your pending order."


# --- tool shape score ---


def test_tool_shape_score_none_when_synthetic_has_no_calls():
    task = _StubTask(id="1", reason_for_call="x", actions=[_StubAction("assistant", "cancel_pending_order")])
    assert tool_shape_score([], task) is None


def test_tool_shape_score_zero_for_disjoint_tools():
    task = _StubTask(id="1", reason_for_call="x", actions=[_StubAction("assistant", "cancel_pending_order")])
    score = tool_shape_score([{"name": "get_order_details", "arguments": {"order_id": "#W1"}}], task)
    assert score == 0.0


def test_tool_shape_score_full_for_matching_tool_and_args():
    task = _StubTask(
        id="1",
        reason_for_call="x",
        actions=[_StubAction("assistant", "cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})],
    )
    score = tool_shape_score(
        [{"name": "cancel_pending_order", "arguments": {"order_id": "#W9", "reason": "changed mind"}}], task
    )
    # same tool, same argument *keys* -- values differ (shared-inventory ids
    # are not part of the signal) and don't affect the score.
    assert score == 1.0


def test_tool_shape_score_ignores_non_assistant_actions():
    task = _StubTask(
        id="1",
        reason_for_call="x",
        actions=[_StubAction("user", "cancel_pending_order"), _StubAction("assistant", "get_order_details")],
    )
    score = tool_shape_score([{"name": "cancel_pending_order", "arguments": {}}], task)
    assert score == 0.0  # only get_order_details counts as a real gold tool; no overlap


# --- split sanity check (cheap, against real pinned data) ---


def test_check_split_matches_documented_facts():
    result = check_split()
    assert result.train_count == 74
    assert result.test_count == 40
    assert result.base_count == 114
    assert result.train_test_disjoint
    assert result.train_union_test_equals_base
    assert result.ok


# --- end-to-end flagging logic on small fixtures ---


def test_flags_a_near_duplicate_but_not_an_unrelated_pair():
    real_tasks = [
        _StubTask(
            id="real-1",
            reason_for_call=(
                "You want to cancel the pending order containing the blue backpack because you no longer need it."
            ),
        ),
        _StubTask(id="real-2", reason_for_call="You want to know the store's return policy for electronics."),
    ]
    synthetic = [
        _synthetic(
            "synth-near-dup",
            "Please cancel my pending order with the blue backpack, I no longer need it.",
        ),
        _synthetic("synth-unrelated", "What's the weather forecast for tomorrow in Seattle?"),
        _synthetic("synth-somewhat", "Can you tell me about your return policy?"),
    ]

    # 3-sigma outlier detection needs more samples than this fixture has to
    # be meaningful; use an explicit threshold to isolate the flagging logic
    # itself from threshold-derivation statistics (covered by the real-data
    # `check_split`/script-run numbers documented in README).
    report = run_decontam_check(real_tasks=real_tasks, synthetic_scenarios=synthetic, similarity_threshold=0.3)

    flagged_ids = {p.synthetic_id for p in report.flagged}
    assert "synth-near-dup" in flagged_ids
    assert "synth-unrelated" not in flagged_ids

    near_dup = next(p for p in report.flagged if p.synthetic_id == "synth-near-dup")
    assert near_dup.real_task_id == "real-1"
    assert near_dup.text_similarity > 0.4


def test_report_json_round_trips_expected_shape():
    real_tasks = [_StubTask(id="real-1", reason_for_call="You want to cancel your pending order right away.")]
    synthetic = [_synthetic("synth-1", "Please cancel my pending order right away.")]

    report = run_decontam_check(real_tasks=real_tasks, synthetic_scenarios=synthetic)
    d = report.to_json()

    assert d["n_real_tasks"] == 1
    assert d["n_synthetic_scenarios"] == 1
    assert "split_check" in d
    assert isinstance(d["flagged_pairs"], list)
    if d["flagged_pairs"]:
        pair = d["flagged_pairs"][0]
        assert set(pair.keys()) == {
            "synthetic_id",
            "real_task_id",
            "text_similarity",
            "tool_shape_score",
            "synthetic_excerpt",
            "real_excerpt",
        }
