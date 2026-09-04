"""Tests for Phase 3, stage 3's sampling, rendering, and verdict capture."""

from __future__ import annotations

import json
import random

import pytest

from tau_forge.validate.human_review import (
    Verdict,
    load_results,
    record_verdict,
    render_markdown,
    sample_cell,
    save_results,
    save_stub_results,
    summarize_results,
)


def _scenarios(cell: str, n: int) -> list[dict]:
    return [
        {
            "id": f"{cell}__{i:03d}",
            "category": cell.split("__")[0],
            "prior_turns": [{"role": "user", "content": f"turn {i}"}],
            "user_message": f"message {i}",
            "expected_tool_calls": [{"name": "get_order_details", "arguments": {"order_id": "#W1"}}],
            "distractor_tool": "calculate",
            "distractor_rationale": "not relevant here",
            "ambiguity_note": "",
        }
        for i in range(1, n + 1)
    ]


def test_sample_cell_is_deterministic_given_same_seed():
    scenarios = _scenarios("happy_path__order_state_confusion", 20)
    a = sample_cell("happy_path__order_state_confusion", scenarios, set(), random.Random(42), base_n=4)
    b = sample_cell("happy_path__order_state_confusion", scenarios, set(), random.Random(42), base_n=4)
    assert [s["id"] for s in a] == [s["id"] for s in b]


def test_sample_cell_different_seed_can_differ():
    scenarios = _scenarios("happy_path__order_state_confusion", 20)
    a = sample_cell("happy_path__order_state_confusion", scenarios, set(), random.Random(1), base_n=4)
    b = sample_cell("happy_path__order_state_confusion", scenarios, set(), random.Random(2), base_n=4)
    assert [s["id"] for s in a] != [s["id"] for s in b]


def test_sample_cell_respects_base_n():
    scenarios = _scenarios("happy_path__order_state_confusion", 20)
    picked = sample_cell("happy_path__order_state_confusion", scenarios, set(), random.Random(42), base_n=4)
    assert len(picked) == 4


def test_sample_cell_oversamples_flagged():
    cell = "ambiguous__order_state_confusion"
    scenarios = _scenarios(cell, 20)
    flagged_ids = {f"{cell}__015", f"{cell}__016"}
    picked = sample_cell(cell, scenarios, flagged_ids, random.Random(42), base_n=4, max_flagged_extra=3)
    picked_ids = {s["id"] for s in picked}
    assert flagged_ids <= picked_ids  # both flagged scenarios must appear
    assert len(picked) <= 4 + 2  # base + however many flagged were actually new


def test_sample_cell_caps_flagged_extras():
    cell = "policy_violation__order_state_confusion"
    scenarios = _scenarios(cell, 20)
    flagged_ids = {f"{cell}__{i:03d}" for i in range(10, 20)}  # 10 flagged, well over the cap
    picked = sample_cell(cell, scenarios, flagged_ids, random.Random(42), base_n=4, max_flagged_extra=3)
    assert len(picked) <= 4 + 3


def test_sample_cell_handles_empty_cell():
    assert sample_cell("empty__cell", [], set(), random.Random(42)) == []


def test_render_markdown_includes_scenario_fields():
    cell = "happy_path__order_state_confusion"
    sample = {cell: _scenarios(cell, 2)}
    md = render_markdown(sample)
    assert f"{cell}__001" in md
    assert "turn 1" in md
    assert "message 1" in md
    assert "get_order_details" in md
    assert "calculate" in md


def test_render_markdown_marks_flagged_scenarios():
    cell = "ambiguous__order_state_confusion"
    sample = {cell: _scenarios(cell, 2)}
    md = render_markdown(sample, flagged_by_cell={cell: {f"{cell}__001"}})
    lines = md.splitlines()
    flagged_line = next(l for l in lines if f"{cell}__001" in l and l.startswith("###"))
    unflagged_line = next(l for l in lines if f"{cell}__002" in l and l.startswith("###"))
    assert "flagged" in flagged_line
    assert "flagged" not in unflagged_line


def test_save_and_load_stub_results_roundtrip(tmp_path):
    cell = "happy_path__order_state_confusion"
    sample = {cell: _scenarios(cell, 3)}
    out_path = tmp_path / "sample_results.json"
    verdicts = save_stub_results(sample, out_path)
    assert len(verdicts) == 3
    assert all(v.verdict == "pending" for v in verdicts)

    loaded = load_results(out_path)
    assert len(loaded) == 3
    assert {v.scenario_id for v in loaded} == {s["id"] for s in sample[cell]}


def test_save_stub_results_preserves_existing_by_default(tmp_path):
    cell = "happy_path__order_state_confusion"
    sample = {cell: _scenarios(cell, 3)}
    out_path = tmp_path / "sample_results.json"
    save_stub_results(sample, out_path)
    record_verdict(out_path, f"{cell}__001", "confirmed_fine", note="looks good", reviewer="dianhaoli@gmail.com")

    # Re-running the stub save (simulating a re-sample) shouldn't clobber the recorded verdict.
    save_stub_results(sample, out_path)
    loaded = {v.scenario_id: v for v in load_results(out_path)}
    assert loaded[f"{cell}__001"].verdict == "confirmed_fine"
    assert loaded[f"{cell}__001"].note == "looks good"


def test_record_verdict_updates_in_place(tmp_path):
    cell = "happy_path__order_state_confusion"
    sample = {cell: _scenarios(cell, 2)}
    out_path = tmp_path / "sample_results.json"
    save_stub_results(sample, out_path)

    record_verdict(out_path, f"{cell}__001", "flagged", note="distractor is weak")
    loaded = {v.scenario_id: v for v in load_results(out_path)}
    assert loaded[f"{cell}__001"].verdict == "flagged"
    assert loaded[f"{cell}__001"].note == "distractor is weak"
    assert loaded[f"{cell}__002"].verdict == "pending"


def test_record_verdict_rejects_unknown_scenario(tmp_path):
    out_path = tmp_path / "sample_results.json"
    save_results([Verdict(scenario_id="x__001", cell="x", verdict="pending")], out_path)
    with pytest.raises(KeyError):
        record_verdict(out_path, "not_sampled__001", "confirmed_fine")


def test_record_verdict_rejects_bad_verdict_value(tmp_path):
    out_path = tmp_path / "sample_results.json"
    save_results([Verdict(scenario_id="x__001", cell="x", verdict="pending")], out_path)
    with pytest.raises(ValueError):
        record_verdict(out_path, "x__001", "not_a_real_verdict")


def test_summarize_results_counts_by_verdict():
    verdicts = [
        Verdict(scenario_id="a", cell="c", verdict="confirmed_fine"),
        Verdict(scenario_id="b", cell="c", verdict="confirmed_fine"),
        Verdict(scenario_id="c", cell="c", verdict="flagged"),
        Verdict(scenario_id="d", cell="c", verdict="pending"),
    ]
    counts = summarize_results(verdicts)
    assert counts == {"confirmed_fine": 2, "flagged": 1, "pending": 1}
