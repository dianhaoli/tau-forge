"""Tests for Phase 3, stage 2's model checker: prompt rendering and, more
importantly, the shape validation that catches a subagent judge producing
output that doesn't actually match the expected report schema (missing a
scenario, duplicate ids, an issues/severity mismatch, an unknown severity).
"""

from __future__ import annotations

import json

import pytest

from tau_forge.validate.model_checker import (
    CellModelCheckReport,
    ScenarioFinding,
    load_reports,
    render_model_check_prompt,
    validate_report_shape,
)


def test_render_model_check_prompt_single_cell():
    prompt = render_model_check_prompt([("happy_path", "electronics_returns_exchanges")])
    assert "happy_path__electronics_returns_exchanges" in prompt
    assert "def cancel_pending_order" in prompt or "cancel_pending_order" in prompt
    assert "narrative" in prompt.lower()
    assert "distractor" in prompt.lower()
    assert "ambiguity_note" in prompt
    assert "Do not" in prompt and "fix" in prompt.lower()


def test_render_model_check_prompt_batches_multiple_cells():
    cells = [
        ("ambiguous", "apparel_footwear_exchanges"),
        ("ambiguous", "order_state_confusion"),
    ]
    prompt = render_model_check_prompt(cells)
    assert "ambiguous__apparel_footwear_exchanges" in prompt
    assert "ambiguous__order_state_confusion" in prompt


def test_render_model_check_prompt_rejects_unknown_cell():
    with pytest.raises(ValueError):
        render_model_check_prompt([("not_a_category", "electronics_returns_exchanges")])


def test_validate_report_shape_clean(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cell = "happy_path__electronics_returns_exchanges"
    scenarios = [{"id": f"{cell}__{i:03d}"} for i in range(1, 4)]
    (raw_dir / f"{cell}.json").write_text(json.dumps(scenarios))
    monkeypatch.setattr("tau_forge.validate.model_checker.RAW_DIR", raw_dir)

    report = CellModelCheckReport(
        cell=cell,
        findings=[ScenarioFinding(scenario_id=s["id"], issues=[], severity="none") for s in scenarios],
    )
    assert validate_report_shape(report) == []


def test_validate_report_shape_catches_missing_and_duplicate(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cell = "happy_path__electronics_returns_exchanges"
    scenarios = [{"id": f"{cell}__{i:03d}"} for i in range(1, 4)]
    (raw_dir / f"{cell}.json").write_text(json.dumps(scenarios))
    monkeypatch.setattr("tau_forge.validate.model_checker.RAW_DIR", raw_dir)

    report = CellModelCheckReport(
        cell=cell,
        findings=[
            ScenarioFinding(scenario_id=f"{cell}__001", issues=[], severity="none"),
            ScenarioFinding(scenario_id=f"{cell}__001", issues=[], severity="none"),  # duplicate
            # 002 missing entirely
            ScenarioFinding(scenario_id=f"{cell}__999", issues=["x"], severity="minor"),  # unknown id
        ],
    )
    problems = validate_report_shape(report)
    assert any("duplicate" in p for p in problems)
    assert any("unknown scenario_id" in p for p in problems)
    assert any("missing findings" in p for p in problems)


def test_validate_report_shape_catches_severity_issues_mismatch(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cell = "ambiguous__order_state_confusion"
    scenarios = [{"id": f"{cell}__001"}]
    (raw_dir / f"{cell}.json").write_text(json.dumps(scenarios))
    monkeypatch.setattr("tau_forge.validate.model_checker.RAW_DIR", raw_dir)

    # severity "major" but no issues listed -- should be flagged.
    report = CellModelCheckReport(
        cell=cell,
        findings=[ScenarioFinding(scenario_id=f"{cell}__001", issues=[], severity="major")],
    )
    problems = validate_report_shape(report)
    assert any("issues list is empty" in p for p in problems)


def test_validate_report_shape_catches_unknown_severity(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cell = "ambiguous__order_state_confusion"
    scenarios = [{"id": f"{cell}__001"}]
    (raw_dir / f"{cell}.json").write_text(json.dumps(scenarios))
    monkeypatch.setattr("tau_forge.validate.model_checker.RAW_DIR", raw_dir)

    report = CellModelCheckReport(
        cell=cell,
        findings=[ScenarioFinding(scenario_id=f"{cell}__001", issues=["x"], severity="catastrophic")],
    )
    problems = validate_report_shape(report)
    assert any("unknown severity" in p for p in problems)


def test_load_reports_reads_json_files(tmp_path):
    mc_dir = tmp_path / "model_check"
    mc_dir.mkdir()
    cell = "happy_path__electronics_returns_exchanges"
    data = [
        {"scenario_id": f"{cell}__001", "issues": [], "severity": "none"},
        {"scenario_id": f"{cell}__002", "issues": ["thin motivation"], "severity": "minor"},
    ]
    (mc_dir / f"{cell}.json").write_text(json.dumps(data))

    reports = load_reports(mc_dir)
    assert len(reports) == 1
    assert reports[0].cell == cell
    assert len(reports[0].findings) == 2
    assert len(reports[0].minor) == 1
    assert len(reports[0].flagged) == 1
