"""Tests for Phase 6 data prep: converting the real τ2-bench retail train
tasks into `TrustedTask`s."""

from __future__ import annotations

import json

from tau_forge.data_prep.trusted_tasks import (
    TrustedTask,
    build_and_save,
    load_saved,
    load_trusted_train_tasks,
)


def test_loads_all_74_train_tasks():
    tasks = load_trusted_train_tasks()
    assert len(tasks) == 74
    assert len({t.task_id for t in tasks}) == 74  # unique ids


def test_every_task_has_a_nonempty_prompt():
    tasks = load_trusted_train_tasks()
    for t in tasks:
        assert t.prompt.strip()
        for a in t.gold_actions:
            assert a.tool_name
            assert isinstance(a.arguments, dict)


def test_some_tasks_legitimately_have_zero_gold_actions():
    """Confirmed against real train tasks (Phase 6 finding, e.g. "24"/"57"):
    the correct behavior for some tasks is purely conversational -- no tool
    call at all. The harness (`tau_forge.harness.rollout`) must treat an
    empty rollout against these as a perfect match, not a vacuous failure."""
    tasks = {t.task_id: t for t in load_trusted_train_tasks()}
    assert tasks["24"].gold_actions == []
    assert any(t.gold_actions for t in tasks.values())  # most tasks do have actions


def test_round_trips_through_json(tmp_path):
    out_path = tmp_path / "train_tasks.json"
    written = build_and_save(out_path)
    assert out_path.exists()

    loaded = load_saved(out_path)
    assert len(loaded) == len(written)
    assert [t.task_id for t in loaded] == [t.task_id for t in written]
    assert loaded[0].gold_actions[0].tool_name == written[0].gold_actions[0].tool_name

    raw = json.loads(out_path.read_text())
    assert isinstance(raw, list)
    assert set(raw[0].keys()) == {"task_id", "prompt", "gold_actions", "reward_basis", "communicate_info"}


def test_task_0_matches_known_reference_trajectory():
    """Pin down one task's shape against what was manually confirmed reading
    tau2's own `Task` objects, as a regression guard on the conversion."""
    tasks = {t.task_id: t for t in load_trusted_train_tasks()}
    t0 = tasks["0"]
    assert [a.tool_name for a in t0.gold_actions] == [
        "find_user_id_by_name_zip",
        "get_order_details",
        "get_product_details",
        "get_product_details",
        "exchange_delivered_order_items",
    ]
    assert "Yusuf Rossi" in t0.prompt
    assert t0.reward_basis == ["DB", "NL_ASSERTION"]
