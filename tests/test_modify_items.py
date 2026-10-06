"""modify_items: change item variants on a pending order (modify_pending_order_items)."""

import json

import pytest

from tau_forge.decontam.real_tasks import RealTaskExclusions
from tau_forge.episodes.generate import generate_tasks, permutation_hashes
from tau_forge.episodes.reference_agents import ReferenceAgent, call
from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import run_episode
from tau_forge.episodes.task import EpisodeTask, ID_RE

EMPTY = RealTaskExclusions(frozenset(), frozenset())


@pytest.fixture(scope="module")
def tasks():
    return generate_tasks(40, 5, templates=("modify_items",), exclusions=EMPTY, log=lambda _: None).tasks


def test_generates_verified_pending_order_tasks(tasks):
    from tau_forge.episodes.task import base_db

    db = base_db()
    assert len(tasks) == 40
    for t in tasks:
        assert db.orders[t.target_order].status == "pending"
        assert t.gold_actions[-1]["name"] == "modify_pending_order_items"
        assert t.slots == [{"id": "main", "tool": "modify_pending_order_items", "record": f"order:{t.target_order}"}]
        assert not t.difficulty["late_correction"] or t.difficulty["n_items"] < 3
        for line in [t.opening] + [x for v in t.profile.values() if isinstance(v, list) for x in v if isinstance(x, str)]:
            assert not [i for i in ID_RE.findall(line) if not i.startswith("#W")]


def test_multi_item_tasks_accept_every_listing_order(tasks):
    multi = [t for t in tasks if t.difficulty["n_items"] >= 2]
    assert multi and all(t.alt_gold_db_hashes for t in multi)
    assert all(not t.alt_gold_db_hashes for t in tasks if t.difficulty["n_items"] == 1)

    class Reversed(ReferenceAgent):
        def write(self, name, arguments):
            if name == "modify_pending_order_items":
                arguments = {**arguments, "item_ids": arguments["item_ids"][::-1],
                             "new_item_ids": arguments["new_item_ids"][::-1]}
            return super().write(name, arguments)

    for t in multi:
        assert score_episode(t, run_episode(t, Reversed(t, "oracle"))).reward == 1.0, t.id


def test_exchange_tool_on_a_pending_order_fails(tasks):
    """The wrong tool for the status: exchange_delivered_order_items refuses a pending order."""

    class WrongTool(ReferenceAgent):
        def write(self, name, arguments):
            return super().write("exchange_delivered_order_items", arguments)

    for t in tasks[:10]:
        r = score_episode(t, run_episode(t, WrongTool(t, "oracle")))
        assert r.reward < 0.5, (t.id, r.reasons)


def test_round_trip_keeps_alt_hashes(tasks):
    t = next(t for t in tasks if t.alt_gold_db_hashes)
    d = json.loads(json.dumps(t.to_dict()))
    assert EpisodeTask.from_dict(d).alt_gold_db_hashes == t.alt_gold_db_hashes
    assert permutation_hashes(t)  # recomputable
