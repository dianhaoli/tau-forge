"""Composite tasks: several requests in one conversation (composite.py)."""

import json

import pytest

from tau_forge.decontam.real_tasks import RealTaskExclusions
from tau_forge.episodes.composite import CompositeAgent, CompositeUser, body_of, compose_tasks
from tau_forge.episodes.generate import generate_tasks, verify_task
from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import Episode, run_episode
from tau_forge.episodes.task import EpisodeTask, STOP, base_db_hash

EMPTY = RealTaskExclusions(frozenset(), frozenset())


@pytest.fixture(scope="module")
def comps():
    pool = generate_tasks(120, 9, templates=("cancel", "exchange", "return_fallback", "modify_items", "status_refusal",
                                               "foreign_order_refusal"), exclusions=EMPTY, log=lambda _: None).tasks
    return compose_tasks(pool, 30, 9, exclusions=EMPTY)


def test_composites_are_verified_multi_order_tasks(comps):
    assert len(comps) == 30
    for t in comps:
        subs = [EpisodeTask.from_dict(d) for d in t.subs]
        assert 2 <= len(subs) <= 3 and len({s.user_id for s in subs}) == 1
        assert len({s.target_order for s in subs}) == len(subs)
        assert verify_task(t).ok
        assert t.slots == [s for i, sub in enumerate(subs) for s in [{**x, "id": f"s{i}"} for x in sub.slots]]
        assert t.max_turns == 14 + 8 * len(subs)
        assert all(body_of(s.opening) for s in subs)


def test_oracle_completes_every_request(comps):
    for t in comps:
        r = score_episode(t, run_episode(t, CompositeAgent(t, "oracle")))
        assert r.reward == 1.0 and r.success, (t.id, t.difficulty["combo"], r.reasons)


def test_requests_are_revealed_one_at_a_time(comps):
    t = comps[0]
    ep = run_episode(t, CompositeAgent(t, "oracle"))
    users = [m["content"] for m in ep.messages if m["role"] == "user"]
    reveals = t.hidden["reveals"]
    first_seen = [next(i for i, u in enumerate(users) if r in u) for r in reveals[1:]]
    assert first_seen == sorted(first_seen) and first_seen[0] > 0
    assert users[-1].endswith(STOP)


def test_stopping_after_the_first_request_gets_partial_credit_only(comps):
    class FirstOnly(CompositeAgent):
        def __call__(self, messages):
            self.i = 0
            return self.agents[0](messages)

    seen = 0
    for t in comps:
        if EpisodeTask.from_dict(t.subs[0]).expect_no_write:
            continue
        r = score_episode(t, run_episode(t, FirstOnly(t, "oracle")))
        assert 0 < r.reward <= 0.3 and not r.success, (t.id, r.reward, r.reasons)
        seen += 1
    assert seen


def test_unconfirmed_writes_are_gated(comps):
    for t in comps:
        r = score_episode(t, run_episode(t, CompositeAgent(t, "no_confirm")))
        assert r.reward <= 0.7, (t.id, r.reward)


def test_consent_is_per_request(comps):
    """A yes for the first request's write never covers the second one."""
    t = next(t for t in comps if sum(1 for d in t.subs if not EpisodeTask.from_dict(d).expect_no_write) >= 2)
    ep = run_episode(t, CompositeAgent(t, "oracle"))
    slots = [w.slot for w in ep.log.writes if w.ok]
    assert len(slots) == len(set(slots)) >= 2 and all(w.confirmed for w in ep.log.writes if w.ok)


def test_round_trip_and_shared_db_untouched(comps):
    for t in comps[:5]:
        d = json.loads(json.dumps(t.to_dict()))
        t2 = EpisodeTask.from_dict(d)
        assert t2.subs == t.subs and t2.alt_gold_db_hashes == t.alt_gold_db_hashes
        run_episode(t2, CompositeAgent(t2, "oracle"))
    assert Episode(comps[0]).env.db_hash() == base_db_hash()
