"""modify_address, modify_user_address and info templates."""

import pytest

from tau_forge.decontam.real_tasks import RealTaskExclusions
from tau_forge.episodes.generate import generate_tasks
from tau_forge.episodes.reference_agents import ReferenceAgent
from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import run_episode
from tau_forge.episodes.user import ScriptedUser, TurnContext

EMPTY = RealTaskExclusions(frozenset(), frozenset())


@pytest.fixture(scope="module")
def tasks():
    return generate_tasks(25, 4, templates=("modify_address", "modify_user_address", "info"), exclusions=EMPTY,
                          log=lambda _: None).tasks


def _ctx(t):
    return TurnContext(read_orders=[t.target_order] if t.target_order else [], authed_user=t.user_id)


def _addr(a):
    return f"{a['address1']}, {a['address2']}, {a['city']}, {a['state']} {a['zip']}"


def test_address_recap_needs_the_new_address(tasks):
    t = next(t for t in tasks if t.template == "modify_address" and t.difficulty["give_order_id"])
    a = t.hidden["new_address"]
    u = ScriptedUser(t)
    assert not u.reply(f"Shall I update the shipping address on order {t.target_order}?", _ctx(t)).confirms
    wrong = dict(a, zip="00000")
    r = ScriptedUser(t).reply(f"I'll change order {t.target_order}'s address to {_addr(wrong)}. Proceed?", _ctx(t))
    assert not r.confirms and r.revokes and a["zip"] in r.text
    r = ScriptedUser(t).reply(f"To confirm: the new shipping address for order {t.target_order} will be "
                              f"**{_addr(a)}**. Shall I proceed?", _ctx(t))
    assert r.confirms == ["main"]


def test_user_address_recap(tasks):
    t = next(t for t in tasks if t.template == "modify_user_address")
    a = t.hidden["new_address"]
    r = ScriptedUser(t).reply(f"I'll update your default address to {_addr(a)}, USA. Would you like me to proceed?", _ctx(t))
    assert r.confirms == ["main"]


@pytest.mark.parametrize("fmt", ["You'd get back ${v:.2f}.", "The refund would be ${v:,.2f} in total.",
                                 "That comes to {v:.2f} USD back to your original payment method."])
def test_info_correct_amount_ends_the_episode(tasks, fmt):
    t = next(t for t in tasks if t.template == "info" and t.hidden["fact"]["kind"] == "refund_total")
    r = ScriptedUser(t).reply(fmt.format(v=t.hidden["fact"]["value"]), _ctx(t))
    assert r.intent == "answered" and r.stop


def test_info_wrong_amount_and_offer_to_act(tasks):
    t = next(t for t in tasks if t.template == "info" and t.hidden["fact"]["kind"] == "refund_total")
    v = t.hidden["fact"]["value"]
    assert ScriptedUser(t).reply(f"You'd get back ${v + 3:.2f}.", _ctx(t)).intent == "answered_wrong"
    r = ScriptedUser(t).reply(f"Would you like me to go ahead and return those items from order {t.target_order}?", _ctx(t))
    assert r.intent == "not_now" and not r.confirms and not r.stop


def test_acting_on_a_question_fails(tasks):
    t = next(t for t in tasks if t.template == "info" and t.target_order)

    class Doer(ReferenceAgent):
        pass

    r = score_episode(t, run_episode(t, Doer(t, "no_confirm")))
    assert r.reward == 0.0, r.reasons


def test_all_oracles_score_one(tasks):
    for t in tasks:
        assert score_episode(t, run_episode(t, ReferenceAgent(t, "oracle"))).reward == 1.0, t.id
