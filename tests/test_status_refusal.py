"""status_refusal: the user's own order in a status the request does not allow."""

import pytest

from tau_forge.decontam.real_tasks import RealTaskExclusions
from tau_forge.episodes.generate import generate_tasks
from tau_forge.episodes.reference_agents import ReferenceAgent, call
from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import run_episode
from tau_forge.episodes.task import base_db
from tau_forge.episodes.user import ScriptedUser, TurnContext

EMPTY = RealTaskExclusions(frozenset(), frozenset())


@pytest.fixture(scope="module")
def tasks():
    return generate_tasks(40, 2, templates=("status_refusal",), exclusions=EMPTY, log=lambda _: None).tasks


def test_tasks_target_an_ineligible_own_order(tasks):
    db = base_db()
    for t in tasks:
        o = db.orders[t.target_order]
        assert o.user_id == t.user_id and t.expect_no_write and not t.slots
        allowed = "pending" if t.hidden["request"] == "cancel" else "delivered"
        assert o.status != allowed and o.status == t.hidden["status"]


def _ctx(t):
    return TurnContext(read_orders=[t.target_order], authed_user=t.user_id)


@pytest.mark.parametrize("text", [
    "Unfortunately, since order {o} has already been delivered, it cannot be cancelled.",
    "I'm sorry, but order {o} has already been delivered, so I'm unable to cancel it. Only pending orders can be cancelled.",
    "Order {o} has already shipped, which means it can no longer be cancelled.",
    "I apologize, but I can't cancel this order. It has already been delivered.",
    "Order {o} is not eligible for cancellation because it was already delivered.",
])
def test_cancel_status_refusals_are_accepted(tasks, text):
    t = next(t for t in tasks if t.hidden["request"] == "cancel" and t.hidden["status"] == "delivered"
             and not t.profile.get("pushback"))
    r = ScriptedUser(t).reply(text.format(o=t.target_order), _ctx(t))
    assert r.intent == "accept_denial" and r.stop


@pytest.mark.parametrize("text", [
    "Order {o} is still pending and hasn't been delivered yet, so it can't be returned.",
    "I'm sorry, but returns are only possible for delivered orders, and order {o} is still pending.",
])
def test_return_status_refusals_are_accepted(tasks, text):
    t = next(t for t in tasks if t.hidden["request"] == "return" and t.hidden["status"] == "pending"
             and not t.profile.get("pushback"))
    r = ScriptedUser(t).reply(text.format(o=t.target_order), _ctx(t))
    assert r.intent == "accept_denial" and r.stop


def test_pushback_needs_a_second_refusal(tasks):
    t = next(t for t in tasks if t.profile.get("pushback") and t.hidden["request"] == "cancel")
    u = ScriptedUser(t)
    text = f"I'm sorry, but order {t.target_order} has already been {t.hidden['status']}, so it can't be cancelled."
    r1 = u.reply(text, _ctx(t))
    assert r1.intent == "pushback" and not r1.stop
    r2 = u.reply("I understand, but I really can't -- only pending orders can be cancelled, and this one isn't.", _ctx(t))
    assert r2.intent == "accept_denial" and r2.stop


def test_attempting_the_write_before_refusing_costs_the_gate(tasks):
    """Comply first (the tool rejects it), then refuse: right outcome, wrong order of operations."""

    class TryThenRefuse(ReferenceAgent):
        def __call__(self, messages):
            if self.wrote:
                self.mode = "oracle"
                self.wrote = False
                self._tried = True
            if getattr(self, "_tried", False):
                self.mode = "oracle"
            return super().__call__(messages)

    t = next(t for t in tasks if not t.profile.get("pushback"))
    r = score_episode(t, run_episode(t, TryThenRefuse(t, "no_confirm")))
    assert r.reward == pytest.approx(0.7) and "write_attempted" in r.gates, r.reasons


def test_wrong_rule_recital_is_not_a_refusal(tasks):
    """C1: a rule about another action, or a status the order does not have, does not refuse this request."""
    t = next(t for t in tasks if t.hidden["request"] == "cancel" and t.hidden["status"] == "delivered"
             and not t.profile.get("pushback"))
    for text in ("Please note that only delivered orders can be returned.",
                 f"I'm sorry, order {t.target_order} is still pending, so it can't be cancelled."):
        assert ScriptedUser(t).reply(text, _ctx(t)).intent != "accept_denial", text


def test_refusal_with_an_alternative_offer_is_accepted(tasks):
    """C3/F1: offering a different or later action does not veto the refusal, and the user never consents to it."""
    t = next(t for t in tasks if t.hidden["request"] == "cancel" and t.hidden["status"] == "delivered"
             and not t.profile.get("pushback"))
    r = ScriptedUser(t).reply(f"Order {t.target_order} has already been delivered, so I can't cancel it. "
                              "Would you like me to start a return instead?", _ctx(t))
    assert r.intent == "accept_denial" and not r.confirms
    t = next(t for t in tasks if t.hidden["request"] == "return" and t.hidden["status"] == "pending"
             and not t.profile.get("pushback"))
    r = ScriptedUser(t).reply(f"Order {t.target_order} hasn't been delivered yet, so I can't return it. Once it's "
                              "delivered, I can help you return it.", _ctx(t))
    assert r.intent == "accept_denial"


def test_offering_the_refused_action_now_vetoes(tasks):
    t = next(t for t in tasks if t.hidden["request"] == "cancel" and t.hidden["status"] == "delivered"
             and not t.profile.get("pushback"))
    r = ScriptedUser(t).reply(f"Order {t.target_order} has been delivered, so normally it can't be cancelled, but "
                              "would you still like me to cancel it?", _ctx(t))
    assert r.intent != "accept_denial"
