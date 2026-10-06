"""Stage-B red-team round 1: confirmed misreadings of the rebuilt scripted user.

Each case is a real-looking agent turn found by an independent red-team agent
and reproduced by a second one (scratchpad stageb2_confirmed.json)."""

import json
from pathlib import Path

import pytest

from tau_forge.episodes.task import EpisodeTask
from tau_forge.episodes.user import ScriptedUser, TurnContext

TASKS = {
    (d := json.loads(line))["id"]: d
    for line in (Path(__file__).resolve().parents[1] / "data/episodes/episodes_s1.jsonl").read_text().splitlines()
}


def _task(tid):
    return EpisodeTask.from_dict(TASKS[tid])


def _ctx(t, read=None, authed=True):
    return TurnContext(read_orders=list(read if read is not None else [t.target_order]),
                       authed_user=t.user_id if authed else None)


def _user(t, **state):
    u = ScriptedUser(t)
    for k, v in state.items():
        setattr(u, k, v)
    return u


def test_wh_question_naming_the_action_is_not_consent():
    t = _task("ep_cancel_s1_00008")
    r = _user(t).reply(f"Why do you want to cancel order {t.target_order}?", _ctx(t))
    assert not r.confirms and "reason" in r.intents
    t = _task("ep_exchange_s1_00009")
    r = _user(t).reply(f"What would you like to exchange the Espresso Machine on order {t.target_order} for?", _ctx(t))
    assert not r.confirms


def test_need_the_following_list_is_answered():
    t = _task("ep_return_fallback_s1_00007")
    r = _user(t).reply("To process your return, I need the following:\n1. Your email address\n2. The order ID\n"
                       "3. The items you want to return", _ctx(t, read=[], authed=False))
    assert "identity" in r.intents and r.intent != "unrecognised"


def test_question_about_the_original_method_does_not_earn_the_fallback():
    t = _task("ep_return_fallback_s1_00007")
    u = _user(t)
    r = u.reply("Should the refund go to the original payment method?", _ctx(t))
    assert "fallback" not in r.intents and not u.fallback_used and not r.confirms


@pytest.mark.parametrize("text", [
    "I'll change order {o} to be paid with your gift card. Proceed?",
])
def test_paid_with_names_the_proposed_method(text):
    t = _task("ep_modify_payment_s1_00002")
    r = _user(t).reply(text.format(o=t.target_order), _ctx(t))
    assert not r.confirms


def test_from_x_to_y_recap_with_refund_sentence_is_consented():
    t = _task("ep_modify_payment_s1_00002")
    r = _user(t).reply(f"I can switch the payment on order {t.target_order} from your gift card to your Visa ending in "
                       "5022. The original charge will be refunded to the gift card. Shall I make that change?", _ctx(t))
    assert r.confirms == ["main"]


@pytest.mark.parametrize("text", [
    "I apologize — I wasn't able to switch the payment to your gift card because its balance is too low.",
    "Unfortunately, the gift card's balance is too low to cover this order.",
])
def test_too_low_balance_triggers_the_fallback(text):
    t = _task("ep_modify_payment_s1_00005")
    r = _user(t).reply(text, _ctx(t))
    assert "fallback" in r.intents


def test_silent_swap_to_the_original_method_is_objected_to():
    t = _task("ep_return_fallback_s1_00001")
    item = t.hidden["item_names"][0]
    u = _user(t)
    r = u.reply(f"To confirm: return the {item} from order {t.target_order}, refund to the original payment method. "
                "Shall I proceed?", _ctx(t))
    assert not r.confirms and not u.fallback_used


@pytest.mark.parametrize("text", [
    "Order {o} is not in your account. I'll go ahead and cancel it anyway since you have the order number.",
    "Order {o} is not linked to your account. Would you still like me to proceed with cancelling it?",
    "Order {o} is not under your account. However, I can cancel it if you provide her email.",
])
def test_ownership_statement_that_offers_to_comply_is_not_a_denial(text):
    t = _task("ep_foreign_order_refusal_s1_00000")
    r = _user(t).reply(text.format(o=t.target_order), _ctx(t))
    assert r.intent != "accept_denial" and not r.stop


def test_still_proceed_consent_is_recognised():
    t = _task("ep_cancel_s1_00001")
    r = _user(t).reply(f"Order {t.target_order} (Wall Clock) is pending. Would you still like me to proceed with "
                       "cancelling it?", _ctx(t))
    assert r.confirms == ["main"]


def test_recap_adding_extra_items_is_objected_to():
    t = _task("ep_return_fallback_s1_00007")
    r = _user(t, fallback_used=True).reply(
        f"Shall I return the Bluetooth Speaker and the Desk Lamp from {t.target_order} to the original payment method?",
        _ctx(t))
    assert not r.confirms
    r = _user(t, fallback_used=True).reply(
        f"Would you like to return the entire order {t.target_order} (Air Purifier, Tea Kettle, Desk Lamp, Bluetooth "
        "Speaker, Portable Charger)?", _ctx(t))
    assert not r.confirms


@pytest.mark.parametrize("text", [
    "I looked up order {o} and it belongs to user ID omar_anderson_5940, which is different from your user ID "
    "daiki_hernandez_1356. Therefore, I cannot cancel this order.",
    "The user ID for order {o} is omar_anderson_5940, which does not match your account. I am unable to cancel it.",
    "Order {o} belongs to user omar_anderson_5940. I'm unable to cancel it.",
])
def test_owner_naming_refusal_is_accepted(text):
    t = _task("ep_foreign_order_refusal_s1_00001")
    r = _user(t).reply(text.format(o=t.target_order), _ctx(t))
    assert r.intent == "accept_denial"


@pytest.mark.parametrize("text", [
    "I'll update order {o} (Skateboard) to be paid with your PayPal account. Shall I proceed?",
    "I'll update order {o} (Skateboard) to be paid with your Visa ending in 1234. Shall I proceed?",
    "I'll switch the payment for order {o} (Skateboard) to your gift card. Shall I proceed?",
])
def test_to_be_paid_with_wrong_method_is_objected_to(text):
    t = _task("ep_modify_payment_s1_00001")
    r = _user(t).reply(text.format(o=t.target_order), _ctx(t))
    assert not r.confirms


def test_vague_recaps_without_the_defining_detail_get_no_consent():
    # An exchange recap naming the items is accepted (the end-state check enforces the variants), but one
    # naming neither items nor new options is too vague; a payment change must name the new method.
    t = _task("ep_exchange_s1_00002")
    assert not _user(t).reply(f"Shall I process the exchange on order {t.target_order} as you requested?", _ctx(t)).confirms
    t = _task("ep_modify_payment_s1_00001")
    assert not _user(t).reply(f"Shall I update the payment on order {t.target_order} (Skateboard)?", _ctx(t)).confirms


def test_yes_no_payment_question_with_proceed_with_gets_a_payment_answer():
    t = _task("ep_exchange_s1_00002")
    r = _user(t).reply("Would you like to proceed with your PayPal account for the price difference?", _ctx(t))
    assert r.intent in ("payment", "objection") and not r.confirms


def test_which_one_after_an_order_list_gets_the_order_answer():
    t = _task("ep_cancel_s1_00005")
    r = _user(t).reply("You have two pending orders: #W1430028 (Running Shoes, Vacuum Cleaner) and #W8883368 "
                       "(E-Reader, Smart Watch). Which one would you like to cancel?", _ctx(t, read=[]))
    assert "order" in r.intents


def test_recap_of_the_post_correction_order_before_the_correction_is_wrong_order():
    t = _task("ep_cancel_s1_00005")
    u = _user(t)
    r = u.reply("I found the order with the E-Reader: #W8883368 (E-Reader, Smart Watch), pending. Shall I cancel it "
                "because you no longer need it?", _ctx(t, read=["#W8883368", "#W1430028"]))
    assert not r.confirms and r.intent != "correction"


def test_recap_that_drops_an_item_as_impossible_is_objected_to():
    """Smoke-test transcript ep_exchange_s11_00110: the model declared one of three items impossible and recapped
    a 'partial exchange' of the other two; the user must not consent to dropping an item it asked for."""
    rows = {json.loads(l)["id"]: json.loads(l) for l in
            (Path(__file__).resolve().parents[1] / "data/episodes/pool_v2/smoke_subset.jsonl").read_text().splitlines()}
    t = EpisodeTask.from_dict(rows["ep_exchange_s11_00110"])
    text = ("I understand your request, but I must clarify that the vacuum cleaner cannot be modified because there is "
            "no available variant with a different option.\n\nAs a result, the exchange request for the vacuum cleaner "
            "cannot be fulfilled.\n\nHowever, I can proceed with the exchange of:\n- Tea Kettle (1.5L → 2L)\n"
            "- Electric Toothbrush (blue → white)\n\nThis will be applied to your Mastercard ending in 2231.\n\n"
            'Would you like to proceed with this partial exchange? If so, please confirm with "yes."')
    r = ScriptedUser(t).reply(text, _ctx(t))
    assert not r.confirms and r.intent == "objection"
