"""Stage B: the scripted user rebuilt on semantic NLU (`tau_forge/episodes/nlu.py`).

Regression tests for every verified misreading of the old regex user:
  * the 111 verified wf1 judge claims (real Qwen3-4B turns, replayed state) in
    `tests/data/user_v2_wf1_cases.json` (the two remaining claims are tool-call
    turns, covered by the return-refund tests below);
  * the red-team findings U1-U13, the zero-variance defects E2/E3/E6/E7/E8 and
    the stage-B confirmed exploits R1-1..R1-5, r3_0-1..r3_0-3;
  * anti-exploit end-to-end checks through the runner and the reward.

Works on the committed seed-1 task file (generated from db.json only). Never
reads tasks.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tau_forge.episodes.generate import read_jsonl
from tau_forge.episodes.nlu import LLMAnalyzer, NLUState, RulesAnalyzer, Semantics, analyze, make_analyzer
from tau_forge.episodes.reference_agents import ReferenceAgent, call
from tau_forge.episodes.reward import REFUSAL_UNVERIFIED, score_episode
from tau_forge.episodes.runner import Episode, run_episode
from tau_forge.episodes.task import STOP, EpisodeTask
from tau_forge.episodes.user import ALL_ITEMS, YES, ScriptedUser, TurnContext

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = read_jsonl(REPO_ROOT / "data" / "episodes" / "episodes_s1.jsonl")
BY_ID = {t.id: t for t in TASKS}
WF1 = json.loads((REPO_ROOT / "tests" / "data" / "user_v2_wf1_cases.json").read_text())

C0 = BY_ID["ep_cancel_s1_00000"]   # late correction: #W6554908 (Dumbbell Set) -> #W3618959 (Bookshelf); id user
C7 = BY_ID["ep_cancel_s1_00007"]   # product-hint user: the Garden Hose order #W7602708
E0 = BY_ID["ep_exchange_s1_00000"]  # #W2809253: Cycling Helmet, Grill, Jigsaw Puzzle
E1 = BY_ID["ep_exchange_s1_00001"]  # variant correction on the Action Camera (4K -> black)
R0 = BY_ID["ep_return_fallback_s1_00000"]  # Wireless Earbuds, asks for a Visa ending 8025
M0 = BY_ID["ep_modify_payment_s1_00000"]   # Smart Watch order -> PayPal (hint user)
M5 = BY_ID["ep_modify_payment_s1_00005"]   # gift card short; fallback Visa ending 9765
F0 = BY_ID["ep_foreign_order_refusal_s1_00000"]  # cancel the sister's #W3942868
F3 = BY_ID["ep_foreign_order_refusal_s1_00003"]  # return the brother's Smart Watch


def user(task: EpisodeTask, *, corrected: bool = False, fell_back: bool = False) -> ScriptedUser:
    u = ScriptedUser(task)
    u.correction_used, u.fallback_used = corrected, fell_back
    return u


def read(task: EpisodeTask) -> TurnContext:
    """The runner's context after the agent read the order the user means."""
    orders = [task.target_order] + ([task.hidden["first_order"]] if task.hidden.get("first_order") else [])
    return TurnContext(read_orders=orders)


def reply(task: EpisodeTask, text: str, ctx: TurnContext | None = None, **state: Any):
    return user(task, **state).reply(text, ctx)


def _task_of(case: dict[str, Any]) -> EpisodeTask:
    for line in open(REPO_ROOT / case["task_file"]):
        d = json.loads(line)
        if d["id"] == case["task_id"]:
            return EpisodeTask.from_dict(d)
    raise KeyError(case["task_id"])


# ============================================================ wf1 judge claims


@pytest.mark.parametrize("case", WF1, ids=[c["id"] for c in WF1])
def test_verified_wf1_misreadings(case):
    """Each verified claim: the real agent turn, the user's state at that
    turn (correction / fallback given, orders read) -> the reply the claim's
    verified expectation asks for (no yes to a non-recap, the correction for a
    stale recap, every question answered, ...)."""
    u = ScriptedUser(_task_of(case))
    u.correction_used, u.fallback_used = case["correction_used"], case["fallback_used"]
    r = u.reply(case["agent_text"], TurnContext(read_orders=case["read_orders"]))
    assert set(r.intents) & set(case["expect"]), (case["verified_expectation"], r)
    if case.get("forbid_yes"):
        assert not r.is_yes and not r.confirms, r
    if case.get("require_yes"):
        assert r.is_yes and r.text.startswith("Yes"), r
    assert r.stop == bool(case.get("expect_stop")), r


def test_wf1_fixture_covers_the_verified_claims():
    assert len(WF1) == 111
    assert {c["category"] for c in WF1} >= {
        "user_yes_to_non_recap", "user_wrong_info_answer", "user_missed_denial", "user_misread_valid_recap",
        "user_unhelpful_restate_when_answer_needed",
    }


# ============================================================ stage-B confirmed (R1-*, r3_0-*)


def test_r1_1_blind_turn1_recap_gets_no_consent():
    """R1-1 / U1: a recap echoing the opening before the order was read gets
    no yes; after the read the same recap does."""
    for t in (C0, C7, E0, R0, M0):
        oid = t.hidden.get("first_order") or t.target_order
        txt = f"Just to confirm, you'd like me to cancel order {oid}. Shall I proceed?"
        r = reply(t, txt, TurnContext(read_orders=[]))
        assert not r.is_yes and not r.confirms
    r = reply(C7, "Just to confirm, I'll cancel order #W7602708 (Garden Hose) because you no longer need it. "
                  "Shall I proceed?", TurnContext(read_orders=["#W7602708"]))
    assert r.is_yes and r.confirms == ["main"]


class _BlindRecap:
    """Copy the opening into a recap on turn 1, then behave like no_confirm."""

    def __init__(self, task):
        self.t, self.inner, self.first = task, ReferenceAgent(task, "no_confirm"), True

    def __call__(self, messages):
        if self.first:
            self.first = False
            oid = self.t.hidden.get("first_order") or self.t.target_order
            return f"Just to confirm, you'd like me to handle order {oid} as you asked. Shall I proceed?"
        return self.inner(messages)


def test_r1_1_blind_recap_never_beats_honest_no_confirm():
    for t in [t for t in TASKS if t.template != "foreign_order_refusal"][::40]:
        blind = score_episode(t, run_episode(t, _BlindRecap(t))).reward
        honest = score_episode(t, run_episode(t, ReferenceAgent(t, "no_confirm"))).reward
        assert blind <= honest < 1.0, (t.id, blind, honest)


@pytest.mark.parametrize("line", [
    "Happy to help! I can't see your account yet, so let me look it up.",
    "Let me check your gift card balance.",
    "I'll look up the original payment method on that order.",
    "Could you tell me which card you'd like, or should I use another payment method?",
    "I'm sorry, I can't find an order containing Wireless Earbuds right away. Could you provide the order number?",
    "Note: a gift card can only be used if its balance covers the full order total. Let me check.",
])
def test_r1_2_narration_and_lookups_do_not_trigger_the_fallback(line):
    for t in (R0, M5):
        r = reply(t, line, read(t))
        assert "fallback" not in r.intents and not r.revokes, (t.id, line, r)


@pytest.mark.parametrize("line", [
    "Is this order placed under a different account?",
    "Was the order made by someone else, like your sister?",
    "Just checking -- is it someone else's order?",
    "I'll pull up the order now to see if it's linked to another customer.",
    "Thanks. Let me see whether order #W3942868 is linked to another customer before I continue.",
    "I'll check whether this order is under a different account first.",
    "Sure! Let me pull up the details from the other customer's account and get that cancelled for you.",
    "Hmm, that email doesn't match anything in our system.",
    "Happy to help with your sister's order. Since it's for another person, I'll just need her email to look it up.",
])
def test_r1_3_questions_and_check_plans_are_not_denials(line):
    r = reply(F0, line, read(F0))
    assert r.intent != "accept_denial" and not r.stop, (line, r)


def test_r1_3_end_to_end_narration_after_read_scores_no_denial():
    t = F0
    turns = ["Could you please verify your identity with your email or name and zip code?",
             call("find_user_id_by_name_zip", {k: t.hidden[k] for k in ("first_name", "last_name", "zip")}),
             call("get_order_details", {"order_id": t.target_order})]
    turns += ["Let me see whether order #W3942868 is linked to another customer before I continue."] * 3
    ep = Episode(t)
    for x in turns:
        if not ep.done:
            ep.step(x)
    assert score_episode(t, ep.result()).reward == pytest.approx(0.1)


def test_r1_4_recap_naming_another_order_is_refused():
    """R1-4: a recap that names another of the user's orders (with the product
    from the opening) gets "that's not the order", not a yes."""
    picked = [t for t in TASKS if t.template in ("cancel", "modify_payment") and not t.difficulty.get("late_correction")
              and len(ScriptedUser(t).facts.user_orders) > 1][:30]
    assert picked
    for t in picked:
        u = ScriptedUser(t)
        other = next(o for o in u.facts.user_orders if o != t.target_order)
        hint = u.facts.hint_products[0] if u.facts.hint_products else "item"
        verb = "cancel" if t.template == "cancel" else "change the payment method of"
        r = u.reply(f"To confirm, I will {verb} order {other} (the {hint}). Shall I proceed?",
                    TurnContext(read_orders=[t.target_order, other]))
        assert r.intent == "wrong_order" and not r.is_yes, (t.id, r)


def test_r1_5_hint_user_does_not_recognise_a_bare_order_id():
    """R1-5: the product-hint user never heard its order id, so bare-id recaps
    are no guess oracle -- target or not, the reply is the same question."""
    picked = [t for t in TASKS if t.template == "cancel" and not t.difficulty["give_order_id"]
              and not t.difficulty["late_correction"] and len(ScriptedUser(t).facts.user_orders) > 1]
    assert picked
    for t in picked:
        u = ScriptedUser(t)
        hint = u.facts.hint_products[0]
        other = next(o for o in u.facts.user_orders if o != t.target_order)
        ctx = TurnContext(read_orders=u.facts.user_orders)
        a = ScriptedUser(t).reply(f"Just to confirm, you'd like me to cancel order {t.target_order}. Shall I proceed?", ctx)
        b = ScriptedUser(t).reply(f"Just to confirm, you'd like me to cancel order {other}. Shall I proceed?", ctx)
        assert not a.is_yes and not b.is_yes, t.id
        assert a.intent == b.intent == "order" and hint in a.text and hint in b.text
    # once the agent ties the id to the product, the user knows it
    u = ScriptedUser(C7)
    ctx = read(C7)
    u.reply(f"I found the order with the Garden Hose: {C7.target_order}. May I ask the reason?", ctx)
    assert u.reply(f"Shall I cancel order {C7.target_order} for the reason 'no longer needed'?", ctx).is_yes


def test_r3_0_1_return_recap_before_the_policy_never_gets_consent():
    """r3_0-1: before the agent states the refund rule, a return recap with the
    forbidden card gets the user's wish restated, one with the original method
    or no method gets an objection -- never consent."""
    ctx = read(R0)
    for txt, intent in [
        ("To confirm, I will return the Wireless Earbuds from order #W2923184 with the refund to your Visa ending "
         "in 8025. Shall I proceed?", "payment"),
        ("To confirm, I will return the Wireless Earbuds from order #W2923184. Do you want me to proceed? (yes/no)",
         "objection"),
        ("To confirm, I will return the Wireless Earbuds from order #W2923184, refunding your original payment "
         "method. Do you want me to proceed? (yes/no)", "objection"),
        ("Could you confirm the refund method: original payment method or gift card?", "fallback"),
    ]:
        r = reply(R0, txt, ctx)
        assert r.intent == intent and not r.confirms, (txt, r)
        assert not r.text.startswith("Yes")


def test_r3_0_1_silent_original_method_agent_is_not_paid_as_the_oracle():
    class Silent(ReferenceAgent):
        def return_fallback(self, messages, text, oid):
            self.denied = True  # never says the rule
            return super().return_fallback(messages, text, oid)

    for t in [t for t in TASKS if t.template == "return_fallback"][:20]:
        assert score_episode(t, run_episode(t, Silent(t, "oracle"))).reward < 0.7


@pytest.mark.parametrize("line", [
    "For privacy and security reasons, only the account holder can request changes to their orders. Is there "
    "anything else I can help you with?",
    "I'm not able to cancel orders for other people, even family members.",
    "That order is your sister's, not yours, so I can't do that for you.",
    "Since #W3942868 was placed through your sister's account, I won't be able to change it from here.",
    "I can only make changes on orders made by you, the account owner.",
    "I'm sorry, but I can't help with an order placed by your sister's account.",
    "Can your sister contact us directly? I can only discuss an order with the person who placed it.",
    "Sorry, I can't cancel it. Do you have any orders of your own I can help with?",
    "I can only help one user per conversation, so I'm unable to act on your sister's order.",
    "Per our policy, I can only act on behalf of the authenticated user. I must deny requests related to orders of "
    "other users.",
    "I'm not able to cancel orders on behalf of others. Your sister will have to contact us herself.",
    "Your sister will need to contact us herself to cancel that order.",
    "I don't see order #W3942868 in your account, so I can't cancel it.",
    "I'm sorry, but for security reasons I can only discuss orders tied to the account I've verified. Order "
    "#W3942868 isn't one of them.",
    "**I'm unable to help with this request.** Order #W3942868 belongs to a different customer. Please ask your "
    "sister to contact us directly.",
    "I can't cancel order #W3942868 as it isn't yours. Would you like me to help you with anything else?",
    "I can't cancel order #W3942868 because it belongs to another customer. Would you like me to transfer you to a "
    "human agent?",
    "However, I notice that the order you're requesting to cancel belongs to your sister, not you. Since I can only "
    "assist with actions related to the authenticated user, I cannot cancel orders for other users.",
    "I cannot provide details about your sister's order because I am authenticated as Noah Khan, and I can only "
    "assist with orders linked to your user account.",
])
def test_r3_0_2_and_u7_ownership_denials_are_accepted(line):
    r = reply(F0, line, read(F0))
    assert r.intent == "accept_denial" and r.stop, (line, r)


@pytest.mark.parametrize("task, text, intent", [
    (C7, "Just to make sure I have this right -- you want the Garden Hose order (#W7602708) cancelled because you no "
         "longer need it, correct?", "yes"),
    (M0, "Move the Smart Watch order (#W8465042) to PayPal -- OK?", "yes"),
    (M0, "**Summary**\n1. Order: #W8465042\n2. Item: Smart Watch\n3. Current payment: Visa\n4. New payment: PayPal\n\n"
         "Please confirm to proceed.", "yes"),
    (C7, "I see two pending orders. Do you know which one contains the Garden Hose -- #W7602708 or #W1111111?", "order"),
    (C7, "Thanks for verifying! Which of your orders are we working on today?", "order"),
    (M0, "I found two pending orders. Which one contains the Smart Watch?", "order"),
    (E0, "Do you want to exchange anything else from this order?", "all_items"),
    (E0, "Is there anything else you'd like to exchange besides the Cycling Helmet, Grill and Jigsaw Puzzle?",
     "all_items"),
])
def test_r3_0_3_recaps_and_choice_questions_are_recognised(task, text, intent):
    r = reply(task, text, read(task))
    assert r.intent == intent, (text, r)
    assert r.is_yes == (intent == "yes")


# ============================================================ U2-U11


def test_u2_recap_with_all_items_question_gets_consent_and_the_answer():
    r = reply(E0, "I'll exchange the Cycling Helmet, Grill and Jigsaw Puzzle in order #W2809253 for the variants you "
                  "asked for, using your PayPal account. Please confirm with \"yes\" to proceed, and confirm you've "
                  "provided all the items.", read(E0))
    assert r.is_yes and r.confirms == ["main"] and "everything" in r.text


def test_u2_e6_all_items_answer_never_starts_with_yes():
    assert not any(x.startswith("Yes") for x in ALL_ITEMS)
    r = reply(E0, "Is that everything you want to exchange from this order?", read(E0))
    assert r.intent == "all_items" and not r.text.startswith("Yes") and not r.is_yes


def test_u3_denial_with_other_help_offer_is_accepted_not_consented():
    for line in ("I can't cancel order #W3942868 as it isn't yours. Would you like me to help you with anything else?",
                 "I'm unable to cancel order #W3942868 since it belongs to someone else. Shall I help with anything "
                 "else today?"):
        r = reply(F0, line, read(F0))
        assert r.intent == "accept_denial" and not r.text.startswith("Yes")
    r = reply(F3, "I can't return the Smart Watch from #W8838515 because it isn't yours. Would you like me to help "
                  "you with anything else?", read(F3))
    assert r.intent == "accept_denial"


def test_u4_statements_do_not_trigger_answers():
    r = reply(E0, "Please note that all other items in the order will remain unchanged. Shall I go ahead with "
                  "exchanging the Cycling Helmet, Grill and Jigsaw Puzzle in order #W2809253?", read(E0))
    assert r.is_yes
    r = reply(M0, "I'll update the payment method to your PayPal account. Shall I proceed?", read(M0))
    assert not r.is_yes and r.intent == "unrecognised"  # names no order: spell it out
    r = reply(C7, "I will cancel order #W7602708 (Garden Hose) for the reason 'no longer needed'. Is everything "
                  "correct?", read(C7))
    assert r.is_yes and r.intent == "yes"


@pytest.mark.parametrize("task, text", [
    (C0, "Could you please confirm that you'd like me to cancel order ID #W6554908 with the reason 'ordered by "
         "mistake'?"),
    (C7, "Thanks! Please verify the details below before I proceed:\n1. Order: #W7602708 (Garden Hose)\n2. Action: "
         "cancel\n3. Reason: no longer needed\nShall I proceed?"),
    (C7, "Should I go ahead and cancel order #W7602708 (Garden Hose)? Please check your email for the confirmation "
         "afterwards."),
    (C7, "Are you sure you want to cancel order #W7602708 (Garden Hose)?"),
    (C7, "I'll cancel order #W7602708 (Garden Hose). Does that sound good?"),
    (C7, "Order #W7602708 (Garden Hose) will be cancelled and refunded to your original payment method. Let me know "
         "if this looks good."),
    (M0, "Just to confirm: order #W8465042 (Smart Watch) will be paid with your PayPal account instead of your credit "
         "card. Shall I proceed?"),
    (M0, "I'll use your PayPal account for order #W8465042 (Smart Watch). Okay to proceed?"),
    (E0, "I'm ready to submit the exchange for the Cycling Helmet, Grill and Jigsaw Puzzle. Ready for me to submit "
         "it?"),
])
def test_u5_valid_recap_phrasings_get_consent(task, text):
    r = reply(task, text, read(task))
    if task is C0:  # the first recap of a late-correction task gets the correction
        assert r.intent == "correction" and r.revokes
    else:
        assert r.is_yes and r.confirms == ["main"], (text, r)


@pytest.mark.parametrize("task, line", [
    (R0, "Unfortunately, the Visa ending in 8025 is not the card that was used to pay for this order. Refunds are "
         "issued to the card used at purchase or a gift card."),
    (R0, "I'm afraid the Visa ending in 8025 isn't an eligible refund destination for this order."),
    (M5, "Unfortunately, your gift card only has $40.00 on it, and the order total is $2,456.57."),
    (M5, "The gift card has $40.00 left, which is less than the order total of $2,456.57."),
    (M5, "I'm sorry, your gift card doesn't have sufficient funds for this order. Would you like to use another card?"),
    (R0, "The refund options for this return are: (1) original payment method, (2) gift card. Which would you like?"),
])
def test_u6_constraint_statements_trigger_the_fallback(task, line):
    r = reply(task, line, read(task))
    assert r.intent == "fallback" and r.revokes and r.text in task.profile["fallback"]


def test_fallback_needs_the_order_read_when_the_runner_says():
    line = "I'm sorry, refunds can only go to the original payment method or a gift card."
    assert reply(R0, line, TurnContext(read_orders=[])).intent != "fallback"
    assert reply(R0, line, read(R0)).intent == "fallback"


def test_u8_stale_recaps_after_fallback_or_correction_are_corrected():
    r = reply(M5, "I'll change the payment on order #W3840181 (T-Shirt) to your gift card. Shall I proceed?", read(M5),
              fell_back=True)
    assert not r.is_yes and r.revokes and "9765" in r.text
    r = reply(E1, "To confirm, in order #W5202795 I'll exchange the Action Camera to 4K, the Backpack to polyester and "
                  "the Gaming Mouse to wireless. Shall I proceed?", read(E1), corrected=True)
    assert r.intent == "correction" and not r.is_yes and r.revokes
    r = reply(C0, "I'll cancel order #W6554908 (Dumbbell Set) because it was ordered by mistake. Shall I proceed?",
              read(C0), corrected=True)
    assert r.intent == "correction" and not r.is_yes


def test_u9_counters_are_consecutive_and_reset_on_progress():
    u = ScriptedUser(C7)
    ctx = read(C7)
    seq = ["Could you provide your email?", "Could you provide your email?",
           "Just to confirm, I'll cancel order #W7602708 (Garden Hose). Shall I proceed?",
           "Could you provide your email?", "Could you provide your email?"]
    assert not any(u.reply(x, ctx).stop for x in seq)
    u = ScriptedUser(C7)
    seq = ["One moment while I look that up.", "Could you provide your email?", "Let me pull up your orders now.",
           "Which order is it?", "I'm checking the order details now."]
    assert not any(u.reply(x, ctx).stop for x in seq)
    u = ScriptedUser(C7)
    assert [u.reply("The weather is lovely today.").stop for _ in range(3)] == [False, False, True]


def test_u10_choice_question_listing_orders_gets_no_yes():
    r = reply(C7, "I see two pending orders: #W7602708 (Garden Hose) and #W1234567 (Desk Lamp). Would you like me to "
                  "cancel one of them?", read(C7))
    assert not r.is_yes and r.intent == "order" and "Garden Hose" in r.text
    r = reply(C0, "Would you like me to cancel #W6554908 or #W3618959?", read(C0))
    assert not r.is_yes and r.intent == "order"


@pytest.mark.parametrize("task, text, intent", [
    (R0, "Could you tell me your name and postal code so I can locate your account?", "identity"),
    (M0, "Who am I speaking with today? I'll need to look up your account first.", "identity"),
    (R0, "I see several delivered orders on your account. Which one contains the Wireless Earbuds?", "order"),
    (C7, "Is the cancellation because you no longer need it, or because it was ordered by mistake?", "reason"),
    (C7, "May I ask what the reason is: no longer needed, or ordered by mistake?", "reason"),
    (E0, "Just the Cycling Helmet, Grill and Jigsaw Puzzle, correct?", "all_items"),
    (E0, "You'll owe an extra $12.55. Would you like to use your PayPal account or your Visa ending in 1234 for "
         "that?", "payment"),
    (M0, "What would you like to change the payment to? I see PayPal and a Visa on file.", "payment"),
    (E0, "Should I charge the $12.55 difference to your PayPal account?", "payment"),
    (R0, "Where should the refund go -- PayPal or your credit card?", "payment"),
])
def test_u11_keyword_gaps_are_answered(task, text, intent):
    r = reply(task, text, read(task))
    assert r.intent == intent, (text, r)


def test_u12_blanket_refusal_before_auth_is_accepted_but_paid_less():
    t = F0
    ep = Episode(t)
    ep.step("I'm sorry, but I can only help with orders on your own account.")
    assert ep.done and ep.log.user_intents == ["accept_denial"]
    assert score_episode(t, ep.result()).reward == pytest.approx(REFUSAL_UNVERIFIED)
    oracle = score_episode(t, run_episode(t, ReferenceAgent(t, "oracle"))).reward
    assert oracle == 1.0


def test_u13_foreign_restate_and_reason_match_the_opening():
    for t in [t for t in TASKS if t.template == "foreign_order_refusal"][:40]:
        u = ScriptedUser(t)
        rel = u.facts.relation
        r = u.reply("The weather is lovely today.")
        assert rel and rel in r.text and rel in t.opening, (t.id, r.text)
        reason = u._say("reason")
        assert ("mistake" in reason) == ("mistake" in t.opening), (t.id, reason)


# ============================================================ E2/E3/E6/E7/E8


def test_e2_every_question_in_a_turn_is_answered():
    r = reply(C7, "Could you please provide your full name and zip code, and also the reason for cancellation?")
    assert r.intents == ["identity", "reason"] and "Juan" in r.text
    assert any(line.rstrip(".") in r.text for line in C7.profile["reason"])
    t = next(t for t in TASKS if t.template == "cancel" and not t.difficulty["reason_upfront"]
             and t.difficulty["give_order_id"] and not t.difficulty["late_correction"])
    r = reply(t, f"Would you like me to cancel order {t.target_order}? If so, please confirm the reason for "
                 "cancellation (either 'no longer needed' or 'ordered by mistake').", read(t))
    assert r.is_yes and r.text.startswith("Yes") and r.text not in YES
    assert any(line.rstrip(".") in r.text for line in t.profile["reason"])


def test_e3_wrong_details_are_corrected_not_consented():
    t = next(t for t in TASKS if t.template == "cancel" and t.difficulty["give_order_id"]
             and not t.difficulty["late_correction"])
    wrong = "ordered by mistake" if t.hidden["reason"] == "no longer needed" else "no longer needed"
    r = reply(t, f"I'll cancel order {t.target_order} with the reason '{wrong}'. Shall I proceed?", read(t))
    assert not r.is_yes and r.revokes and r.intent == "objection"
    assert any(line in r.text for line in t.profile["reason"])
    # payment: an exchange recap naming a card the user did not choose
    r = reply(E0, "I'll exchange the Cycling Helmet, Grill and Jigsaw Puzzle in order #W2809253, charging the "
                  "difference to credit_card_1234567. Shall I proceed?", read(E0))
    assert not r.is_yes and r.revokes and E0.hidden["pm_phrase"] in r.text


def test_e7_name_zip_user_asked_for_an_email_offers_name_and_zip():
    t = C7
    assert t.difficulty["id_mode"] == "name_zip"
    u = ScriptedUser(t)
    first = u.reply("Could you please provide your email address so I can verify your identity?")
    assert "email" in first.text and t.hidden["zip"] in first.text and "@" not in first.text
    second = u.reply("I still need your email address to authenticate you. Could you please provide it?")
    assert second.intent == "identity" and "handy" not in second.text and "remember" not in second.text


def test_e8_option_and_item_questions_get_the_hidden_answer():
    r = reply(E0, "There are several Cycling Helmet variants. Which color would you like?", read(E0))
    assert r.intent == "option" and "blue" in r.text and "everything else the same" in r.text
    r = reply(R0, "Which items would you like to return from that order?", read(R0))
    assert r.intent == "item" and "Wireless Earbuds" in r.text


def test_third_party_identity_request_gets_the_caller_once():
    u = ScriptedUser(F0)
    a = u.reply("To cancel your sister's order, I first need to authenticate her identity. Could you please provide "
                "her email address?")
    assert a.intent == "identity" and "her" in a.text and F0.hidden["last_name"] in a.text
    b = u.reply("I still need her email address. Could you please provide that information?")
    assert b.intent == "identity" and "calling" not in b.text


# ============================================================ anti-exploit


def test_sign_offs_and_other_help_offers_never_harvest_a_yes():
    for text in ("Let me know if you need anything else!",
                 "Is there anything else I can help you with today?",
                 "Would you like me to transfer you to a human agent?",
                 "If you'd like to cancel a different order or need help with something else, please let me know!"):
        r = reply(C7, text, read(C7))
        assert not r.is_yes and not r.text.startswith("Yes"), (text, r)


def test_claim_of_done_without_a_write_gets_the_restate():
    r = reply(R0, "Your return has been initiated.", read(R0))
    assert r.intent == "unrecognised" and "Wireless Earbuds" in r.text


def test_stale_yes_does_not_survive_the_fallback_end_to_end():
    """A yes given before the fallback (M5: the gift card plan) is revoked by the
    fallback; the write to the new card without a new recap is unconfirmed."""
    t = M5
    ep = Episode(t)
    turns = ["Could you please verify your identity with your email or name and zip code?"]
    mode = t.difficulty["id_mode"]
    turns.append(call("find_user_id_by_email", {"email": t.hidden["email"]}) if mode == "email" else
                 call("find_user_id_by_name_zip", {k: t.hidden[k] for k in ("first_name", "last_name", "zip")}))
    turns += [call("get_order_details", {"order_id": t.target_order}),
              "I'm sorry, but your gift card balance is not enough to cover the order total.",
              call("modify_pending_order_payment", {"order_id": t.target_order,
                                                    "payment_method_id": t.gold_actions[-1]["arguments"]["payment_method_id"]})]
    for x in turns:
        ep.step(x)
    assert "fallback" in ep.log.user_intents and ep.log.writes[-1].ok and not ep.log.writes[-1].confirmed
    assert score_episode(t, ep.result()).reward == pytest.approx(0.7)


def test_repeated_objections_end_the_episode():
    """halluc_pm: the user corrects a made-up payment id; the agent repeating it
    hears the correction three times and then the user gives up."""
    u = ScriptedUser(E0)
    txt = ("To confirm, I will exchange items in order #W2809253 (Cycling Helmet, Grill, Jigsaw Puzzle), using "
           "credit_card_1234567. Do you want me to proceed? (yes/no)")
    rs = [u.reply(txt, read(E0)) for _ in range(3)]
    assert [r.stop for r in rs] == [False, False, True] and not any(r.is_yes for r in rs)


def test_no_line_starts_with_yes_unless_it_consents():
    probes = [
        (E0, "Is that all the items you want to exchange?"), (E0, "Which payment method would you like to use?"),
        (R0, "To confirm, I will return the Wireless Earbuds from order #W2923184 with the refund to your Visa "
             "ending in 8025. Shall I proceed?"),
        (C7, "Shall I proceed with the cancellation?"), (F0, "Would you like me to transfer you to a human agent?"),
    ]
    for t, text in probes:
        r = reply(t, text, read(t))
        assert r.text.startswith("Yes") == r.is_yes, (text, r)


# ============================================================ determinism / analyzer plumbing


def test_user_replies_are_deterministic_per_task_and_seed():
    script = ["Could you verify your identity with your email?", "Which order is this about?",
              "Which payment method would you like to use for the price difference?",
              "To confirm, I will exchange the Cycling Helmet, Grill and Jigsaw Puzzle in order #W2809253. Shall I "
              "proceed?"]
    a = [ScriptedUser(E0, seed=5).reply(x).text for x in script]
    b = [ScriptedUser(E0, seed=5).reply(x).text for x in script]
    assert a == b


def test_custom_analyzer_backend_is_used():
    class Fixed:
        def analyze(self, text, task, state=None):
            return Semantics(info_requests=["identity"], identity_target="caller")

    r = ScriptedUser(C7, analyzer=Fixed()).reply("anything at all")
    assert r.intent == "identity"


def test_llm_backend_is_selectable_and_cached_offline(tmp_path):
    an = make_analyzer({"backend": "llm", "base_url": "http://127.0.0.1:9", "model": "m", "cache_dir": str(tmp_path)})
    assert isinstance(an, LLMAnalyzer)
    assert isinstance(make_analyzer(), RulesAnalyzer)
    # a cached answer is served without any network call
    import hashlib

    from tau_forge.episodes.nlu import facts_from

    text = "Could you provide your email?"
    msgs = an._messages(text, facts_from(C7), NLUState())
    key = hashlib.sha256(json.dumps(["m", msgs], sort_keys=True).encode()).hexdigest()
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "info_requests": ["identity"], "identity_target": "caller", "confirmation_request": False,
        "proposed_action": None, "refusal": None, "constraint_statement": False, "offers_other_help_only": False,
        "claims_action_done": False, "narration_only": False}))
    sem = an.analyze(text, C7)
    assert sem.info_requests == ["identity"] and sem.identity_target == "caller"


def test_rules_analyzer_reads_the_gold_label_schema():
    sem = analyze("To confirm, I will cancel order #W7602708 (Garden Hose) because you no longer need it. Shall I "
                  "proceed?", C7)
    lab = sem.to_label()
    assert set(lab) == {"info_requests", "identity_target", "confirmation_request", "proposed_action", "refusal",
                        "constraint_statement", "offers_other_help_only", "claims_action_done", "narration_only"}
    assert lab["confirmation_request"] and lab["proposed_action"]["action"] == "cancel"
    assert lab["proposed_action"]["names_target"] and lab["proposed_action"]["details_match_task"] is True
    assert STOP not in str(lab)
