"""Grader fixes from the zero-variance audit: decision-graded transfers,
`autojunk=False` similarity, empty replies on no-call golds, bare untagged
tool-call JSON, and shaping that no longer pays a premature write on a
read-gold scenario. Each test pins one behaviour change; the numbers in the
docstrings are what the pre-fix grader produced on the same inputs.

Runs on CPU against the shipped db.json and data/synthetic/raw, like
tests/test_grpo_signal.py.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

import pytest

from tau_forge.envs.retail import RetailEnv
from tau_forge.reward.reward import (
    TEXT_SIMILARITY_FLOOR,
    TRANSFER_SUMMARY_MIN_CHARS,
    TRANSFER_TRIVIAL_SUMMARY_SCORE,
    Action,
    _text_similarity_score,
    reward,
    transfer_decision_score,
)
from tau_forge.train import reward_adapter, shaping
from tau_forge.train.completion_parsing import MALFORMED_TOOL_CALL, parse_completion
from tau_forge.train.dataset import load_scenarios
from tau_forge.train.reward_adapter import _get_shared_db, grpo_reward_func, score_completion

TRANSFER = "transfer_to_human_agents"
PENDING_ORDER = "#W5918442"  # pending, user sofia_rossi_8776 (see tests/test_reward.py)


def _call(name, args=None, key="arguments"):
    return "<tool_call>\n" + json.dumps({"name": name, key: args or {}}) + "\n</tool_call>"


def _gold(scenario):
    calls = scenario.get("expected_tool_calls") or []
    return (calls[0]["name"], calls[0].get("arguments", {})) if calls else (None, {})


@pytest.fixture(scope="module")
def env():
    return RetailEnv()


@pytest.fixture(scope="module")
def db():
    return _get_shared_db()


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


# --------------------------------------------------------------------------
# 1. transfer_to_human_agents is graded on the decision, never on gold prose
# --------------------------------------------------------------------------

GOLD_TRANSFER = Action(
    TRANSFER,
    {
        "summary": (
            "User sofia_rossi_8776 wants the refund for order #W5918442 issued as a paper check; "
            "policy only allows the original payment method or a gift card."
        )
    },
)


def test_transfer_with_real_summary_scores_full_regardless_of_gold_wording(db):
    """Pre-fix, a correct paraphrase of this request scored max(0.3, ratio) --
    on the corpus half of 50 hand-written correct paraphrases sat at 0.300."""
    paraphrase = Action(TRANSFER, {"summary": "Customer asks to get an order refund sent as a check."})
    result = reward(paraphrase, GOLD_TRANSFER, db)
    assert (result.score, result.reason) == (1.0, "transfer_decision")


def test_gold_wording_with_wrong_content_no_longer_outscores_a_correct_summary(db):
    """Pre-fix, gold's wording with the wrong order/user id averaged 0.974 while
    correct paraphrases averaged 0.378. Now neither is compared to gold."""
    wrong_ids = Action(
        TRANSFER, {"summary": GOLD_TRANSFER.tool_input["summary"].replace("#W5918442", "#W0000000")}
    )
    correct = Action(TRANSFER, {"summary": "Customer wants order #W5918442 refunded by paper check."})
    assert reward(wrong_ids, GOLD_TRANSFER, db).score == reward(correct, GOLD_TRANSFER, db).score == 1.0


def test_transfer_score_does_not_depend_on_gold_summary(db):
    rollout = Action(TRANSFER, {"summary": "Customer wants a refund issued as a paper check."})
    other_gold = Action(TRANSFER, {"summary": "Completely unrelated gold text about a warranty extension."})
    assert reward(rollout, GOLD_TRANSFER, db) == reward(rollout, other_gold, db)
    # And the grader itself only ever sees the rollout's arguments.
    assert transfer_decision_score(rollout.tool_input) == reward(rollout, GOLD_TRANSFER, db)


@pytest.mark.parametrize("summary", ["", "   \n\t", "n/a", "transfer", "Customer needs help."])
def test_empty_or_trivial_transfer_summary_scores_partial(db, summary):
    """Pre-fix an empty summary scored 0.300 on 107/107 out_of_scope scenarios
    -- identical to a good paraphrase that happened to fall under the floor."""
    result = reward(Action(TRANSFER, {"summary": summary}), GOLD_TRANSFER, db)
    assert (result.score, result.reason) == (TRANSFER_TRIVIAL_SUMMARY_SCORE, "transfer_trivial_summary")


def test_transfer_summary_threshold_counts_non_whitespace_characters():
    just_under = " ".join("x" * (TRANSFER_SUMMARY_MIN_CHARS - 1))  # padded with spaces
    at = "x" * TRANSFER_SUMMARY_MIN_CHARS
    assert transfer_decision_score({"summary": just_under}).score == TRANSFER_TRIVIAL_SUMMARY_SCORE
    assert transfer_decision_score({"summary": at}).score == 1.0


def test_trivial_transfer_sits_between_the_right_tool_floor_and_full_credit():
    assert 0.3 < TRANSFER_TRIVIAL_SUMMARY_SCORE < 1.0


def test_transfer_keeps_the_other_tiers(db):
    # Hallucinated extra argument: still the schema tier, not a decision grade.
    extra = Action(TRANSFER, {"summary": "Customer wants a paper check refund.", "priority": "high"})
    assert reward(extra, GOLD_TRANSFER, db).reason == "schema_invalid_or_hallucinated_args"
    # Transfer against a different gold, or against silence, is still a miss.
    good = Action(TRANSFER, {"summary": "Customer wants a paper check refund."})
    assert reward(good, Action("cancel_pending_order", {"order_id": PENDING_ORDER}), db).score == 0.0
    assert reward(good, Action(None), db).score == 0.0


def test_transfer_scoring_never_executes_gold():
    """A transfer is graded from the rollout alone, so the adapter must not pay
    for (or cache) a 2.8MB gold execution to grade it."""
    reward_adapter._gold_cache.clear()
    score = score_completion(
        _call(TRANSFER, {"summary": "Customer wants a paper check refund."}),
        TRANSFER,
        GOLD_TRANSFER.tool_input,
    )
    assert score == 1.0
    assert not reward_adapter._gold_cache


def test_every_out_of_scope_scenario_rewards_the_transfer_decision(scenarios):
    """Corpus-wide: restating the user's request as the summary scores 1.0 on
    all 107 out_of_scope scenarios (pre-fix: mean 0.344, 71/107 at 0.300), and
    an empty summary scores the partial tier on all of them (pre-fix 0.300)."""
    oos = [s for s in scenarios if s["category"] == "out_of_scope"]
    assert len(oos) == 107
    for s in oos:
        name, args = _gold(s)
        assert name == TRANSFER
        restated = _call(TRANSFER, {"summary": "Customer request: " + s["user_message"]})
        assert score_completion(restated, name, args) == 1.0, s["id"]
        empty = _call(TRANSFER, {"summary": ""})
        assert score_completion(empty, name, args) == TRANSFER_TRIVIAL_SUMMARY_SCORE, s["id"]


# --------------------------------------------------------------------------
# 2. SequenceMatcher autojunk is off
# --------------------------------------------------------------------------

def test_similarity_is_not_distorted_on_long_strings(scenarios):
    """difflib's autojunk kicks in at 200+ chars and junks every common letter.
    The audit's example: a 283-char string from the corpus and a two-edit
    paraphrase of it (User->Customer, '; '->' -- ') measured 0.724 with
    autojunk and 0.986 without."""
    by_id = {s["id"]: s for s in scenarios}
    long_text = _gold(by_id["out_of_scope__address_payment_modification__006"])[1]["summary"]
    assert len(long_text) >= 200
    paraphrase = long_text.replace("User ", "Customer ", 1).replace("; ", " -- ")
    expected_ratio = SequenceMatcher(None, paraphrase.lower(), long_text.lower(), autojunk=False).ratio()
    junked_ratio = SequenceMatcher(None, paraphrase.lower(), long_text.lower()).ratio()
    # The bug being fixed is real on this input, not hypothetical...
    assert expected_ratio - junked_ratio > 0.2
    # ...and the grader now uses the un-junked ratio.
    floor = TEXT_SIMILARITY_FLOOR
    assert _text_similarity_score(paraphrase, long_text) == pytest.approx(
        (expected_ratio - floor) / (1 - floor)
    )
    assert _text_similarity_score(paraphrase, long_text) > 0.95


# --------------------------------------------------------------------------
# 3. An empty reply is not a correct no-call
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\n", "\t \n"])
def test_empty_completion_on_no_call_gold_scores_zero(text):
    """Pre-fix: 1.0 on every one of the 182 no-call scenarios."""
    assert score_completion(text, None, {}) == 0.0
    result = reward(Action(None, text=text), Action(None), _get_shared_db())
    assert (result.score, result.reason) == (0.0, "empty_reply")


def test_non_empty_reply_on_no_call_gold_still_scores_full():
    assert score_completion("Which of your two orders did you mean?", None, {}) == 1.0


def test_action_without_text_keeps_the_old_contract():
    """Callers that only have parsed actions (the multi-step harness) never set
    `text`; `None` means unknown, so they are graded exactly as before."""
    assert reward(Action(None), Action(None), _get_shared_db()).reason == "correct_no_call"


def test_empty_reply_through_the_trl_adapter():
    out = grpo_reward_func(
        prompts=["p", "p"],
        completions=["", "Could you confirm the order number?"],
        expected_tool_name=[None, None],
        expected_tool_arguments_json=["{}", "{}"],
    )
    assert out == [0.0, 1.0]


def test_empty_completion_on_call_gold_is_still_a_missing_call():
    assert score_completion("", "cancel_pending_order", {"order_id": PENDING_ORDER}) == 0.0


# --------------------------------------------------------------------------
# 4. A bare (untagged) tool-call object is an attempted call, not a reply
# --------------------------------------------------------------------------

BARE = json.dumps(
    {"name": "cancel_pending_order", "arguments": {"order_id": PENDING_ORDER, "reason": "no longer needed"}}
)


@pytest.mark.parametrize(
    "text",
    [
        BARE,
        "```json\n" + BARE + "\n```",
        "Sure, cancelling that now.\n" + BARE,
        json.dumps({"name": "cancel_pending_order", "parameters": {"order_id": PENDING_ORDER}}),
        json.dumps({"function": {"name": "cancel_pending_order", "arguments": {"order_id": PENDING_ORDER}}}),
        json.dumps([{"name": "get_order_details", "arguments": {"order_id": PENDING_ORDER}}]),
    ],
)
def test_bare_tool_call_json_is_malformed_not_a_no_call(text):
    assert parse_completion(text) == (MALFORMED_TOOL_CALL, {})
    # Pre-fix: 1.0 on every no-call gold -- the reward-hacking path.
    assert score_completion(text, None, {}) == 0.0


def test_bare_gold_call_is_not_accepted_as_the_call():
    """vLLM's hermes parser would not see a call here at eval, so training must
    not pay for one either."""
    args = {"order_id": PENDING_ORDER, "reason": "no longer needed"}
    assert score_completion(BARE, "cancel_pending_order", args) == 0.0
    assert shaping.shaping_score(BARE, "cancel_pending_order", args, RetailEnv()) == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Your order details: {\"status\": \"pending\", \"items\": 2}. Shall I proceed?",
        'I have you down as {"name": "Sofia Rossi"} -- is that right?',
        "The \"name\" on the account doesn't match; can you confirm your email?",
        "Unbalanced { brace and \"name\" mention",
    ],
)
def test_prose_with_incidental_json_is_still_a_reply(text):
    assert parse_completion(text) == (None, {})


def test_tagged_call_still_wins_over_any_bare_json():
    text = _call("get_order_details", {"order_id": PENDING_ORDER}) + "\n" + BARE
    assert parse_completion(text) == ("get_order_details", {"order_id": PENDING_ORDER})


def test_other_wire_formats_inside_tags_are_not_repaired():
    """Parity with eval's hermes parser beats leniency: these are not accepted
    as the calls they resemble (see completion_parsing's module docstring)."""
    # A ```json fence inside the tags: hermes' json.loads fails -> no call.
    fenced = "<tool_call>\n```json\n" + BARE + "\n```\n</tool_call>"
    assert parse_completion(fenced) == (MALFORMED_TOOL_CALL, {})
    # 'parameters' instead of 'arguments': arguments are never read from it.
    params = _call("get_order_details", {"order_id": PENDING_ORDER}, key="parameters")
    assert parse_completion(params) == ("get_order_details", {})
    # Arguments as a JSON-encoded string are never decoded.
    as_string = (
        '<tool_call>\n{"name": "get_order_details", "arguments": '
        + json.dumps(json.dumps({"order_id": PENDING_ORDER}))
        + "}\n</tool_call>"
    )
    assert parse_completion(as_string) == ("get_order_details", {})


# --------------------------------------------------------------------------
# 5. Shaping never pays a premature write on a read-gold scenario
# --------------------------------------------------------------------------


def test_premature_write_on_gold_order_scores_below_a_legitimate_other_read(env, db):
    """Verifier 'missed' #1, on policy_violation__order_state_confusion__002's
    shape: gold is a status check. Pre-fix the write scored 0.13, the read 0.10."""
    gold_name, gold_args = "get_order_details", {"order_id": "#W2611340"}
    owner = db.orders["#W2611340"].user_id
    write = shaping.shaping_score(
        _call("cancel_pending_order", {"order_id": "#W2611340", "reason": "no longer needed"}),
        gold_name,
        gold_args,
        env,
    )
    other_read = shaping.shaping_score(
        _call("get_user_details", {"user_id": owner}), gold_name, gold_args, env
    )
    assert write == pytest.approx(shaping.PARSEABLE_CALL + shaping.REAL_TOOL)
    assert other_read == pytest.approx(
        shaping.PARSEABLE_CALL + shaping.REAL_TOOL + shaping.SCHEMA_VALID + shaping.SAME_TOOL_CLASS
    )
    assert write < other_read


def test_premature_write_ranks_last_among_real_tools_on_every_pv_status_check(env, db, scenarios):
    """All 26 policy_violation scenarios whose gold is get_order_details."""
    pv = [
        s for s in scenarios
        if s["category"] == "policy_violation" and _gold(s)[0] == "get_order_details"
    ]
    assert len(pv) == 26
    for s in pv:
        name, args = _gold(s)
        order_id = args["order_id"]
        write = _call("cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"})
        read = _call("get_user_details", {"user_id": db.orders[order_id].user_id})
        w = score_completion(write, name, args) + shaping.shaping_score(write, name, args, env)
        r = score_completion(read, name, args) + shaping.shaping_score(read, name, args, env)
        assert w < r, s["id"]


def test_write_when_gold_is_a_transfer_gets_only_the_real_tool_floor(env):
    score = shaping.shaping_score(
        _call("cancel_pending_order", {"order_id": PENDING_ORDER, "reason": "no longer needed"}),
        TRANSFER,
        {"summary": "Customer wants order " + PENDING_ORDER + " refunded by check."},
        env,
    )
    assert score == pytest.approx(shaping.PARSEABLE_CALL + shaping.REAL_TOOL)


def test_read_on_gold_record_when_gold_is_a_write_keeps_its_credit(env):
    """One-directional on purpose: looking up the order before writing is the
    step the policy usually requires, so it keeps right-record credit."""
    score = shaping.shaping_score(
        _call("get_order_details", {"order_id": PENDING_ORDER}),
        "cancel_pending_order",
        {"order_id": PENDING_ORDER, "reason": "no longer needed"},
        env,
    )
    assert score == pytest.approx(
        shaping.PARSEABLE_CALL + shaping.REAL_TOOL + shaping.SCHEMA_VALID + shaping.RIGHT_TARGET_RECORD
    )


def test_wrong_write_on_gold_record_when_gold_is_a_write_is_unchanged(env):
    score = shaping.shaping_score(
        _call(
            "modify_pending_order_address",
            {
                "order_id": PENDING_ORDER,
                "address1": "1 Main St",
                "address2": "",
                "city": "Austin",
                "state": "TX",
                "country": "USA",
                "zip": "78701",
            },
        ),
        "cancel_pending_order",
        {"order_id": PENDING_ORDER, "reason": "no longer needed"},
        env,
    )
    assert score == pytest.approx(shaping.WRONG_TOOL_CEILING)
    assert shaping.WRONG_TOOL_CEILING < 0.2
