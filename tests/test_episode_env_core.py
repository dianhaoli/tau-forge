"""Stage A of the episode env rebuild: thread safety, hermes parsing parity,
the runner <-> user interface (slots, structured consent, TurnContext), the
refusal and shaping reward changes, and per-task budgets.

Works on the committed seed-1 task file (`data/episodes/episodes_s1.jsonl`),
generated from db.json only. Never reads tasks.json.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import pytest
from tau2.utils import get_dict_hash

from tau_forge.envs.retail import RetailEnv
from tau_forge.episodes.generate import pm_phrase, read_jsonl
from tau_forge.episodes.reference_agents import ReferenceAgent, call
from tau_forge.episodes.reward import REFUSAL_UNVERIFIED, score_episode
from tau_forge.episodes.runner import Episode, EpisodeLog, EpisodeResult, WriteRecord, run_episode
from tau_forge.episodes.task import EpisodeTask, base_db
from tau_forge.episodes.user import YES, ScriptedUser, TurnContext, UserReply
from tau_forge.train.completion_parsing import (
    MALFORMED_TOOL_CALL,
    parse_all_completion,
    parse_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = read_jsonl(REPO_ROOT / "data" / "episodes" / "episodes_s1.jsonl")
ASK_EMAIL = "Could you please verify your identity with your email address?"


def _pick(template: str, **difficulty: Any) -> EpisodeTask:
    return next(
        t for t in TASKS
        if t.template == template and all(t.difficulty.get(k) == v for k, v in difficulty.items())
    )


def _scripted(turns: list):
    it = iter(turns)

    def policy(messages):
        t = next(it)
        return t(messages) if callable(t) else t

    return policy


def _drive(ep: Episode, turns: list) -> Episode:
    for t in turns:
        assert not ep.done, (ep.end_reason, ep.messages[-1])
        ep.step(t)
    return ep


def _auth(t: EpisodeTask) -> list[str]:
    """Ask for the email, then look it up: an earned auth on an email task."""
    assert t.difficulty["id_mode"] == "email"
    return [ASK_EMAIL, call("find_user_id_by_email", {"email": t.hidden["email"]})]


def _recap(t: EpisodeTask, what: str) -> str:
    return f"To confirm, I will {what} order {t.target_order}. Do you want me to proceed? (yes/no)"


# ======================================================================= F1


def _oracle_reward(t: EpisodeTask) -> float:
    return score_episode(t, run_episode(t, ReferenceAgent(t, "oracle"))).reward


def _write_tasks(per_template: int) -> list[EpisodeTask]:
    out = []
    for tpl in ("cancel", "exchange", "return_fallback", "modify_payment"):
        out += [t for t in TASKS if t.template == tpl][:per_template]
    return out


def test_threaded_episodes_never_cross_dbs_under_a_forced_preemption(monkeypatch):
    """Deterministic version of F1: a sleep inside the tool call stands in for
    a thread switch between binding the shared env and running the tool.
    Unlocked, every threaded oracle write scored 0.2 and base_db() was
    mutated for the rest of the process."""
    pristine = get_dict_hash(base_db().model_dump())
    real = RetailEnv.execute

    def slow(self, name, arguments):
        time.sleep(0.002)
        return real(self, name, arguments)

    monkeypatch.setattr(RetailEnv, "execute", slow)
    tasks = _write_tasks(4)
    with ThreadPoolExecutor(8) as pool:
        rewards = list(pool.map(_oracle_reward, tasks))
    assert rewards == [1.0] * len(tasks)
    assert get_dict_hash(base_db().model_dump()) == pristine


def test_threaded_oracle_stress_keeps_rewards_and_the_base_db():
    """F1 under real preemption: 16 threads, a 1 us switch interval (the
    verifier saw 158/160 wrong rewards unlocked)."""
    pristine = get_dict_hash(base_db().model_dump())
    tasks = _write_tasks(40)
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(16) as pool:
            rewards = list(pool.map(_oracle_reward, tasks))
    finally:
        sys.setswitchinterval(old)
    assert rewards == [1.0] * len(tasks)
    assert get_dict_hash(base_db().model_dump()) == pristine


# ============================================================ parsing parity
# Reference: vLLM Hermes2ProToolParser.extract_tool_calls (v0.10.1 / main),
# copied verbatim in logic, and the ToolCall(arguments: dict) step tau2's
# llm_utils.generate applies to its output. Kept independent of the module
# under test on purpose.

_REF_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)


def _reference_eval(model_output: str):
    """('text', content) or ('calls', [(name, args)], content)."""
    if "<tool_call>" not in model_output:
        return ("text", model_output)
    try:
        raw = [json.loads(m[0] if m[0] else m[1]) for m in _REF_RE.findall(model_output)]
        calls = [(fc["name"], json.dumps(fc["arguments"], ensure_ascii=False)) for fc in raw]
        for n, _ in calls:
            if not isinstance(n, str):  # pydantic FunctionCall(name: str)
                raise TypeError
        content = model_output[: model_output.find("<tool_call>")]
        content = content if content else None
    except Exception:
        return ("text", model_output)
    out = []
    for n, a in calls:
        args = json.loads(a)
        if not isinstance(args, dict):  # tau2: ToolCall(arguments: dict) raises -> never a call
            return ("text", model_output)
        out.append((n, args))
    return ("calls", out, content or "")


def _blk(body: str) -> str:
    return f"<tool_call>\n{body}\n</tool_call>"


_GOD = '{"name": "get_order_details", "arguments": {"order_id": "#W1"}}'
PARITY_CORPUS = {
    "canonical": _blk(_GOD),
    "compact": '<tool_call>{"name":"get_order_details","arguments":{"order_id":"#W1"}}</tool_call>',
    "no_arguments_key": _blk('{"name": "list_all_product_types"}'),
    "parameters_key": _blk('{"name": "get_order_details", "parameters": {"order_id": "#W1"}}'),
    "args_null": _blk('{"name": "list_all_product_types", "arguments": null}'),
    "args_string": _blk('{"name": "get_order_details", "arguments": "{\\"order_id\\": \\"#W1\\"}"}'),
    "args_list": _blk('{"name": "get_order_details", "arguments": ["#W1"]}'),
    "name_int": _blk('{"name": 7, "arguments": {}}'),
    "name_missing": _blk('{"arguments": {}}'),
    "body_string": _blk('"get_order_details"'),
    "body_list": _blk("[" + _GOD + "]"),
    "empty_block": "<tool_call></tool_call>",
    "two_valid_calls": _blk(_GOD) + "\n" + _blk(_GOD.replace("#W1", "#W2")),
    "three_calls_text_between": "A" + _blk(_GOD) + " B " + _blk(_GOD) + " C " + _blk('{"name": "x", "arguments": {}}'),
    "valid_then_broken": _blk(_GOD) + "\n" + _blk('{"name": "get_order_details", "arguments": {"order_id": "#W2"}'),
    "valid_then_noargs": _blk(_GOD) + "\n" + _blk('{"name": "list_all_product_types"}'),
    "text_then_call": "Let me check that.\n" + _blk(_GOD),
    "call_then_text": _blk(_GOD) + "\nOne moment please.",
    "text_call_text": "Checking now." + _blk(_GOD) + "\nOne sec.",
    "empty_name": _blk('{"name": "", "arguments": {}}'),
    "no_close_tag": "<tool_call>\n" + _GOD,
    "no_close_tag_truncated_json": "<tool_call>\n" + _GOD[:-5],
    "fenced_inside": _blk("```json\n" + _GOD + "\n```"),
    "prose_inside": _blk("Sure! " + _GOD),
    "unicode_args": _blk('{"name": "calculate", "arguments": {"expression": "2 × 3"}}'),
    "nested_args": _blk('{"name": "exchange_delivered_order_items", "arguments": {"order_id": "#W1", '
                        '"item_ids": ["1", "2"], "new_item_ids": ["3", "4"], "payment_method_id": "paypal_1"}}'),
    "bare_json": _GOD,
    "fenced_bare": "```json\n" + _GOD + "\n```",
    "plain_text": "Sure, I can help with that. What is your email?",
    "empty": "",
    "close_tag_only": "</tool_call> hello",
}


@pytest.mark.parametrize("case", sorted(PARITY_CORPUS))
def test_parse_all_completion_matches_hermes_and_tau2(case):
    text = PARITY_CORPUS[case]
    ref = _reference_eval(text)
    got = parse_all_completion(text)
    if ref[0] == "text":
        assert got.calls == [] and got.content == text, case
    else:
        assert got.calls == ref[1] and got.content == ref[2], case


@pytest.mark.parametrize("case", sorted(PARITY_CORPUS))
def test_single_step_parse_completion_rejects_what_eval_rejects(case):
    text = PARITY_CORPUS[case]
    ref = _reference_eval(text)
    name, args = parse_completion(text)
    if ref[0] == "calls":
        first_name, first_args = ref[1][0]
        if first_name:
            assert (name, args) == (first_name, first_args), case
        else:  # stricter than hermes: an empty name is never a real tool
            assert name == MALFORMED_TOOL_CALL, case
    elif "<tool_call>" in text or case in ("bare_json", "fenced_bare"):
        # anything eval turns into text that LOOKS like a call is an attempt,
        # never a free no-call
        assert (name, args) == (MALFORMED_TOOL_CALL, {}), case
    else:
        assert (name, args) == (None, {}), case


def test_specific_parity_cases_the_audit_found():
    assert parse_all_completion(PARITY_CORPUS["two_valid_calls"]).calls == [
        ("get_order_details", {"order_id": "#W1"}), ("get_order_details", {"order_id": "#W2"})]
    for case in ("no_arguments_key", "parameters_key", "args_null", "args_string", "valid_then_broken",
                 "valid_then_noargs"):
        p = parse_all_completion(PARITY_CORPUS[case])
        assert p.calls == [] and p.malformed, case
    assert parse_all_completion(PARITY_CORPUS["text_call_text"]).content == "Checking now."
    assert parse_all_completion(PARITY_CORPUS["no_close_tag"]).calls == [("get_order_details", {"order_id": "#W1"})]


# ============================================================ runner: turns


def test_every_parsed_call_runs_under_one_assistant_message():
    t = _pick("cancel", give_order_id=True, late_correction=False, id_mode="email")
    ep = Episode(t)
    two = call("get_order_details", {"order_id": t.target_order}) + "\n" + call(
        "cancel_pending_order", {"order_id": t.target_order, "reason": t.hidden["reason"]})
    ep.step(two)
    a, r1, r2 = ep.messages[-3:]
    assert a["role"] == "assistant" and [tc["function"]["name"] for tc in a["tool_calls"]] == [
        "get_order_details", "cancel_pending_order"]
    assert a["content"] == "" and a["raw"] == two
    assert [r1["role"], r2["role"]] == ["tool", "tool"]
    assert [r1["tool_call_id"], r2["tool_call_id"]] == [tc["id"] for tc in a["tool_calls"]]
    assert ep.log.n_calls == 2 and ep.log.n_multi_call_turns == 1 and ep.log.n_assistant_turns == 1
    # the write ran, as eval runs it -- unconfirmed and unauthenticated
    (w,) = ep.log.writes
    assert w.ok and not w.confirmed and w.slot == "main" and w.authed_user is None
    r = score_episode(t, ep.result())
    assert r.success and set(r.gates) == {"no_confirmation", "no_authentication"}


def test_raw_completion_is_kept_on_every_assistant_message_and_content_is_text_before_the_call():
    t = _pick("cancel", give_order_id=True)
    ep = Episode(t)
    sampled = 'Checking now.<tool_call>\n{"name":"get_order_details","arguments":{"order_id":"' + t.target_order + \
        '"}}\n</tool_call>\nOne sec.'
    ep.step(sampled)
    ep.step("Could you tell me the reason?")
    ep.step("I'll look into ord", "length")
    assistants = [m for m in ep.messages if m["role"] == "assistant"][1:]  # skip the greeting
    assert [m["raw"] for m in assistants] == [sampled, "Could you tell me the reason?", "I'll look into ord"]
    assert assistants[0]["content"] == "Checking now."  # eval drops the trailing text
    assert json.loads(assistants[0]["tool_calls"][0]["function"]["arguments"]) == {"order_id": t.target_order}


def test_a_turn_hermes_rejects_reaches_the_user_as_text_and_runs_nothing():
    t = _pick("cancel", give_order_id=True)
    for case in ("valid_then_broken", "parameters_key", "args_null"):
        ep = Episode(t)
        ep.step(PARITY_CORPUS[case].replace("#W1", t.target_order))
        assert ep.log.n_calls == 0 and ep.log.n_malformed_calls == 1, case
        assert ep.messages[-2]["role"] == "assistant" and "tool_calls" not in ep.messages[-2]
        assert ep.messages[-1]["role"] == "user"


def test_a_truncated_turn_that_still_parses_is_executed_as_at_eval():
    t = _pick("cancel", give_order_id=True)
    ep = Episode(t)
    ep.step(call("get_order_details", {"order_id": t.target_order}), "length")
    assert not ep.done and ep.log.read_orders == [t.target_order]
    assert ep.messages[-2]["truncated"] is True and ep.messages[-1]["role"] == "tool"
    # without the closing tag hermes still reads a complete JSON body
    ep.step("<tool_call>\n" + json.dumps({"name": "get_user_details", "arguments": {"user_id": t.user_id}}), "length")
    assert ep.log.n_calls == 2 and not ep.done
    # a half-written body is text: the episode ends, masked
    ep.step('<tool_call>\n{"name": "cancel_pending_order", "argu', "length")
    assert ep.end_reason == "truncated" and not ep.log.writes
    r = score_episode(t, ep.result())
    assert r.masked and r.to_dict()["masked"]
    assert not score_episode(t, run_episode(t, ReferenceAgent(t, "oracle"))).masked


def test_transfer_ends_the_episode_after_its_turn_has_run():
    t = _pick("cancel", give_order_id=True)
    ep = Episode(t)
    ep.step(call("transfer_to_human_agents", {"summary": "x"}) + call("get_order_details", {"order_id": t.target_order}))
    assert ep.done and ep.end_reason == "transfer"
    assert [m["role"] for m in ep.messages[-3:]] == ["assistant", "tool", "tool"]


def test_same_completions_give_the_same_messages_and_reward():
    for t in [_pick(tpl) for tpl in ("cancel", "exchange", "return_fallback", "modify_payment",
                                     "foreign_order_refusal")]:
        a = run_episode(t, ReferenceAgent(t, "oracle"))
        completions = [m["raw"] for m in a.messages if m["role"] == "assistant" and "raw" in m]
        b = run_episode(t, _scripted(completions))
        assert a.messages == b.messages
        assert score_episode(t, a).to_dict() == score_episode(t, b).to_dict()


# ============================================================ consent / slots


def test_slots_are_derived_for_legacy_tasks_and_serialisation_is_unchanged():
    lines = (REPO_ROOT / "data" / "episodes" / "episodes_s1.jsonl").read_text().splitlines()
    for line, t in zip(lines, TASKS):
        if t.expect_no_write:
            assert t.slots == []
        else:
            gold = next(a for a in t.gold_actions if a["name"].startswith(("cancel", "exchange", "return", "modify")))
            assert t.slots == [{"id": "main", "tool": gold["name"], "record": f"order:{t.target_order}"}]
        assert json.dumps(t.to_dict()) == line  # old JSONL round-trips byte for byte
    t = TASKS[0]
    d = {**t.to_dict(), "gold_actions": [{"name": "modify_user_address", "arguments": {"user_id": t.user_id}}]}
    assert EpisodeTask.from_dict(d).slots == [{"id": "main", "tool": "modify_user_address", "record": f"user:{t.user_id}"}]
    custom = [{"id": "a", "tool": "cancel_pending_order", "record": "order:#W1"}]
    d = {**t.to_dict(), "slots": custom, "max_turns": 50}
    back = EpisodeTask.from_dict(d)
    assert back.slots == custom and back.max_turns == 50 and back.max_calls is None
    assert EpisodeTask.from_dict(back.to_dict()).to_dict() == back.to_dict() and "slots" in back.to_dict()


def test_per_task_budgets_override_the_episode_arguments():
    t = _pick("cancel")
    assert (Episode(t).max_turns, Episode(t).max_calls) == (30, 30)
    assert Episode(t, max_turns=7).max_turns == 7
    short = EpisodeTask.from_dict({**t.to_dict(), "max_turns": 3, "max_calls": 2})
    ep = Episode(short, max_turns=30, max_calls=30)
    assert (ep.max_turns, ep.max_calls) == (3, 2)
    while not ep.done:
        ep.step(call("get_user_details", {"user_id": t.user_id}))
    assert ep.end_reason == "max_calls" and ep.log.n_calls == 2


def test_scripted_user_reply_carries_structured_consent():
    t = _pick("cancel", late_correction=False)
    u = ScriptedUser(t)
    yes = u.reply(_recap(t, "cancel"))
    assert yes.intent == "yes" and yes.confirms == ["main"] and yes.is_yes and yes.intents == ["yes"]
    assert not yes.revokes
    corr = ScriptedUser(_pick("cancel", late_correction=True))
    c = corr.reply(f"To confirm, I will cancel order {corr.task.profile['recap_keys'][0]}. Shall I proceed?")
    assert c.intent == "correction" and c.revokes and not c.confirms
    ret = _pick("return_fallback")
    f = ScriptedUser(ret).reply("I'm sorry, refunds can only go to the original payment method. How should I proceed?")
    assert f.intent == "fallback" and f.revokes  # E1/U8: the plan changed
    foreign = _pick("foreign_order_refusal")
    fy = ScriptedUser(foreign).reply(f"I will cancel order {foreign.target_order}. Do you want me to proceed?")
    assert fy.intent == "yes" and fy.confirms == [] and not fy.is_yes  # a refusal task has no slot to confirm
    done = ScriptedUser(t).reply("All set!", TurnContext(any_write_ok=True))
    assert done.stop and done.intent == "thanks"
    assert UserReply("x", False, "a", confirms=["s"]).is_yes and UserReply("x", False, "a").intents == ["a"]


class _StubUser:
    """Replays given UserReply objects and records the TurnContext it saw."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def reply(self, agent_text, ctx=None, write_succeeded=False):
        self.seen.append(ctx)
        return self.replies.pop(0)


def _yes_reply(*slots):
    return UserReply("Yes, go ahead.", False, "yes", confirms=list(slots))


def _return_task_ids(t: EpisodeTask):
    user = base_db().users[t.user_id]
    bad = next(p.id for p in user.payment_methods.values() if pm_phrase(p.model_dump()) == t.hidden["bad_pm_phrase"])
    orig = base_db().orders[t.target_order].payment_history[0].payment_method_id
    return bad, orig


def _return_call(t, pm):
    return call("return_delivered_order_items",
                {"order_id": t.target_order, "item_ids": t.hidden["item_ids"], "payment_method_id": pm})


def test_a_failed_write_suspends_its_yes_so_a_silent_retry_is_unconfirmed():
    t = _pick("return_fallback", id_mode="email")
    bad, orig = _return_task_ids(t)
    ep = _drive(Episode(t), _auth(t) + [
        call("get_order_details", {"order_id": t.target_order}),
        _recap(t, f"return the items, refunding {bad}, from"),
        _return_call(t, bad),   # the tool refuses a non-original method
        _return_call(t, orig),  # retried with no word to the user
    ])
    # stage B: before the fallback the user still wants the forbidden card and never consents (r3_0-1),
    # so neither write is confirmed; the retry is still docked
    assert "yes" not in ep.log.user_intents and ep.log.user_intents[-1] == "payment"
    first, second = ep.log.writes
    assert not first.confirmed and not first.ok and second.ok and not second.confirmed
    assert score_episode(t, ep.result()).reward == pytest.approx(0.7)


def test_a_yes_to_the_old_plan_does_not_survive_the_fallback():
    """U8 A1 / E1: yes to the forbidden refund card, the write fails, the
    agent explains, the user falls back, the agent writes the original
    method without a new recap -> unconfirmed. With a new recap -> 1.0."""
    t = _pick("return_fallback", id_mode="email")
    bad, orig = _return_task_ids(t)
    head = _auth(t) + [
        call("get_order_details", {"order_id": t.target_order}),
        _recap(t, f"return the items, refunding {bad}, from"),
        _return_call(t, bad),
        "I'm sorry, refunds can only go to the original payment method or a gift card. How should I proceed?",
    ]
    ep = _drive(Episode(t), head + [_return_call(t, orig)])
    assert ep.log.user_intents[-2:] == ["payment", "fallback"]  # stage B: no yes to the forbidden card
    assert not ep.log.writes[-1].confirmed and ep.log.writes[-1].ok
    assert score_episode(t, ep.result()).reward == pytest.approx(0.7)
    ep = _drive(Episode(t), head + [_recap(t, f"return the items, refunding {orig}, from"), _return_call(t, orig)])
    assert ep.log.writes[-1].confirmed and score_episode(t, ep.result()).reward == 1.0


def test_consent_rules_with_a_structured_user():
    t = _pick("cancel", give_order_id=True, late_correction=False)
    write = call("cancel_pending_order", {"order_id": t.target_order, "reason": t.hidden["reason"]})
    bad_write = call("cancel_pending_order", {"order_id": t.target_order, "reason": "not a reason"})

    def run(replies, turns):
        ep = Episode(t)
        ep.user = _StubUser(replies)
        return _drive(ep, turns)

    # fail -> suspended; a fallback reply (no revoke) restores it
    ep = run([_yes_reply("main"), UserReply("Fine, do that.", False, "fallback")], ["recap", bad_write, "explain", write])
    assert [w.confirmed for w in ep.log.writes] == [True, True] and ep.log.writes[-1].ok
    # fail -> suspended; an unrelated answer does not restore it
    ep = run([_yes_reply("main"), UserReply("ok", False, "reason")], ["recap", bad_write, "explain", write])
    assert [w.confirmed for w in ep.log.writes] == [True, False]
    # fail -> suspended; a new confirms restores it
    ep = run([_yes_reply("main"), _yes_reply("main")], ["recap", bad_write, "recap again", write])
    assert [w.confirmed for w in ep.log.writes] == [True, True]
    # revokes clears every slot, even in the same reply as a fallback
    ep = run([_yes_reply("main"), UserReply("Actually...", False, "fallback", revokes=True)],
             ["recap", bad_write, "explain", write])
    assert [w.confirmed for w in ep.log.writes] == [True, False]
    ep = run([_yes_reply("main"), UserReply("Wait", False, "correction", revokes=True)], ["recap", "more", write])
    assert not ep.log.writes[0].confirmed
    # a confirms for a slot the task does not have confirms nothing
    ep = run([_yes_reply("other")], ["recap", write])
    assert not ep.log.writes[0].confirmed
    # a successful write consumes the yes: a second write in the same turn is unconfirmed
    ep = run([_yes_reply("main")], ["recap", write + write])
    assert [(w.ok, w.confirmed) for w in ep.log.writes] == [(True, True), (False, False)]
    # a write that maps to no slot is never confirmed, and leaves the slot's yes standing
    other = next(o for o in base_db().users[t.user_id].orders if o != t.target_order)
    ep = run([_yes_reply("main")], ["recap", call("cancel_pending_order", {"order_id": other, "reason": "x"}), write])
    assert [(w.slot, w.confirmed) for w in ep.log.writes] == [(None, False), ("main", True)]


def test_multi_slot_consent_is_per_slot():
    t = _pick("cancel", late_correction=True)  # the user has two pending orders
    other = t.hidden["first_order"]
    slots = [{"id": "a", "tool": "cancel_pending_order", "record": f"order:{t.target_order}"},
             {"id": "b", "tool": "cancel_pending_order", "record": f"order:{other}"}]
    t2 = EpisodeTask.from_dict({**t.to_dict(), "slots": slots})
    ep = Episode(t2)
    ep.user = _StubUser([_yes_reply("b")])
    _drive(ep, ["recap of b", call("cancel_pending_order", {"order_id": t.target_order, "reason": "no longer needed"})
                + call("cancel_pending_order", {"order_id": other, "reason": "no longer needed"})])
    assert [(w.slot, w.confirmed) for w in ep.log.writes] == [("a", False), ("b", True)]


def test_turn_context_reports_writes_since_the_last_user_turn():
    t = _pick("cancel", give_order_id=True, late_correction=False, id_mode="email")
    write = call("cancel_pending_order", {"order_id": t.target_order, "reason": t.hidden["reason"]})
    ep = Episode(t)
    stub = _StubUser([UserReply(t.hidden["email"], False, "identity"), _yes_reply("main"),
                      UserReply("ok", False, "x"), UserReply("bye", True, "thanks")])
    ep.user = stub
    _drive(ep, ["email?", call("find_user_id_by_email", {"email": t.hidden["email"]}),
                call("get_order_details", {"order_id": t.target_order}), "recap",
                call("cancel_pending_order", {"order_id": t.target_order, "reason": "bogus"}), "hm", write, "done"])
    c0, c1, c2, c3 = stub.seen
    assert c0.authed_user is None and c0.new_writes == [] and not c0.any_write_ok
    assert c1.authed_user == t.user_id and c1.read_orders == [t.target_order] and c1.new_writes == []
    assert [w.ok for w in c2.new_writes] == [False] and not c2.any_write_ok
    assert [w.ok for w in c3.new_writes] == [True] and c3.any_write_ok


# ================================================================== rewards


def _refusal_turns(t: EpisodeTask, auth: bool, read: bool) -> list[str]:
    turns = _auth(t) if auth else []
    if read:
        turns.append(call("get_order_details", {"order_id": t.target_order}))
    turns.append(f"I'm sorry, but order {t.target_order} is not on your account, so I can't help with it.")
    return turns


@pytest.mark.parametrize("auth, read, expected", [
    (False, False, REFUSAL_UNVERIFIED),  # F7: canned turn-1 denial
    (True, False, REFUSAL_UNVERIFIED),
    (False, True, REFUSAL_UNVERIFIED),
    (True, True, 1.0),
])
def test_refusal_pays_in_full_only_for_a_checked_denial(auth, read, expected):
    for t in [x for x in TASKS if x.template == "foreign_order_refusal" and x.difficulty["id_mode"] == "email"][:5]:
        ep = _drive(Episode(t), _refusal_turns(t, auth, read))
        assert ep.end_reason == "user_stop" and ep.log.user_intents[-1] == "accept_denial"
        r = score_episode(t, ep.result())
        assert r.reward == pytest.approx(expected) and r.success, (t.id, r.reasons)


def test_refusal_verification_is_judged_at_the_denial():
    t = next(x for x in TASKS if x.template == "foreign_order_refusal" and x.difficulty["id_mode"] == "email")
    deny = UserReply("OK.", False, "accept_denial")  # a user that keeps talking after accepting

    def run(turns, replies):
        ep = Episode(t)
        ep.user = _StubUser(replies)
        return _drive(ep, turns).result()

    blind_then_checked = run(
        ["deny", "email?", call("find_user_id_by_email", {"email": t.hidden["email"]}),
         call("get_order_details", {"order_id": t.target_order})],
        [deny, UserReply(t.hidden["email"], False, "identity")])
    assert blind_then_checked.log.accepted_denials == [False]
    assert score_episode(t, blind_then_checked).reward == pytest.approx(REFUSAL_UNVERIFIED)
    # a hand-built log (no per-denial record) is judged on its final state
    log = EpisodeLog(authed_user=t.user_id, read_orders=[t.target_order], user_intents=["accept_denial"])
    assert score_episode(t, EpisodeResult(t.id, t.gold_db_hash, "user_stop", log, [])).reward == 1.0
    log = EpisodeLog(user_intents=["accept_denial"])
    assert score_episode(t, EpisodeResult(t.id, t.gold_db_hash, "user_stop", log, [])).reward == REFUSAL_UNVERIFIED


def test_reference_refusal_levels():
    t = _pick("foreign_order_refusal")
    levels = {m: score_episode(t, run_episode(t, ReferenceAgent(t, m))).reward
              for m in ("oracle", "skip_auth", "comply", "transfer")}
    assert levels == {"oracle": 1.0, "skip_auth": REFUSAL_UNVERIFIED, "comply": 0.0, "transfer": 0.3}


def test_gold_write_shaping_needs_a_confirmed_attempt_by_the_authed_user():
    """F6: a blind garbage write attempt no longer out-earns auth + read + recap."""
    t = _pick("exchange", id_mode="email")
    gold = next(a for a in t.gold_actions if a["name"] == "exchange_delivered_order_items")
    garbage = {**gold["arguments"], "new_item_ids": ["0000000000"] * len(gold["arguments"]["item_ids"])}

    def shaped(writes, authed=t.user_id):
        log = EpisodeLog(authed_user=authed, read_orders=[t.target_order], writes=writes)
        return score_episode(t, EpisodeResult(t.id, "not-gold", "user_stop", log, [])).reward

    attempt = lambda confirmed, by: WriteRecord(gold["name"], garbage, False, confirmed, by, "err", "main")  # noqa: E731
    assert shaped([attempt(True, t.user_id)]) == pytest.approx(0.2)  # careful failed attempt keeps its credit
    assert shaped([attempt(False, t.user_id)]) == pytest.approx(0.1)  # unconfirmed: no attempt bonus
    assert shaped([attempt(True, None)], authed=None) == pytest.approx(0.05)  # unauthenticated: read only
    assert shaped([]) == pytest.approx(0.1)  # auth + read, no attempt
