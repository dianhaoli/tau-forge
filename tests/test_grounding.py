"""Grounding inserts the lookups prior_turns narrate as real tool turns."""

import json

import pytest

from tau_forge.envs.retail import RetailEnv
from tau_forge.train.dataset import build_examples, load_scenarios
from tau_forge.train.grounding import grounded_prior_turns, plan_lookups

ID_KEYS = {"order_id", "user_id", "item_ids", "new_item_ids", "product_id", "payment_method_id", "item_id"}


@pytest.fixture(scope="module")
def env():
    return RetailEnv()


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="module")
def grounded(env, scenarios):
    return {s["id"]: grounded_prior_turns(s, env) for s in scenarios}


def _visible(messages):
    return " ".join((m.get("content") or "") + json.dumps(m.get("tool_calls") or "") for m in messages)


def _unreachable(scenario, visible):
    calls = scenario.get("expected_tool_calls") or []
    if not calls:
        return []
    out = []
    for k, v in calls[0].get("arguments", {}).items():
        if k not in ID_KEYS:
            continue
        for x in v if isinstance(v, list) else [v]:
            if str(x) not in visible and str(x).lstrip("#") not in visible:
                out.append((k, x))
    return out


def test_grounding_never_mutates_the_db(env, scenarios, grounded):
    before = env.db_hash()
    for s in scenarios[:50]:
        grounded_prior_turns(s, env)
    assert env.db_hash() == before


def test_grounded_turns_are_well_formed_tool_exchanges(grounded):
    for sid, messages in grounded.items():
        for i, m in enumerate(messages):
            if m.get("tool_calls"):
                (call,) = m["tool_calls"]
                assert m["role"] == "assistant" and m["content"] == ""
                assert isinstance(call["function"]["arguments"], str)  # as tau2/litellm send it
                json.loads(call["function"]["arguments"])
                nxt = messages[i + 1]
                assert nxt["role"] == "tool" and nxt["tool_call_id"] == call["id"], sid
            if m["role"] == "tool":
                assert messages[i - 1].get("tool_calls"), sid


def test_original_prose_turns_survive_in_order(scenarios, grounded):
    for s in scenarios:
        prose = [(m["role"], m["content"]) for m in grounded[s["id"]] if m["role"] != "tool" and not m.get("tool_calls")]
        assert prose == [(t["role"], t["content"]) for t in s.get("prior_turns", [])]


def test_grounding_never_pre_answers_the_gold_call(scenarios, grounded):
    for s in scenarios:
        calls = s.get("expected_tool_calls") or []
        if not calls:
            continue
        gold = (calls[0]["name"], json.dumps(calls[0]["arguments"], sort_keys=True))
        for m in grounded[s["id"]]:
            for c in m.get("tool_calls") or []:
                shown = (c["function"]["name"], json.dumps(json.loads(c["function"]["arguments"]), sort_keys=True))
                assert shown != gold, s["id"]


def test_grounding_only_ever_shows_one_users_records(env, scenarios, grounded):
    """One user per conversation (policy): never leak another account's data."""
    for s in scenarios:
        users = set()
        for m in grounded[s["id"]]:
            for c in m.get("tool_calls") or []:
                args = json.loads(c["function"]["arguments"])
                if "user_id" in args:
                    users.add(args["user_id"])
                if "order_id" in args:
                    users.add(env.db.orders[args["order_id"]].user_id)
        assert len(users) <= 1, (s["id"], users)


def test_no_lookups_without_an_authentication_signal(env):
    scenario = {
        "prior_turns": [
            {"role": "user", "content": "Hi, I want to cancel order #W7736708."},
            {"role": "assistant", "content": "Sure -- first, can you give me your email or name and zip code?"},
        ],
        "user_message": "Why do you need that?",
        "expected_tool_calls": [],
    }
    assert plan_lookups(scenario, env.db) == {}


def test_lookups_named_only_in_the_final_message_are_left_to_the_policy(env):
    user_id, user = next(iter(env.db.users.items()))
    scenario = {
        "prior_turns": [
            {"role": "user", "content": f"Hi, my email is {user.email}."},
            {"role": "assistant", "content": "Thanks, I've found your account. What can I do for you?"},
        ],
        "user_message": f"What's the status of order {user.orders[0]}?",
        "expected_tool_calls": [{"name": "get_order_details", "arguments": {"order_id": user.orders[0]}}],
    }
    plan = plan_lookups(scenario, env.db)
    assert plan == {1: [("find_user_id_by_email", {"email": user.email}), ("get_user_details", {"user_id": user_id})]}


def test_grounding_makes_narrated_ids_reachable(env, scenarios, grounded):
    """The motivating bug: ids a narrated lookup would have returned were
    missing from the prompt. Pin a known case, and the corpus-wide drop."""
    by_id = {s["id"]: s for s in scenarios}
    sid = "happy_path__apparel_footwear_exchanges__002"
    s = by_id[sid]
    assert _unreachable(s, _visible(s["prior_turns"] + [{"content": s["user_message"]}]))
    assert not _unreachable(s, _visible(grounded[sid] + [{"content": s["user_message"]}]))

    def n_unreachable(view):
        return sum(
            bool(_unreachable(s, _visible(view(s) + [{"content": s["user_message"]}])))
            for s in scenarios
            if (s.get("expected_tool_calls") or []) and env.tool_mutates_state(s["expected_tool_calls"][0]["name"])
        )

    prose, ground = n_unreachable(lambda s: s["prior_turns"]), n_unreachable(lambda s: grounded[s["id"]])
    assert prose >= 100 and ground <= 0.6 * prose, (prose, ground)


def test_build_examples_grounds_by_default_and_can_opt_out():
    grounded = build_examples()
    plain = build_examples(ground_lookups=False)
    assert any(m["role"] == "tool" for ex in grounded for m in ex.prompt_messages)
    assert not any(m["role"] == "tool" for ex in plain for m in ex.prompt_messages)
    assert [e.id for e in grounded] == [e.id for e in plain]
