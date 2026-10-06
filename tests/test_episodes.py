"""Multi-step episode tasks: generation, scripted user, runner, reward, decontam
hook and the audit loop.

Never reads tasks.json: every generation here passes an explicit (empty or
injected) `RealTaskExclusions`, and the decontam fingerprinting is exercised
on stub tasks shaped like tau2's `Task`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.utils import get_dict_hash

from tau_forge.decontam.real_tasks import RealTaskExclusions, fingerprint_tasks, load_real_task_exclusions
from tau_forge.envs.retail import RetailEnv
from tau_forge.episodes.audit import ChatTemplatePolicy, Generation, run_audit, summarize
from tau_forge.episodes.generate import KNOB_SPACE, generate_tasks, read_jsonl, user_lines, verify_task, write_jsonl
from tau_forge.episodes.reference_agents import ReferenceAgent, call
from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import GREETING, Episode, EpisodeLog, EpisodeResult, WriteRecord, run_episode
from tau_forge.episodes.task import ID_RE, STOP, TEMPLATES, base_db, base_db_hash
from tau_forge.episodes.user import BE_SPECIFIC, YES, ScriptedUser

REPO_ROOT = Path(__file__).resolve().parents[1]
EMPTY = RealTaskExclusions(frozenset(), frozenset())
N = 10

# (template, behaviour) -> the reward every such episode must get. Mirrors the
# design doc's table; "|corr" = a late-correction task, "|gc" = the user owns
# a gift card.
EXPECTED = {
    ("exchange", "oracle"): 1.0,
    ("exchange", "no_confirm"): 0.7,
    # wrote before the correction: wrong variant; the unconfirmed attempt earns no shaping
    ("exchange", "no_confirm|corr"): 0.1,
    ("exchange", "wrong_variant"): 0.2,
    ("exchange", "halluc_pm"): 0.2,
    ("exchange", "skip_auth"): 0.7,
    ("exchange", "transfer"): 0.0,
    ("cancel", "oracle"): 1.0,
    ("cancel", "no_confirm"): 0.7,
    ("cancel", "no_confirm|corr"): 0.0,  # cancelled the wrong order: shaping withdrawn
    ("cancel", "skip_auth"): 0.7,
    ("cancel", "transfer"): 0.0,
    ("return_fallback", "oracle"): 1.0,
    ("return_fallback", "no_confirm"): 0.7,
    ("return_fallback", "comply"): 0.2,  # the tool refuses a non-original method
    ("return_fallback", "giftcard_fallback|gc"): 0.2,
    ("return_fallback", "giftcard_fallback"): 1.0,  # no gift card: lands on the original
    ("return_fallback", "transfer"): 0.0,
    ("modify_payment", "oracle"): 1.0,
    ("modify_payment", "no_confirm"): 0.7,
    ("modify_payment", "halluc_pm"): 0.2,
    ("modify_payment", "skip_auth"): 0.7,
    ("modify_payment", "transfer"): 0.0,
    ("foreign_order_refusal", "oracle"): 1.0,
    ("foreign_order_refusal", "comply"): 0.0,
    ("foreign_order_refusal", "transfer"): 0.5,
}


def _behaviour(task, mode: str) -> str:
    if mode == "no_confirm" and task.difficulty.get("late_correction"):
        return "no_confirm|corr"
    if mode == "giftcard_fallback" and task.difficulty.get("has_gift_card"):
        return "giftcard_fallback|gc"
    return mode


def _modes(task) -> list[str]:
    modes = sorted({m.split("|")[0] for (t, m) in EXPECTED if t == task.template})
    # skip_auth looks the order up by the id the user gave; without one there is nothing to skip to.
    return [m for m in modes if m != "skip_auth" or task.difficulty.get("give_order_id")]


@pytest.fixture(scope="module")
def pristine_hash():
    return get_dict_hash(base_db().model_dump())


@pytest.fixture(scope="module")
def report(pristine_hash):
    return generate_tasks(N, 0, exclusions=EMPTY, log=lambda _: None)


@pytest.fixture(scope="module")
def tasks(report):
    # Force coverage of the knob branches the reward table depends on.
    extra = generate_tasks(
        2,
        11,
        exclusions=EMPTY,
        knob_overrides={
            "cancel": {"late_correction": True, "give_order_id": False},
            "exchange": {"late_correction": True, "n_items": 2, "give_order_id": False},
            "modify_payment": {"gift_card_short": True},
        },
        templates=("cancel", "exchange", "modify_payment"),
        log=lambda _: None,
    )
    return report.tasks + extra.tasks


@pytest.fixture(scope="module")
def episodes(tasks):
    """(task, behaviour, EpisodeResult, EpisodeReward) for every task x mode."""
    out = []
    for t in tasks:
        for mode in _modes(t):
            ep = run_episode(t, ReferenceAgent(t, mode))
            out.append((t, _behaviour(t, mode), ep, score_episode(t, ep)))
    return out


# ------------------------------------------------------------------ generation


def test_every_template_generates_n_verified_tasks(report):
    by_template = defaultdict(list)
    for t in report.tasks:
        by_template[t.template].append(t)
    assert set(by_template) == set(TEMPLATES)
    for template, ts in by_template.items():
        assert len(ts) == N, template
        for t in ts:
            v = verify_task(t)
            assert v.ok, (t.id, v.problems)
            assert v.gold_db_hash == t.gold_db_hash
            assert set(KNOB_SPACE[template]) <= set(t.difficulty)
    assert len({t.id for t in report.tasks}) == len(report.tasks)
    assert len({t.dedupe_key for t in report.tasks}) == len(report.tasks)


def test_generation_is_deterministic_by_seed():
    a = generate_tasks(3, 5, exclusions=EMPTY, log=lambda _: None)
    b = generate_tasks(3, 5, exclusions=EMPTY, log=lambda _: None)
    c = generate_tasks(3, 6, exclusions=EMPTY, log=lambda _: None)
    assert [t.to_dict() for t in a.tasks] == [t.to_dict() for t in b.tasks]
    assert [t.to_dict() for t in a.tasks] != [t.to_dict() for t in c.tasks]


def test_users_never_say_an_id_other_than_an_order_id(tasks):
    for t in tasks:
        for line in user_lines(t):
            assert all(i.startswith("#W") for i in ID_RE.findall(line)), (t.id, line)
        assert t.user_id not in t.opening


def test_writes_and_refusals_change_the_db_as_expected(tasks):
    base = base_db_hash()
    for t in tasks:
        assert (t.gold_db_hash == base) == t.expect_no_write, t.id


def test_knob_overrides_shape_the_task(tasks):
    hard = [t for t in tasks if t.template == "exchange" and t.difficulty["n_items"] == 2
            and t.difficulty["late_correction"] and not t.difficulty["give_order_id"]]
    assert hard
    for t in hard:
        assert "#W" not in t.opening  # the order must be found by scanning
        assert t.profile.get("correction")
        write = t.gold_actions[-1]
        assert len(write["arguments"]["item_ids"]) == 2
        # the scan reads orders from the profile until it reaches the target
        assert sum(a["name"] == "get_order_details" for a in t.gold_actions) >= 1


def test_verify_rejects_an_unreachable_id(tasks):
    t = next(t for t in tasks if t.template == "exchange")
    broken = type(t).from_dict(t.to_dict())
    broken.gold_actions = [a for a in broken.gold_actions if a["name"] != "get_product_details"]
    v = verify_task(broken)
    assert not v.ok and any("unrevealed" in p for p in v.problems)


def test_verify_rejects_a_leaked_id_in_a_user_line(tasks):
    t = next(t for t in tasks if t.template == "cancel")
    broken = type(t).from_dict(t.to_dict())
    broken.profile = {**broken.profile, "identity": [f"My user id is {t.user_id}."]}
    v = verify_task(broken)
    assert not v.ok and any("leaks" in p for p in v.problems)


def test_tasks_round_trip_through_jsonl(tmp_path, report):
    path = tmp_path / "eps.jsonl"
    write_jsonl(report.tasks, path)
    assert [t.to_dict() for t in read_jsonl(path)] == [t.to_dict() for t in report.tasks]


# -------------------------------------------------------------------- reward


def test_oracle_scores_one_on_every_task(episodes, tasks):
    oracle = [(t, ep, r) for t, b, ep, r in episodes if b == "oracle"]
    assert len(oracle) == len(tasks)
    for t, ep, r in oracle:
        assert r.reward == 1.0, (t.id, r.reasons, ep.end_reason)


def test_near_miss_agents_land_on_the_design_reward_levels(episodes):
    seen = set()
    for t, behaviour, ep, r in episodes:
        assert r.reward == pytest.approx(EXPECTED[(t.template, behaviour)]), (t.id, behaviour, r.reasons)
        seen.add((t.template, behaviour))
    assert seen == set(EXPECTED), set(EXPECTED) - seen
    assert {round(v, 3) for v in EXPECTED.values()} == {0.0, 0.1, 0.2, 0.5, 0.7, 1.0}


def test_gates_stack_on_success_and_never_go_negative(tasks):
    t = next(t for t in tasks if t.template == "cancel")
    write = t.gold_actions[-1]
    log = EpisodeLog(writes=[WriteRecord(write["name"], write["arguments"], True, confirmed=False, authed_user=None)])
    r = score_episode(t, EpisodeResult(t.id, t.gold_db_hash, "user_stop", log, []))
    assert r.success and r.reward == pytest.approx(0.4)
    assert set(r.gates) == {"no_confirmation", "no_authentication"}


def test_shaping_is_capped_and_withdrawn_after_touching_another_record(tasks):
    t = next(t for t in tasks if t.template == "exchange")
    gold = t.gold_actions[-1]
    attempt = WriteRecord(gold["name"], gold["arguments"], False, True, t.user_id)
    log = EpisodeLog(authed_user=t.user_id, read_orders=[t.target_order], writes=[attempt, attempt])
    r = score_episode(t, EpisodeResult(t.id, "not-the-gold-hash", "user_stop", log, []))
    assert not r.success and r.reward == pytest.approx(0.2)
    stray = WriteRecord("cancel_pending_order", {"order_id": "#W0000000", "reason": "no longer needed"}, True, True, t.user_id)
    log.writes.append(stray)
    r = score_episode(t, EpisodeResult(t.id, "not-the-gold-hash", "user_stop", log, []))
    assert r.reward == 0.0 and not r.shaping


# ---------------------------------------------------------------- scripted user


def test_scripted_user_is_deterministic(tasks):
    t = next(t for t in tasks if t.template == "exchange" and t.profile.get("correction"))
    script = [
        "Could you verify your identity with your email?",
        "Which order is this about? I need the order id.",
        "Which payment method would you like to use for the price difference?",
        f"To confirm, I will exchange items in order {t.target_order}. Shall I proceed?",
        f"Updated: exchange in order {t.target_order}. Shall I proceed?",
    ]
    a, b = ScriptedUser(t, seed=3), ScriptedUser(t, seed=3)
    ra = [a.reply(s) for s in script]
    rb = [b.reply(s) for s in script]
    assert [r.text for r in ra] == [r.text for r in rb]
    assert [r.intent for r in ra] == ["identity", "order", "payment", "correction", "yes"]
    assert ra[3].answered_recap and not ra[3].is_yes and ra[4].is_yes


def test_scripted_user_says_yes_only_to_a_recap_naming_the_target(tasks):
    t = next(t for t in tasks if t.template == "cancel" and not t.profile.get("correction"))
    u = ScriptedUser(t)
    vague = u.reply("Shall I proceed with the cancellation?")
    assert not vague.is_yes and vague.text in BE_SPECIFIC
    named = u.reply(f"I will cancel order {t.target_order}. Do you want me to proceed?")
    assert named.is_yes and named.text in YES


def test_scripted_user_stops_after_three_unrecognised_turns(tasks):
    u = ScriptedUser(tasks[0])
    replies = [u.reply("The weather is lovely today.") for _ in range(3)]
    assert [r.stop for r in replies] == [False, False, True]
    assert replies[-1].text.endswith(STOP)


def test_scripted_user_fallback_and_refusal(tasks):
    ret = next(t for t in tasks if t.template == "return_fallback")
    u = ScriptedUser(ret)
    deny = u.reply("I'm sorry, refunds can only go to the original payment method. How should I proceed?")
    assert deny.intent == "fallback" and not deny.stop
    foreign = next(t for t in tasks if t.template == "foreign_order_refusal")
    r = ScriptedUser(foreign).reply("I'm sorry, but I can't help with an order on someone else's account.")
    assert r.stop and r.intent == "accept_denial"
    done = ScriptedUser(ret).reply("All set!", write_succeeded=True)
    assert done.stop and done.intent == "thanks"


# ----------------------------------------------------------------------- runner


def test_runner_never_mutates_the_shared_db(episodes, pristine_hash):
    assert any(r.success for _, b, _, r in episodes if b != "oracle")  # writes really happened
    assert get_dict_hash(base_db().model_dump()) == pristine_hash
    assert get_dict_hash(RetailDB.load(RETAIL_DB_PATH).model_dump()) == pristine_hash


def test_copy_on_write_end_state_equals_a_full_deep_copy_replay(episodes):
    one_each = {}
    for e in episodes:  # one episode per (template, behaviour): every write path, 0.2 s each
        one_each.setdefault((e[0].template, e[1]), e)
    for t, behaviour, ep, _ in one_each.values():
        env = RetailEnv(db=base_db().model_copy(deep=True))
        for m in ep.messages:
            for tc in m.get("tool_calls") or []:
                env.execute(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        assert env.db_hash() == ep.final_db_hash, (t.id, behaviour)


def test_runner_message_shape_matches_grounding(episodes):
    t, _, ep, _ = next(e for e in episodes if e[1] == "oracle" and e[0].template == "exchange")
    assert ep.messages[0] == {"role": "assistant", "content": GREETING}
    assert ep.messages[1] == {"role": "user", "content": t.opening}
    calls = [i for i, m in enumerate(ep.messages) if m.get("tool_calls")]
    assert calls
    for i in calls:
        tc = ep.messages[i]["tool_calls"][0]
        assert tc["type"] == "function" and isinstance(tc["function"]["arguments"], str)
        res = ep.messages[i + 1]
        assert res["role"] == "tool" and res["tool_call_id"] == tc["id"] and res["name"] == tc["function"]["name"]


def test_runner_termination_rules(tasks):
    t = next(t for t in tasks if t.template == "cancel")
    sys_msg = {"role": "system", "content": "SYS"}

    ep = Episode(t, system_message=sys_msg)
    assert ep.messages[0] == sys_msg
    ep.step(call("transfer_to_human_agents", {"summary": "x"}))
    assert ep.done and ep.end_reason == "transfer" and ep.log.transfer

    ep = Episode(t)
    half = call("cancel_pending_order", {"order_id": t.target_order, "reason": "no longer needed"})[:-20]
    ep.step(half, "length")
    assert ep.end_reason == "truncated" and not ep.log.writes  # a half-written call is never executed

    ep = Episode(t)
    ep.step('<tool_call>{"name": oops</tool_call>')
    assert ep.log.n_malformed_calls == 1 and ep.messages[-1]["role"] == "user"

    ep = Episode(t, max_turns=4)
    while not ep.done:
        ep.step(call("get_user_details", {"user_id": t.user_id}))
    assert ep.end_reason == "max_turns" and ep.log.n_assistant_turns == 4

    ep = Episode(t)
    ep.step("", "context")
    assert ep.end_reason == "context_budget"
    with pytest.raises(RuntimeError):
        ep.step("hello")


# ---------------------------------------------------------------------- decontam


@dataclass
class _StubAction:
    name: str
    arguments: dict[str, Any]
    requestor: str = "assistant"


@dataclass
class _StubCriteria:
    actions: list[_StubAction]


@dataclass
class _StubTask:
    actions: list[_StubAction]
    initial_state: Optional[Any] = None
    evaluation_criteria: Any = field(init=False)

    def __post_init__(self) -> None:
        self.evaluation_criteria = _StubCriteria(self.actions)


def _as_stub(task) -> _StubTask:
    return _StubTask([_StubAction(a["name"], a["arguments"]) for a in task.gold_actions])


def test_fingerprint_matches_the_generator_gold_hash_and_collects_users(tasks):
    cancel = next(t for t in tasks if t.template == "cancel")
    fp = fingerprint_tasks([_as_stub(cancel)], base_db())
    assert fp.gold_db_hashes == {cancel.gold_db_hash}
    assert cancel.user_id in fp.user_ids
    assert repr(fp) == "RealTaskExclusions(<1 user ids>, <1 gold hashes>)"


def test_generator_rejects_real_task_users_and_gold_hashes():
    first = generate_tasks(4, 3, exclusions=EMPTY, templates=("cancel",), log=lambda _: None).tasks
    banned_users = RealTaskExclusions(frozenset({first[0].user_id}), frozenset())
    lines: list[str] = []
    again = generate_tasks(4, 3, exclusions=banned_users, templates=("cancel",), log=lines.append)
    assert first[0].user_id not in {t.user_id for t in again.tasks}
    assert again.stats["cancel"]["decontam_user"] >= 1
    # only counts are logged, never an id
    assert lines and first[0].user_id not in lines[0] and "decontam_user=" in lines[0]

    other_user = next(t for t in first if t.user_id != first[0].user_id)
    banned_hash = RealTaskExclusions(frozenset(), frozenset({other_user.gold_db_hash, base_db_hash()}))
    again = generate_tasks(4, 3, exclusions=banned_hash, templates=("cancel", "foreign_order_refusal"), log=lambda _: None)
    assert other_user.gold_db_hash not in {t.gold_db_hash for t in again.tasks}
    assert again.stats["cancel"]["decontam_gold_hash"] >= 1
    # the untouched-db hash identifies nothing, so refusal tasks are not rejected by it
    assert again.stats["foreign_order_refusal"]["decontam_gold_hash"] == 0
    assert again.stats["foreign_order_refusal"]["accepted"] == 4


def test_decontam_fails_loudly_when_tasks_cannot_be_loaded(monkeypatch):
    import tau2.domains.retail.environment as retail_env

    def boom(*_a, **_k):
        raise FileNotFoundError("tasks.json")

    monkeypatch.setattr(retail_env, "get_tasks", boom)
    load_real_task_exclusions.cache_clear()
    with pytest.raises(RuntimeError, match="FileNotFoundError"):
        load_real_task_exclusions()
    with pytest.raises(RuntimeError):
        generate_tasks(1, 0, templates=("cancel",), log=lambda _: None)  # no exclusions -> must load, must fail
    load_real_task_exclusions.cache_clear()


# ------------------------------------------------------------------------ audit


def _scripted_generator(modes_by_sample: list[str]):
    agents: dict = {}

    def generate(requests):
        out = []
        for r in requests:
            agent = agents.setdefault(r.key, ReferenceAgent(r.task, modes_by_sample[r.key[1]]))
            out.append(Generation(agent(r.messages)))
        return out

    return generate


def test_audit_loop_with_a_fake_generator(tasks):
    picked = [t for t in tasks if t.template == "cancel" and not t.difficulty["late_correction"]][:2]
    picked += [t for t in tasks if t.template == "foreign_order_refusal"][:2]
    result = run_audit(picked, _scripted_generator(["oracle", "no_confirm", "transfer"]), n_samples=3)
    per_task = {r["id"]: r for r in result["per_task"]}
    for t in picked:
        rec = per_task[t.id]
        if t.template == "cancel":
            assert rec["rewards"] == [1.0, 0.7, 0.0] and rec["successes"] == [True, True, False]
        else:  # no_confirm complies with a foreign order here
            assert rec["rewards"] == [1.0, 0.0, 0.5]
    s = result["summary"]
    assert s["overall"]["n_tasks"] == 4 and s["overall"]["n_episodes"] == 12
    assert s["overall"]["effective_variance_fraction"] == 1.0 and s["overall"]["flat_fraction"] == 0.0
    assert set(s["per_template"]) == {"cancel", "foreign_order_refusal"}
    assert "id_mode" in s["per_knob"]["cancel"]


def test_summary_effective_variance_threshold():
    recs = [
        {"template": "x", "difficulty": {"k": True}, "rewards": [0.0, 0.05], "successes": [False, False], "end_reasons": ["a", "a"]},
        {"template": "x", "difficulty": {"k": False}, "rewards": [0.0, 0.1], "successes": [False, False], "end_reasons": ["a", "a"]},
    ]
    s = summarize(recs)
    assert s["overall"]["effective_variance_fraction"] == 0.5  # std 0.025 < 0.05 <= std 0.05
    assert s["per_knob"]["x"]["k"]["False"]["effective_variance_fraction"] == 1.0


class _FakeTokenizer:
    def __init__(self):
        self.rendered: list[tuple[list[dict], int]] = []

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        self.rendered.append((messages, len(tools)))
        return " ".join(f"<{m['role']}> {m.get('content') or ''}" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_chat_template_policy_batches_and_enforces_the_context_budget(tasks):
    picked = [t for t in tasks if t.template == "foreign_order_refusal"][:3]
    tok = _FakeTokenizer()
    batches: list[int] = []

    def engine(prompts):
        batches.append(len(prompts))
        return [(call("transfer_to_human_agents", {"summary": "s"}), "stop") for _ in prompts]

    sys_msg = {"role": "system", "content": "SYSTEM PROMPT"}
    policy = ChatTemplatePolicy(engine, tok.apply_chat_template, RetailEnv().all_openai_schemas(),
                                lambda s: len(tok(s)["input_ids"]), max_new_tokens=10, max_model_len=10_000)
    result = run_audit(picked, policy, n_samples=2, system_message=sys_msg)
    assert batches == [6]  # one engine call for all 3 tasks x 2 samples
    assert all(n_tools == 16 and msgs[0] == sys_msg for msgs, n_tools in tok.rendered)
    assert all(r["rewards"] == [0.5, 0.5] for r in result["per_task"])

    tight = ChatTemplatePolicy(engine, tok.apply_chat_template, [], lambda s: len(s), max_new_tokens=10, max_model_len=20)
    result = run_audit(picked[:1], tight, n_samples=2)
    assert result["per_task"][0]["end_reasons"] == ["context_budget", "context_budget"]
    assert tight.n_context_overflows == 2


def _load_script():
    spec = importlib.util.spec_from_file_location("episode_audit", REPO_ROOT / "scripts" / "episode_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_episode_audit_script_end_to_end_with_injected_model(tmp_path, report):
    path = tmp_path / "eps.jsonl"
    write_jsonl([t for t in report.tasks if t.template in ("cancel", "foreign_order_refusal")][:4], path)
    out = tmp_path / "audit.json"
    script = _load_script()

    def engine(prompts):
        return [(call("transfer_to_human_agents", {"summary": "s"}), "stop") for _ in prompts]

    script.main(["--tasks", str(path), "--samples-per-task", "2", "--output", str(out)],
                tokenizer=_FakeTokenizer(), engine=engine)
    data = json.loads(out.read_text())
    assert len(data["per_task"]) == 4 and all(len(r["rewards"]) == 2 for r in data["per_task"])
    assert data["config"]["n_tools"] == 16 and data["config"]["context_overflows"] == 0
    assert data["summary"]["overall"]["end_reasons"] == {"transfer": 8}


def test_episode_audit_script_dry_run_with_a_scripted_policy(tmp_path, report):
    path = tmp_path / "eps.jsonl"
    write_jsonl(report.tasks[:6], path)
    out = tmp_path / "audit.json"
    _load_script().main(["--tasks", str(path), "--samples-per-task", "2", "--fake-policy", "oracle", "--output", str(out)])
    data = json.loads(out.read_text())
    assert data["summary"]["overall"]["success_rate"] == 1.0


def test_episode_audit_script_imports_no_gpu_stack():
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('ea', {str(REPO_ROOT / 'scripts' / 'episode_audit.py')!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "bad = [x for x in ('torch', 'vllm', 'transformers') if x in sys.modules];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


# ------------------------------------------------------- review fixes (round 2)


def test_tool_exceptions_become_tool_errors_and_the_episode_continues(tasks):
    """tau2's calculate / find_user_id_by_email raise SyntaxError,
    ZeroDivisionError and AttributeError on malformed input; uncaught, one
    such sample used to abort the whole batched audit."""
    t = next(t for t in tasks if t.template == "cancel")
    ep = Episode(t)
    bad_calls = [
        ("calculate", {"expression": "(54.04 - 49.99"}, "SyntaxError"),
        ("calculate", {"expression": "2 3"}, "SyntaxError"),
        ("calculate", {"expression": "1/0"}, "ZeroDivisionError"),
        ("find_user_id_by_email", {"email": None}, "AttributeError"),
    ]
    for name, args, err in bad_calls:
        ep.step(call(name, args))
        assert not ep.done
        assert ep.messages[-1]["role"] == "tool" and ep.messages[-1]["content"].startswith(f"Error: {err}: ")
    assert ep.log.n_calls == 4 and ep.log.authed_user is None


def test_audit_survives_a_malformed_call_and_a_crashing_episode(tasks, monkeypatch):
    picked = [t for t in tasks if t.template == "cancel"][:3]

    def generate(requests):
        return [
            Generation(call("calculate", {"expression": "(54.04 - 49.99"}) if r.key == (0, 0)
                       else "Could you verify your identity with your email?")
            for r in requests
        ]

    done: list[str] = []
    result = run_audit(picked, generate, n_samples=2, on_task_done=lambda rec: done.append(rec["id"]))
    assert sorted(done) == sorted(t.id for t in picked)  # every task checkpointed once
    assert all(len(r["rewards"]) == 2 for r in result["per_task"])

    # An exception escaping `step` (a harness bug) ends only that episode.
    real_step = Episode.step

    def flaky(self, completion, finish_reason="stop"):
        if self.task.id == picked[1].id and self.log.n_assistant_turns == 1:
            raise KeyError("boom")
        return real_step(self, completion, finish_reason)

    monkeypatch.setattr(Episode, "step", flaky)
    result = run_audit(picked, generate, n_samples=2)
    rec = next(r for r in result["per_task"] if r["id"] == picked[1].id)
    assert rec["end_reasons"] == ["runner_error", "runner_error"]
    assert rec["runner_errors"] == ["KeyError: 'boom'"] * 2
    assert all("runner_error" not in r["end_reasons"] for r in result["per_task"] if r["id"] != picked[1].id)


def test_episode_audit_script_checkpoints_each_task(tmp_path, report):
    path = tmp_path / "eps.jsonl"
    write_jsonl(report.tasks[:3], path)
    out = tmp_path / "audit.json"
    _load_script().main(["--tasks", str(path), "--samples-per-task", "2", "--fake-policy", "oracle", "--output", str(out)])
    lines = (tmp_path / "audit.json.partial.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["id"] for x in lines) == sorted(t.id for t in report.tasks[:3])


class _EarlyConfirmAgent:
    """no_confirm, but its first turn asks to 'confirm your email' while
    naming the order from the opening -- a very common instruct-model line."""

    def __init__(self, task):
        self.t, self.inner, self.first = task, ReferenceAgent(task, "no_confirm"), True

    def __call__(self, messages):
        if self.first:
            self.first = False
            return f"I can help with order {self.t.target_order}. Could you please confirm your email address first?"
        return self.inner(messages)


def test_a_confirm_request_for_information_is_not_a_recap(tasks):
    cancel = [t for t in tasks if t.template == "cancel" and t.difficulty["give_order_id"]
              and not t.difficulty["late_correction"]]
    assert cancel
    t = cancel[0]
    u = ScriptedUser(t)
    r = u.reply(f"I can help with order {t.target_order}. Could you please confirm your email address first?")
    assert r.intent == "identity" and not r.is_yes
    r = u.reply(f"Could you confirm your email so I can cancel order {t.target_order}?")
    assert r.intent == "identity" and not r.is_yes
    r = u.reply(f"Can you confirm order {t.target_order} is the right one?")  # target, but no action
    assert not r.is_yes
    r = u.reply(f"I will cancel order {t.target_order}. Would you like me to proceed?")
    assert r.is_yes
    # End to end: the early "confirm" no longer buys the write a yes.
    for t in [t for t in tasks if t.difficulty.get("give_order_id") and not t.difficulty.get("late_correction")
              and t.template in ("cancel", "exchange", "return_fallback", "modify_payment")]:
        ep = run_episode(t, _EarlyConfirmAgent(t))
        r = score_episode(t, ep)
        assert r.reward == pytest.approx(0.7) and "no_confirmation" in r.gates, (t.id, r.reasons)


class _SelfAuthAgent(ReferenceAgent):
    """skip_auth, plus 'authenticating' with the email it read from the db."""

    def __init__(self, task):
        super().__init__(task, "skip_auth")
        self.done_auth = False

    def __call__(self, messages):
        d = self.user_details(messages)
        if d is not None and not self.done_auth:
            self.done_auth = True
            return call("find_user_id_by_email", {"email": d["email"]})
        return super().__call__(messages)


def test_self_authentication_with_db_values_does_not_pass_the_auth_gate(tasks):
    picked = [t for t in tasks if t.template != "foreign_order_refusal" and t.difficulty.get("give_order_id")
              and not t.difficulty.get("identity_upfront")]
    assert picked
    for t in picked:
        ep = run_episode(t, _SelfAuthAgent(t))
        assert "identity" not in ep.log.user_intents  # the user was never asked
        assert ep.log.unearned_auths == [t.user_id] and ep.log.authed_user is None
        r = score_episode(t, ep)
        assert r.reward == pytest.approx(0.7) and "no_authentication" in r.gates, (t.id, r.reasons)


def test_earned_authentication_matches_what_the_user_said(tasks):
    t = next(t for t in tasks if t.difficulty.get("id_mode") == "email" and t.template != "foreign_order_refusal")
    ep = Episode(t)
    ep.step(call("find_user_id_by_email", {"email": t.hidden["email"]}))
    earned = t.difficulty["identity_upfront"]
    assert (ep.log.authed_user == t.user_id) == earned
    ep.step("Could you please verify your identity with your email address?")
    ep.step(call("find_user_id_by_email", {"email": t.hidden["email"].upper()}))  # case-insensitive
    assert ep.log.authed_user == t.user_id


@pytest.mark.parametrize(
    "policy, finish",
    [
        ("asdf qwerty", "stop"),
        ("", "stop"),
        ('<tool_call>{not json</tool_call>', "stop"),
        ("Sorry, I can't find your account. What is your email?", "stop"),
        ("I'll look into order", "length"),
        ("", "context"),
    ],
)
def test_refusal_without_a_denial_scores_below_a_transfer(tasks, policy, finish):
    refusals = [t for t in tasks if t.template == "foreign_order_refusal"]
    for t in refusals[:5]:
        ep = Episode(t)
        while not ep.done:
            ep.step(policy, finish)
        r = score_episode(t, ep.result())
        assert r.reward == pytest.approx(0.1) and not r.success, (t.id, ep.end_reason, r.reasons)
        assert r.reward < EXPECTED[("foreign_order_refusal", "transfer")]


@pytest.mark.parametrize(
    "denial",
    [
        "I'm sorry, but order {o} isn't associated with your account, so I can't make changes to it.",
        "That order doesn't belong to you -- only the account holder can request this.",
        "Unfortunately, order {o} is not on your account. Your roommate will need to contact us directly.",
    ],
)
def test_refusal_denial_phrasings_are_accepted(tasks, denial):
    t = next(t for t in tasks if t.template == "foreign_order_refusal")
    r = ScriptedUser(t).reply(denial.format(o=t.target_order))
    assert r.intent == "accept_denial" and r.stop


def test_restate_after_a_correction_or_fallback_names_the_gold_target(tasks):
    exch = next(t for t in tasks if t.template == "exchange" and t.hidden.get("correction"))
    u = ScriptedUser(exch)
    assert u.reply(f"Shall I proceed with the exchange in order {exch.target_order}?").intent == "correction"
    line = u.reply("Hmm.").text
    name, diff = exch.hidden["correction"]
    (key, val), = diff.items()
    old_diff = next(d for n, d in exch.hidden["targets"] if n == name)
    assert name in line and val in line and next(iter(old_diff.values())) not in line

    ret = next(t for t in tasks if t.template == "return_fallback")
    u = ScriptedUser(ret)
    assert u.reply("I'm sorry, that is not possible.").intent == "fallback"
    line = u.reply("Hmm.").text
    assert "original payment method" in line and ret.hidden["bad_pm_phrase"] not in line

    mp = next(t for t in tasks if t.template == "modify_payment" and t.profile.get("fallback"))
    u = ScriptedUser(mp)
    assert u.reply("I'm sorry, that is not possible.").intent == "fallback"
    line = u.reply("Hmm.").text
    assert mp.hidden["target_pm_phrase"] in line and mp.hidden["asked_pm_phrase"] not in line


@pytest.mark.parametrize(
    "template, line",
    [
        ("modify_payment", "Unfortunately your gift card balance of $40.00 won't cover the order total of $120.50."),
        ("modify_payment", "Your gift card has a balance of $40.00, but the order total is $120.50. "
                           "Which payment method would you like to use?"),
        ("modify_payment", "Would you like to use another payment method instead?"),
        ("return_fallback", "Our policy is that refunds go back to the payment method used for the purchase, "
                            "or to a gift card. Which would you prefer?"),
        ("return_fallback", "Which would you prefer for the refund: the payment method you used for the purchase, "
                            "or a gift card?"),
        ("return_fallback", "Refunds can only go to the original payment method or a gift card."),
    ],
)
def test_fallback_fires_on_denial_phrasing_variants(tasks, template, line):
    t = next(t for t in tasks if t.template == template and t.profile.get("fallback"))
    r = ScriptedUser(t).reply(line)
    assert r.intent == "fallback" and r.text in t.profile["fallback"]


def test_payment_question_before_any_constraint_still_gets_the_original_ask(tasks):
    """The fallback is a reaction to the policy, not to any payment question:
    an agent that just asks "which method?" hears the forbidden method again."""
    t = next(t for t in tasks if t.template == "return_fallback")
    r = ScriptedUser(t).reply("Which payment method should the refund go to?")
    assert r.intent == "payment" and t.hidden["bad_pm_phrase"] in r.text


# --- exchange wording: unambiguous correction and target variant --------------

def _exchange_tasks(tasks):
    return [t for t in tasks if t.template == "exchange"]


def test_exchange_opening_and_restate_pin_the_variant(tasks):
    for t in _exchange_tasks(tasks):
        assert "everything else the same" in t.opening
        assert "everything else the same" in t.profile["restate"]
        if t.profile.get("restate_after_correction"):
            assert "everything else the same" in t.profile["restate_after_correction"]


def test_exchange_correction_replaces_first_change(tasks):
    corrected = [t for t in _exchange_tasks(tasks) if t.profile.get("correction")]
    assert corrected
    for t in corrected:
        for line in t.profile["correction"]:
            assert "instead" in line
            assert "same as" in line and "now" in line
            assert "Everything else is right" not in line


def test_exchange_gold_variant_differs_in_exactly_the_requested_option(tasks):
    db = base_db()
    checked = 0
    for t in _exchange_tasks(tasks):
        order = db.orders[t.target_order]
        gold = next(a for a in t.gold_actions if a["name"] == "exchange_delivered_order_items")["arguments"]
        asked = {n: d for n, d in t.hidden["targets"]}
        if t.hidden["correction"]:
            asked[t.hidden["correction"][0]] = t.hidden["correction"][1]
        for old_id, new_id in zip(gold["item_ids"], gold["new_item_ids"]):
            old = next(i for i in order.items if i.item_id == old_id)
            new = db.products[old.product_id].variants[new_id]
            diff = {k: v for k, v in new.options.items() if old.options.get(k) != v}
            assert diff == asked[old.name]
            checked += 1
    assert checked


# ------------------------------------------- scripted-user audit fixes (1-3)
_RETURN_PROCEED = (
    "The order #W4017490 is delivered, so we can proceed with the return of the Luggage Set. Here are the details:\n\n"
    "- **Item to be returned**: Luggage Set (2-piece, red, softshell)\n- **Item ID**: 7160999700\n\n"
    "The refund will be processed to the original payment method, which is the gift card.\n\n"
    "Could you please confirm if you would like to proceed with this return? "
    '(Please reply with "yes" to confirm.)'
)
_OWNERSHIP_DENIAL = (
    "I understand you'd like to return the Smart Watch for a family member. However, I can only process returns "
    "for the account associated with your user ID. Since the order is under a different user ID (yara_li_8961), "
    "I cannot proceed with this request."
)


def test_refusal_task_does_not_accept_a_return_offer_as_a_denial(tasks):
    foreign = next(t for t in tasks if t.template == "foreign_order_refusal")
    r = ScriptedUser(foreign).reply(_RETURN_PROCEED)
    assert not r.stop and r.intent != "accept_denial"
    for line in (
        "Sure, the refund goes to the original payment method. Please confirm with a yes to proceed.",
        "That exceeds the balance, but we can proceed. Shall I go ahead?",
    ):
        assert ScriptedUser(foreign).reply(line).intent != "accept_denial"


def test_refusal_task_accepts_an_ownership_denial(tasks):
    foreign = next(t for t in tasks if t.template == "foreign_order_refusal")
    for line in (
        _OWNERSHIP_DENIAL,
        "Sorry, that order does not belong to you.",
        "That order belongs to another customer, so I can't help with it. Is there anything else?",
        "I can only help with orders on your own account.",
    ):
        r = ScriptedUser(foreign).reply(line)
        assert r.stop and r.intent == "accept_denial", line


def test_refusal_reward_is_not_paid_to_an_agent_that_offers_the_return(tasks):
    foreign = next(t for t in tasks if t.template == "foreign_order_refusal")
    u = ScriptedUser(foreign)
    assert u.reply(_RETURN_PROCEED).intent != "accept_denial"
    assert ScriptedUser(foreign).reply(_OWNERSHIP_DENIAL).intent == "accept_denial"


_IMPERATIVE_RECAP = (
    "I've found your order {o}, which is currently pending. Here are the details: ...\n\n"
    'Please confirm with a "yes" if you would like me to proceed with the cancellation.'
)


def test_imperative_confirmation_request_counts_as_a_recap(tasks):
    t = next(t for t in tasks if t.template == "cancel" and not t.profile.get("correction"))
    r = ScriptedUser(t).reply(_IMPERATIVE_RECAP.format(o=t.target_order))
    assert r.is_yes
    # still needs the target and the action, and no information ask
    assert not ScriptedUser(t).reply('Please confirm with a "yes" if you would like me to proceed.').is_yes
    assert not ScriptedUser(t).reply(f"Please reply yes to go ahead with order {t.target_order}.").is_yes
    assert not ScriptedUser(t).reply(
        f"To cancel order {t.target_order}, please confirm your email address."
    ).is_yes


def test_write_after_imperative_recap_is_confirmed_and_without_it_is_docked(tasks):
    t = next(t for t in tasks if t.template == "cancel" and not t.profile.get("correction"))

    class Imperative(ReferenceAgent):
        def recap(self, summary: str) -> str:
            return f'Please confirm with a "yes" if you would like me to {summary}.'

    res = run_episode(t, Imperative(t, "oracle"))
    assert score_episode(t, res).reward == 1.0
    res = run_episode(t, ReferenceAgent(t, "no_confirm"))
    assert score_episode(t, res).reward < 1.0


def test_statements_do_not_trigger_identity_or_order_answers(tasks):
    t = next(t for t in tasks if t.template == "cancel")
    for line in (
        "I have already authenticated your identity using your email.",
        "The order with the Bluetooth Speaker is #W7538230.",
    ):
        assert ScriptedUser(t).reply(line).intent == "unrecognised"
    for line in (
        "Could you provide your email?",
        "Please provide your email address.",
        "I need your email to verify your identity.",
    ):
        assert ScriptedUser(t).reply(line).intent == "identity"
    for line in ("Which order would you like to cancel?", "Please provide your order id."):
        assert ScriptedUser(t).reply(line).intent == "order"


def test_repeated_identical_information_request_ends_the_episode(tasks):
    t = next(t for t in tasks if t.template == "cancel")
    u = ScriptedUser(t)
    replies = [u.reply("Could you provide your email?") for _ in range(3)]
    assert [r.stop for r in replies] == [False, False, True]
    assert replies[-1].intent == "give_up" and replies[-1].text.endswith(STOP)
    # a different intent in between resets the streak
    u = ScriptedUser(t)
    seq = ["Could you provide your email?", "Which order is it?", "Could you provide your email?",
           "Which order is it?"]
    assert not any(u.reply(x).stop for x in seq)
    # a stuck agent now ends in a handful of turns, not max_turns
    res = run_episode(t, lambda m: "Could you provide your email?")
    assert res.end_reason != "max_turns" and len(res.messages) <= 12


def test_ownership_denial_with_transfer_offer_is_accepted():
    """A refusal that offers a transfer is still a refusal, not an offer to proceed."""
    task = next(t for t in generate_tasks(4, 0, templates=["foreign_order_refusal"]).tasks)
    user = ScriptedUser(task)
    r = user.reply(
        "I'm sorry, but this order belongs to a different account, so I can't cancel it. "
        "Would you like me to transfer you to a human agent?"
    )
    assert r.intent == "accept_denial"


@pytest.mark.parametrize(
    "line",
    [
        "For privacy and security reasons, I can only assist the account holder. Your roommate will need to "
        "contact us directly to cancel her order.",
        "Unfortunately I can't help with that request, since the order isn't yours. The account owner has to "
        "request the cancellation.",
        "I'm sorry, but that's not your order, so I can't make changes to it.",
    ],
)
def test_account_holder_and_not_yours_denials_are_accepted(tasks, line):
    foreign = next(t for t in tasks if t.template == "foreign_order_refusal")
    assert ScriptedUser(foreign).reply(line).intent == "accept_denial"


def test_exchange_recap_may_say_change_and_an_all_items_question_is_not_a_recap(tasks):
    t = next(t for t in tasks if t.template == "exchange" and not t.profile.get("correction"))
    name = t.profile["recap_keys"][1]
    r = ScriptedUser(t).reply(
        f"I'll change the {name} in order {t.target_order} to the new option, with the difference going to "
        "your card. Shall I proceed?"
    )
    assert r.is_yes
    r = ScriptedUser(t).reply(
        f"Before I proceed with the exchange for order {t.target_order}, is that all the items you want to exchange?"
    )
    assert r.intent == "all_items" and not r.is_yes
