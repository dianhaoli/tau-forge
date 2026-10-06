"""The episode loop: policy <-> scripted user <-> a live, private RetailEnv.

The policy is any `callable(messages) -> assistant text`. Its text is parsed
with `completion_parsing.parse_all_completion`, a mirror of what eval does
(vLLM's hermes tool parser, then tau2 executing every call it returns): every
`<tool_call>{json}</tool_call>` block is a call, and a turn with any block
hermes rejects is text. The calls of one turn are recorded as tau2 records
them -- ONE assistant message carrying N `tool_calls` (content = the text
before the first block), followed by N `role: "tool"` results -- and executed
in order, each with its own bookkeeping. That is the exact shape
`tau_forge.train.grounding` renders and tau2 feeds its agent at eval time.
Every assistant message also keeps the sampled completion verbatim under
`"raw"`: re-rendering `content` + `tool_calls` with the chat template does not
reproduce non-canonical samples (trailing text, compact JSON), so a trainer
must take the sampled tokens from there. A text turn goes to the scripted
user, whose reply is appended as a user message.

The conversation starts the way tau2's orchestrator starts every task: an
assistant greeting ("Hi! How can I help you today?", tau2's
DEFAULT_FIRST_AGENT_MESSAGE), then the task's opening user message.

Consent. Each write the task expects is a slot (`EpisodeTask.slots`). The
runner keeps the set of slots the user has said yes to: a write is
`confirmed` iff the slot it maps to (tool + record) holds a yes at call time;
a successful write consumes that yes; a failed one suspends it until the user
confirms that slot again or gives a fallback; a reply with `revokes` (the
plan changed) clears every slot, suspended ones included. A write that maps
to no slot is never confirmed.

Termination: user STOP (done, gave up after 3 unrecognised turns, or accepted
a denial); a `transfer_to_human_agents` call (after the rest of its turn's
calls ran, as tau2 runs them all); `max_turns` assistant messages (30, the
design's budget) or `max_calls` tool calls, per task when the task sets them
(both checked after a whole turn, so a multi-call turn may overshoot
`max_calls`); a truncated completion (`finish_reason="length"`) that hermes
reads as text, which ends the episode as "truncated" for the trainer to mask
-- a truncated completion that still parses into calls is executed, as eval
executes it, and the message is flagged `"truncated": True`; or an over-budget
prompt (`"context"`).

Cost. A full `RetailDB.model_copy(deep=True)` of the 2.8MB db measured
0.094 s, more than everything else in an episode (env 0.02 s, end-state hash
0.04 s). Each episode instead runs on a `CowEnv`: fresh top-level
`users`/`orders` dicts sharing every record with the base db, plus private
deep copies, made just before a non-read tool executes, of the few records
that tool can write. An episode that never attempts a write copies nothing,
and its end-state hash is the cached base hash. `tests/test_episodes.py`
checks that the shared db is unchanged after write-heavy episodes and that
copy-on-write end states equal a full-deep-copy replay.

Threads. All episodes of a process share one `RetailEnv` that is re-pointed
at each episode's db per call, so isolate + bind + execute run under one
module lock (`_ENV_LOCK`). Without it, a thread switch between binding and
the tool body ran one episode's write against another episode's view, where
the record was never privately copied -- it wrote into `base_db()` itself and
every later write episode in that process scored as a failure. The lock costs
nothing measurable: tool calls are pure Python and GIL-bound anyway.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tau2.domains.retail.data_model import RetailDB
from tau2.utils import get_dict_hash

from tau_forge.envs.retail import RetailEnv, ToolResult
from tau_forge.episodes.task import (
    AUTH_TOOLS,
    READ_TOOLS,
    STOP,
    TRANSFER_TOOL,
    WRITE_TOOLS,
    EpisodeTask,
    base_db,
    base_db_hash,
    write_record,
)
from tau_forge.episodes.user import ScriptedUser, TurnContext, UserReply
from tau_forge.train.completion_parsing import ParsedCompletion, parse_all_completion

GREETING = "Hi! How can I help you today?"


@dataclass
class WriteRecord:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    # Gate inputs, captured at the moment of the call.
    confirmed: bool
    authed_user: Optional[str]
    error: Optional[str] = None
    # The consent slot the call maps to (by tool + record); None if none.
    slot: Optional[str] = None


@dataclass
class EpisodeLog:
    # The user id of the last successful auth lookup whose identifying
    # arguments the user had actually said (see `Episode._earned_auth`).
    authed_user: Optional[str] = None
    # Successful auth lookups with identifying values the user never said --
    # e.g. the email copied out of `get_user_details` -- by returned user id.
    unearned_auths: list[str] = field(default_factory=list)
    read_orders: list[str] = field(default_factory=list)
    writes: list[WriteRecord] = field(default_factory=list)
    transfer: bool = False
    n_calls: int = 0
    n_assistant_turns: int = 0
    n_malformed_calls: int = 0
    # Assistant turns that carried more than one tool call.
    n_multi_call_turns: int = 0
    # Primary intent of every user reply, one per user turn.
    user_intents: list[str] = field(default_factory=list)
    # One entry per user reply that accepted a denial: whether, at that
    # moment, the agent had earned auth of the task's user AND read the
    # target order (the refusal reward's verification condition).
    accepted_denials: list[bool] = field(default_factory=list)
    # The order each accepted refusal was about (parallel to accepted_denials).
    denied_orders: list[str] = field(default_factory=list)
    # Set when the audit loop caught an exception from `step` (a harness bug,
    # not a policy error): "<type>: <message>".
    runner_error: Optional[str] = None


@dataclass
class EpisodeResult:
    task_id: str
    final_db_hash: str
    end_reason: str
    log: EpisodeLog
    messages: list[dict[str, Any]]
    # Composite tasks: final hash of each sub-request's target order record.
    record_hashes: dict[str, str] = field(default_factory=dict)


def cow_view(db: RetailDB) -> RetailDB:
    """A RetailDB whose top-level dicts are new but whose records are shared
    with `db`. Safe only behind `CowEnv`, which replaces a record by a private
    copy before anything can write to it."""
    return RetailDB.model_construct(products=db.products, users=dict(db.users), orders=dict(db.orders))


_SHARED_ENV: Optional[RetailEnv] = None
# Guards _SHARED_ENV's creation and every isolate + bind + execute (see the
# module docstring, "Threads"). Re-entrant so `_shared_env()` can be called
# while it is held.
_ENV_LOCK = threading.RLock()


def _shared_env() -> RetailEnv:
    """One `RetailEnv` per process, re-pointed at each episode's db per call.

    Building a RetailEnv costs 0.04 s, nearly all of it tau2 deriving a
    pydantic argument model per tool -- identical for every episode. The
    tools are bound methods that read `toolkit.db` at call time (tau2's own
    `update_db` reassigns it the same way), so swapping the db is enough.
    Callers that bind it must hold `_ENV_LOCK` until the call returns."""
    global _SHARED_ENV
    with _ENV_LOCK:
        if _SHARED_ENV is None:
            _SHARED_ENV = RetailEnv(db=cow_view(base_db()))
        return _SHARED_ENV


class CowEnv:
    """A retail tool executor over a copy-on-write view of a shared base db.

    `execute` copies the records a non-read tool can touch before running it:
    the order named by `order_id`, that order's owner (payment-method gift
    card balances live on the user), and the user named by `user_id`. That
    covers every retail write -- cancel, exchange, return, the three order
    modifications, modify_user_address -- none of which writes a product.
    Thread-safe: see the module docstring."""

    def __init__(self, base: RetailDB, base_hash: Optional[str] = None):
        self.base = base
        self._base_hash = base_hash
        self.db = cow_view(base)
        self.copied_orders: set[str] = set()
        self.copied_users: set[str] = set()

    def _own_user(self, uid: Any) -> None:
        if isinstance(uid, str) and uid in self.db.users and uid not in self.copied_users:
            self.db.users[uid] = self.db.users[uid].model_copy(deep=True)
            self.copied_users.add(uid)

    def _isolate(self, arguments: dict[str, Any]) -> None:
        oid = arguments.get("order_id")
        if isinstance(oid, str) and oid in self.db.orders:
            if oid not in self.copied_orders:
                self.db.orders[oid] = self.db.orders[oid].model_copy(deep=True)
                self.copied_orders.add(oid)
            self._own_user(self.db.orders[oid].user_id)
        self._own_user(arguments.get("user_id"))

    def _bound(self) -> RetailEnv:
        env = _shared_env()
        env.db = self.db
        env._toolkit.db = self.db
        return env

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Never raises. `RetailEnv.execute` turns ValueError/TypeError into a
        failed ToolResult, but tau2's tools raise other things on ordinary
        malformed model input -- `calculate("(54.04 - 49.99")` and
        `calculate("2 3")` raise SyntaxError, `calculate("1/0")`
        ZeroDivisionError, `find_user_id_by_email(email=None)`
        AttributeError. Uncaught, one such sample ended a whole batched audit
        (every episode of every task, results only written at the end). tau2's
        own environment answers any tool exception with "Error: ..." and lets
        the agent carry on; so does this."""
        with _ENV_LOCK:
            try:
                if name not in READ_TOOLS and name != TRANSFER_TOOL:
                    self._isolate(arguments)
                return self._bound().execute(name, arguments)
            except Exception as e:  # noqa: BLE001 -- any tool failure is the policy's error, not the runner's
                return ToolResult(
                    ok=False, tool_name=name, arguments=arguments, error=f"{type(e).__name__}: {e}",
                    error_type=type(e).__name__,
                )

    def tool_mutates_state(self, name: str) -> bool:
        return _shared_env().tool_mutates_state(name)

    def db_hash(self) -> str:
        """tau2's db hash. An env that never copied a record is still the base
        db, whose hash is computed once (0.04 s saved per such episode)."""
        if not self.copied_orders and not self.copied_users:
            if self._base_hash is None:
                self._base_hash = get_dict_hash(self.base.model_dump())
            return self._base_hash
        return get_dict_hash(self.db.model_dump())


def _tool_content(result) -> str:
    if not result.ok:
        return f"Error: {result.error}"
    return result.value if isinstance(result.value, str) else json.dumps(result.value, default=str)


class Episode:
    """One conversation, driven one assistant completion at a time via
    `step`, so a batch generator can advance many episodes per forward pass
    (`scripts/episode_audit.py`). `run_episode` is the single-policy loop."""

    def __init__(
        self,
        task: EpisodeTask,
        *,
        system_message: Optional[dict[str, str]] = None,
        user_seed: int = 0,
        max_turns: int = 30,
        max_calls: int = 30,
        db: Optional[RetailDB] = None,
    ):
        self.task = task
        self.env = CowEnv(db, None) if db is not None else CowEnv(base_db(), base_db_hash())
        if task.subs:
            from tau_forge.episodes.composite import CompositeUser

            self.user = CompositeUser(task, seed=user_seed)
        else:
            self.user = ScriptedUser(task, seed=user_seed)
        self.max_turns = task.max_turns if task.max_turns is not None else max_turns
        self.max_calls = task.max_calls if task.max_calls is not None else max_calls
        self.messages: list[dict[str, Any]] = []
        if system_message is not None:
            self.messages.append(dict(system_message))
        self.messages.append({"role": "assistant", "content": GREETING})
        self.messages.append({"role": "user", "content": task.opening})
        self.log = EpisodeLog()
        self.done = False
        self.end_reason: Optional[str] = None
        # Consent state (see the module docstring).
        self._slot_ids = {s["id"] for s in task.slots}
        self._slot_by_key = {(s["tool"], s["record"]): s["id"] for s in task.slots}
        self._yes_slots: set[str] = set()
        self._suspended_slots: set[str] = set()
        self._writes_reported = 0  # writes already shown to the user in a TurnContext

    def final_db_hash(self) -> str:
        return self.env.db_hash()

    # ---- stepping -------------------------------------------------------

    def _finish(self, reason: str) -> None:
        self.done = True
        self.end_reason = reason

    def abort(self, error: str) -> None:
        """End the episode on a harness failure, keeping what it did so far."""
        self.log.runner_error = error
        if not self.done:
            self._finish("runner_error")

    def step(self, completion: str, finish_reason: str = "stop") -> None:
        if self.done:
            raise RuntimeError(f"episode {self.task.id} already ended ({self.end_reason})")
        if finish_reason == "context":
            self._finish("context_budget")
            return
        completion = completion or ""
        self.log.n_assistant_turns += 1
        parsed = parse_all_completion(completion)
        truncated = finish_reason == "length"
        if parsed.calls:
            # Eval executes whatever hermes parses, cut off or not.
            self._tool_turn(completion, parsed, truncated)
        elif truncated:
            # A cut-off text turn (or half-written call) ends the episode; the
            # trainer masks it. Nothing reaches the user.
            self.messages.append({"role": "assistant", "content": completion, "raw": completion, "truncated": True})
            self._finish("truncated")
            return
        else:
            if parsed.malformed:
                # Eval shows a rejected call to the user as text; so does this.
                self.log.n_malformed_calls += 1
            self._text_turn(completion)
        if not self.done and self.log.n_assistant_turns >= self.max_turns:
            self._finish("max_turns")
        if not self.done and self.log.n_calls >= self.max_calls:
            self._finish("max_calls")

    # ---- consent ----------------------------------------------------------

    def _slot_for(self, name: str, arguments: dict[str, Any]) -> Optional[str]:
        return self._slot_by_key.get((name, write_record(name, arguments)))

    def _apply_consent(self, reply: UserReply) -> None:
        intents = reply.intents or [reply.intent]
        if reply.revokes:
            self._yes_slots.clear()
            self._suspended_slots.clear()
        if "fallback" in intents and self._suspended_slots:
            self._yes_slots |= self._suspended_slots
            self._suspended_slots.clear()
        for slot in reply.confirms:
            if slot in self._slot_ids:
                self._yes_slots.add(slot)
                self._suspended_slots.discard(slot)

    # ---- turns ------------------------------------------------------------

    def _tool_turn(self, completion: str, parsed: ParsedCompletion, truncated: bool = False) -> None:
        tool_calls = []
        for name, arguments in parsed.calls:
            self.log.n_calls += 1
            tool_calls.append(
                {"id": f"call_{self.log.n_calls}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}
            )
        msg: dict[str, Any] = {"role": "assistant", "content": parsed.content, "tool_calls": tool_calls, "raw": completion}
        if truncated:
            msg["truncated"] = True
        self.messages.append(msg)
        if len(tool_calls) > 1:
            self.log.n_multi_call_turns += 1
        for tc, (name, arguments) in zip(tool_calls, parsed.calls):
            self._execute(tc["id"], name, arguments)
        if self.log.transfer:
            self._finish("transfer")

    def _execute(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        result = self.env.execute(name, arguments)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": _tool_content(result)})

        if result.ok and name in AUTH_TOOLS:
            if self._earned_auth(name, arguments):
                self.log.authed_user = result.value
            else:
                self.log.unearned_auths.append(result.value)
        if result.ok and name == "get_order_details":
            self.log.read_orders.append(arguments.get("order_id"))
        if name in WRITE_TOOLS:
            slot = self._slot_for(name, arguments)
            confirmed = slot is not None and slot in self._yes_slots
            self.log.writes.append(
                WriteRecord(name, dict(arguments), result.ok, confirmed, self.log.authed_user, result.error, slot)
            )
            if slot is not None:
                if result.ok:
                    # one yes buys one successful write
                    self._yes_slots.discard(slot)
                    self._suspended_slots.discard(slot)
                elif confirmed:
                    # a failed write may not be silently retried on the same yes
                    self._yes_slots.discard(slot)
                    self._suspended_slots.add(slot)
        if name == TRANSFER_TOOL:
            self.log.transfer = True

    def _earned_auth(self, name: str, arguments: dict[str, Any]) -> bool:
        """Whether an auth lookup's identifying values came from the user.

        policy.md wants the agent to authenticate *the user*: the lookup
        proves identity only if its email (or first name, last name and zip)
        is something the user said. Without this check the no_authentication
        gate was satisfied by self-authentication -- read the order from the
        id in the opening, `get_user_details` on its owner, then
        `find_user_id_by_email` with the email just read from the db -- which
        took every give_order_id task without identity in the opening from
        0.7 to the full 1.0 (cancel 46/46, exchange 53/53, modify_payment
        35/35, return_fallback 57/57 on 200 tasks per template), the user
        never having been asked who they are. Matching is case-insensitive
        substring on the user's turns, the opening included."""
        said = "\n".join(m["content"] for m in self.messages if m["role"] == "user").lower()
        if name == "find_user_id_by_email":
            keys = ("email",)
        else:
            keys = ("first_name", "last_name", "zip")
        values = [arguments.get(k) for k in keys]
        return all(isinstance(v, str) and v.strip() and v.strip().lower() in said for v in values)

    def _turn_context(self) -> TurnContext:
        ctx = TurnContext(
            new_writes=list(self.log.writes[self._writes_reported:]),
            authed_user=self.log.authed_user,
            read_orders=list(self.log.read_orders),
            any_write_ok=any(w.ok for w in self.log.writes),
        )
        self._writes_reported = len(self.log.writes)
        return ctx

    def _text_turn(self, completion: str) -> None:
        self.messages.append({"role": "assistant", "content": completion, "raw": completion})
        reply: UserReply = self.user.reply(completion, self._turn_context())
        self.log.user_intents.append(reply.intent)
        if "accept_denial" in (reply.intents or [reply.intent]):
            denied = reply.denied_order or self.task.target_order
            self.log.accepted_denials.append(
                self.log.authed_user == self.task.user_id and denied in self.log.read_orders
            )
            self.log.denied_orders.append(denied)
        self._apply_consent(reply)
        self.messages.append({"role": "user", "content": reply.text})
        if reply.stop:
            self._finish("user_stop")

    def record_hashes(self, order_ids: list[str]) -> dict[str, str]:
        """Hash of each named order record in the episode's db (composite partial credit)."""
        out = {}
        for oid in order_ids:
            rec = self.env.db.orders.get(oid)
            out[oid] = get_dict_hash(rec.model_dump()) if rec is not None else ""
        return out

    def result(self) -> EpisodeResult:
        records = {}
        if self.task.subs:
            records = self.record_hashes([d["target_order"] for d in self.task.subs])
        return EpisodeResult(
            task_id=self.task.id,
            final_db_hash=self.final_db_hash(),
            end_reason=self.end_reason or "running",
            log=self.log,
            messages=self.messages,
            record_hashes=records,
        )


def run_episode(
    task: EpisodeTask,
    policy: Callable[[list[dict[str, Any]]], str],
    **episode_kwargs: Any,
) -> EpisodeResult:
    """Drive one episode to termination with a synchronous policy. The policy
    receives the live message list (system message first if one was given)
    and must not mutate it."""
    ep = Episode(task, **episode_kwargs)
    while not ep.done:
        ep.step(policy(ep.messages))
    return ep.result()


__all__ = [
    "CowEnv", "Episode", "EpisodeLog", "EpisodeResult", "GREETING", "STOP", "TurnContext", "WriteRecord", "cow_view",
    "run_episode",
]
