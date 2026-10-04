"""The episode loop: policy <-> scripted user <-> a live, private RetailEnv.

The policy is any `callable(messages) -> assistant text`. Its text is parsed
with the same `completion_parsing.parse_completion` the single-step reward
uses, so a tool call means one `<tool_call>{json}</tool_call>` block (Qwen's
native convention; only the first block is executed -- tau2's policy allows
one call per turn). A call is executed and fed back as an assistant
`tool_calls` message plus a `role: "tool"` result, the exact shape
`tau_forge.train.grounding` renders and tau2 feeds its agent at eval time.
A text turn goes to the scripted user, whose reply is appended as a user
message.

The conversation starts the way tau2's orchestrator starts every task: an
assistant greeting ("Hi! How can I help you today?", tau2's
DEFAULT_FIRST_AGENT_MESSAGE), then the task's opening user message.

Termination: user STOP (done, gave up after 3 unrecognised turns, or accepted
a denial); a `transfer_to_human_agents` call; `max_turns` assistant messages
(30, the design's budget); `max_calls` tool calls; a truncated completion
(`finish_reason="length"`, ended without executing a half-written call, to be
masked by the trainer); or an over-budget prompt (`"context"`).

Cost. A full `RetailDB.model_copy(deep=True)` of the 2.8MB db measured
0.094 s, more than everything else in an episode (env 0.02 s, end-state hash
0.04 s). Each episode instead runs on a `CowEnv`: fresh top-level
`users`/`orders` dicts sharing every record with the base db, plus private
deep copies, made just before a non-read tool executes, of the few records
that tool can write. An episode that never attempts a write copies nothing,
and its end-state hash is the cached base hash. `tests/test_episodes.py`
checks that the shared db is unchanged after write-heavy episodes and that
copy-on-write end states equal a full-deep-copy replay.
"""

from __future__ import annotations

import json
import re
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
)
from tau_forge.episodes.user import ScriptedUser, UserReply
from tau_forge.train.completion_parsing import MALFORMED_TOOL_CALL, parse_completion

GREETING = "Hi! How can I help you today?"
_TOOL_BLOCK_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL)


@dataclass
class WriteRecord:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    # Gate inputs, captured at the moment of the call.
    confirmed: bool
    authed_user: Optional[str]
    error: Optional[str] = None


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
    user_intents: list[str] = field(default_factory=list)
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


def cow_view(db: RetailDB) -> RetailDB:
    """A RetailDB whose top-level dicts are new but whose records are shared
    with `db`. Safe only behind `CowEnv`, which replaces a record by a private
    copy before anything can write to it."""
    return RetailDB.model_construct(products=db.products, users=dict(db.users), orders=dict(db.orders))


_SHARED_ENV: Optional[RetailEnv] = None


def _shared_env() -> RetailEnv:
    """One `RetailEnv` per process, re-pointed at each episode's db per call.

    Building a RetailEnv costs 0.04 s, nearly all of it tau2 deriving a
    pydantic argument model per tool -- identical for every episode. The
    tools are bound methods that read `toolkit.db` at call time (tau2's own
    `update_db` reassigns it the same way), so swapping the db is enough.
    Calls are synchronous; do not share one process's episodes across
    threads."""
    global _SHARED_ENV
    if _SHARED_ENV is None:
        _SHARED_ENV = RetailEnv(db=cow_view(base_db()))
    return _SHARED_ENV


class CowEnv:
    """A retail tool executor over a copy-on-write view of a shared base db.

    `execute` copies the records a non-read tool can touch before running it:
    the order named by `order_id`, that order's owner (payment-method gift
    card balances live on the user), and the user named by `user_id`. That
    covers every retail write -- cancel, exchange, return, the three order
    modifications, modify_user_address -- none of which writes a product."""

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
        if name not in READ_TOOLS and name != TRANSFER_TOOL:
            self._isolate(arguments)
        try:
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
        self.user = ScriptedUser(task, seed=user_seed)
        self.max_turns = max_turns
        self.max_calls = max_calls
        self.messages: list[dict[str, Any]] = []
        if system_message is not None:
            self.messages.append(dict(system_message))
        self.messages.append({"role": "assistant", "content": GREETING})
        self.messages.append({"role": "user", "content": task.opening})
        self.log = EpisodeLog()
        self.done = False
        self.end_reason: Optional[str] = None
        self._yes = False

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
        if finish_reason == "length":
            # Never execute a half-written call; the trainer masks these.
            self.messages.append({"role": "assistant", "content": completion})
            self.log.n_assistant_turns += 1
            self._finish("truncated")
            return
        self.log.n_assistant_turns += 1
        name, arguments = parse_completion(completion)
        if name is not None and name != MALFORMED_TOOL_CALL:
            self._tool_turn(completion, name, arguments)
        else:
            if name == MALFORMED_TOOL_CALL:
                # No parsable name means no tool message to attach a result
                # to; it reaches the user as text, who will not recognise it.
                self.log.n_malformed_calls += 1
            self._text_turn(completion)
        if not self.done and self.log.n_assistant_turns >= self.max_turns:
            self._finish("max_turns")
        if not self.done and self.log.n_calls >= self.max_calls:
            self._finish("max_calls")

    def _tool_turn(self, completion: str, name: str, arguments: dict[str, Any]) -> None:
        self.log.n_calls += 1
        call_id = f"call_{self.log.n_calls}"
        self.messages.append(
            {
                "role": "assistant",
                "content": _TOOL_BLOCK_RE.sub("", completion).strip(),
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                ],
            }
        )
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
            self.log.writes.append(
                WriteRecord(name, dict(arguments), result.ok, self._yes, self.log.authed_user, result.error)
            )
            if result.ok:
                self._yes = False  # one yes buys one write
        if name == TRANSFER_TOOL:
            self.log.transfer = True
            self._finish("transfer")

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

    def _text_turn(self, completion: str) -> None:
        self.messages.append({"role": "assistant", "content": completion})
        wrote = any(w.ok for w in self.log.writes)
        reply: UserReply = self.user.reply(completion, write_succeeded=wrote)
        self.log.user_intents.append(reply.intent)
        if reply.is_yes:
            self._yes = True
        elif reply.answered_recap:
            self._yes = False
        self.messages.append({"role": "user", "content": reply.text})
        if reply.stop:
            self._finish("user_stop")

    def result(self) -> EpisodeResult:
        return EpisodeResult(
            task_id=self.task.id,
            final_db_hash=self.final_db_hash(),
            end_reason=self.end_reason or "running",
            log=self.log,
            messages=self.messages,
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


__all__ = ["CowEnv", "Episode", "EpisodeLog", "EpisodeResult", "GREETING", "STOP", "WriteRecord", "cow_view", "run_episode"]
