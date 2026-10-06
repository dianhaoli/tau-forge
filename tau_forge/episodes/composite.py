"""Composite episode tasks: 2-3 requests in one conversation.

Half of tau2 retail's tasks have two or more writes, often on different orders
("cancel this one and return that one"). The single-request templates train
each skill in isolation; a composite strings verified single-request tasks of
the SAME user on DIFFERENT orders into one conversation:

  * the opening is the first sub-request's opening; every later request is
    revealed only once the previous one is resolved ("Thanks! One more thing:
    ..."), the way tau2's LLM user works through its task instructions;
  * a write sub-request is resolved by a successful write on its order, a
    refusal sub-request (status / foreign) by an accepted refusal;
  * each sub-request has its own consent slot (`s0`, `s1`, ...): one yes buys
    one write on one order;
  * success = the final db equals the gold chain's end state (any permutation
    of a multi-item modify) AND every refusal sub-request was refused; gates as
    for single tasks. A failure earns partial credit per sub-request done right
    (capped at 0.3, below the worst success).

Composites are built from already generated, verified and decontaminated
single tasks (`compose_tasks`), then verified and decontaminated again as a
whole. The scripted user is a thin wrapper around one `ScriptedUser` per
sub-request (`CompositeUser`); the reference agent is one `ReferenceAgent` per
sub-request (`CompositeAgent`).
"""

from __future__ import annotations

import itertools
import random
import re
from collections import defaultdict
from typing import Any, Optional

from tau2.utils import get_dict_hash

from tau_forge.episodes.task import ORDER_ID_RE, STOP, EpisodeTask, base_db

THANKS = ["Great, thank you! That's all I needed.", "Perfect, thanks for your help.", "Thanks, that's everything."]
NEXT = ["Thanks! One more thing: ", "Great, thank you. I also need something else: ", "Perfect. There's one more thing: "]
_GREETING_RE = re.compile(r"^(?:Hi|Hello|Hey there)(?:!|,[^.!?]*[.!?])\s*")

# Combos (sub-request templates in reveal order) and weights, after the write-type combinations of tau2
# retail's train split (aggregate counts only): repeated cancels / returns / exchanges on different orders,
# mixed write types, and "partial" requests where one part must be refused.
COMBOS: list[tuple[tuple[str, ...], int]] = [
    (("cancel", "cancel"), 8),
    (("cancel", "return_fallback"), 10),
    (("exchange", "exchange"), 11),
    (("return_fallback", "return_fallback"), 8),
    (("exchange", "return_fallback"), 6),
    (("modify_items", "return_fallback"), 5),
    (("cancel", "exchange"), 5),
    (("exchange", "modify_items"), 4),
    (("cancel", "modify_items"), 4),
    (("cancel", "status_refusal"), 4),
    (("return_fallback", "status_refusal"), 4),
    (("exchange", "foreign_order_refusal"), 2),
    (("cancel", "modify_address"), 4),
    (("modify_address", "return_fallback"), 3),
    (("cancel", "return_fallback", "exchange"), 3),
    (("cancel", "cancel", "return_fallback"), 3),
]


def body_of(opening: str) -> str:
    """A sub-request's opening without its greeting / identity clause."""
    return _GREETING_RE.sub("", opening, count=1).strip()


def _orders_of(task: EpisodeTask) -> set[str]:
    out = {task.target_order}
    if (task.hidden or {}).get("first_order"):
        out.add(task.hidden["first_order"])
    return out


# ------------------------------------------------------------------- scripted user


class CompositeUser:
    """One `ScriptedUser` per sub-request; the current one answers, the next is revealed when it is done."""

    def __init__(self, task: EpisodeTask, seed: int = 0, analyzer: Any = None):
        from tau_forge.episodes.user import ScriptedUser

        self.task = task
        self.subs = [EpisodeTask.from_dict(d) for d in task.subs]
        self.users = [ScriptedUser(s, seed=seed, analyzer=analyzer) for s in self.subs]
        self.rng = random.Random(f"{task.id}/composite/{seed}")
        self.i = 0
        self.reveals = list((task.hidden or {}).get("reveals") or [])
        # slot id per sub-request (refusal sub-requests have none)
        self.slot_of = {}
        for i, sub in enumerate(self.subs):
            if sub.slots:
                self.slot_of[i] = f"s{i}"

    # The scripted users' state that the runner / tests look at.
    @property
    def correction_used(self) -> bool:
        return self.users[self.i].correction_used

    @property
    def fallback_used(self) -> bool:
        return self.users[self.i].fallback_used

    def _next(self, prefix: str = "") -> "Any":
        from tau_forge.episodes.user import UserReply

        self.i += 1
        if self.i >= len(self.subs):
            return UserReply(f"{prefix}{self.rng.choice(THANKS)} {STOP}".strip(), True, "thanks")
        text = f"{prefix}{self.rng.choice(NEXT)}{self.reveals[self.i]}".strip()
        return UserReply(text, False, "next_request", intents=["next_request"], revokes=True)

    def _filter(self, txt: str) -> str:
        """Drop sentences only about orders of requests already done, so the current user does not read
        "I've cancelled #W1" as a proposal about another order."""
        done = set()
        for j in range(self.i):
            done |= _orders_of(self.subs[j])
        cur = _orders_of(self.subs[self.i])
        if not done:
            return txt
        from tau_forge.episodes.nlu import sentences

        keep = []
        for s in sentences(txt):
            ids = set(ORDER_ID_RE.findall(s))
            if ids and ids <= done and not ids & cur:
                continue
            keep.append(s)
        return " ".join(keep) if keep else txt

    def reply(self, agent_text: str, ctx: Any = None, write_succeeded: bool = False):
        from tau_forge.episodes.user import TurnContext, UserReply

        sub = self.subs[self.i]
        sid = self.slot_of.get(self.i)
        if ctx is not None and sid is not None and any(w.ok and w.slot == sid for w in ctx.new_writes):
            return self._next()
        sub_ctx = None
        if ctx is not None:
            sub_ctx = TurnContext(
                new_writes=[w for w in ctx.new_writes if w.slot == sid],
                authed_user=ctx.authed_user, read_orders=list(ctx.read_orders), any_write_ok=False,
            )
        r = self.users[self.i].reply(self._filter(agent_text), sub_ctx)
        confirms = [sid for c in r.confirms if c == "main" and sid] if r.confirms else []
        if "accept_denial" in (r.intents or [r.intent]):
            line = r.text.replace(STOP, "").strip()
            nxt = self._next(prefix=line + " ")
            return UserReply(nxt.text, nxt.stop, "accept_denial", intents=["accept_denial"] + nxt.intents[:1],
                             revokes=True, denied_order=sub.target_order)
        return UserReply(r.text, r.stop, r.intent, intents=list(r.intents), confirms=confirms, revokes=r.revokes,
                         answered_recap=r.answered_recap)


# ------------------------------------------------------------------- composition


def compose(subs: list[EpisodeTask], rng: random.Random) -> EpisodeTask:
    """One composite task from single tasks of one user (verified separately by the caller)."""
    first = subs[0]
    gold = list(first.gold_actions[:2])  # auth + get_user_details, once
    for s in subs:
        gold += s.gold_actions[2:]
    reveals = [body_of(s.opening) for s in subs]
    n = len(subs)
    write_subs = [s for s in subs if not s.expect_no_write]
    return EpisodeTask(
        id="", template="composite", user_id=first.user_id, opening=first.opening,
        profile={"reveal": [f"{x}{r}" for r in reveals[1:] for x in NEXT]},
        gold_actions=gold, target_order=(write_subs[0] if write_subs else first).target_order,
        expect_no_write=not write_subs,
        difficulty={"n_subs": n, "combo": [s.template for s in subs],
                    "id_mode": first.difficulty.get("id_mode"), "identity_upfront": first.difficulty.get("identity_upfront"),
                    "sub_difficulty": [s.difficulty for s in subs]},
        hidden={"reveals": reveals, "sub_ids": [s.id for s in subs], "item_ids": [i for s in subs for i in s.hidden.get("item_ids", [])]},
        involved_users=sorted({u for s in subs for u in s.involved_users}),
        subs=[s.to_dict() for s in subs],
        max_turns=14 + 8 * n, max_calls=14 + 8 * n,
    )


def gold_variants(task: EpisodeTask, db=None, base_hash: Optional[str] = None) -> tuple[list[str], dict[str, list[str]]]:
    """(end-state hashes, per-sub-target record hashes) over every listing order of every multi-item
    modify write in the gold chain."""
    from tau_forge.episodes.runner import CowEnv

    idx = [i for i, a in enumerate(task.gold_actions)
           if a["name"] == "modify_pending_order_items" and len(a["arguments"]["item_ids"]) > 1]
    choices = []
    for i in idx:
        args = task.gold_actions[i]["arguments"]
        choices.append(list(itertools.permutations(list(zip(args["item_ids"], args["new_item_ids"])))))
    targets = [d["target_order"] for d in task.subs] if task.subs else [task.target_order]
    hashes: set[str] = set()
    records: dict[str, set[str]] = defaultdict(set)
    for combo in itertools.product(*choices) if choices else [()]:
        env = CowEnv(db if db is not None else base_db(), base_hash)
        perm = dict(zip(idx, combo))
        for j, a in enumerate(task.gold_actions):
            args = a["arguments"]
            if j in perm:
                args = {**args, "item_ids": [p[0] for p in perm[j]], "new_item_ids": [p[1] for p in perm[j]]}
            env.execute(a["name"], args)
        hashes.add(env.db_hash())
        for oid in targets:
            rec = env.db.orders.get(oid)
            records[oid].add(get_dict_hash(rec.model_dump()) if rec is not None else "")
    return sorted(hashes), {k: sorted(v) for k, v in records.items()}


def compose_tasks(pool: list[EpisodeTask], n: int, seed: int = 0, *, exclusions: Any = None,
                  log: Any = None) -> list[EpisodeTask]:
    """Up to `n` verified composites from a pool of generated single tasks (any templates)."""
    from tau_forge.envs.retail import RetailEnv
    from tau_forge.episodes.generate import verify_task

    log = log or (lambda _m: None)
    rng = random.Random(f"composite/{seed}")
    db = base_db()
    base_hash = RetailEnv(db=db).db_hash()
    real_hashes = set(getattr(exclusions, "gold_db_hashes", ()) or ()) - {base_hash}
    by_user: dict[tuple, dict[str, list[EpisodeTask]]] = defaultdict(lambda: defaultdict(list))
    for t in pool:
        if t.subs or t.template == "modify_payment":
            continue
        by_user[(t.user_id, t.difficulty.get("id_mode"))][t.template].append(t)
    combos, weights = zip(*COMBOS)
    out: list[EpisodeTask] = []
    seen: set[tuple] = set()
    misses = 0
    while len(out) < n and misses < 5000:
        misses += 1
        combo = rng.choices(combos, weights=weights)[0]
        users = [u for u, d in by_user.items() if all(len(d[c]) >= combo.count(c) for c in set(combo))]
        if not users:
            continue
        key = rng.choice(sorted(users))
        picks: list[EpisodeTask] = []
        used: set[str] = set()
        for c in combo:
            cands = [t for t in by_user[key][c] if t not in picks and not (_orders_of(t) & used)]
            if not cands:
                break
            t = rng.choice(cands)
            picks.append(t)
            used |= _orders_of(t)
        if len(picks) != len(combo):
            continue
        # the identity only comes up front if the first request says it
        dkey = tuple(sorted(t.id for t in picks)) + tuple(t.template for t in picks)
        if dkey in seen:
            continue
        seen.add(dkey)
        task = compose(picks, rng)
        report = verify_task(task, db, base_hash)
        if not report.ok:
            continue
        hashes, records = gold_variants(task, db, base_hash)
        if real_hashes & set(hashes):
            continue
        task.gold_db_hash = report.gold_db_hash
        task.alt_gold_db_hashes = [h for h in hashes if h != report.gold_db_hash]
        task.hidden["gold_records"] = records
        task.id = f"ep_composite_s{seed}_{len(out):05d}"
        out.append(task)
        misses = 0
    log(f"[episodes.composite] composed {len(out)}/{n}")
    return out


# ------------------------------------------------------------------- reference agent


class CompositeAgent:
    """One `ReferenceAgent` per sub-request; switches when the user reveals the next request."""

    def __init__(self, task: EpisodeTask, mode: str = "oracle"):
        from tau_forge.episodes.reference_agents import ReferenceAgent

        self.t = task
        self.subs = [EpisodeTask.from_dict(d) for d in task.subs]
        self.agents = [ReferenceAgent(s, mode) for s in self.subs]
        self.reveals = task.hidden["reveals"]
        self.i = 0

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        users = [m["content"] for m in messages if m["role"] == "user"]
        while self.i + 1 < len(self.subs) and self.reveals[self.i + 1] in users[-1]:
            self.i += 1
            self.agents[self.i].user_from = len(users) - 1
        return self.agents[self.i](messages)
