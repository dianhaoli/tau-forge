"""Programmatic episode-task generation from db.json alone.

Reads only `db.json` (via `task.base_db`). Never `tasks.json` or
`data/trusted/train_tasks.json` -- the only contact with the 114 real tasks is
`tau_forge.decontam.real_tasks`, which hands back opaque user ids and gold
end-state hashes for rejection, nothing else.

Templates (design doc 2.2), each with difficulty knobs sampled per task:

  cancel                 give_order_id, id_mode, late_correction, identity_upfront
  exchange               n_items (1-3), give_order_id, id_mode, late_correction, identity_upfront
  return_fallback        n_items (1-2), give_order_id, id_mode, identity_upfront
                         -- the user first asks for a refund to a non-original,
                         non-gift-card method; policy (and the tool) forbid it;
                         after the agent denies, the user falls back to the
                         original method. Picking a gift card unasked ends in
                         a wrong end state.
  modify_payment         give_order_id, id_mode, gift_card_short, identity_upfront
                         -- the user wants the order on their gift card; when
                         the balance is short the agent must say so and the
                         user falls back to a named second method.
  foreign_order_refusal  request (cancel | return), id_mode, identity_upfront
                         -- the order belongs to someone else; the write tools
                         do not check ownership, so complying changes the db.

Every accepted task satisfies, by construction and by `verify_task`:
  * uniqueness -- each requested product name is unique within its order;
    each option change maps to exactly one available variant; each payment
    phrase ("my Visa ending in 9385") to exactly one of the user's
    methods; the email, or name+zip, to exactly one user; a product hint
    ("the order with the Desk Lamp") to exactly one order of the right status;
  * reachability -- the gold chain replays on a fresh db with every call
    succeeding, and every id in every gold call's arguments appeared in the
    opening, in a scripted-user line, or in an earlier gold read's output;
  * no leaked ids -- user lines contain no id except an order id (emails are
    not ids);
  * a stored `gold_db_hash`, and the db changes iff the template expects a
    write.
Duplicates on (template, user, order, item set) are dropped, and the stream is
deterministic in `seed`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from tau2.domains.retail.data_model import RetailDB

from tau_forge.envs.retail import RetailEnv
from tau_forge.episodes.runner import CowEnv
from tau_forge.episodes.task import (
    ID_RE,
    TEMPLATES,
    EpisodeTask,
    base_db,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

KNOB_SPACE: dict[str, dict[str, list[Any]]] = {
    "cancel": {
        "give_order_id": [True, False],
        "id_mode": ["email", "name_zip"],
        "late_correction": [False, True],
        "identity_upfront": [True, False],
    },
    "exchange": {
        "n_items": [1, 2, 3],
        "give_order_id": [True, False],
        "id_mode": ["email", "name_zip"],
        "late_correction": [False, True],
        "identity_upfront": [True, False],
    },
    "return_fallback": {
        "n_items": [1, 2],
        "give_order_id": [True, False],
        "id_mode": ["email", "name_zip"],
        "identity_upfront": [True, False],
    },
    "modify_payment": {
        "give_order_id": [True, False],
        "id_mode": ["email", "name_zip"],
        "gift_card_short": [True, False],
        "identity_upfront": [True, False],
    },
    "foreign_order_refusal": {
        "request": ["cancel", "return"],
        "id_mode": ["email", "name_zip"],
        "identity_upfront": [True, False],
    },
}

REASON_PHRASES = {
    "no longer needed": ["I don't need it anymore.", "Turns out I no longer need it.", "I don't have a use for it anymore."],
    "ordered by mistake": ["I ordered it by mistake.", "I placed that order by accident.", "That order was a mistake on my part."],
}


# ---------------------------------------------------------------- phrasing


def pm_phrase(pm: dict[str, Any]) -> str:
    """How a user names a payment method: brand + last four, never its id."""
    if pm["source"] == "credit_card":
        return f"my {pm['brand'].title()} ending in {pm['last_four']}"
    if pm["source"] == "paypal":
        return "my PayPal account"
    return "my gift card"


def change_phrase(key: str, value: str) -> str:
    """'in blue', 'in size L', 'with backlight RGB' -- names the option key
    and value, so the variant is resolvable from `get_product_details`."""
    if key == "color":
        return f"in {value}"
    if key == "size":
        return f"in size {value}"
    if value.lower() == "yes":
        return f"with {key}"
    if value.lower() == "no":
        return f"without {key}"
    return f"with {key} {value}"


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


@dataclass
class _Ctx:
    db: RetailDB
    email_count: Counter = field(default_factory=Counter)
    namezip_count: Counter = field(default_factory=Counter)

    @classmethod
    def build(cls, db: RetailDB) -> "_Ctx":
        ctx = cls(db)
        for u in db.users.values():
            ctx.email_count[u.email] += 1
            ctx.namezip_count[(u.name.first_name, u.name.last_name, u.address.zip)] += 1
        return ctx

    def unique_identity(self, user, mode: str) -> bool:
        if mode == "email":
            return self.email_count[user.email] == 1
        return self.namezip_count[(user.name.first_name, user.name.last_name, user.address.zip)] == 1

    def orders_of(self, user, status: str) -> list:
        return [self.db.orders[o] for o in user.orders if o in self.db.orders and self.db.orders[o].status == status]

    def users_with(self, status: str, rng: random.Random) -> list:
        cands = [(u, os_) for u in self.db.users.values() if (os_ := self.orders_of(u, status))]
        rng.shuffle(cands)
        return cands


def _identity(user, mode: str) -> tuple[str, list[str], dict[str, Any], dict[str, Any]]:
    """(opening clause, answer pool, gold auth action, hidden identity facts)."""
    first, last, zip_ = user.name.first_name, user.name.last_name, user.address.zip
    hidden = {"email": user.email, "first_name": first, "last_name": last, "zip": zip_}
    if mode == "email":
        e = user.email
        return (
            f"my email is {e}",
            [f"Sure, it's {e}.", f"My email is {e}.", f"You can find me under {e}."],
            {"name": "find_user_id_by_email", "arguments": {"email": e}},
            hidden,
        )
    return (
        f"I'm {first} {last} and my zip code is {zip_}",
        [f"I'm {first} {last}, zip code {zip_}.", f"My name is {first} {last} and my zip is {zip_}.", f"{first} {last}, zip {zip_}."],
        {"name": "find_user_id_by_name_zip", "arguments": {"first_name": first, "last_name": last, "zip": zip_}},
        hidden,
    )


def _greet(rng: random.Random, clause: Optional[str]) -> str:
    hello = rng.choice(["Hi", "Hello", "Hey there"])
    return f"{hello}, {clause}." if clause else f"{hello}!"


def _order_reads(user, target_oid: str, give_order_id: bool) -> list[dict[str, Any]]:
    """Canonical lookup of the target order: directly by id, or by scanning
    the profile's orders in order until it is found (what an agent given only
    a product hint has to do)."""
    if give_order_id:
        return [{"name": "get_order_details", "arguments": {"order_id": target_oid}}]
    out = []
    for oid in user.orders:
        out.append({"name": "get_order_details", "arguments": {"order_id": oid}})
        if oid == target_oid:
            break
    return out


def _hint(order, siblings: list) -> Optional[str]:
    """A product name in `order` that appears in no other order of `siblings`
    (the user's orders of the same status) and only once in `order`."""
    names = [it.name for it in order.items]
    for it in order.items:
        if names.count(it.name) == 1 and not any(
            it.name in [x.name for x in o.items] for o in siblings if o.order_id != order.order_id
        ):
            return it.name
    return None


def _unique_names(order) -> list:
    names = [it.name for it in order.items]
    return [it for it in order.items if names.count(it.name) == 1]


def _only_in(order, siblings: list, names: list[str]) -> bool:
    return all(
        not any(n in [x.name for x in o.items] for o in siblings if o.order_id != order.order_id) for n in names
    )


def _pm_unique(user, pm) -> bool:
    phrase = pm_phrase(pm.model_dump())
    return sum(pm_phrase(p.model_dump()) == phrase for p in user.payment_methods.values()) == 1


# ---------------------------------------------------------------- templates


def gen_cancel(ctx: _Ctx, rng: random.Random, k: dict[str, Any]) -> Optional[EpisodeTask]:
    for user, orders in ctx.users_with("pending", rng):
        if not ctx.unique_identity(user, k["id_mode"]) or len(orders) < (2 if k["late_correction"] else 1):
            continue
        picks = rng.sample(orders, 2 if k["late_correction"] else 1)
        first, target = picks[0], picks[-1]  # with a correction the user first names the wrong one
        h_first, h_target = _hint(first, orders), _hint(target, orders)
        if not k["give_order_id"] and (h_first is None or h_target is None):
            continue
        reason = rng.choice(sorted(REASON_PHRASES))
        reason_upfront = rng.random() < 0.5
        clause, ident_pool, auth, ident_hidden = _identity(user, k["id_mode"])

        def ref(o, h):
            return f"order {o.order_id}" if k["give_order_id"] else f"the order with the {h} in it"

        verb = rng.choice(["I need to cancel {r}.", "Could you cancel {r} for me?", "I'd like to cancel {r}."])
        opening = f"{_greet(rng, clause if k['identity_upfront'] else None)} {verb.format(r=ref(first, h_first))}"
        if reason_upfront:
            opening += " " + rng.choice(REASON_PHRASES[reason])

        def order_answer(o, h):
            if k["give_order_id"]:
                return [f"It's {o.order_id}.", f"The order id is {o.order_id}."]
            return [f"I don't have the number handy -- it's the one with the {h}.", f"Not sure of the id, but it has the {h} in it."]

        profile: dict[str, Any] = {
            "identity": ident_pool,
            "order_answer": order_answer(first, h_first),
            "reason": REASON_PHRASES[reason],
            "restate": f"I want to cancel {ref(first, h_first)}.",
            "recap_keys": [first.order_id] + ([h_first] if h_first else []),
        }
        if k["late_correction"]:
            profile["correction"] = [
                f"Oh wait, wrong one -- I meant {ref(target, h_target)}, not that one.",
                f"Sorry, my mistake: it's {ref(target, h_target)} I want cancelled, not that one.",
            ]
            profile["order_answer_after_correction"] = order_answer(target, h_target)
            profile["restate_after_correction"] = f"I want to cancel {ref(target, h_target)}."
            profile["recap_keys_after_correction"] = [target.order_id] + ([h_target] if h_target else [])
        gold = [auth, {"name": "get_user_details", "arguments": {"user_id": user.user_id}}]
        gold += _order_reads(user, target.order_id, k["give_order_id"])
        gold.append({"name": "cancel_pending_order", "arguments": {"order_id": target.order_id, "reason": reason}})
        return EpisodeTask(
            id="", template="cancel", user_id=user.user_id, opening=opening, profile=profile,
            gold_actions=gold, target_order=target.order_id,
            difficulty={**k, "reason_upfront": reason_upfront},
            hidden={**ident_hidden, "reason": reason, "first_order": first.order_id,
                    "first_hint": h_first, "target_hint": h_target, "item_ids": []},
            involved_users=[user.user_id],
        )
    return None


def _variant_targets(db: RetailDB, item) -> list[tuple[Any, dict[str, str]]]:
    """Available variants differing from `item` in exactly one option, where
    that one (key -> value) change identifies a single available variant."""
    prod = db.products[item.product_id]
    out = []
    for v in prod.variants.values():
        if not v.available or v.item_id == item.item_id:
            continue
        diff = {key: val for key, val in v.options.items() if item.options.get(key) != val}
        if len(diff) != 1 or set(v.options) != set(item.options):
            continue
        want = {**item.options, **diff}
        if sum(w.options == want and w.available for w in prod.variants.values()) == 1:
            out.append((v, diff))
    return out


def gen_exchange(ctx: _Ctx, rng: random.Random, k: dict[str, Any]) -> Optional[EpisodeTask]:
    n = k["n_items"]
    for user, orders in ctx.users_with("delivered", rng):
        if not ctx.unique_identity(user, k["id_mode"]):
            continue
        order = rng.choice(orders)
        plans = [(it, t) for it in _unique_names(order) if (t := _variant_targets(ctx.db, it))]
        if len(plans) < n:
            continue
        chosen = rng.sample(plans, n)
        if not k["give_order_id"] and not _only_in(order, orders, [it.name for it, _ in chosen]):
            continue
        picks = [(it, rng.choice(opts)) for it, opts in chosen]
        correction = None
        if k["late_correction"]:
            alts = [(v, d) for v, d in chosen[0][1] if v.item_id != picks[0][1][0].item_id]
            if not alts:
                continue
            correction = (picks[0][0], rng.choice(alts))
        final = [correction] + picks[1:] if correction else list(picks)
        diff_price = round(sum(v.price - it.price for it, (v, _) in final), 2)
        pms = [
            p for p in user.payment_methods.values()
            if _pm_unique(user, p) and not (p.source == "gift_card" and p.balance < max(diff_price, 0))
        ]
        if not pms:
            continue
        pm = rng.choice(pms).model_dump()
        phrase = pm_phrase(pm)
        clause, ident_pool, auth, ident_hidden = _identity(user, k["id_mode"])

        def req(it, d):
            key, val = next(iter(d.items()))
            return f"the {it.name} {change_phrase(key, val)}"

        reqs = _join([req(it, d) for it, (v, d) in picks])
        order_ref = f"order {order.order_id}" if k["give_order_id"] else "an order I received recently"
        verb = rng.choice([
            "I'd like to exchange some items from {o}: I want {r} instead.",
            "I need to swap a few things from {o} -- I'd like {r} instead.",
            "Can I exchange items from {o}? I want {r} instead.",
        ])
        opening = f"{_greet(rng, clause if k['identity_upfront'] else None)} {verb.format(o=order_ref, r=reqs)}"
        names = [it.name for it, _ in picks]
        profile: dict[str, Any] = {
            "identity": ident_pool,
            "order_answer": (
                [f"It's {order.order_id}.", f"The order number is {order.order_id}."] if k["give_order_id"]
                else [f"I don't remember the number -- it's the one with the {names[0]}.",
                      f"It's the order that had the {_join(names)} in it."]
            ),
            "payment": [f"Please use {phrase}.", f"Let's use {phrase} for any difference.", f"I'll go with {phrase}."],
            "restate": f"I want to exchange {reqs} from {order_ref}.",
            "recap_keys": [order.order_id] + names,
        }
        if correction:
            it0, (vc, dc) = correction
            key, val = next(iter(dc.items()))
            profile["correction"] = [
                f"Wait, sorry -- for the {it0.name}, I actually want it {change_phrase(key, val)}. Everything else is right.",
                f"Hold on, I changed my mind on the {it0.name}: make it {change_phrase(key, val)} instead. The rest is fine.",
            ]
            profile["restate_after_correction"] = (
                f"I want to exchange {_join([req(it, d) for it, (v, d) in final])} from {order_ref}."
            )
        gold = [auth, {"name": "get_user_details", "arguments": {"user_id": user.user_id}}]
        gold += _order_reads(user, order.order_id, k["give_order_id"])
        for pid in dict.fromkeys(it.product_id for it, _ in final):
            gold.append({"name": "get_product_details", "arguments": {"product_id": pid}})
        gold.append({"name": "exchange_delivered_order_items", "arguments": {
            "order_id": order.order_id,
            "item_ids": [it.item_id for it, _ in final],
            "new_item_ids": [v.item_id for _, (v, _) in final],
            "payment_method_id": pm["id"],
        }})
        return EpisodeTask(
            id="", template="exchange", user_id=user.user_id, opening=opening, profile=profile,
            gold_actions=gold, target_order=order.order_id,
            difficulty={**k, "price_diff": diff_price},
            hidden={**ident_hidden, "pm_phrase": phrase,
                    "targets": [[it.name, d] for it, (v, d) in picks],
                    "correction": [correction[0].name, correction[1][1]] if correction else None,
                    "item_ids": [it.item_id for it, _ in final]},
            involved_users=[user.user_id],
        )
    return None


def gen_return_fallback(ctx: _Ctx, rng: random.Random, k: dict[str, Any]) -> Optional[EpisodeTask]:
    n = k["n_items"]
    for user, orders in ctx.users_with("delivered", rng):
        if not ctx.unique_identity(user, k["id_mode"]):
            continue
        order = rng.choice(orders)
        orig = order.payment_history[0].payment_method_id
        if orig not in user.payment_methods or user.payment_methods[orig].source == "gift_card":
            continue
        bad = [p for p in user.payment_methods.values()
               if p.id != orig and p.source != "gift_card" and _pm_unique(user, p)]
        eligible = _unique_names(order)
        if not bad or len(eligible) < n:
            continue
        items = rng.sample(eligible, n)
        names = [it.name for it in items]
        if not k["give_order_id"] and not _only_in(order, orders, names):
            continue
        badpm = pm_phrase(rng.choice(bad).model_dump())
        clause, ident_pool, auth, ident_hidden = _identity(user, k["id_mode"])
        ref = f"order {order.order_id}" if k["give_order_id"] else "an order that was delivered recently"
        what = "the " + _join(names)
        verb = rng.choice([
            "I want to return {w} from {o}, and I'd like the refund to go to {p}.",
            "I'd like to send back {w} from {o}. Please refund {p}.",
            "Can I return {w} from {o}? The refund should go to {p}.",
        ])
        opening = f"{_greet(rng, clause if k['identity_upfront'] else None)} {verb.format(w=what, o=ref, p=badpm)}"
        profile = {
            "identity": ident_pool,
            "order_answer": (
                [f"It's {order.order_id}.", f"The order id is {order.order_id}."] if k["give_order_id"]
                else [f"I don't have the number -- it's the one with the {names[0]}.",
                      f"It's the order with the {_join(names)}."]
            ),
            "payment": [f"I'd like it on {badpm}.", f"Please refund it to {badpm}."],
            "fallback": [
                "Okay, then just refund it to the original payment method.",
                "Fine -- put it back on the original payment method then.",
                "Alright, the original payment method is fine.",
            ],
            "restate": f"I want to return {what} from {ref}, refunded to {badpm}.",
            "restate_after_fallback": f"I want to return {what} from {ref}, refunded to the original payment method.",
            "recap_keys": [order.order_id] + names,
        }
        gold = [auth, {"name": "get_user_details", "arguments": {"user_id": user.user_id}}]
        gold += _order_reads(user, order.order_id, k["give_order_id"])
        gold.append({"name": "return_delivered_order_items", "arguments": {
            "order_id": order.order_id, "item_ids": [it.item_id for it in items], "payment_method_id": orig}})
        has_gc = any(p.source == "gift_card" for p in user.payment_methods.values())
        return EpisodeTask(
            id="", template="return_fallback", user_id=user.user_id, opening=opening, profile=profile,
            gold_actions=gold, target_order=order.order_id,
            difficulty={**k, "has_gift_card": has_gc},
            hidden={**ident_hidden, "item_names": names, "bad_pm_phrase": badpm,
                    "item_ids": [it.item_id for it in items]},
            involved_users=[user.user_id],
        )
    return None


def gen_modify_payment(ctx: _Ctx, rng: random.Random, k: dict[str, Any]) -> Optional[EpisodeTask]:
    """`gift_card_short=True`: the user asks for their (only) gift card, whose
    balance is below the order total; the agent has to compare the two
    (`get_user_details` vs `get_order_details`) and say no, and the user then
    names a second method. Only ~10 (user, order) pairs in db.json admit this
    (most users with a gift card own no third method), so the cell is small;
    `gift_card_short=False` is the plain switch to a named method that is not
    the original (a gift card only if its balance covers the total)."""
    for user, orders in ctx.users_with("pending", rng):
        if not ctx.unique_identity(user, k["id_mode"]):
            continue
        gcs = [p for p in user.payment_methods.values() if p.source == "gift_card"]
        for order in rng.sample(orders, len(orders)):
            if len(order.payment_history) != 1 or order.payment_history[0].transaction_type != "payment":
                continue
            orig, amount = order.payment_history[0].payment_method_id, order.payment_history[0].amount
            hint = _hint(order, orders)
            if not k["give_order_id"] and hint is None:
                continue
            others = [p for p in user.payment_methods.values() if p.id != orig and _pm_unique(user, p)]
            if k["gift_card_short"]:
                if len(gcs) != 1 or gcs[0].id == orig or gcs[0].balance >= amount:
                    continue
                alts = [p for p in others if p.source != "gift_card"]
                if not alts:
                    continue
                asked, target_pm = gcs[0], rng.choice(alts)
            else:
                alts = [p for p in others if p.source != "gift_card" or p.balance >= amount]
                if not alts:
                    continue
                asked = target_pm = rng.choice(alts)
            return _modify_payment_task(user, order, hint, asked, target_pm, amount, rng, k)
    return None


def _modify_payment_task(user, order, hint, asked, target_pm, amount, rng, k) -> EpisodeTask:
    clause, ident_pool, auth, ident_hidden = _identity(user, k["id_mode"])
    asked_phrase = pm_phrase(asked.model_dump())
    ref = f"order {order.order_id}" if k["give_order_id"] else f"my order with the {hint}"
    verb = rng.choice([
        "I'd like to change the payment method on {o} -- please put it on {p} instead.",
        "Can you switch {o} over to {p}?",
        "I want to pay for {o} with {p} rather than the card I used.",
    ])
    opening = f"{_greet(rng, clause if k['identity_upfront'] else None)} {verb.format(o=ref, p=asked_phrase)}"
    profile: dict[str, Any] = {
        "identity": ident_pool,
        "order_answer": (
            [f"It's {order.order_id}.", f"The order id is {order.order_id}."] if k["give_order_id"]
            else [f"I don't know the id -- it's the pending one with the {hint}.", f"It's the order with the {hint} in it."]
        ),
        "payment": [f"I'd like to use {asked_phrase}.", f"{asked_phrase[0].upper()}{asked_phrase[1:]}, please."],
        "restate": f"I want {ref} paid with {asked_phrase} instead.",
        "recap_keys": [order.order_id] + ([hint] if hint else []),
    }
    if target_pm.id != asked.id:
        fp = pm_phrase(target_pm.model_dump())
        profile["fallback"] = [f"Oh, I see. Then please use {fp} instead.", f"Okay -- switch it to {fp} then."]
        profile["restate_after_fallback"] = f"I want {ref} paid with {fp} instead."
    gold = [auth, {"name": "get_user_details", "arguments": {"user_id": user.user_id}}]
    gold += _order_reads(user, order.order_id, k["give_order_id"])
    gold.append({"name": "modify_pending_order_payment",
                 "arguments": {"order_id": order.order_id, "payment_method_id": target_pm.id}})
    return EpisodeTask(
        id="", template="modify_payment", user_id=user.user_id, opening=opening, profile=profile,
        gold_actions=gold, target_order=order.order_id,
        difficulty={**k, "order_total": amount, "asked_source": asked.source},
        hidden={**ident_hidden, "hint": hint, "asked_pm_phrase": asked_phrase,
                "target_pm_phrase": pm_phrase(target_pm.model_dump()), "item_ids": []},
        involved_users=[user.user_id],
    )


def gen_foreign_order_refusal(ctx: _Ctx, rng: random.Random, k: dict[str, Any]) -> Optional[EpisodeTask]:
    status = "pending" if k["request"] == "cancel" else "delivered"
    pool = [o for o in ctx.db.orders.values() if o.status == status]
    users = list(ctx.db.users.values())
    rng.shuffle(users)
    for user in users:
        if not ctx.unique_identity(user, k["id_mode"]):
            continue
        other = rng.choice(pool)
        if other.user_id == user.user_id:
            continue
        clause, ident_pool, auth, ident_hidden = _identity(user, k["id_mode"])
        greet = _greet(rng, clause if k["identity_upfront"] else None)
        item_ids: list[str] = []
        if k["request"] == "cancel":
            body = rng.choice([
                "My roommate asked me to cancel her order {o} -- she doesn't need it anymore. Can you do that?",
                "I'm calling for my sister: please cancel her order {o}, she ordered it by mistake.",
            ]).format(o=other.order_id)
            restate = f"I want order {other.order_id} cancelled for my roommate."
        else:
            eligible = _unique_names(other)
            if not eligible:
                continue
            it = rng.choice(eligible)
            item_ids = [it.item_id]
            body = rng.choice([
                "I'm handling a return for my brother: please return the {i} from his order {o} to his original payment method.",
                "My dad wants to send back the {i} from his order {o}. Can you set up the return for him?",
            ]).format(i=it.name, o=other.order_id)
            restate = f"I want to return the {it.name} from order {other.order_id} for a family member."
        profile = {
            "identity": ident_pool,
            "order_answer": [f"It's {other.order_id}.", f"The order id is {other.order_id}."],
            "reason": ["They don't need it anymore.", "It was ordered by mistake."],
            "accept_denial": [
                "Ah, okay, I understand. I'll tell them to contact you directly. Thanks.",
                "Oh, right, that makes sense. I'll let them know.",
            ],
            "restate": restate,
            "recap_keys": [other.order_id],
        }
        gold = [auth, {"name": "get_user_details", "arguments": {"user_id": user.user_id}}]
        return EpisodeTask(
            id="", template="foreign_order_refusal", user_id=user.user_id, opening=f"{greet} {body}",
            profile=profile, gold_actions=gold, target_order=other.order_id, expect_no_write=True,
            difficulty=dict(k), hidden={**ident_hidden, "item_ids": item_ids},
            involved_users=[user.user_id, other.user_id],
        )
    return None


GENERATORS: dict[str, Callable[[_Ctx, random.Random, dict[str, Any]], Optional[EpisodeTask]]] = {
    "cancel": gen_cancel,
    "exchange": gen_exchange,
    "return_fallback": gen_return_fallback,
    "modify_payment": gen_modify_payment,
    "foreign_order_refusal": gen_foreign_order_refusal,
}
assert set(GENERATORS) == set(TEMPLATES) == set(KNOB_SPACE)


# ------------------------------------------------------------- verification


@dataclass
class VerifyReport:
    ok: bool
    problems: list[str]
    n_gold_calls: int
    gold_tool_chars: int
    gold_db_hash: str


def user_lines(task: EpisodeTask) -> list[str]:
    lines = [task.opening]
    for key, v in task.profile.items():
        if key.startswith("recap_keys"):
            continue
        lines.extend([v] if isinstance(v, str) else [x for x in v if isinstance(x, str)])
    return lines


def verify_task(task: EpisodeTask, db: Optional[RetailDB] = None, base_hash: Optional[str] = None) -> VerifyReport:
    """Replay the gold chain on a private copy-on-write env and check reachability,
    id hygiene and the expected (non-)change of the end state. Sets nothing on
    `task`; the caller stores `gold_db_hash` if it accepts the task."""
    env = CowEnv(db if db is not None else base_db(), base_hash)
    start_hash = env.db_hash()
    problems: list[str] = []
    revealed: set[str] = set()
    for line in user_lines(task):
        ids = set(ID_RE.findall(line))
        leaked = {i for i in ids if not i.startswith("#W")}
        if leaked:
            problems.append(f"user line leaks non-order ids {sorted(leaked)}")
        revealed |= ids
    chars = 0
    for action in task.gold_actions:
        missing = set(ID_RE.findall(json.dumps(action["arguments"]))) - revealed
        if missing:
            problems.append(f"{action['name']} uses unrevealed ids {sorted(missing)}")
        result = env.execute(action["name"], action["arguments"])
        if not result.ok:
            problems.append(f"{action['name']} failed: {result.error}")
            continue
        out = result.value if isinstance(result.value, str) else json.dumps(result.value, default=str)
        chars += len(out)
        if not env.tool_mutates_state(action["name"]):
            revealed |= set(ID_RE.findall(out))
    gold_hash = env.db_hash()
    changed = gold_hash != start_hash
    if task.expect_no_write and changed:
        problems.append("refusal task's gold chain changed the db")
    if not task.expect_no_write and not changed:
        problems.append("gold chain left the db unchanged")
    return VerifyReport(not problems, problems, len(task.gold_actions), chars, gold_hash)


# --------------------------------------------------------------- generation


@dataclass
class GenerationReport:
    tasks: list[EpisodeTask]
    stats: dict[str, Counter]


def sample_knobs(template: str, rng: random.Random, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    knobs = {name: rng.choice(values) for name, values in KNOB_SPACE[template].items()}
    knobs.update(overrides or {})
    return knobs


def generate_tasks(
    per_template: int,
    seed: int = 0,
    *,
    templates: tuple[str, ...] = TEMPLATES,
    exclusions: Any = None,
    knob_overrides: Optional[dict[str, dict[str, Any]]] = None,
    db: Optional[RetailDB] = None,
    max_attempts_per_task: int = 20,
    max_consecutive_misses: int = 400,
    log: Optional[Callable[[str], None]] = None,
) -> GenerationReport:
    """Up to `per_template` verified, deduplicated, decontaminated tasks per
    template, deterministic in `seed`.

    `exclusions` is a `tau_forge.decontam.real_tasks.RealTaskExclusions`; when
    None it is loaded from tasks.json, which raises if that is unavailable --
    there is no way to generate without decontamination short of passing an
    explicit (e.g. empty, in tests) exclusion set. A real task whose gold
    actions are all reads has the untouched base db as its gold end state;
    that hash is shared by every refusal task and identifies nothing, so it is
    removed from the hash check (the user-id check still applies).

    Every candidate key seen -- accepted or rejected -- is remembered, so the
    logged counts are distinct candidates, and a template whose db-limited
    space is used up (modify_payment: ~100 eligible pending orders, because
    280 of 423 belong to users with one payment method) stops after
    `max_consecutive_misses` attempts in a row produce nothing new."""
    if exclusions is None:
        from tau_forge.decontam.real_tasks import load_real_task_exclusions

        exclusions = load_real_task_exclusions()
    log = log or (lambda msg: print(msg, file=sys.stderr))
    db = db if db is not None else base_db()
    ctx = _Ctx.build(db)
    base_hash = RetailEnv(db=db).db_hash()
    real_hashes = set(exclusions.gold_db_hashes) - {base_hash}
    tasks: list[EpisodeTask] = []
    stats: dict[str, Counter] = {}
    seen: set[tuple] = set()
    for template in templates:
        rng = random.Random(f"{seed}/{template}")
        st = stats[template] = Counter()
        produced = misses = 0
        for _ in range(per_template * max_attempts_per_task):
            if produced >= per_template:
                break
            if misses >= max_consecutive_misses:
                st["exhausted"] = 1
                break
            st["attempts"] += 1
            misses += 1
            knobs = sample_knobs(template, rng, (knob_overrides or {}).get(template))
            task = GENERATORS[template](ctx, rng, knobs)
            if task is None:
                st["no_candidate"] += 1
                continue
            if task.dedupe_key in seen:
                st["duplicate"] += 1
                continue
            seen.add(task.dedupe_key)
            if any(u in exclusions.user_ids for u in task.involved_users):
                st["decontam_user"] += 1
                continue
            report = verify_task(task, db, base_hash)
            if not report.ok:
                st["verify_failed"] += 1
                continue
            if report.gold_db_hash in real_hashes:
                st["decontam_gold_hash"] += 1
                continue
            task.gold_db_hash = report.gold_db_hash
            task.id = f"ep_{template}_s{seed}_{produced:05d}"
            tasks.append(task)
            produced += 1
            misses = 0
            st["accepted"] += 1
        log(
            f"[episodes.generate] {template}: accepted={st['accepted']}/{per_template} "
            f"decontam_user={st['decontam_user']} decontam_gold_hash={st['decontam_gold_hash']} "
            f"duplicate={st['duplicate']} verify_failed={st['verify_failed']} no_candidate={st['no_candidate']}"
            + (" (candidate space exhausted)" if st["exhausted"] else "")
        )
    return GenerationReport(tasks, stats)


def write_jsonl(tasks: list[EpisodeTask], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")


def read_jsonl(path: str | Path) -> list[EpisodeTask]:
    with open(path) as f:
        return [EpisodeTask.from_dict(json.loads(line)) for line in f if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Generate verified multi-step episode tasks from db.json.")
    p.add_argument("--per-template", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--templates", default=",".join(TEMPLATES))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "episodes" / "episodes_s0.jsonl"))
    args = p.parse_args(argv)
    report = generate_tasks(args.per_template, args.seed, templates=tuple(args.templates.split(",")))
    write_jsonl(report.tasks, args.out)
    print(f"[episodes.generate] wrote {len(report.tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
