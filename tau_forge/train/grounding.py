"""Renders the lookups a scenario's `prior_turns` narrate as real tool turns.

Every scenario's `prior_turns` is user/assistant prose only. When the narrative
says the agent already looked something up ("I've located your account... order
#W7736708 is delivered and includes the red Headphones"), the *result* of that
lookup -- the item id, the payment method id, the user id -- never reaches the
prompt. Measured over the corpus: 103 of the 150 scenarios whose gold is a
write tool needed at least one id argument that appears nowhere in the prompt
text. The policy model can only hallucinate it (schema-valid call that raises:
flat 0.2 across the whole group) or go and fetch it (a lookup, graded
`wrong_tool`: flat 0.0) -- either way a zero-variance group and no gradient,
independent of how good the model is.

This module replays those lookups against the same `db.json` snapshot the
reward function grades against and inserts them where the narrative implies
they happened, as native assistant `tool_calls` + `tool` result messages --
the same shape tau2 itself feeds the agent at evaluation time
(`tau2.utils.llm_utils.to_litellm_messages`), with the result serialised the
way `tau2.environment.Environment.to_json_str` serialises it.

What gets inserted, and where:
  * authentication -- the first user turn carrying a real account email, or a
    name + zip matching exactly one user, gets `find_user_id_by_email` /
    `find_user_id_by_name_zip` + `get_user_details` before the next assistant
    turn. When the prose never shows identifying details but an assistant turn
    claims the account is open (authentication happened off-screen), only
    `get_user_details` is inserted, before that assistant turn.
  * orders -- every order id the prose mentions gets `get_order_details`
    before the first assistant turn following both its mention and
    authentication. The gold order, if the prose never names it but the user
    is authenticated, is looked up before the last assistant turn.
  * products -- the product behind each gold `new_item_ids` entry gets
    `get_product_details` before the last assistant turn (the turn that
    narrates the variant choice).

What is never inserted:
  * anything mentioned only in the final `user_message` -- there is no
    assistant turn after it for a lookup to precede; fetching it is the
    policy's own job this turn.
  * a call identical in name to the gold call on the same record (a gold
    `get_order_details(#W1)` never sees `get_order_details(#W1)` in context),
    and no authentication lookups at all when the gold *is* authentication.
  * any lookup when no authentication signal exists -- grounding must not
    authenticate a user the scenario deliberately left unauthenticated.
  * a lookup that fails against the db (it is dropped, not shown as an error).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from tau2.domains.retail.data_model import RetailDB

from tau_forge.envs.retail import RetailEnv

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
ZIP_RE = re.compile(r"\b\d{5}\b")
ORDER_RE = re.compile(r"#?\bW\d{7}\b")
USER_ID_RE = re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")
# An assistant turn asserting it already has the account -- authentication
# happened before the transcript starts, or in a turn the prose skipped.
ACCOUNT_OPEN_RE = re.compile(
    r"\b(located|found|pulled up|verified|authenticated|identified|have|opened|confirmed|matched)\b"
    r"[^.?!]{0,40}\b(your|the)\s+(account|profile)\b"
    r"|\byour account\s*\(|\b(i have|i've got) your account\b"
    r"|\b(you're|you are) (now )?(verified|authenticated)\b"
    r"|\b(confirmed|verified|authenticated) your identity\b"
    r"|\bidentity (is |has been )?(verified|confirmed)\b",
    re.IGNORECASE,
)


def _norm_order(raw: str) -> str:
    return "#" + raw.lstrip("#")


def _identify(text: str, db: RetailDB) -> Optional[tuple[str, dict[str, Any]]]:
    """(user_id, authentication call args) if `text` carries a real account
    email, or a name + zip that pins down exactly one user."""
    lowered = text.lower()
    for email in EMAIL_RE.findall(text):
        email = email.rstrip(".").lower()
        for uid, user in db.users.items():
            if user.email.lower() == email:
                return uid, {"tool": "find_user_id_by_email", "args": {"email": user.email}}
    for z in ZIP_RE.findall(text):
        matches = [
            (uid, u)
            for uid, u in db.users.items()
            if u.address.zip == z
            and u.name.first_name.lower() in lowered
            and u.name.last_name.lower() in lowered
        ]
        if len(matches) == 1:
            uid, u = matches[0]
            return uid, {
                "tool": "find_user_id_by_name_zip",
                "args": {"first_name": u.name.first_name, "last_name": u.name.last_name, "zip": z},
            }
    return None


def _orders_in(text: str, db: RetailDB) -> list[str]:
    seen: list[str] = []
    for raw in ORDER_RE.findall(text):
        oid = _norm_order(raw)
        if oid in db.orders and oid not in seen:
            seen.append(oid)
    return seen


def _owner_of_orders(order_ids: list[str], db: RetailDB) -> Optional[str]:
    owners = {db.orders[o].user_id for o in order_ids if o in db.orders}
    return owners.pop() if len(owners) == 1 else None


def plan_lookups(scenario: dict[str, Any], db: RetailDB) -> dict[int, list[tuple[str, dict[str, Any]]]]:
    """Map from prior-turn index -> lookups to insert immediately *before*
    that (assistant) turn. Pure planning, no execution."""
    turns = scenario.get("prior_turns", [])
    asst_idx = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    if not asst_idx:
        return {}

    calls = scenario.get("expected_tool_calls") or []
    gold_name = calls[0]["name"] if calls else None
    gold_args = calls[0].get("arguments", {}) if calls else {}

    def next_asst(after: int) -> Optional[int]:
        return next((i for i in asst_idx if i > after), None)

    plan: dict[int, list[tuple[str, dict[str, Any]]]] = {}

    def add(at: Optional[int], tool: str, args: dict[str, Any]) -> None:
        if at is None:
            return
        if gold_name == tool and all(gold_args.get(k) == v for k, v in args.items()):
            return  # never pre-answer the gold call
        if any((tool, args) == existing for slot in plan.values() for existing in slot):
            return
        plan.setdefault(at, []).append((tool, args))

    # --- authentication ---------------------------------------------------
    user_id: Optional[str] = None
    auth_at: Optional[int] = None
    if gold_name not in ("find_user_id_by_email", "find_user_id_by_name_zip"):
        for i, t in enumerate(turns):
            if t["role"] != "user":
                continue
            ident = _identify(t["content"], db)
            if ident and next_asst(i) is not None:
                user_id, auth = ident
                auth_at = next_asst(i)
                add(auth_at, auth["tool"], auth["args"])
                break
        if user_id is None:
            # Off-screen authentication: an assistant turn says the account is
            # open, without the user ever showing identifying details here.
            for i in asst_idx:
                content = turns[i]["content"]
                if ACCOUNT_OPEN_RE.search(content) or USER_ID_RE.search(content):
                    mentioned = _orders_in(" ".join(t["content"] for t in turns), db)
                    uid_hits = [u for u in USER_ID_RE.findall(content) if u in db.users]
                    gold_owner = None
                    if gold_args.get("order_id") in db.orders:
                        gold_owner = db.orders[gold_args["order_id"]].user_id
                    user_id = (
                        (uid_hits[0] if uid_hits else None)
                        or (gold_args.get("user_id") if gold_args.get("user_id") in db.users else None)
                        or gold_owner
                        or _owner_of_orders(mentioned, db)
                    )
                    if user_id is not None:
                        auth_at = i
                    break
    if user_id is None or auth_at is None:
        return plan
    add(auth_at, "get_user_details", {"user_id": user_id})

    # --- orders mentioned in the prose ------------------------------------
    for i, t in enumerate(turns):
        for oid in _orders_in(t["content"], db):
            if db.orders[oid].user_id != user_id:
                continue  # one user per conversation; never leak another account
            if i < auth_at:
                at = auth_at  # named before authentication: looked up right after it
            elif t["role"] == "assistant":
                at = i  # the assistant names it in this very turn: it looked it up first
            else:
                at = next_asst(i)
            add(at, "get_order_details", {"order_id": oid})

    # --- gold entities the prose never names ------------------------------
    last_asst = asst_idx[-1] if asst_idx[-1] >= auth_at else None
    gold_order = gold_args.get("order_id")
    if gold_order in db.orders and db.orders[gold_order].user_id == user_id:
        add(last_asst, "get_order_details", {"order_id": gold_order})
    for new_item in gold_args.get("new_item_ids", []) or []:
        for pid, product in db.products.items():
            if new_item in product.variants:
                add(last_asst, "get_product_details", {"product_id": pid})
                break
    return plan


def _tool_turns(env: RetailEnv, lookups: list[tuple[str, dict[str, Any]]], start: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k, (tool, args) in enumerate(lookups):
        result = env.execute(tool, args)
        if not result.ok:
            continue
        call_id = f"call_{start + k}"
        out.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args)},
                    }
                ],
            }
        )
        content = result.value if isinstance(result.value, str) else json.dumps(result.value, default=str)
        out.append({"role": "tool", "tool_call_id": call_id, "name": tool, "content": content})
    return out


def grounded_prior_turns(scenario: dict[str, Any], env: RetailEnv) -> list[dict[str, Any]]:
    """`prior_turns` with the narrated lookups inserted as real tool turns.
    Only READ tools are ever executed, so `env.db` is not mutated."""
    plan = plan_lookups(scenario, env.db)
    messages: list[dict[str, Any]] = []
    n_calls = 0
    for i, turn in enumerate(scenario.get("prior_turns", [])):
        if i in plan:
            tool_turns = _tool_turns(env, plan[i], n_calls)
            n_calls += len(tool_turns) // 2
            messages.extend(tool_turns)
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages
