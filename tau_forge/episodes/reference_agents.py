"""Scripted reference policies: the oracle and the near-misses a 4B policy
plausibly samples.

The ORACLE resolves every id only from tool outputs it has seen in the
episode, and identity/order/payment facts only from what the scripted user has
actually said; it reads `task.hidden` for the natural-language targets alone
(which product, which option change), never for an id. So an oracle score of
1.0 proves the task is solvable through the protocol. Each near-miss mode
changes one decision, to show the reward separates them:

  no_confirm         writes without a recap + yes
  wrong_variant      exchanges to an available variant other than the asked one
  halluc_pm          invents a payment method id
  skip_auth          looks up the order and its owner instead of authenticating
  comply             does what the user asked even when policy forbids it
  giftcard_fallback  return_fallback: refunds to a gift card without asking
  transfer           transfers to a human immediately

Like any policy here, an agent is `callable(messages) -> assistant text`, and
emits tool calls as `<tool_call>{json}</tool_call>` so they go through the
same parser a model's completions do.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from tau_forge.episodes.generate import pm_phrase
from tau_forge.episodes.task import ORDER_ID_RE, TRANSFER_TOOL, EpisodeTask
from tau_forge.episodes.user import YES

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
MODES = (
    "oracle", "no_confirm", "wrong_variant", "halluc_pm", "skip_auth", "comply", "giftcard_fallback", "transfer",
)


def call(name: str, arguments: dict[str, Any]) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def tool_outputs(messages: list[dict[str, Any]], tool: str) -> list[tuple[dict[str, Any], Any]]:
    """(arguments, parsed output) for every successful call of `tool`."""
    args_by_id: dict[str, dict[str, Any]] = {}
    out = []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            args_by_id[tc["id"]] = json.loads(tc["function"]["arguments"])
        if m["role"] == "tool" and m.get("name") == tool and not m["content"].startswith("Error"):
            try:
                value = json.loads(m["content"])
            except json.JSONDecodeError:
                value = m["content"]
            out.append((args_by_id.get(m.get("tool_call_id"), {}), value))
    return out


class ReferenceAgent:
    def __init__(self, task: EpisodeTask, mode: str = "oracle"):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.t, self.mode = task, mode
        self.wrote = False
        self.denied = False

    # ---- conversation helpers ---------------------------------------------

    def user_msgs(self, messages) -> list[str]:
        """The user's messages since this agent's request started (`user_from`: how many to skip; a
        composite task's later sub-requests start mid-conversation)."""
        return [m["content"] for m in messages if m["role"] == "user"][getattr(self, "user_from", 0):]

    def said(self, messages, key: str) -> Optional[str]:
        """The user line from profile pool `key` the user has said, if any."""
        pool = self.t.profile.get(key) or []
        for line in self.user_msgs(messages):
            if any(p in line for p in pool):
                return line
        return None

    def user_id(self, messages) -> Optional[str]:
        for tool in ("find_user_id_by_email", "find_user_id_by_name_zip"):
            outs = tool_outputs(messages, tool)
            if outs:
                return outs[-1][1]
        return None

    def user_details(self, messages) -> Optional[dict[str, Any]]:
        outs = tool_outputs(messages, "get_user_details")
        return outs[-1][1] if outs else None

    def order(self, messages, oid: str) -> Optional[dict[str, Any]]:
        return next((o for a, o in tool_outputs(messages, "get_order_details") if a.get("order_id") == oid), None)

    def product(self, messages, pid: str) -> Optional[dict[str, Any]]:
        return next((p for a, p in tool_outputs(messages, "get_product_details") if a.get("product_id") == pid), None)

    def item_names(self, messages, oid: str) -> str:
        """The order's item names, as a recap names them: a product-hint user
        recognises its order by a product, not by an id it never heard."""
        return ", ".join(dict.fromkeys(i["name"] for i in self.order(messages, oid)["items"]))

    def recap(self, summary: str) -> str:
        return f"To confirm, I will {summary}. Do you want me to proceed? (yes/no)"

    def ready(self, messages) -> bool:
        if self.mode == "no_confirm":
            return True
        last = self.user_msgs(messages)[-1]
        return last in YES

    def write(self, name: str, arguments: dict[str, Any]) -> str:
        self.wrote = True
        return call(name, arguments)

    # ---- main policy ------------------------------------------------------

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        t = self.t
        if self.mode == "transfer":
            return call(TRANSFER_TOOL, {"summary": "Customer request: " + t.opening})
        if self.wrote:
            return "All done -- your request has been processed. Is there anything else I can help with?"
        text = "\n".join(self.user_msgs(messages))
        uid = self.user_id(messages)
        # skip_auth needs an order id to skip to; on a product-hint task it
        # authenticates like the oracle (the tests skip that combination).
        if uid is None and self.mode == "skip_auth" and ORDER_ID_RE.search(text):
            oid = ORDER_ID_RE.findall(text)[-1]
            od = self.order(messages, oid)
            if od is None:
                return call("get_order_details", {"order_id": oid})
            uid = od["user_id"]
        elif uid is None:
            if m := EMAIL_RE.search(text):
                return call("find_user_id_by_email", {"email": m.group(0).rstrip(".")})
            h = t.hidden
            if all(h[k] in text for k in ("first_name", "last_name", "zip")):
                return call("find_user_id_by_name_zip", {"first_name": h["first_name"], "last_name": h["last_name"], "zip": h["zip"]})
            return "Before I can help, could you please verify your identity with your email, or your full name and zip code?"
        if self.user_details(messages) is None:
            return call("get_user_details", {"user_id": uid})
        if t.template == "foreign_order_refusal":
            return self.foreign(messages, text)
        oid = self.find_order(messages, text)
        if not oid.startswith("#W"):
            return oid  # a tool call or a question
        return getattr(self, t.template)(messages, text, oid)

    def find_order(self, messages, text: str) -> str:
        ids = ORDER_ID_RE.findall(text)
        if ids:
            oid = ids[-1]  # latest mention wins: handles a late correction
            return oid if self.order(messages, oid) is not None else call("get_order_details", {"order_id": oid})
        h, t = self.t.hidden, self.t.template
        if t == "cancel":
            names = [h["target_hint"] if self.said(messages, "correction") else h["first_hint"]]
        elif t in ("exchange", "modify_items"):
            names = [n for n, _ in h["targets"]]
        elif t == "return_fallback":
            names = h["item_names"]
        elif t == "status_refusal":
            names = [h["hint"]] if h.get("hint") else h["item_names"]
        else:
            names = [h["hint"]]
        status = "pending" if t in ("cancel", "modify_payment", "modify_items") else "delivered"
        if t == "status_refusal":
            status = h["status"]
        for oid in self.user_details(messages)["orders"]:
            od = self.order(messages, oid)
            if od is None:
                return call("get_order_details", {"order_id": oid})
            if od["status"] == status and all(any(i["name"] == n for i in od["items"]) for n in names):
                return oid
        return "I couldn't find that order -- could you give me the order id?"

    # ---- templates --------------------------------------------------------

    def cancel(self, messages, text: str, oid: str) -> str:
        if re.search(r"no longer need|don't need|use for it", text):
            reason = "no longer needed"
        elif re.search(r"mistake|accident", text):
            reason = "ordered by mistake"
        else:
            return "May I ask the reason for the cancellation?"
        if not self.ready(messages):
            return self.recap(f"cancel order {oid} ({self.item_names(messages, oid)}; status "
                              f"{self.order(messages, oid)['status']}) with reason '{reason}'")
        return self.write("cancel_pending_order", {"order_id": oid, "reason": reason})

    def modify_items(self, messages, text: str, oid: str) -> str:
        return self.exchange(messages, text, oid, tool="modify_pending_order_items", verb="modify")

    def exchange(self, messages, text: str, oid: str, tool: str = "exchange_delivered_order_items",
                 verb: str = "exchange") -> str:
        od = self.order(messages, oid)
        targets = [tuple(x) for x in self.t.hidden["targets"]]
        if self.t.hidden.get("correction") and self.said(messages, "correction"):
            targets[0] = tuple(self.t.hidden["correction"])
        item_ids, new_ids = [], []
        for name, diff in targets:
            it = next(i for i in od["items"] if i["name"] == name)
            prod = self.product(messages, it["product_id"])
            if prod is None:
                return call("get_product_details", {"product_id": it["product_id"]})
            want = {**it["options"], **diff}
            cands = [v for v in prod["variants"].values() if v["available"] and v["item_id"] != it["item_id"]]
            if self.mode == "wrong_variant":
                pick = next(v for v in cands if v["options"] != want)
            else:
                pick = next(v for v in cands if v["options"] == want)
            item_ids.append(it["item_id"])
            new_ids.append(pick["item_id"])
        methods = self.user_details(messages)["payment_methods"].values()
        pm = next((p["id"] for p in methods if pm_phrase(p) in text), None)
        if pm is None:
            return "Which payment method would you like to use for the price difference?"
        if self.mode == "halluc_pm":
            pm = "credit_card_1234567"
        if not self.ready(messages):
            return self.recap(f"{verb} items {item_ids} in order {oid} ({', '.join(n for n, _ in targets)}) for {new_ids}, using {pm}")
        return self.write(tool,
                          {"order_id": oid, "item_ids": item_ids, "new_item_ids": new_ids, "payment_method_id": pm})

    def return_fallback(self, messages, text: str, oid: str) -> str:
        od = self.order(messages, oid)
        items = [next(i for i in od["items"] if i["name"] == n)["item_id"] for n in self.t.hidden["item_names"]]
        methods = list(self.user_details(messages)["payment_methods"].values())
        orig = od["payment_history"][0]["payment_method_id"]
        if self.mode == "comply":
            pm = next(p["id"] for p in methods if p["source"] != "gift_card" and pm_phrase(p) in self.t.opening)
        elif self.mode == "giftcard_fallback":
            pm = next((p["id"] for p in methods if p["source"] == "gift_card"), orig)
        else:
            if not self.said(messages, "fallback") and not self.denied:
                self.denied = True
                return ("I'm sorry, but refunds can only go to the original payment method or an existing "
                        "gift card. How would you like to proceed?")
            pm = orig
        if not self.ready(messages):
            return self.recap(f"return items {items} ({', '.join(self.t.hidden['item_names'])}) from order {oid}, refunding {pm}")
        return self.write("return_delivered_order_items", {"order_id": oid, "item_ids": items, "payment_method_id": pm})

    def modify_payment(self, messages, text: str, oid: str) -> str:
        od = self.order(messages, oid)
        methods = list(self.user_details(messages)["payment_methods"].values())
        fallback = self.said(messages, "fallback")
        source_text = fallback if fallback else self.t.opening
        target = next(p for p in methods if pm_phrase(p) in source_text)
        amount = od["payment_history"][0]["amount"]
        if target["source"] == "gift_card" and target["balance"] < amount and self.mode != "comply":
            return (f"I'm sorry, but your gift card balance (${target['balance']}) is not enough to cover the order "
                    f"total (${amount}). Is there another payment method you'd like to use?")
        pm = "credit_card_1234567" if self.mode == "halluc_pm" else target["id"]
        if not self.ready(messages):
            return self.recap(f"change the payment method of order {oid} ({self.item_names(messages, oid)}) to {pm}")
        return self.write("modify_pending_order_payment", {"order_id": oid, "payment_method_id": pm})

    def status_refusal(self, messages, text: str, oid: str) -> str:
        od = self.order(messages, oid)  # find_order has read it
        req = self.t.hidden["request"]
        if self.mode not in ("comply", "no_confirm"):
            if req == "cancel":
                return (f"I'm sorry, but order {oid} is already {od['status']}, so it can't be cancelled -- "
                        "only pending orders can be cancelled.")
            return (f"I'm sorry, but order {oid} is {od['status']} and hasn't been delivered yet, so it can't be "
                    "returned -- only delivered orders can be returned.")
        if req == "cancel":
            name, args = "cancel_pending_order", {"order_id": oid, "reason": self.t.hidden["reason"]}
        else:
            items = [i["item_id"] for i in od["items"] if i["name"] in self.t.hidden["item_names"]]
            name, args = "return_delivered_order_items", {
                "order_id": oid, "item_ids": items, "payment_method_id": od["payment_history"][0]["payment_method_id"]}
        if not self.ready(messages):
            return self.recap(f"{name.replace('_', ' ')} for order {oid} ({self.item_names(messages, oid)})")
        return self.write(name, args)

    def foreign(self, messages, text: str) -> str:
        oid = ORDER_ID_RE.findall(text)[-1]
        # Every mode reads the order first: the refusal reward pays a denial
        # in full only after the agent checked whose order it is.
        od = self.order(messages, oid)
        if od is None:
            return call("get_order_details", {"order_id": oid})
        if self.mode not in ("comply", "no_confirm"):
            return (f"I'm sorry, but order {oid} is not on your account, and I can only help with your own orders. "
                    "They will need to contact us themselves.")
        if self.t.difficulty["request"] == "cancel":
            name, args = "cancel_pending_order", {"order_id": oid, "reason": "no longer needed"}
        else:
            items = [i["item_id"] for i in od["items"] if i["name"] in text]
            name, args = "return_delivered_order_items", {
                "order_id": oid, "item_ids": items, "payment_method_id": od["payment_history"][0]["payment_method_id"]}
        if not self.ready(messages):
            return self.recap(f"{name.replace('_', ' ')} for order {oid}")
        return self.write(name, args)

