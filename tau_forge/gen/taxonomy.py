"""Phase 2 generation taxonomy: 5 task categories x 6 domain sub-themes = 30 cells.

Sub-themes are grounded in the real tau2-bench retail domain -- its actual 50
product types, tools.py, and policy.md -- rather than the illustrative theme
names in the original task spec. In particular "subscription/recurring-order
edge cases" was dropped: this domain has no subscription concept anywhere in
tools.py, policy.md, or db.json (single one-off orders only), so a theme built
around it would push the generator toward inventing behavior no tool supports.
Replaced with two mechanic-flavored themes (order_state_confusion,
identity_and_order_lookup) that are richly grounded in real policy rules instead,
keeping the 5-6-themes / ~25-30-cells shape the spec calls for.
"""

from __future__ import annotations

CATEGORIES: dict[str, str] = {
    "happy_path": (
        "A clean, single-request case solvable in one (or a short, obviously "
        "necessary sequence of) tool call(s) given the context. No ambiguity, "
        "no policy conflict, nothing missing."
    ),
    "requires_earlier_context": (
        "The correct action is only recoverable by using specific information "
        "stated in an earlier turn (an order id, an item choice, a stated "
        "preference) that the current user message does not repeat. An agent "
        "that only reads the current message cannot get this right."
    ),
    "ambiguous": (
        "The user's request is genuinely underspecified given everything in "
        "the context so far -- e.g. they have two eligible orders and don't say "
        "which, or the request could map to more than one tool/action. The "
        "correct agent behavior is to ask a clarifying question, not guess. "
        "expected_tool_calls must be empty, and ambiguity_note must explain "
        "exactly what's underspecified."
    ),
    "policy_violation": (
        "The user asks for something policy.md explicitly disallows (e.g. "
        "acting on someone else's order, cancelling a non-pending order, a "
        "cancellation reason outside the two allowed values, modifying an "
        "order a second time after it was already modified, a mutating action "
        "without confirmation). The correct action is a policy-compliant "
        "refusal/explanation -- expected_tool_calls must be empty, unless the "
        "correct first move is a READ tool needed to establish the state that "
        "makes the refusal correct (e.g. checking order status to discover "
        "it's not pending), in which case include just that READ call."
    ),
    "out_of_scope": (
        "The request is outside the agent's tool set entirely -- nothing in "
        "tools.py can address it, and it isn't a policy violation of an "
        "in-scope action, it's just not something this agent does at all. The "
        "correct action is exactly one call to transfer_to_human_agents."
    ),
}

THEMES: dict[str, str] = {
    "electronics_returns_exchanges": (
        "Returns and exchanges of electronics/appliances/small-tech products "
        "(e.g. Laptop, Smartphone, Tablet, Headphones, Smart Watch, Vacuum "
        "Cleaner, Espresso Machine, Digital Camera -- draw from the real "
        "catalog, don't invent product names). These tend to be higher-priced, "
        "which makes gift-card-balance-sufficiency edge cases natural here."
    ),
    "apparel_footwear_exchanges": (
        "Exchanges of apparel/footwear/wearable-accessory products (e.g. "
        "T-Shirt, Sneakers, Hiking Boots, Running Shoes, Fleece Jacket, "
        "Sunglasses, Cycling Helmet) where the new item must be the same "
        "product but a different option (size/color/style) -- exercise the "
        "'same product type, different variant, availability required' "
        "constraint from policy.md and modify/exchange tools."
    ),
    "address_payment_modification": (
        "modify_pending_order_address / modify_pending_order_payment / "
        "modify_user_address mechanics: default user address vs. one order's "
        "shipping address, the single-payment-history precondition, gift card "
        "balance checks, and the immediate-vs-5-7-business-days refund-timing "
        "rule depending on payment method type."
    ),
    "order_state_confusion": (
        "Scenarios that hinge on correctly reading order.status before acting: "
        "pending vs. delivered vs. already 'pending (item modified)' vs. "
        "already 'exchange requested'/'return requested' vs. cancelled. The "
        "richest source of same-tool-family distractors -- cancel vs. "
        "modify-items vs. modify-address vs. exchange vs. return all look "
        "superficially similar but only one is valid for a given order's "
        "actual current status."
    ),
    "identity_and_order_lookup": (
        "Authentication and disambiguation mechanics: find_user_id_by_email "
        "vs. find_user_id_by_name_zip, a user with multiple orders where the "
        "current request must resolve to the right one, and the 'one user per "
        "conversation, refuse requests about anyone else' boundary."
    ),
    "damaged_or_defective_item_narratives": (
        "The user describes a damaged, defective, or wrong item received. "
        "tools.py has NO 'damage'/'defect' reason code anywhere -- the correct "
        "action still routes through return_delivered_order_items or "
        "exchange_delivered_order_items exactly as written, with no extra "
        "argument invented for the reason. A model that hallucinates a reason "
        "field here should score poorly in Phase 4 -- a good source of "
        "realistic hallucination pressure."
    ),
}


def all_cells() -> list[tuple[str, str]]:
    """All 30 (category, theme) cells in a fixed, stable order."""
    return [(category, theme) for category in CATEGORIES for theme in THEMES]


def cell_id(category: str, theme: str) -> str:
    return f"{category}__{theme}"
