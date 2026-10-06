"""What an agent's text turn DOES, for the scripted user (stage B).

`analyze(text, task, state)` turns one assistant text turn into `Semantics`,
whose fields are exactly the gold semantic labels of
`data/episodes/nlu_gold/labels.jsonl` (schema: the stage contract, "Semantic
NLU labels"): which pieces of information the agent asks the user for, whether
it asks consent to act now, the action it proposes or reports and whether that
action's details match the task, whether it refuses and why, and so on.
`ScriptedUser` (`user.py`) decides its reply from these semantics, never from
raw keyword hits on the whole turn.

Two backends implement the `Analyzer` protocol:

* `RulesAnalyzer` -- deterministic, sentence- and clause-level rules. This is
  what training and every test use: a GRPO group must differ only by the
  policy's samples, so the user has to reply identically to identical text.
  Measured against the 806 adjudicated gold turns by `scripts/nlu_eval.py`
  (every 5th turn is a held-out split nobody tuned on); `tests/test_nlu_gold.py`
  pins the held-out accuracies.
* `LLMAnalyzer` -- an OpenAI-compatible chat endpoint asked for the same JSON
  at temperature 0, with the label definitions and the task facts in the
  prompt, cached on disk by (model, prompt). For drift checks against the
  rules, selected with `make_analyzer({"backend": "llm", ...})`; never used in
  tests or training.

How the rules read a turn:

* The turn is split into sentences (markdown stripped; bullets and line breaks
  are sentence boundaries). A sentence ASKS something only if it is a question
  ("?") or an imperative request ("Please provide ...", "I'll need your ...",
  "Reply with yes ..."). Negated or informational sentences are not requests:
  "I cannot provide details about her order", "Please note that ...",
  "If you need anything else, let me know" (a sign-off).
* Information requests are read only from asking sentences. A sentence asking
  for the email / name + zip is an identity request -- of a third party when
  it asks for "her email" or "your sister's details".
* A confirmation request is an asking sentence that seeks consent to act
  ("Shall I proceed?", "Please confirm with yes", "Would you like me to cancel
  it?"); offers of other help ("Would you like me to transfer you?",
  "anything else?") and requests to confirm a piece of information ("confirm
  your email", "confirm the reason") are not.
* Refusals are recognised by meaning per clause: an inability / policy modal
  ("can't", "unable", "only ... can", "not allowed") together with an
  ownership object (another user / person / customer, the account holder,
  "your sister's order", "not yours", "the authenticated user's own orders",
  "one user per conversation"), a status object (delivered / processed, "only
  pending orders") or a constraint object (refund destination, gift card
  balance, the asked payment method). Questions about ownership ("Is this
  order under a different account?") and plans to check ("let me see whether
  it is linked to another customer") are not refusals.
* Details (reason, new options, payment, items) are extracted against the
  task's known values -- the reason enum, the option values of the target
  items' products in the catalogue, the user's payment methods (brand + last
  four, "ending in NNNN", PayPal, gift card, "original payment method", pm
  ids), the target order's item names -- and compared with what the task wants
  (`details_match_task`). Order ids other than the target, and products that
  only appear in the user's other orders, make `names_wrong_target`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from tau_forge.episodes.task import ORDER_ID_RE, base_db

INFO_KINDS = (
    "identity", "order_id", "reason", "payment_method", "all_items_complete",
    "option_choice", "item_choice", "order_choice", "other_info",
)
ACTIONS = (
    "cancel", "exchange", "return", "modify_payment", "modify_items", "modify_address",
    "modify_user_address", "transfer", "other",
)
REFUSALS = ("ownership", "status", "constraint", "other")
TEMPLATE_ACTION = {
    "cancel": "cancel",
    "exchange": "exchange",
    "return_fallback": "return",
    "modify_payment": "modify_payment",
}


# =============================================================================
# Result types
# =============================================================================


@dataclass
class ProposedAction:
    action: str
    names_target: bool = False
    names_wrong_target: bool = False
    details: dict[str, Any] = field(
        default_factory=lambda: {"reason": None, "new_options": None, "payment": None, "items": None}
    )
    details_match_task: Optional[bool] = None
    # Not part of the gold schema; for the user policy.
    names_current_target: bool = False  # the order the user means NOW (first order before a late correction)
    names_stale_target: bool = False    # the order a late correction retracted
    named_orders: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)  # "reason" | "options" | "payment" | "items"
    missing: list[str] = field(default_factory=list)     # details the recap leaves out
    payment_pm: Optional[str] = None  # resolved pm id, or "unknown"
    stale_options: bool = False       # states an option value a late correction retracted

    def to_label(self) -> dict[str, Any]:
        return {
            "action": self.action, "names_target": self.names_target,
            "names_wrong_target": self.names_wrong_target, "details": dict(self.details),
            "details_match_task": self.details_match_task,
        }


@dataclass
class Semantics:
    info_requests: list[str] = field(default_factory=list)
    identity_target: Optional[str] = None
    confirmation_request: bool = False
    proposed_action: Optional[ProposedAction] = None
    refusal: Optional[str] = None
    constraint_statement: bool = False
    offers_other_help_only: bool = False
    claims_action_done: bool = False
    narration_only: bool = False
    # Not part of the gold schema; for the user policy.
    offers_other_help: bool = False      # any other-help offer / sign-off question
    proceeds_anyway: bool = False        # offers to go ahead with the action in the same turn
    choice_orders: list[str] = field(default_factory=list)  # order ids offered as a choice
    constraint_about_payment: bool = False  # a constraint stated about the asked payment method / balance
    ownership_question: bool = False     # asks whether the order is someone else's
    payment_yesno_pm: Optional[str] = None  # a yes/no question about one payment method ("charge it to your PayPal?")

    def to_label(self) -> dict[str, Any]:
        return {
            "info_requests": list(self.info_requests),
            "identity_target": self.identity_target,
            "confirmation_request": self.confirmation_request,
            "proposed_action": self.proposed_action.to_label() if self.proposed_action else None,
            "refusal": self.refusal,
            "constraint_statement": self.constraint_statement,
            "offers_other_help_only": self.offers_other_help_only,
            "claims_action_done": self.claims_action_done,
            "narration_only": self.narration_only,
        }


@dataclass
class NLUState:
    """What the user has said so far that changes what the task wants."""

    correction_used: bool = False
    fallback_used: bool = False


# =============================================================================
# Task facts
# =============================================================================


@dataclass
class TaskFacts:
    template: str
    target_order: str
    expect_no_write: bool
    user_id: Optional[str]
    first_order: Optional[str]            # cancel with a late correction: the order first named
    give_order_id: bool                   # the user knows (and said) the order id
    user_orders: list[str]
    order_items: dict[str, list[str]]     # order id -> item names (user's orders + target)
    target_items: list[dict[str, Any]]    # name, product_id, item_id, options
    hint_products: list[str]              # products that name the target to the user
    first_hint_products: list[str]
    reason: Optional[str]
    exchange_targets: list[tuple[str, dict[str, str]]]
    exchange_correction: Optional[tuple[str, dict[str, str]]]
    payment_methods: dict[str, dict[str, Any]]
    original_pm: Optional[str]
    gold_pm: Optional[str]
    asked_pm: Optional[str]
    fallback_pm: Optional[str]
    request_items: list[str]              # items the user asks to return (return, foreign return)
    relation: Optional[str]               # foreign: "sister", "roommate", ...
    foreign_request: Optional[str]        # foreign: "cancel" | "return"
    product_options: dict[str, dict[str, set[str]]] = field(default_factory=dict)  # item name -> key -> values
    # Old (s0) correction wording "Everything else is right" adds the option change to the first one;
    # the current wording ("forget the change I asked for ... instead") replaces it.
    correction_additive: bool = False
    # modify_items: the "exchange" is a modification of a pending order's items, so "modify / change /
    # update the items" names the action too
    pending_item_change: bool = False

    def wanted_targets(self, state: Optional[NLUState]) -> list[tuple[str, dict[str, str]]]:
        targets = [tuple(t) for t in self.exchange_targets]
        if self.exchange_correction and state is not None and state.correction_used:
            name, diff = self.exchange_correction
            if self.correction_additive:
                targets = [(name, {**d, **diff}) if n == name else (n, d) for n, d in targets]
            else:
                targets = [(name, dict(diff)) if n == name else (n, d) for n, d in targets]
        return [(n, dict(d)) for n, d in targets]

    def retracted_options(self, state: Optional[NLUState]) -> list[tuple[str, dict[str, str]]]:
        if self.exchange_correction and state is not None and state.correction_used and not self.correction_additive:
            name = self.exchange_correction[0]
            return [(n, dict(d)) for n, d in self.exchange_targets if n == name]
        return []

    def current_order(self, state: Optional[NLUState]) -> str:
        if self.first_order and not (state is not None and state.correction_used):
            return self.first_order
        return self.target_order

    def current_hints(self, state: Optional[NLUState]) -> list[str]:
        if self.first_order and not (state is not None and state.correction_used):
            return self.first_hint_products
        return self.hint_products

    def wanted_pm(self, state: Optional[NLUState]) -> Optional[str]:
        """The payment method the user wants right now."""
        fb = state is not None and state.fallback_used
        if self.template == "return_fallback":
            return self.original_pm if fb else self.asked_pm
        if self.template == "modify_payment":
            return self.fallback_pm if (fb and self.fallback_pm) else self.asked_pm
        if self.template == "exchange":
            return self.gold_pm
        return None


def _pm_phrase(pm: dict[str, Any]) -> str:
    if pm["source"] == "credit_card":
        return f"my {pm['brand'].title()} ending in {pm['last_four']}"
    if pm["source"] == "paypal":
        return "my PayPal account"
    return "my gift card"


_RELATIONS = (
    "sister", "brother", "mother", "mom", "father", "dad", "roommate", "friend", "wife", "husband",
    "partner", "spouse", "son", "daughter", "cousin", "aunt", "uncle", "grandmother", "grandfather",
    "neighbor", "neighbour", "colleague", "coworker", "boss", "family member", "relative",
)


def _find_user(db, email: Optional[str], first: Optional[str], last: Optional[str], zip_: Optional[str]):
    for u in db.users.values():
        if email and u.email == email:
            return u
    for u in db.users.values():
        if first and u.name.first_name == first and u.name.last_name == last and u.address.zip == zip_:
            return u
    return None


def facts_from(task: Any) -> TaskFacts:
    """Facts from an `EpisodeTask` or a gold-turn task dict (`turns.jsonl`)."""
    if hasattr(task, "_nlu_facts"):
        return task._nlu_facts
    db = base_db()
    if isinstance(task, dict):
        hidden = task.get("hidden") or {}
        template = task.get("template") or ""
        target = task["target_order"]
        opening = task.get("opening", "")
        expect_no_write = bool(task.get("expect_no_write"))
        gw = task.get("gold_write") or {}
        gold_pm = (gw.get("arguments") or {}).get("payment_method_id")
        profile = task.get("profile_lines") or {}
        recap_keys = task.get("recap_keys") or []
        user = _find_user(db, hidden.get("email"), hidden.get("first_name"), hidden.get("last_name"), hidden.get("zip"))
        user_id = user.user_id if user else None
        give_order_id = bool(ORDER_ID_RE.search(opening))
        if not template:
            template = "foreign_order_refusal" if expect_no_write else ""
    else:
        hidden = task.hidden
        # modify_items is an exchange of item variants on a pending order: same request shape (items, new
        # options, a payment method for the difference), so the NLU reads it as an exchange
        template = "exchange" if task.template == "modify_items" else task.template
        target = task.target_order
        opening = task.opening
        expect_no_write = task.expect_no_write
        gold_pm = next((a["arguments"].get("payment_method_id") for a in task.gold_actions
                        if a["arguments"].get("payment_method_id")), None)
        profile = task.profile
        recap_keys = profile.get("recap_keys") or []
        user_id = task.user_id
        user = db.users.get(user_id)
        give_order_id = bool(task.difficulty.get("give_order_id", bool(ORDER_ID_RE.search(opening))))
    first_order = hidden.get("first_order") if hidden.get("first_order") not in (None, target) else None
    user_orders = list(user.orders) if user else []
    order_items: dict[str, list[str]] = {}
    for oid in set(user_orders) | {target} | ({first_order} if first_order else set()):
        o = db.orders.get(oid)
        if o is not None:
            order_items[oid] = [it.name for it in o.items]
    tord = db.orders.get(target)
    target_items = []
    product_options: dict[str, dict[str, set[str]]] = {}
    if tord is not None:
        for it in tord.items:
            target_items.append({"name": it.name, "product_id": it.product_id, "item_id": it.item_id,
                                 "options": dict(it.options)})
            prod = db.products.get(it.product_id)
            if prod is not None:
                opts: dict[str, set[str]] = {}
                for v in prod.variants.values():
                    for k, val in v.options.items():
                        opts.setdefault(k, set()).add(val)
                product_options[it.name] = opts
    original_pm = tord.payment_history[0].payment_method_id if tord is not None and tord.payment_history else None
    # Payment methods: the task user's, plus the target order owner's (foreign).
    pms: dict[str, dict[str, Any]] = {}
    owners = [user] if user else []
    if tord is not None and tord.user_id in db.users:
        owners.append(db.users[tord.user_id])
    for u in owners:
        for pid, pm in u.payment_methods.items():
            d = pm.model_dump()
            d["owner"] = u.user_id
            pms.setdefault(pid, d)
    own_pms = {k: v for k, v in pms.items() if v["owner"] == user_id}

    def by_phrase(phrase: Optional[str]) -> Optional[str]:
        if not phrase:
            return None
        hits = [pid for pid, pm in own_pms.items() if _pm_phrase(pm) == phrase]
        return hits[0] if len(hits) == 1 else None

    reason = hidden.get("reason")
    asked_pm = fallback_pm = None
    if template == "modify_payment":
        asked_pm = by_phrase(hidden.get("asked_pm_phrase"))
        tp = by_phrase(hidden.get("target_pm_phrase"))
        fallback_pm = tp if tp and tp != asked_pm else None
    elif template == "return_fallback":
        asked_pm = by_phrase(hidden.get("bad_pm_phrase"))
    elif template == "exchange":
        asked_pm = by_phrase(hidden.get("pm_phrase")) or gold_pm
    targets = [(n, dict(d)) for n, d in (hidden.get("targets") or [])]
    corr = hidden.get("correction")
    exchange_correction = (corr[0], dict(corr[1])) if corr else None
    relation = foreign_request = None
    request_items = list(hidden.get("item_names") or [])
    if template == "foreign_order_refusal":
        low = opening.lower()
        relation = next((r for r in _RELATIONS if re.search(rf"\b{r}\b", low)), None)
        foreign_request = "return" if re.search(r"\breturn|send back", low) else "cancel"
        if foreign_request == "cancel":
            reason = "ordered by mistake" if "mistake" in low else "no longer needed"
        if tord is not None:
            request_items = [it.name for it in tord.items if it.name.lower() in low]
    hint_products = [k for k in recap_keys if not k.startswith("#W")]
    if profile.get("recap_keys_after_correction"):
        hint_products = [k for k in profile["recap_keys_after_correction"] if not k.startswith("#W")]
    first_hints: list[str] = []
    if first_order:
        first_hints = [k for k in recap_keys if not k.startswith("#W")]
        if hidden.get("target_hint"):
            hint_products = [hidden["target_hint"]]
        if hidden.get("first_hint"):
            first_hints = [hidden["first_hint"]]
    if template == "exchange":
        hint_products = [n for n, _ in targets] or hint_products
    if template == "return_fallback":
        hint_products = request_items or hint_products
    facts = TaskFacts(
        template=template, target_order=target, expect_no_write=expect_no_write, user_id=user_id,
        first_order=first_order, give_order_id=give_order_id, user_orders=user_orders, order_items=order_items,
        target_items=target_items, hint_products=hint_products, first_hint_products=first_hints, reason=reason,
        exchange_targets=targets, exchange_correction=exchange_correction, payment_methods=pms,
        original_pm=original_pm, gold_pm=gold_pm, asked_pm=asked_pm, fallback_pm=fallback_pm,
        request_items=request_items, relation=relation, foreign_request=foreign_request,
        product_options=product_options,
        pending_item_change=(not isinstance(task, dict) and task.template == "modify_items"),
        correction_additive=bool(profile.get("correction")) and not any(
            re.search(r"\b(?:forget|drop)\b", c) for c in profile.get("correction") or []),
    )
    try:
        object.__setattr__(task, "_nlu_facts", facts) if not isinstance(task, dict) else None
    except Exception:  # noqa: BLE001 -- caching is best effort
        pass
    return facts


# =============================================================================
# Sentence segmentation
# =============================================================================

_MD_RE = re.compile(r"\*\*|__|`|^#+\s*|^\s*>\s*", re.M)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(]*[A-Z0-9#*])|\n+")


def clean(text: str) -> str:
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("—", " -- ").replace("–", "-")
    t = _MD_RE.sub("", t)
    t = _BULLET_RE.sub("", t)
    return t


def sentences(text: str) -> list[str]:
    out = []
    for s in _SPLIT_RE.split(clean(text)):
        s = s.strip()
        if s:
            out.append(s)
    return out


_PLAN_INTRO_RE = re.compile(
    r"\b(?:i(?:'ll| will)(?: (?:now|then|first|do the following|need to))?|here(?:'s| is| are)|the (?:details|summary|items?|options?)|"
    r"you (?:want|would like|'d like|wish)|details|summary|available options|options available|the following|steps?)\b[^:]{0,60}:\s*$",
    re.I,
)


@dataclass
class Unit:
    """One asking unit: `text` for consent detection, `info` for information requests."""

    text: str
    info: str
    confirm_details: bool = False  # "Please confirm:" followed by a list of action details


_DETAIL_ITEM_RE = re.compile(
    r"^[^:?]{1,40}:\s*\S|#W\d{7}|\b(?:will be|is|are|was|has been|remain)\b|"
    r"^(?:you (?:want|agree|would like|'d like|wish|are)|\u2705|exchange|return|refund|cancel|change|switch|update)\b", re.I)
_ASK_ITEM_RE = re.compile(r"\b(?:you(?:'d| would) like|you want|you prefer|your preferred|which|whether|if you|what)\b", re.I)


def request_units(text: str) -> tuple[list[str], list[Unit]]:
    """(all sentences, asking units). An asking unit is a request sentence; a
    request that ends with a colon ("Please confirm:", "I'll need:") is merged
    with the list items that follow it in the same paragraph, which are its
    object -- bare noun phrases ("The reason for cancellation") and questions
    among them are information requests, "Label: value" lines are action
    details for a consent request. List items after a plan / details intro
    ("I will:", "Here are the details:") are statements even when they start
    with a verb ("Verify her user ID.")."""
    all_sents: list[str] = []
    units: list[Unit] = []
    for para in re.split(r"\n\s*\n", clean(text)):
        ps = [x.strip() for x in _SPLIT_RE.split(para) if x and x.strip()]
        i = 0
        while i < len(ps):
            s0 = ps[i]
            all_sents.append(s0)
            header = bool(re.search(r"^[\w ]{0,30}\b(?:needed|required|info(?:rmation)? needed)\W*$", s0, re.I))
            asking_intro = bool(re.search(r"\b(?:please|kindly|confirm|provide|could you|can you|let me know)\b", s0, re.I)) or bool(
                re.search(r"\b(?:i|we)(?:'ll| will|'d| would)?(?: (?:just|also|first|still))? (?:need|require)\b(?! to)", s0, re.I))
            if (s0.endswith(":") or header) and (is_request(s0) or NEED_RE.search(s0) or header) and (
                    asking_intro or not _PLAN_INTRO_RE.search(s0)):
                items = []
                j = i + 1
                while j < len(ps) and len(items) < 6:
                    items.append(ps[j])
                    all_sents.append(ps[j])
                    j += 1
                def is_ask(x: str) -> bool:
                    return "?" in x or not _DETAIL_ITEM_RE.search(x) or (
                        bool(_ASK_ITEM_RE.search(x)) and not re.search(r"^you (?:want|agree|would like|'d like|wish)\b", x, re.I))
                asks = [x for x in items if is_ask(x)]
                details = [x for x in items if not is_ask(x)]
                plain_confirm = bool(re.search(
                    r"^(?:[\w']+[, ]+){0,3}(?:please |kindly |could you please |can you |now,? please |i need you to )?confirm(?: the following| these details|"
                    r" the details(?: below)?| below)?\s*:\s*$", s0, re.I))
                # a colon intro followed by questions is a question list: its consent words do not ask consent
                consent_text = s0 + " " + " ".join(items) if not asks else " ".join(items)
                units.append(Unit(consent_text, s0 + " " + " ".join(asks),
                                  confirm_details=bool(details) and plain_confirm and not asks))
                i = j
                continue
            if s0.endswith(":") and _PLAN_INTRO_RE.search(s0) and not asking_intro:
                j = i + 1
                while j < len(ps):
                    all_sents.append(ps[j])
                    if "?" in ps[j]:
                        units.append(Unit(ps[j], ps[j]))
                    j += 1
                i = j
                continue
            if is_request(s0) and not _negated_request(s0):
                units.append(Unit(s0, s0))
            i += 1
    return all_sents, units


# =============================================================================
# Lexicon
# =============================================================================

NEG = r"(?:can(?:no|')?t|cannot|unable to|not able to|won't|will not|am not|are not|is not|isn't|aren't|don't|do not|doesn't|does not|didn't|did not|never|not)"

# A question addressed to the user, or an imperative request.
REQUEST_RE = re.compile(
    r"^(?:(?:so|and|also|then|now|first|next|finally|lastly|additionally|meanwhile|alternatively|otherwise|"
    r"to (?:proceed|continue|confirm|assist|help)[^,]{0,40}|once (?:you|confirmed)[^,]{0,40}|"
    r"if (?:so|yes|you agree|this is correct|that's correct|everything is correct|correct)|in the meantime|"
    r"before (?:i|we) (?:proceed|continue|go ahead|do that|make)[^,]{0,40}|for (?:security|verification)[^,]{0,30}|"
    r"thank you|thanks|great|perfect|got it|sure|okay|ok|alright)[,:]?\s+)*"
    r"(?:please|kindly|could you|can you|would you|will you|may i|might i|let me know|tell me|share|provide|"
    r"send|give me|enter|specify|clarify|choose|select|pick|reply|respond|type|confirm|verify|indicate|"
    r"double-check|check)\b",
    re.I,
)
NEED_RE = re.compile(
    r"\b(?:i|we)(?:'ll| will|'d| would)?(?: (?:still|also|just|first|now|only|then|again|additionally|actually|really))*"
    r" (?:need|require|must have|must ask for|have to ask for)\b(?! to (?:check|verify that|look|locate|find|review|confirm whether))",
    re.I,
)
NEED_YOUR_RE = re.compile(r"\b(?:need|require)s?\b[^.?!]{0,30}\b(?:your|her|his|their|the (?:order|email|user|reason|payment))\b", re.I)
PLEASE_NOTE_RE = re.compile(r"\b(?:please|kindly) (?:note|be aware|keep in mind|remember|allow|expect|feel free|don't hesitate|do not hesitate)\b", re.I)
SIGNOFF_RE = re.compile(
    r"\b(?:if (?:you (?:have|need|want|require)|there(?:'s| is)) (?:anything|any (?:other |more |further |additional )?"
    r"(?:questions?|concerns?|requests?|issues?|help|assistance|needs?)|(?:other|more|further|additional) (?:questions?|concerns?|requests?|"
    r"help|assistance))|"
    r"(?:anything|something) else (?:i can|we can|you need|to help|i may)|"
    r"other (?:questions|requests|concerns)|further (?:assistance|help|questions)|any more (?:help|assistance|questions)|"
    r"assist(?:ance)? with (?:anything|something) else|feel free to (?:ask|reach|contact|let)|"
    r"have a (?:great|nice|good|wonderful)|need (?:any )?(?:more|further|additional) (?:help|assistance))",
    re.I,
)
OTHER_HELP_RE = re.compile(
    r"\b(?:transfer|human agent|human representative|live agent|representative|specialist|supervisor|"
    r"anything else|something else|other (?:orders?|requests?|questions?|options?|issues?)|another (?:order|request)|"
    r"your own (?:orders?|account)|under your (?:own )?(?:name|account)|own orders?|"
    r"explore (?:other|alternative)|alternatives?|different request|else (?:i|we) can)\b",
    re.I,
)

IDENTITY_RE = re.compile(
    r"\b(?:e-?mail(?: address)?|zip(?: ?code)?|postal code|post code|first and last name|full name|first name|"
    r"last name|your name|her name|his name|their name|name and (?:zip|postal|email)|user ?id|"
    r"verify (?:your|her|his|their|the (?:caller|account holder|customer))(?:'s)? (?:identity|account|details|information|info)|"
    r"(?:authenticate|verify|identify) (?:you|yourself|her|him|them|your identity|her identity|his identity|their identity)|"
    r"(?:your|her|his|their) identity|who (?:am i|i am) (?:speaking|talking|chatting) (?:with|to)|"
    r"(?:locate|find|look up|pull up) (?:your|her|his|their) (?:account|profile|user))\b",
    re.I,
)
THIRD_PARTY_RE = re.compile(
    r"\b(?:her|his|their|(?:your )?(?:" + "|".join(_RELATIONS) + r")(?:'s)?|the account (?:holder|owner)(?:'s)?|"
    r"the (?:other|order) (?:owner|user|customer)(?:'s)?|the person who (?:placed|made|owns)|"
    r"the (?:account|person|user|customer) (?:that|who|which) (?:placed|made|owns))\b",
    re.I,
)
ORDER_ID_ASK_RE = re.compile(
    r"\b(?:order (?:id|number|#(?!w?\d)|no\.)|order's id|id of the order|number of the order|which order|what order|"
    r"which (?:one|of (?:these|those|them))\b(?=[^.?!]{0,40}\border)|which of (?:your|the) orders|"
    r"which (?:one|order) (?:contains|has|includes|had|included|is it)|"
    r"(?:which|what|specify|provide|identify|confirm)\b[^.?!]{0,20}\bthe order (?:you(?:'d| would)? (?:like|want|mean|are referring))|"
    r"(?:more )?details about the order|identify the (?:correct )?order|locate the (?:correct )?order|"
    r"(?:order|delivery) date)\b",
    re.I,
)
REASON_ASK_RE = re.compile(
    r"\b(?:reason|why|no longer needed\b.{0,40}\bordered by mistake|ordered by mistake\b.{0,40}\bno longer needed|"
    r"no longer need it\b.{0,40}\bmistake)",
    re.I,
)
PAYMENT_ASK_RE = re.compile(
    r"\b(?:(?:which|what) (?:payment|card|method|account|form of payment)|"
    r"(?:provide|confirm|specify|choose|select|let me know|tell me|share|indicate|give me|know|clarify|decide|verify)(?: \w+){0,4} "
    r"(?:payment method|method of payment|form of payment|refund method|payment option|payment method id)\b(?! to your)|"
    r"(?:a different|another|alternative|other|an alternative) (?:payment (?:method|option)|card|method of payment|form of payment)|"
    r"(?:payment method|method|card|account) (?:you(?:'d| would) like|you want|you prefer|to use|for (?:the|any) (?:price )?difference|"
    r"for (?:the|your) refund)|"
    r"how (?:would|do|will) you (?:like|want|prefer) to (?:pay|be refunded|receive|get|cover|handle)|"
    r"where (?:would|do) you (?:like|want) (?:the|your) (?:refund|money)|where should (?:the|your|i send the) (?:refund|money|difference)|"
    r"refund destination|(?:would|do) you (?:like|want) (?:the|your) refund (?:to go|sent|issued|processed)|"
    r"(?:would|do) you (?:like|want|prefer) (?:to use )?(?:your |the |my )?(?:paypal|gift card|credit card|visa|mastercard|amex|discover|original)"
    r"\b[^.?!]{0,60}\bor\b|"
    r"\b(?:paypal|gift card|credit card|visa|mastercard|card)\b[^.?!]{0,40}\bor\b[^.?!]{0,40}\b(?:paypal|gift card|credit card|visa|mastercard|card|"
    r"original|another|different|other)(?: (?:payment|method|card))?|which (?:one|would you prefer|do you prefer)\b[^.?!]{0,30}\b(?:pay|refund|card)|"
    r"(?:a|another|alternative|different|new) (?:payment method|card|method of payment)(?: you(?:'d| would) like)?(?: to use)?\??$|"
    r"(?:charge|bill) (?:the|any|it)\b[^.?!]{0,40}\bto\b[^.?!]{0,30}\?|use (?:for|to pay for) (?:the|any) (?:price )?difference|"
    r"what would you like to (?:change|switch) (?:the payment|it) to)",
    re.I,
)
ALL_ITEMS_ASK_RE = re.compile(
    r"\b(?:all (?:of )?the items|all items|any other items?|(?:no|any) other (?:items?|changes|adjustments|modifications|products)|"
    r"other items (?:to|you)|(?:anything|something) else (?:you(?:'d| would) like )?to "
    r"(?:exchange|return|modify|change|swap|send back)|(?:anything|something) else (?:from|in) (?:this|the|that) order|"
    r"(?:is that|are those|is this|are these) (?:all|everything|the only)|(?:the )?only (?:item|items|one|thing)s? (?:you|to)|the only item\b|"
    r"everything you (?:want|would like|'d like)|(?:any|other) (?:additional|more) items?|besides the|in addition to the|"
    r"provided all|listed all|complete list|full list|"
    r"just (?:the|those|these|that)\b[^.?!]{0,80}\b(?:correct|right)\?|or other items as well|(?:keep|leave) the \w+(?: \w+)? unchanged\?)",
    re.I,
)
OPTION_ASK_RE = re.compile(
    r"\b(?:which (?:option|variant|version|configuration|model|color|colour|size|one would you|of these options|combination)|"
    r"(?:specify|clarify|choose|select|pick|prefer|provide)\b[^.?!]{0,60}\b(?:option|variant|version|configuration|size|color|colour|"
    r"material|capacity|storage|style|specifications|details for the new)|"
    r"(?:options?|variants?|configurations?|combinations?) (?:you(?:'d| would) (?:like|prefer)|you want)|"
    r"would you (?:like|prefer) (?:the|a|an)\b[^.?!]{0,60}\bor\b|(?:what|which) (?:color|colour|size|material|type|capacity|brightness|"
    r"set type|options?|connectivity)\b[^.?!]{0,40}\b(?:would you|do you|you(?:'d| would))|"
    r"(?:do you have|any) (?:a )?(?:preference|specific (?:option|preference))|exact (?:option|variant|item|options|specifications)|"
    r"(?:other|another|alternative|different) (?:available )?(?:options?|variants?|sizes?|combinations?|product)|"
    r"keep the current|let me know your preference|(?:exact|preferred|final) (?:selections?|choices?|options?)|your (?:selections?|choice) (?:of|for)|"
    r"(?:color|colour|material|size|set type|brightness|connectivity|capacity|type) and (?:color|colour|material|size|set type|"
    r"brightness|connectivity|capacity|type)\b|which one (?:you(?:'d| would) like|do you want|would you like|you prefer))\b",
    re.I,
)
ITEM_CHOICE_RE = re.compile(
    r"\b(?:which (?:item|items|product|products)\b|specify which (?:item|product)|(?:only|just) one of (?:these|them|the items)|"
    r"item id of the|(?:exchange|return) (?:just|only) the\b(?![^.?!]{0,60}\b(?:other items?|anything else|as well))[^.?!]{0,60}\?|"
    r"which item\(s\))",
    re.I,
)


ACTION_WORDS = {
    "cancel": re.compile(r"\bcancel\w*", re.I),
    "exchange": re.compile(r"\b(?:exchang\w*|swap\w*|replac\w*)", re.I),
    "return": re.compile(r"\b(?:return(?:ed|ing|s)?\b(?! (?:policy|to (?:you|us)))|send(?:ing)? back|refund(?:ed|ing)?\b)", re.I),
    "modify_payment": re.compile(
        r"\b(?:(?:chang|switch|updat|modif|mov|replac|set)\w*\b[^.?!]{0,60}\b(?:payment|paypal|gift card|card|visa|mastercard|"
        r"amex|discover)|(?:payment|paypal|gift card|visa|mastercard|card)\b[^.?!]{0,40}\b(?:chang|switch|updat|modif)\w*|"
        r"(?:paid|pay|charged?) (?:for )?(?:with|using|via|to|on|by)\b|new payment(?: method)?\b|switch(?:ing)? (?:it|this|the order|order)\b[^.?!]{0,30}\bto)",
        re.I,
    ),
    "transfer": re.compile(r"\btransfer\w*\b[^.?!]{0,40}\b(?:human|agent|representative|specialist)", re.I),
}
EXCHANGE_SOFT_RE = re.compile(r"\b(?:chang\w*|switch\w*|updat\w*)\b", re.I)

DONE_RE = re.compile(
    r"\b(?:(?:has|have) (?:now )?(?:been|successfully been) (?:successfully |now )?(?:cancel\w*|exchang\w*|return\w*|"
    r"processed|updated|changed|modified|refunded|submitted|requested|placed|initiated|completed|switched|completed)|"
    r"(?:is|are) now (?:cancel\w*|processed|updated|changed|being processed|confirmed|complete|in progress|switched)|"
    r"\bi(?:'ve| have) (?:successfully )?(?:cancel\w*|exchang\w*|processed|updated|changed|modified|submitted|"
    r"initiated|completed|switched|requested|placed|set up)|"
    r"(?:was|were) (?:successfully )?(?:cancel\w*|processed|updated|changed|exchang\w*|returned|submitted)|"
    r"successfully (?:cancel\w*|processed|updated|changed|exchang\w*|returned|submitted|requested|initiated|modified|switched)|"
    r"(?:request|exchange|return|cancellation|change|update) (?:is|has been) (?:confirmed|complete|completed|successful|done|"
    r"processed|submitted|initiated)|"
    r"(?:your|the) (?:return|exchange|cancellation|refund|order) (?:status )?(?:is|will be) (?:now )?(?:updated|marked) (?:to|as)|"
    r"status (?:has been|is now|is) (?:updated|changed) to|"
    r"(?:return|exchange) has been (?:successfully )?requested)\b",
    re.I,
)

# ---- refusal / constraint lexicon -------------------------------------------
INABILITY_RE = re.compile(
    r"\b(?:can(?:no|')?t|cannot|unable to|not able to|won't be able to|will not be able to|not allowed to|"
    r"not permitted to|am not permitted|not authorized to|not possible|won't|will not|must (?:deny|decline|refuse)|"
    r"have to (?:deny|decline|refuse)|not in a position to|no way to|(?:i|we) (?:can|am able to|are able to) only|"
    r"(?:can|may) only|only (?:able|allowed|permitted|authorized) to|limited to|restricted to|"
    r"(?:is|are) not (?:allowed|permitted|possible|eligible|supported)|isn't (?:allowed|permitted|possible|eligible)|"
    r"(?:don't|do not) have (?:access|permission|the ability)|without (?:their|her|his) (?:authorization|consent)|"
    r"(?:need|needs|will need|would need|has to|have to|must) (?:to )?(?:contact|reach out to|call|request|do (?:this|that|it)|"
    r"handle|make (?:this|the|that) request)(?: us)?(?: directly| (?:her|him|them)sel(?:f|ves)| on (?:her|his|their) own)?|"
    r"(?:only|just) (?:the )?(?:account (?:holder|owner)|authenticated user|person who placed|owner)|"
    r"against (?:our|the) policy|policy (?:doesn't|does not) allow|not (?:in|within) (?:my|our) (?:scope|policy))\b",
    re.I,
)
OWNERSHIP_OBJ_RE = re.compile(
    r"\b(?:(?:other|another|different|separate|a third)[- ](?:people|persons?|users?|customers?|accounts?|individuals?|profiles?|parties|party)|"
    r"someone else(?:'s)?|somebody else(?:'s)?|others(?:'s)?\b|third[- ]part(?:y|ies)|"
    r"(?:your )?(?:" + "|".join(_RELATIONS) + r")(?:'s|s)?\b|"
    r"account (?:holder|owner)s?|(?:the )?owner of|person who (?:placed|made|owns)|"
    r"(?:authenticated|verified|logged[- ]in|signed[- ]in|current) (?:user|account|customer|caller)(?:'s)?|"
    r"(?:your|their) own (?:account|orders?|user|profile)|own orders?|orders? (?:on|under|in|from|tied to|linked to|associated with|belonging to) your|"
    r"(?:your|the) (?:account|user (?:account|id|profile)|profile) (?:that|which|i(?:'ve| have)|we(?:'ve| have))|"
    r"one (?:user|customer|account|person) per (?:conversation|session|call|chat)|on behalf of|for other (?:people|users|customers)|"
    r"not (?:yours|your own|on your account|in your account|under your|associated with your|linked to your|tied to your|"
    r"part of your|listed (?:on|under|in) your|registered (?:to|under) you)|isn't yours|aren't yours|'s not yours|"
    r"(?:belongs?|belonging) to|under (?:a |the |her |his |their |another )?(?:different |other )?(?:user|account|customer|name)|"
    r"(?:placed|made|owned|purchased) by|(?:her|his|their) (?:order|account|orders))\b",
    re.I,
)
OWNERSHIP_STRONG_RE = re.compile(
    r"\b(?:(?:you are|you're|you aren't) not (?:the|an?) (?:user|customer|account (?:holder|owner)|owner|person)(?: associated| linked| who| on| of)|"
    r"not (?:yours|your own)|(?:not|isn't|is not) your (?:own )?(?:order|account)\b|isn't yours|aren't yours|'s not yours|is not yours|"
    r"one (?:user|customer|account|person) per (?:conversation|session|call|chat)|"
    r"(?:not|isn't|aren't|wasn't|is not|are not|was not) (?:associated|linked|tied|registered|connected|listed|under|on|in|part of|one of)"
    r"(?: with| to| under| on| in)? (?:your (?:own )?(?:account|profile|user|orders\b|order history|name)|you\b|this account|"
    r"the account (?:i|we) (?:verified|authenticated))|"
    r"(?:doesn't|does not|don't|do not) (?:belong|appear) (?:to|in|on|under) (?:you|your)|"
    r"(?:don't|do not|can't|cannot) see (?:it|that order|this order|order #?w?\d*|the order)?\s*(?:in|on|under) your account|"
    r"(?:will|would) (?:need|have) to (?:contact|reach out to|call) us (?:her|him|them)sel(?:f|ves)|"
    r"(?:needs?|has) to (?:contact|reach out to) us (?:directly|(?:her|him|them)sel(?:f|ves)))\b",
    re.I,
)
OWNERSHIP_STATEMENT_RE = re.compile(  # whose order it is, said declaratively
    r"\b(?:belongs? to (?:a different|another|someone|somebody|your " + "|your ".join(_RELATIONS) + r"|her\b|him\b|them\b)|"
    r"(?:placed|made|owned) (?:by|under|through) (?:a different|another|someone|your|her|his|their)|"
    r"under (?:a different|another|someone else's|her|his|their) (?:account|user|name)|"
    r"(?:is|'s|isn't|is not) (?:registered|linked|associated) (?:to|with|under) (?:a different|another|someone)|"
    r"(?:ask|tell) (?:your|her|him|them)\b[^.?!]{0,30}\bto contact us|"
    r"belongs? to (?:the )?(?:user|customer|account)\b|(?:user ?id|account)\b[^.?!]{0,40}\b[a-z]+_[a-z]+_\d{4}|"
    r"(?:different from|(?:does not|doesn't|do not|don't) match|not the same as) (?:your|the (?:one|account|user) you))",
    re.I,
)
OWNERSHIP_DENY_CTX_RE = re.compile(  # a sentence describing whose order it is -- needed with OWNERSHIP_OBJ for a refusal
    r"\b(?:order|orders|account|request|cancel\w*|return\w*|chang\w*|modif\w*|assist|help|process|act|handle|discuss|"
    r"access|proceed|action|details)\b",
    re.I,
)
STATUS_OBJ_RE = re.compile(
    r"\b(?:(?:already |been )?(?:delivered|processed|shipped|cancelled|canceled)\b|only (?:pending|delivered) orders?|"
    r"(?:status|order) (?:is|was) (?:not )?(?:pending|delivered|processed|shipped)|not (?:in )?pending|no longer pending|"
    r"(?:pending|delivered) status)",
    re.I,
)
PAYMENT_OBJ_RE = re.compile(
    r"\b(?:refunds?|payment methods?|paypal|gift card|credit card|visa|mastercard|amex|discover|card|balance|"
    r"funds|original (?:payment|method|card|form)|method used|payment)\b",
    re.I,
)
CONSTRAINT_RE = re.compile(
    r"\b(?:must (?:be|go|either)|can only (?:be|go)|(?:can|could|will|may) only be (?:refunded|issued|sent|processed|returned|made)|"
    r"only (?:be )?(?:refunded|issued|sent|processed|returned) to|either (?:the )?original|original payment method or|"
    r"(?:to|back to) the original (?:payment )?(?:method|card|form)|(?:refunds?|refunded) (?:are|is|will be)? ?(?:only )?(?:issued|processed|sent|made|returned|go(?:es)?)"
    r"(?: back)? to the (?:original|same|card|payment method used)|"
    r"(?:payment method|card) (?:that was )?used (?:for|to (?:make|pay for|purchase)|at) (?:the )?(?:purchase|order)|"
    r"(?:not|isn't) (?:an )?eligible|(?:doesn't|does not|don't|do not) have (?:enough|sufficient)|(?:insufficient|not enough|not sufficient|isn't enough|not cover|doesn't cover|does not cover|"
    r"won't cover|cannot cover|can't cover|less than|lower than|exceeds?|short of|only has|only have)|"
    r"(?:only|just) (?:pending|delivered) orders?|(?:can|may) only be (?:cancel|exchang|return|modif)\w*|"
    r"(?:reason|reasons) (?:must|has to|should|can only|needs to)|must be (?:either )?\"?(?:no longer|ordered)|"
    r"(?:either|only) \"?no longer needed\"? or \"?ordered by mistake|only (?:be done|be exchanged|be returned|be modified) once|"
    r"(?:can|could) only be (?:done|performed|used)|same product type|policy|not (?:allowed|permitted|possible)|"
    r"one (?:user|customer) per)\b",
    re.I,
)
# A stated RULE about where a refund / payment may go or what a balance allows. Unlike CONSTRAINT_RE (tuned
# for the gold constraint_statement label), a plain description of the plan ("refund to the original payment
# method") or a question about it is not one. Only these earn the payment fallback.
PAY_RULE_RE = re.compile(
    r"\b(?:must (?:be|go|either)|can only (?:be|go)|(?:can|could|will|may) only be (?:refunded|issued|sent|processed|returned|made|used)|"
    r"only (?:be )?(?:refunded|issued|sent|processed|returned|used) (?:to|if|when)|either (?:the )?original|original payment method or|"
    r"or (?:to )?(?:a|the|an existing|your) gift card|(?:refunds?|refunded) (?:are|is|will be|can be)? ?only|only (?:to|back to) the original|"
    r"(?:not|isn't|aren't) (?:an )?(?:eligible|allowed|permitted|possible|supported)|"
    r"(?:doesn't|does not|don't|do not) have (?:enough|sufficient)|insufficient|not enough|not sufficient|isn't enough|isn't sufficient|"
    r"(?:not|doesn't|does not|won't|cannot|can't) (?:fully )?cover|less than|lower than|exceeds?|short of|only has|only have|"
    r"too (?:low|small)|not (?:high|large|big) enough|policy|"
    r"(?:can(?:no|')?t|cannot|unable to|not able to|won't be able to|wasn't able to|was not able to) (?:be )?(?:refund|use|send|process|switch|change|issue|put|go)\w*)\b"
    r"|\$\s?[\d,.]+\s*<",
    re.I,
)
CHECK_PLAN_RE = re.compile(
    r"\b(?:let me|i(?:'ll| will| am going to|'m going to)|allow me to|i need to|i have to|i'd like to|i will now|"
    r"one moment while i|while i)\b[^.?!]{0,30}\b(?:check|see|verify|look|confirm|find out|determine|review|pull up|look up)\b",
    re.I,
)

# ---- details lexicon ---------------------------------------------------------
REASON_NLN_RE = re.compile(r"\b(?:no longer need(?:ed|s)?|don'?t need (?:it|them)|do not need (?:it|them)|not needed|no use for|"
                           r"(?:don'?t|do not) have (?:a|any) use for|"
                           r"doesn'?t need (?:it|them)|no longer require)", re.I)
REASON_OBM_RE = re.compile(r"\b(?:ordered by mistake|by mistake|by accident|mistaken(?:ly)?|accidental(?:ly)?|in error)\b", re.I)
REASON_PAIR_RE = re.compile(r"no longer needed\W{0,6}\s*(?:or|and|/)\s*\W{0,6}ordered by mistake|ordered by mistake\W{0,6}\s*(?:or|and|/)\s*"
                            r"\W{0,6}no longer needed|no longer need it\b.{0,30}\bor\b.{0,30}mistake", re.I)

PM_ID_RE = re.compile(r"\b(credit_card|gift_card|paypal)_\d{7}\b", re.I)
LAST4_RE = re.compile(r"\b(?:ending(?: in| with)?|last (?:four|4)(?: digits)?(?: of)?|ends in|xxxx|\*{2,}|x{2,})\s*:?\s*\(?(\d{4})\b", re.I)
BRAND_RE = re.compile(r"\b(visa|mastercard|master card|amex|american express|discover)\b", re.I)
PAYPAL_RE = re.compile(r"\bpay ?pal\b", re.I)
GIFT_RE = re.compile(r"\bgift ?cards?\b", re.I)
ORIGINAL_RE = re.compile(
    r"\b(?:original (?:payment(?: method)?|method|card|form of payment|credit card|paypal|gift card)|"
    r"(?:payment method|card|method) (?:that was |you )?used (?:for|to (?:make|pay for|purchase)|at|when you)|"
    r"same (?:payment method|card|method)|card (?:used )?at purchase)\b",
    re.I,
)
CREDIT_RE = re.compile(r"\bcredit card\b", re.I)
# "to be paid with X", "will be paid using X", "so it is paid with X": X is the method proposed, not the current one
FUTURE_PAID_RE = re.compile(
    r"\b(?:to be|will be|would be|shall be|should be|be|so (?:that )?it(?:'s| is)|it(?:'ll| will) be|instead) (?:now )?(?:paid|charged|covered|billed)"
    r" (?:for )?(?:with|via|using|by|on|through|to)\W*(?:your |the |a |an |my )?$",
    re.I,
)
NOT_PROPOSED_BEFORE_RE = re.compile(
    r"(?:\bfrom|\bcurrent(?:ly)?(?: payment(?: method)?)?(?: is| was|:)?|\bwas (?:paid|made|purchased|charged)(?: for)?(?: with| via| using| by| on| through)?|"
    r"\bpaid (?:for )?(?:with|via|using|by|on|through)|\bpurchased (?:with|using)|\binstead of|\brather than|\bnot|\bother than|"
    r"\boriginally(?: paid)?(?: with| via| using)?|\bexisting payment(?: method)?(?: is)?|\bused (?:to pay|for (?:the )?(?:purchase|order))(?: was)?|"
    r"\bcan(?:no|')?t (?:be )?(?:use|refund|send|switch|change|process)\w*(?: (?:it|the refund|this|the payment))?(?: (?:to|on|with))?|"
    r"\bunable to (?:use|refund|send|process)\w*(?: (?:it|the refund))?(?: (?:to|on|with))?|\bbalance (?:of|on)(?: your)?)\W*(?:your |the |a |an |my )?$",
    re.I,
)


# =============================================================================
# Sentence classification
# =============================================================================


def _strip_lead(s: str) -> str:
    return re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", s).strip()


def is_request(s: str) -> bool:
    """An asking sentence: a question, or an imperative / need statement
    addressed to the user that is not negated, informational or a sign-off."""
    s = _strip_lead(s)
    if PLEASE_NOTE_RE.search(s) and "?" not in s:
        return False
    if "?" in s:
        return True
    if REQUEST_RE.search(s):
        # "Please confirm if you'd like..." / "Kindly provide ..." -- but not "If you need anything, please let me know"
        if SIGNOFF_RE.search(s) and not re.search(r"\b(?:confirm|provide|reply|respond|share)\b", s, re.I):
            return False
        return True
    if re.search(r"\b(?:please|kindly)\b", s, re.I):
        if SIGNOFF_RE.search(s) and not re.search(r"\b(?:confirm|provide|reply|respond|share|specify)\b", s, re.I):
            return False
        if re.search(r"\b(?:please|kindly)\s+(?:confirm|provide|reply|respond|share|send|tell|let me know|specify|clarify|choose|"
                     r"select|enter|type|verify|indicate|double-check|check|give|answer|say)\b", s, re.I):
            return True
    if re.search(r"\blet me know\b", s, re.I) and not SIGNOFF_RE.search(s):
        return True
    if re.search(r"\bplease\W*$", s, re.I) and not SIGNOFF_RE.search(s):
        return True
    if re.search(r"\b(?:reply|respond|answer|type|say)\b[^.?!]{0,30}\byes\b", s, re.I):
        return True
    if NEED_RE.search(s) and not re.search(rf"\b(?:i|we) {NEG}\b", s, re.I):
        # "I need your email" / "I'll need her first name" -- not "I need to check the order"
        if NEED_YOUR_RE.search(s) or re.search(r"\b(?:need|require)\b[^.?!]{0,20}\b(?:the|a|an|some|more|additional)\b", s, re.I):
            return True
    if re.search(r"\bconfirm\b", s, re.I) and re.search(r"^(?:to proceed|before (?:i|we)|once|if)\b", s, re.I) is None:
        if re.search(r"^(?:confirm|please confirm|kindly confirm)\b", s, re.I):
            return True
    return False


def _negated_request(s: str) -> bool:
    """'I cannot provide details', 'I can't share that' -- a request verb under negation."""
    return bool(re.search(rf"\b(?:i|we)\s+{NEG}\s+(?:\w+\s+){{0,2}}(?:provide|share|give|disclose|access|see)\b", s, re.I))



CONSENT_RE = re.compile(
    r"\b(?:shall i\b(?! (?:transfer|connect|check|look|help|assist))|"
    r"should i (?:still )?(?:go ahead|proceed|continue|cancel|exchange|return|change|update|process|modify|switch|submit|initiate|place|move|set)|"
    r"(?:would|do|will) you (?:still )?(?:like|want|wish) me to (?!transfer|help|assist|check|look|search|provide|explain|guide|connect|find|"
    r"verify|locate|pull|review|recommend|walk|send you|escalate|create|add)|"
    r"(?:would|do) you (?:like|want|wish) to (?:proceed|go ahead|continue)\b(?! with (?:a|another|other|different|one of|either|only|just))|"
    r"(?:if|whether) (?:you(?:'d| would| want| wish)?|you'd) (?:still )?(?:like|want|wish)? ?(?:me )?to (?:proceed|go ahead|continue|cancel|exchange|"
    r"return|switch|move|submit)\b|"
    r"(?:if|whether) (?:i (?:should|can|may)|to) (?:proceed|go ahead|continue)|"
    r"(?:would|do) you (?:still )?(?:like|want|wish) (?:me )?to (?:cancel|exchange|return|change|switch|update|modify|move|submit) (?:this|it|the|order|your|these)|"
    r"ok(?:ay)? to (?:proceed|go ahead|continue)|good to go|ready for me to|green light|"
    r"(?:reply|respond|answer|type|say|confirm|send|write)\b[^.?!]{0,40}[\"'(]yes\b|\(yes|yes/no|\"yes\" or \"no\"|yes or no\b|"
    r"confirm (?:with|by) (?:a |replying|typing|saying)|"
    r"(?:please |kindly )?confirm (?:that|if|whether) (?:you|this|these|the|everything|all|it|i)\b(?! (?:have|meant|mean|need|know|are referring)\b)|"
    r"(?:please |kindly |can you |could you )?confirm (?:the|these|this|your|all|my) (?:details|exchange|return|cancellation|change|request|action|"
    r"order(?! id| number)|following|summary|above|modification|update|swap|refund|plan)|"
    r"confirm (?:to proceed|and i(?:'ll| will)|so (?:that )?i can (?:proceed|go ahead|process|cancel|submit|complete))|"
    r"is (?:this|that|everything|all of this|this all|that all|all this|the above) (?:information )?(?:correct|right|accurate|ok(?:ay)?|fine)\b|"
    r"(?:does|do) (?:that|this|these|everything) (?:sound|look) (?:good|right|correct|ok(?:ay)?|fine)|sounds? good\?|looks? good\?|"
    r"(?:,|--|\u2014|\.)\s*(?:correct|right|is that (?:right|correct|ok(?:ay)?)|ok(?:ay)?|sound good|go ahead|agreed)\s*\?|"
    r"^(?:correct|right|ok(?:ay)?|go ahead|proceed|shall we|all good)\s*\?|confirm with (?:a )?\W?yes\b|"
    r"(?:reply|respond|answer|type|say)\s+(?:with\s+)?(?:a\s+)?\W?yes\b|^(?:so |just )?to confirm,? (?:you|i)\b[^?]*\?|"
    r"are you sure|once you confirm, i(?:'ll| will) proceed|let me know if (?:this|that|everything|it) (?:looks|sounds|is) (?:good|right|correct|ok(?:ay)?|fine)|"
    r"(?:verify|confirm|check) (?:that )?(?:these|the|all|this|everything)\b[^.?!]{0,25}\b(?:is|are) (?:correct|right|accurate)|"
    r"(?:do|can|could|would) you (?:please )?confirm\s*(?:\(yes/no\))?\s*\?|^(?:please |kindly )?confirm\W*$|"
    r"confirm (?:that's|that is|this is|it's) (?:correct|right|ok(?:ay)?)|confirm (?:you(?:'d| would)? (?:like|want)|your (?:approval|consent|agreement))|"
    r"(?:your|you) (?:approval|consent|go-ahead|authori[sz]ation)\b|do you (?:approve|agree|authori[sz]e|consent)|"
    r"(?:can|may) i (?:proceed|go ahead|cancel|process|submit|place|make (?:this|the) change))",
    re.I,
)
CONSENT_EXCLUDE_RE = re.compile(
    r"\b(?:transfer\w*|human|representative|specialist|anything else|something else|"
    r"how (?:you(?:'d| would) like|to) (?:proceed|move forward|continue)|how i can (?:assist|help))\b",
    re.I,
)


def _clauses(s: str) -> list[str]:
    """A sentence split at clause joints that commonly glue a consent ask to a sign-off or an alternative."""
    parts = re.split(r";|, (?:and |but )?(?:also |or )?(?=(?:please|let me know|if you|or if|otherwise|alternatively)\b)| -- ", s)
    return [p.strip() for p in parts if p.strip()]


CONSENT_ALT_RE = re.compile(
    r"\b(?:a different|another|other (?:options?|variants?|orders?|requests?|methods?|cards?)|one of (?:these|them|your|the)|either|"
    r"your own|any of (?:these|them)|"
    r"which (?:one|option|variant|combination|item)|different (?:payment|request|method|option|variant|combination|item|card))\b|"
    r"\bor (?:would you like to |if you(?:'d| would) (?:like|prefer) to |do you want to |you(?:'d| would) like to )?"
    r"(?:check|review|modify|make a|add|use|explore|choose|select|keep|consider|search|cancel the exchange|revert)\b|"
    r"\b(?:return\w*|exchang\w*|cancel\w*|modif\w*) or (?:return\w*|exchang\w*|cancel\w*|modif\w*)\b|"
    r"\bproceed with (?:this|that|the) (?:information|info|details you (?:provided|gave))\b",
    re.I,
)


def _consent_clause(c: str) -> bool:
    if not CONSENT_RE.search(c):
        return False
    m0 = CONSENT_RE.search(c)
    if re.match(r"^\W*(?:why|what|which|how|when|where|who)\b", c, re.I) and not re.search(r"\byes\b", c, re.I):
        return False  # "Why do you want to cancel order #W1?" asks for information, not consent
    if re.search(r"\b(?:so (?:that )?(?:i|we) can|before (?:i|we)|(?:i|we) (?:can|will|'ll|could|need to))\s*$", c[:m0.start()], re.I):
        return False  # "so I can confirm the exchange": the agent confirming, not asking
    if CONSENT_ALT_RE.search(c) and not re.search(r"\byes\b", c, re.I):
        return False
    if IDENTITY_RE.search(c) and not re.search(r"\b(?:cancel\w*|exchang\w*|return\w*|refund\w*|switch\w*|chang\w*|modif\w*|"
                                              r"updat\w*|proceed|go ahead)\b", c, re.I):
        return False  # "It's jane@example.com, correct?" checks an identity detail, it asks no consent
    m = CONSENT_RE.search(c)
    cl = c.lower()
    if CONSENT_EXCLUDE_RE.search(c) and not re.search(r"\b(?:yes|proceed with (?:the |this )?(?:cancel|exchang|return|chang|refund|modif))", cl):
        return False
    # "Please confirm your email" -- consent words around an information object
    if re.search(r"\bconfirm (?:your|her|his|their|the) (?:e-?mail|zip|name|identity|address|order (?:id|number)|user ?id|reason|payment method|item id)", cl) \
            and not re.search(r"\b(?:proceed|go ahead|yes)\b", cl):
        return False
    if re.search(r"^(?:once|after|when) (?:you(?:'ve| have)? )?(?:provide|confirm|give|share)", cl) and "proceed" not in cl[m.start():] and "?" not in c:
        return False
    return True

# =============================================================================
# The rules analyzer
# =============================================================================


class Analyzer(Protocol):
    def analyze(self, text: str, task: Any, state: Optional[NLUState] = None) -> Semantics: ...


@dataclass
class _PMMention:
    start: int
    end: int
    pm: str            # pm id, "original", "unknown", "credit_card" (ambiguous), "gift_card" (ambiguous)
    proposed: bool = True


class RulesAnalyzer:
    """Deterministic rules (see the module docstring)."""

    def analyze(self, text: str, task: Any, state: Optional[NLUState] = None) -> Semantics:
        facts = facts_from(task)
        state = state or NLUState()
        return _analyze(text or "", facts, state)


def _resolve_pm_mentions(sent: str, facts: TaskFacts) -> list[_PMMention]:
    pms = facts.payment_methods
    own = {k: v for k, v in pms.items() if v.get("owner") == facts.user_id} or pms
    out: list[_PMMention] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def add(a: int, b: int, pm: str) -> None:
        if free(a, b):
            out.append(_PMMention(a, b, pm))
            taken.append((a, b))

    for m in PM_ID_RE.finditer(sent):
        pid = m.group(0).lower()
        add(m.start(), m.end(), pid if pid in pms else "unknown_id")
    for m in LAST4_RE.finditer(sent):
        last4 = m.group(1)
        hits = [k for k, v in pms.items() if v.get("last_four") == last4]
        # include a preceding brand word in the span
        a = m.start()
        bm = None
        for b in BRAND_RE.finditer(sent[max(0, a - 40):a]):
            bm = b
        if bm is not None:
            a = max(0, a - 40) + bm.start()
        add(a, m.end(), hits[0] if len(hits) == 1 else "unknown_id")
    for m in BRAND_RE.finditer(sent):
        brand = m.group(1).lower().replace("master card", "mastercard").replace("american express", "amex")
        hits = [k for k, v in own.items() if v.get("source") == "credit_card" and (v.get("brand") or "").lower() == brand]
        if not hits:
            hits_all = [k for k, v in pms.items() if v.get("source") == "credit_card" and (v.get("brand") or "").lower() == brand]
            add(m.start(), m.end(), hits_all[0] if len(hits_all) == 1 else "unknown")
        else:
            add(m.start(), m.end(), hits[0] if len(hits) == 1 else "credit_card")
    for m in ORIGINAL_RE.finditer(sent):
        add(m.start(), m.end(), "original")
    for m in PAYPAL_RE.finditer(sent):
        hits = [k for k, v in own.items() if v.get("source") == "paypal"]
        if not hits:
            hits = [k for k, v in pms.items() if v.get("source") == "paypal"]
        add(m.start(), m.end(), hits[0] if len(hits) == 1 else "unknown")
    for m in GIFT_RE.finditer(sent):
        hits = [k for k, v in own.items() if v.get("source") == "gift_card"]
        if not hits:
            hits = [k for k, v in pms.items() if v.get("source") == "gift_card"]
        add(m.start(), m.end(), hits[0] if len(hits) == 1 else ("unknown" if not hits else "gift_card"))
    for m in CREDIT_RE.finditer(sent):
        hits = [k for k, v in own.items() if v.get("source") == "credit_card"]
        add(m.start(), m.end(), hits[0] if len(hits) == 1 else "credit_card")
    out.sort(key=lambda x: x.start)
    for x in out:
        before = sent[max(0, x.start - 45):x.start]
        if re.search(r"\b(?:e\.g\.|for example|such as|like)\W*(?:\"|')?(?:[\w ,'\"]{0,30}\bor\b\W*)?$", before, re.I):
            x.proposed = False
            continue
        if NOT_PROPOSED_BEFORE_RE.search(before) and not FUTURE_PAID_RE.search(before):
            x.proposed = False
        if re.search(r"\b(?:balance|has|have|only has)\b[^.?!]{0,10}$", before, re.I) and x.pm != "original":
            pass
    # "from X to Y": X is the current method
    for i, x in enumerate(out):
        if i + 1 < len(out) and re.search(r"^\W*(?:\([^)]*\))?\W*(?:to|->|=>|→)\W", sent[x.end:out[i + 1].start] + " ", re.I):
            if re.search(r"\bfrom\b[^.?!]{0,10}$", sent[max(0, x.start - 25):x.start], re.I) or re.search(r"(?:->|→)", sent[x.end:out[i + 1].start]):
                x.proposed = False
    return out


def _canon_pm(pm: str, facts: TaskFacts) -> str:
    if pm == "original":
        return facts.original_pm or "original"
    return pm


def _item_spans(text: str, names: list[str]) -> dict[str, list[tuple[int, int]]]:
    """For each item name, the text spans from each of its mentions to the next mention of any item name."""
    low = text.lower()
    hits = []
    for n in set(names):
        for m in re.finditer(re.escape(n.lower()), low):
            hits.append((m.start(), m.end(), n))
    hits.sort()
    spans: dict[str, list[tuple[int, int]]] = {}
    for i, (a, b, n) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        spans.setdefault(n, []).append((a, end))
    return spans


def _value_rx(key: str, val: str) -> re.Pattern:
    v = re.escape(val.lower())
    k = re.escape(key.lower())
    if val.lower() in ("yes", "no", "none"):
        # boolean / none values only count next to their key: "backlight none", "no backlight", "waterproof: yes"
        alts = [rf"{k}\W{{0,3}}(?:\w+\W{{0,3}})?{v}\b", rf"\b{v}\W{{0,3}}{k}"]
        if val.lower() == "none":
            alts.append(rf"\bno {k}")
            alts.append(rf"\bwithout (?:a |an )?{k}")
        if val.lower() == "no":
            alts.append(rf"\bwithout (?:a |an )?{k}|\bnon-?{k}")
        if val.lower() == "yes":
            alts.append(rf"\bwith (?:a |an )?{k}")
        return re.compile("|".join(alts), re.I)
    toks = []
    for tok in re.split(r"[\s-]+", val.lower()):
        t = re.escape(tok)
        if len(tok) > 3 and tok.endswith("es"):
            t = re.escape(tok[:-2]) + "(?:es)?"
        elif len(tok) > 3 and tok.endswith("s"):
            t = re.escape(tok[:-1]) + "s?"
        toks.append(t)
    return re.compile(r"(?<![\w-])" + r"[\s-]*".join(toks) + r"(?![\w-])", re.I)


def _analyze(text: str, facts: TaskFacts, state: NLUState) -> Semantics:
    sem = Semantics()
    sents, units = request_units(text)
    req_sents = [u.info for u in units]
    full = clean(text)
    low = full.lower()
    q_sents = [s for s in sents if "?" in s]

    # ---- other-help offers / sign-offs ----------------------------------------
    def other_help_sentence(s: str) -> bool:
        if re.search(r"\b(?:one of )?your own (?:orders?|request)|(?:a different|another) request\b", s, re.I) and not IDENTITY_RE.search(s):
            return True
        if IDENTITY_RE.search(s) or ORDER_ID_ASK_RE.search(s) or re.search(r"\b(?:reason|payment method)\b", s, re.I) \
                or (ALL_ITEMS_ASK_RE.search(s) and facts.template != "cancel"):
            return False
        if SIGNOFF_RE.search(s):
            return True
        if re.search(r"\btransfer\w*\b[^.?!]{0,50}\b(?:human|agent|representative|specialist|supervisor|team)", s, re.I):
            return True
        if OTHER_HELP_RE.search(s) and not re.search(r"\b(?:cancel|exchange|return|change|modify|switch|update)\w* (?:it|this|that|the order|order #|your sister|her|his)", s, re.I):
            return True
        return False

    action_req_sents = [s for s in req_sents if not other_help_sentence(s)]
    sem.offers_other_help = any(other_help_sentence(s) for s in sents if is_request(s) or "?" in s or SIGNOFF_RE.search(s))

    # ---- info requests -------------------------------------------------------
    infos: list[str] = []

    def add_info(k: str) -> None:
        if k not in infos:
            infos.append(k)

    order_ids = list(dict.fromkeys(ORDER_ID_RE.findall(full)))
    for s in action_req_sents:
        sl = s.lower()
        confirm_like = bool(re.search(r"\bconfirm\b", sl))
        # identity
        if IDENTITY_RE.search(s) and not re.search(r"\b(?:verified|authenticated|confirmed|located|found)\b[^.?!]{0,30}\b(?:identity|account|email)", sl) \
                and not re.search(r"(?<!double-)\b(?:check|see) your (?:email|inbox)\b(?! address)|\bemail (?:confirmation|notification|with)|receive an email|send you an email|via email|by email", sl):
            add_info("identity")
            if THIRD_PARTY_RE.search(s) and not re.search(r"\byour (?:own )?(?:e-?mail|name|zip|identity|details)\b", sl):
                sem.identity_target = "third_party"
            elif sem.identity_target is None:
                sem.identity_target = "caller"
        # order id / choice
        if ORDER_ID_ASK_RE.search(s) and not re.search(r"\b(?:the|this|your) order (?:id|number) (?:is|was)\b", sl):
            if confirm_like and len(order_ids) == 1 and not re.search(r"\bwhich\b", sl) and re.search(r"order (?:id|number)\s*\(?#?w?\d", sl):
                pass  # "confirm order ID #W123" names the order, it does not ask for it
            elif len(order_ids) >= 2 and re.search(r"\b(?:which|one of|or)\b", sl):
                add_info("order_choice")
            else:
                add_info("order_id")
        elif len(re.findall(r"#W\d{7}", s)) >= 2 and re.search(r"\b(?:which|or)\b", sl):
            add_info("order_choice")
        elif re.search(r"\b(?:meant|mean|referring to|thinking of) (?:a different|another|the other) order", sl):
            add_info("order_choice")
        elif len(order_ids) >= 2 and re.search(r"\bone of (?:them|these|those|the orders|your orders)\b", sl):
            add_info("order_choice")
        # reason
        if facts.template in ("cancel", "foreign_order_refusal") or re.search(r"\breason\b", sl):
            one_value = bool(REASON_OBM_RE.search(sl)) != bool(REASON_NLN_RE.search(sl)) and not REASON_PAIR_RE.search(sl)
            if REASON_ASK_RE.search(s) and not one_value and not re.search(
                    r"\breason (?:is|was)\s*(?:that|because|\"|'|no longer|ordered|you)", sl):
                if re.search(r"\breason\b|\bwhy\b|no longer need", sl):
                    add_info("reason")
        # payment
        if PAYMENT_ASK_RE.search(s) and (
                not re.search(r"\b(?:link|add|set up|setting up)\b[^.?!]{0,30}\b(?:paypal|account|card)\b", sl)
                or re.search(r"\b(?:a different|another|other|alternative) (?:payment|card|method)", sl)):
            named = {m.pm for m in _resolve_pm_mentions(s, facts) if m.proposed}
            if len(named) == 1 and not re.search(r"\b(?:or|which|what|different|another|other|alternative|how)\b", sl):
                pass  # "refund to your original Visa?" -- a yes/no about one method, not a request to choose
            else:
                add_info("payment_method")
        elif re.search(r"\bwhich (?:one |of (?:these|them|those) )?(?:should|would|do|shall)\b", sl) and "?" in s and len(
                {full[m.start:m.end].lower() for m in _resolve_pm_mentions(full, facts)}) >= 2:
            add_info("payment_method")
        # all items
        if ALL_ITEMS_ASK_RE.search(s) and facts.template != "cancel" and not re.search(r"\b(?:remain|stay)s? (?:the same|unchanged)\b", sl):
            add_info("all_items_complete")
        # option choice
        if OPTION_ASK_RE.search(s) and not (facts.template == "modify_payment" and PAYMENT_ASK_RE.search(s)):
            add_info("option_choice")
        if ITEM_CHOICE_RE.search(s):
            add_info("item_choice")
    if "order_choice" in infos and "order_id" in infos:
        infos.remove("order_id")
    sem.info_requests = infos
    if "identity" not in infos:
        sem.identity_target = None
    sem.choice_orders = order_ids if "order_choice" in infos else []

    # ---- confirmation request ------------------------------------------------
    sem.confirmation_request = any(
        u.confirm_details or any(_consent_clause(c) for c in _clauses(u.text))
        for u in units if not other_help_sentence(u.text) or CONSENT_RE.search(u.text)
    )

    # ---- refusal, constraint ---------------------------------------------------
    decl = [s for s in sents if "?" not in s]
    own_hit = status_hit = cons_hit = other_hit = False
    pay_cons = False
    for s in decl:
        sl = s.lower()
        if CHECK_PLAN_RE.search(s) and not INABILITY_RE.search(s):
            continue
        inab = INABILITY_RE.search(s)
        payish = PAYMENT_OBJ_RE.search(s)
        about_order_owner = re.search(r"\border\b|someone|somebody|other (?:users?|customers?|persons?|people)|account holder|"
                                      r"one user|on behalf|" + "|".join(_RELATIONS), sl)
        if IDENTITY_RE.search(s) and not about_order_owner:
            continue
        if payish and not about_order_owner:
            own_cand = False
        else:
            own_cand = (inab and OWNERSHIP_OBJ_RE.search(s) and OWNERSHIP_DENY_CTX_RE.search(s)) or OWNERSHIP_STRONG_RE.search(s)
        if own_cand:
            # "I can help with your sister's order" is not a refusal; need a real inability / restriction
            if not re.search(r"\b(?:i can|we can|i'll|i will|happy to)\b(?! only)[^.?!]{0,40}\b(?:cancel|process|return|help|assist)\b", sl) or inab:
                own_hit = True
                continue
        if inab and STATUS_OBJ_RE.search(s) and re.search(r"\b(?:cancel\w*|modif\w*|chang\w*|return\w*|exchang\w*)\b", sl):
            status_hit = True
            continue
        if re.search(r"\b(?:not|cannot|can't|unable|no longer)\b[^.?!]{0,30}\b(?:be )?(?:cancel\w*|modif\w*)\b[^.?!]{0,60}\b(?:delivered|processed|shipped|status)\b", sl) or \
                re.search(r"\b(?:delivered|processed|shipped)\b[^.?!]{0,60}\b(?:cannot|can't|not|unable)\b[^.?!]{0,20}\b(?:be )?(?:cancel\w*|modif\w*)", sl):
            status_hit = True
            continue
        if payish and (inab or re.search(r"\b(?:insufficient|not enough|not sufficient|(?:doesn't|does not|don't|do not) have (?:enough|sufficient)|less than|lower than|only has|only have|exceeds?|"
                                          r"does not cover|doesn't cover|won't cover|cannot cover|can't cover|not (?:an )?eligible|too (?:low|small)|not (?:high|large|big) enough|"
                                          r"must (?:go|be (?:refunded|issued|sent|processed|returned))|can only (?:go|be)|"
                                          r"only be (?:refunded|issued|sent|processed|returned))\b", sl)):
            if not re.search(r"\bcan(?:no|')?t (?:see|find|locate)\b", sl) or re.search(r"\b(?:paypal|card|payment method)\b", sl):
                cons_hit = True
                pay_cons = True
                continue
        if inab and re.search(r"\b(?:not available|unavailable|no such variant|doesn't exist|does not exist|same product type|"
                              r"only exchange|only be exchanged|out of stock)\b", sl):
            other_hit = True
            continue
        if re.search(r"\b(?:not available|unavailable|no such variant|out of stock)\b", sl) and (
                re.search(r"\b(?:cannot|can't|unable|not possible)\b", low)
                or re.search(r"\b(?:for exchange|inventory|in stock|variants?)\b", sl)):
            other_hit = True
    if not own_hit and any(INABILITY_RE.search(s) and not CHECK_PLAN_RE.search(s) for s in decl) and any(
            OWNERSHIP_STATEMENT_RE.search(s) for s in decl):
        own_hit = True
    if not own_hit and facts.template == "foreign_order_refusal" and any(
            re.search(r"\b(?:can(?:no|')?t|cannot|unable to|not able to|won't)\b[^.?!]{0,30}\b(?:cancel|return|help with|process|"
                      r"change|modify)\b", s, re.I) for s in decl) and re.search(
            r"\b(?:orders? of your own|your own orders?|orders? (?:on|under) your (?:own )?account)\b", low):
        own_hit = True  # "I can't cancel it. Any orders of your own I can help with?"
    if cons_hit and any(CHECK_PLAN_RE.search(s) for s in sents) and not re.search(
            r"\b(?:insufficient|not enough|less than|lower than|only has|doesn't cover|does not cover|won't cover|can(?:no|')?t "
            r"(?:be )?(?:refund|use|send|process|switch)|cannot use|unable to (?:use|refund))", low):
        cons_hit = pay_cons = False  # a rule stated before checking ("can only be used if its balance covers ... let me check")
    if own_hit:
        sem.refusal = "ownership"
    elif status_hit:
        sem.refusal = "status"
    elif cons_hit:
        sem.refusal = "constraint"
    elif other_hit:
        sem.refusal = "other"
    sem.constraint_about_payment = pay_cons
    checking = any(CHECK_PLAN_RE.search(s) for s in sents)
    cs = False
    for s in sents:
        if CONSTRAINT_RE.search(s) and not CHECK_PLAN_RE.search(s):
            cs = True
            break
    sem.constraint_statement = cs or sem.refusal is not None
    if sem.refusal is None and cs:
        # constraint statements about the payment method (refund destination, balance) count for the fallback --
        # not a conditional rule announced before a check ("can only be used if ..., let me check")
        for s in sents:
            if "?" not in s and PAY_RULE_RE.search(s) and PAYMENT_OBJ_RE.search(s) and not CHECK_PLAN_RE.search(s) and not (
                    checking and re.search(r"\b(?:if|as long as|provided|unless|once)\b", s, re.I)):
                sem.constraint_about_payment = True
    sem.ownership_question = any(OWNERSHIP_OBJ_RE.search(s) for s in q_sents)
    for s in q_sents:
        named = {_canon_pm(m.pm, facts) for m in _resolve_pm_mentions(s, facts) if m.proposed}
        if len(named) == 1 and re.search(r"\b(?:use|using|charge|refund|pay|put|go|send|bill|cover|proceed with)\b|"
                                         r"\bfor the (?:price )?difference\b", s, re.I) and not re.search(
                r"\b(?:or|which|what)\b", s, re.I):
            sem.payment_yesno_pm = next(iter(named))
    for s in sents:
        sl = s.lower()
        # "Your gift card has a balance of $40.00, but the order total is $120.50": the constraint, stated by numbers
        if re.search(r"\bbalance\b[^.?!]{0,40}\$\s?[\d,.]+[^.?!]{0,40}\b(?:but|while|whereas|and|which is less)\b[^.?!]{0,40}"
                     r"\b(?:total|cost|amount|price|order)\b", sl):
            sem.constraint_about_payment = True
            sem.constraint_statement = True
        # offering exactly the allowed refund destinations: the original method or a gift card
        if re.search(r"\b(?:original (?:payment|method|card|form)|(?:payment )?method (?:you )?used (?:for|to (?:make|pay for)|at) "
                     r"(?:the )?purchase|card (?:you )?used)\b[^.?!]{0,80}\bgift card|\bgift card\b[^.?!]{0,80}\b(?:original "
                     r"(?:payment|method|card|form)|(?:payment )?method (?:you )?used)", sl) and (
                "?" in s or re.search(r"\b(?:only|either|must|can|options?|choices?)\b", sl)):
            sem.constraint_about_payment = True
            sem.constraint_statement = True

    # ---- claims done -----------------------------------------------------------
    done = False
    for s in sents:
        if DONE_RE.search(s):
            m = DONE_RE.search(s)
            pre = s[:m.start()].lower()
            if re.search(r"\b(?:once|after|when|if|before|until)\b[^,.]{0,60}$", pre) or "?" in s:
                continue
            done = True
            break
    sem.claims_action_done = done

    # ---- proposed action -------------------------------------------------------
    sem.proposed_action = _proposed_action(full, sents, req_sents, facts, state, sem)
    if sem.proposed_action is not None:
        pa = sem.proposed_action
        # an offer to proceed with the requested action in the same turn
        sem.proceeds_anyway = sem.confirmation_request and pa.action in ("cancel", "return", "exchange", "modify_payment")

    # ---- offers other help only ----------------------------------------------
    sem.offers_other_help_only = bool(
        sem.offers_other_help and not sem.info_requests and not sem.confirmation_request
        and (sem.refusal is not None or sem.proposed_action is None) and not sem.claims_action_done
    )
    # ---- narration --------------------------------------------------------------
    sem.narration_only = bool(
        not sem.info_requests and not sem.confirmation_request and sem.refusal is None and not sem.claims_action_done
        and sem.proposed_action is None and not sem.offers_other_help and not q_sents and not req_sents
    )
    return sem


def _detect_action(full: str, facts: TaskFacts) -> Optional[str]:
    hits = {a: len(rx.findall(full)) for a, rx in ACTION_WORDS.items()}
    hits = {a: n for a, n in hits.items() if n}
    if facts.pending_item_change:
        mods = len(re.findall(r"\b(?:modif\w*|chang\w*|updat\w*|switch\w*)\b", full, re.I))
        if mods and any(n.lower() in full.lower() for n, _ in facts.exchange_targets):
            hits["exchange"] = hits.get("exchange", 0) + mods
    if facts.template == "exchange" and not hits.get("exchange") and re.search(r"(?:->|\u2192|=>)", full) and any(
            n.lower() in full.lower() for n, _ in facts.exchange_targets):
        hits["exchange"] = 1
    if re.search(r"\b(?:use|set|put)\b[^.?!]{0,25}\b(?:paypal|gift card|card|visa|mastercard|amex|discover)\b[^.?!]{0,30}"
                 r"\b(?:for|on|as the payment (?:method )?for) (?:order|this order|the order|your order)", full, re.I):
        hits["modify_payment"] = hits.get("modify_payment", 0) + 1
    if facts.template == "modify_payment" and not hits.get("modify_payment") and re.search(
            r"\b(?:updat|chang|switch|mov)\w*\b[^.?!]{0,20}\border\b", full, re.I) and re.search(
            r"\b(?:paypal|gift card|visa|mastercard|amex|discover|credit card|payment)\b", full, re.I):
        hits["modify_payment"] = 1
    if not hits:
        if facts.template == "exchange" and EXCHANGE_SOFT_RE.search(full) and any(
                n.lower() in full.lower() for n, _ in facts.exchange_targets):
            return "exchange"
        return None
    pref = TEMPLATE_ACTION.get(facts.template)
    if facts.template == "foreign_order_refusal":
        pref = facts.foreign_request
    if pref in ("exchange", "return") and "modify_payment" in hits and not re.search(
            r"\b(?:payment method|pay)\b[^.?!]{0,30}\b(?:for|of|on) (?:the |your |this )?order\b", full, re.I):
        # payment talk inside an exchange / return is about the price difference or the refund
        hits[pref] = hits.get(pref, 0) + hits.pop("modify_payment")
    if pref == "return" and "return" in hits:
        return "return"
    if pref in hits:
        # "refund" words also fire inside cancel / exchange / modify texts
        if pref != "return" or hits.get("return"):
            return pref
    if "return" in hits and len(hits) > 1:
        # refunds are mentioned by every action; prefer the specific one
        rest = {a: n for a, n in hits.items() if a != "return"}
        if rest:
            return max(rest, key=lambda a: (rest[a], a == pref))
    return max(hits, key=lambda a: (hits[a], a == pref))


def _proposed_action(full: str, sents: list[str], req_sents: list[str], facts: TaskFacts, state: NLUState,
                     sem: Semantics) -> Optional[ProposedAction]:
    action = _detect_action(full, facts)
    if action is None or action == "transfer":
        return None
    low = full.lower()
    # "instead of #W1, I'll cancel #W2": an id the turn explicitly sets aside is not a proposal target
    order_ids = list(dict.fromkeys(
        m.group(0) for m in ORDER_ID_RE.finditer(full)
        if not re.search(r"\b(?:instead of|rather than|not|no longer|other than|replacing)\W*(?:order\W*)?$",
                         full[max(0, m.start() - 25):m.start()], re.I)))
    target = facts.target_order
    names_target = target in order_ids
    others = [o for o in order_ids if o != target]
    # Specificity: an order, an item, a payment or a reason must be named, or the turn asks consent / claims done.
    item_names = facts.order_items.get(target, [])
    spans_all = _item_spans(full, item_names)
    items_named = [n for n in dict.fromkeys(item_names) if n.lower() in low and not all(
        re.search(r"\b(?:remain(?:s)? unchanged|unchanged|remain the same|stay(?:s)? the same|keep(?:ing)? (?:the )?current|"
                  r"not be (?:exchanged|returned)|no change|keeping it|(?:is|are) (?:currently )?(?:not |un)available|"
                  r"unavailable|cannot be (?:exchanged|returned))\b", full[a:min(b, a + 120)], re.I)
        or re.search(r"\b(?:not (?:exchanging|returning|exchange|return)|without (?:exchanging|returning)|except(?: for)?|excluding|"
                     r"(?:cannot|can't|unable to) (?:proceed with )?(?:the )?(?:exchange|return)(?: (?:for|of))?|drop(?:ping)?|"
                     r"keep(?:ing)?|instead of)\W+(?:the |your |current )*\W*$", full[max(0, a - 60):a], re.I)
        for a, b in spans_all.get(n, [(0, 0)]))]
    reason = None
    if REASON_PAIR_RE.search(low):
        reason = None
    elif REASON_OBM_RE.search(low):
        reason = "ordered by mistake"
    elif REASON_NLN_RE.search(low):
        reason = "no longer needed"
    # "... with the exchange of the Tea Kettle and Desk Lamp only": an explicit item list overrides other mentions
    for s in sents:
        if re.search(r"\bonly\b", s, re.I) and re.search(r"\b(?:exchang|return|proceed|swap)\w*", s, re.I):
            in_s = [n for n in items_named if n.lower() in s.lower()]
            if in_s and len(in_s) < len(items_named) and re.search(
                    r"\b(?:only (?:the )?" + "|".join(re.escape(n.lower()) for n in in_s) + r")|(?:" +
                    "|".join(re.escape(n.lower()) for n in in_s) + r") only\b", s.lower()):
                items_named = in_s
    pm_mentions = []
    for s in sents:
        for m in _resolve_pm_mentions(s, facts):
            pm_mentions.append((s, m))
    proposed_pms = [(_canon_pm(m.pm, facts), s, m) for s, m in pm_mentions if m.proposed
                    and m.pm not in ("credit_card", "gift_card", "unknown")
                    and not (action == "modify_payment" and m.pm == "original")
                    and not (action == "modify_payment" and re.search(r"\brefund", s, re.I))]
    # a method the user does not own, proposed as the new one ("switch it to your gift card" with no gift card)
    proposed_unknown = any(m.proposed and m.pm == "unknown" for s, m in pm_mentions
                           if not (action == "modify_payment" and re.search(r"\brefund", s, re.I)))
    # "the original payment method (Mastercard ending 1111)": the parenthesised method is an alias of "original"
    # (for a payment change, the current method -- dropped with it)
    orig_spans = [(s, m) for s, m in pm_mentions if m.pm == "original"]
    proposed_pms = [x for x in proposed_pms if not any(
        x[1] == s2 and 0 <= x[2].start - m2.end <= 12 and re.match(r"^\W*(?:\(|,|--|which is|i\.e\.)", s2[m2.end:x[2].start] + "(")
        for s2, m2 in orig_spans)]
    specific = bool(order_ids or items_named or sem.confirmation_request or sem.claims_action_done or proposed_pms
                    or (reason and action == "cancel"))
    if not specific:
        return None
    # Identity-only turns ("to cancel your order I first need your email") propose nothing yet.
    if "identity" in sem.info_requests and not (sem.confirmation_request or sem.claims_action_done):
        return None
    if sem.refusal == "ownership" and not sem.confirmation_request:
        return None
    if not names_target and not others and facts.template == "exchange" and _names_hint(low, facts.hint_products):
        # Gold labels count an exchange that names the target's items as naming the target (most batches).
        names_target = True
    pa = ProposedAction(action=action, names_target=names_target, names_wrong_target=bool(others), named_orders=order_ids)
    cur = facts.current_order(state)
    pa.names_current_target = cur in order_ids or (not order_ids and _names_hint(low, facts.current_hints(state)))
    if facts.first_order:
        stale = facts.first_order if state.correction_used else None
        if stale and stale in order_ids:
            pa.names_stale_target = True
        if state.correction_used and not order_ids and _names_hint(low, facts.first_hint_products) and not _names_hint(low, facts.hint_products):
            pa.names_stale_target = True
    # products that only appear in the user's other orders
    if not others and facts.template != "foreign_order_refusal":
        own_names = set(n.lower() for n in facts.order_items.get(cur, []))
        other_names = set()
        for oid, names in facts.order_items.items():
            if oid != cur:
                other_names |= {n.lower() for n in names}
        foreign_only = [n for n in other_names - own_names if len(n) > 3 and re.search(rf"\b{re.escape(n)}\b", low)]
        if foreign_only and not any(n.lower() in low for n in facts.current_hints(state)):
            pa.names_current_target = False
    details = pa.details
    mismatches: list[str] = []
    missing: list[str] = []
    # reason
    if action == "cancel":
        details["reason"] = reason
        if reason is None:
            missing.append("reason")
        elif facts.reason and reason != facts.reason:
            mismatches.append("reason")
    # payment
    pm_choice = set(p for p, _, _ in proposed_pms if p not in ("unknown",))
    pay_pm: Optional[str] = None
    if action in ("exchange", "return", "modify_payment", "modify_items"):
        cands = [p for p, _, _ in proposed_pms]
        distinct = list(dict.fromkeys(cands))
        if len(distinct) == 1:
            pay_pm = distinct[0]
        elif len(distinct) > 1:
            if "payment_method" in sem.info_requests:
                pay_pm = None  # a choice between methods
            else:
                pay_pm = _pick_proposed_pm(proposed_pms, facts, action)
        if pay_pm is not None:
            details["payment"] = _pm_label(pay_pm, facts)
            pa.payment_pm = pay_pm
        want = _expected_pm(facts, state, action)
        if pay_pm is None and proposed_unknown:
            mismatches.append("payment")
        elif pay_pm is None:
            if action in ("exchange", "modify_payment", "return"):
                missing.append("payment")
        elif want is not None:
            if not _pm_equal(pay_pm, want, facts):
                mismatches.append("payment")
        del pm_choice
    # items
    if action in ("return", "exchange"):
        if items_named:
            details["items"] = items_named
        wanted_items = facts.request_items if action == "return" else [n for n, _ in facts.wanted_targets(state)]
        if facts.template == "foreign_order_refusal":
            wanted_items = facts.request_items
        whole = re.search(r"\b(?:(?:return|exchang\w*|send back|refund)\w*\s+(?:all (?:the |of the )?items|everything)|"
                          r"(?:the )?(?:entire|whole) order|all (?:the |of the )?items in (?:the |your |this )?order)\b", low)
        if wanted_items and items_named:
            # extra items count only when named in a sentence that proposes the action itself, not in a listing
            # of the order's contents ("the order contains the following items: ...")
            act_rx = r"\b(?:return\w*|send(?:ing)? back|exchang\w*|swap\w*)\b"
            prop_items = {n for s_ in sents if re.search(act_rx, s_, re.I) and not re.search(
                r"\b(?:contains?|includes?|including|following items|other items|items in (?:the|your|this) order|"
                r"currently|options?:|available)\b", s_, re.I) for n in items_named if n.lower() in s_.lower()}
            if not all(n in items_named for n in wanted_items) or any(n not in wanted_items for n in prop_items):
                mismatches.append("items")
        elif wanted_items and whole and len(set(wanted_items)) < len(set(item_names)):
            mismatches.append("items")
        elif wanted_items and not items_named:
            missing.append("items")
    # options (exchange)
    if action == "exchange" and facts.exchange_targets:
        opts, bad, stale = _option_check(full, facts, state)
        if opts:
            details["new_options"] = opts
        if bad:
            mismatches.append("options")
        pa.stale_options = stale
        if not opts:
            missing.append("options")
    pa.mismatches = mismatches
    pa.missing = missing
    stated = any(details[k] for k in ("reason", "payment", "items", "new_options"))
    relevant_stated = stated
    if action == "cancel":
        relevant_stated = details["reason"] is not None
    if not relevant_stated:
        pa.details_match_task = None
    else:
        pa.details_match_task = not mismatches
    want_action = TEMPLATE_ACTION.get(facts.template) or (facts.foreign_request if facts.template == "foreign_order_refusal" else None)
    if want_action and action != want_action:
        pa.mismatches.append("action")
        pa.details_match_task = False if stated else None
    return pa


def _names_hint(low: str, hints: list[str]) -> bool:
    return any(h and h.lower() in low for h in hints)


def _pick_proposed_pm(proposed: list[tuple[str, str, _PMMention]], facts: TaskFacts, action: str) -> Optional[str]:
    # Prefer a mention inside an explicit proposal frame ("to your X", "refund ... X", "New payment method: X").
    frame = re.compile(r"(?:\bto|\binto|\bonto|\bvia|\busing|\buse|\bwith|\bthrough|method:|refund(?:ed)? to|charged to|go(?:es)? to|"
                       r"sent to|issued to|processed to|new payment method(?: is| will be)?:?)\W*(?:your |the |my |a |this |that )?(?:existing |original )?$", re.I)
    framed = [p for p, s, m in proposed if frame.search(s[max(0, m.start - 30):m.start])]
    if framed:
        return framed[-1]
    return proposed[-1][0] if proposed else None


def _expected_pm(facts: TaskFacts, state: NLUState, action: str) -> Optional[str]:
    if facts.template == "return_fallback":
        return facts.gold_pm or facts.original_pm
    if facts.template == "foreign_order_refusal":
        return facts.original_pm
    if facts.template == "modify_payment":
        return facts.wanted_pm(state)
    if facts.template == "exchange":
        return facts.gold_pm
    return None


def _pm_equal(said: str, want: str, facts: TaskFacts) -> bool:
    if said == want:
        return True
    if said == "credit_card":
        return facts.payment_methods.get(want, {}).get("source") == "credit_card"
    if said == "gift_card":
        return facts.payment_methods.get(want, {}).get("source") == "gift_card"
    return False


def _pm_label(pm: str, facts: TaskFacts) -> str:
    d = facts.payment_methods.get(pm)
    if d is None:
        return pm
    return _pm_phrase(d).replace("my ", "", 1)


def _option_check(full: str, facts: TaskFacts, state: NLUState) -> tuple[list[str], bool, bool]:
    """(stated new options, contradicts the wanted change, states a retracted option)."""
    names = [it["name"] for it in facts.target_items]
    spans = _item_spans(full, names)
    current = {it["name"]: it["options"] for it in facts.target_items}
    stated: list[str] = []
    bad = False
    stale = False
    retracted = {n: d for n, d in facts.retracted_options(state)}
    for name, diff in facts.wanted_targets(state):
        cur = current.get(name, {})
        prod_opts = facts.product_options.get(name, {})
        texts = [full[a:b] for a, b in spans.get(name, [])]
        if not texts:
            continue
        seg = " ".join(texts)
        found = {key: bool(_value_rx(key, val).search(seg)) for key, val in diff.items()}
        for key, val in diff.items():
            if found[key]:
                stated.append(f"{key} {val}")
        n_before = len(stated)
        for key, values in prod_opts.items():
            want = diff.get(key, cur.get(key))
            for v in values:
                if v == want or v == cur.get(key):
                    continue
                if len(v) <= 1:
                    continue
                rx = _value_rx(key, v)
                for t in texts:
                    if not rx.search(t):
                        continue
                    if name in retracted and retracted[name].get(key) == v:
                        stale = True
                        bad = True
                    elif _listing(t):
                        continue
                    else:
                        bad = True
                        stated.append(f"{key} {v}")
                    break
        # new options stated for this item, but not every wanted change among them (e.g. a recap that
        # ignores a late correction): the stated plan is not what the user wants
        if (any(found.values()) or len(stated) > n_before) and not all(found.values()):
            bad = True
            if facts.exchange_correction and state.correction_used and facts.exchange_correction[0] == name:
                stale = True
    return stated, bad, stale


def _listing(seg: str) -> bool:
    """A segment that lists several alternatives (an option-choice question), not a single proposal."""
    return len(re.findall(r"item id|\$\d", seg, re.I)) >= 3 or bool(re.search(r"\b(?:options|alternatives|available (?:variants|options|configurations))\b", seg, re.I))


# =============================================================================
# LLM backend (drift checks only)
# =============================================================================

LABEL_DEFINITIONS = """You label what ONE customer-service agent turn DOES, independent of how a customer would react.
Fields:
- info_requests: things the agent asks the user to PROVIDE or DECIDE this turn, any of identity, order_id, reason,
  payment_method, all_items_complete, option_choice, item_choice, order_choice, other_info. A sentence like
  "I cannot provide details" or "please note ..." is not a request.
- identity_target: "caller" | "third_party" | null -- whose identity an identity request is about.
- confirmation_request: true only when the agent asks consent to perform an action NOW. Sign-offs ("let me know
  if you need anything else") and offers of other help ("would you like me to transfer you?") are not.
- proposed_action: null, or the action proposed / described: {action: cancel|exchange|return|modify_payment|
  modify_items|modify_address|modify_user_address|transfer|other, names_target (the task's target order id is named),
  names_wrong_target (another order id is named), details: {reason, new_options, payment, items},
  details_match_task (compare the stated reason / options / payment / items with the task facts; null when nothing
  is stated)}.
- refusal: null | ownership | status | constraint | other.
- constraint_statement: the turn states a policy rule or limit.
- offers_other_help_only: the only offer/question is other help (transfer, anything else).
- claims_action_done: the agent says the action was carried out.
- narration_only: pure narration with no request, offer, refusal or claim.
Answer with one JSON object with exactly these keys."""

LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "info_requests": {"type": "array", "items": {"type": "string", "enum": list(INFO_KINDS)}},
        "identity_target": {"type": ["string", "null"], "enum": ["caller", "third_party", None]},
        "confirmation_request": {"type": "boolean"},
        "proposed_action": {"type": ["object", "null"]},
        "refusal": {"type": ["string", "null"], "enum": [*REFUSALS, None]},
        "constraint_statement": {"type": "boolean"},
        "offers_other_help_only": {"type": "boolean"},
        "claims_action_done": {"type": "boolean"},
        "narration_only": {"type": "boolean"},
    },
    "required": ["info_requests", "identity_target", "confirmation_request", "proposed_action", "refusal",
                 "constraint_statement", "offers_other_help_only", "claims_action_done", "narration_only"],
}


def task_facts_prompt(facts: TaskFacts, state: NLUState) -> str:
    want_pm = facts.wanted_pm(state)
    return json.dumps({
        "template": facts.template,
        "target_order": facts.target_order,
        "order_first_named_by_user": facts.first_order,
        "late_correction_given": state.correction_used,
        "payment_fallback_given": state.fallback_used,
        "cancel_reason": facts.reason,
        "exchange_wanted_changes": facts.wanted_targets(state),
        "items_to_return": facts.request_items,
        "wanted_payment_method": _pm_label(want_pm, facts) if want_pm else None,
        "gold_payment_method": _pm_label(facts.gold_pm, facts) if facts.gold_pm else None,
        "original_payment_method": _pm_label(facts.original_pm, facts) if facts.original_pm else None,
        "expect_refusal": facts.expect_no_write,
        "user_orders": facts.order_items,
    }, default=list)


def _semantics_from_label(d: dict[str, Any]) -> Semantics:
    pa = d.get("proposed_action")
    pact = None
    if isinstance(pa, dict) and pa.get("action"):
        det = pa.get("details") or {}
        pact = ProposedAction(
            action=pa["action"], names_target=bool(pa.get("names_target")),
            names_wrong_target=bool(pa.get("names_wrong_target")),
            details={k: det.get(k) for k in ("reason", "new_options", "payment", "items")},
            details_match_task=pa.get("details_match_task"),
        )
        pact.names_current_target = pact.names_target
    return Semantics(
        info_requests=[x for x in d.get("info_requests") or [] if x in INFO_KINDS],
        identity_target=d.get("identity_target"), confirmation_request=bool(d.get("confirmation_request")),
        proposed_action=pact, refusal=d.get("refusal"), constraint_statement=bool(d.get("constraint_statement")),
        offers_other_help_only=bool(d.get("offers_other_help_only")), claims_action_done=bool(d.get("claims_action_done")),
        narration_only=bool(d.get("narration_only")),
    )


class LLMAnalyzer:
    """Labels a turn with an OpenAI-compatible chat model (temperature 0, JSON
    schema output), caching every answer on disk by a hash of (model, prompt).
    Never used in tests or training: an API call per user turn and a second
    sampler inside every reward would break GRPO's group comparison."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 cache_dir: str | Path = "data/episodes/nlu_llm_cache", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def _messages(self, text: str, facts: TaskFacts, state: NLUState) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": LABEL_DEFINITIONS},
            {"role": "user", "content": f"Task facts:\n{task_facts_prompt(facts, state)}\n\nAgent turn:\n{text}"},
        ]

    def analyze(self, text: str, task: Any, state: Optional[NLUState] = None) -> Semantics:
        facts = facts_from(task)
        state = state or NLUState()
        messages = self._messages(text, facts, state)
        key = hashlib.sha256(json.dumps([self.model, messages], sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return _semantics_from_label(json.loads(path.read_text()))
        import urllib.request

        body = {
            "model": self.model, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "turn_semantics", "schema": LABEL_SCHEMA}},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 -- configured endpoint
            out = json.loads(resp.read())
        label = json.loads(out["choices"][0]["message"]["content"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(label))
        return _semantics_from_label(label)


_DEFAULT = RulesAnalyzer()


def make_analyzer(config: Optional[dict[str, Any]] = None) -> Analyzer:
    """`{"backend": "rules"}` (default) or `{"backend": "llm", "base_url": ..., "model": ..., "cache_dir": ...}`.
    The env var TAU_FORGE_NLU_BACKEND=llm (+ TAU_FORGE_NLU_BASE_URL / _MODEL) selects the LLM backend
    when no config is given."""
    cfg = dict(config or {})
    if not cfg and os.environ.get("TAU_FORGE_NLU_BACKEND") == "llm":
        cfg = {"backend": "llm", "base_url": os.environ.get("TAU_FORGE_NLU_BASE_URL", ""),
               "model": os.environ.get("TAU_FORGE_NLU_MODEL", "")}
    if cfg.get("backend", "rules") == "rules":
        return _DEFAULT
    if cfg["backend"] == "llm":
        return LLMAnalyzer(cfg["base_url"], cfg["model"], cfg.get("api_key"),
                           cfg.get("cache_dir", "data/episodes/nlu_llm_cache"))
    raise ValueError(f"unknown NLU backend {cfg['backend']!r}")


def analyze(text: str, task: Any, state: Optional[NLUState] = None) -> Semantics:
    """The rules analyzer: `RulesAnalyzer().analyze`."""
    return _DEFAULT.analyze(text, task, state)


__all__ = [
    "ACTIONS", "INFO_KINDS", "REFUSALS", "Analyzer", "LLMAnalyzer", "NLUState", "ProposedAction", "RulesAnalyzer",
    "Semantics", "TaskFacts", "analyze", "facts_from", "make_analyzer", "sentences",
]
