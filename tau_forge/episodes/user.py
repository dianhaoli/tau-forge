"""The deterministic scripted user an episode's policy talks to (stage B).

Why scripted, not tau2's LLM user simulator: a GRPO group must differ only by
the policy's samples. An LLM user adds its own sampling noise to every reward
(and an API call per turn); a rules user replies identically to identical
agent text, so all variance within a group is the policy's. The cost is that
GRPO probes the script for quirks -- hence the anti-exploit rules below, and
the README's note to track drift against tau2's LLM user.

How it reads a turn. Every agent TEXT turn (tool calls never reach the user)
goes through `nlu.analyze` (`tau_forge/episodes/nlu.py`), which says what the
turn DOES: which information it asks for, whether it asks consent to act now,
what action it proposes with which details and whether those match the task,
whether it refuses and why. The reply policy below works on those semantics,
never on keyword hits over the whole turn. The NLU is measured against 806
adjudicated real Qwen3-4B turns (`scripts/nlu_eval.py`).

Reply policy, in priority order:
  1. a write has succeeded                     -> thanks + STOP
  2. refusal task: an ownership refusal (declarative, by meaning -- not a
     question about ownership, not a plan to check) that does not offer to go
     ahead with the refused action in the same turn -> accept + STOP. Offers
     of other help (transfer, "anything else?") do not veto it, and neither
     does a missing auth: a blind refusal is accepted and the reward gate pays
     it 0.4 instead of 1.0. Questions about whose order it is get an answer.
  3. fallback pending, and the turn refuses the asked payment method or states
     the constraint about it (refund destination, gift card balance), after
     the agent has read the order -> the fallback (once; revokes all consent).
     If the same turn already recaps the post-fallback plan, the fallback line
     is followed by the consent.
  4. a consent request proposing the template action:
       * a choice between several orders, or a recap of another of the user's
         orders                              -> the order answer, no yes
       * a recap of the order a late correction retracted -> the correction again
       * names no order / product the user means -> "spell out the details"
       * the target order not read yet (when the runner says)     -> same
       * a product-hint user and a bare order id it was never told -> "is that
         the one with the <product>?" (the user does not know its order ids)
       * a late correction not used yet       -> the correction (revokes)
       * a fallback not used yet: the user still wants the method it asked for,
         so it restates it ("that's what I want") or objects to another one,
         and never consents -- consent is per slot, not per payment method, so
         a yes here would also cover a write to the method the agent never
         explained
       * details that contradict the task (reason, option values, payment,
         items, the action) -> a correction line stating the right detail,
         no consent, revokes
       * otherwise yes (`confirms` = the task's slots); if a detail the user
         has to provide is missing and the user never said it (cancel reason,
         exchange / modify payment method) -> "Yes -- and <the detail>"
     Every information request in the same turn is answered after the consent.
  5. information requests, every one answered in one reply in a fixed order:
     identity (a third party's identity -> "I don't have their details, I'm the
     one calling", once; an email request to a name+zip user -> "I don't have
     it handy, look me up by name and zip", once), order id / which order,
     which item, which option (the hidden option change), reason, payment
     method, "is that everything?" ("That's everything." -- no line starts
     with "Yes" unless it consents), other.
  6. a claim that the action was done when no write succeeded, a bare consent
     request, a sign-off, an offer of other help, anything else -> restate
     (unrecognised); pure narration -> "Okay." (also unrecognised).
Give-up: the third consecutive identical information answer (or the same
correction / objection three times) with no progress in between, or the third
consecutive unrecognised turn, ends the episode with STOP.

Every line is drawn from a paraphrase pool with an rng seeded by (task id,
seed): deterministic per task, but no single surface form for the policy to
key on across tasks. The user never says an id other than an order id or an
email (`generate.verify_task` checks the profile; the lines built here use
only profile phrases, product names and option values).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from tau_forge.episodes.nlu import (
    NLUState,
    Semantics,
    TEMPLATE_ACTION,
    TaskFacts,
    _pm_phrase,
    facts_from,
    make_analyzer,
    sentences,
)
from tau_forge.episodes.task import ORDER_ID_RE, STOP, EpisodeTask

if TYPE_CHECKING:
    from tau_forge.episodes.runner import WriteRecord

MAX_SAME_ANSWER = 3  # consecutive identical answers (no progress in between) before giving up
MAX_UNRECOGNISED = 3  # consecutive unrecognised turns before giving up
YES = ["Yes, please proceed.", "Yes, go ahead.", "Yes, that's right -- please do it.", "Yes, I confirm."]
YES_AND = ["Yes, please go ahead -- and {d}", "Yes -- and {d}"]
THANKS = ["Great, thank you! That's all I needed.", "Perfect, thanks for your help.", "Thanks, that's everything."]
RESTATE = ["Sorry, to be clear: ", "Just to repeat what I need: ", "Let me say it again: "]
GIVE_UP = ["I don't think this is going anywhere. Bye.", "Never mind, I'll try again some other time."]
BE_SPECIFIC = [
    "Before I say yes, can you tell me exactly which order and items this is for?",
    "Sorry, what exactly are you going to do? Please spell out the details first.",
]
GROUND_FIRST = [
    "Before I say yes, could you pull up the order and tell me exactly what you'll do?",
    "Hold on -- could you look up the order first and walk me through the details?",
]
ALL_ITEMS = ["That's everything.", "That's all of it.", "Just those, nothing else."]
ACK = ["Okay.", "Alright.", "Sure."]
NOT_THAT_ORDER = ["No, that's not the order I mean. ", "That's not the right order. "]
DONT_KNOW_ID = [
    "I don't know the order number -- is that the one with the {h}?",
    "I'm not sure about the order number. Is that the order with the {h}?",
]
NO_EMAIL = [
    "I don't have my email handy -- could you look me up by name and zip code instead? {i}",
    "I can't remember which email I used. Can you find me by name and zip? {i}",
]
THIRD_PARTY = [
    "I don't have {p} account details -- I'm the one calling, and {i}",
    "I can't give you {p} details, sorry -- it's me calling. {i}",
]
RELATION = ["It's my {r}'s order -- I'm asking on {p} behalf.", "It's my {r}'s, not mine. I'm just helping {o} out."]
STILL_WANT = {
    "return_fallback": ["That's what I want -- the refund to {pm}.", "Right, I'd like the refund to go to {pm}."],
    "modify_payment": ["That's what I want -- put it on {pm}.", "Right, I'd like it paid with {pm}."],
}
OBJECT_PM = {
    "return_fallback": ["Wait -- I asked for the refund to go to {pm}.", "No, I'd like the refund on {pm}, please."],
    "modify_payment": ["Wait -- I want it paid with {pm}.", "No, I'd like it on {pm}, please."],
    "exchange": ["No -- please use {pm} for any difference.", "Not that one -- I'd like to use {pm}."],
    "post_fallback_return": ["No -- the original payment method is fine.", "No, please refund it to the original payment method."],
}
WRONG_REASON = ["Actually, no -- {r}", "That's not quite it -- {r}"]
WRONG_ITEMS = ["That's not quite right -- I want to {verb} the {items}.", "Not exactly: it's the {items} I want to {verb}."]
WRONG_OPTIONS = [
    "That's not what I asked for -- I want {changes}, with everything else the same as what I have now.",
    "No, the change I need is {changes}; everything else stays the same as what I have now.",
]
WRONG_ACTION = ["No, that's not what I want. ", "That's not what I asked for. "]

_FEM = {"sister", "mother", "mom", "wife", "daughter", "aunt", "grandmother"}
_MASC = {"brother", "father", "dad", "husband", "son", "uncle", "grandfather"}
_INFO_ORDER = ("identity", "order_id", "order_choice", "item_choice", "option_choice", "reason", "payment_method",
               "all_items_complete", "other_info")
_VERB = {"exchange": "exchange", "return_fallback": "return", "cancel": "cancel", "modify_payment": "change the payment on"}


@dataclass
class TurnContext:
    """What the runner tells the user about the episode before a text turn.
    The user never sees tool calls; this is the state a real customer would
    know or be told about (and what the runner's own gates use)."""

    # Write-tool calls since the previous user turn, succeeded or failed.
    new_writes: list["WriteRecord"] = field(default_factory=list)
    # Earned auth so far (runner log); None before any.
    authed_user: Optional[str] = None
    read_orders: list[str] = field(default_factory=list)
    # Some write has succeeded in this episode.
    any_write_ok: bool = False


@dataclass
class UserReply:
    text: str
    stop: bool
    # Primary intent, for logs and back-compat.
    intent: str
    # Every intent answered this turn (multi-answer replies); defaults to [intent].
    intents: list[str] = field(default_factory=list)
    # Consent slot ids (`EpisodeTask.slots`) this reply says yes to -- the
    # runner's confirmation gate keys off this, not off the text.
    confirms: list[str] = field(default_factory=list)
    # The plan changed (late correction, fallback, a corrected detail): clear all pending consent.
    revokes: bool = False
    # Back-compat: True iff `confirms` is non-empty.
    is_yes: bool = False
    # Back-compat: this reply answered a target-naming confirmation request
    # without a yes (the late correction).
    answered_recap: bool = False

    def __post_init__(self) -> None:
        if not self.intents:
            self.intents = [self.intent]
        if self.confirms:
            self.is_yes = True


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _lower_first(s: str) -> str:
    return s if re.match(r"^I\b", s) else s[0].lower() + s[1:]


def _change_phrase(key: str, value: str) -> str:
    from tau_forge.episodes.generate import change_phrase

    return change_phrase(key, value)


COMPLY_RE = re.compile(
    r"\b(?:i(?:'ll| will(?! not)| can(?!'t|’t|not| only)| could(?!n't|n’t| not| only)| am going to|'m going to)|let me(?! know)|"
    r"we(?:'ll| will(?! not)| can(?!'t|’t|not| only))|"
    r"(?:would|do) you (?:still )?(?:like|want) me to|shall i|should i)\b"
    r"(?:(?!\b(?:not|cannot|unable|never|only)\b|n't\b)[^.?!]){0,50}\b(?:cancel\w*|return\w*|refund\w*|process\w*|proceed|go ahead)\b",
    re.I,
)
REFUSING_RE = re.compile(r"\b(?:can(?:no|'|’)?t|cannot|unable to|not able to|won't|will not|not (?:allowed|permitted|possible))\b", re.I)
OWN_ORDERS_RE = re.compile(r"\b(?:your own|instead|another order|other orders?|one of your orders|orders? (?:on|in|under) your)\b", re.I)


class ScriptedUser:
    def __init__(self, task: EpisodeTask, seed: int = 0, analyzer: Any = None):
        self.task = task
        self.p = task.profile
        self.rng = random.Random(f"{task.id}/{seed}")
        self.analyzer = analyzer if analyzer is not None else make_analyzer()
        self.facts: TaskFacts = facts_from(task)
        self.correction_used = False
        self.fallback_used = False
        self.n_unrecognised = 0  # consecutive
        self._last_answer: Optional[tuple] = None
        self._answer_streak = 0
        self._third_party_said = False
        self._no_email_said = False
        self._known_ids: set[str] = set()  # order ids the agent tied to a product the user knows
        self._correction_repeated = False
        hidden = task.hidden or {}
        mode = task.difficulty.get("id_mode")
        if mode is None:
            mode = "email" if any("@" in x for x in self.p.get("identity", [])) else "name_zip"
        self.id_mode = mode
        self._told: set[str] = set()
        low_open = task.opening.lower()
        if any(line.lower().rstrip(".") in low_open for line in self.p.get("reason", [])) or (
                hidden.get("reason") and hidden["reason"] in low_open):
            self._told.add("reason")
        if self.facts.template in ("return_fallback", "modify_payment"):
            self._told.add("payment")
        if self.facts.template == "foreign_order_refusal":
            self._told.add("reason")

    # ------------------------------------------------------------------ lines

    def _pick(self, pool: list[str]) -> str:
        return self.rng.choice(pool)

    def _say(self, key: str) -> str:
        """A line from the task's pool for `key`; after the late correction (or
        the fallback) its `<key>_after_correction` (`_after_fallback`) variant
        when one exists -- the order the user now means, or a restate that no
        longer names the forbidden method."""
        if self.correction_used and f"{key}_after_correction" in self.p:
            key = f"{key}_after_correction"
        elif self.fallback_used and f"{key}_after_fallback" in self.p:
            key = f"{key}_after_fallback"
        if self.facts.template == "foreign_order_refusal" and key in ("restate", "reason"):
            return self._foreign_line(key)
        pool = self.p[key]
        return pool if isinstance(pool, str) else self.rng.choice(pool)

    def _foreign_line(self, key: str) -> str:
        """Restate and reason consistent with the opening (whose relation and
        reason the generator's profile did not always match)."""
        rel = self.facts.relation or "family member"
        oid = self.facts.target_order
        if key == "reason":
            return "It was ordered by mistake." if self.facts.reason == "ordered by mistake" else "They don't need it anymore."
        if self.facts.foreign_request == "return":
            items = self.facts.request_items
            what = f"the {_join(items)}" if items else "the item"
            return f"I want to return {what} from order {oid} for my {rel}."
        return f"I want order {oid} cancelled for my {rel}."

    def _pronoun(self) -> tuple[str, str]:
        rel = self.facts.relation or ""
        if rel in _FEM:
            return "her", "her"
        if rel in _MASC:
            return "his", "him"
        return "their", "them"

    def _caller_identity(self) -> str:
        h = self.task.hidden or {}
        if self.id_mode == "email" and h.get("email"):
            return f"My email is {h['email']}."
        if h.get("first_name"):
            return f"I'm {h['first_name']} {h['last_name']}, zip code {h['zip']}."
        return self._say("identity")

    def _name_zip(self) -> str:
        h = self.task.hidden or {}
        return f"I'm {h.get('first_name', '')} {h.get('last_name', '')}, zip code {h.get('zip', '')}."

    def _pm_text(self, pm_id: Optional[str]) -> str:
        d = self.facts.payment_methods.get(pm_id or "")
        if d is None:
            return "the original payment method"
        return _pm_phrase(d)

    def _wanted_pm_text(self) -> str:
        f = self.facts
        if f.template == "return_fallback":
            return "the original payment method" if self.fallback_used else (self.task.hidden.get("bad_pm_phrase")
                                                                              or self._pm_text(f.asked_pm))
        if f.template == "modify_payment":
            if self.fallback_used and self.task.hidden.get("target_pm_phrase"):
                return self.task.hidden["target_pm_phrase"]
            return self.task.hidden.get("asked_pm_phrase") or self._pm_text(f.asked_pm)
        if f.template == "exchange":
            return self.task.hidden.get("pm_phrase") or self._pm_text(f.gold_pm)
        return ""

    def _changes_text(self) -> str:
        parts = []
        for name, diff in self.facts.wanted_targets(self._state()):
            if diff:
                parts.append(f"the {name} " + " and ".join(_change_phrase(k, v) for k, v in diff.items()))
        return _join(parts) if parts else "the change I asked for"

    def _state(self) -> NLUState:
        return NLUState(correction_used=self.correction_used, fallback_used=self.fallback_used)

    # ------------------------------------------------------------ bookkeeping

    def _reply(self, text: str, intents: list[str], *, confirms: Optional[list[str]] = None, revokes: bool = False,
               answered_recap: bool = False, key: Optional[tuple] = None, progress: bool = False) -> UserReply:
        """Every non-terminal reply goes through here: the same-answer streak
        (identical answers with no progress in between) and the consecutive
        unrecognised counter."""
        if intents[0] == "unrecognised":
            self.n_unrecognised += 1
            self._last_answer, self._answer_streak = None, 0
            if self.n_unrecognised >= MAX_UNRECOGNISED:
                return UserReply(f"{self._pick(GIVE_UP)} {STOP}", True, "give_up")
            return UserReply(text, False, "unrecognised", intents)
        self.n_unrecognised = 0
        if progress or key is None:
            self._last_answer, self._answer_streak = None, 0
        else:
            self._answer_streak = self._answer_streak + 1 if key == self._last_answer else 1
            self._last_answer = key
            if self._answer_streak >= MAX_SAME_ANSWER:
                return UserReply(f"{self._pick(GIVE_UP)} {STOP}", True, "give_up")
        return UserReply(text, False, intents[0], intents, confirms=list(confirms or []), revokes=revokes,
                         answered_recap=answered_recap)

    def _unrecognised(self, line: str) -> UserReply:
        return self._reply(line, ["unrecognised"])

    def _restate(self) -> UserReply:
        return self._unrecognised(self._pick(RESTATE) + self._say("restate"))

    def _yes_confirms(self) -> list[str]:
        return [s["id"] for s in self.task.slots]

    # ------------------------------------------------------------ main policy

    def reply(self, agent_text: str, ctx: Optional[TurnContext] = None, write_succeeded: bool = False) -> UserReply:
        txt = agent_text or ""
        if ctx is not None and ctx.any_write_ok:
            write_succeeded = True
        if write_succeeded:
            return UserReply(f"{self._pick(THANKS)} {STOP}", True, "thanks")
        f = self.facts
        sem = self.analyzer.analyze(txt, self.task, self._state())
        self._learn_ids(txt)

        # 2. refusal tasks
        if f.expect_no_write:
            if sem.refusal == "ownership" and not self._offers_refused_action(sem, txt):
                return UserReply(f"{self._say('accept_denial')} {STOP}", True, "accept_denial")

        # 3. the fallback
        if self.p.get("fallback") and not self.fallback_used and (
                sem.refusal == "constraint" or sem.constraint_about_payment):
            if self._grounded(ctx):
                return self._fallback(txt, sem, ctx)

        parts: list[str] = []
        intents: list[str] = []
        confirms: list[str] = []
        revokes = False
        answered_recap = False
        progress = False
        covered: set[str] = set()

        # 4. consent requests
        pa = sem.proposed_action
        if sem.confirmation_request and pa is not None and pa.action != "transfer":
            d = self._consent(sem, ctx, txt)
            if d is not None:
                kind, text, intent, covered = d
                parts.append(text)
                intents.append(intent)
                if kind == "yes":
                    confirms = self._yes_confirms()
                    progress = True
                elif kind == "revoke":
                    revokes = True
                    progress = intent == "correction" and not self._correction_repeated
                    answered_recap = intent == "correction"
                elif kind == "unrecognised" and not self._info_requests(sem, covered):
                    return self._unrecognised(text)
                elif kind == "unrecognised":
                    parts.pop()
                    intents.pop()

        # a proposal (no consent asked) that still targets the retracted or another order: object
        if pa is not None and not intents and not sem.claims_action_done and not f.expect_no_write \
                and pa.action == TEMPLATE_ACTION.get(f.template):
            cur = f.current_order(self._state())
            if pa.names_stale_target:
                parts.append(self._say("correction"))
                intents.append("correction")
                revokes = True
            elif pa.named_orders and cur not in pa.named_orders and any(o in f.order_items for o in pa.named_orders):
                parts.append(self._pick(NOT_THAT_ORDER) + self._order_answer())
                intents.append("wrong_order")
                revokes = True
                covered = covered | {"order_id", "order_choice"}

        # a yes/no question about one payment method
        if sem.payment_yesno_pm and not intents and "payment_method" not in sem.info_requests and f.template in (
                "exchange", "return_fallback", "modify_payment"):
            want = f.wanted_pm(self._state())
            said = sem.payment_yesno_pm
            if want and said == want:
                parts.append(f"That works -- please use {self._wanted_pm_text()}.")
                intents.append("payment")
            else:
                key = "post_fallback_return" if (f.template == "return_fallback" and self.fallback_used) else f.template
                parts.append(self._pick(OBJECT_PM[key]).format(pm=self._wanted_pm_text()))
                intents.append("objection")

        # refusal task: ownership questions get an answer
        if f.expect_no_write and sem.ownership_question and sem.refusal is None and not intents \
                and "identity" not in sem.info_requests:
            p1, p2 = self._pronoun()
            parts.append(self._pick(RELATION).format(r=f.relation or "family member", p=p1, o=p2))
            intents.append("relation")

        # 5. information requests
        for kind in self._info_requests(sem, covered):
            ans = self._answer(kind, txt, sem)
            if ans is None:
                continue
            text, intent = ans
            if text not in parts:
                parts.append(text)
                intents.append(intent)

        if parts:
            key = None if progress else tuple(intents)
            return self._reply(" ".join(parts), intents, confirms=confirms, revokes=revokes,
                               answered_recap=answered_recap, key=key, progress=progress)

        # 6. everything else
        if sem.confirmation_request:
            return self._unrecognised(self._pick(BE_SPECIFIC))
        if sem.narration_only and re.search(r"\b(?:let me|one moment|just a moment|i(?:'ll| will| am going to|'m going to) "
                                            r"(?:check|look|pull|verify|review|see))\b", txt, re.I):
            return self._unrecognised(self._pick(ACK))
        return self._restate()

    # ------------------------------------------------------------ pieces

    def _grounded(self, ctx: Optional[TurnContext]) -> bool:
        """The agent has read the order the user means now (when the runner tells)."""
        if ctx is None:
            return True
        return self.facts.current_order(self._state()) in (ctx.read_orders or [])

    def _learn_ids(self, txt: str) -> None:
        """A product-hint user learns an order id once the agent ties it to a
        product of that order in the same sentence ("the order with the Pet Bed
        is #W4642822")."""
        f = self.facts
        if f.give_order_id:
            return
        units = [txt] if len(set(ORDER_ID_RE.findall(txt))) == 1 else sentences(txt)
        for s in units:
            ids = set(ORDER_ID_RE.findall(s))
            if len(ids) != 1:
                continue
            oid = next(iter(ids))
            names = f.order_items.get(oid, [])
            if any(n.lower() in s.lower() for n in names):
                self._known_ids.add(oid)

    def _offers_refused_action(self, sem: Semantics, txt: str = "") -> bool:
        pa = sem.proposed_action
        if sem.confirmation_request and pa is not None and pa.action in ("cancel", "return") \
                and (pa.names_target or not pa.named_orders):
            return True
        # "...is not in your account. I'll go ahead and cancel it anyway" / "I can cancel it if you provide her
        # email": an ownership statement that still plans or offers the refused action is not a denial
        sents = sentences(txt)
        refusing = [i for i, x in enumerate(sents) if REFUSING_RE.search(x)]
        last_refusal = refusing[-1] if refusing else -1
        for i, sent in enumerate(sents):
            # an offer (a question), or a plan stated after the last refusing sentence ("...not in your account.
            # I'll go ahead and cancel it anyway"); "it's pending, so we can proceed. However, I cannot..." is not
            if COMPLY_RE.search(sent) and not OWN_ORDERS_RE.search(sent) and ("?" in sent or i > last_refusal):
                return True
        return False

    def _fallback(self, txt: str, sem: Semantics, ctx: Optional[TurnContext]) -> UserReply:
        self.fallback_used = True
        line = self._say("fallback")
        # The same turn may already recap the post-fallback plan.
        sem2 = self.analyzer.analyze(txt, self.task, self._state())
        if sem2.confirmation_request and sem2.proposed_action is not None:
            d = self._consent(sem2, ctx, txt)
            if d is not None and d[0] == "yes":
                return self._reply(f"{line} {d[1]}", ["fallback", "yes"], confirms=self._yes_confirms(), revokes=True,
                                   progress=True)
        return self._reply(line, ["fallback"], revokes=True, progress=True)

    def _consent(self, sem: Semantics, ctx: Optional[TurnContext], txt: str = ""):
        """(kind, text, intent, covered info kinds) for a consent request, or
        None when the request is not about the task's action at all."""
        f = self.facts
        pa = sem.proposed_action
        st = self._state()
        cur = f.current_order(st)
        want_action = TEMPLATE_ACTION.get(f.template) or (f.foreign_request if f.expect_no_write else None)
        others = [o for o in pa.named_orders if o != cur]
        low = txt.lower()
        hint_named = any(h and h.lower() in low for h in f.current_hints(st))
        # a product-hint user knows its orders by their products, not by ids: a recap naming only ids it was
        # never told gets the same question whichever order it is (no guess oracle, R1-5)
        if not f.give_order_id and pa.named_orders and not f.expect_no_write and not hint_named \
                and not any(o in self._known_ids for o in pa.named_orders) and not self._products_named(low):
            hints = f.current_hints(st)
            return ("info", self._pick(DONT_KNOW_ID).format(h=hints[0] if hints else "items I mentioned"), "order",
                    {"order_id", "order_choice"})
        # several orders offered as a choice -> which one
        if sem.choice_orders or (others and cur in pa.named_orders):
            return ("info", self._order_answer(), "order", {"order_id", "order_choice"})
        # the order a late correction retracted
        self._correction_repeated = False
        if pa.names_stale_target:
            self._correction_repeated = True
            return ("revoke", self._say("correction"), "correction", set())
        # the order the user is about to switch to, before the switch: the late correction now
        if self.p.get("correction") and not self.correction_used and f.first_order and f.target_order in pa.named_orders:
            return ("revoke", self._pick(NOT_THAT_ORDER) + self._order_answer(), "wrong_order",
                    {"order_id", "order_choice"})
        # another order of the user's
        if others and cur not in pa.named_orders:
            if any(o in f.order_items for o in others) or f.give_order_id:
                return ("revoke", self._pick(NOT_THAT_ORDER) + self._order_answer(), "wrong_order",
                        {"order_id", "order_choice"})
            return ("unrecognised", self._pick(BE_SPECIFIC), "unrecognised", set())
        if not pa.names_current_target and cur not in pa.named_orders:
            return ("unrecognised", self._pick(BE_SPECIFIC), "unrecognised", set())
        if want_action and pa.action != want_action:
            return ("unrecognised", self._pick(WRONG_ACTION) + self._say("restate"), "unrecognised", set())
        if not self._grounded(ctx):
            return ("unrecognised", self._pick(GROUND_FIRST), "unrecognised", set())
        if not f.give_order_id and cur in pa.named_orders and cur not in self._known_ids and not hint_named:
            hints = f.current_hints(st)
            return ("info", self._pick(DONT_KNOW_ID).format(h=hints[0] if hints else "items I mentioned"), "order",
                    {"order_id", "order_choice"})
        if f.expect_no_write:
            # complying with a foreign order: the caller happily says yes (the task has no slot to confirm)
            return ("yes", self._pick(YES), "yes", set())
        if self.p.get("correction") and not self.correction_used:
            self.correction_used = True
            return ("revoke", self._say("correction"), "correction", set())
        # a fallback the agent has not earned yet: the user still wants what it asked for
        if self.p.get("fallback") and not self.fallback_used:
            asked = f.asked_pm
            if pa.payment_pm and pa.payment_pm == asked:
                return ("info", self._pick(STILL_WANT[f.template]).format(pm=self._wanted_pm_text()), "payment",
                        {"payment_method"})
            return ("revoke", self._pick(OBJECT_PM[f.template]).format(pm=self._wanted_pm_text()), "objection",
                    {"payment_method"})
        # an exchange recap naming no item and no new option ("as you requested") is too vague to agree to
        if f.template == "exchange" and "options" in pa.missing and not pa.mismatches and not (pa.details or {}).get("items") \
                and not self._names_new_item_ids(txt):
            return ("unrecognised", self._pick(BE_SPECIFIC), "unrecognised", set())
        if f.template == "modify_payment" and "payment" in pa.missing and not pa.mismatches:
            return ("unrecognised", self._pick(BE_SPECIFIC), "unrecognised", set())
        mm = [m for m in pa.mismatches]
        if pa.stale_options:
            self._correction_repeated = True
            return ("revoke", self._say("correction") if self.p.get("correction") else
                    self._pick(WRONG_OPTIONS).format(changes=self._changes_text()), "correction", set())
        if "action" in mm:
            return ("unrecognised", self._pick(WRONG_ACTION) + self._say("restate"), "unrecognised", set())
        if "reason" in mm and self.p.get("reason"):
            self._told.add("reason")
            return ("revoke", self._pick(WRONG_REASON).format(r=self._say("reason")), "objection", {"reason"})
        if "options" in mm:
            return ("revoke", self._pick(WRONG_OPTIONS).format(changes=self._changes_text()), "objection",
                    {"option_choice"})
        if "payment" in mm:
            key = "post_fallback_return" if f.template == "return_fallback" else (
                "exchange" if f.template == "exchange" else "modify_payment")
            return ("revoke", self._pick(OBJECT_PM[key]).format(pm=self._wanted_pm_text()), "objection",
                    {"payment_method"})
        if "items" in mm:
            names = f.request_items if f.template == "return_fallback" else [n for n, _ in f.wanted_targets(st)]
            return ("revoke", self._pick(WRONG_ITEMS).format(verb=_VERB.get(f.template, "change"), items=_join(names)),
                    "objection", {"item_choice"})
        # consent; a missing detail the user has to provide comes with it
        detail = None
        covered: set[str] = set()
        if f.template == "cancel" and "reason" in pa.missing and ("reason" not in self._told or "reason" in sem.info_requests):
            detail = self._say("reason")
            covered.add("reason")
            self._told.add("reason")
        elif f.template in ("exchange", "modify_payment") and "payment" in pa.missing and (
                "payment" not in self._told or "payment_method" in sem.info_requests):
            pm = self._wanted_pm_text()
            detail = f"please use {pm} for any difference." if f.template == "exchange" else f"please put it on {pm}."
            covered.add("payment_method")
            self._told.add("payment")
        elif f.template == "return_fallback" and "payment" in pa.missing and "payment_method" in sem.info_requests:
            detail = "the refund should go to the original payment method."
            covered.add("payment_method")
        if "all_items_complete" in sem.info_requests:
            covered.add("all_items_complete")
            line = "Yes, that's everything -- please go ahead." if detail is None else \
                self._pick(YES_AND).format(d=_lower_first(detail)) + " That's everything."
            return ("yes", line, "yes", covered)
        if detail is not None:
            return ("yes", self._pick(YES_AND).format(d=_lower_first(detail)), "yes", covered)
        return ("yes", self._pick(YES), "yes", covered)

    def _names_new_item_ids(self, txt: str) -> bool:
        """The recap states the new variants by item id (as precise as option values)."""
        ids = [i for a in self.task.gold_actions for i in (a.get("arguments") or {}).get("new_item_ids") or []]
        return bool(ids) and all(i in txt for i in ids)

    def _products_named(self, low: str) -> bool:
        """Some product of the user's orders is named (so a hint user can tell which order it is)."""
        return any(len(n) > 3 and n.lower() in low for names in self.facts.order_items.values() for n in names)

    def _info_requests(self, sem: Semantics, covered: set[str]) -> list[str]:
        return [k for k in _INFO_ORDER if k in sem.info_requests and k not in covered]

    def _order_answer(self) -> str:
        return self._say("order_answer")

    def _answer(self, kind: str, txt: str, sem: Semantics) -> Optional[tuple[str, str]]:
        f = self.facts
        if kind == "identity":
            if sem.identity_target == "third_party" and not self._third_party_said:
                self._third_party_said = True
                p1, _ = self._pronoun()
                return self._pick(THIRD_PARTY).format(p=p1, i=self._caller_identity()), "identity"
            low = txt.lower()
            if (self.id_mode == "name_zip" and not self._no_email_said and re.search(r"e-?mail", low)
                    and not re.search(r"\bname\b|\bzip\b|postal", low)):
                self._no_email_said = True
                return self._pick(NO_EMAIL).format(i=self._name_zip()), "identity"
            return self._say("identity"), "identity"
        if kind in ("order_id", "order_choice"):
            return self._order_answer(), "order"
        if kind == "reason":
            if self.p.get("reason"):
                self._told.add("reason")
                return self._say("reason"), "reason"
            return None
        if kind == "payment_method":
            if f.template == "exchange" or (self.p.get("payment") and not self.fallback_used):
                self._told.add("payment")
                return self._say("payment"), "payment"
            if self.fallback_used and self.p.get("fallback"):
                return self._say("fallback"), "payment"
            return None
        if kind == "all_items_complete":
            return self._pick(ALL_ITEMS), "all_items"
        if kind == "option_choice":
            if f.template == "exchange":
                return (f"I'd like {self._changes_text()} -- everything else the same as what I have now.", "option")
            if ORDER_ID_RE.search(txt) or not f.give_order_id:
                return self._order_answer(), "order"
            return None
        if kind == "item_choice":
            if f.template == "exchange":
                return f"It's the {_join([n for n, _ in f.wanted_targets(self._state())])}.", "item"
            if f.template == "return_fallback" and f.request_items:
                return f"The {_join(f.request_items)}.", "item"
            return None
        return None


def question_sentences(text: str) -> str:
    """Back-compat helper: the asking units of `text`, joined."""
    from tau_forge.episodes.nlu import request_units

    return " ".join(u.text for u in request_units(text)[1])


def asks_confirmation(text: str, task: Optional[EpisodeTask] = None) -> bool:
    """Back-compat helper: does the turn ask consent to act now."""
    from tau_forge.episodes.nlu import _consent_clause, _clauses, request_units

    return any(u.confirm_details or any(_consent_clause(c) for c in _clauses(u.text)) for u in request_units(text)[1])
