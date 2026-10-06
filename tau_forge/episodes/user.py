"""The deterministic scripted user an episode's policy talks to.

Why scripted, not tau2's LLM user simulator: a GRPO group must differ only by
the policy's samples. An LLM user adds its own sampling noise to every reward
(and an API call per turn); a regex user replies identically to identical
agent text, so all variance within a group is the policy's. The cost is that
GRPO will probe the script for quirks, hence the anti-exploit rules below --
and the README's note to track drift against tau2's LLM user on a held-out
generated set.

Intent matching on each agent TEXT turn (tool calls never reach the user), in
priority order -- first match wins:
  1. a write has succeeded              -> thanks + STOP
  2. refusal task, agent denies (and is not asking who the user is)
                                        -> accept the denial + STOP
  3. fallback pending, agent denies or states the constraint (the balance,
     the original/purchase method, "another payment method")
                                        -> give the fallback (once)
  4. a recap: a confirmation request naming the target order/product AND
     the action, whose questions ask for no information
                                        -> the late correction (once), else yes
  5. identity asked                     -> identity answer
  6. all/other items asked              -> "that's everything"
  7. order id asked                     -> order id, or a product hint
  8. reason asked                       -> reason
  9. payment asked                      -> payment phrase (post-fallback if used)
 10. a confirmation request that names no target -> ask for specifics
 11. anything else                      -> restate the request
Rules 10 and 11 count as unrecognised; the third unrecognised turn ends the
episode with STOP.

Consent is structured, not textual (the runner's side is `runner.Episode`):
a yes returns `confirms` = the slot ids it consents to (`EpisodeTask.slots`;
for now every slot of the task), and the late correction and the fallback
return `revokes=True` -- the plan changed, so every pending yes is cleared,
including one a failed write had suspended. The runner passes a
`TurnContext` (writes since the last user turn, earned auth, orders read);
this user only reads `any_write_ok` from it.

Anti-exploit rules:
  * "yes" only answers a recap: a confirmation request that names the current
    target (an order id or product name from `profile["recap_keys"]`) and the
    template's action (`ACTION_RE`), and none of whose question sentences
    asks for identity, an order id or a choice. So a bare "shall I proceed?"
    cannot harvest a yes, after a late correction the recap has to name the
    corrected target, and "I can help with order #W... Could you please
    confirm your email address first?" -- the word "confirm" plus the order
    id from the opening -- gets the identity answer, not a yes. That
    opening line used to harvest a yes that stayed valid until the first
    write, which took a never-recapping agent from 0.7 to 1.0 on every
    give_order_id task without a correction (44/44 on 30 per template).
  * every line is drawn from a paraphrase pool with an rng seeded by
    (task id, seed): deterministic per task, but no single surface form for
    the policy to key on across tasks.
  * the user never says an id other than an order id or an email
    (`generate.verify_task` enforces it on every line of the profile).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from tau_forge.episodes.task import STOP, EpisodeTask

if TYPE_CHECKING:
    from tau_forge.episodes.runner import WriteRecord

MAX_SAME_ANSWER = 3  # consecutive identical information answers before giving up
YES = ["Yes, please proceed.", "Yes, go ahead.", "Yes, that's right -- please do it.", "Yes, I confirm."]
THANKS = ["Great, thank you! That's all I needed.", "Perfect, thanks for your help.", "Thanks, that's everything."]
RESTATE = ["Sorry, to be clear: ", "Just to repeat what I need: ", "Let me say it again: "]
GIVE_UP = ["I don't think this is going anywhere. Bye.", "Never mind, I'll try again some other time."]
BE_SPECIFIC = [
    "Before I say yes, can you tell me exactly which order and items this is for?",
    "Sorry, what exactly are you going to do? Please spell out the details first.",
]
ALL_ITEMS = ["Yes, that's everything.", "That's all of it.", "Just those, nothing else."]

CONFIRM_RE = re.compile(
    r"\b(confirm|proceed|shall i|should i|go ahead|would you like me to|do you want me to|"
    r"yes/no|is (that|this) (correct|right)|ok(ay)? to)\b|\(yes",
    re.I,
)
IDENT_RE = re.compile(r"\b(e-?mail|zip|verify|authenticat\w*|identity|identify|full name|first and last name)\b", re.I)
ALL_ITEMS_RE = re.compile(
    r"\b(all (of )?the items|all items|any other items?|other items?|"
    r"anything else (you('d| would) like )?to (exchange|return|modify|cancel)|"
    r"(is that|are those) all (the items)?)\b",
    re.I,
)
ORDER_ASK_RE = re.compile(r"\b(order (id|number)|which order)\b", re.I)
REASON_RE = re.compile(r"\b(reason|why)\b", re.I)
PAY_RE = re.compile(
    r"\b(payment method|which (card|payment|account)|refund (to|go)|pay (for )?the (price )?difference|"
    r"how would you like to (pay|be refunded|receive))\b",
    re.I,
)
DENY_RE = re.compile(
    r"\b(can(no|'|’)t|unable|not able|only (be )?(help|assist|refund|process|go|return)\w*|"
    r"not (allowed|possible|permitted|eligible)|must (go|be refunded)|original payment method|"
    r"insufficient|not enough|(doesn't|does not) (have|cover) enough|against (our|the) policy|"
    r"won(no|'|’)?t (cover|work|be (possible|able))|(will|would) not (cover|work)|"
    r"(does not|doesn't|do not|don't) cover|(not|isn't|aren't) (sufficient|enough)|exceeds?|"
    r"go(es)? back to|(used|use) (for|to (make|pay for)) the (original )?(purchase|order)|"
    r"(not|isn't|aren't) (associated|linked|tied|registered|listed|on|under|in|part of|one of) (with |to )?your|"
    r"(does not|doesn't) belong|belongs to (someone|another|a different))\b",
    re.I,
)
# Ownership denial, for refusal (expect_no_write) tasks only. DENY_RE also
# matches fallback/payment phrasing ("original payment method", "exceeds",
# "can't") that a compliant agent uses while PROCEEDING with a return, so
# using it to accept a refusal paid 1.0 to an agent that offered the return.
OWNERSHIP_DENY_RE = re.compile(
    r"\b((different|another|other|someone else'?s?|a third|separate) (user|account|customer|person|individual|profile)|"
    r"(under|belongs? to|owned by|registered (to|under)|placed by|made by|associated with) (a |an |the )?"
    r"(different|another|other|someone|a third|separate)|"
    r"(not|isn't|aren't|wasn't) (associated|linked|tied|registered|connected|placed|made|part of|one of|under|on|in|listed)"
    r"( with| to| under| on| in| by)? (your|you|this)|"
    r"(does not|doesn't|do not|don't) (belong|appear|match|show up)( to| in| on| under)?( you| your)?|"
    r"(only|just) (help|assist|process|handle|manage|access|modify|cancel|return|discuss)\w*[^.?!]{0,60}"
    r"(your (own )?(account|orders?|user)|(account|orders?) (associated|linked|tied) (with|to) your|"
    r"(the )?account (associated|linked|tied))|"
    r"(can(no|'|’)t|cannot|unable to|not able to|not allowed to|am not permitted to|won(no|'|’)t)[^.?!]{0,60}"
    r"(someone else|another (person|user|customer|account)|a different (person|user|customer|account)|"
    r"(other|another|different) (person|user|customer|account)('s|’s)?|third part\w+)|"
    r"(someone else|another person|another user|another customer)('s|’s)? (order|account)|"
    r"(only|just) (help|assist|process|handle|manage|access|modify|cancel|return|discuss)\w*[^.?!]{0,60}"
    r"(the )?account (holder|owner)|(the )?account (holder|owner) (has|have|must|needs?|will need|should) to|"
    r"(isn't|is not|aren't|are not|wasn't|was not|'s not) (yours|your own)|not your (own )?(order|account))\b",
    re.I,
)
# A request in imperative form: "Please confirm with a "yes" ...", "reply yes",
# "let me know if you'd like me to proceed". Together with a "?" question
# these are the turns that ask the user to confirm.
IMPERATIVE_CONFIRM_RE = re.compile(
    r"(\b(please|kindly)\s+(confirm|reply|respond|answer|say|type|let me know)\b|"
    r"\bconfirm\b[^.?!]{0,40}\b(yes|proceed|go ahead)\b|"
    r"\b(reply|respond|answer|type|say|send)\b[^.?!]{0,30}[\"'“”‘’(]?\byes\b|"
    r"\blet me know (if|whether)\b[^.?!]{0,40}\b(proceed|go ahead|like me to|want me to)\b|"
    r"\bconfirm (if|whether|that) you\b)",
    re.I,
)
# A confirm-style question that only offers OTHER help ("Would you like me to
# transfer you to a human agent?", "... help with one of your own orders?") is
# not an offer to proceed with the refused request.
OTHER_HELP_RE = re.compile(
    r"\b(transfer|human|representative|anything else|something else|other orders?|your own|another order)\b",
    re.I,
)
# Imperative / statement forms that still ASK the user for something (so
# IDENT_RE / ORDER_ASK_RE only fire on requests, not on statements like "I
# have already authenticated your identity using your email").
REQUEST_RE = re.compile(
    r"\b(please|kindly|provide|let me know|tell me|share|send me|give me|enter|"
    r"i('d| would) need|i('ll| will) need|i need|i require|need (to know )?your|require)\b",
    re.I,
)

# Statements of the constraint behind a fallback that need no denial word:
# "your gift card has a balance of $40.00, but the order total is $120.50 ...
# which payment method would you like?" or "refunds go to the method used
# for the purchase or a gift card -- which would you prefer?". A correct agent
# phrasing it so was not recognised and the user kept asking for the
# forbidden / short method.
CONSTRAINT_RE = re.compile(
    r"\b(balance|original (payment|method|card|form)|(another|a different|other|alternative|second) "
    r"(payment|card|method|form of payment))\b",
    re.I,
)
# What a recap must say is about to happen, per template: a target-naming
# "confirm" that names no action is a request, not a recap.
ACTION_RE = {
    "cancel": re.compile(r"\bcancel\w*", re.I),
    "exchange": re.compile(r"\b(exchang\w*|swap\w*|replac\w*|chang\w*|switch\w*)", re.I),
    "return_fallback": re.compile(r"\b(return\w*|refund\w*|send(ing)? back)", re.I),
    "modify_payment": re.compile(r"\b(chang\w*|switch\w*|updat\w*|modif\w*|mov(e|ing)|charg\w*)", re.I),
    "foreign_order_refusal": re.compile(r"\b(cancel\w*|return\w*|refund\w*)", re.I),
}
# A question sentence matching this asks for information, so it is not a
# recap even if the turn also names the target and action.
INFO_ASK_RE = re.compile(
    r"\b(e-?mail|zip|verify|authenticat\w*|identity|identify|full name|first and last name|"
    r"order (id|number)|which (payment|card|method|order|item|one|option)|provide)\b",
    re.I,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def question_sentences(text: str) -> str:
    """The sentences of `text` that ask the user something, joined: those
    containing a question mark, plus imperative requests ("Please provide your
    order id.", "I need your email to verify.")."""
    return " ".join(x for x in _SENTENCE_SPLIT_RE.split(text) if "?" in x or REQUEST_RE.search(x))


def asks_confirmation(text: str) -> bool:
    """True when the turn asks the user to confirm / approve: a confirm word
    in a question sentence, or an imperative confirmation request."""
    return bool(CONFIRM_RE.search(" ".join(x for x in _SENTENCE_SPLIT_RE.split(text) if "?" in x))) or bool(
        IMPERATIVE_CONFIRM_RE.search(text)
    )


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
    # The plan changed (late correction, fallback): clear all pending consent.
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


class ScriptedUser:
    def __init__(self, task: EpisodeTask, seed: int = 0):
        self.task = task
        self.p = task.profile
        self.rng = random.Random(f"{task.id}/{seed}")
        self.correction_used = False
        self.fallback_used = False
        self.n_unrecognised = 0
        self._last_info_intent = ""
        self._info_streak = 0

    def _answer(self, intent: str, text: str) -> UserReply:
        """An information answer (identity / order / reason / payment / all
        items). The third consecutive answer of the same kind means the agent
        is looping on one question: give up instead of repeating until
        max_turns."""
        self._info_streak = self._info_streak + 1 if intent == self._last_info_intent else 1
        self._last_info_intent = intent
        if self._info_streak >= MAX_SAME_ANSWER:
            return UserReply(f"{self._pick(GIVE_UP)} {STOP}", True, "give_up")
        return UserReply(text, False, intent)

    def _say(self, key: str) -> str:
        """A line from the task's pool for `key`; after the late correction
        (or the fallback) has been given, its `<key>_after_correction` (or
        `<key>_after_fallback`) variant when one exists -- e.g. the order the
        user now means, or a restate that no longer names the forbidden
        method. Without them a later unrecognised turn made the user re-ask
        for what it had just retracted."""
        if self.correction_used and f"{key}_after_correction" in self.p:
            key = f"{key}_after_correction"
        elif self.fallback_used and f"{key}_after_fallback" in self.p:
            key = f"{key}_after_fallback"
        pool = self.p[key]
        return pool if isinstance(pool, str) else self.rng.choice(pool)

    def _pick(self, pool: list[str]) -> str:
        return self.rng.choice(pool)

    def _names_target(self, text: str) -> bool:
        keys = self.p["recap_keys"]
        if self.correction_used and self.p.get("recap_keys_after_correction"):
            keys = self.p["recap_keys_after_correction"]
        low = text.lower()
        return any(k.lower() in low for k in keys)

    def _is_recap(self, text: str) -> bool:
        """A confirmation request that names the target and the action and
        asks for nothing else -- the only thing "yes" answers."""
        if not (asks_confirmation(text) and self._names_target(text)):
            return False
        action = ACTION_RE.get(self.task.template)
        if action is not None and not action.search(text):
            return False
        asks = question_sentences(text)
        # "Before I proceed with the exchange for order #W1, is that all the
        # items?" names the target and the action, but asks about the item
        # list: a yes to it is not a yes to a recap.
        return not INFO_ASK_RE.search(asks) and not ALL_ITEMS_RE.search(asks)

    def _unrecognised(self, line: str) -> UserReply:
        self.n_unrecognised += 1
        if self.n_unrecognised >= 3:
            return UserReply(f"{self._pick(GIVE_UP)} {STOP}", True, "give_up")
        return UserReply(line, False, "unrecognised")

    def _yes(self) -> UserReply:
        """A yes to a recap. Interim (stage A): it confirms every slot of the
        task -- "main" on every single-write task, nothing on a refusal task;
        binding a yes to the slots its recap actually describes is the NLU's
        job."""
        return UserReply(self._pick(YES), False, "yes", confirms=[s["id"] for s in self.task.slots])

    def reply(self, agent_text: str, ctx: Optional[TurnContext] = None, write_succeeded: bool = False) -> UserReply:
        p, txt = self.p, agent_text or ""
        if ctx is not None and ctx.any_write_ok:
            write_succeeded = True
        if write_succeeded:
            return UserReply(f"{self._pick(THANKS)} {STOP}", True, "thanks")
        deny = bool(DENY_RE.search(txt))
        is_recap = self._is_recap(txt)
        is_confirm = asks_confirmation(txt)
        asks = question_sentences(txt)
        # "Sorry, I can't find your account. What is your email?" is a request
        # for identity, not a denial of the request.
        asks_identity = bool(IDENT_RE.search(asks))
        # A refusal is only accepted for an ownership denial, and never in a
        # turn that also offers to proceed / asks for confirmation.
        if (
            self.task.expect_no_write and OWNERSHIP_DENY_RE.search(txt)
            and not asks_identity and not is_recap
            and not (is_confirm and not OTHER_HELP_RE.search(txt))
        ):
            return UserReply(f"{self._say('accept_denial')} {STOP}", True, "accept_denial")
        if (
            p.get("fallback") and not self.fallback_used and not is_recap and not asks_identity
            and (deny or CONSTRAINT_RE.search(txt))
        ):
            self.fallback_used = True
            # The plan changed: a yes given to the old plan (e.g. a refund to
            # the forbidden card, whose write then failed) no longer stands.
            return UserReply(self._say("fallback"), False, "fallback", revokes=True)
        if is_recap:
            if p.get("correction") and not self.correction_used:
                self.correction_used = True
                return UserReply(self._say("correction"), False, "correction", revokes=True, answered_recap=True)
            return self._yes()
        if asks_identity:
            return self._answer("identity", self._say("identity"))
        if ALL_ITEMS_RE.search(txt):
            return self._answer("all_items", self._pick(ALL_ITEMS))
        if ORDER_ASK_RE.search(asks):
            return self._answer("order", self._say("order_answer"))
        if REASON_RE.search(txt) and p.get("reason"):
            return self._answer("reason", self._say("reason"))
        if PAY_RE.search(txt) and p.get("payment"):
            key = "fallback" if self.fallback_used and p.get("fallback") else "payment"
            return self._answer("payment", self._say(key))
        if is_confirm:
            return self._unrecognised(self._pick(BE_SPECIFIC))
        return self._unrecognised(self._pick(RESTATE) + self._say("restate"))
