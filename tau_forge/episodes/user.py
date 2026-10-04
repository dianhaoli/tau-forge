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
  2. refusal task, agent denies         -> accept the denial + STOP
  3. agent denies, fallback pending     -> give the fallback (once)
  4. confirmation request naming the target order/product
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

Anti-exploit rules:
  * "yes" only answers a confirmation request that names the current target
    (an order id or product name from `profile["recap_keys"]`), so a bare
    "shall I proceed?" cannot harvest a yes, and after a late correction the
    recap has to name the corrected target.
  * every line is drawn from a paraphrase pool with an rng seeded by
    (task id, seed): deterministic per task, but no single surface form for
    the policy to key on across tasks.
  * the user never says an id other than an order id or an email
    (`generate.verify_task` enforces it on every line of the profile).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from tau_forge.episodes.task import STOP, EpisodeTask

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
    r"\b(can(no|'|’)t|unable|not able|only (be )?(help|assist|refund|process)\w*|"
    r"not (allowed|possible|permitted|eligible)|must (go|be refunded)|original payment method|"
    r"insufficient|not enough|(doesn't|does not) (have|cover) enough|against (our|the) policy)\b",
    re.I,
)


@dataclass
class UserReply:
    text: str
    stop: bool
    intent: str
    # True only for a "yes" to a recap that named the target -- the runner's
    # confirmation gate keys off this, not off the text.
    is_yes: bool = False
    # True when this reply answered a target-naming confirmation request
    # without a yes (the late correction): it revokes any earlier yes.
    answered_recap: bool = False


class ScriptedUser:
    def __init__(self, task: EpisodeTask, seed: int = 0):
        self.task = task
        self.p = task.profile
        self.rng = random.Random(f"{task.id}/{seed}")
        self.correction_used = False
        self.fallback_used = False
        self.n_unrecognised = 0

    def _say(self, key: str) -> str:
        """A line from the task's pool for `key`; after the late correction
        has been given, its `<key>_after_correction` variant when one exists
        (e.g. the order the user now means)."""
        if self.correction_used and f"{key}_after_correction" in self.p:
            key = f"{key}_after_correction"
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

    def _unrecognised(self, line: str) -> UserReply:
        self.n_unrecognised += 1
        if self.n_unrecognised >= 3:
            return UserReply(f"{self._pick(GIVE_UP)} {STOP}", True, "give_up")
        return UserReply(line, False, "unrecognised")

    def reply(self, agent_text: str, write_succeeded: bool = False) -> UserReply:
        p, txt = self.p, agent_text or ""
        if write_succeeded:
            return UserReply(f"{self._pick(THANKS)} {STOP}", True, "thanks")
        deny = bool(DENY_RE.search(txt))
        is_confirm = "?" in txt and bool(CONFIRM_RE.search(txt))
        names_target = self._names_target(txt)
        if self.task.expect_no_write and deny:
            return UserReply(f"{self._say('accept_denial')} {STOP}", True, "accept_denial")
        if p.get("fallback") and not self.fallback_used and deny and not (is_confirm and names_target):
            self.fallback_used = True
            return UserReply(self._say("fallback"), False, "fallback")
        if is_confirm and names_target:
            if p.get("correction") and not self.correction_used:
                self.correction_used = True
                return UserReply(self._say("correction"), False, "correction", answered_recap=True)
            return UserReply(self._pick(YES), False, "yes", is_yes=True)
        if IDENT_RE.search(txt):
            return UserReply(self._say("identity"), False, "identity")
        if ALL_ITEMS_RE.search(txt):
            return UserReply(self._pick(ALL_ITEMS), False, "all_items")
        if ORDER_ASK_RE.search(txt):
            return UserReply(self._say("order_answer"), False, "order")
        if REASON_RE.search(txt) and p.get("reason"):
            return UserReply(self._say("reason"), False, "reason")
        if PAY_RE.search(txt) and p.get("payment"):
            key = "fallback" if self.fallback_used and p.get("fallback") else "payment"
            return UserReply(self._say(key), False, "payment")
        if is_confirm:
            return self._unrecognised(self._pick(BE_SPECIFIC))
        return self._unrecognised(self._pick(RESTATE) + self._say("restate"))
