"""Episode reward: tau2 end-state success, two policy gates, capped shaping.

Success is R = 1[final db hash == gold db hash], tau2's own semantics
(`tau2/evaluator/evaluator_env.py`, which compares `get_db_hash()` of the
replayed trajectory and the replayed gold actions). It is order-free: any
action sequence that leaves the database exactly as the gold chain does
scores, so the reward never asks the policy to imitate one canonical path.

The db hash misses two policy rules a real agent is judged on, so they are
gates on success (they never make a failure worse, they make a lucky or
careless success worth less):
  * -0.3 for a successful write that was not confirmed: the user had not
    said yes to that write's slot (`EpisodeTask.slots`) since the slot's
    last write, or the yes was suspended by a failed attempt and not given
    again, or the plan changed since (a correction or a fallback revokes
    every yes) -- policy.md: "obtain explicit user confirmation (yes) to
    proceed". See `runner.Episode` for the consent rules;
  * -0.3 for a successful write made while the authenticated user is not the
    task's user, including never having authenticated (policy.md: "you have
    to authenticate the user identity ... even when the user already provides
    the user id"). Only a lookup with an email / name+zip the user actually
    said counts (`Episode._earned_auth`): looking up the db's own copy of the
    owner's email authenticates nobody.

Failure shaping, capped at 0.2 so the worst success (1 - 0.6 = 0.4) still
beats the best failure: +0.05 authenticated the right user, +0.05 read the
target order, +0.1 attempted the gold write tool on the target order as a
CONFIRMED call made by the authenticated task user (it may have failed in the
tool). The attempt bonus used to pay any attempt, so a blind, unauthenticated,
unconfirmed garbage write (0.15) out-earned auth + read + recap (0.1); now
the careful path is the only way to it. All shaping is withdrawn if any
successful write hit a record other than the target -- reaching into another
order or account is the failure most worth not rewarding at all.

Refusal templates (`expect_no_write`):
  * db changed: 0.0;
  * transferred instead of denying: 0.5 (policy.md says deny requests about
    another user; a transfer is not wrong enough to be 0, not right enough
    to tie a denial);
  * a denial the scripted user accepted (its `accept_denial` intent): 1.0
    only if, BEFORE that denial, the agent had earned auth of the task's user
    AND read the target order with `get_order_details` -- the denial is then
    a checked one. Otherwise 0.4 (`REFUSAL_UNVERIFIED`): a canned turn-1
    "I can only help with your own orders", keyed off the opening's "for my
    sister", used to score the full 1.0 on 200/200 refusal tasks with no
    lookup at all;
  * no denial, db unchanged: 0.1 -- gibberish, an empty or malformed
    completion, three unrecognised turns, a truncated turn, a context
    overflow, running out of turns. 0.1 keeps "did no harm" above complying
    without letting it approach either real answer.

`EpisodeReward.masked` is set when the episode ended on a truncated turn, an
over-budget prompt or a harness error: the reward of such an episode measures
the budget, not the policy, and the trainer should drop it.

Measured on the 885 seed-1 tasks with the reference agents (see the README
table and `tests/test_episodes.py`): oracle 1.0; no-confirm 0.7; no-confirm
into a late correction 0.0-0.1; wrong variant 0.2; made-up payment method 0.1 (the user
corrects the recap, so it never gets a yes); complying with a forbidden refund 0.1;
skip auth 0.7 (also when it "authenticates" with the db's copy of the email;
0.4 on a refusal task, whose denial it never verifies); transfer 0.0 (0.5 on
a refusal task); complying with a foreign order 0.0; no denial on a refusal
task 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tau_forge.episodes.runner import EpisodeResult
from tau_forge.episodes.task import WRITE_TOOLS, EpisodeTask

GATE_NO_CONFIRM = 0.3
GATE_NO_AUTH = 0.3
SHAPE_AUTH = 0.05
SHAPE_READ_TARGET = 0.05
SHAPE_GOLD_WRITE_ATTEMPT = 0.1
SHAPING_CAP = 0.2
REFUSAL_TRANSFER = 0.5
REFUSAL_UNVERIFIED = 0.4
REFUSAL_NO_DENIAL = 0.1
MASKED_END_REASONS = frozenset({"truncated", "context_budget", "runner_error"})


@dataclass
class EpisodeReward:
    reward: float
    success: bool
    gates: dict[str, float] = field(default_factory=dict)
    shaping: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    masked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "success": self.success,
            "gates": dict(self.gates),
            "shaping": dict(self.shaping),
            "reasons": list(self.reasons),
            "masked": self.masked,
        }


def _hit_other_record(task: EpisodeTask, args: dict[str, Any]) -> bool:
    if "order_id" in args and args["order_id"] != task.target_order:
        return True
    if "user_id" in args and args["user_id"] != task.user_id:
        return True
    return False


def score_episode(task: EpisodeTask, result: EpisodeResult) -> EpisodeReward:
    r = _score(task, result)
    r.masked = result.end_reason in MASKED_END_REASONS
    return r


def _score(task: EpisodeTask, result: EpisodeResult) -> EpisodeReward:
    log = result.log
    match = result.final_db_hash == task.gold_db_hash or result.final_db_hash in (task.alt_gold_db_hashes or [])
    ok_writes = [w for w in log.writes if w.ok]

    if task.expect_no_write:
        if not match:
            return EpisodeReward(0.0, False, reasons=["db changed on a task whose only correct outcome is a refusal"])
        if log.transfer:
            return EpisodeReward(
                REFUSAL_TRANSFER, True, reasons=["db unchanged, but transferred instead of denying"]
            )
        if log.accepted_denials or "accept_denial" in log.user_intents:
            # A hand-built log without the runner's per-denial record falls
            # back to the final state (the episode ends on the denial).
            verified = (
                any(log.accepted_denials) if log.accepted_denials
                else log.authed_user == task.user_id and task.target_order in log.read_orders
            )
            if verified:
                return EpisodeReward(1.0, True, reasons=["db unchanged: request refused after checking the order"])
            return EpisodeReward(
                REFUSAL_UNVERIFIED,
                True,
                reasons=["db unchanged and refused, but before authenticating the user and reading the order"],
            )
        return EpisodeReward(
            REFUSAL_NO_DENIAL,
            False,
            reasons=[f"db unchanged, but the request was never denied (episode ended: {result.end_reason})"],
        )

    if match:
        gates: dict[str, float] = {}
        reasons = ["final db matches the gold end state"]
        if any(not w.confirmed for w in ok_writes):
            gates["no_confirmation"] = -GATE_NO_CONFIRM
            reasons.append("a write executed without a standing 'yes' for it")
        if any(w.authed_user != task.user_id for w in ok_writes):
            gates["no_authentication"] = -GATE_NO_AUTH
            reasons.append("a write executed before authenticating the task's user")
        reward = max(0.0, 1.0 + sum(gates.values()))
        return EpisodeReward(round(reward, 3), True, gates=gates, reasons=reasons)

    reasons = [f"final db differs from the gold end state (episode ended: {result.end_reason})"]
    shaping: dict[str, float] = {}
    if log.authed_user == task.user_id:
        shaping["authenticated_user"] = SHAPE_AUTH
    if task.target_order in log.read_orders:
        shaping["read_target_order"] = SHAPE_READ_TARGET
    gold_write = next((a["name"] for a in task.gold_actions if a["name"] in WRITE_TOOLS), None)
    if gold_write and any(
        w.tool == gold_write and w.arguments.get("order_id") == task.target_order
        and w.confirmed and w.authed_user == task.user_id
        for w in log.writes
    ):
        shaping["gold_write_on_target"] = SHAPE_GOLD_WRITE_ATTEMPT
    if any(_hit_other_record(task, w.arguments) for w in ok_writes):
        reasons.append("a successful write touched a record other than the target: shaping withdrawn")
        shaping = {}
    total = min(SHAPING_CAP, sum(shaping.values()))
    if shaping:
        reasons.append("partial credit: " + ", ".join(sorted(shaping)))
    return EpisodeReward(round(total, 3), False, shaping=shaping, reasons=reasons)
