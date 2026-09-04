"""Phase 6 harness: turn-level scoring of a multi-step rollout against a
`TrustedTask`'s gold action sequence, built entirely on the Phase 1 env
wrapper and the Phase 4 `reward()` function -- no new grading logic.

Grading is **teacher-forced**: turn i's reward compares the rollout's turn-i
action against gold's turn-i action, both executed from the DB state
produced by replaying gold actions `0..i-1` (not the rollout's own prior
actions). This is a deliberate choice, distinct from tau2's own official
retail-task grading (`RewardType.DB`: only the *final* DB state after a full
conversation has to match, any path there is fine) -- that coarse,
end-of-episode signal is what Phase 8's live-benchmark eval uses. Training
(Phase 7) needs denser, per-step credit assignment, and teacher-forcing the
context is what makes a step's reward well-defined and comparable across
rollouts regardless of where an earlier step diverged.

A rollout shorter than gold has its missing turns graded as "no call made"
(via `reward()`'s existing `missing_call` case). A rollout longer than gold
has its extra turns scored 0 (`extra_unrequested_turn`) -- there is no gold
counterpart to grade them against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.domains.retail.data_model import RetailDB

from tau_forge.data_prep.trusted_tasks import GoldAction, TrustedTask
from tau_forge.envs.retail import execute_against
from tau_forge.reward.reward import Action, RewardBreakdown, reward


def _load_initial_db() -> RetailDB:
    return RetailDB.load(RETAIL_DB_PATH)


def replay_prefix_state(gold_actions: list[GoldAction], upto: int, initial_db: Optional[RetailDB] = None) -> RetailDB:
    """The DB state after replaying gold actions `[0, upto)` starting from
    the shared initial `db.json` (or `initial_db` if given, e.g. to avoid
    reloading it once per task/turn in a hot loop).

    A gold action can legitimately fail to execute (e.g. a lookup for an id
    that doesn't exist -- the error is itself the correct, graded outcome;
    see `reward.reward`'s `expected_failure_but_succeeded`/`error_match`
    handling). A failed call never mutates state, so replay simply carries
    the prior DB state forward through it rather than treating it as a
    broken label.

    O(upto) deep copies; prefer `replay_all_prefixes` when every prefix of a
    trajectory is needed (`score_rollout`'s case) -- that computes all of
    them in one forward pass instead of redoing the prefix from scratch per
    turn."""
    db = (initial_db if initial_db is not None else _load_initial_db()).model_copy(deep=True)
    for a in gold_actions[:upto]:
        _result, db = execute_against(db, a.tool_name, a.arguments)
    return db


def replay_all_prefixes(gold_actions: list[GoldAction], initial_db: RetailDB) -> list[RetailDB]:
    """`states[i]` is the DB state before gold turn `i`, for every `i` in
    `range(len(gold_actions))`, computed in one forward pass (O(n) deep
    copies total, vs. O(n^2) from calling `replay_prefix_state` per turn)."""
    states = []
    db = initial_db.model_copy(deep=True)
    for a in gold_actions:
        states.append(db)
        _result, db = execute_against(db, a.tool_name, a.arguments)
    return states


@dataclass
class TrajectoryScore:
    task_id: str
    turn_scores: list[RewardBreakdown] = field(default_factory=list)

    @property
    def mean(self) -> float:
        if not self.turn_scores:
            return 0.0
        return sum(t.score for t in self.turn_scores) / len(self.turn_scores)

    @property
    def min(self) -> float:
        return min((t.score for t in self.turn_scores), default=0.0)


def score_rollout(
    task: TrustedTask, rollout_actions: list[Action], initial_db: Optional[RetailDB] = None
) -> TrajectoryScore:
    """Grade `rollout_actions` turn-by-turn against `task.gold_actions`,
    teacher-forcing context from the gold trajectory (see module docstring).

    Pass `initial_db` (e.g. loaded once by a caller scoring many tasks) to
    avoid re-reading and re-parsing the shipped `db.json` on every call --
    `RetailDB.load` dominates runtime otherwise.

    Some real train tasks (Phase 6 finding, e.g. tasks "24"/"57") have zero
    gold actions -- the correct behavior is purely conversational, no tool
    call at all. A rollout that also makes no calls is a perfect match on
    such a task (`correct_no_call`, mirroring `reward.reward`'s own
    single-turn case for this), not the vacuous 0.0 an empty turn list would
    otherwise average to."""
    gold = task.gold_actions
    if not gold and not rollout_actions:
        return TrajectoryScore(task_id=task.task_id, turn_scores=[RewardBreakdown(1.0, "correct_no_call")])
    n_turns = max(len(gold), len(rollout_actions))
    db = initial_db if initial_db is not None else _load_initial_db()
    prefix_states = replay_all_prefixes(gold, db)

    scores: list[RewardBreakdown] = []
    for i in range(n_turns):
        if i >= len(gold):
            scores.append(RewardBreakdown(0.0, "extra_unrequested_turn"))
            continue
        gold_action = Action(gold[i].tool_name, gold[i].arguments)
        rollout_action = (
            rollout_actions[i] if i < len(rollout_actions) else Action(tool_name=None)
        )
        scores.append(reward(rollout_action, gold_action, prefix_states[i]))

    return TrajectoryScore(task_id=task.task_id, turn_scores=scores)


def gold_rollout(task: TrustedTask) -> list[Action]:
    """The trivial "perfect" rollout: replay gold verbatim. Scoring this
    against itself should always come out to 1.0 per turn -- that's the
    Phase 6 smoke test's core assertion (env + reward + data prep agree with
    each other on real, trusted data)."""
    return [Action(a.tool_name, a.arguments) for a in task.gold_actions]
