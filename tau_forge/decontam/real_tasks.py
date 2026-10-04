"""Exclusion fingerprints of the 114 real retail tasks, for episode generation.

`tau_forge.episodes.generate` builds tasks from the same shared db.json the
real tasks use, so a generated (user, order, action) can coincide with a real
task's gold outcome even though nothing was copied from it. This module is
the sanctioned place (with `check.py`) to read tasks.json; it reduces it to
two sets of opaque identifiers and nothing else:

  * `user_ids` -- every user a real task's gold actions touch: `user_id`
    arguments, owners of `order_id` arguments, and the users the gold
    `find_user_id_*` calls resolve to;
  * `gold_db_hashes` -- tau2's end-state hash after replaying each task's gold
    actions from its initial state, computed exactly as tau2's
    `EnvironmentEvaluator` computes the gold side.

No text, argument value other than a user id, or task id ever leaves this
module, and nothing here prints: the repr is counts only, and any load or
replay failure raises with the exception type, never its message (pydantic
validation messages quote the input). The generator logs only how many
candidates each set rejected.

Loading fails loudly. A generator that silently skipped decontamination when
tasks.json is missing would produce a corpus that looks clean and is not.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Iterable

_USER_TOOLS = ("find_user_id_by_email", "find_user_id_by_name_zip")


@dataclass(frozen=True)
class RealTaskExclusions:
    user_ids: frozenset[str]
    gold_db_hashes: frozenset[str]

    def __repr__(self) -> str:  # counts only -- never the ids themselves
        return f"RealTaskExclusions(<{len(self.user_ids)} user ids>, <{len(self.gold_db_hashes)} gold hashes>)"

    __str__ = __repr__


def _gold_actions(task: Any) -> list[Any]:
    criteria = getattr(task, "evaluation_criteria", None)
    return list(getattr(criteria, "actions", None) or [])


def fingerprint_tasks(tasks: Iterable[Any], db: Any) -> RealTaskExclusions:
    """Pure core of `load_real_task_exclusions`: replays each task's gold
    actions on a private copy of `db` (never mutating it). `tasks` only needs
    tau2's `Task` attribute shape, so tests can pass stubs."""
    from loguru import logger

    # tau2 logs replayed tool responses at DEBUG (enabled by default) --
    # silence it so no task content reaches stderr.
    logger.disable("tau2")
    try:
        return _fingerprint(tasks, db)
    finally:
        logger.enable("tau2")


def _fingerprint(tasks: Iterable[Any], db: Any) -> RealTaskExclusions:
    from tau2.domains.retail.environment import get_environment

    user_ids: set[str] = set()
    hashes: set[str] = set()
    for k, task in enumerate(tasks):
        env = get_environment(db=db.model_copy(deep=True))
        initial = getattr(task, "initial_state", None)
        try:
            if initial is not None:
                env.set_state(
                    initialization_data=initial.initialization_data,
                    initialization_actions=initial.initialization_actions,
                    message_history=initial.message_history or [],
                )
        except Exception as e:  # noqa: BLE001 -- re-raised without content
            raise RuntimeError(f"real task #{k}: could not restore its initial state ({type(e).__name__})") from None
        for action in _gold_actions(task):
            args = dict(action.arguments or {})
            uid = args.get("user_id")
            if isinstance(uid, str):
                user_ids.add(uid)
            oid = args.get("order_id")
            if isinstance(oid, str) and oid in env.tools.db.orders:
                user_ids.add(env.tools.db.orders[oid].user_id)
            try:
                out = env.make_tool_call(
                    tool_name=action.name, requestor=getattr(action, "requestor", "assistant"), **args
                )
            except Exception:  # noqa: BLE001 -- tau2's evaluator also tolerates failing gold calls
                continue
            if action.name in _USER_TOOLS and isinstance(out, str):
                user_ids.add(out)
        hashes.add(env.get_db_hash())
    return RealTaskExclusions(frozenset(user_ids), frozenset(hashes))


@functools.lru_cache(maxsize=1)
def load_real_task_exclusions() -> RealTaskExclusions:
    """All 114 real retail tasks (tau2's `base` split: train + test), reduced
    to exclusion fingerprints. Raises RuntimeError if they cannot be loaded.

    Measured: 52 distinct user ids and 99 distinct gold hashes (one of them
    the untouched base db -- some real tasks' gold actions are all reads).
    The replay takes ~28 s, so the result is cached per process."""
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.environment import get_tasks
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    try:
        tasks = get_tasks("base")
    except Exception as e:  # noqa: BLE001 -- re-raised without content
        raise RuntimeError(
            "episode decontamination needs tau2's retail tasks.json and could not load it "
            f"({type(e).__name__}). Initialise third_party/tau2-bench; do not generate without it."
        ) from None
    if not tasks:
        raise RuntimeError("episode decontamination loaded zero real retail tasks; refusing to generate.")
    return fingerprint_tasks(tasks, RetailDB.load(RETAIL_DB_PATH))
