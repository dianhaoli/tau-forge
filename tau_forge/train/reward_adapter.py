"""Adapts `tau_forge.reward.reward()` into the `reward_funcs` contract TRL's
`GRPOTrainer` expects: `def fn(prompts, completions, **kwargs) -> list[float]`,
where `**kwargs` receives every non-`prompt` dataset column, each a list
aligned with `completions` (TRL's own documented reward-function contract).

Deliberately torch/trl-free -- only `grpo_train.py` needs the GPU training
stack. `score_completion` is the actual scoring logic and is unit-testable on
its own against the real synthetic data without a GPU (see
`tests/test_train_pipeline.py`); `grpo_reward_func` is the thin adapter TRL
calls directly.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Optional

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH

from tau_forge.envs.retail import ToolResult, execute_against
from tau_forge.reward.reward import Action, reward
from tau_forge.train.completion_parsing import parse_completion

_shared_db: Optional[RetailDB] = None


def _get_shared_db() -> RetailDB:
    """The plain shipped `db.json`, loaded once and reused for every reward
    call in a run -- matches `tau_forge.validate.rule_checker`'s own
    re-execution methodology (see `dataset.py`'s module docstring). `reward()`
    and `execute_against()` never mutate the `db_state` they're given, so one
    shared load is safe to reuse across the whole run instead of re-parsing
    the 2.8MB `db.json` per rollout."""
    global _shared_db
    if _shared_db is None:
        _shared_db = RetailDB.load(RETAIL_DB_PATH)
    return _shared_db


# Gold's end state is a property of the scenario, not of the completion being
# graded, so every sample in a GRPO group re-derives the same one: 16
# completions of one prompt, 16 identical `execute_against` calls, each
# deep-copying the 2.8MB retail db. Skipping the repeats takes a right-tool
# scoring call from ~256ms to ~141ms.
#
# Kept small on purpose. Each entry pins a whole db copy, so an unbounded cache
# over the 541-scenario corpus would hold hundreds of them. Both callers feed
# completions grouped by scenario -- TRL emits a group's `num_generations`
# samples contiguously, and `zero_shot_baseline` repeats each row
# `samples_per_scenario` times in place -- so a single live entry is enough and
# four is slack for interleaving.
_GOLD_CACHE_SIZE = 4
_gold_cache: "OrderedDict[tuple[str, str], tuple[ToolResult, RetailDB]]" = OrderedDict()


def _gold_outcome(
    expected_tool_name: str, expected_tool_arguments: dict[str, Any]
) -> tuple[ToolResult, RetailDB]:
    """Cached `execute_against` for a gold action against the shared db. The
    `RetailDB` handed back is shared between callers; `reward()` only reads and
    compares it, which is what makes that safe."""
    key = (expected_tool_name, json.dumps(expected_tool_arguments, sort_keys=True, default=str))
    cached = _gold_cache.get(key)
    if cached is not None:
        _gold_cache.move_to_end(key)
        return cached
    outcome = execute_against(_get_shared_db(), expected_tool_name, expected_tool_arguments)
    _gold_cache[key] = outcome
    while len(_gold_cache) > _GOLD_CACHE_SIZE:
        _gold_cache.popitem(last=False)
    return outcome


def score_completion(
    completion_text: str,
    expected_tool_name: Optional[str],
    expected_tool_arguments: dict[str, Any],
) -> float:
    predicted_name, predicted_args = parse_completion(completion_text)
    rollout_action = Action(tool_name=predicted_name, tool_input=predicted_args)
    gold_action = Action(tool_name=expected_tool_name, tool_input=expected_tool_arguments)
    # Only pay for gold when `reward()` will get far enough to use it. A
    # mismatched or missing tool name is graded without ever executing
    # anything, and that is the common case on a cold-start policy -- eagerly
    # populating the cache there would make the usual completion slower, not
    # faster.
    gold_outcome = (
        _gold_outcome(expected_tool_name, expected_tool_arguments)
        if expected_tool_name is not None and predicted_name == expected_tool_name
        else None
    )
    return reward(rollout_action, gold_action, _get_shared_db(), gold_outcome=gold_outcome).score


def grpo_reward_func(
    prompts: list[str],
    completions: list[str],
    expected_tool_name: list[Optional[str]],
    expected_tool_arguments_json: list[str],
    **kwargs: Any,
) -> list[float]:
    return [
        score_completion(completion, name, json.loads(args_json))
        for completion, name, args_json in zip(completions, expected_tool_name, expected_tool_arguments_json)
    ]
