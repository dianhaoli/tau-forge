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
from typing import Any, Optional

from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.utils import RETAIL_DB_PATH

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


def score_completion(
    completion_text: str,
    expected_tool_name: Optional[str],
    expected_tool_arguments: dict[str, Any],
) -> float:
    predicted_name, predicted_args = parse_completion(completion_text)
    rollout_action = Action(tool_name=predicted_name, tool_input=predicted_args)
    gold_action = Action(tool_name=expected_tool_name, tool_input=expected_tool_arguments)
    return reward(rollout_action, gold_action, _get_shared_db()).score


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
