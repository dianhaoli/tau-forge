"""Auxiliary shaping reward -- the direct fix for zero-variance-at-0.0 groups.

Why this exists
---------------
`tau_forge.reward.reward()` collapses *every* wrong-tool outcome to exactly
`0.0`: `wrong_tool`, `unknown_tool`, and a malformed `<tool_call>` block all
score identically. That flat floor is the mechanical reason a cold-start
scenario shows zero within-group variance at n=16 -- the policy is wrong in
many materially different ways, and the reward says all of them are the same.
GRPO's advantage is `(r - group_mean) / group_std`, so a group of sixteen
distinct-but-equally-scored failures produces no gradient at all. Raising the
sampling temperature or the group size cannot fix that: the degeneracy is in
the reward surface, not in the sampling.

This module adds a small, strictly-bounded amount of partial credit *inside*
that flat region, so "called `get_order_details` on the right order when gold
was `cancel_pending_order`" scores above "emitted a malformed blob". That
turns a flat group into a varying one and gives GRPO a gradient pointing at
the right record and the right tool class.

Deliberately a **separate** TRL `reward_funcs` entry rather than an edit to
`reward.py`: Phase 4's adversarial test table (README) is a validated
artifact, and this keeps it true by construction. TRL sums its reward
functions, so the effective score is `reward() + shaping()`.

Ordering guarantee (asserted in tests, not just claimed): the shaping term is
capped at `WRONG_TOOL_CEILING = 0.15`, strictly below `reward()`'s 0.2
schema-invalid tier and far below its 0.3 right-tool floor. No amount of
shaping can make a wrong tool outscore a right one, so this cannot invert any
ranking Phase 4 established.

Where it deliberately stays silent (returns 0.0):
  * gold is "no call" (`ambiguous` / most `policy_violation`) -- crediting a
    tool call there would push the policy toward acting when silence is
    correct, the exact failure this corpus is meant to train against.
  * the policy called the right tool -- `reward()` already grades it in the
    0.2-1.0 band with real resolution; adding to that just rescales.
  * the policy said nothing when a call was required (`missing_call`) --
    there is no partial signal in an empty turn to grade, and paying anything
    for silence competes with the no-call scenarios above.

Torch/trl-free, like the rest of `tau_forge.train` except `grpo_train`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from tau_forge.reward.reward import ID_LIKE_KEYS
from tau_forge.train.completion_parsing import MALFORMED_TOOL_CALL, parse_completion

# Every component below sums to exactly this. Kept strictly under reward.py's
# 0.2 `schema_invalid_or_hallucinated_args` tier -- see module docstring.
WRONG_TOOL_CEILING = 0.15

PARSEABLE_CALL = 0.02  # emitted a well-formed <tool_call> with a string name
REAL_TOOL = 0.03  # ...and that name is an actual retail tool
SCHEMA_VALID = 0.03  # ...and the arguments validate against that tool's schema
SAME_TOOL_CLASS = 0.02  # ...and it mutates state iff gold does
RIGHT_TARGET_RECORD = 0.05  # ...and it names the same order/user/item as gold

# A turn containing two or more <tool_call> blocks violates the retail policy's
# "at most one tool call per turn" rule and only the first is ever parsed --
# both by `completion_parsing` here and by vLLM's tool parser at eval time. A
# flat penalty makes that visible to the gradient instead of silently free.
MULTI_CALL_PENALTY = 0.05

_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>")


def count_tool_call_blocks(text: str) -> int:
    return len(_TOOL_CALL_OPEN_RE.findall(text))


def _id_values(arguments: dict[str, Any]) -> set[str]:
    """Every id-like value in an argument dict, flattened across scalars and
    lists (`item_ids`/`new_item_ids` are lists of item ids). Compared as
    strings so an int order id and its string form still match."""
    out: set[str] = set()
    for key, value in arguments.items():
        base = key[:-1] if key.endswith("s") else key
        if key not in ID_LIKE_KEYS and base not in ID_LIKE_KEYS:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for v in values:
            if isinstance(v, (str, int)) and not isinstance(v, bool):
                out.add(str(v))
    return out


def wrong_tool_partial_credit(
    predicted_name: str,
    predicted_args: dict[str, Any],
    expected_name: str,
    expected_args: dict[str, Any],
    env: Any,
) -> float:
    """Graded credit for a wrong-tool call, in `[0.0, WRONG_TOOL_CEILING]`.

    `env` is a `RetailEnv` (injected so this module needs no DB load of its
    own and the caller can reuse one across a whole batch)."""
    if predicted_name == MALFORMED_TOOL_CALL:
        return 0.0

    score = PARSEABLE_CALL
    if not env.has_tool(predicted_name):
        return score

    score += REAL_TOOL

    schema_ok, _ = env.validate_arguments(predicted_name, predicted_args)
    if schema_ok and not env.extra_arguments(predicted_name, predicted_args):
        score += SCHEMA_VALID

    if env.has_tool(expected_name) and env.tool_mutates_state(predicted_name) == env.tool_mutates_state(
        expected_name
    ):
        score += SAME_TOOL_CLASS

    gold_ids = _id_values(expected_args)
    if gold_ids and gold_ids & _id_values(predicted_args):
        score += RIGHT_TARGET_RECORD

    return min(score, WRONG_TOOL_CEILING)


def shaping_score(
    completion_text: str,
    expected_tool_name: Optional[str],
    expected_tool_arguments: dict[str, Any],
    env: Any,
    penalize_multi_call: bool = True,
) -> float:
    """The full auxiliary term for one completion. Can go slightly negative
    when `penalize_multi_call` fires on an otherwise-worthless completion;
    that is intended -- two tool calls in a turn is worse than one wrong one."""
    penalty = (
        MULTI_CALL_PENALTY
        if penalize_multi_call and count_tool_call_blocks(completion_text) > 1
        else 0.0
    )

    predicted_name, predicted_args = parse_completion(completion_text)

    # Gold is silence, or the policy stayed silent, or it got the tool right --
    # see the module docstring for why each of these gets no shaping.
    if expected_tool_name is None or predicted_name is None or predicted_name == expected_tool_name:
        return -penalty

    return wrong_tool_partial_credit(
        predicted_name, predicted_args, expected_tool_name, expected_tool_arguments, env
    ) - penalty


def make_grpo_shaping_func(penalize_multi_call: bool = True):
    """Builds the `reward_funcs`-shaped callable TRL calls, closing over a
    single shared `RetailEnv` so the 2.8MB `db.json` is parsed once per run,
    not once per completion. The env is only ever asked schema/type questions
    here -- never executed against -- so sharing it is safe."""
    from tau_forge.envs.retail import RetailEnv

    env = RetailEnv()

    def shaping_reward_func(
        prompts: list[str],
        completions: list[str],
        expected_tool_name: list[Optional[str]],
        expected_tool_arguments_json: list[str],
        **kwargs: Any,
    ) -> list[float]:
        return [
            shaping_score(c, name, json.loads(args_json), env, penalize_multi_call)
            for c, name, args_json in zip(completions, expected_tool_name, expected_tool_arguments_json)
        ]

    shaping_reward_func.__name__ = "shaping_reward"
    return shaping_reward_func
