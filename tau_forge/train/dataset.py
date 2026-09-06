"""Builds the GRPO training dataset from the Phase 2 synthetic scenarios.

Per the held-out data policy (README), this is the **only** data source a live
training run may touch -- never `data/tau2/domains/retail/tasks.json` (the real
114 tasks; those are for Phase 5 decontamination and Phase 8 evaluation only).

Every scenario is graded against the plain, unmodified shipped `db.json` --
deliberately matching `tau_forge.validate.rule_checker`'s own re-execution
methodology (`RetailEnv()` with no args, no per-scenario snapshot derived from
`prior_turns`; see that module's re-execution comment for why a scenario needing
special DB state was authored by executing the setup call for real and writing
it into `prior_turns` as narrated dialogue, not by shipping a derived snapshot).

Deliberately torch/trl-free: only `tau_forge.train.grpo_train` needs the GPU
training stack. This module (and its tests) run anywhere `uv sync` (no
`--extra train`) already works.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_GLOB = str(REPO_ROOT / "data" / "synthetic" / "raw" / "*.json")

# Appended to tau2's own AGENT_INSTRUCTION (see _system_message) so the policy
# model is told the exact wire format its tool calls will be parsed with --
# `tau_forge.train.completion_parsing` parses this same <tool_call> convention,
# which is Qwen's own native tool-calling format, not an invented one.
TOOL_CALL_FORMAT_INSTRUCTION = (
    "\n\nWhen you decide to make a tool call, emit exactly one, in this exact "
    'form:\n<tool_call>\n{"name": "<tool_name>", "arguments": {...}}\n</tool_call>\n'
    "If the correct action is not calling a tool right now -- e.g. the request "
    "is ambiguous, out of policy, or a plain reply is what's needed -- send a "
    "message with no <tool_call> block instead.\n\n"
    "If the user's request cannot be satisfied by any tool available to you "
    "-- there is simply no capability that does what they're asking, even "
    "though the request itself is clear -- that is not a case for a plain "
    "reply. Call transfer_to_human_agents with a summary of the issue. Do "
    "not try to solve it with an unrelated tool, and do not just apologize "
    "or ask clarifying questions in place of escalating."
)


@dataclass
class TrainingExample:
    id: str
    category: str
    theme: str
    prompt_messages: list[dict[str, str]]
    expected_tool_name: Optional[str]
    expected_tool_arguments: dict[str, Any] = field(default_factory=dict)


def load_scenarios(data_glob: str = DEFAULT_DATA_GLOB) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for path in sorted(glob.glob(data_glob)):
        scenarios.extend(json.loads(Path(path).read_text()))
    return scenarios


def _system_message(policy_text: str, include_format_instruction: bool = True) -> dict[str, str]:
    """The exact system message a training prompt carries.

    `tau_forge.eval.prompt_parity` patches tau2's module-level
    `AGENT_INSTRUCTION` with the same suffix so the Phase 8 eval prompt matches
    this one byte for byte -- which means this function can be called in a
    process where the suffix is *already* on `AGENT_INSTRUCTION`. Appending
    unconditionally there would silently build training prompts carrying it
    twice, so the append is idempotent. `include_format_instruction=False`
    builds the stock-tau2 prompt instead, for the alternative parity regime
    (train against the unmodified prompt, evaluate with --stock-prompt)."""
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    instruction = AGENT_INSTRUCTION
    if include_format_instruction and not instruction.endswith(TOOL_CALL_FORMAT_INSTRUCTION):
        instruction = instruction + TOOL_CALL_FORMAT_INSTRUCTION
    elif not include_format_instruction and instruction.endswith(TOOL_CALL_FORMAT_INSTRUCTION):
        instruction = instruction[: -len(TOOL_CALL_FORMAT_INSTRUCTION)]

    content = SYSTEM_PROMPT.format(agent_instruction=instruction, domain_policy=policy_text)
    return {"role": "system", "content": content}


def _default_policy_text() -> str:
    from tau2.domains.retail.utils import RETAIL_POLICY_PATH

    return Path(RETAIL_POLICY_PATH).read_text()


def scenario_to_example(scenario: dict[str, Any], system_message: dict[str, str]) -> TrainingExample:
    messages = [system_message]
    for turn in scenario.get("prior_turns", []):
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": scenario["user_message"]})

    calls = scenario.get("expected_tool_calls") or []
    if calls:
        expected_tool_name: Optional[str] = calls[0]["name"]
        expected_tool_arguments = calls[0].get("arguments", {})
    else:
        expected_tool_name = None
        expected_tool_arguments = {}

    return TrainingExample(
        id=scenario["id"],
        category=scenario["category"],
        theme=scenario["theme"],
        prompt_messages=messages,
        expected_tool_name=expected_tool_name,
        expected_tool_arguments=expected_tool_arguments,
    )


def build_examples(
    data_glob: str = DEFAULT_DATA_GLOB,
    policy_text: Optional[str] = None,
    include_format_instruction: bool = True,
) -> list[TrainingExample]:
    system_message = _system_message(
        policy_text if policy_text is not None else _default_policy_text(),
        include_format_instruction=include_format_instruction,
    )
    return [scenario_to_example(s, system_message) for s in load_scenarios(data_glob)]


def render_prompt(
    example: TrainingExample,
    apply_chat_template: Callable[..., str],
    tools: list[dict[str, Any]],
) -> str:
    """`apply_chat_template` is injected (normally
    `tokenizer.apply_chat_template` bound with `tokenize=False,
    add_generation_prompt=True`) rather than imported directly, so this
    module never needs a real tokenizer/model download to be tested."""
    return apply_chat_template(example.prompt_messages, tools=tools)


def to_hf_rows(
    examples: list[TrainingExample],
    apply_chat_template: Callable[..., str],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One dict per example, in the shape `datasets.Dataset.from_list` and
    `GRPOTrainer` expect: a `prompt` column plus extra columns
    (`expected_tool_name`, `expected_tool_arguments_json`) that TRL passes
    through to the reward function as kwargs, aligned per-completion."""
    rows = []
    for ex in examples:
        rows.append(
            {
                "id": ex.id,
                "category": ex.category,
                "theme": ex.theme,
                "prompt": render_prompt(ex, apply_chat_template, tools),
                "expected_tool_name": ex.expected_tool_name,
                "expected_tool_arguments_json": json.dumps(ex.expected_tool_arguments),
            }
        )
    return rows
