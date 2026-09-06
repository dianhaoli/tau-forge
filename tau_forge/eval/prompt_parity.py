"""Keeps the training-time system prompt and the Phase 8 evaluation-time
system prompt byte-identical.

The gap this closes
-------------------
`tau_forge.train.dataset._system_message` builds every training prompt as

    SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION + TOOL_CALL_FORMAT_INSTRUCTION,
        domain_policy=retail_policy,
    )

while tau2's own `LLMAgent.system_prompt` (`tau2/agent/llm_agent.py`) builds it as

    SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=...)

-- with no `TOOL_CALL_FORMAT_INSTRUCTION`. So `tau2 run` sends a system prompt
the policy has never been optimized under. That matters more here than generic
prompt drift usually does, for two specific reasons:

* RLVR moves the policy *conditioned on* the prompt it trained under. A prompt
  the trained weights never saw is off-distribution for exactly the behavior
  the run paid for.
* The suffix is not cosmetic. Its escalation paragraph (added in "Tell the
  model to escalate, not go silent, when no tool covers a request") is what
  roughly doubled the `out_of_scope` mean score and dropped that category's
  zero-variance rate from ~85% to ~49%. Under the stock tau2 prompt, that
  instruction is simply absent at eval time.

`patch_agent_instruction()` appends the same suffix to tau2's module-level
`AGENT_INSTRUCTION` before any agent is constructed, so the two prompts match.

Applying this is a change to the benchmark harness, and the honest way to use
it is the only way it produces a valid number: **apply it identically to the
baseline run and to every trained-checkpoint run.** The reported improvement is
then a difference in weights under a fixed prompt, which is what the claim
"RLVR improved the model" actually means. Running the baseline stock and the
trained model patched would measure the prompt and the training together and
attribute all of it to training.

The alternative -- drop the suffix from training and match stock tau2 -- is
equally valid and needs no patch at all; `assert_prompts_match` is what tells
you which regime you are in, either way. Pick one, record it, do not mix.
"""

from __future__ import annotations


def training_system_prompt(policy_text: str | None = None) -> str:
    from tau_forge.train.dataset import _default_policy_text, _system_message

    return _system_message(policy_text if policy_text is not None else _default_policy_text())["content"]


def eval_system_prompt(policy_text: str | None = None) -> str:
    """What tau2's `LLMAgent` will actually send, given the current (possibly
    patched) module-level `AGENT_INSTRUCTION`. Reads tau2's own constants
    rather than reconstructing the format string here, so it cannot drift from
    the harness it is meant to mirror."""
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    from tau_forge.train.dataset import _default_policy_text

    return SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=policy_text if policy_text is not None else _default_policy_text(),
    )


def patch_agent_instruction() -> str:
    """Append `TOOL_CALL_FORMAT_INSTRUCTION` to tau2's `AGENT_INSTRUCTION`,
    in place, idempotently. Must run before any `LLMAgent` is constructed --
    `system_prompt` reads the module global at call time, so importing this
    and calling it at the top of an eval entrypoint is sufficient."""
    from tau2.agent import llm_agent

    from tau_forge.train.dataset import TOOL_CALL_FORMAT_INSTRUCTION

    if not llm_agent.AGENT_INSTRUCTION.endswith(TOOL_CALL_FORMAT_INSTRUCTION):
        llm_agent.AGENT_INSTRUCTION = llm_agent.AGENT_INSTRUCTION + TOOL_CALL_FORMAT_INSTRUCTION
    return llm_agent.AGENT_INSTRUCTION


def prompts_match(policy_text: str | None = None) -> bool:
    return training_system_prompt(policy_text) == eval_system_prompt(policy_text)


def assert_prompts_match(policy_text: str | None = None) -> None:
    if prompts_match(policy_text):
        return
    training = training_system_prompt(policy_text)
    evaluation = eval_system_prompt(policy_text)
    raise AssertionError(
        "Training and evaluation system prompts differ "
        f"({len(training)} vs {len(evaluation)} chars). Call patch_agent_instruction() "
        "before constructing the agent, or build the training dataset without "
        "TOOL_CALL_FORMAT_INSTRUCTION -- and use the same regime for the baseline run."
    )
