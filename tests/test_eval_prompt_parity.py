"""The guard on the train/eval prompt gap: a policy optimized under one system
prompt and evaluated under another is measured off-distribution, and the
`TOOL_CALL_FORMAT_INSTRUCTION` suffix is not cosmetic (its escalation
paragraph is what fixed the out_of_scope category). These tests fail loudly if
the two prompts drift apart again."""

import pytest

from tau_forge.eval import prompt_parity
from tau_forge.train.dataset import TOOL_CALL_FORMAT_INSTRUCTION, _system_message, build_examples


@pytest.fixture
def unpatched():
    """tau2's AGENT_INSTRUCTION is a module global; patching it is process-wide,
    so restore it or every later test inherits the patch."""
    from tau2.agent import llm_agent

    original = llm_agent.AGENT_INSTRUCTION
    yield llm_agent
    llm_agent.AGENT_INSTRUCTION = original


def test_stock_tau2_prompt_does_not_match_training_prompt(unpatched):
    """The finding: without the patch, `tau2 run` sends a prompt the trained
    weights have never seen."""
    assert not prompt_parity.prompts_match()
    with pytest.raises(AssertionError, match="differ"):
        prompt_parity.assert_prompts_match()


def test_patch_makes_the_two_prompts_byte_identical(unpatched):
    prompt_parity.patch_agent_instruction()
    assert prompt_parity.prompts_match()
    prompt_parity.assert_prompts_match()


def test_patch_is_idempotent_and_does_not_double_append(unpatched):
    once = prompt_parity.patch_agent_instruction()
    twice = prompt_parity.patch_agent_instruction()
    assert once == twice
    assert twice.count(TOOL_CALL_FORMAT_INSTRUCTION) == 1


def test_dataset_build_is_stable_whether_or_not_tau2_is_patched(unpatched):
    """Building the dataset inside an already-patched process must produce the
    same prompt, not one carrying the suffix twice."""
    before = _system_message("POLICY")["content"]
    prompt_parity.patch_agent_instruction()
    after = _system_message("POLICY")["content"]
    assert before == after
    assert after.count(TOOL_CALL_FORMAT_INSTRUCTION) == 1


def test_stock_prompt_regime_strips_the_suffix(unpatched):
    stock = _system_message("POLICY", include_format_instruction=False)["content"]
    assert TOOL_CALL_FORMAT_INSTRUCTION not in stock

    prompt_parity.patch_agent_instruction()
    still_stock = _system_message("POLICY", include_format_instruction=False)["content"]
    assert still_stock == stock, "the stock regime must ignore a patched AGENT_INSTRUCTION too"


def test_build_examples_honours_the_prompt_regime():
    augmented = build_examples()[0].prompt_messages[0]["content"]
    stock = build_examples(include_format_instruction=False)[0].prompt_messages[0]["content"]
    assert TOOL_CALL_FORMAT_INSTRUCTION in augmented
    assert TOOL_CALL_FORMAT_INSTRUCTION not in stock


def test_run_config_holds_the_comparison_fixed():
    from tau_forge.eval.run_tau2 import build_run_config, parse_args

    args = parse_args(["--label", "baseline"])
    config = build_run_config(args)
    assert config.domain == "retail"
    assert config.task_split_name == "test"
    assert config.num_trials == 4, "one trial cannot resolve a training delta through a stochastic user"
    assert config.llm_args_agent["temperature"] == 0.0
    assert config.llm_args_agent["api_base"].endswith("/v1")
    assert "baseline" in config.save_to
