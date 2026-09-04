"""Tests for the torch/trl-free half of the Phase 7 training pipeline:
dataset construction, completion parsing, and reward scoring. Runs anywhere
`uv sync` (no `--extra train`) already works -- no GPU, no real tokenizer, no
model download needed. `grpo_train.py` itself (the GRPOTrainer wiring) is not
covered here; it can only be exercised on the GPU box."""

import json

from tau_forge.train.completion_parsing import MALFORMED_TOOL_CALL, parse_completion
from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples, load_scenarios, to_hf_rows
from tau_forge.train.reward_adapter import score_completion


def _fake_apply_chat_template(messages, tools=None, tokenize=False, add_generation_prompt=True):
    """Stand-in for `tokenizer.apply_chat_template` -- exercises the plumbing
    without needing a real Qwen tokenizer or a model download."""
    parts = [f"[{m['role']}] {m['content']}" for m in messages]
    if tools:
        parts.insert(0, f"[tools] {json.dumps(tools)}")
    return "\n".join(parts)


def test_load_scenarios_matches_rule_checker_tally():
    # README: "Final tally: 30/30 cells, 541 scenarios".
    assert len(load_scenarios(DEFAULT_DATA_GLOB)) == 541


def test_build_examples_no_duplicate_ids():
    examples = build_examples()
    assert len(examples) == 541
    assert len({e.id for e in examples}) == 541


def test_build_examples_expected_tool_matches_scenario_shape():
    for ex in build_examples():
        if ex.category in ("ambiguous", "policy_violation"):
            # policy_violation can also be a single READ call -- only assert
            # the no-call cases here to avoid over-constraining.
            continue
        if ex.category == "out_of_scope":
            assert ex.expected_tool_name == "transfer_to_human_agents"


def test_to_hf_rows_shape_and_json_roundtrip():
    examples = build_examples()[:5]
    rows = to_hf_rows(
        examples,
        _fake_apply_chat_template,
        tools=[{"type": "function", "function": {"name": "noop"}}],
    )
    for row in rows:
        assert row["prompt"]
        assert set(row) == {
            "id",
            "category",
            "theme",
            "prompt",
            "expected_tool_name",
            "expected_tool_arguments_json",
        }
        json.loads(row["expected_tool_arguments_json"])  # must round-trip


def test_parse_completion_message_only():
    name, args = parse_completion("Sure, happy to help with that.")
    assert name is None
    assert args == {}


def test_parse_completion_valid_tool_call():
    text = (
        '<tool_call>\n{"name": "get_order_details", '
        '"arguments": {"order_id": "#W1234567"}}\n</tool_call>'
    )
    name, args = parse_completion(text)
    assert name == "get_order_details"
    assert args == {"order_id": "#W1234567"}


def test_parse_completion_malformed_tool_call_is_not_silently_a_no_call():
    # A garbled <tool_call> block must not be scored as if the model had
    # correctly chosen not to call a tool -- see completion_parsing.py's
    # module docstring for why that would be a reward-hacking path.
    name, _args = parse_completion("<tool_call>\nnot valid json\n</tool_call>")
    assert name == MALFORMED_TOOL_CALL


def test_score_completion_exact_match_on_real_scenario_scores_perfect():
    ex = next(e for e in build_examples() if e.expected_tool_name is not None)
    completion_text = (
        "<tool_call>\n"
        + json.dumps({"name": ex.expected_tool_name, "arguments": ex.expected_tool_arguments})
        + "\n</tool_call>"
    )
    score = score_completion(completion_text, ex.expected_tool_name, ex.expected_tool_arguments)
    assert score == 1.0


def test_score_completion_correct_silence_on_no_call_scenario_scores_perfect():
    assert any(e.expected_tool_name is None for e in build_examples())  # sanity: such scenarios exist
    score = score_completion("Let me ask a clarifying question first.", None, {})
    assert score == 1.0


def test_score_completion_unexpected_call_on_no_call_scenario_scores_zero():
    completion_text = (
        '<tool_call>\n{"name": "cancel_pending_order", '
        '"arguments": {"order_id": "#W1234567"}}\n</tool_call>'
    )
    assert score_completion(completion_text, None, {}) == 0.0


def test_score_completion_malformed_call_on_no_call_scenario_scores_zero_not_one():
    # The reward-hacking tripwire from completion_parsing.py, exercised
    # through the full scoring path.
    score = score_completion("<tool_call>\nnot valid json\n</tool_call>", None, {})
    assert score == 0.0
