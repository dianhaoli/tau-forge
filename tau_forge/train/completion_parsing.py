"""Parses a policy-model completion into tool calls, exactly as eval does.

Qwen's own tool-calling convention wraps a call in a `<tool_call>...</tool_call>`
block containing a JSON object `{"name": ..., "arguments": {...}}` -- the format
`tau_forge.train.dataset.TOOL_CALL_FORMAT_INSTRUCTION` tells the policy model to
use, matching how Qwen chat templates render tool calls natively (not an
invented convention this project made up).

A completion with no `<tool_call>` block is message-only -- the correct answer
for `ambiguous`/`policy_violation`/most `out_of_scope` scenarios, per
`reward.reward`'s `Action(tool_name=None)` convention.

A `<tool_call>` block that's present but malformed is deliberately **not**
treated the same as no call at all by the single-step grader: doing so would
let a garbled tool-call attempt score a free `correct_no_call` 1.0 on a
scenario where the right answer actually is silence -- a cheap reward-hacking
path this project explicitly flagged as a risk to full-parameter RLVR (see
docs/phase7_aws_setup.md, "Methodology risks"). Instead `parse_completion`
grades it as an attempted call to a sentinel tool name that can't match any
real gold tool, which `reward()` correctly scores 0 either way (wrong tool, or
an unexpected call when none was expected).

The same holds for a *bare* tool-call object with no tags at all -- e.g.
`{"name": "cancel_pending_order", "arguments": {...}}` as plain text, or inside
a ```json fence. That used to parse as message-only and earned the full 1.0 on
every no-call gold (182 of 541 scenarios), the same hole through a different
door. It is now `MALFORMED_TOOL_CALL` too: an attempted call that can never
match gold.

Parity with eval, deliberately over leniency
--------------------------------------------
Phase 8 serves the policy with vLLM's `--tool-call-parser hermes`
(`tau_forge/eval/run_tau2.py`), and tau2 executes every call it returns.
`parse_all_completion` mirrors `Hermes2ProToolParser.extract_tool_calls`
(vLLM v0.10.1 and main; `hermes_extract_tool_calls` below is that function's
logic verbatim) plus the step tau2's `llm_utils.generate` adds on top:
  * every `<tool_call>` block is parsed (regex
    `<tool_call>(.*?)</tool_call>|<tool_call>(.*)`, so an unclosed final block
    still counts), and each body must `json.loads` to an object with a string
    `name` and an `arguments` key;
  * if ANY block fails that, hermes returns the whole completion as plain
    text -- no call at all, even for the blocks that were fine;
  * `arguments` that is present but not a JSON object (null, a string, a
    list) passes hermes but makes tau2's `ToolCall(arguments: dict)` raise and
    end the simulation; here it makes the turn text, never a call with `{}`;
  * the assistant content tau2 records is the text before the first block.
So at eval a bare JSON object, a ```json fence inside the tags, trailing prose
inside the tags, a `"parameters"` key, a missing `arguments` key or a
`{"function": {...}}` wrapper is *never* a tool call, and two valid blocks are
two calls. This module mirrors that rather than "repairing" those formats into
valid calls: a parser more forgiving than eval's would train the policy to
emit formats that silently stop working the moment it is evaluated, and the
measured reward would overstate what eval will see.

`parse_completion` is the single-call view the single-step trainer grades:
the first call `parse_all_completion` returns; `MALFORMED_TOOL_CALL` for
anything hermes rejects and, stricter than hermes, for an empty name or a bare
call object with no tags; else `(None, {})`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

MALFORMED_TOOL_CALL = "__malformed_tool_call__"

# vLLM Hermes2ProToolParser.tool_call_regex, verbatim. The second alternative
# matches an unclosed final block through to end-of-string.
HERMES_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
HERMES_START_TOKEN = "<tool_call>"

# Keys that, next to a string `name`, mark a JSON object as a tool-call attempt.
# `parameters` is the other common spelling models fall into; it is detected
# here only to classify the attempt, never accepted as a valid call.
_CALL_ARGUMENT_KEYS = ("arguments", "parameters")

_JSON_DECODER = json.JSONDecoder()


def _is_call_object(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and bool(obj["name"])
        and any(key in obj for key in _CALL_ARGUMENT_KEYS)
    )


def contains_bare_tool_call(text: str) -> bool:
    """True if `text` (assumed to have no `<tool_call>` tag) contains a JSON
    object shaped like a tool call anywhere in it -- plain, fenced, or nested
    inside a wrapper such as `{"function": {...}}` or a list, since every `{`
    is tried as a start position. A reply that merely mentions a name, or
    contains JSON without an `arguments`/`parameters` key, is not a call."""
    if '"name"' not in text:  # cheap reject for the overwhelmingly common prose reply
        return False
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = _JSON_DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if _is_call_object(obj):
            return True
    return False


def hermes_extract_tool_calls(model_output: str) -> tuple[bool, list[tuple[str, str]], Optional[str]]:
    """`Hermes2ProToolParser.extract_tool_calls` (non-streaming), logic
    verbatim: returns `(tools_called, [(name, arguments_json)], content)`.
    vLLM builds `FunctionCall(name=...)`, a pydantic `str` field, so a
    non-string name raises there; the explicit check reproduces that."""
    if HERMES_START_TOKEN not in model_output:
        return False, [], model_output
    try:
        function_call_tuples = HERMES_TOOL_CALL_RE.findall(model_output)
        raw_function_calls = [json.loads(match[0] if match[0] else match[1]) for match in function_call_tuples]
        tool_calls = []
        for function_call in raw_function_calls:
            name = function_call["name"]
            arguments = json.dumps(function_call["arguments"], ensure_ascii=False)
            if not isinstance(name, str):
                raise TypeError("FunctionCall.name must be a string")
            tool_calls.append((name, arguments))
        content = model_output[: model_output.find(HERMES_START_TOKEN)]
        return True, tool_calls, content if content else None
    except Exception:  # noqa: BLE001 -- hermes catches everything and returns the text
        return False, [], model_output


@dataclass(frozen=True)
class ParsedCompletion:
    """What eval makes of one completion.

    `calls` are the tool calls tau2 would execute, in order. `content` is the
    assistant text tau2 records: the text before the first block for a tool
    turn ("" when there is none; hermes says None), the whole completion for a
    text turn. `malformed` marks a text turn that LOOKS like an attempted
    call -- a `<tool_call>` tag eval rejected, or a bare call object without
    tags -- so callers can count it; eval shows it to the user as text."""

    content: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    malformed: bool = False


def parse_all_completion(text: str) -> ParsedCompletion:
    text = text or ""
    called, raw_calls, content = hermes_extract_tool_calls(text)
    if called:
        calls = []
        for name, arguments_json in raw_calls:
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                # tau2 would crash building ToolCall(arguments=...): never a call.
                return ParsedCompletion(text, [], True)
            calls.append((name, arguments))
        return ParsedCompletion(content or "", calls, False)
    return ParsedCompletion(text, [], HERMES_START_TOKEN in text or contains_bare_tool_call(text))


def parse_completion(text: str) -> tuple[Optional[str], dict[str, Any]]:
    """Single-call view for the single-step trainer (see the module docstring).
    With several valid blocks it returns the first; eval would run them all."""
    parsed = parse_all_completion(text)
    if parsed.calls:
        name, arguments = parsed.calls[0]
        if not name:
            return MALFORMED_TOOL_CALL, {}
        return name, arguments
    if parsed.malformed:
        return MALFORMED_TOOL_CALL, {}
    return None, {}
