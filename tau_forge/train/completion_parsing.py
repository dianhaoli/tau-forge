"""Parses a policy-model completion into a graded `Action`.

Qwen's own tool-calling convention wraps a call in a `<tool_call>...</tool_call>`
block containing a JSON object `{"name": ..., "arguments": {...}}` -- the format
`tau_forge.train.dataset.TOOL_CALL_FORMAT_INSTRUCTION` tells the policy model to
use, matching how Qwen chat templates render tool calls natively (not an
invented convention this project made up).

A completion with no `<tool_call>` block is message-only -- the correct answer
for `ambiguous`/`policy_violation`/most `out_of_scope` scenarios, per
`reward.reward`'s `Action(tool_name=None)` convention.

A `<tool_call>` block that's present but malformed (bad JSON, missing/non-string
`name`) is deliberately **not** treated the same as no call at all: doing so
would let a garbled tool-call attempt score a free `correct_no_call` 1.0 on a
scenario where the right answer actually is silence -- a cheap reward-hacking
path this project explicitly flagged as a risk to full-parameter RLVR (see
docs/phase7_aws_setup.md, "Methodology risks"). Instead it's graded as an
attempted call to a sentinel tool name that can't match any real gold tool,
which `reward()` correctly scores 0 either way (wrong tool, or an unexpected
call when none was expected).

The same holds for a *bare* tool-call object with no tags at all -- e.g.
`{"name": "cancel_pending_order", "arguments": {...}}` as plain text, or inside
a ```json fence. That used to parse as message-only and earned the full 1.0 on
every no-call gold (182 of 541 scenarios), the same hole through a different
door. It is now `MALFORMED_TOOL_CALL` too: an attempted call that can never
match gold.

Parity with eval, deliberately over leniency
--------------------------------------------
Phase 8 serves the policy with vLLM's `--tool-call-parser hermes`
(`tau_forge/eval/run_tau2.py`). That parser only looks inside `<tool_call>`
tags, `json.loads` the whole body, and reads `name` and `arguments`; anything
it cannot parse is passed through as plain assistant text with no call. So at
eval a bare JSON object, a ```json fence inside the tags, trailing prose inside
the tags, a `"parameters"` key, or a `{"function": {...}}` wrapper is *never* a
tool call. This module mirrors that rather than "repairing" those formats into
valid calls: a parser more forgiving than eval's would train the policy to
emit formats that silently stop working the moment it is evaluated, and the
measured reward would overstate what eval will see. Concretely, a `"parameters"`
key or a non-object `arguments` value still parses with empty arguments (and
so fails schema validation in `reward()`), and none of the above formats is
ever accepted as the call it resembles.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

MALFORMED_TOOL_CALL = "__malformed_tool_call__"

# Matches through to end-of-string if the closing tag is missing (e.g. a
# completion truncated by max_completion_length mid-call) rather than failing
# to match at all -- a truncated tool-call attempt is still an attempt, not a
# silent no-call. Body is whatever's between the tags, valid JSON or not; a
# non-JSON or brace-less body (e.g. "not valid json", no braces at all) must
# still be caught as an attempted call below, not fall through as if the tag
# were never there.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)

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


def parse_completion(text: str) -> tuple[Optional[str], dict[str, Any]]:
    match = _TOOL_CALL_RE.search(text)
    if not match:
        if contains_bare_tool_call(text):
            return MALFORMED_TOOL_CALL, {}
        return None, {}
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return MALFORMED_TOOL_CALL, {}
    name = payload.get("name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name:
        return MALFORMED_TOOL_CALL, {}
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments
