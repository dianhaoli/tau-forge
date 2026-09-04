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


def parse_completion(text: str) -> tuple[Optional[str], dict[str, Any]]:
    match = _TOOL_CALL_RE.search(text)
    if not match:
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
