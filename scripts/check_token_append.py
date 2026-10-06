"""GPU-box check: appending token pieces == tokenizing a full chat-template
re-render, for every message type an episode produces, on the real Qwen
tokenizer.

GRPO rollouts (`tau_forge.train.episode_rollout`) build each sequence by
APPENDING ids: the opening prompt, each turn's sampled ids + `<|im_end|>`,
then the tokenized environment delta (`render_env_delta`). A tau2 eval instead
re-renders the whole conversation with the chat template and tokenizes it in
one go. This script asserts the two give identical ids (and identical text),
so the policy is trained on exactly the token sequences it sees at eval:

  1. tokenizer facts the trainer relies on: `<|im_end|>` and `<|im_start|>`
     are single tokens and eos == `<|im_end|>` (else TRL's
     mask_truncated_completions masks every episode);
  2. synthetic conversations covering each message type: assistant text ->
     user, bare tool call -> tool, content + tool call -> tool, two tool calls
     -> two tool messages in one user turn, tool error, unicode / leading
     newline / trailing space user text, conversation ending on an assistant;
  3. real episodes of the reference agents (oracle, no_confirm, transfer, ...)
     on a few tasks per template, with the real system prompt and 16 tools.

It also counts assistant messages whose raw sampled text (`"raw"`, when the
runner stores it) differs from the canonical re-render -- how far training
history (raw ids) drifts from what a tau2 eval feeds back (informational).

Usage (needs `transformers`; no GPU actually required):
    python scripts/check_token_append.py --model Qwen/Qwen3-4B-Instruct-2507
Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tau_forge.train.episode_rollout import (  # noqa: E402
    IM_END,
    IM_START,
    check_append_equals_rerender,
    encode,
    render_assistant_body,
    special_token_id,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--tasks", default=str(REPO_ROOT / "data" / "episodes" / "episodes_s1_sub50.jsonl"))
    p.add_argument("--per-template", type=int, default=2)
    p.add_argument("--modes", default="oracle,no_confirm,transfer,comply")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def _tc(name: str, args: dict[str, Any], k: int) -> dict[str, Any]:
    return {"id": f"call_{k}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def synthetic_cases(opening: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Conversations continuing `opening` (system, greeting, opening user)."""
    o = list(opening)
    lookup = _tc("find_user_id_by_email", {"email": "a.b@example.com"}, 1)
    details = _tc("get_user_details", {"user_id": "ab_1234"}, 2)
    order = _tc("get_order_details", {"order_id": "#W0000001"}, 3)
    return {
        "text_then_user": o + [
            {"role": "assistant", "content": "Could you share your email?"},
            {"role": "user", "content": "Sure, it's a.b@example.com."},
            {"role": "assistant", "content": "Thanks!"},
        ],
        "tool_call_then_tool": o + [
            {"role": "assistant", "content": "", "tool_calls": [lookup]},
            {"role": "tool", "tool_call_id": "call_1", "name": "find_user_id_by_email", "content": "ab_1234"},
        ],
        "content_and_tool_call": o + [
            {"role": "assistant", "content": "Let me look that up.", "tool_calls": [details]},
            {"role": "tool", "tool_call_id": "call_2", "name": "get_user_details",
             "content": json.dumps({"user_id": "ab_1234", "orders": ["#W0000001"]})},
            {"role": "assistant", "content": "Found it."},
        ],
        "two_calls_two_tools": o + [
            {"role": "assistant", "content": "", "tool_calls": [details, order]},
            {"role": "tool", "tool_call_id": "call_2", "name": "get_user_details", "content": "{\"user_id\": \"ab_1234\"}"},
            {"role": "tool", "tool_call_id": "call_3", "name": "get_order_details", "content": "{\"status\": \"pending\"}"},
            {"role": "assistant", "content": "Your order is pending."},
            {"role": "user", "content": "Great, thanks."},
        ],
        "tool_error": o + [
            {"role": "assistant", "content": "", "tool_calls": [order]},
            {"role": "tool", "tool_call_id": "call_3", "name": "get_order_details", "content": "Error: Order not found"},
        ],
        "odd_user_text": o + [
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "\nCafé — order #W0000001 😊 please  "},
            {"role": "assistant", "content": "On it."},
            {"role": "user", "content": "###STOP###"},
        ],
    }


def reference_episodes(args: argparse.Namespace, system_message: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    from tau_forge.episodes.generate import read_jsonl
    from tau_forge.episodes.reference_agents import ReferenceAgent
    from tau_forge.episodes.runner import run_episode

    tasks = read_jsonl(args.tasks)
    picked: dict[str, list] = {}
    for t in tasks:
        if len(picked.setdefault(t.template, [])) < args.per_template:
            picked[t.template].append(t)
    out = {}
    for template, ts in sorted(picked.items()):
        for t in ts:
            for mode in args.modes.split(","):
                try:
                    res = run_episode(t, ReferenceAgent(t, mode), system_message=system_message)
                except Exception as e:  # noqa: BLE001 -- a mode that does not apply to this template
                    print(f"  skip {t.id} {mode}: {type(e).__name__}: {e}")
                    continue
                out[f"{t.id}:{mode}"] = res.messages
    return out


def main(argv: Optional[list[str]] = None, *, tokenizer: Any = None, tools: Any = None,
         system_message: Optional[dict[str, str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tools is None:
        from tau_forge.envs.retail import RetailEnv

        tools = RetailEnv().all_openai_schemas()
    if system_message is None:
        from tau_forge.train.dataset import _default_policy_text, _system_message

        system_message = _system_message(_default_policy_text())

    failures: list[str] = []
    facts = {
        "im_end_ids": encode(tokenizer, IM_END),
        "im_start_ids": encode(tokenizer, IM_START),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    im_end = special_token_id(tokenizer, IM_END)
    if len(facts["im_end_ids"]) != 1 or len(facts["im_start_ids"]) != 1:
        failures.append(f"special tokens are not single ids: {facts}")
    if facts["eos_token_id"] != im_end:
        failures.append(f"eos_token_id {facts['eos_token_id']} != <|im_end|> {im_end}")

    opening = [system_message, {"role": "assistant", "content": "Hi! How can I help you today?"},
               {"role": "user", "content": "I want to cancel an order."}]
    cases = {f"synthetic:{k}": v for k, v in synthetic_cases(opening).items()}
    cases.update({f"episode:{k}": v for k, v in reference_episodes(args, system_message).items()})

    kinds: Counter = Counter()
    raw_total = raw_diff = 0
    results = {}
    for name, messages in cases.items():
        rep = check_append_equals_rerender(tokenizer, messages, tools=tools)
        results[name] = rep
        kinds.update(rep["kinds"])
        if not rep["ok"]:
            failures.append(f"{name}: {json.dumps({k: v for k, v in rep.items() if 'mismatch' in k}, ensure_ascii=False)}")
        for m in messages[3:]:
            if m["role"] == "assistant" and isinstance(m.get("raw"), str):
                raw_total += 1
                raw_diff += m["raw"] != render_assistant_body(m)

    print(f"[check_token_append] tokenizer facts: {facts}")
    print(f"[check_token_append] {len(cases)} conversations, message kinds covered: {dict(kinds)}")
    print(f"[check_token_append] passed {sum(r['ok'] for r in results.values())}/{len(results)}")
    if raw_total:
        print(f"[check_token_append] raw sampled text != canonical re-render on {raw_diff}/{raw_total} assistant turns "
              "(training history keeps the raw ids; a tau2 eval re-renders)")
    for f in failures:
        print(f"[check_token_append] FAIL {f}")
    summary = {"facts": facts, "kinds": dict(kinds), "n": len(results), "failures": failures,
               "raw_turns": raw_total, "raw_differs": raw_diff}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"summary": summary, "results": results}, indent=1, default=str))
    return summary


if __name__ == "__main__":
    sys.exit(1 if main()["failures"] else 0)
