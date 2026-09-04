"""Renders the Phase 2 per-cell scenario-generation prompt.

One call to `render_cell_prompt` produces the complete, self-contained brief for
one subagent covering one (category, theme) cell -- including the real tool
source, policy text, and DB schema (read fresh off disk each render, so this
always reflects whatever's actually checked out under third_party/tau2-bench),
plus the running dedup registry. Reused for all 30 cells, not just the Phase 2
pilot -- keep this the single source of truth for the template rather than
copy-pasting it per cell.
"""

from __future__ import annotations

from pathlib import Path

from tau_forge.gen.taxonomy import CATEGORIES, THEMES

REPO_ROOT = Path(__file__).resolve().parents[2]
_TAU2 = REPO_ROOT / "third_party" / "tau2-bench"
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"
REGISTRY_UPDATES_DIR = REPO_ROOT / "data" / "synthetic" / "registry_updates"


def _read(relpath: str) -> str:
    return (_TAU2 / relpath).read_text()


def render_cell_prompt(
    category: str,
    theme: str,
    seen_so_far: list[str],
    n_scenarios: str = "15-20",
) -> str:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category}. Known: {sorted(CATEGORIES)}")
    if theme not in THEMES:
        raise ValueError(f"Unknown theme: {theme}. Known: {sorted(THEMES)}")

    tools_source = _read("src/tau2/domains/retail/tools.py")
    policy_text = _read("data/tau2/domains/retail/policy.md")
    data_model_source = _read("src/tau2/domains/retail/data_model.py")

    seen_block = (
        "\n".join(f"- {line}" for line in seen_so_far)
        if seen_so_far
        else "(none yet -- you're in an early wave, nothing to avoid overlapping yet)"
    )

    out_path = RAW_DIR / f"{category}__{theme}.json"
    registry_path = REGISTRY_UPDATES_DIR / f"{category}__{theme}.jsonl"

    return f"""You are generating training data for a retail customer-service tool-calling
agent (the tau2-bench retail domain). This data will be used to train and
evaluate an LLM agent with reinforcement learning, so gold answers must be
genuinely correct, not just plausible-looking -- a wrong gold label silently
teaches the wrong behavior and the reward function has no way to catch it later.

## Working environment

Repo root: {REPO_ROOT} (already fully set up -- a `uv`-managed venv exists,
don't reinstall or re-sync anything). Run python via `uv run python3 ...` from
that directory.

You have a real mock database and a real executor to check your work against --
USE THEM. Do not write expected_tool_calls from imagination alone.

```python
from tau_forge.envs.retail import RetailEnv, execute_against

env = RetailEnv()  # loads a FRESH in-memory copy of the real db.json -- never
                    # writes back to disk, so it's always safe to make more of
                    # these; nothing you do here can corrupt the shared fixture
                    # other work in this project depends on.

# Explore for real entities that fit your scenario -- never invent an id:
for order in env.db.orders.values():
    if order.status == "delivered":
        ...  # find one that actually fits your theme, inspect order.items etc.

# Propose an action, actually run it, look at the REAL result:
result = env.execute("return_delivered_order_items", {{
    "order_id": "#W1234567", "item_ids": ["1008292230"],
    "payment_method_id": "gift_card_0000000",
}})
print(result.ok, result.value, result.error)
# If it failed, your proposed action or arguments were wrong -- fix and retry,
# don't paper over it or hand-wave the JSON anyway.

# You MAY chain several real executions here to explore what a multi-step
# interaction would actually look like end to end (e.g. look up the order,
# then look up the product, then try the mutation) -- that's a legitimate way
# to build confidence in a scenario. But `expected_tool_calls` in your OUTPUT
# must never contain more than one call. policy.md is explicit: "You should
# at most make one tool call at a time, and if you take a tool call, you
# should not respond to the user at the same time." A chained
# [lookup, mutation] answer describes two agent turns, not one, and isn't
# something a policy-compliant agent would ever emit in a single turn -- it's
# also not gradable by the Phase 4 reward function, which only scores one
# Action. If your exploration takes several real calls, that just tells you
# which ONE of them is correct for THIS scenario's turn; write that one down,
# and if the scenario is naturally later-stage (the lookup already happened),
# say so in `prior_turns` as text instead of trying to cram both steps in.
```

For a scenario with an empty expected_tool_calls (ambiguous / policy_violation /
out_of_scope), there's nothing to execute for the final answer, but any order_id
/ user_id / product_id / item_id / payment_method_id you reference anywhere
(prior_turns, user_message, distractor rationale) must still be a real one you
looked up in `env.db` -- never invent one.

Every id appearing anywhere in your output must be real. Fabricated ids make a
scenario unusable: Phase 3's automated rule checker re-executes every
expected_tool_calls against this exact same database and discards anything that
doesn't check out -- including a hard check that `expected_tool_calls` never has
more than one entry.

## Match the real benchmark's tool-use distribution, not just its tool list

Across the real 114 τ²-bench retail tasks' reference trajectories (550 tool
calls total), **64.9% are lookup/identity calls** (`get_order_details` alone is
30.5% of every call made; `find_user_id_by_name_zip`/`find_user_id_by_email`,
`get_user_details`, `get_product_details`, `get_item_details` account for most
of the rest) and only **35.1% are the mutating/generic "final" action**
(`return_delivered_order_items`, `modify_pending_order_items`,
`exchange_delivered_order_items`, `cancel_pending_order`,
`modify_pending_order_address`, `calculate`, `modify_user_address`,
`transfer_to_human_agents`, `modify_pending_order_payment`, roughly in that
order of frequency). A first pilot batch of this generation pipeline skewed
almost entirely toward the mutating/generic end (near-zero lookup calls as the
*answer* to a scenario) because every scenario assumed identity/order lookup
was already resolved in `prior_turns`. That's a real, measured gap from the
real distribution above, not a hypothetical one -- correct for it here:

**At least a third of your scenarios in this cell should have a lookup tool
itself as the correct `expected_tool_calls` answer** -- i.e. the scenario
snapshot represents an EARLIER point in the interaction, before identity or
order/product details have been established, and the correct single next
action is `find_user_id_by_name_zip`, `find_user_id_by_email`,
`get_user_details`, `get_order_details`, `get_product_details`,
`get_item_details`, `list_all_product_types`, or `calculate` -- not the
downstream mutation. (If this cell's theme is `identity_and_order_lookup`,
skew even higher than a third -- that theme exists specifically to cover this
stage.) For your other scenarios, where the correct answer legitimately is the
mutating/generic action, write `prior_turns` so it's clear the necessary
lookup already happened earlier in the conversation (stated as fixed
assistant/user text, e.g. "I found your order -- it's currently delivered and
contains an Action Camera (4K, black) and a Water Bottle."), not silently
assumed.

## Tools available (full source, verbatim -- the exact interface the agent sees)

```python
{tools_source}
```

## Agent policy (governs both scenario correctness and what counts as a policy_violation)

{policy_text}

## Database schema

```python
{data_model_source}
```

(50 products, 500 users, 1000 orders, 591 variant items in the real db.json --
explore it yourself as shown above; don't assume specific ids or contents
beyond what you actually look up.)

## Your assignment

Generate {n_scenarios} STATIC multi-turn scenarios in category "{category}" with
theme "{theme}".

**Category "{category}" means:** {CATEGORIES[category]}

**Theme "{theme}" means:** {THEMES[theme]}

Each scenario is a snapshot, not a live dialogue -- you are writing:
- 1-3 turns of prior conversation history (fixed text, written by you)
- the current user message
- the correct next tool call(s) given everything in that history, with exact
  arguments -- verified by actually executing them as shown above
- which OTHER plausible-but-wrong tool from the schema should also be available
  in this scenario's context as a distractor, and why a careful agent would
  reject it. Pick a distractor a model could genuinely be tempted by (e.g. a
  scenario needing cancel_pending_order should also plausibly have
  modify_pending_order_items "in scope," since both are pending-order actions)
  -- not a random unrelated tool.

Vary how much the correct action depends on info from earlier turns vs. just the
current message. For tool calls with optional/variable slots, don't always fill
every possible field the same way across your scenarios -- vary which ones are
present, the way a real user would under- or over-specify a request. (Most
retail tool parameters are required, not optional -- where a tool genuinely has
no optional slots, vary instead how much the user pre-supplies inline vs. leaves
for the agent to look up itself.)

Do not reproduce or lightly paraphrase any real tau2-bench benchmark task you
might recall from pretraining -- write genuinely novel scenarios grounded only
in the schemas, policy, and live database above. You have not been shown and
must not try to recall or reconstruct any specific real eval task from this
benchmark.

## Scenarios already generated for OTHER cells (avoid overlapping these)

{seen_block}

## Output

Write your output as a JSON array to exactly this file path (create it and any
missing parent directories):
`{out_path}`

Each element:
```json
{{
  "id": "{category}__{theme}__001",
  "category": "{category}",
  "theme": "{theme}",
  "prior_turns": [{{"role": "user|assistant", "content": "..."}}],
  "user_message": "...",
  "expected_tool_calls": [{{"name": "...", "arguments": {{"...": "..."}}}}],
  "expected_tool_calls_verified": true,
  "distractor_tool": "...",
  "distractor_rationale": "...",
  "ambiguity_note": "..."
}}
```

`expected_tool_calls` must NEVER contain more than one entry, in any category
-- see "Match the real benchmark's tool-use distribution" above for why. Rules
by category:
- **ambiguous**: must be `[]`; `ambiguity_note` must explain exactly what's
  underspecified and why guessing would be wrong.
- **policy_violation**: must be `[]`, UNLESS the correct move THIS turn is a
  single READ call needed to establish the state that makes the refusal
  correct (e.g. checking order status to discover it's not pending) -- if so,
  include just that one READ call and explain the refusal in a
  prior_turns/user_message-adjacent way that makes the intended eventual
  behavior clear.
- **out_of_scope**: must be exactly one call to `transfer_to_human_agents` with
  a real, specific `summary` argument.
- **happy_path / requires_earlier_context**: exactly one real, verified call --
  either a lookup call (if this scenario represents an earlier stage of the
  interaction) or the mutating/generic action (if `prior_turns` already
  establishes that the necessary lookups happened). See the tool-use
  distribution section above for the target mix.

Set `expected_tool_calls_verified: true` only if you actually executed that
call via `env.execute(...)` and confirmed the result matched your intent --
false if you couldn't (e.g. a `[]`-expected-tool-calls scenario has nothing to
execute, so mark `false` for those and rely on real ids instead; be honest
here, this field is read as a trust signal downstream, not a formality).

Also write a second file -- one-line semantic summaries for the dedup registry
(short enough for another generator to recognize "this is basically the same
setup as X" without reading the full scenario), one JSON object per line, to:
`{registry_path}`
```json
{{"id": "...", "one_line": "10-15 word summary of the setup+ask, not the answer"}}
```

When done, report back in your final message: how many scenarios you produced,
how many required you to revise your first proposed tool call after seeing a
real execution failure (a signal of how much the interactive-authoring step is
actually catching, not a score to optimize), and anything about the
category/theme definitions that was unclear or that the real data pushed back
on.
"""
