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

# Build a multi-step expected_tool_calls the same way: keep ONE RetailEnv alive
# for that one scenario, execute step 1 for real, inspect the resulting state,
# then decide step 2 against what actually resulted -- not what you assumed
# would result.
```

For a scenario with an empty expected_tool_calls (ambiguous / policy_violation /
out_of_scope), there's nothing to execute for the final answer, but any order_id
/ user_id / product_id / item_id / payment_method_id you reference anywhere
(prior_turns, user_message, distractor rationale) must still be a real one you
looked up in `env.db` -- never invent one.

Every id appearing anywhere in your output must be real. Fabricated ids make a
scenario unusable: Phase 3's automated rule checker re-executes every
expected_tool_calls against this exact same database and discards anything that
doesn't check out.

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

**Hard cap: `expected_tool_calls` has AT MOST ONE element, always, in every
category.** This isn't a generation convenience, it's a fact about the domain:
policy.md itself says the real agent may emit at most one tool call per turn,
never a tool call and a reply in the same turn. A "scenario" here is one
decision point -- the single next tool call correct given everything before
it -- not a multi-step plan. If a real task genuinely needs two calls (e.g.
look up a product's variant, then exchange using the id you find), do NOT put
both in expected_tool_calls. Instead, either:
- write the scenario so the correct next call IS the lookup (the exchange
  happens on a *later*, separately-written scenario/turn you don't need to
  produce here), or
- put the first call and its real result into `prior_turns` (as an
  assistant tool-call/tool-result exchange, phrased in prose since prior_turns
  is plain dialogue text) so the current turn's correct action is only the
  second, final call.
Pick whichever framing produces a more natural scenario; either is fine, but
never emit a 2+-element expected_tool_calls list -- the rule checker hard-fails
on it.

**Lookup-only scenarios: aim for roughly 1 in 3 of your happy_path /
requires_earlier_context scenarios (those with non-empty expected_tool_calls)
to have a READ tool as the entire correct answer** (`find_user_id_by_name_zip`,
`find_user_id_by_email`, `get_order_details`, `get_product_details`,
`get_item_details`, `get_user_details`, `list_all_product_types`) -- i.e. the
correct move this turn is to look something up, full stop, not to also act on
it. This matches how real conversations actually unfold turn-by-turn (per the
cap above, a lookup-then-act need spans two turns/scenarios, not one), and
without deliberately aiming for it generators default to writing the mutating
action as the "real" answer almost every time, which the pilot found produces
a badly skewed ~1-in-20 lookup share. Don't pad this with trivial lookups --
each one should be a case where looking something up is genuinely the correct
and complete next action (e.g. the user references an order by description
only, or asks a question that IS a lookup, like "what's the status of my
order" or "is the blue one in stock" with nothing further to act on this
turn).

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

Rules for `expected_tool_calls` by category (remember the global one-call cap
above -- none of these ever exceed length 1):
- **ambiguous**: must be `[]`; `ambiguity_note` must explain exactly what's
  underspecified and why guessing would be wrong.
- **policy_violation**: must be `[]`, UNLESS the correct first move is a READ
  tool needed to establish the state that makes the refusal correct (e.g.
  checking order status to discover it's not pending) -- if so, include just
  that one READ call and explain the refusal in a prior_turns/user_message-adjacent
  way that makes the intended final behavior clear.
- **out_of_scope**: must be exactly one call to `transfer_to_human_agents` with
  a real, specific `summary` argument.
- **happy_path / requires_earlier_context**: exactly one real, verified call --
  see the lookup-only guidance above for how to split a naturally two-step need.

Set `expected_tool_calls_verified: true` only if you actually executed every
call in the list via `env.execute(...)` (or a chained sequence of them) and
confirmed the result matched your intent -- false if you couldn't (e.g. a
`[]`-expected-tool-calls scenario has nothing to execute, so mark `false` for
those and rely on real ids instead; be honest here, this field is read as a
trust signal downstream, not a formality).

Also write a second file -- one-line semantic summaries for the dedup registry
(short enough for another generator to recognize "this is basically the same
setup as X" without reading the full scenario), one JSON object per line, to:
`{registry_path}`
```json
{{"id": "...", "one_line": "10-15 word summary of the setup+ask, not the answer"}}
```

Before reporting done, run the rule checker against everything under
`data/synthetic/raw/` (it checks all cells present, not just yours, but that's
fine -- ignore failures in files other than `{out_path.name}`):
`uv run python3 -m tau_forge.validate.rule_checker`
Fix anything it flags in your own cell's file (it re-executes every
expected_tool_calls against the real db.json, so a failure means a real bug --
a fabricated id, a stale-state assumption, an extra hallucinated argument, or a
cap/shape violation) before finishing. It also prints your cell's lookup-only
share -- if it's flagged as far from ~1/3, revise a few scenarios rather than
ignoring it.

When done, report back in your final message: how many scenarios you produced,
how many required you to revise your first proposed tool call after seeing a
real execution failure (a signal of how much the interactive-authoring step is
actually catching, not a score to optimize), and anything about the
category/theme definitions that was unclear or that the real data pushed back
on.
"""
