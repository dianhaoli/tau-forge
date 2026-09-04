# tau-forge

RLVR fine-tuning of `Qwen/Qwen3-4B-Instruct-2507` with GRPO on synthetic multi-step
tool-calling scenarios grounded in τ²-bench's real retail domain, evaluated against
the real τ²-bench retail benchmark (plus a zero-shot airline check and BFCL v3
multi-turn).

Companion design doc: `qwen3-4b-rl-toolcalling-plan.md` (referenced by the original
task spec; not yet present in this repo — add it here if/when available for
continuity across sessions).

## Phase status

| Phase | Description | Status |
|---|---|---|
| 0 | Repo setup + data substrate extraction | Done |
| 1 | Environment wrapper / mock tool executor | Done |
| 2 | Synthetic scenario generation (methodology + first cells) | **Pilot done (3/30 cells) — paused for reordering, see below** |
| 3 | Validation pipeline (rule / model / human / difficulty) | **Rule checker (stage 1) done, see below; model/human/difficulty stages not started** |
| 4 | Reward function + adversarial tests | **Done — 6/6 adversarial cases pass, see below** |
| 5 | Decontamination vs. real 114 τ²-bench tasks | Not started |
| 6 | Harness smoke test on real 74 train tasks | Not started |
| 7 | Real GRPO training run | Not started |
| 8 | Evaluation (τ²-bench retail, airline zero-shot, BFCL v3) | Not started |

Each phase after the current one is gated on a STOP checkpoint for review — see the
originating task spec. Do not advance a phase past its STOP without explicit
go-ahead.

**Sequencing note:** the task spec's own recommended execution order is
`0 → 1 → 4 → (Phase 6 data prep) → 6 → 2 → 3 → 5 → 7 → 8`, specifically so a
broken harness/reward function surfaces against the 74 trusted real train tasks
before a full 400-500-scenario synthetic sweep is built on top of it. This repo's
history went `0 → 1 → 2 (pilot)` instead, skipping straight to the Phase 2 pilot.
That pilot (3/30 cells, 57 scenarios) is not wasted — it's a valid dry run of the
generation methodology — but the full 30-cell sweep is being held pending Phase 6.
Phase 4 (this session) was done next, out of file order, to get back on the
recommended track. **Do not run the remaining Phase 2 cells until Phase 6's
harness smoke test has passed.**

## Data substrate

`third_party/tau2-bench` is a **git submodule** of
[`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench),
pinned at `a2c0247` (`v1.0.1-31-ga2c0247`, 2026-08-18). It is vendored rather than
copied so later phases can `import` `tau2.*` directly and so `tau2 run` is available
verbatim for Phase 6/9 evaluation. Set up with `uv sync --extra gym` inside that
directory (do not commit changes to `uv.lock` there — it's upstream's file, we only
track the pinned commit).

Key retail-domain files (read verbatim, unmodified, in Phase 0):

- `src/tau2/domains/retail/tools.py` — `RetailTools(ToolKitBase)`, 16 tools:
  - **READ (7)**: `find_user_id_by_name_zip`, `find_user_id_by_email`,
    `get_order_details`, `get_product_details`, `get_item_details`,
    `get_user_details`, `list_all_product_types`
  - **WRITE (7)**: `cancel_pending_order`, `exchange_delivered_order_items`,
    `modify_pending_order_address`, `modify_pending_order_items`,
    `modify_pending_order_payment`, `modify_user_address`,
    `return_delivered_order_items`
  - **GENERIC (2)**: `calculate`, `transfer_to_human_agents`
  - (A `think` tool exists in source but is commented out / inactive.)
- `data/tau2/domains/retail/policy.md` — business rules. Highlights: agent must
  authenticate the user (by email, or name+zip) before doing anything, even if a
  user id is offered; one user per conversation; explicit yes/no confirmation
  required before any DB-mutating action; no fabricated info/policy; at most one
  tool call per turn, never a tool call and a reply in the same turn; transfer to
  human only when truly out of scope for the tool set; exchange/modify-items tools
  are **single-use per order**.
- `data/tau2/domains/retail/db.json` (2.8 MB) + `src/tau2/domains/retail/data_model.py`
  — mock DB (`RetailDB`: `products`, `users`, `orders`, all pydantic models).
  Live stats: **50 products / 500 users / 1000 orders / 591 variant items**.
- `data/tau2/domains/retail/split_tasks.json` — confirmed
  **train: 74, test: 40, base: 114**, train/test disjoint, union == base. Matches
  expectation exactly; repo has not changed in a way that affects this split.
- `data/tau2/domains/airline/split_tasks.json` — train: 30, **test: 20**, base: 50
  (airline test split needed zero-shot in Phase 8; no airline data used in training).

### `tasks.json` handling (decontamination discipline)

Per the task spec, the 114 real retail task instances in
`data/tau2/domains/retail/tasks.json` must never be seen by the synthetic-data
generator. In Phase 0 this file was touched **only** to confirm its length (114,
matching `split_tasks.json`'s `base`) and its top-level dict keys (`description`,
`evaluation_criteria`, `id`, `initial_state`, `user_scenario` — field names only,
no values). No task content (description text, scenarios, evaluation criteria) has
been read or is present anywhere in this repo or in generation context. This file's
actual content is only to be loaded in Phase 5, as an isolated decontamination
check, never fed into scenario generation prompts.

## Environment wrapper (`tau_forge/envs/retail.py`)

Our own lightweight executor — not `tau2.gym`, not the `tau2 run` orchestrator
(those are reserved for Phase 9's live-harness evaluation only). `RetailEnv` wraps
one `RetailDB` snapshot and calls tau2's own `RetailTools` methods directly, so
results and DB side effects are exactly what the real benchmark would produce.

- `RetailEnv(db=None)` — live, stateful; `execute(tool_name, arguments)` mutates
  `self.db` in place. Used for interactive gold-answer authoring (Phase 2) and DB
  inspection.
- `execute_against(db, tool_name, arguments)` — stateless; runs one call against a
  deep copy of `db` and returns `(ToolResult, resulting_db)` without touching the
  input. This is the primitive the Phase 4 reward function uses to compare a
  rollout's and the gold action's end states from the same starting snapshot.
- Tool metadata (`tool_names`, `tool_type`, `openai_schema`, `args_model`) is read
  straight from tau2's own `Tool` objects — not reimplemented — so schemas shown to
  the policy model match the real benchmark exactly.
- `tool_mutates_state(name)` exposes tau2's own READ/WRITE/GENERIC-derived flag
  (e.g. `transfer_to_human_agents` is GENERIC but does *not* mutate state). Phase
  4's reward function should use this to pick `state_match_score` (DB side effect
  exists) vs. `arg_match_score` (no DB state to diff — read tools, `transfer_to_human_agents`-style actions) — this is the fallback rule the plan calls for, not a
  guess.
- **Confirmed empirically** (see `tests/test_retail_env.py`): tau2's auto-generated
  per-tool pydantic parameter model uses pydantic's default `extra` policy
  (`ignore`), so `validate_arguments()` alone does **not** catch a hallucinated
  extra argument the tool doesn't accept — it silently passes. Use
  `extra_arguments(name, arguments)` for that; Phase 4's hallucination penalty
  needs it, not `validate_arguments()` alone.
- 18 tests in `tests/test_retail_env.py`, run via `uv run pytest`, covering:
  execution correctness for a READ and two WRITE tools, clean (non-exception)
  error handling for bad ids/state/reasons, snapshot/mutation isolation,
  `execute_against` determinism and non-mutation of its input, and the schema
  validation edge cases above. All passing against the real shipped `db.json`.

## Synthetic generation (`tau_forge/gen/`) — Phase 2 pilot

`taxonomy.py` defines 5 categories x 6 domain-grounded themes = 30 cells (see its
docstring for why "subscription/recurring-order edge cases" from the original task
spec was replaced — no subscription concept exists anywhere in this domain).
`prompt_template.py` renders the full per-cell generation brief programmatically
(real tool source/policy/schema read fresh off disk, plus the running dedup
registry) — reused for all 30 cells, not copy-pasted per cell.

Piloted 3 cells (one subagent each, run in parallel) before committing to the full
sweep: `happy_path`×electronics (20 scenarios), `ambiguous`×apparel (17),
`policy_violation`×order_state_confusion (20) — 57 total, in
`data/synthetic/raw/*.json`. Each subagent authored gold answers interactively
against the Phase 1 `RetailEnv` (executing real calls, not just asserting
correctness) and ran its own post-hoc verification pass against the written JSON.
Per-cell one-line summaries are merged into `data/synthetic/registry.jsonl` for
the next wave's dedup context.

**Finding surfaced during the pilot, not before:** `modify_pending_order_address`
and `modify_pending_order_payment` gate on `RetailTools._is_pending_order`, a
*substring* check (`"pending" in order.status`), while `cancel_pending_order` and
`modify_pending_order_items` gate on exact equality (`order.status == "pending"`).
On an order in `"pending (item modified)"`, the substring check still matches --
so `modify_pending_order_address` will **not** raise, even though policy.md is
explicit that no further modification is allowed once items have been modified.
`modify_pending_order_payment` happens to still be blocked in that state by an
independent invariant (payment history must have exactly one entry), but
`modify_pending_order_address` has no such backstop. Verified directly against
`third_party/tau2-bench/src/tau2/domains/retail/tools.py` (not just taken on the
pilot subagent's word) and encoded as `policy_violation__order_state_confusion__011`.
Implication for Phase 3/4: "the tool call didn't raise" is not sufficient evidence
of policy compliance for this action -- the rule checker and reward function both
need to consult policy.md's written rule for this specific state transition, not
just tool-level exceptions.

## Reward function (`tau_forge/reward/reward.py`) — Phase 4

Grades a rollout `Action(tool_name, tool_input)` against a scenario's gold `Action`
by comparing **outcomes** through the Phase 1 `RetailEnv`/`execute_against`, not
literal arguments -- matching how tau2-bench itself grades (DB-end-state
equivalence, not trajectory match). Score tiers:

- `0.0` -- wrong tool, or a call made/withheld when the opposite was correct
  (message-only is itself the correct action for `ambiguous`/`policy_violation`
  scenarios with no `expected_tool_calls`, so calling a tool there scores 0, not
  just "unhelpful").
- `0.2` -- schema-invalid args, a hallucinated argument outside the tool's schema
  (checked via `RetailEnv.extra_arguments`, since pydantic's auto-derived param
  model on tau2's tools uses `extra="ignore"`, not `"forbid"` -- confirmed in
  Phase 1's own tests, not assumed here), or the call raising at execution despite
  valid schema (bad id / wrong state).
- `0.3 - 1.0` -- right tool, schema-valid, executes cleanly: graded on how close
  the outcome is to gold. For a **mutating** tool, diffs the resulting DB record
  (touching a different order/user than gold scores at the floor; a mismatch on a
  *critical* field -- status, items, payment history, exchange/return ids -- also
  floors; a mismatch confined to a *graded* field -- `cancel_reason`,
  `exchange_price_difference` -- gets real partial credit). For a **read-only or
  generic** tool with no DB side effect, falls back to `arg_match_score` (exact for
  IDs/enums/short strings, numeric tolerance, similarity-with-a-floor for free
  text) -- except tools in `OUTPUT_DETERMINES_CORRECTNESS` (the READ tools, plus
  `calculate`), where matching the *returned value* is checked first, so two
  different-but-equivalent inputs both score 1.0.

**Finding surfaced while building this, not before:** `transfer_to_human_agents`
always returns the literal string `"Transfer successful"` regardless of `summary`
-- so an output-equality shortcut applied uniformly to all non-mutating tools would
let a padded, content-free `summary` score a perfect 1.0 purely because the return
value matched. `OUTPUT_DETERMINES_CORRECTNESS` deliberately excludes it (and any
other tool whose return value doesn't actually vary with its arguments) so it's
always graded via `arg_match_score` on the argument content itself.

A second, separate finding from writing the adversarial tests: `modify_pending_order_items`
has a latent bug in tau2's own implementation (not ours) where `item.price` /
`item.options` get set from the **last** variant processed in an earlier loop, not
each item's own new variant -- so reordering the `item_ids`/`new_item_ids` pairs in
that tool's call *does* change the resulting DB state, unlike a well-behaved
"order shouldn't matter" tool. Verified empirically (see the comment in
`tests/test_reward.py`) before picking `exchange_delivered_order_items` instead for
the equivalent-args test, since that tool sorts `exchange_items`/`exchange_new_items`
independently and never mutates `order.items` directly.

### Adversarial test results (`tests/test_reward.py`, `uv run pytest`)

All 6 cases the Phase 4 spec calls out, plus baseline sanity checks (exact match →
1.0 for both a mutating and a read-only tool; correctly withholding a call → 1.0;
calling one when none was expected → 0.0) -- 12 tests total, all passing:

| Case | Score | Reason |
|---|---|---|
| Padded, generic free-text `summary` (right tool/record) | **0.300** | `arg_match` (similarity floored to 0) |
| Subtly wrong item variant (schema-valid, off-by-one choice) | **0.300** | `state_match_partial` / critical field mismatch |
| Right tool/args + one hallucinated extra field | **0.200** | `schema_invalid_or_hallucinated_args` |
| Equivalent reordered args, identical resulting DB state | **1.000** | `state_match_exact` |
| Equivalent `calculate` expression (`"4 - 0"` vs `"2 + 2"`) | **1.000** | `output_match` |
| Near miss: same order, only `cancel_reason` differs | **0.948** | `state_match_partial` (graded field only) |
| Wrong order touched entirely (same tool, same call shape) | **0.300** | `state_match_partial` / `wrong_record` |

The wrong-record case (0.300) scores well below the descriptive-field near miss
(0.948) despite superficially looking similar (same tool, same argument shape) --
the two are not just numerically different, `state_match_score` tags the
wrong-record case explicitly (`detail.reason == "wrong_record"`) so this
distinction is inspectable, not just a coincidence of the weights chosen.

STOP for review, per the Phase 4 spec: reward function built and adversarially
tested before any training or further synthetic-data generation. Next per the
reordering above is Phase 6's data prep (converting the real 74 train tasks to
static snapshots) and smoke test, not the remaining 27 Phase 2 cells.

## Rule checker (`tau_forge/validate/rule_checker.py`) — Phase 3, stage 1

Ran an informal resemblance check between the pilot's 57 scenarios and the real
114 τ²-bench retail tasks (still isolated -- `tasks.json` was read for
comparison only, never fed into a generation prompt) and it surfaced a real,
structural finding, not just a contamination scare:

- **Tool-call count per scenario:** real tasks average 4.8 tool calls (median
  5, up to 13); the pilot averaged 0.49 (median 0). Real tasks make the agent
  do the *entire* chain -- authenticate, look up the order, look up the
  product, then act. The pilot's scenarios start mid-conversation with
  identity/order context already resolved in `prior_turns`, so they only ever
  exercise the *last* decision.
- **Tool-use distribution:** across 550 real reference-trajectory tool calls,
  **64.9% are lookup/identity calls** (`get_order_details` alone is 30.5% of
  every call made) and only 35.1% are the mutating/generic "final" action. The
  pilot's `expected_tool_calls` were almost entirely the mutating/generic end
  -- essentially zero lookup calls as the actual answer to a scenario.
- **One scenario, `happy_path__electronics_returns_exchanges__010`, chained two
  calls** (`get_product_details` then `exchange_delivered_order_items`) into
  one `expected_tool_calls`. That's not just untidy -- policy.md is explicit
  that the agent may make **at most one tool call per turn** ("You should at
  most make one tool call at a time..."), so a chained answer describes two
  agent turns as if they were one, misrepresenting what a policy-compliant
  agent would ever emit in a single turn. It's also not gradable by the Phase
  4 reward function, which only scores a single `Action`. Fixed by trimming
  the gold answer to just the `get_product_details` lookup (the correct action
  *this* turn) and updating the distractor rationale to explain why the
  exchange itself belongs to a later turn.

Built `tau_forge/validate/rule_checker.py` (Phase 3's stage-1 rule checker) to
catch this class of issue mechanically going forward: it re-executes every
scenario's `expected_tool_calls` against a fresh copy of the real `db.json`
and flags (a) more than one call in `expected_tool_calls` -- now a hard
failure, not just a style note, (b) category-shape violations (`ambiguous`
non-empty, `policy_violation`'s single call not a READ, `out_of_scope` not
exactly one `transfer_to_human_agents`), (c) unknown tools, schema-invalid or
hallucinated arguments, and execution failures, and (d) an
`expected_tool_calls_verified` flag inconsistent with whether there's
actually anything to verify. Run via `uv run python3 -m
tau_forge.validate.rule_checker`. Current result: **57/57 pilot scenarios
pass** (was 56/57 before the fix above).

**Fixed `tau_forge/gen/prompt_template.py`** so the remaining 27 cells don't
reproduce either finding:
- `expected_tool_calls` is now hard-capped at one entry in every category, with
  the policy.md quote inline as the reason, and explicit guidance that
  multi-step *exploration* against `RetailEnv` during authoring is fine and
  encouraged, but only one call goes in the output.
- Added a "match the real benchmark's tool-use distribution" section stating
  the 64.9%/35.1% split above and instructing each cell to make **at least a
  third of its scenarios' correct answer a lookup call itself** (representing
  an earlier stage of the conversation, before identity/order/product details
  are resolved), skewed even higher for the `identity_and_order_lookup` theme
  -- rather than always assuming that stage already happened in `prior_turns`.

This doesn't close the deeper live-multi-turn generalization gap discussed
separately (still tracked as this project's headline Phase 8 question), but it
does make the static-snapshot dataset's *aggregate* tool-usage shape resemble
the real benchmark's, which it measurably did not before this fix.
