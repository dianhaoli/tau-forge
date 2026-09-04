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
| 2 | Synthetic scenario generation (methodology + first cells) | **Pilot done (3/30 cells) — awaiting review** |
| 3 | Validation pipeline (rule / model / human / difficulty) | Not started |
| 4 | Reward function + adversarial tests | Not started |
| 5 | Decontamination vs. real 114 τ²-bench tasks | Not started |
| 6 | Harness smoke test on real 74 train tasks | Not started |
| 7 | Real GRPO training run | Not started |
| 8 | Evaluation (τ²-bench retail, airline zero-shot, BFCL v3) | Not started |

Each phase after the current one is gated on a STOP checkpoint for review — see the
originating task spec. Do not advance a phase past its STOP without explicit
go-ahead.

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

### Rule checker -- Phase 3, stage 1 (`tau_forge/validate/rule_checker.py`)

Deterministic, no-LLM, mechanical checks -- run after every cell (or small
batch), not saved for the end: `uv run python3 -m tau_forge.validate.rule_checker`.
This is stage 1 only (mechanical correctness); stages 2-4 (model checker,
human review, difficulty calibration) are not built yet.

What it checks, per scenario:
- required JSON keys present; `id`/`category`/`theme` consistent with the
  cell's filename; no duplicate ids within a cell.
- **global hard cap: `expected_tool_calls` has at most 1 element**, always.
  This is a fact about the domain, not a generation convenience -- policy.md
  itself says the real agent emits at most one tool call per turn, so a
  scenario (one decision point) can never legitimately need a 2-call chain in
  a single `expected_tool_calls`. A genuinely two-step need must be split:
  either the correct answer *is* the lookup (the mutating call is a separate,
  later scenario/turn), or the first call+result is folded into `prior_turns`
  as already-resolved dialogue, leaving only the final call as this turn's
  answer.
- category-specific shape: `ambiguous` → `[]` + non-empty `ambiguity_note`;
  `policy_violation` → `[]`, or exactly one call that must be a READ tool;
  `out_of_scope` → exactly one `transfer_to_human_agents` call with a
  non-empty `summary`; `happy_path`/`requires_earlier_context` → exactly one
  call.
- **re-execution**: builds a fresh `RetailEnv` (real shipped `db.json`) and
  actually runs every `expected_tool_calls` entry against it, in order --
  catches fabricated ids, stale-state assumptions, and extra/hallucinated
  arguments (via `extra_arguments()`, not just `validate_arguments()`, per
  the Phase 1 finding above) that interactive authoring might have missed.
  This is the check that matters most: a scenario that reads fine but doesn't
  actually re-execute is not usable gold data.
- distractor sanity (real tool name, not identical to the expected call).
- soft/informational: each cell's share of `happy_path`/
  `requires_earlier_context` scenarios whose single call is a READ tool
  ("lookup-only") is reported, flagged if it's far from the ~1/3 target below.
  This isn't a hard failure (doesn't affect PASS/FAIL) since it's a
  distributional property of a whole cell, not a per-scenario correctness
  fact.

**Two fixes to `prompt_template.py` this drove**, both required for every cell
generated after the pilot:
1. **The one-call hard cap** above -- the pilot's `happy_path` cell had one
   scenario with a 2-call `expected_tool_calls` (look up a product variant,
   then exchange using the id found); fixed by making the lookup itself the
   scenario's answer (`happy_path__electronics_returns_exchanges__010`).
2. **~1/3 lookup-only share target** -- without deliberately aiming for it,
   generators default to writing the mutating action as "the" answer almost
   always; the pilot's `happy_path` cell (pre-fix) was skewed to ~1-in-20
   lookup-only. The template now explicitly asks for roughly 1 in 3 of a
   cell's non-empty-`expected_tool_calls` scenarios to be a READ tool as the
   complete correct answer, matching how a lookup-then-act need actually
   spans two turns under the cap above rather than being crammed into one.

All 3 pilot cells (57 scenarios) pass the rule checker 57/57 clean after the
one fix above; the pilot's lookup-only share (reported, not failing) still
reflects the pre-fix generation and is not being retroactively rebalanced --
the ~1/3 target applies to cells generated with the fixed template.

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
