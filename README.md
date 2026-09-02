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
| 1 | Environment wrapper / mock tool executor | **Done — awaiting review** |
| 2 | Synthetic scenario generation (methodology + first cells) | Not started |
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
