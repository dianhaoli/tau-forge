# tau-forge

RLVR fine-tuning of `Qwen/Qwen3-4B-Instruct-2507` with GRPO on synthetic multi-step
tool-calling scenarios grounded in τ²-bench's real retail domain, evaluated against
the real τ²-bench retail benchmark (plus a zero-shot airline check and BFCL v3
multi-turn).

Companion design doc: `qwen3-4b-rl-toolcalling-plan.md` (referenced by the original
task spec; not yet present in this repo — add it here if/when available for
continuity across sessions).

## Held-out data policy — all 114 real retail tasks, always

**All 114 real τ²-bench retail tasks — the full `train` (74) + `test` (40) split,
not just `test` — are permanently off-limits to anything that updates real model
weights.** This is a deliberate project-level decision, and it overrides τ²-bench's
own upstream convention (where `train` is meant to be trained on and only `test` is
held out) -- final evaluation here runs against **all 114**, so training on the
`train` half would contaminate that evaluation.

Concretely:
- No GRPO training run, and no *live* (real-model) GRPO smoke test, may use any of
  the 114 real retail tasks as training data. Live smoke tests use the Phase 2
  synthetic pilot data (`data/synthetic/raw/*.json`) instead.
- Phase 6's smoke test (below) is the one sanctioned exception, and only because it
  never touches model weights at all -- it runs two scripted stand-in policies
  (`gold`, `noisy`) through the scoring code on CPU to validate the harness itself,
  not a trained model. No neural network has ever seen any of the 114 tasks in this
  repo.
- If a live smoke test is ever run against real data *for validation purposes only*
  (not recommended -- the synthetic pilot data validates the same code path without
  this risk), the resulting checkpoint must be discarded and Phase 7's real run
  started from a fresh, untouched copy of the base model. Given the synthetic data
  works just as well for that check with none of this discipline required, prefer
  it and skip this case entirely.

## Phase status

| Phase | Description | Status |
|---|---|---|
| 0 | Repo setup + data substrate extraction | Done |
| 1 | Environment wrapper / mock tool executor | Done |
| 2 | Synthetic scenario generation (methodology + first cells) | **Pilot done (3/30 cells) — paused for reordering, see below** |
| 3 | Validation pipeline (rule / model / human / difficulty) | Not started |
| 4 | Reward function + adversarial tests | **Done — 6/6 adversarial cases pass, see below** |
| 5 | Decontamination vs. real 114 τ²-bench tasks | Not started |
| 6 | Harness smoke test on real 74 train tasks (scoring code only, no model) | **Passed — gold policy 1.0000/1.0000 (mean/min), see below** |
| 7 | Real GRPO training run (synthetic data only — see held-out policy above) | Not started — needs a GPU box, none available in this environment |
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
Phase 4 was done next, out of file order, to get back on the recommended track.
Phase 6 (this session) is now done and passing — see below, which removes the
technical blocker on resuming Phase 2. Per the STOP-checkpoint policy above,
resuming the full 30-cell sweep still needs explicit go-ahead, not just a green
smoke test — and note there is a concurrent session already working the Phase 2
cells; coordinate before starting new ones to avoid duplicate work.

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

## Data prep + harness smoke test (`tau_forge/data_prep/`, `tau_forge/harness/`) — Phase 6

**Data prep.** `tau_forge/data_prep/trusted_tasks.py` converts the real τ²-bench
retail **train** split (74 tasks) into `TrustedTask`s: a rendered user-scenario
prompt plus the gold assistant action sequence, pulled straight from tau2's own
`Task` objects (`tau2.domains.retail.environment.get_tasks("train")`) rather than
reimplementing any parsing. This is the first point in the project where real task
*content* is read -- deliberately: the Phase 0/5 decontamination rule is about
never letting task content leak into the synthetic-data *generator*'s prompts,
which grounding RLVR training directly on the real train split doesn't touch. The
real **test** split's content is still never read anywhere in this repo. All 114
retail tasks ship with `initial_state=None` (confirmed in Phase 0) -- every task
starts from the same shipped `db.json`, so there is no per-task DB snapshot to
extract; the actual conversion work is the prompt/action-sequence extraction.
Output: `data/trusted/train_tasks.json` (74 entries, committed).

**Harness.** `tau_forge/harness/rollout.py` adds multi-turn scoring on top of the
Phase 4 `reward()` function -- no new grading logic, just sequencing it.
Teacher-forced: turn `i`'s reward compares the rollout's turn-`i` action against
gold's turn-`i` action, both executed from the DB state produced by replaying
**gold's** actions `0..i-1` (not the rollout's own prior actions, which could have
already diverged). This is a deliberate departure from tau2's own official retail
grading (`RewardType.DB`: only the *final* DB state after a full conversation has
to match, any path there is fine) -- that coarse, episode-end signal is what
Phase 8's live-benchmark eval uses; training needs denser per-step credit
assignment, which requires a well-defined "correct" context for every step
regardless of where the rollout diverged. A rollout shorter than gold has its
missing turns scored via `reward()`'s existing `missing_call` case; a rollout
longer than gold has its extra turns scored 0 (`extra_unrequested_turn`).

**Two real findings, surfaced by running this harness against the real trusted
tasks, not synthetic data:**

- Some gold actions are themselves expected to fail. Task `"2"`'s gold trajectory
  calls `get_product_details` on a product id that genuinely doesn't exist in
  `db.json` -- the resulting error *is* the correct outcome (the user apparently
  mentioned a product that isn't real), not a broken label. `reward()` originally
  raised `ValueError` on this. Fixed: when the gold action itself fails, grade the
  rollout by whether it reproduces the same failure (`error_match` for an exact
  reproduction, `error_partial_match` via text similarity otherwise,
  `expected_failure_but_succeeded` -- scored like the schema-invalid tier -- if the
  rollout's call succeeds when it was supposed to fail). Also affects tasks `"3"`,
  `"4"`, `"35"`, `"37"`, `"46"`, `"47"`, `"54"`, `"67"`, `"105"`.
- Two train tasks (`"24"`, `"57"`) have **zero** gold actions -- the correct
  behavior is purely conversational (no tool call needed at all, akin to the
  Phase 2 pilot's `ambiguous`/`policy_violation` scenarios). `score_rollout`
  originally scored an empty rollout against these as `0.0` (an empty turn-score
  list's mean defaults to 0, not 1) instead of the `1.0` a correct empty match
  deserves. Fixed as an explicit `correct_no_call` case.

**Smoke test** (`tau_forge/harness/smoke_test.py`, `uv run python -m
tau_forge.harness.smoke_test`): runs two scripted stand-in policies over all 74
trusted tasks --

- `gold` -- replays each task's gold action sequence verbatim against itself.
- `noisy` -- gold actions with randomly injected corruption (wrong tool, corrupted
  argument, or a dropped call; ~50% of turns affected).

| Policy | mean-of-task-means | min task mean |
|---|---|---|
| `gold` | **1.0000** | **1.0000** |
| `noisy` | 0.5975 | -- |

**PASSED.** Every one of the 74 trusted train tasks scores a perfect 1.0 when
graded against itself, confirming env (Phase 1) + reward (Phase 4) + data prep
(this phase) agree with each other on real, trusted ground truth. The noisy
policy scoring well below gold confirms the reward function actually
discriminates good from bad rollouts rather than degenerately returning a
constant. Full per-task breakdown in `data/trusted/phase6_smoke_report.json`
(committed).

**Scope note -- no GRPO steps run here.** This environment has no GPU and no ML
training stack (`torch`/`trl`/`vllm` all absent -- confirmed, not assumed). This
smoke test validates the *plumbing* a GRPO trainer sits on top of using scripted
policies as a stand-in for a live model, not a trained one -- which is exactly why
it's the sanctioned exception to the held-out-data policy above (no model weights
are ever produced or updated here). It does not run the "50-100 step GRPO smoke
test" the Phase 6 spec also calls for -- that needs a GPU box (Phase 7) and, per
the held-out policy, must run against the **Phase 2 synthetic pilot data**, not
these 74 real tasks (draft EC2 setup notes are being worked out with the user
separately).

STOP for review, per the Phase 6 spec: harness validated against real trusted
data before resuming Phase 2/3 or attempting any GPU training.
