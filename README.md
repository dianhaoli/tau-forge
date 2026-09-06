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
| 2 | Synthetic scenario generation (methodology + full sweep) | **Done (30/30 cells, 541 scenarios)** |
| 3 | Validation pipeline (rule / model / human / difficulty) | **Stages 1, 2, 4 done (541/541 scenarios each); stage 3 (human review) sample generated + delivered — 1 scenario reviewed and a systemic 23-scenario authentication bug it surfaced fixed, rest pending — see below** |
| 4 | Reward function + adversarial tests | **Done — 6/6 adversarial cases pass, see below** |
| 5 | Decontamination vs. real 114 τ²-bench tasks | **Done — 0/8 flagged pairs confirmed as true positives on spot-check, see below** |
| 6 | Harness smoke test on real 74 train tasks (scoring code only, no model) | **Passed — gold policy 1.0000/1.0000 (mean/min), see below** |
| 7 | Real GRPO training run (synthetic data only — see held-out policy above) | Not yet run — needs a GPU box, none available in this environment. AWS EC2 setup fully prepped (`docs/phase7_aws_setup.md` / `infra/`: full-parameter GRPO not LoRA, `g6e.12xlarge` 4x L40S ZeRO-2, timing model, methodology risks) **and the actual training code is written** (`tau_forge/train/`: dataset/prompt building off tau2's own real system prompt + tool schemas, `<tool_call>` completion parsing with a caught reward-hacking bug fixed, a `reward()`-backed TRL `GRPOTrainer` reward adapter, the `grpo_train.py`/`zero_shot_baseline.py` entrypoints) — unit-tested against all 541 real scenarios without a GPU (`tests/test_train_pipeline.py`), but never executed end-to-end since no GPU/torch/trl was available while writing it; the smoke test is what actually validates it. Still awaits go-ahead for the full run |
| 8 | Evaluation (τ²-bench retail, airline zero-shot, BFCL v3) | Harness written (`tau_forge/eval/`: prompt-parity-checked `run_tau2` entrypoint, fixed-comparison run config) — **no run executed, no baseline number exists yet**. See "Phase 8 evaluation harness" below |

Each phase after the current one is gated on a STOP checkpoint for review — see the
originating task spec. Do not advance a phase past its STOP without explicit
go-ahead.

**Sequencing note:** the task spec's own recommended execution order is
`0 → 1 → 4 → (Phase 6 data prep) → 6 → 2 → 3 → 5 → 7 → 8`, specifically so a
broken harness/reward function surfaces against the 74 trusted real train tasks
before a full 400-500-scenario synthetic sweep is built on top of it. This repo's
actual history split into parallel lines of work that have since been merged
back together: one line went `0 → 1 → 2 (pilot) → 4 → 6` (reward function built
and the harness smoke-tested against real trusted data before touching more
synthetic data, per the recommended order), while a concurrent line did the
`2 (full 30-cell sweep) → 3 (stage-1 rule checker)` work in parallel rather than
waiting on Phase 6 to land first. Both lines are now reconciled into this
history: Phase 2 is done (30/30 cells), Phase 3 stages 1, 2, and 4 are done
(stage 3's sample is generated and delivered, verdicts pending), Phase 4 and 6
are both done and passing. Per the STOP-checkpoint policy above, Phase 5 and
Phase 7 still need explicit go-ahead before starting — none of this
reconciliation implies permission to advance further un-asked.

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
`policy_violation`×order_state_confusion (20) — 57 total. Each subagent authored
gold answers interactively against the Phase 1 `RetailEnv` (executing real calls,
not just asserting correctness) and ran its own post-hoc verification pass
against the written JSON. Per-cell one-line summaries are merged into
`data/synthetic/registry.jsonl` for the next wave's dedup context.

## Synthetic generation -- Phase 2 full sweep (all 30 cells)

Completed the remaining 27 cells in 5 waves (5-6 subagents in parallel per wave,
sequential across waves so each wave's `seen_so_far` dedup context reflects every
prior wave): wave 1 = the 5 remaining `happy_path` cells, wave 2 = all 6
`requires_earlier_context` cells, wave 3 = all 6 `ambiguous` cells, wave 4 = the
5 remaining `policy_violation` cells, wave 5 = all 6 `out_of_scope` cells. Same
methodology as the pilot throughout: each subagent got the full
`render_cell_prompt` brief (rendered fresh, so it reflected every prior wave's
registry additions), authored gold answers interactively against a live
`RetailEnv`, ran the rule checker against its own cell before reporting done, and
appended its one-line summaries to the registry for the next wave.

**Final tally: 30/30 cells, 541 scenarios, rule checker clean (541/541, 0 errors,
0 warnings)** -- `happy_path` 110, `requires_earlier_context` 108, `ambiguous`
104, `policy_violation` 112, `out_of_scope` 107; roughly even across the 6 themes
(88-92 each). All post-pilot `happy_path`/`requires_earlier_context` cells hit the
~1/3 lookup-only-share target (6/18 = 33% almost everywhere); only the pilot's
`happy_path`×electronics cell (generated before the target existed, and
intentionally not regenerated) sits off it at 2/20 = 10%, exactly as expected per
the "Rule checker" section above.

**Tried and deferred: multi-turn out_of_scope scenarios (README stays at
541).** Phase 7 zero-shot baselining on an L40S box surfaced that
`out_of_scope` scenarios are all single-step, zero-lookup escalations
(identity already resolved in `prior_turns`, the very next call is
`transfer_to_human_agents`) -- a narrower pattern than 3 of the 4 real
tau2-bench retail tasks that actually need this tool (tasks 10/12/26), where
the agent does a full normal lookup/action workflow first and escalates only
the one specific sub-request no tool can satisfy (task 50 is the only real
task matching our pure single-step pattern). To test whether GRPO training on
the narrower pattern risks teaching an "escalate immediately when it smells
hard" shortcut that would misgeneralize to that more common real-task shape,
3 scenarios were hand-authored in this richer style -- prior_turns narrate an
already-resolved, tool-executed partial request, and `expected_tool_calls`
covers only the next, genuinely-unsatisfiable ask (e.g. redirecting a refund
to a *different* payment method than the order's own, which
`return_delivered_order_items` hard-rejects). Grounded in real `db.json`
data and rule-checker clean, same discipline as every other scenario.

Zero-shot baselined twice at n=16 (temperature 1.0): both times, 0/48 across
all three, 100% zero-variance -- confirmed cold start, not just low
probability. The first pass conflated this with an authoring bug (a missing
item id forced hallucination on every wrong-tool attempt); after fixing that,
the result held, and the actual failure sharpened into something more
specific than "doesn't know to escalate": the model copies the
`payment_method_id` from the already-resolved action in `prior_turns` onto
the new request, checking "is this a real payment method on this account" as
its precondition instead of "is this *this order's own* original payment
method" -- a precondition-scoping error. Since zero variance means zero GRPO
signal regardless, and 3 examples is too few to include safely even if a
future higher-sample rerun (or one giving the model room to reason before the
tool call, untested here) surfaced occasional successes, these were pulled
from the active training set rather than left in `data/synthetic/raw/`. Kept
at `data/synthetic/deferred/out_of_scope_multiturn_coldstart.json` (outside
`DEFAULT_DATA_GLOB`, so not loaded by `tau_forge.train.dataset`) as candidate
material for a future SFT warm-start pass or a larger (10+) batch of the same
shape -- see `data/synthetic/deferred/README.md`.

**A related gap this dataset does not close, surfaced by an independent
resemblance check against the real 114 τ²-bench retail tasks** (tasks.json
read for aggregate comparison only, never fed into a generation prompt): real
reference trajectories average **4.8 tool calls per task** (median 5, up to
13), since a real task makes the agent do the *entire* chain -- authenticate,
look up the order, look up the product, then act -- while every synthetic
scenario here is a single static snapshot graded as one decision point (by
construction, per the one-call cap below), averaging under 1 call each. This
is a static-snapshot-vs-live-multi-turn-conversation difference in what's
being measured, not a bug to fix in this dataset -- but it means aggregate
tool-count resemblance to the real benchmark is out of scope for Phase 2/3 and
stays an open question for Phase 8's live-benchmark evaluation.

A few real-data findings surfaced during the sweep, beyond the pilot's finding
above (all independently re-discoverable from `third_party/tau2-bench/src/tau2/domains/retail/tools.py`,
not just taken on a subagent's word):
- `modify_pending_order_items` always appends a `payment_history` entry, even for
  a $0 price difference -- so once an order has had its items modified once (status
  `"pending (item modified)"`), `modify_pending_order_payment` can no longer
  succeed (its "exactly one payment history entry" precondition breaks), even
  though `modify_pending_order_address`'s looser substring status check still
  allows an address change on the same order. Two different tools reading the
  same nominal "pending" state disagree on whether a second mutation is legal, for
  two unrelated reasons (one an explicit status check, one a side-effect of an
  earlier call) -- exercised directly in several `order_state_confusion` cells.
- `return_delivered_order_items` only allows a refund to the order's *original*
  payment method or an existing gift card -- never an arbitrary other credit card
  or PayPal account on file, even though `exchange_delivered_order_items` has no
  such restriction on where a price difference is charged. Several cells'
  interactive authoring first proposed a "refund to a different card" scenario
  that failed real execution for exactly this reason before being corrected.
- The base `db.json` fixture has no orders in `"pending (item modified)"`,
  `"exchange requested"`, or `"return requested"` -- those states only exist
  after a mutating call runs. Cells needing them constructed the state live
  (executed the first action for real against a `RetailEnv`, confirmed the
  resulting status, then wrote that first action into `prior_turns` as narrated
  dialogue) rather than searching for a pre-existing example or fabricating one,
  consistent with the one-call cap.

**Status: Phase 3 stages 2 (model checker) and 4 (difficulty calibration) are
now done, and stage 3 (human review)'s sample is generated and delivered —
see the "Model checker" / "Human review" / "Difficulty calibration" sections
below.**

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

### Model checker -- Phase 3, stage 2 (`tau_forge/validate/model_checker.py`)

Stage 1 is mechanical and cannot judge whether a scenario is actually *good*
training data -- that needs an LLM judge. Following this repo's established
pattern for judgment-heavy work (one subagent per group of cells, run in
parallel, given the real tool schemas/policy.md as grounding -- same
methodology as Phase 2's generation), 5 subagents each judged all 6 themes of
one category (~105-112 scenarios per subagent) against three axes: (a) does
the narrative actually motivate `expected_tool_calls`, (b) is `distractor_tool`
a plausible wrong answer with a rationale that's actually correct for *this*
scenario, and (c) (ambiguous/policy_violation only) is `ambiguity_note`
specific rather than generic boilerplate. Each judge wrote one
`{scenario_id, issues, severity}` record per scenario to
`data/synthetic/model_check/<cell>.json` -- deliberately the same
per-scenario/per-cell report shape as stage 1's `Finding`/`CellReport`, so it
composes with the same kind of tooling. `model_checker.py` does not call an
LLM itself; it renders the judge brief and validates/aggregates whatever the
subagents wrote (`validate_report_shape` catches a missing scenario id, a
duplicate, an unknown severity, or an issues/severity mismatch -- structural
guarantees a subagent's free-text judgment could otherwise violate silently).

**Result (as first judged): 541/541 scenarios judged, 22 major, 63 minor, 456
clean** (`uv run python3 -m tau_forge.validate.model_checker`). Per this
project's rule that stage 2 only *reports*, none of the flagged scenarios were
auto-edited at the time -- deciding what to regenerate is stage 3's call. One
class of finding (the authentication-policy gap, below) has since been fixed
after human review confirmed it; see "Human review" below for the current
tally (7 major, 55 minor, 479 clean). The findings themselves are real, not
noise:

- **Two mislabeled-gold-answer bugs**, both confirmed directly against
  `db.json`: `policy_violation__electronics_returns_exchanges__006`'s
  `distractor_rationale` names the wrong Smart Thermostat variant as the
  (unavailable) exchange target -- the variant the user actually asked for
  (white, Apple HomeKit) is real and *available*, so the scripted refusal is
  wrong, the exchange should proceed. `policy_violation__apparel_footwear_exchanges__017`
  labels a fully-specified, executable exchange (same new item id reused for
  two identical order items, which the tool's signature explicitly supports)
  as a policy violation requiring refusal.
- **Three scenarios built on a false DB premise**:
  `ambiguous__order_state_confusion__014/015/016` each narrate an order as
  already having a prior action applied (item-modified / exchange-requested /
  return-requested) to justify why it's distinguishable from a sibling order
  -- but in the real `db.json` all three orders are untouched (`pending` /
  `delivered` / `delivered` respectively, none of the claimed fields set). A
  real `get_order_details` call would contradict what `prior_turns` tells the
  user, undermining the scenario's own ambiguity reasoning.
- **A systemic authentication-policy gap** (now fixed, see below): 23 of the
  original 85 flagged scenarios -- 13 in `happy_path__order_state_confusion`
  (9 major, 4 minor) and 10 in `happy_path__damaged_or_defective_item_narratives`
  (6 major, 4 minor) -- had the user self-identify with only *one* factor (a
  bare stated `user_id`, or a name with no zip/email) with the assistant
  proceeding straight to a money-moving write action, contradicting
  policy.md's explicit "this has to be done even when the user already
  provides the user id" authentication requirement. Severity was `major` when
  the resulting action mutates state, `minor` for a read-only lookup on the
  same pattern.
- **A structural consistency gap** (52 of the 63 minor flags): three
  `policy_violation` cells (`electronics_returns_exchanges`,
  `apparel_footwear_exchanges`, `address_payment_modification`) leave
  `ambiguity_note` empty on every scenario, while the other three
  `policy_violation` cells populate it with violation-specific reasoning
  throughout. The substantive reasoning is present in `distractor_rationale`
  in every case checked, so this is a field-population gap, not a missing-
  information one -- still worth fixing for consistency since axis (c)
  explicitly applies to this category.
- Several smaller single-scenario issues: a gold call whose arguments
  (`new_item_ids`, `payment_method_id`) are never actually stated anywhere in
  the visible conversation (`happy_path__electronics_returns_exchanges__014`,
  inconsistent with a sibling scenario in the same file that correctly emits
  a lookup instead of guessing); a `get_user_details` call keyed to a
  `user_id` with zero textual grounding for who the customer even is
  (`requires_earlier_context__apparel_footwear_exchanges__018`); a handful of
  `out_of_scope` scenarios that bundle an achievable in-scope action with an
  unsupported condition, making "transfer immediately" a debatable rather
  than clearly-correct single answer.

None of these are stage-1 failures -- every flagged scenario's
`expected_tool_calls` still re-executes cleanly, which is exactly the gap
stage 2 exists to close (a scenario can be mechanically valid and still teach
the wrong thing).

### Human review -- Phase 3, stage 3 (`tau_forge/validate/human_review.py`)

Inherently manual, and deliberately not automated further than making a
*sample* review efficient. `build_sample()` draws a stratified random sample
with a fixed, documented seed (`SAMPLE_SEED = 42`, per-cell sub-seeded so the
sample is stable regardless of dict ordering): `BASE_PER_CELL = 4` scenarios
per cell (120 baseline across all 30 cells) plus up to `MAX_FLAGGED_EXTRA = 3`
additional stage-2-flagged scenarios per cell not already in the random draw
-- oversampling exactly what stage 2 already found questionable rather than
trusting a plain random sample to happen to cover it. Actual sample:
**140 scenarios** (120 base + 20 flagged extras, every one of the 22 major and
a good share of the 63 minor findings included).

The sample is rendered two ways, both generated by `uv run python3 -m
tau_forge.validate.human_review`:
- `data/synthetic/human_review/sample.md` -- a single readable file, prior
  turns / user message / expected call / distractor shown side by side per
  scenario (not raw JSON), committed to the repo as the durable, diffable
  record.
- A filterable HTML page (by category, by flagged-only) built from
  `data/synthetic/human_review/sample.json`, sent directly to the repo owner
  for interactive review (artifact publishing was blocked by this session's
  auto-mode classifier, so it was delivered as a file rather than a hosted
  link).

Verdicts are captured in `data/synthetic/human_review/sample_results.json`
via `record_verdict()` -- one `{scenario_id, cell, verdict, note, reviewer,
timestamp}` entry per sampled scenario, `"pending"` until a reviewer sets it
to `"confirmed_fine"` or `"flagged"` (with a note). `save_stub_results()`
never clobbers an already-recorded verdict on a re-sample, so this is a real,
auditable record of what was and wasn't reviewed, not just a conversation
that happened and left no trace.

**First verdict, and a real fix.** The repo owner reviewed
`happy_path__order_state_confusion__003` (a scenario flagged `major` by
stage 2 for the authentication gap above) and confirmed it: the assistant
treats a bare `"I'm lei_li_6575"` as authenticated before a payment-method
change, with no email or zip ever given, contradicting policy.md. Verdict
recorded as `"flagged"` with the repo owner as reviewer.

Since the same defect was mechanical and systemic (23 scenarios, two
patterns -- a bare `user_id`, or a name with no zip/email -- both traceable
to a real backing user via the order id already in the scenario), it was
fixed programmatically rather than scenario-by-scenario: for each of the 23,
the self-identification snippet in `prior_turns`/`user_message` was replaced
with `"Hi, this is <real first+last name>, my email is <that user's real
email>."`, looked up from the real `db.json` (directly by `user_id` where one
was stated, or via the stated `order_id` -> `order.user_id` where only a name
was given). Nothing else in any scenario changed -- `expected_tool_calls`,
`distractor_tool`/`distractor_rationale`, and every other turn are untouched,
since the defect was purely in how the user authenticated, not what they
asked for or the correct answer.

Verified after the fix:
- Rule checker: still **541/541 clean**, 0 errors, 0 warnings (re-execution
  is unaffected since no tool call arguments changed).
- Model checker severities for those 23 scenario ids updated to `"none"` in
  their `data/synthetic/model_check/<cell>.json` files (the fix directly
  resolves the exact issue text they were flagged for). New stage-2 tally:
  **7 major, 55 minor, 479 clean** (down from 22/63/456).
- Stage 4 difficulty and the stage-3 sample were both regenerated against the
  updated data (`build_sample()`'s flagged-oversampling naturally pulls in
  fewer extras now that fewer scenarios are flagged -- 134 vs. the original
  140 -- while `save_stub_results()` preserved every already-recorded
  verdict, including the one above).

**Status: one scenario reviewed and its underlying systemic bug fixed;
remaining sample (134 scenarios, including the 7 still-major and 55
still-minor stage-2 flags) still awaiting review.**

### Difficulty calibration -- Phase 3, stage 4 (`tau_forge/validate/difficulty.py`)

Every scenario is a static single-decision-point snapshot, not a live
multi-turn conversation (see "Synthetic generation -- Phase 2 full sweep"
above for why), so there's no "how many turns did it take" signal to use.
`compute_difficulty()` instead combines four scenario features into a
`difficulty_score` in `[0, 1]` (weights: 0.40 category + 0.30 action type +
0.20 distractor closeness + up to 0.15 stage-2-flag bump, clipped to `[0,
1]`), each independently justified rather than picked arbitrarily:

1. **Category base rate** (`happy_path` 0.20 < `out_of_scope` 0.30 <
   `requires_earlier_context` 0.55 < `policy_violation` 0.70 < `ambiguous`
   0.75) -- `ambiguous`/`policy_violation` require the harder judgment call of
   recognizing when *not* to act, with no gold call to fall back on and no
   partial credit the way a mutating call's tiered scoring gives.
2. **Action type** (mutating 0.80 > no-call 0.75 > read-only/transfer 0.30)
   -- a mutating call has more ways to get subtly wrong per the Phase 4
   reward function's tiered scoring; withholding a call entirely requires the
   same recognize-not-to-act judgment as (1); a fixed-shape
   `transfer_to_human_agents` call is easy once scope is correctly
   recognized.
3. **Distractor closeness**, via a `TOOL_FAMILIES` grouping (order-mutation /
   user-mutation / identity-lookup / read-lookup / generic) -- a distractor in
   the same family as the correct answer scores 1.0 (hardest to rule out, per
   the order_state_confusion findings above: cancel vs. modify-items vs.
   modify-address vs. exchange vs. return all look superficially similar);
   for `[]`-answer scenarios with no tool to compare against, a mutating
   distractor scores 0.9 (calling it at all is the specific mistake the
   scenario is designed to tempt) vs. 0.4 for a read distractor.
4. **Stage 2 severity bump** (major +0.15, minor +0.07, none +0) -- a
   scenario the model checker had to flag is treated as genuinely harder to
   get right, on the theory that a policy model is more likely to stumble on
   it too.

Output is a **sidecar file per cell**
(`data/synthetic/difficulty/<cell>.json`, `{id, difficulty_score,
difficulty_label, components}`), not a mutated field on the raw scenario
JSON -- the raw files already passed stage-1 review and are the audit trail
for the generation sweep; a derived, regenerable sidecar avoids touching that
record and makes it trivial to recompute if the formula changes.
`difficulty_label` buckets the score at 0.40/0.65 into easy/medium/hard, for
callers (e.g. Phase 7 curriculum ordering or stratified batch sampling) that
want a coarser tag instead of the raw float.

**Result** (`uv run python3 -m tau_forge.validate.difficulty`, 541/541
scored, re-run after the authentication-gap fix above): 143 easy, 169 medium,
229 hard overall. By category (mean score): `out_of_scope` 0.302, `happy_path`
0.446, `requires_earlier_context` 0.569, `policy_violation` 0.650, `ambiguous`
0.702 -- exactly the ordering the category-base weights were designed to
produce, with the other three signals adding real spread within each
category rather than collapsing to the base rate (e.g. `happy_path` ranges
0.220-0.670 depending on lookup-vs-mutating and distractor family alone). The
23 fixed scenarios' scores dropped slightly (their stage-2 severity bump went
to 0), shifting some from `hard` to `medium` -- expected, since difficulty
partly reflects "how much did stage 2 have to flag."

**Status: stages 2 and 4 complete (and re-run after the fix above); stage
3's sample is generated and delivered for review, one scenario reviewed and
its systemic bug fixed, rest pending.** Per the STOP-checkpoint policy,
Phase 5 and Phase 7 still need explicit go-ahead -- this reconciliation of
stages 2-4 does not itself constitute that go-ahead.

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

*(Editorial note from reconciling parallel work: a concurrent session did not
wait for this STOP and completed the remaining 27 Phase 2 cells plus Phase 3's
stage-1 rule checker in parallel with this session's Phase 6 work -- see the
"Phase 2 full sweep" and "Rule checker" sections above. Recorded here as
historical fact, not as retroactive permission for either session's choice.)*

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
these 74 real tasks. AWS EC2 setup for that GPU box is prepped in
`docs/phase7_aws_setup.md` and `infra/ec2_bootstrap.sh` (instance sizing reasoning,
bootstrap script, security checklist) — launching the instance and writing the
actual training script are separate, still-gated next steps.

STOP for review, per the Phase 6 spec: harness validated against real trusted
data before resuming Phase 2/3 or attempting any GPU training.

## Phase 7 signal audit — what was actually limiting the run

Written while diagnosing why a stratified n=16 zero-shot audit came back ~72.5%
zero-variance. Four findings, in descending order of how much each one costs the
Phase 8 number. The first two are bugs; the second two are design choices that
were quietly working against the goal.

### 1. Every training prompt was being truncated (fixed)

`grpo_train.py` shipped `--max-prompt-length 2048`. Measured against the real
rendered prompts:

| | tokens (approx) |
|---|---|
| tau2 system prompt + retail policy | 1,950–2,600 |
| 16 retail tool schemas | 3,080–4,100 |
| **full rendered prompt, median** | **5,100–6,800** |
| scenarios exceeding 2,048 tokens | **541 / 541** |

TRL truncates a prompt by keeping its **last** `max_prompt_length` tokens, so
this does not raise -- it silently deletes the system prompt, the retail policy
and every tool definition from all 541 examples, then grades the resulting
completions with a reward function that still assumes they were there. The
default is now `None` (no truncation), and `check_prompt_lengths()` tokenizes
the real prompts at startup and refuses to run if a supplied value would
truncate (`--allow-prompt-truncation` overrides deliberately). Covered by
`tests/test_grpo_signal.py::test_preflight_refuses_a_truncating_max_prompt_length`.

### 2. The training prompt and the eval prompt were different (fixed)

`dataset.TOOL_CALL_FORMAT_INSTRUCTION` is appended to tau2's `AGENT_INSTRUCTION`
for training. tau2's own `LLMAgent.system_prompt` does not append it, so `tau2
run` at Phase 8 would send a system prompt the trained weights had never seen --
and, specifically, would drop the escalation paragraph that roughly doubled the
`out_of_scope` mean score and took that category's zero-variance rate from ~85%
to ~49%. `tau_forge/eval/prompt_parity.py` patches tau2's module-level
constant so the two prompts are byte-identical, and `tau_forge/eval/run_tau2.py`
asserts it before launching. The patch must be applied to the **baseline** run
too -- otherwise the reported delta mixes a prompt change into a weights change.
The alternative regime (train with `include_format_instruction=False`, evaluate
`--stock-prompt`) is equally valid; the tests enforce that one of the two is
consistently in force, not that a particular one is chosen.

Writing the parity check immediately caught a second-order bug it created:
`_system_message` appended the suffix unconditionally, so building a dataset
inside a patched process produced prompts carrying it twice. The append is now
idempotent.

### 3. The reward function's flat 0.0 floor is what makes cold starts dead

`reward()` returns exactly `0.0` for `wrong_tool`, `unknown_tool`,
`missing_call`, `unexpected_call` and a malformed `<tool_call>` alike. A
cold-start scenario is therefore not "the model can't do it" -- it is "the model
is wrong in sixteen materially different ways and the reward says all sixteen
are identical." GRPO's advantage is `(r - mean) / std`; a group of sixteen
distinct failures scoring the same number produces no gradient at all. **Raising
the sampling temperature or the group size cannot fix that** -- the degeneracy
is in the reward surface, not the sampler, which is why the temperature sweep
was not the right first move.

`tau_forge/train/shaping.py` adds a second TRL reward function supplying bounded
partial credit *inside* that floor: a parseable call (0.02), naming a real tool
(+0.03), schema-valid args (+0.03), the same read/write class as gold (+0.02),
and naming the same order/user record as gold (+0.05). Capped at 0.15, strictly
below `reward()`'s 0.2 tier, so no wrong tool can ever outscore a right one --
asserted in tests, which also confirm the four-way ordering
`malformed < fake tool < wrong class < right record` that `reward()` collapses
to a single 0.0. It stays silent where `reward()` already grades: gold-is-silence
scenarios, empty completions, and right-tool calls. A `-0.05` penalty applies to
a turn emitting more than one `<tool_call>` block (a retail-policy violation, and
only the first is ever parsed). Re-run the audit with
`zero_shot_baseline --with-shaping` to get both zero-variance fractions in one
pass; the difference between them is how many dead groups this revives.

### 4. Over half the corpus trains the model *not* to do retail work

Counted directly off `data/synthetic/raw/`:

| category | n | gold action |
|---|---|---|
| `ambiguous` | 104 | no tool call (all) |
| `policy_violation` | 112 | no tool call (78 of 112) |
| `out_of_scope` | 107 | `transfer_to_human_agents` |
| `happy_path` | 110 | a real retail action |
| `requires_earlier_context` | 108 | a real retail action |

That is 182 scenarios (33.6%) graded on withholding a call and another 107
(19.8%) on escalating out of the domain: **289 of 541, 53.4% of the training
signal, spent on not acting.** The benchmark is the opposite shape -- this
project's own finding is that 4 of 114 real retail tasks (3.5%) ever need
`transfer_to_human_agents`, Phase 6 found 2 of 74 train tasks (2.7%) whose
correct behavior is purely conversational, and real tasks average 4.8 sequential
tool calls. On an all-or-nothing multi-turn benchmark, a policy nudged toward
asking a clarifying question where it should have authenticated and mutated
fails the whole task.

`tau_forge/train/curriculum.py` adds `REAL_TASK_ALIGNED_MIX`, a target
distribution shaped like the benchmark rather than like the generation taxonomy.
Measured effect:

| | n | no-call share | non-acting share |
|---|---|---|---|
| corpus as generated | 541 | 33.6% | **53.4%** |
| `--category-mix real` | 300 | 18.7% | **23.7%** |

It keeps a real minority of guardrail scenarios on purpose -- refusing an
out-of-policy mutation is worth genuine points too, and zeroing them would trade
one lopsided policy for its mirror image. Downsampling is round-robin across
themes, so all 30 `category__theme` cells survive. Note the cost: 541 → 300
prompts, because the mixture is capped by how many acting scenarios exist. **The
better version of this fix is generating more `happy_path` /
`requires_earlier_context` scenarios rather than discarding the rest** -- Phase
2's generator can produce them, and that is the single highest-value data task
left.

The same module supplies `--exclude-zero-variance-from` (drop scenarios measured
constant-at-0.0 in a `zero_shot_baseline` output; constant-at-1.0 are kept by
default as cheap regression insurance) and a stratified `--val-fraction` split
for checkpoint selection. The validation set is **synthetic**, deliberately:
selecting a checkpoint by a real task's score steers weights by that task as
surely as training on it would, which the held-out data policy forbids.

### Other GRPO settings changed, and why

| setting | was | now | reason |
|---|---|---|---|
| `--scale-rewards` | on (TRL default) | **off** | Dividing the advantage by the group's reward std rescales every group to unit variance -- so a group whose sixteen samples nearly agree gets its remaining noise amplified to the same magnitude as a group with real signal. Backwards on a corpus measured ~72% near-degenerate. |
| `--loss-type` | `grpo` | `dr_grpo` | Removes the length normalization that otherwise pays the policy to pad a one-tool-call answer. |
| `--gradient-accumulation-steps` | 1 | 4 | At `--num-generations 16` and an effective batch of 32 across 4 GPUs, accumulation of 1 means each optimizer step sees **two** unique prompts. Group size buys a good advantage estimate within a prompt; only accumulation buys diversity across the update. |
| `--top-p` / `--top-k` | not exposed | exposed | The plain-HF baseline backend inherited Qwen3's shipped `generation_config` (`top_p=0.8`, `top_k=20`), which narrows exactly the distribution the variance audit exists to measure -- and does not match what the trainer samples with. Both are now explicit on both backends. |
| checkpoint selection | last step | best `eval_reward` | The overfitting mitigation `docs/phase7_aws_setup.md` asks for, and what makes a lower `--beta` safe to try. |
| `--beta` | 0.04 | 0.04 (unchanged) | Left as-is deliberately, but the flag help now records the tradeoff: KL to the reference is a ceiling on how far the policy can move, so it also caps the Phase 8 gain. With held-out checkpoint selection in place, 0.0–0.01 is the setting to try for maximum measured improvement. |

`--dry-run` runs the whole dataset/mixture/prompt-length preflight without
importing torch. Run it before every launch.

## Phase 8 evaluation harness (`tau_forge/eval/`) — the number to beat

`python -m tau_forge.eval.run_tau2 --label baseline` runs the real τ²-bench
retail benchmark against a locally-served checkpoint, with prompt parity
asserted and every non-weight variable (user-simulator model and temperature,
seed, trials, max steps) pinned as a flag and recorded in the results directory
name. Serve the policy with vLLM first:

```
vllm serve <model-or-checkpoint> --served-model-name tau-forge-policy \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 16384 --port 8000
```

`--tool-call-parser hermes` is the parser for Qwen's
`<tool_call>{...}</tool_call>` convention -- the same one
`completion_parsing.py` grades during training. Without it vLLM returns the
call as plain assistant text, tau2 sees an agent that never calls a tool, and
every task fails for a reason unrelated to the policy.

**Run the baseline before any training.** It is both the referent for
"improvement" and the only source of real failure trajectories -- and the
current synthetic taxonomy was designed a priori, not from observed failures on
the actual benchmark. `--num-trials 4` rather than 1: the user simulator is
itself stochastic, so a single trial cannot resolve a training delta.

### Recommended order for the GPU box

1. **Baseline eval.** `run_tau2 --label baseline --task-split-name test --num-trials 4`.
   Nothing downstream is interpretable without it, and its failure trajectories
   are the only evidence-based input to what the corpus should contain.
2. **Variance audit at training settings.** `zero_shot_baseline --use-vllm
   --samples-per-scenario 16 --with-shaping --save-completions`, over all 541,
   with `--temperature/--top-p/--top-k` matching what `grpo_train` will use.
   One pass now reports both the raw and the shaped zero-variance fraction.
3. **`grpo_train --dry-run`** with the chosen `--category-mix` and
   `--exclude-zero-variance-from` pointing at step 2's output. Read the printed
   mixture and prompt-token distribution before spending a GPU-hour.
4. **50–100 step smoke test** (`--smoke-test`), then the full run.
5. **Eval every saved checkpoint** with the same `run_tau2` flags as step 1,
   changing only `--label`.

Temperature sweeps and a larger `--num-generations` are further down this list
than they look. Neither can rescue a scenario whose failures all score exactly
0.0 -- finding 3 is that intervention, and a genuine cold start needs prompting
or an SFT warm-start, not more samples.

## Decontamination check (`tau_forge/decontam/check.py`) — Phase 5

**What this checks, precisely.** The 541 synthetic scenarios and the 114 real
τ²-bench retail tasks draw from the same shared `db.json` (1000 orders, 500
users, 50 products), so a synthetic scenario reusing a real order/product id
is normal shared-inventory overlap, not evidence of copying. What this phase
checks instead is *narrative*-level duplication -- does a synthetic scenario
tell essentially the same story as a real task -- via two independent
signals, computed only for this isolated comparison and never fed into
`tau_forge.gen.prompt_template` or any other generation-facing code:

- **Narrative text similarity**: TF-IDF (word unigrams + bigrams) cosine
  similarity between a synthetic scenario's dialogue (`prior_turns` +
  `user_message`) and a real task's `reason_for_call` -- the free-text
  narrative substance of `user_scenario`, not its whole templated object
  (which is mostly constant boilerplate headers and persona/instruction
  voice notes, not story content). No embedding model/API was available in
  this session; TF-IDF cosine over word n-grams was chosen over a
  character-level measure like `difflib.SequenceMatcher` because it's a
  better fit for "same story, different words" duplication across two
  structurally different text registers (synthetic dialogue vs. real
  third-person narration), and needs nothing heavier than `numpy`, already
  present transitively via `tau2` -- no new dependency was added.
- **Tool-call shape**: Jaccard overlap between the tools a synthetic
  scenario's `expected_tool_calls` uses and the tools a real task's gold
  assistant actions use, plus (only when the same tool appears in both)
  Jaccard overlap of argument *keys* -- deliberately never argument
  *values*, since values are exactly where shared-inventory ids would leak
  in and produce a spurious signal. Reported alongside each flagged pair as
  corroborating (or undermining) context, not as a second independent
  trigger for flagging.

**Threshold.** There is no meaningful *absolute* cosine cutoff across two
registers this different -- even a genuinely similar narrative (same kind of
request, different specifics) tends to score low in absolute terms simply
because the phrasing conventions differ; the real 114 tasks' `reason_for_call`
text is third-person case-file narration, the synthetic scenarios are
first/second-person dialogue. So the threshold is derived from this run's own
similarity distribution rather than picked as a fixed number: for each of the
541 synthetic scenarios, take its best (max) similarity against any of the
114 real tasks, then flag scenarios whose best match is a statistical outlier
at `mean + 3*std` (a standard 3-sigma rule) of that per-scenario-max
distribution.

**Results** (`data/decontam/decontam_report.json`, committed):
- Split-level sanity re-check (per the held-out policy, still all 114):
  **train 74 / test 40 / base 114, train/test disjoint, train ∪ test == base
  -- confirmed unchanged** for the pinned tau2-bench commit.
- Distribution of per-scenario best-match similarity across all 541 x 114 =
  61,674 pairs: mean **0.0882**, std **0.0362** → threshold **0.1969**.
- **8 / 541 scenarios flagged** (1.5%), similarity 0.2003-0.2577, all sharing
  a real task's tool and general theme (t-shirt exchange, laptop upgrade,
  garden-hose return) but not its specifics:

  | Synthetic scenario | Real task | Similarity | Tool shape |
  |---|---|---|---|
  | `happy_path__apparel_footwear_exchanges__015` | 85 | 0.2577 | 0.0 |
  | `policy_violation__apparel_footwear_exchanges__004` | 80 | 0.2537 | n/a (no call) |
  | `requires_earlier_context__apparel_footwear_exchanges__001` | 80 | 0.2351 | 1.0 |
  | `happy_path__apparel_footwear_exchanges__001` | 80 | 0.2274 | 1.0 |
  | `happy_path__damaged_or_defective_item_narratives__009` | 28 | 0.2226 | 0.6 |
  | `happy_path__electronics_returns_exchanges__003` | 93 | 0.2153 | 1.0 |
  | `requires_earlier_context__damaged_or_defective_item_narratives__003` | 56 | 0.2010 | 0.0 |
  | `requires_earlier_context__damaged_or_defective_item_narratives__014` | 44 | 0.2003 | 0.0 |

**Spot-check verdict: 0/8 confirmed true positives -- all 8 are false
positives.** Manually read every flagged pair's full text against the
excerpts in the report. Representative examples:
  - `happy_path__apparel_footwear_exchanges__001`/`__requires_earlier_context...__001`
    both flag against real task 80 (exchange a red XXL crew-neck cotton
    t-shirt) purely because both are t-shirt color/size/style exchanges on a
    delivered order, with a tool-shape score of 1.0 to match -- but different
    order (`#W6552785` vs. `#W7209932`), different starting/target
    color-size-style combination, and different payment method (PayPal /
    store credit vs. gift card in the real task). Same taxonomy cell theme
    (`apparel_footwear_exchanges`), different story.
  - `happy_path__damaged_or_defective_item_narratives__009` flags against
    real task 28 because both mention a defective garden hose -- but the
    synthetic scenario is a single-item return, while task 28 is a five-item
    return (skateboard, garden hose, backpack, keyboard, bed) plus a
    same-item order cancellation and a running refund total, a materially
    different (and much larger) request.
  - `policy_violation__apparel_footwear_exchanges__004` has no
    `expected_tool_calls` at all (`tool_shape_score: null`) -- it's flagged
    on text alone, and reads as a generic "swap the style" follow-up request
    that happens to lexically overlap real task 80's t-shirt exchange
    narrative without describing the same event.

  This is consistent with how the synthetic data was generated: each
  taxonomy cell (`tau_forge/gen/taxonomy.py`) targets a small, fixed set of
  domain-grounded themes (e.g. `apparel_footwear_exchanges`,
  `damaged_or_defective_item_narratives`), and the real 114 tasks were
  authored against the same domain and the same 50-product catalog -- so
  some thematic resonance within a theme is expected and is exactly what
  this check is designed to tell apart from actual narrative copying. It
  found none.

No scenario was auto-deleted or auto-regenerated -- per the phase spec, that
call belongs to the repo owner after reviewing the report; none is
recommended here given the spot-check above.

**Tests** (`tests/test_decontam.py`, `uv run pytest`): 12 tests covering
tokenization, TF-IDF cosine on identical/disjoint documents, narrative
rendering, tool-shape scoring (matching tools, disjoint tools, non-assistant
actions excluded, argument-key-only comparison), the split sanity check
against real pinned data, and the end-to-end flagging logic against small
fixture data (a deliberate near-duplicate pair flagged, an unrelated pair
not). The full 541x114 comparison is exercised by
`uv run python -m tau_forge.decontam.check`, not by the test suite.

STOP for review, per the Phase 5 spec and the project's STOP-checkpoint
policy: **do not start Phase 7 (real GRPO training) off the back of this
without an explicit go-ahead**, even though decontamination came back clean.
