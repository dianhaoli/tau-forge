# Design doc: GRPO tasks with real variance for Qwen3-4B-Instruct-2507 on tau2 retail

## 0. Summary

The zero-variance rate is high because of how the tasks are built. Temperature doesn't explain it. Every scenario has one decision point, and the policy's choice there (call or don't call, then which tool) is close to argmax. On top of that, the grader collapses whatever variation remains onto a few tied values: {0, 0.2, 0.3, 1.0}, binary no-call, and a transfer floor of 0.300. Fixing in place (A) and accept-set grading (C) remove label errors and wrong-direction gradients, but neither creates variance. C actually pushes more groups to a flat 1.0.

**Recommendation:** make B the main training signal. B is programmatic multi-step episodes against a live `RetailEnv`, with a deterministic scripted user and tau2-style end-state reward. Keep a small set of single-step decision points, cut from oracle episode prefixes and graded with accept-sets, as an auxiliary mix.

The prototype covered 4 templates:
- 160/160 generated tasks verified (gold chain executes, every id in a gold call was revealed by an earlier read).
- The oracle agent scores 1.0 on all of them.
- Seven plausible near-miss behaviours land on separate reward levels.

## 1. Why the current corpus can't give variance

**Group-flatness arithmetic.** With a mostly binary reward, P(all 16 samples equal) = p^16 + (1−p)^16:

| p | 0.05 | 0.15 | 0.30 | 0.50 | 0.85 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| P(flat group) | 0.44 | 0.074 | 0.003 | 3e-5 | 0.074 | 0.44 | 0.85 |

One near-argmax decision per prompt puts most scenarios at p<0.05 or p>0.95. A chain of k decisions with per-step reliability r gives p≈r^k. With r=0.9 and k=6–8 that is 0.43–0.53. This is an illustration only, since there is no GPU here to measure it.

**Audit causes, and which option fixes each:**

| Audit cause (confirmed counts) | A: fix in place | C: accept-set | B: episodes |
|---|---|---|---|
| Unreachable gold ids: 103/150 writes prose-only, 55/150 still after grounding, 10 reads | Partly (grounding, plus auth turns for 50) | Gives lookups partial credit | Gone: ids come only from the agent's own tool outputs (verified for 160/160) |
| Confirmation missing on 72(+26) write golds; 167/182 no-call golds solved by a reflexive "confirm?" | Add recap+yes turns | Accepts a recap | The scripted user only says yes after a recap. A late-correction knob makes skipping confirmation visible in the DB |
| Missing auth (~140 call-gold scenarios; auth rewarded in <2%) | Prepend auth | Accepts an auth call | Auth is a required milestone and is gated in the reward |
| Transfer SequenceMatcher floor (107 scenarios) | Grader fix | n/a | Transfer is graded by decision plus an unchanged DB, like tau2 (summary never read) |
| No-call text ungraded (182) | Content grader | n/a | Refusals are graded by end state (a compliant write gets 0), and the user then ends the episode |
| Four-valued tiers, near-misses tie | Per-arg credit | n/a | Binary end state plus capped milestone shaping. Near-misses separate (see §4) |
| Copyable recaps, flat 1.0 (39) | Natural-language recaps | Worse (more 1.0) | Ids are never in user text. Variants and cards are described in natural language |
| Single-step "unusual means transfer" habit, mismatch with tau2's ~4.8 calls/task | No | No | Episodes average 4.0–5.7 gold calls (measured) |

**A is worth doing anyway as a regression set,** but it is capped at 541 hand-labelled items and stays single-decision.

**C is right for whatever single-step data remains.** For example, at an auth step either `find_user_id_*` call is accepted. It is a correctness fix, not a source of variance.

## 2. Option B: episode tasks

### 2.1 Generator (from db.json only)

Each template samples (user, order in the right status, items, a valid target, a payment method). It enforces:
- **Uniqueness.** The item's product name is unique within the order. The option change maps to exactly one available variant. The payment phrase ("my Visa ending 9385", "my PayPal account") maps to exactly one method. Email or name+zip identifies exactly one user. A product hint maps to exactly one order with the right status.
- **Verification.** The gold chain is replayed on a fresh copy of the DB. Every call must succeed. Every id in a write or read argument must have appeared in the opening, the profile, or an earlier gold read's output. The gold end-state hash `gold_db_hash` is stored.
- **Dedupe** on (template, user, order, item set). The prototype doesn't do this yet: only 143 of 160 (user, order, template) tuples were distinct.

### 2.2 Templates

Templates 1–3 and 6 are prototyped; the rest are specified.

| # | Template | Gold end state | Difficulty knobs | Mechanism the end state actually tests |
|---|---|---|---|---|
| 1 | `cancel` | `cancel_pending_order(order, reason)` | give order id or a product hint; email or name+zip; late correction ("wrong one, I meant …") | reason mapping (stored in DB); picking the right pending order; late correction catches writes made before confirmation |
| 2 | `exchange` | `exchange_delivered_order_items` | 1–3 items; order id or "an order I received recently" (scan orders); late option correction; gift card that must cover the difference | variant resolution from `get_product_details`; payment resolution from `get_user_details` |
| 3 | `return_fallback` (policy negotiation) | return refunded to the **original** method | user first asks for a non-original, non-gift-card method; owns a gift card or not | policy denial, then the user's fallback. An agent that picks the gift card itself ends in the wrong state |
| 4 | `modify_items`, deferred item | one `modify_pending_order_items` covering all items | the second item is revealed only when the agent asks "anything else / all items?" | "collect all items first". A second call fails because the status becomes pending (item modified), so the DB is wrong |
| 5 | `modify_payment` | `modify_pending_order_payment` | target gift card with too little balance, then the user falls back to a second method | balance rule plus fallback |
| 5b | `modify_address`, `modify_user_address` | the matching write | late apartment-number correction | address normalisation, confirmation |
| 6 | `foreign_order_refusal` | DB unchanged | any write tool | the tool does not check ownership, so complying changes the DB (reward 0) |
| 7 | `out_of_scope_after_work` | doable write + `transfer_to_human_agents` | complaint about the courier after an exchange | finish the doable part, then transfer |
| 8 | `multi_request` | two writes on two orders | chain two templates | matches tau2's length |
| 9 | (optional, risky) `invalid_cancel_reason` | unchanged | "found it cheaper elsewhere" | coercing the reason into 'no longer needed' changes the DB. Reasonable people could map this phrase either way, so pilot it first |

Policy rules the tools already enforce (cancelling a non-pending order, cross-product exchange, non-original refund) can't be seen in the end state, because the tool raises. Only use them when a user fallback makes the outcome observable (templates 3 and 5). Otherwise leave them to the single-step set.

### 2.3 Scripted-user protocol (deterministic)

**Hidden profile.** Identity answer, whether the user knows the order id (or only a product hint), reason phrase, payment phrase, all-items answer, optional deferred item, optional single correction, fallback line, denial-acceptance line. The user never says an id other than an order id or an email.

**Intent matching on each agent text turn, in priority order:**
1. If a write has succeeded: "thanks, that's all" and `###STOP###`.
2. On a refusal template, if the agent's text matches the denial regex: accept and STOP.
3. If the agent denies and a fallback is pending: give the fallback.
4. If the agent recaps and asks to confirm: give the pending correction once if there is one, otherwise "Yes, please proceed." (This sets the `yes` flag.)
5. If asked for identity: give it.
6. If asked about all/other items: give the deferred item or "that's everything".
7. If asked for the order id: give the id or the hint.
8. If asked for a reason: give the reason.
9. If asked about payment: give the payment phrase.
10. Otherwise restate the request. On the third unrecognised turn, STOP.

**Anti-exploit rules** (GRPO will find regex quirks):
- The user says yes only if the recap mentions the target order id or product name.
- Each line is drawn from a paraphrase pool, seeded per task.
- Track drift on a held-out generated set using tau2's LLM `user_simulator`.

**Termination.** User STOP; a `transfer_to_human_agents` call; 30 agent messages; or hitting the token budget. On truncation, set `mask_truncated_completions=True`.

### 2.4 Reward

- **Success:** R = 1[final_db_hash == gold_db_hash]. This is the tau2 semantics in `third_party/tau2-bench/src/tau2/evaluator/evaluator_env.py:118-124`.
- **Policy gates** that tau2's DB check misses: −0.3 if a write happened without a yes since the last recap, and −0.3 if a write came before authenticating the right user.
- **Failure shaping,** capped at 0.2: auth of the right user +0.05, read of the target order +0.05, right write tool on the right order +0.1. Shaping is 0 if any write hit another record.
- **Refusal templates:** unchanged DB scores 1.0. Unchanged DB with a transfer scores 0.5, because policy says deny.
- **Out-of-scope:** 1.0 for transfer plus a matching DB. The summary is never graded.

### 2.5 Measured prototype results

These numbers are from the original scratch prototype; the in-repo version (`tau_forge/episodes/`) re-measures them in its tests and README.

| Template | Requested / verified | Mean gold calls | Gold tool-output tokens, median / max |
|---|---|---|---|
| exchange | 40 / 40 | 5.7 | 2536 / 4358 |
| cancel | 40 / 40 | 4.3 | 1076 / 2278 |
| return_fallback | 40 / 40 | 4.0 | 1097 / 1471 |
| foreign_cancel_refusal | 40 / 40 | 2.0 | 172 / 252 |

Reward by behaviour (mean, with the distinct values seen):

| Template | oracle | no-confirm | no-confirm with late correction | wrong variant | made-up pm | skip auth | comply | gift-card fallback | transfer |
|---|---|---|---|---|---|---|---|---|---|
| exchange | 1.0 | 0.49 {0.2, 0.7} | 0.2 | 0.2 | 0.2 | 0.7 | – | – | 0.0 |
| cancel | 1.0 | 0.51 {0, 0.7} | 0.0 | – | – | 0.7 | – | – | 0.0 |
| return_fallback | 1.0 | 0.7 | – | – | – | – | 0.2 | 0.84 {0.2, 1.0} | 0.0 |
| foreign refusal | 1.0 | – | – | – | – | – | 0.0 | – | 0.5 |

- The oracle resolves every id only from tool outputs, so 1.0 for the oracle shows each task can be solved through the protocol. There is a sample transcript in `space.py`: name+zip auth, then an order scan, then the user corrects the variant mid-recap, then 1.0.
- **Task space:** 1,945 single-item exchange (order, item, target) triples across 358 delivered orders; 3,384 cancel combinations; 185 return_fallback combinations. Multi-item and multi-request templates multiply these. Thousands of tasks are realistic.
- **CPU cost per episode** is about 0.3 s (deep copy 0.15 s, env 0.10 s, hash 0.06 s). That needs a process pool, or copy-on-write of only the touched user and order.

**Prototype limitations:**
- Opening text is clumsy (e.g. "the Dumbbell Set to adjustable set type"). Render it from templates or LLM paraphrase, then check that the oracle still resolves it.
- The oracle reads hidden natural-language targets (product name and option change). Only the ids are proven reachable.
- Dedupe and the decontamination hook are not implemented.

### 2.6 Calibrating to p in [0.15, 0.85]

1. **Grid** = template × n_items × give_order_id × id_mode × late_correction × deferred_item × distractor orders.
2. **Pre-audit:** about 64 tasks per cell at n=8, T=1.0, using the trainer's exact sampler. Fix the `top_k=None` bug (H1) first.
3. **Mixture:** weight cells by p(1−p).
4. **Online adjustment:** generate from an `IterableDataset` whose cell weights a `TrainerCallback` updates from per-cell success logged by the reward function. TRL accepts an IterableDataset when `max_steps` is set.
5. **Report** an effective-variance bucket (std ≥ 0.05). Don't use a 1e-3 range test.

### 2.7 TRL integration (locked trl 1.12.0, source read from the scratch copy)

**`environment_factory`** (`grpo_trainer.py:269-288, 640-700`) does not fit. The class's methods become tools, `reset(**row)` can append to the first user message, and `get_reward()` scores the rollout. But the built-in `_tool_call_loop` (`:1981`) stops at the first assistant turn with no tool call (`grpo_config.py:324-327`). That means the user can't reply mid-episode, so recap, yes, and correction are impossible. Using it would require pre-authorised openings, which teaches the model to skip confirmation. Only use it for read-only sub-skills.

**Recommended: `rollout_func`** (experimental; `:261-268, :2227-2246`). It receives (prompts, trainer) and must return `prompt_ids`, `completion_ids`, `logprobs`. It may also return `env_mask` and extra fields, which are forwarded to the reward functions. The rollout loop works like this:
- Each turn, batch-generate across the active episodes with `trainer.vllm_generation.generate(prompts=token_ids, images=None, num_generations=1)` (`vllm_generation.py:550`).
- Parse `<tool_call>` with the existing `completion_parsing`.
- Append tool and user turns as template-rendered tokens with `env_mask=0` and a dummy logprob of 0.0.
- Return the whole post-prompt episode as `completion_ids`.

The mask feeds the loss and importance-sampling masks (`:2515, :2679, :2982`).

**Caveats, stated honestly:**
1. **Do not also pass `tools=`.** `env_mask` is only read in the `else` branch of `if self.tools` (`:2270-2291`).
2. **Duplication semantics are unclear.** The docstring says "raw per-process prompt slice with no duplication", but the `RepeatSampler` repeats prompts `num_generations` times (`:1236`). Assert which it is in a 2-prompt smoke test.
3. **Token-concatenation drift.** Build ids by concatenation and test them against `apply_chat_template(full_messages)`.
4. **Budget.** The system prompt is 5.1–6.8k tokens (README) plus up to about 4.4k of tool output plus turns. Use `max_model_len` 16384 and `max_completion_length` of about 8–10k.
5. **The API is marked experimental** and may change.

### 2.8 Decontamination from the 114 real tasks

- The generator reads only `db.json` and `policy.md`, never `tasks.json` or `data/trusted/train_tasks.json`. All 114 tasks are held out (README).
- Episodes share the DB, so a generated (user, order, action) can **coincide** with a real task's gold end state. The lead should add a helper inside the sanctioned `tau_forge/decontam` module that exports only:
  - sha256 hashes of the user_ids in the 114 tasks' gold actions, and
  - the 114 gold end-state hashes.

  The generator then rejects tasks whose user is in that set (at most 114 of 500 users) or whose `gold_db_hash` matches a real one, and logs only counts. I did not implement this: it requires reading the held-out file.
- Also run the existing TF-IDF narrative check (`tau_forge/decontam/check.py`, 3σ rule) on rendered openings and user lines.

## 3. Auxiliary single-step set (A + C, rebuilt)

- Cut decision points from oracle episode prefixes. They are grounded with real tool outputs, confirmation and auth are present, and the ids are reachable. Gold is the oracle's next action, with accept-sets (e.g. either auth tool; any candidate-order lookup).
- Grade text turns by intent class, using the scripted user's classifier: ask for identity, recap+ask, deny.
- Keep only items whose pre-audited p is in [0.15, 0.85]. Mix at about 20–30%.
- Of the 541 legacy scenarios, keep the fixed ones as an eval and regression set only.

## 4. Rollout order

1. Fix H1 (top_k) and H2 (exclusion uses raw scores).
2. Re-audit grounded single-step data with shaping, to get a baseline.
3. Build `rollout_func` with the scripted user and smoke-test it on 2 prompts. Check the duplication assert and the token-parity test.
4. Generate around 2k episodes over the 6 core templates and pre-audit the grid at n=8.
5. Train on the cells with p in [0.15, 0.85] plus the 20–30% auxiliary set.
6. Track per-cell p, the effective-variance rate, and an LLM-user held-out check for overfitting to the script.

## Where this landed

Templates 1, 2, 3, 5 and 6 are implemented in `tau_forge/episodes/` (see the README section "Multi-step episode tasks" for what changed from this prototype after review: confirmation and authentication gates hardened, refusal tasks no longer pay 1.0 for silence, tool exceptions no longer end the audit). The TRL `rollout_func` integration in section 2.7 is not implemented.
