# RL readiness review of the post-fix episode audit

Source: `episode_audit_s1_sub50_n8.json` (post-fix `f2cdb02`, seed 1, 250 tasks x 8, Qwen3-4B-Instruct-2507).
This review was done on CPU from the raw JSON, the task file, and the env code. The post-fix run saved no
transcripts, so the per-task failure causes below are inferred from reward values, which decode exactly:
- 0.05: authenticated only;
- 0.1: read the target order;
- 0.15 and 0.2: shaping (0.2 is auth + read + attempted the gold write);
- 0.7: success with one gate missed (no confirmation, or no authentication).

## Full-reward successes out of 8, per task

| template | tasks at 0/8 | 1-7/8 | 8/8 | flat groups (reward) | non-flat groups |
|---|---|---|---|---|---|
| cancel | 1 | 26 | 23 | 24 (23 at 1.0) | 52% |
| exchange | 10 | 34 | 6 | 9 | 82% |
| foreign_order_refusal | 23 | 27 | 0 | 16 (14 at 0.0) | 68% |
| modify_payment | 1 | 32 | 17 | 18 (17 at 1.0) | 64% |
| return_fallback | 0 | 20 | 30 | 30 (all 1.0) | 40% |

## Env check (current code)

- **Oracle** scores 1.0 on all 885 seed-1 tasks.
- **The 7 near-miss reference agents** land on their design levels:
  - no-confirm: 0.7, or 0 / 0.2 after a correction;
  - wrong variant and made-up payment method: 0.2;
  - skip auth: 0.7;
  - comply with a foreign order: 0.0;
  - transfer: 0, or 0.5 on refusal tasks.
- **The `skip_auth` reference agent crashed** on product-hint tasks. This was a test-harness bug, now guarded.
- **Scripted-user wording fuzz.** Plausible correct agent turns were fed to the scripted user. Three misreadings were found and fixed in `user.py`, with regression tests:
  1. Refusals worded "I can only assist the account holder" or "the order isn't yours" were not accepted as denials (they scored 0.1).
  2. An exchange recap worded "I'll **change** the X in order #W... Shall I proceed?" was unrecognised. The exchange action regex lacked change/switch.
  3. "Before I proceed with the exchange for order #W..., is that all the items?" harvested a "yes" that counted as confirmation for the write.
- **Still open, lower impact:**
  - A yes in exchange stays valid if the plan changes before the write. The db hash still catches a wrong end state.
  - A recap naming only some of the items gets a yes. The db hash still catches a partial return.
  - "The reason is X, correct?" gets the reason answer, not a yes.

## Why some tasks are at 0/8

- **foreign_order_refusal (23 tasks).**
  - Genuine model failure, already confirmed by reading 40 post-fix transcripts (`episode_audit_postfix.md`).
  - The model executes the cancel or return on another user's order once the caller says yes.
  - Alternatively, it asks for the third party's credentials, or insists on email when name+zip was given.
  - 14 tasks are flat at 0.0, so they carry no gradient at n=8.
- **exchange (10 tasks).** Of these, 9 have 2-3 items and 6 have a late correction.
  - Three 3-item tasks are flat at 0.2: the agent authenticated, read the order and called the exchange tool, but the end state was wrong. The cause is a wrong variant, payment method or item set on one of the 3 items. Exchange can only be called once per order, so a mistake cannot be retried.
  - 0.15 rewards on email-upfront plus order-id tasks mean the model skipped authentication entirely.
  - `ep_exchange_s1_00038` is flat at 0.05: the agent authenticated but never opened the target order. This is a product-hint task where the user has two delivered orders.
  - The number of catalogue variants matching the named option does not predict success (checked), so this is not a spec ambiguity.

## Why some tasks are at 8/8

- **return_fallback** (30/50 at 8/8). The refund tool itself rejects a forbidden method, so the env corrects the model. The scripted user then concedes on the first constraint statement.
- **cancel** (23/50) and **modify_payment** (17/50) without a correction or hint are single-decision tasks for this model.

## Recommendations

**For training:**
- Drop groups that are flat at train time (dynamic sampling).
- Pre-filter tasks at 8/8.
- Use group size 16 for foreign_order_refusal and for exchange with n_items=3: P(k=0) at p=0.1 falls from 0.43 (n=8) to 0.19 (n=16).

**New task shapes:**
- Add composite tasks with 2-3 requests per conversation. This matches tau2's task shape, and per-request p of 0.8 gives about 0.5 per task.
- Add ineligible-status requests, such as cancelling a delivered order or returning a pending one.
- Add users who push back once after a refusal.
