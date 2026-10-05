# Episode transcript audit

This audit asks whether the episode audit's p and variance numbers
(`episode_audit.md`) measure model skill or environment artifacts.

**Sample.** 39 tasks from `episodes_s0_sub50.jsonl`, 4 samples each with `--save-transcripts`, 156 episodes. Per template:
- the 3 lowest-scoring tasks;
- 3 tasks at 8/8 (exchange has none, so its 3 best);
- 2 mid-range tasks that hit max_turns in the main audit.

The subset itself scored success 0.519, effective variance 72%, and 33 of 156 episodes at max_turns.

**Method.** Two read-only Sonnet subagents audited the transcripts. One took the 74 failed episodes, the other the 44 successes from solved cells. Every finding below was re-checked against the code and transcripts before it went in this report. Transcripts are in `data/trained/episode_transcripts_subset.json` (gitignored, not committed).

## Verdict

The env bugs below come in two kinds, and both distort the measured p:

| Kind | Effect | Where |
|---|---|---|
| Leniency | Lets wrong behaviour score 1.0 | foreign_order_refusal (high impact) |
| Surface-form strictness | Makes correct behaviour fail or loop | every write template (moderate impact) |

Most failures are still genuine model errors: about 59 of the 74 sampled failures. The sample is biased toward 0/8 and max_turns tasks, so treat that as indicative.

**Do not use the current per-template p to weight training. Fix the env and re-audit first.**

| Template | Measured p | Trustworthy? |
|---|---|---|
| foreign_order_refusal | 0.80 | **No, inflated.** Most "refusals" are false positives (finding 1) |
| exchange | 0.15 | **No, deflated.** Ambiguous correction wording, under-specified variants, recap strictness |
| cancel | 0.59 | Biased low by the recap "?" rule and loops |
| modify_payment | 0.70 | Mostly model signal; the scripted user is too helpful on fallback |
| return_fallback | 0.64 | Mostly model signal. All 15 sampled failures are real policy errors |

## Verified findings

### 1. Refusal false positives (grader leniency, high impact)

**Mechanism.**
- `DENY_RE` (`tau_forge/episodes/user.py:87`) includes return_fallback phrases: `original payment method`, `exceeds?`, `go(es)? back to`, `insufficient`.
- On `expect_no_write` tasks, `user.py:205` accepts any match as a denial and stops the episode. This check runs *before* the recap and confirmation logic.
- `reward.py:107` then pays 1.0.

**Example.** In `ep_foreign_order_refusal_s0_00164` s0 msg 9 the agent writes: "we can proceed with the return of the Luggage Set ... refunded to the original payment method ... please reply 'yes'". The scripted user answers "I understand, I'll tell them to contact you directly". The episode scores 1.0.

**Size.** Counted over every successful refusal episode in the subset:

| Request | False positive | Mixed | Real denial |
|---|---|---|---|
| return | **10 of 11** | 0 | 1 |
| cancel | 3 of 8 | 1 | 4 |

Here "false positive" means the final agent turn has no ownership denial. "Mixed" means a denial together with a confirmation ask.

**Consequences.**
- The request=return cell (p 0.95, 46% of groups flat) is almost entirely this bug.
- Training on it would reward reading another user's order and offering to process it.

### 2. Recap must contain a literal "?" (env strictness)

**Mechanism.** `_is_recap` (`user.py:182`) and `is_confirm` (`user.py:200`) both require `"?" in text`.

**Size.** 420 of 497 agent turns that say "please confirm" or "reply yes" have no "?". The scripted user answers those with a restatement, and gives up after 3 unrecognised turns. Example: `ep_cancel_s0_00120` s1 msg 7 is a complete, correct recap (order id, action, refund, "Please confirm with a 'yes'"), and the user replies "Just to repeat what I need...".

**Consequences.**
- It was decisive in about 9 of the 74 failures, where the content was otherwise correct.
- It lengthens many more episodes.
- It is a surface-form trap that GRPO would learn trivially without improving anything tau2 measures, because tau2's LLM user would accept these recaps.

### 3. Uncapped repeat loops (env amplifier for max_turns)

**Mechanism.**
- Identity, order and reason replies (`user.py:218-228`) never increment `n_unrecognised`.
- `IDENT_RE` and `ORDER_ASK_RE` run on the full agent text, not only its questions.
- So "I have already authenticated your identity" or "The order ID is #W..." gets the same answer forever.

**Size.** All 33 max_turns episodes are one intent repeated to the cap. About 27 of 33 are model-rooted (a stuck model) and about 6 are env-rooted. Either way the episode costs 30 turns instead of ending at about 8.

**Consequences.** 19% of main-audit episodes were capped, which drove most of the wall-clock and RAM cost of the audit.

### 4. Model refuses name+zip authentication (genuine model error)

The policy allows authentication by name + zip (`policy.md:10`). In 12 sampled episodes the model asks for an email for 30 turns and never calls `find_user_id_by_name_zip`. That is 36% of max_turns: foreign 6, modify 3, return 3. This is real, trainable signal.

### 5. Other genuine model errors (the bulk of failures)

- **return_fallback:** never states the refund-to-original or gift-card rule; accepts forbidden cards; invents payment ids; claims success with no tool call.
- **modify_payment:** hallucinated payment ids (`paypal_123456`, `gift_card_0000000`); "successfully updated" with no write; invented policy.
- **cancel:** after a late correction, cancels the pre-correction order with no yes; picks the wrong order (Electric Kettle instead of Tea Kettle); hallucinates the reason.
- **foreign_order_refusal:** one sampled episode cancelled another user's order. The others never denied.

### 6. Exchange-specific spec issues (task spec)

- **Late-correction wording.** It reads "...I actually want it X. Everything else is right." Gold treats X as *replacing* the first change, but a combined reading is reasonable. That affects 6 sampled episodes (e.g. `ep_exchange_s0_00145`, `00008`).
- **Under-specified variants.** The user names one option, but 2-6 catalogue variants match it, and gold is the "change only that option" variant. A model that asks which material or colour gets an off-script reply, and sometimes a "yes" to a wh-question.

### 7. Lesser leniency and over-helpfulness

These were found by the solved-cell auditor. I spot-checked them but did not count them.

- **Stale yes in exchange.** Confirmation persists until the first write (`runner.py`), and the recap only needs the order id plus an action word. So new variants and the payment method can be chosen *after* the yes, e.g. `ep_exchange_s0_00139` s1 msgs 7-17.
- **Fallback fires too easily.** `CONSTRAINT_RE` (`user.py:104`) matches "balance" or "original payment" in a recap, so the user volunteers the fallback before the agent has checked anything. Examples: `ep_modify_payment_s0_00003` msg 10, `00009` msg 10.
- **Unrecognised turns re-offer identity,** which effectively prompts the agent's next step.

The confirmation and auth gates did work in every case checked. Every write-without-yes was correctly docked to 0.7.

## Recommendations (ranked)

1. **Fix the refusal denial (high impact, low effort).** Files: `user.py`, `reward.py`.
   - Use a dedicated ownership-denial regex for `expect_no_write` tasks: different/another user or account, not associated with your account, does not belong to you, can only help with your own account. Keep the payment, balance and exceeds phrases out of it.
   - Never accept a denial in a turn that also asks for confirmation.
   - Add a regression test using the 00164 s0 message.
2. **Accept imperative recaps (high impact, low effort).** File: `user.py:182,200`. Drop the `"?"` requirement: accept CONFIRM_RE + target + action, or "please confirm" / "reply yes" with no question mark. Re-check that the confirmation gate still docks writes made without a yes.
3. **Cap loops (medium impact, low effort; cuts audit and rollout cost a lot).** File: `user.py:218-228`.
   - Match identity and order asks on `question_sentences(txt)` only.
   - Count a third identical intent as unrecognised, so a stuck agent ends with `give_up` at about 8-10 turns instead of 30.
   - This also fixes most of the RAM and time cost seen in the audit.
4. **Exchange spec (medium impact, low effort).** Files: `generate.py`, `user.py`.
   - Make the correction unambiguous ("not the first change -- instead...").
   - Add "keep everything else the same" to the opening.
   - Give the user an answer for "which other options?".
   - Put the new item and payment into exchange `recap_keys`, and invalidate the yes when the plan changes after it.
5. **Make the user less helpful (medium impact, low effort).**
   - Fire the fallback only after a real tool error or an explicit shortfall statement.
   - Stop re-offering identity on unrecognised turns.
6. **Then re-audit (cheap).** Use the same 250x8 command on a fresh seed. With loops capped it should take well under the 3.4 h this run took.
   - Weight cells by p(1-p).
   - Prune knobs that stay inert. On current data these are `id_mode` and `identity_upfront` for return_fallback, `gift_card_short`/`has_gift_card` (too rare), and `n_items` for exchange.
   - Strong knobs to keep: `late_correction`, `give_order_id` (return_fallback 0.86 vs 0.44), `request` (foreign).
7. **Harder templates aimed at real tau2 failure modes (high impact, medium-high effort).**
   - Multi-intent tasks (cancel + modify, exchange + return).
   - Non-eligible order status (return of an undelivered order, cancel of a delivered one).
   - Out-of-stock exchange targets.
   - A user who insists after a refusal.
   - The name+zip-only user, which is already a strong signal source; consider making it more common.

## What still holds

Even after discounting the artifacts, the episode format clearly produces more within-group variance than the single-step corpus did (24% effective; `diagnosis.txt`). The genuine failure modes are the behaviours tau2 retail punishes:
- skipping the refund rule;
- hallucinating ids;
- claiming success without a tool call;
- mishandling corrections;
- refusing name+zip authentication.

That supports `docs/variance_audit.md` step 3 (train mainly on episodes) once fixes 1-4 land and a re-audit confirms the p spread.
