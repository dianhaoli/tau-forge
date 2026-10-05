# Episode variance audit (Step 4b, reduced)

Model Qwen/Qwen3-4B-Instruct-2507, 250 tasks x 8 samples = 2000 episodes; temperature 1.0, top_p 1.0, top_k 0 (disabled), max_new_tokens 1024, max_model_len 16384, max_turns 30. Tasks: 50 per template drawn with seed 0 from `episodes_s0.jsonl` (`generate --per-template 200 --seed 0`; modify_payment exhausted at 85). Reduced from 885x16 because the full run ran out of host RAM (15 GB, no swap) on this A10G box. Wall-clock 3.4 h. vLLM run with `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc on the box).

**Overall: success p = 0.574, mean reward 0.554, effective variance (group std >= 0.05) 81.2%, flat 13.2%.** Single-step comparison (same day, n=16): 24.4% effective overall, 20.7% on clean scenarios.

End reasons: {'user_stop': 1594, 'max_turns': 376, 'transfer': 29, 'context_budget': 1}. **376 episodes (19%) hit the 30-turn cap**; turn-by-turn active counts sat at ~376 from turn 25 to 30, so these are loops, not slow progress. See the transcript audit before trusting the variance numbers.

## Per template

| template | tasks | p | mean reward | eff. var | flat | p excl. max_turns | max_turns | transfer | tasks at k=0 / k=8 (of 8) |
|---|---|---|---|---|---|---|---|---|---|
| cancel | 50 | 0.593 | 0.587 | 94% | 4% | 0.762 | 89 | 7 | 3 / 6 |
| exchange | 50 | 0.145 | 0.233 | 74% | 0% | 0.172 | 63 | 5 | 20 / 0 |
| foreign_order_refusal | 50 | 0.795 | 0.804 | 54% | 46% | 0.914 | 52 | 3 | 0 / 23 |
| modify_payment | 50 | 0.703 | 0.609 | 92% | 8% | 0.930 | 98 | 8 | 0 / 8 |
| return_fallback | 50 | 0.635 | 0.538 | 92% | 8% | 0.779 | 74 | 6 | 2 / 10 |

## Per difficulty knob (p, effective variance, tasks)

**cancel**

| knob | value | tasks | p | eff. var | max_turns eps |
|---|---|---|---|---|---|
| give_order_id | False | 38 | 0.58 | 95% | 76 |
| give_order_id | True | 12 | 0.64 | 92% | 13 |
| id_mode | email | 23 | 0.60 | 91% | 36 |
| id_mode | name_zip | 27 | 0.59 | 96% | 53 |
| identity_upfront | False | 28 | 0.63 | 93% | 39 |
| identity_upfront | True | 22 | 0.54 | 95% | 50 |
| late_correction | False | 33 | 0.68 | 94% | 56 |
| late_correction | True | 17 | 0.43 | 94% | 33 |
| reason_upfront | False | 23 | 0.44 | 96% | 52 |
| reason_upfront | True | 27 | 0.72 | 93% | 37 |

**exchange**

| knob | value | tasks | p | eff. var | max_turns eps |
|---|---|---|---|---|---|
| give_order_id | False | 29 | 0.12 | 79% | 40 |
| give_order_id | True | 21 | 0.18 | 67% | 23 |
| id_mode | email | 24 | 0.19 | 79% | 21 |
| id_mode | name_zip | 26 | 0.11 | 69% | 42 |
| identity_upfront | False | 23 | 0.11 | 57% | 21 |
| identity_upfront | True | 27 | 0.17 | 89% | 42 |
| late_correction | False | 29 | 0.22 | 90% | 41 |
| late_correction | True | 21 | 0.05 | 52% | 22 |
| n_items | 1 | 18 | 0.16 | 78% | 18 |
| n_items | 2 | 17 | 0.11 | 71% | 28 |
| n_items | 3 | 15 | 0.17 | 73% | 17 |

**foreign_order_refusal**

| knob | value | tasks | p | eff. var | max_turns eps |
|---|---|---|---|---|---|
| id_mode | email | 24 | 0.87 | 46% | 13 |
| id_mode | name_zip | 26 | 0.73 | 62% | 39 |
| identity_upfront | False | 26 | 0.88 | 38% | 5 |
| identity_upfront | True | 24 | 0.71 | 71% | 47 |
| request | cancel | 29 | 0.68 | 79% | 44 |
| request | return | 21 | 0.95 | 19% | 8 |

**modify_payment**

| knob | value | tasks | p | eff. var | max_turns eps |
|---|---|---|---|---|---|
| asked_source | credit_card | 27 | 0.78 | 85% | 36 |
| asked_source | gift_card | 5 | 0.65 | 100% | 8 |
| asked_source | paypal | 18 | 0.60 | 100% | 54 |
| gift_card_short | False | 45 | 0.71 | 91% | 90 |
| gift_card_short | True | 5 | 0.65 | 100% | 8 |
| give_order_id | False | 19 | 0.64 | 95% | 43 |
| give_order_id | True | 31 | 0.74 | 90% | 55 |
| id_mode | email | 22 | 0.74 | 95% | 38 |
| id_mode | name_zip | 28 | 0.67 | 89% | 60 |
| identity_upfront | False | 32 | 0.76 | 91% | 47 |
| identity_upfront | True | 18 | 0.60 | 94% | 51 |

**return_fallback**

| knob | value | tasks | p | eff. var | max_turns eps |
|---|---|---|---|---|---|
| give_order_id | False | 27 | 0.44 | 93% | 53 |
| give_order_id | True | 23 | 0.86 | 91% | 21 |
| has_gift_card | False | 44 | 0.64 | 91% | 61 |
| has_gift_card | True | 6 | 0.56 | 100% | 13 |
| id_mode | email | 23 | 0.61 | 91% | 28 |
| id_mode | name_zip | 27 | 0.66 | 93% | 46 |
| identity_upfront | False | 28 | 0.68 | 93% | 29 |
| identity_upfront | True | 22 | 0.57 | 91% | 45 |
| n_items | 1 | 23 | 0.71 | 91% | 32 |
| n_items | 2 | 27 | 0.57 | 93% | 42 |

Note: effective variance and flat are computed on the episode *reward*, which has partial credit, not on binary success. That's why exchange shows 0% flat while 20 of its 50 tasks have k=0 successes: their rewards still differ within the group.

**Caveat (transcript audit):** these p values are distorted by environment bugs, chiefly false-positive refusals in foreign_order_refusal and the literal-"?" recap rule. Read `episode_transcript_audit.md` before using them.
