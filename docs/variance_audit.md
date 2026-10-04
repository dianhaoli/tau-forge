# Why GRPO groups had no variance, and what changed

A zero-shot n=16 audit (temperature 1.0) found ~72.5% of the 541 synthetic
scenarios zero-variance: all 16 samples got the same reward, so GRPO's advantage
is zero and the scenario contributes no gradient. The raw audit JSON was not in
the repo, so this diagnosis was done from code and data on CPU. Every scenario
was read in full, every claim was reproduced with a script, and the findings
were re-checked by an adversarial verifier. The per-scenario results are in
`data/synthetic/label_audit.json`.

The causes sit in three layers. Each layer alone is enough to flatten groups.

## 1. Environment / data: the correct answer was unreachable or against policy

| Defect | Scenarios | What a policy-following model does | Score it gets |
|---|---|---|---|
| Gold id never in the prompt (prior lookups narrated as prose, ids dropped) | 103 of 150 write golds (+20 reads) | guesses the id, or looks it up | flat 0.2 (call raises) / flat 0.0 (`wrong_tool`) |
| Write gold with no explicit "yes" from the user (`policy.md`: confirm before any DB write) | 72 of 150 writes (+26 ambiguous) | lists the action, asks for yes | 0.0 (`missing_call`) |
| Write gold for a user who never authenticated | 55 of 150 writes | asks for email or name+zip | 0.0 |
| out_of_scope transfer where the policy says deny (31) or just answer (20); contradicts policy_violation labels for the same rule | 51 of 107 | refuses or answers | 0.0 |
| out_of_scope with no authentication | 50 of 107 | authenticates first | 0.0 |
| Read gold with no authentication | 26 of 102 | authenticates first | 0.0 |
| Training prompt said "don't ask clarifying questions, transfer instead" (unscoped) | 104 ambiguous and the policy_violation golds | either one: transfer or ask | one category flat 0 |

Union: **222 of 541 scenarios have a gold under which the policy-correct action
scores 0** (the `blocking` list in `label_audit.json`). That is a flat group,
and it is also signal pointing away from the benchmark's policy.

**Fixed:**
- `tau_forge/train/grounding.py` replays the lookups the prose narrates against
  `db.json` and inserts them as real assistant `tool_calls` + `tool` messages,
  the shape tau2 feeds the agent at eval. Unreachable write golds: 103 -> 51
  (the rest are mostly unauthenticated golds, which grounding deliberately does
  not authenticate on the scenario's behalf).
- `grpo_train` drops label-defect scenarios by default
  (`--keep-label-defects` to opt out). `zero_shot_baseline --split train|val`
  applies the same exclusions, so its val split matches the trainer's.
- The escalation instruction is scoped to clear, permitted requests no tool can
  do.

**Consequence you need to know:** after exclusions, 319 scenarios remain, and
56.1% of them are no-call golds (it was 33.6%). With `--category-mix real` only
about 112 remain. The cleaned single-step corpus is small and skewed. That is
the main argument for section 3.

## 2. Grader: different behaviours got identical scores

| Issue | Effect | Fix |
|---|---|---|
| `transfer_to_human_agents` graded by `SequenceMatcher` similarity to the generator's summary prose. Reward is `max(0.3, ratio)` with autojunk on | an empty summary scored the same as a good paraphrase (0.300); the gold's wording with wrong content scored 0.84; 46/107 gold summaries name ids the model never saw | grade the decision: right tool with a non-trivial summary = 1.0, empty or trivial = 0.6, gold prose never read (tau2 does not read it either) |
| No-call golds (182) were content-blind | an empty completion scored 1.0 | empty or whitespace-only now scores 0.0 |
| A bare tool-call JSON without `<tool_call>` tags parsed as a reply | a free 1.0 on every no-call gold | now treated as a malformed call |
| Shaping gave right-record credit to a premature write on policy_violation status-check golds | 0.13 for the exact violation, more than a legitimate read (0.10) | writes get no record credit when gold is not a write |
| `inspect_stuck.py` re-graded completions without the raw text | diagnostics disagreed with training | single `reward_adapter.grade_completion` path |

Trivial-policy corpus means (reward / reward+shaping), before -> after:

| Policy | Before | After |
|---|---|---|
| empty string | 0.336 / 0.336 | 0.000 / 0.000 |
| bare untagged JSON call | 0.336 / 0.336 | 0.000 / 0.000 |
| always transfer, good summary | 0.068 / 0.109 | 0.198 / 0.239 |

**Still open:** a non-empty reply on a no-call gold still scores 1.0 whatever
it says (a refusal, a clarifying question, or a hallucinated "Done!"). Grading
reply content needs either an intent classifier or the episode format, where
the end state does the grading.

## 3. Task design: one near-argmax decision per prompt

Every scenario asks for one next action. For a 4B instruct model, "call or
not, and which tool" is close to argmax, so p(success) sits near 0 or 1. A
binary group of 16 is flat with probability p^16 + (1-p)^16: 0.44 at p = 0.05
or 0.95, and 0.003 at p = 0.3. Raising the temperature or the group size does
not move p off the extremes, and fixing labels and the grader removes wrong
gradients without creating variance.

**Added:** `tau_forge/episodes/` adds multi-step episode tasks generated
programmatically from `db.json`. They have:
- a deterministic scripted user that says yes only to a recap naming the
  target and action;
- a live env, with tool results fed back to the model;
- tau2-style end-state reward with policy gates for confirmation and
  authentication.

The five templates are cancel, exchange, return_fallback, modify_payment and
foreign_order_refusal. Their difficulty knobs (order id vs product hint, email
vs name+zip, late correction, 1-3 items, gift-card shortfall) are the dials for
calibrating per-cell p into 0.15-0.85. There are decontamination hooks against
the 114 real tasks.

Episodes average 4.2-6.4 gold calls, matching tau2's ~4.8. Reference agents
for oracle, no-confirm, skip-auth, wrong-variant and comply land on distinct
reward levels (README, "Multi-step episode tasks"). Design rationale:
`docs/episode_tasks_design.md`.

## 4. Sampling / harness

- `grpo_train` passed `top_k=None`. The HF path then silently fell back to
  Qwen's shipped `top_k=20`, and the vLLM path raised. The trainer sampled a
  narrower distribution than the audit measured. It is now explicit 0
  (disabled).
- `--exclude-zero-variance-from` read raw scores, so it dropped every group
  that shaping revives. It now uses shaped scores, plus an
  effective-variance threshold (`--min-std`, default 0.05). Shaping-only
  micro-differences of 0.02 no longer count as live groups. The audit,
  bucket analysis and scorecard report the effective number next to the old
  one.

## What to run next (GPU box)

1. **Re-audit the single-step corpus** with the new prompt, grounding and grader:
   `zero_shot_baseline --use-vllm --samples-per-scenario 16 --with-shaping --save-completions`
   (default `--max-model-len` is now 12288 for the grounded prompts).
   The 72.5% figure predates every fix above, so it no longer describes the
   corpus. Compare clean vs defective scenarios in `scripts/diagnose.sh`'s
   report.
2. **Run the episode audit**
   (`python -m tau_forge.episodes.generate ...` then `scripts/episode_audit.py`,
   see the README). It gives per-template and per-knob p. Weight cells by
   p(1-p).
3. Only if step 2 shows a healthy p spread, **implement the TRL `rollout_func`
   integration** (`docs/episode_tasks_design.md` section 2.7) and train mainly
   on episodes, with the cleaned single-step set as a minority auxiliary mix.

Unverified here, because huggingface.co is blocked from this environment:
whether Qwen3's `<tool_call>` token is non-special (all evidence says yes), and
vLLM's hermes parser behaviour, which `completion_parsing` mirrors. Check both
on the box with
`tok.decode(tok.encode('<tool_call>'), skip_special_tokens=True)`.
