# Deferred scenarios

Not loaded by `tau_forge.train.dataset.DEFAULT_DATA_GLOB` (only `data/synthetic/raw/*.json`
is) -- kept here for reference, not part of the active 541-scenario training set.

## `out_of_scope_multiturn_coldstart.json` (3 scenarios)

Hand-authored to test whether GRPO training on our `out_of_scope` scenarios --
all single-step, zero-lookup escalations -- would generalize to the more
common real tau2-bench pattern (tasks 10/12/26): a full normal workflow
first, then escalating only the one sub-request no tool can satisfy.

Zero-shot baselined twice at n=16 (Qwen3-4B-Instruct-2507, temperature 1.0):
first pass conflated the result with an authoring bug (a missing item id
forced hallucination on every wrong-tool attempt); after fixing that, still
0/48 across all three, 100% zero-variance both times. Diagnosis: the model
copies the `payment_method_id` from an already-resolved action narrated in
`prior_turns` onto the new request, checking "is this a real payment method
on the account" instead of "is this the *original* payment method of *this
order*" -- a precondition-scoping error, not simply "doesn't know to
escalate." Not yet tested with reasoning space before the tool call, which
may or may not be the same failure once the model can deliberate first.

Excluded from this training run: zero variance means zero GRPO signal
regardless, and 3 examples is too few to safely include even if a future
rerun surfaces occasional successes (real risk of memorizing their specific
surface details rather than the general skill). Candidate material for a
future SFT warm-start pass or an expanded (10+) synthetic batch of the same
shape, not for GRPO as-is.
