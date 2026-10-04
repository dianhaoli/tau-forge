"""Multi-step episode tasks: programmatic, end-state-graded retail conversations.

Why this package exists. The single-decision scenarios in `data/synthetic/raw`
ask the policy for ONE next action. That action is close to argmax for a
4B instruct model -- p(success) sits near 0 or near 1 -- so a GRPO group of 16
samples is flat with probability p^16 + (1-p)^16 (0.44 at p=0.05 or 0.95), and
the zero-shot n=16 audit measured ~72.5% zero-variance scenarios. A chain of k
decisions with per-step reliability r succeeds with p ~ r^k, which for r=0.9
and k=6-8 lands at 0.43-0.53, where a flat group is ~1e-5 likely. Episodes
also remove the corpus-level defects the audit found (unreachable gold ids,
ungraded refusal text, missing auth/confirmation) by construction: every id
the policy needs comes from its own tool outputs, and grading is the tau2
end-state hash, not text matching.

Modules:
  * `task`     -- the `EpisodeTask` record, shared tool sets, the cached base db.
  * `generate` -- templates sampled from db.json only, verification, dedupe,
                  decontamination against the 114 real tasks.
  * `user`     -- the deterministic scripted user.
  * `runner`   -- the agent <-> user <-> RetailEnv loop (copy-on-write db).
  * `reward`   -- end-state success, policy gates, capped failure shaping.
  * `reference_agents` -- scripted oracle and near-miss policies used by the
                  tests and the CPU dry run of `scripts/episode_audit.py`.
  * `audit`    -- the batched multi-turn variance-audit loop, generator-agnostic
                  (vLLM is wired in by `scripts/episode_audit.py`).

Not done yet: TRL `rollout_func` integration (see README, "Multi-step episode
tasks"). Nothing here is wired into `grpo_train`.
"""
