# Phase 7 AWS setup — GPU box for the real GRPO run

**Status: infra prep only.** This document and the scripts next to it get an EC2
box ready to run Phase 7. They do not start Phase 7 itself (no training code has
been written yet, and per the phase-status table Phase 7 still needs explicit
go-ahead before it begins) — this is groundwork so that when the go-ahead comes,
launching the box is a five-minute, well-reasoned decision instead of a scramble.

Nothing here provisions anything automatically. Launching an actual instance is a
real-money, real-AWS-account action and should be a deliberate step you take
(console or CLI), not something run unattended.

## What Phase 7 actually needs to do

Per the held-out data policy and the phase table: **one policy model** (`Qwen3-4B-Instruct-2507`)
trained with **GRPO** against the reward function from Phase 4, sampling rollouts
against the Phase 2 synthetic pilot/sweep data (541 scenarios, single-decision-point
episodes — one user turn + optional `prior_turns` context, then one tool call or
message as the graded action). That "single decision point" framing matters a lot
for sizing: this is not a long-horizon multi-turn RL run with 4-8k-token episodes:
prompts are short (a rendered scenario + tool schemas), and generations are short
(one tool call or one short message). Sequence length in the hundreds to low
thousands of tokens, not tens of thousands.

## Sizing: FLOPs

For a dense N-parameter transformer, the standard estimator is:
- Forward pass: ~2N FLOPs/token
- Forward + backward (training): ~6N FLOPs/token

`Qwen3-4B-Instruct-2507` is dense (not MoE), N ≈ 4.02B, so:
- Rollout generation (forward only): ~8 GFLOPs/token
- Policy-gradient update (fwd+bwd): ~24 GFLOPs/token

541 scenarios × a handful of epochs × a GRPO group size of, say, 8-16 samples/prompt
puts total tokens processed in the tens of millions, not billions — total training
compute for this run is on the order of a few PFLOP-days at most, which is trivial
for a single modern GPU (an L4 alone does ~121 TFLOPS bf16 dense). **FLOPs are not
the constraint here.** The real constraints are (a) GPU memory to hold everything
GRPO needs simultaneously, and (b) generation being memory-bandwidth-bound /
latency-bound (autoregressive decoding), not compute-bound — so a GPU with strong
memory bandwidth matters more than one with the highest peak FLOPs.

## Sizing: memory

GRPO needs, live on the GPU(s) at once: the trainable policy, a frozen reference
policy (for the KL term), optimizer state, gradients, activations, and a rollout
generation engine (KV cache) sampling a group of completions per prompt.

**This is RLVR, so this is a full-parameter fine-tune, not LoRA.** LoRA restricts
updates to a low-rank subspace, which can bottleneck exactly the kind of policy
shift RLVR's (often sparse/binary, verifiable) reward signal is trying to induce —
that's a real cost, not just a memory-saving convenience, and not one worth trading
away just because it makes the memory story easier. Full-parameter it is:

**Full fine-tune, bf16 weights + fp32 AdamW (mixed precision):**

| Component | Size |
|---|---|
| Policy weights (bf16) | 8 GB |
| Reference weights (bf16, frozen copy — needed for the KL term; not free like LoRA's "disable the adapter" trick, but it's just one static forward-only copy) | 8 GB |
| Gradients (bf16) | 8 GB |
| AdamW optimizer state (fp32 m + v) | 32 GB |
| FP32 master weights (mixed precision) | 16 GB |
| **Subtotal, before activations/KV cache** | **~72 GB** |

That's already past a single-GPU card's memory before a single activation or
KV-cache byte — and AWS doesn't sell a single GPU bigger than the L40S's 48 GB (no
standalone A100/H100 instance exists; `p4d`/`p4de`/`p5` are fixed 8-GPU boxes). So
this genuinely needs sharding across a few GPUs, not because of exotic scale, just
because ~72 GB doesn't fit on one card.

**ZeRO-2 (optimizer + gradient sharding, weights replicated) across a handful of
GPUs is the right tool here** — much simpler than full ZeRO-3/FSDP parameter
sharding (which is built for models that don't fit *replicated* either; a 4B model
easily does), and it's a standard, well-supported `accelerate`/`deepspeed` config,
not custom infra. Sharding the 32 GB optimizer state + 8 GB gradients (40 GB total)
across **4 GPUs** leaves ~10 GB/GPU for that, plus the replicated 16 GB
(policy + reference weights) on every GPU → **~26 GB/GPU** before activations/KV
cache, comfortably inside a 48 GB L40S with room to spare for rollout generation.

## Recommendation

| Instance | GPUs | VRAM (total) | Mem bandwidth/GPU | vCPU / RAM | ~On-demand (us-east-1)* | Fit |
|---|---|---|---|---|---|---|
| **g6e.12xlarge** (primary) | 4x L40S | 192 GB | 864 GB/s | 48 / 384 GB | ~$14/hr | Full-FT GRPO via ZeRO-2, ~26 GB/GPU used, plenty of headroom for rollout batch/group size |
| g6e.2xlarge (only if deliberately going back to LoRA) | 1x L40S | 48 GB | 864 GB/s | 8 / 64 GB | ~$2.5/hr | LoRA-only fallback, not the RLVR default — see note above |
| p4d.24xlarge | 8x A100 40GB | 320 GB | 1555 GB/s | 96 / 1152 GB | ~$32/hr | Works too, but 8 GPUs is more than this 4B/541-scenario run needs — mostly idle capacity, not idle-cost you're avoiding by ignoring $/hr, but wasted throughput/GPU-hours nonetheless |

*Prices are ballpark and change — confirm current on-demand/spot pricing at launch
time via the EC2 console or `aws ec2 describe-spot-price-history`.

**Pick `g6e.12xlarge`.** Reasoning:
- Full-parameter GRPO's ~72 GB working set needs sharding across GPUs no matter
  what — this isn't about buying more speed, it's the minimum needed to do
  full-parameter RLVR at all with a single-GPU cap of 48 GB.
- ZeRO-2 across 4 GPUs is the smallest/simplest sharding config that clears that
  bar with real headroom (~26 GB/GPU of ~48 GB used), not the maximal one — no
  need for full parameter sharding (ZeRO-3/FSDP) when replicated weights already
  fit easily.
- L40S's high memory bandwidth (864 GB/s/GPU) still helps the bandwidth-bound
  rollout-generation step, same reasoning as before, now with 4 GPUs available to
  parallelize rollout sampling across too (e.g. vLLM tensor-parallel or just
  round-robining prompts across GPUs).
- Stops short of `p4d.24xlarge`'s 8 GPUs, which would mostly sit idle for a model
  and dataset this size — "reasonable but fast" reads as "enough headroom to move
  quickly, not paying for capacity that can't be used."

## Timing estimate

Compute isn't the bottleneck for this run (see FLOPs section above) — wall-clock
time scales roughly linearly with **total rollouts** `R = 541 prompts × epochs ×
GRPO group size`. Per-rollout time has two dominant pieces (generation, training),
plus an overhead multiplier for reward computation, generation↔training weight
sync, logging, and checkpointing:

```
T ≈ R × [ L_completion / TP_gen  +  (L_prompt + L_completion) × 32 GFLOPs / TP_train ] × overhead
```

using `L_completion` ≈ 150 tok, `L_prompt` ≈ 1000 tok, `TP_gen` (aggregate decode
throughput on 4x L40S) 4,000–12,000 tok/s, `TP_train` (aggregate sustained training
FLOPs, ZeRO-2 comm overhead included) 150–300 TFLOPS, and 32 GFLOPs/token = 24
(policy fwd+bwd) + 8 (reference-model forward, for the KL term). Overhead
multiplier: 1.15x (optimistic) – 1.5x (conservative).

That works out to **~0.15–0.4 sec/rollout** (~3–7 rollouts/sec) on `g6e.12xlarge`:

| Total rollouts R | Optimistic | Conservative |
|---|---|---|
| 5,000 | ~13 min | ~35 min |
| 17,312 *(541 × 4 epochs × group 8)* | ~45 min | ~2.0 hr |
| 30,000 | ~1.3 hr | ~3.5 hr |
| 50,000 | ~2.2 hr | ~5.9 hr |
| 100,000 | ~4.3 hr | ~11.8 hr |

`R = 541 × epochs × group_size` for whatever hyperparameters end up chosen.

**Weakest parts of this estimate** — not modeled precisely, and the reason to treat
the table as a planning number, not a commitment:
- Reward computation (`RetailEnv` deep-copy + tool exec + diff, per rollout, on
  CPU) — folded into the overhead multiplier, but could dominate if not
  parallelized across the box's ~48 vCPUs.
- Generation↔training weight-sync cost — a known real bottleneck in RL
  post-training pipelines, depends on implementation (in-process weight sharing
  vs. reload-from-disk) that doesn't exist yet.
- All throughput numbers are first-principles, not measured on this model/box.

The smoke test below (50-100 steps) replaces this estimate with a real measured
seconds/rollout number — plug that into the same formula for a trustworthy
full-run estimate instead of the first-principles range.

## Methodology risks — and why full-parameter GRPO needs more than just the box

Getting the instance right doesn't by itself mean the training run works well.
Three risks are specific to this project's data/setup, not to the infra, and are
worth deciding on *before* spending GPU-hours on the full run:

1. **GRPO's signal needs within-group reward variance — untested here.** GRPO
   normalizes advantage by each group's own mean/std across its `G` sampled
   completions. If a scenario is trivially easy (base model gets it right every
   time) or too hard/ambiguous (consistently wrong for reasons unrelated to
   sampling), every sample in the group gets the same reward and that prompt
   contributes ~zero gradient. **Phase 3 stage 4 (difficulty calibration) was never
   done** — nobody has measured how hard the 541 scenarios are for
   `Qwen3-4B-Instruct-2507` zero-shot. Running the full training job before knowing
   this risks a lot of compute going into degenerate, zero-signal groups.
2. **Full-parameter capacity + a small, static, repeatedly-seen prompt set is real
   overfitting/reward-hacking exposure.** 541 unique prompts seen across multiple
   epochs, full-parameter updates (much more capacity to memorize or exploit than
   LoRA would allow), against an engineered reward function that already has one
   documented edge case (`transfer_to_human_agents`'s constant return value, fixed
   in Phase 4 but a sign more exist). Mitigate with a meaningfully tuned KL penalty
   against the frozen reference, a small learning rate (RLVR full fine-tunes
   typically want `1e-6`–`5e-6`, well below supervised-FT LRs), held-out-eval-based
   checkpoint selection rather than last-step, and watching the reward-tier
   distribution (Phase 4's 0 / 0.2 / 0.3-1.0 tiers) over training for a shift toward
   suspiciously uniform near-miss behavior.
3. **Train/eval mismatch already flagged in the README.** Every synthetic scenario
   is a single decision point graded in isolation; real τ²-bench tasks average 4.8
   sequential tool calls per conversation (Phase 2 section). Training here is
   closer to a contextual bandit than long-horizon RL — good for GRPO stability
   (no credit-assignment problem), but it means strong training reward doesn't
   guarantee proportional Phase 8 live-benchmark improvement, since Phase 8 chains
   errors across multiple sequential actions that this training never rehearses
   together.

**Concrete recommended approach, in order:**
1. Zero-shot base-model pass over all 541 scenarios (inference only, no training)
   — gives the missing difficulty signal and a pre-training baseline.
2. The Phase 6-flagged 50-100 step GRPO smoke test — confirms the
   generation→training→reward loop works end-to-end and gives a real
   seconds/rollout number (see Timing estimate above) before committing to a full
   run blind.
3. Tuned KL coefficient, small LR with warmup, GRPO group size ~16 (better
   advantage estimate given how few unique prompts exist).
4. Held-out eval during training, checkpoint by best held-out score rather than
   final step.
5. Track tier-distribution / generation-length / tool-diversity metrics during
   training as cheap reward-hacking tripwires.

None of this changes the instance pick — `g6e.12xlarge` runs whichever of these
gets decided. It does mean "will it work well" is a methodology question the
hardware can't answer by itself, and steps 1-2 above are cheap enough to be the
actual first thing run on the box, not something skipped on the way to the full
job.

## AMI and software

Use an AWS Deep Learning AMI (Ubuntu, GPU PyTorch variant) so CUDA/cuDNN/NVIDIA
drivers are preinstalled and version-matched — don't hand-install drivers on a
bare Ubuntu AMI. Everything above that (uv, the training stack) is handled by
`infra/ec2_bootstrap.sh` in this directory.

Training dependencies (`torch`, `trl`, `deepspeed`, `vllm`, `accelerate`,
`bitsandbytes`) are declared as an optional `train` dependency group in
`pyproject.toml` — not part of the default `uv sync`, since this repo is developed
in GPU-less environments and those packages need a matching CUDA toolchain to even
install cleanly. On the GPU box: `uv sync --extra train`. `deepspeed` is included
for the ZeRO-2 sharding the full-parameter run needs (see sizing above); `peft`
is dropped from the default set since this is a full-FT run, not LoRA — worth
re-adding only if you deliberately switch back.

## Security / cost-safety checklist (do before launching)

- [ ] Security group: inbound SSH (22) restricted to your IP only, no other
      inbound ports open (no public HTTP/Jupyter exposure).
- [ ] Use an existing key pair or generate one for this box specifically; don't
      reuse a shared/prod key.
- [ ] Set a billing alarm (e.g. CloudWatch billing alarm) before launching —
      this is a GPU instance and idle time costs real money.
- [ ] Prefer launching via a script/CLI you can re-run rather than manual console
      clicks, so the exact config is reproducible and diffable.
- [ ] **Stop or terminate the instance when not actively training** — even with
      credits to spend, an idle GPU instance left running overnight is pure waste.
- [ ] EBS volume sized for base model weights (~8 GB bf16) + full-parameter
      checkpoints (~8 GB bf16 each, or ~24 GB if saving optimizer state for
      resumability) + vLLM/HF caches — 300 GB gp3 is a safer starting point than
      a LoRA-sized volume, especially if keeping more than one checkpoint.
- [ ] Confirm the held-out data policy before any live run: only Phase 2 synthetic
      data, never the 114 real τ²-bench retail tasks, touches model weights.

## Next steps (not done by this doc)

1. Review this sizing/instance choice — confirm `g6e.12xlarge` (or an alternative
   above) is the pick.
2. Actually launch the instance (console or CLI) once ready — this is the step
   that costs money and should be a deliberate, explicit action.
3. Run `infra/ec2_bootstrap.sh` on the box to get the repo + training deps
   installed.
4. Zero-shot base-model pass over all 541 scenarios (difficulty signal +
   baseline) — cheap, inference-only, do this before writing a line of training
   code.
5. The 50-100 step GRPO smoke test (Phase 6 spec) — validates the loop end-to-end
   and produces a real seconds/rollout number to replace the Timing estimate
   above.
6. Only then does Phase 7's full run itself (the full-parameter GRPO training
   script with a ZeRO-2 `accelerate`/`deepspeed` config, wiring it to
   `tau_forge.reward`/`tau_forge.harness`/`tau_forge.envs`) start — still gated on
   its own explicit go-ahead per the phase-status table.

## Why finish this setup

Every phase after the current one is gated on an explicit go-ahead (README,
"Phase status") — Phase 7 has been sitting as **"Not started — needs a GPU box,
none available in this environment"** since Phase 6 passed. That's not a soft
blocker: Phases 0, 1, 4, and 6 are all done and green (env wrapper, reward
function adversarially tested, harness validated against all 74 real trusted
tasks scoring a perfect 1.0), and Phase 2/3-stage-1 produced 541 rule-checker-clean
synthetic scenarios — the entire project is built and validated up to the one step
that actually produces a trained model. Phase 8 (the real evaluation against
τ²-bench retail, airline zero-shot, and BFCL v3 — the actual point of this repo)
can't start until Phase 7 has a checkpoint to evaluate. Finishing this setup is the
difference between "everything is ready and waiting" and "everything is ready
except the one step nothing downstream can happen without."

What "finishing this setup" concretely unblocks, once you give the go-ahead:
launch `g6e.12xlarge` → run `infra/ec2_bootstrap.sh` → run the zero-shot pass and
smoke test above to de-risk the full run methodologically (not just
infra-wise) → write and run the actual full-parameter GRPO training script → get a
checkpoint → Phase 8 evaluates it. Every piece up to "write and run the training
script" is either already done (this doc/scripts) or a cheap, fast diagnostic step
(zero-shot pass, smoke test) — the setup work here removes the excuse for Phase 7
to keep sitting idle once you're ready to say go.
