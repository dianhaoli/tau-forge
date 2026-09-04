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

GRPO needs, live on the GPU at once: the trainable policy, a frozen reference
policy (for the KL term), optimizer state, gradients, activations, and a rollout
generation engine (KV cache) sampling a group of completions per prompt.

**Full fine-tune, bf16 weights + fp32 AdamW (mixed precision):**

| Component | Size |
|---|---|
| Policy weights (bf16) | 8 GB |
| Reference weights (bf16, frozen copy) | 8 GB |
| Gradients (bf16) | 8 GB |
| AdamW optimizer state (fp32 m + v) | 32 GB |
| FP32 master weights (mixed precision) | 16 GB |
| **Subtotal, before activations/KV cache** | **~72 GB** |

That's already past a single 48 GB card before a single activation or KV-cache
byte, and would need multi-GPU (FSDP/DeepSpeed ZeRO-3 sharding) to fit — a lot of
infra complexity and cost for a 4B model trained on 541 short scenarios.

**LoRA/PEFT instead — the right call for this run's scale:**
- Base weights frozen, bf16: 8 GB (one copy serves as both the generation/rollout
  model *and* the reference model — disable the adapter to get the reference
  policy's logits, no second full copy needed)
- LoRA adapter (rank 16-32 on attention + MLP projections): tens of millions of
  trainable params → optimizer state + gradients for the adapter alone, well under
  1 GB
- Activations (with gradient checkpointing) and rollout-generation KV cache for a
  GRPO group (e.g. 8-16 completions/prompt, short sequences): a few GB, scales with
  batch size and group size, tunable to fit whatever card is used

Total working set comfortably fits in **16-24 GB**, with room to spare for KV cache
and batch size even on a 24 GB card. This is the approach to use: it also sidesteps
the multi-GPU sharding complexity entirely, which matters a lot given this project
has no existing multi-GPU training infra.

## Recommendation

| Instance | GPU | VRAM | vCPU / RAM | ~On-demand (us-east-1)* | Fit |
|---|---|---|---|---|---|
| **g6.2xlarge** (primary) | 1x L4 | 24 GB | 8 / 32 GB | ~$0.98/hr | LoRA GRPO fits with headroom; best $/hr for this workload |
| g5.2xlarge (fallback) | 1x A10G | 24 GB | 8 / 32 GB | ~$1.21/hr | Same fit as above; use if g6 capacity/region is an issue |
| g6e.2xlarge (if more headroom wanted) | 1x L40S | 48 GB | 8 / 64 GB | ~$2.5/hr | Room for larger GRPO group size / longer context without tuning |
| p4d.24xlarge | 8x A100 40GB | 320 GB | 96 / 1152 GB | ~$32/hr | Overkill — only justified for full fine-tune or a much bigger run |

*Prices are ballpark and change — confirm current on-demand/spot pricing at launch
time via the EC2 console or `aws ec2 describe-spot-price-history`.

**Pick `g6.2xlarge`.** Reasoning:
- FLOPs aren't the bottleneck (see above), so paying for an A100/H100's raw compute
  buys nothing this run will use.
- 24 GB comfortably fits LoRA GRPO for a 4B model with short episodes, per the
  memory table above.
- L4 has good memory bandwidth per dollar, which matters for the
  memory-bandwidth-bound rollout-generation step.
- Single GPU avoids all multi-GPU orchestration (FSDP/DeepSpeed/NCCL) complexity,
  which this project has zero existing infra for — not worth introducing for a
  541-scenario run.
- Spot pricing on g6.2xlarge is typically 50-70% off on-demand and this workload
  (checkpointed training, not latency-sensitive serving) tolerates interruption
  fine if checkpointing is wired up — worth using once the run is validated on
  on-demand first.

If a first run turns out to need more headroom (bigger GRPO group size, longer
`prior_turns` context than expected), step up to `g6e.2xlarge` (L40S, 48 GB) rather
than jumping straight to multi-GPU — still single-GPU, just more room.

## AMI and software

Use an AWS Deep Learning AMI (Ubuntu, GPU PyTorch variant) so CUDA/cuDNN/NVIDIA
drivers are preinstalled and version-matched — don't hand-install drivers on a
bare Ubuntu AMI. Everything above that (uv, the training stack) is handled by
`infra/ec2_bootstrap.sh` in this directory.

Training dependencies (`torch`, `trl`, `peft`, `vllm`, `accelerate`,
`bitsandbytes`) are declared as an optional `train` dependency group in
`pyproject.toml` — not part of the default `uv sync`, since this repo is developed
in GPU-less environments and those packages need a matching CUDA toolchain to even
install cleanly. On the GPU box: `uv sync --extra train`.

## Security / cost-safety checklist (do before launching)

- [ ] Security group: inbound SSH (22) restricted to your IP only, no other
      inbound ports open (no public HTTP/Jupyter exposure).
- [ ] Use an existing key pair or generate one for this box specifically; don't
      reuse a shared/prod key.
- [ ] Set a billing alarm (e.g. CloudWatch billing alarm) before launching —
      this is a GPU instance and idle time costs real money.
- [ ] Prefer launching via a script/CLI you can re-run rather than manual console
      clicks, so the exact config is reproducible and diffable.
- [ ] **Stop or terminate the instance when not actively training** — a `g6.2xlarge`
      left running idle overnight is a real, avoidable cost.
- [ ] EBS volume sized for base model weights (~8 GB bf16) + checkpoints +
      vLLM/HF caches — 100 GB gp3 is a safe starting point.
- [ ] Confirm the held-out data policy before any live run: only Phase 2 synthetic
      data, never the 114 real τ²-bench retail tasks, touches model weights.

## Next steps (not done by this doc)

1. Review this sizing/instance choice — confirm `g6.2xlarge` (or the fallback) is
   the pick.
2. Actually launch the instance (console or CLI) once ready — this is the step
   that costs money and should be a deliberate, explicit action.
3. Run `infra/ec2_bootstrap.sh` on the box to get the repo + training deps
   installed.
4. Only then does Phase 7 itself (writing the GRPO training script, wiring it to
   `tau_forge.reward`/`tau_forge.harness`/`tau_forge.envs`) start — still gated on
   its own explicit go-ahead per the phase-status table.
