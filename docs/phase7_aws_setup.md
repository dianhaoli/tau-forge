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

## Three length knobs, only one of which is a memory knob

These get conflated, and conflating them is how the `--max-prompt-length 2048`
default survived. They are not interchangeable.

| knob | where | what it does | is it a memory control? |
|---|---|---|---|
| `--max-prompt-length` | TRL `GRPOConfig` | Keeps the **last** N tokens of the prompt and discards the rest, silently | **No.** Use it only to reject over-long prompts you intended to reject |
| `--max-model-len` | vLLM | Sizes the KV cache; a request exceeding it is **rejected**, never trimmed | **Yes** |
| `--max-completion-length` / `--max-new-tokens` | both | Caps generated tokens | Yes, and cheaply |

The rendered retail prompt is a fixed ~5-7k tokens and is not compressible
without changing the task: it is tau2's system prompt, the full retail policy,
and 16 tool schemas. Cutting it to fit a GPU does not make the run cheaper, it
makes the run meaningless -- the policy model loses the tool definitions while
the reward function keeps grading as if it had them, so every completion scores
in the flat 0.0 floor for a reason that has nothing to do with the policy.

**If a single GPU is the constraint, these are the knobs that actually help**,
in order of how much they buy per unit of damage:

1. `--max-new-tokens` / `--max-completion-length`. A single `<tool_call>` block
   needs well under 512 tokens; 256 is usually plenty and halves the generated
   KV.
2. Concurrency. vLLM's `--max-num-seqs` and `gpu_memory_utilization`; TRL's
   `--per-device-train-batch-size` with `--gradient-accumulation-steps` raised
   to compensate, which keeps the effective batch identical.
3. Scenario count per pass. `--scenario-ids-file` runs a subset; the stratified
   sampler in `scripts/sample_audit_ids.py` keeps all 30 cells represented.
4. For training specifically: ZeRO-3 with CPU optimizer offload, or LoRA.

On KV-cache sizing, the formula is
`2 (K and V) x n_layers x n_kv_heads x head_dim x 2 bytes` per token. For
Qwen3-4B's published config (36 layers, 8 KV heads, head_dim 128 -- verify
against the checkpoint rather than trusting this line) that is ~147 KB per
token, so an 8192-token budget is ~1.2 GB per concurrent sequence on top of
~8 GB of bf16 weights. Sixteen concurrent sequences, one full GRPO group, is
therefore ~27 GB for inference alone.

**Full-parameter GRPO on a 4B model does not fit one GPU**, which is the reason
the recommendation above is a 4x L40S box. Weights, gradients, fp32 Adam moments
and the fp32 master copy come to roughly 64 GB before a single activation. A
single-GPU pass is an inference budget (the zero-shot audit, the Phase 8 eval
against a served checkpoint), not a training budget. If a single-GPU *training*
smoke test is wanted before the quota lands, it needs LoRA or ZeRO-3 with
offload, and its result is a validation of the plumbing, not a preview of the
full run's learning curve.

## Methodology risks — and why full-parameter GRPO needs more than just the box

Getting the instance right doesn't by itself mean the training run works well.
Three risks are specific to this project's data/setup, not to the infra, and are
worth deciding on *before* spending GPU-hours on the full run:

1. **GRPO's signal needs within-group reward variance.** GRPO normalizes
   advantage by each group's own mean/std across its `G` sampled completions. If a
   scenario is trivially easy (base model gets it right every time) or too
   hard/ambiguous (consistently wrong for reasons unrelated to sampling), every
   sample in the group gets the same reward and that prompt contributes ~zero
   gradient. **Phase 3 stage 4 (difficulty calibration) has since landed**
   (`tau_forge/validate/difficulty.py`, `data/synthetic/difficulty/*.json`) — but
   it's a *structural* heuristic (category, action-kind, distractor closeness,
   stage-2 model-checker flags), not an empirical measurement of this specific
   model's zero-shot behavior. It's a good prior for which scenarios are likely
   risky, not a substitute for actually checking: a scenario the heuristic calls
   "easy" can still turn out zero-variance for a *different* reason (the model
   is confidently, consistently wrong on it, not confidently right). `python -m
   tau_forge.train.zero_shot_baseline` (written alongside this doc, see below) is
   the empirical check — run it and cross-reference against the heuristic scores
   before trusting either alone.
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

## Launch runbook (AWS CLI)

Concrete, copy-pasteable steps. Console equivalents exist for every step (EC2 →
Launch Instance) if you'd rather click through, but the CLI is what the checklist
above means by "reproducible and diffable." Run these from your own machine (or
CloudShell) — this session has no AWS credentials and cannot run them for you.
Replace `us-east-1` throughout if you're in a different region; `g6e` availability
varies by region, so check first (step 0).

**0. Confirm `aws` is configured and `g6e` is available/quota'd in your region**
```
aws sts get-caller-identity
aws ec2 describe-instance-type-offerings --location-type region \
    --filters Name=instance-type,Values=g6e.12xlarge --region us-east-1
```
If that returns nothing, `g6e` isn't offered in that region — pick another (check
the docs' fallback options) or a nearby region. Then check your quota — new/
lightly-used accounts often start at **0** for GPU instances:
```
aws service-quotas get-service-quota --service-code ec2 \
    --quota-code L-DB2E81BA --region us-east-1  # "Running On-Demand G and VT instances" (vCPUs)
```
`g6e.12xlarge` needs 48 vCPUs of quota. If the value is below that, request an
increase (`request-service-quota-increase` or via Service Quotas console) — this
is the single most common thing that silently blocks a first GPU launch, and
approval can take from minutes to a business day, so do this check first.

**1. Set a billing alarm** (console: Billing → Budgets → Create budget — a CLI
path exists too but the console is faster for a one-off alarm) before doing
anything else below.

**2. Create a key pair** (skip if reusing one you already trust for this):
```
aws ec2 create-key-pair --key-name tau-forge-phase7 \
    --query 'KeyMaterial' --output text > ~/.ssh/tau-forge-phase7.pem
chmod 400 ~/.ssh/tau-forge-phase7.pem
```

**3. Create a security group, SSH-only from your own IP:**
```
VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
    --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 create-security-group --group-name tau-forge-phase7 \
    --description "Phase 7 GPU box, SSH only" --vpc-id "$VPC_ID" \
    --query 'GroupId' --output text)
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32"
```
No other inbound rules — confirms the checklist's "no public HTTP/Jupyter
exposure" item by construction.

**4. Find the current Deep Learning AMI (Ubuntu, GPU PyTorch) for your region:**
```
AMI_ID=$(aws ec2 describe-images --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text \
    --region us-east-1)
echo "$AMI_ID"
```
Sanity-check the returned name/date look right before using it — AMI naming
occasionally changes; if this comes back empty, search "Deep Learning AMI GPU
PyTorch" in the EC2 console's AMI catalog instead and copy the ID.

**5. Launch the instance:**
```
aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type g6e.12xlarge \
    --key-name tau-forge-phase7 \
    --security-group-ids "$SG_ID" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tau-forge-phase7}]' \
    --region us-east-1
```
Note the returned `InstanceId`. Get its public IP once it's running:
```
aws ec2 describe-instances --instance-ids <InstanceId> \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

**6. SSH in and run the bootstrap script:**
```
ssh -i ~/.ssh/tau-forge-phase7.pem ubuntu@<PublicIpAddress>
# on the box:
curl -O https://raw.githubusercontent.com/dianhaoli/tau-forge/main/infra/ec2_bootstrap.sh
chmod +x ec2_bootstrap.sh
./ec2_bootstrap.sh
```
(Deep Learning AMIs use the `ubuntu` login user; double-check the AMI's own
listing if that doesn't connect.) The bootstrap script prints the exact
zero-shot-baseline / smoke-test / full-run commands at the end, matching "Next
steps" below.

**7. When done for the session:**
```
aws ec2 stop-instances --instance-ids <InstanceId>   # keeps the EBS volume, resume later
# or, once truly finished with this box:
aws ec2 terminate-instances --instance-ids <InstanceId>
```
`stop` (not `terminate`) is the right default between working sessions — it
keeps the 300 GB volume (and anything on it: cloned repo, downloaded model
weights, checkpoints) so you're not re-downloading the model each time, at the
cost of EBS storage while stopped (much cheaper than the running GPU-hour rate).

## Next steps (not done by this doc)

1. Review this sizing/instance choice — confirm `g6e.12xlarge` (or an alternative
   above) is the pick.
2. Actually launch the instance (console or CLI) once ready — this is the step
   that costs money and should be a deliberate, explicit action.
3. Run `infra/ec2_bootstrap.sh` on the box to get the repo + training deps
   installed.
4. Zero-shot base-model pass over all 541 scenarios (difficulty signal +
   baseline) — cheap, inference-only, do this before writing a line of training
   code:
   ```
   uv run python3 -m tau_forge.train.zero_shot_baseline
   ```
5. The 50-100 step GRPO smoke test (Phase 6 spec) — validates the loop end-to-end
   and produces a real seconds/rollout number to replace the Timing estimate
   above:
   ```
   uv run accelerate launch --config_file infra/accelerate_zero2.yaml \
       -m tau_forge.train.grpo_train --smoke-test
   ```
6. Only after both of those look sane, the full run (still full-parameter GRPO,
   same ZeRO-2 `accelerate`/`deepspeed` config, wiring
   `tau_forge.reward`/`tau_forge.envs` via `tau_forge.train.reward_adapter`):
   ```
   uv run accelerate launch --config_file infra/accelerate_zero2.yaml \
       -m tau_forge.train.grpo_train
   ```
   This last step is still gated on its own explicit go-ahead per the
   phase-status table — steps 4-5 are cheap diagnostics, not the full job.

## What's implemented vs. what still needs the GPU box

`tau_forge/train/` now has the actual training code, written and unit-tested
against the real 541 scenarios (`tests/test_train_pipeline.py`, runs without a
GPU) but never executed end-to-end (no GPU/torch/trl available in the
environment this was written in):

- `dataset.py` — builds the GRPO training set from `data/synthetic/raw/*.json`,
  reusing tau2's own real agent system prompt (`tau2.agent.llm_agent`) and real
  tool schemas (`RetailEnv.all_openai_schemas()`), graded against the plain
  shipped `db.json` (matching `tau_forge.validate.rule_checker`'s own
  re-execution methodology, not a derived per-scenario snapshot).
- `completion_parsing.py` — parses a policy completion's `<tool_call>` block
  (Qwen's native tool-calling format) into an `Action`. Deliberately treats a
  *malformed* tool-call attempt differently from *no* attempt (a caught bug,
  not a design given from the start — see its module docstring) so a garbled
  call can't be mistaken for correctly-chosen silence on an
  `ambiguous`/`policy_violation` scenario.
- `reward_adapter.py` — adapts `tau_forge.reward.reward()` into TRL's
  `reward_funcs(prompts, completions, **kwargs) -> list[float]` contract.
- `grpo_train.py` — the actual `GRPOTrainer` entrypoint, `--smoke-test` flag
  included, `beta` (KL coefficient) deliberately overridden from TRL's default
  of `0.0` per the methodology risks above.
- `zero_shot_baseline.py` — step 4's diagnostic script.
- `infra/ds_zero2.json` / `infra/accelerate_zero2.yaml` — the ZeRO-2 configs
  for the 4-GPU launch.

**What's unverified**: everything past "does the data/reward plumbing work" —
the actual `GRPOTrainer`/`accelerate`/`deepspeed` wiring, vLLM integration (off
by default, see `--use-vllm`'s help text), and whether the chosen
`per_device_train_batch_size`/`num_generations` combination fits in the ~26
GB/GPU budget without adjustment. The smoke test (step 5) is what actually
answers that — treat the first run of it as a debugging session, not a
guaranteed clean pass, and report back what breaks.

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
