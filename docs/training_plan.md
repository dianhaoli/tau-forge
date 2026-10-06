# Multi-turn GRPO on episode tasks: how to run it

This is the runbook for `tau_forge/train/grpo_episodes.py`. It trains
Qwen3-4B-Instruct-2507 with TRL 1.12 `GRPOTrainer`, a custom `rollout_func`
and vLLM in colocate mode. Every rollout is a full multi-turn episode against
the scripted user and a private copy-on-write retail db.

The framework choice, the TRL 1.12 gotchas and the cost model come from the
training-efficiency plan (the `wf1_train_eff.md` research report). This page
covers what was built from that plan and how to run it.

## What runs where

| File | Role | Needs GPU stack |
|---|---|---|
| `tau_forge/train/episode_rollout.py` | Rollout core. Token-append sequences, in-flight cap with refill, `score_episode` rewards, flat-group detection and masking, the Qwen3 delta renderer, `check_append_equals_rerender`. | no |
| `tau_forge/train/grpo_episodes.py` | TRL wiring: config, dataset, `rollout_func`, synthetic-val callback, per-micro-batch padding trim. | only past `--dry-run` |
| `tau_forge/train/task_pool.py`, `scripts/prefilter_tasks.py` | Drop solved tasks and most dead tasks, using an episode audit. | no |
| `scripts/check_token_append.py` | Checks on the real tokenizer that appended ids equal a full re-render. | `transformers` only |
| `scripts/episode_audit.py` | Variance audit. `--max-inflight 64` and token-append prompts by default. | vLLM |
| `tests/test_episode_rollout.py` | CPU tests (30). They use a char-level tokenizer that renders with the real Qwen3-2507 template, and a reference-agent engine. | no |

### How a rollout is built

These are the rules that make the masks and importance sampling correct.

- **Groups.** Each dataset row is one `EpisodeTask` serialized to JSON in `prompt`. TRL's `RepeatSampler` repeats each row G times, so one rollout is one episode and a group is the G episodes of one task.
- **Prompt ids.** The chat template is rendered once, over the system message, 16 tools, greeting and opening, with the generation prompt.
- **Each turn.** The sampled ids are appended with `env_mask=1` and vLLM's logprobs. Then the environment reply is appended as `render_env_delta` ids with `env_mask=0` and logprob 0.0.
  - The env reply is the `\n` after `<|im_end|>`, then the tool results inside a `<|im_start|>user` turn (or the user reply), then `<|im_start|>assistant\n`.
  - History is never re-rendered.
- **End of sequence.** Each sequence ends after its last sampled token.
  - If the last turn completed, the sequence ends on `<|im_end|>`, which is Qwen's eos.
  - If the last turn hit `max_tokens`, the sequence ends mid-turn. `mask_truncated_completions` then drops it.
  - An episode whose first prompt does not fit becomes a single masked `<|im_end|>`.
- **Rewards.** The reward is `score_episode` on the finished episode, returned as `episode_reward`.
  - A harness crash returns `None`, which TRL treats as unscorable, and its tokens are masked.
  - All G episodes of a task share one scripted-user seed per step, `crc32(seed:task:step)`.
- **Flat groups.** A group whose G rewards are all equal gets `env_mask` zeroed, so it drops out of the DAPO token denominator (`--no-mask-flat-groups` turns this off).
  - This only applies to whole groups on one rank, so `(P*G/n_gpus) % G == 0` must hold. The run prints whether it does.
- **Scheduling.** At most `--max-inflight` episodes (default 64) are active per rank, and finished ones are replaced from the queue. Every engine call is one `llm.generate` over the active episodes.
  - Each request extends the previous one exactly, so vLLM's prefix cache covers the whole history.
  - TRL already sets vLLM `max_num_seqs` to `steps_per_generation`, which is 64 here.

## Config (defaults)

| Setting | Value | Why |
|---|---|---|
| Group size G (`--num-generations`) | 8 | P(flat) vs cost trade-off from the plan |
| Prompts per step P (`--prompts-per-step`) | 16 | 128 episodes per step |
| `per_device_train_batch_size` | 1 | No padding on 7–14k-token rows |
| `gradient_accumulation_steps` | 128 / n_gpus (64 on 2 GPUs) | `steps_per_generation` defaults to it, so the generation batch is 128 |
| `num_iterations` | 1 | |
| `loss_type` | `dapo` | Token-level. `dr_grpo` divides by a meaningless 512 for multi-turn |
| `beta` | 0.0 | No reference model: about −17% FLOPs and 8 GB less |
| `epsilon` / `epsilon_high` | 0.2 / 0.28 | Same as the slime reference |
| `scale_rewards` | `none` | |
| Learning rate | 1e-6 full-param, 1e-5 with `--lora` | |
| LoRA | r=64, alpha=64, all linear layers (not `lm_head`), dropout 0 | `--lora` |
| Weight decay | 0.1 full, 0.0 LoRA | |
| Schedule | `constant_with_warmup`, 5 warmup steps, grad clip 1.0 | |
| `mask_truncated_completions` | on | |
| `vllm_importance_sampling_correction` | on, `sequence_mask` (clip max 3.0) | `--is-mode token_truncate` if more than 5% of sequences are masked |
| vLLM | colocate, `gpu_memory_utilization` 0.35, `max_model_len` 16384, sleep mode off | |
| Sampling | temperature 1.0, top_p 1.0, top_k disabled (-1 to vLLM), `max_tokens` 512 per turn, `logprobs=0`, stop at `<|im_end|>` | Matches the variance audit's sampler |
| Episode limits | `max_turns` 20, `max_calls` 20 | |
| Precision | `bf16`; fp32 master weights for full-param (TRL's default load dtype); bf16 base for LoRA | |
| Gradient checkpointing | on (non-reentrant) | |
| Liger fused loss | off by default | `--liger` |
| Padding trim | on | `--no-trim-padding`. TRL pads every micro-batch to the longest sequence in the per-rank generation batch; the trim cuts each micro-batch to its own length |
| Synthetic val | 64 held-out pool tasks, 1 episode each, user seed 0, at step 0 and every 25 steps | |
| Checkpoints | every 25 steps | |

**Task pool.** Build it as follows.

1. Generate tasks with `python -m tau_forge.episodes.generate`.
2. Audit them with `scripts/episode_audit.py --samples-per-task 8`.
3. Run `scripts/prefilter_tasks.py`. It drops tasks solved 8/8 and keeps a seeded 20% of the dead ones (flat rewards with no success) in each template.
   - Alternatively, pass `--prefilter-audit` to the trainer.

On the existing `data/trained/episode_audit_s1_sub50_n8.json`, the flat share goes from 0.388 (the plan's 38.8%) to 0.019, keeping 156 of 250 tasks.

| Template | Tasks | Live | Solved | Dead | Kept |
|---|---|---|---|---|---|
| cancel | 50 | 26 | 23 | 1 | 26 |
| exchange | 50 | 41 | 6 | 3 | 41 |
| foreign_order_refusal | 50 | 34 | 0 | 16 | 37 |
| modify_payment | 50 | 32 | 17 | 1 | 32 |
| return_fallback | 50 | 20 | 30 | 0 | 20 |

These counts are from that audit's environment version. Re-audit after any env, reward or user change.

## GPU-box verification (before any paid run)

Run these on the training box after `uv sync --extra train`.

1. **Token append equals re-render**, on the real tokenizer. This takes minutes and needs no GPU:
   ```bash
   uv run --extra train python scripts/check_token_append.py --model Qwen/Qwen3-4B-Instruct-2507 \
       --json-out data/trained/check_token_append.json
   ```
   - It must print `passed N/N` and exit 0.
   - It also asserts that `<|im_end|>` and `<|im_start|>` are single tokens and that `eos_token_id == <|im_end|>`. If eos were anything else, `mask_truncated_completions` would mask every episode.
   - If the runner stores `"raw"` on assistant messages, the script also reports how many raw sampled turns differ from the canonical re-render that a tau2 eval feeds back.
2. **CPU tests**: `uv run pytest -q tests/test_episode_rollout.py tests/test_episodes.py`.
3. **Config and data dry run**, without torch:
   ```bash
   uv run python -m tau_forge.train.grpo_episodes --tasks data/episodes/pool_s1.jsonl --num-processes 2 --dry-run
   ```
4. **Two-step GPU run.** Use small batches and run val every step:
   ```bash
   uv run --extra train accelerate launch --config_file infra/accelerate_zero2.yaml --num_processes 2 \
       -m tau_forge.train.grpo_episodes --tasks data/episodes/pool_s1.jsonl \
       --max-steps 2 --prompts-per-step 4 --val-holdout 8 --val-every 1 --output-dir data/trained/grpo_episodes_dry
   ```
   Check each of these in the log:
   - **`episodes/n_appended_im_end` is 0.** That means vLLM returns the eos id in `token_ids`. If it is non-zero, the turn is still closed correctly with a masked `<|im_end|>`, but say so in the run notes.
   - **`sampling/importance_sampling_ratio/mean` is about 1.0, and `sampling/sampling_logp_difference/mean` is small.** A ratio far from 1 means the token stream that training scores differs from what vLLM sampled.
   - **`episodes/mean_reward` agrees with the audit's success level for the pool,** and `episodes/end/*` shows no `runner_error`.
   - **Memory headroom.** Neither the training nor the vLLM side may OOM. If they do, lower `--vllm-gpu-memory-utilization` or `--max-inflight`, or add `--liger`.
   - **Step time.** Divide by 128/16 = 8 for the per-episode cost, then put it into the plan's `cost_model.py`.

## How to run

Smoke run, 30 steps with 128 episodes per step, on 2x H100:
```bash
uv run --extra train accelerate launch --config_file infra/accelerate_zero2.yaml --num_processes 2 \
    -m tau_forge.train.grpo_episodes --tasks data/episodes/pool_s1.jsonl --smoke --wandb --run-name ep-smoke
```

Main run, 200 steps (or LoRA: add `--lora`, which also suits 1x L40S):
```bash
uv run --extra train accelerate launch --config_file infra/accelerate_zero2.yaml --num_processes 2 \
    -m tau_forge.train.grpo_episodes --tasks data/episodes/pool_s1.jsonl --max-steps 200 --wandb --run-name ep-main
```

Outputs go to `data/trained/grpo_episodes[_smoke]/`:
- checkpoints every 25 steps;
- `val_log.jsonl`, with the synthetic-val summary per evaluation;
- `final/`.

Pick the checkpoint by `val/mean_reward`. Run the tau2 eval (gpt-4.1 user, 4 trials, concurrency 32) only for the baseline and the selected final checkpoint.

## What to watch

| Metric | Healthy | Action if not |
|---|---|---|
| `reward`, `rewards/episode_reward_func/mean` | Rising over the run | Flat for over 50 steps: check flat share and IS masking |
| `val/mean_reward`, `val/<template>/mean_reward` | Rising; no template collapsing | Use it to select checkpoints; a collapsing template suggests reward hacking, so read transcripts |
| `completions/clipped_ratio` | Under 2% | Episodes whose last turn hit 512 tokens (all masked). If high, look for repeat loops |
| `sampling/importance_sampling_ratio/mean` and `/min` | Mean about 1; min well above 0 | `sequence_mask` sets masked sequences to ratio 0, so a mean below about 0.95 means more than 5% are masked. Switch to `--is-mode token_truncate` |
| `episodes/flat_group_share`, `frac_reward_zero_std` | 15% or less (2% on the prefiltered s1 pool at the base policy) | Re-audit and re-prefilter; flat groups are masked but their compute is still spent |
| `clip_ratio/region_mean` | A few percent | Much higher means the policy is moving too fast: lower the lr |
| `episodes/end/context_budget`, `episodes/end/max_turns` | Rare | Loops; the 16k budget is ample for real episodes |
| `episodes/n_incomplete_groups` | 0 | Groups split across ranks: fix P·G/n_gpus |
| `episodes/seconds_engine` vs step time | Rollout is about 25–30% of the step on H100 | If rollout dominates, check `--max-inflight` against KV capacity (the vLLM log line "KV cache ... tokens") |

## Expected step time and cost

**Estimates, not measurements.** This is the plan's model, ±40%. The smoke run's measured step time replaces it. Each range runs from continuous scheduling (low) to simple lockstep (high). The rollout here caps in-flight episodes and refills them between engine calls, which lands in between.

| Setup | Training s/step | Rollout s/step | Step s | 30 steps | 200 steps | GPU-h (200 steps) |
|---|---|---|---|---|---|---|
| 1x A10G LoRA, E=64 | 678 | 192–346 | 884–1037 | 8.7–10.2 h | 57–67 h | 57–67 |
| 1x L40S LoRA, E=128 | 299 | 178–321 | 492–634 | 5.0–6.3 h | 32–41 h | 32–41 |
| **2x H100 full, E=128** | 70 | 28–51 | **109–132** | **1.3–1.5 h** | **7.2–8.7 h** | 14–17 |
| 2x H100 LoRA, E=128 | 56 | 28–51 | 98–121 | 1.2–1.4 h | 6.5–8.0 h | 13–16 |
| 4x H100 full, E=128 | 35 | 18–32 | 64–78 | 0.9–1.0 h | 4.3–5.2 h | 17–21 |
| 8x H100 full, E=256, 100 steps (same episode count) | 35 | 18–32 | 64–78 | 0.6 h (15 steps) | 2.3–2.7 h | 18–22 |

| Setup (estimated $) | 30-step smoke | 150 steps | 200 steps | 300 steps |
|---|---|---|---|---|
| 1x A10G LoRA (E=64) | $9–10 / spot $3–5 | $43–50 / $15–25 | $57–67 / $20–33 | $86–101 / $30–50 |
| 1x L40S LoRA | $4–12 / $2–5 | $20–57 / $12–25 | $27–76 / $16–33 | $40–113 / $24–49 |
| **2x H100 full** | **$5–9** / spot $3–6 / AWS $10–21 | $22–39 | **$29–52** / spot $17–35 / AWS $58–122 | $43–77 |
| 2x H100 LoRA | $5–8 | $20–36 | $26–48 | $39–71 |
| 4x H100 full | $7–12 | $27–48 | $35–63 / spot $21–42 | $51–93 |
| 8x H100 (equal episode count) | $9–15 | $29–51 | $37–66 | $53–96 |

The plan also estimates these costs:
- Prefilter audit, 1,000 tasks × 8 on 2x H100: about 30–55 min, $2–5.
- Each tau2 eval: $15–35 of gpt-4.1 user-simulator calls.
- First credible run (smoke + prefilter + main + 2 evals): about $70–140, or about $50–100 on spot.

The training column assumes no padding waste. The padding trim delivers that for the training forward and backward. The importance-sampling old-logprob forward still runs on TRL's batch-wide padding, about 1/5 of training FLOPs (see open risks).

## Open risks (GPU-only)

1. **vLLM eos handling.** The code assumes `token_ids` ends with `<|im_end|>` when generation stops on it, which is TRL's own assumption. If it does not, a masked `<|im_end|>` is appended and the run still works. Check `episodes/n_appended_im_end`.
2. **Tokenizer boundary merges.** The CPU tests prove append equals re-render for a character-level tokenizer on the real template. Only `check_token_append.py` proves it for Qwen's BPE.
3. **Raw history vs eval re-render.** Training keeps the policy's raw sampled text in its history. A tau2 eval re-renders past tool calls from the parsed calls. Small formatting drift (whitespace, key order, text after `</tool_call>`) is possible. `check_token_append.py` counts it once the runner stores `"raw"`.
4. **Sequence-level IS masking.** `sequence_mask` multiplies about 650 per-token ratios. A systematic bf16 vLLM-vs-training logprob gap could mask many sequences. Watch `sampling/importance_sampling_ratio/mean`.
5. **Padding in the old-logprob forward.** TRL computes old logprobs over the padded per-rank generation batch, one row at a time, and the trim does not reach it. A `_get_per_token_logps_and_entropies` override would cut a further ~10% of step time.
6. **Private TRL attributes.** The code touches `trainer._metrics`, `trainer._last_loaded_step` and `trainer.vllm_generation.llm`/`sync_weights` (TRL 1.12.0). It also overrides `_prepare_inputs` for the padding trim. Re-check these after any TRL upgrade.
7. **Memory.** 2x H100 at `vllm_gpu_memory_utilization=0.35` is the plan's estimate: about 44 GB training plus 28 GB vLLM per GPU. It has not been measured.
