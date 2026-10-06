"""Multi-turn GRPO on episode tasks: TRL 1.12 `GRPOTrainer` + `rollout_func`
+ vLLM colocate. The rollout itself is `tau_forge.train.episode_rollout`
(torch-free, CPU-tested); this module is the TRL/vLLM wiring.

Design (docs/training_plan.md; wf1_train_eff.md §1-2 for the TRL 1.12 gotchas):
  * Dataset: one row per `EpisodeTask`, serialized as JSON into `prompt` --
    `rollout_func(prompts, trainer)` receives only the prompts, already
    repeated `num_generations` times by `RepeatSampler`, so each prompt is one
    episode and a group is the G episodes of one task.
  * Rollout: `trainer.vllm_generation.llm.generate` called directly, once per
    round of active episodes, with our own `SamplingParams(max_tokens=512,
    logprobs=0, stop at <|im_end|>)`. Never `trainer.vllm_generation.generate`:
    with sleep mode on it does a level-2 sleep after every call (weights and
    prefix cache dropped per turn), and it uses `max_completion_length` as the
    per-call cap. Sleep mode is forced off.
  * Token append, env_mask 1 only on sampled tokens, sequences end at the last
    sampled `<|im_end|>` (so `mask_truncated_completions` masks exactly the
    episodes whose final turn hit the token cap), logprobs from vLLM for
    sampled tokens and 0.0 for environment tokens (masked anyway).
  * Reward: `score_episode`, returned as the extra field `episode_reward` and
    passed through by `episode_reward_func`. One user seed per (task, step):
    a group's spread is the policy's alone.
  * Flat groups (all G rewards equal) get env_mask zeroed: they carry zero
    advantage under `scale_rewards="none"` and would otherwise dilute the DAPO
    token denominator. Needs whole groups per rank: (16*8/n_gpus) % 8 == 0.
  * Synthetic-val hook: every `--val-every` steps, N held-out tasks are scored
    with the same rollout core (user seed 0, one sample each), logged as
    `val/mean_reward`, `val/success_rate`, `val/<template>/mean_reward` and
    appended to `<output_dir>/val_log.jsonl`.

Usage (GPU box, `uv sync --extra train`; 2x H100 shown):
    accelerate launch --num_processes 2 --config_file infra/accelerate_zero2.yaml \\
        -m tau_forge.train.grpo_episodes --tasks data/episodes/pool.jsonl --smoke   # 30 steps
    accelerate launch --num_processes 2 ... -m tau_forge.train.grpo_episodes \\
        --tasks data/episodes/pool.jsonl --max-steps 200 --wandb
    python -m tau_forge.train.grpo_episodes --tasks ... --dry-run   # config + data check, no torch

Nothing at module level imports torch, transformers, TRL or vLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zlib
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TASKS = REPO_ROOT / "data" / "episodes" / "pool_v2" / "train.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "trained" / "grpo_episodes"
SMOKE_STEPS = 30
MAIN_STEPS = 200
LR_FULL = 1e-6
LR_LORA = 1e-5


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tasks", default=str(DEFAULT_TASKS), help="EpisodeTask JSONL (tau_forge.episodes.generate).")
    p.add_argument("--templates", default=None, help="Comma-separated templates to keep (default: all).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke", action="store_true", help=f"{SMOKE_STEPS}-step smoke run (output dir gets a _smoke suffix).")
    p.add_argument("--max-steps", type=int, default=None, help=f"Optimizer steps (default {MAIN_STEPS}, {SMOKE_STEPS} with --smoke).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-processes", type=int, default=None, help="Default: $WORLD_SIZE (set by accelerate), else 1.")

    g = p.add_argument_group("batch")
    g.add_argument("--num-generations", type=int, default=8, help="G: episodes per task per step.")
    g.add_argument("--prompts-per-step", type=int, default=16, help="P: tasks per optimizer step (E = P*G episodes).")

    g = p.add_argument_group("episode")
    g.add_argument("--max-turns", type=int, default=20)
    g.add_argument("--max-calls", type=int, default=20)
    g.add_argument("--max-new-tokens", type=int, default=512, help="Per assistant turn.")
    g.add_argument("--max-model-len", type=int, default=16384, help="vLLM context; an episode whose next request + max-new-tokens exceeds it ends as context_budget.")
    g.add_argument("--max-inflight", type=int, default=64, help="Episodes in flight per rank (keep within the KV cache).")
    g.add_argument("--no-mask-flat-groups", dest="mask_flat_groups", action="store_false",
                   help="Keep flat groups' tokens in the DAPO denominator.")

    g = p.add_argument_group("sampling")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=0, help="0 = disabled (sent to vLLM as -1, as the audit does).")

    g = p.add_argument_group("optimization")
    g.add_argument("--lora", action="store_true", help="LoRA r=64 on all linear layers (not lm_head); lr 1e-5.")
    g.add_argument("--lora-r", type=int, default=64)
    g.add_argument("--lora-alpha", type=int, default=64)
    g.add_argument("--learning-rate", type=float, default=None, help=f"Default {LR_FULL} full, {LR_LORA} LoRA.")
    g.add_argument("--weight-decay", type=float, default=None, help="Default 0.1 full, 0.0 LoRA.")
    g.add_argument("--warmup-steps", type=int, default=5)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--beta", type=float, default=0.0, help="KL coefficient. 0 = no reference model.")
    g.add_argument("--epsilon", type=float, default=0.2)
    g.add_argument("--epsilon-high", type=float, default=0.28)
    g.add_argument("--loss-type", default="dapo")
    g.add_argument("--scale-rewards", default="none")
    g.add_argument("--no-mask-truncated", dest="mask_truncated", action="store_false")
    g.add_argument("--no-is-correction", dest="is_correction", action="store_false",
                   help="Disable vllm_importance_sampling_correction.")
    g.add_argument("--is-mode", default="sequence_mask",
                   choices=("sequence_mask", "sequence_truncate", "token_mask", "token_truncate"),
                   help="Switch to token_truncate if sequence_mask drops >5%% of sequences.")
    g.add_argument("--model-dtype", default=None, help="Load dtype. Default: float32 master weights (full), bfloat16 (LoRA).")
    g.add_argument("--liger", action="store_true", help="use_liger_kernel (fused GRPO loss; not with a PEFT lm_head).")
    g.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    g.add_argument("--no-trim-padding", dest="trim_padding", action="store_false",
                   help="Keep TRL's generation-batch-wide padding in every training micro-batch.")
    g.add_argument("--deepspeed", default=None, help="DeepSpeed JSON, e.g. infra/ds_zero2.json.")
    g.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    g.add_argument("--save-steps", type=int, default=25)

    g = p.add_argument_group("task pool")
    g.add_argument("--prefilter-audit", default=None,
                   help="Episode audit JSON: drop tasks solved at n/n, keep --keep-dead-fraction of the flat dead ones.")
    g.add_argument("--keep-dead-fraction", type=float, default=0.2)
    g.add_argument("--keep-solved-fraction", type=float, default=0.0)
    g.add_argument("--drop-flat-other", action="store_true")

    g = p.add_argument_group("synthetic validation")
    g.add_argument("--val-tasks", default=None, help="Held-out EpisodeTask JSONL (ids also in --tasks are removed from training).")
    g.add_argument("--val-holdout", type=int, default=64, help="Without --val-tasks: hold out this many pool tasks (0 = no val).")
    g.add_argument("--val-every", type=int, default=25)
    g.add_argument("--no-val-at-start", dest="val_at_start", action="store_false")

    g = p.add_argument_group("logging")
    g.add_argument("--wandb", action="store_true", help="report_to wandb (needs WANDB_API_KEY).")
    g.add_argument("--run-name", default=None)
    g.add_argument("--dry-run", action="store_true", help="Load and filter tasks, print the config, exit before torch.")
    args = p.parse_args(argv)

    if args.max_steps is None:
        args.max_steps = SMOKE_STEPS if args.smoke else MAIN_STEPS
    if args.learning_rate is None:
        args.learning_rate = LR_LORA if args.lora else LR_FULL
    if args.weight_decay is None:
        args.weight_decay = 0.0 if args.lora else 0.1
    if args.model_dtype is None:
        args.model_dtype = "bfloat16" if args.lora else "float32"
    if args.num_processes is None:
        args.num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    if args.output_dir is None:
        args.output_dir = str(DEFAULT_OUTPUT_DIR) + ("_smoke" if args.smoke else "")
    return args


# ------------------------------------------------------------------ config


def batch_layout(args: argparse.Namespace) -> dict[str, int]:
    """E = P*G episodes per optimizer step, split evenly over ranks; one
    episode per micro-batch (no padding on 7-14k-token rows)."""
    episodes = args.prompts_per_step * args.num_generations
    if episodes % args.num_processes:
        raise ValueError(f"{episodes} episodes per step do not split over {args.num_processes} processes")
    per_rank = episodes // args.num_processes
    return {
        "episodes_per_step": episodes,
        "episodes_per_rank": per_rank,
        "gradient_accumulation_steps": per_rank,
        "whole_groups_per_rank": int(per_rank % args.num_generations == 0),
    }


def build_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """GRPOConfig kwargs. Pure, so tests check the defaults without TRL."""
    layout = batch_layout(args)
    top_k = args.top_k if args.top_k and args.top_k > 0 else 0
    kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        seed=args.seed,
        # batch: generation batch = 1 * n_procs * steps_per_generation(=grad accum) = E
        per_device_train_batch_size=1,
        gradient_accumulation_steps=layout["gradient_accumulation_steps"],
        num_generations=args.num_generations,
        num_iterations=1,
        shuffle_dataset=True,
        # optimizer
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        # loss
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        mask_truncated_completions=args.mask_truncated,
        vllm_importance_sampling_correction=args.is_correction,
        vllm_importance_sampling_mode=args.is_mode,
        # sampling as TRL records it (our rollout builds its own SamplingParams from the same args)
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
        max_completion_length=args.max_new_tokens,
        # vLLM colocate, sleep mode off (gotcha 4)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.max_model_len,
        vllm_enable_sleep_mode=False,
        # precision / kernels
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.liger,
        model_init_kwargs={"dtype": args.model_dtype},
        # bookkeeping
        remove_unused_columns=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=["wandb"] if args.wandb else ["none"],
        log_completions=False,
    )
    if args.gradient_checkpointing:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if args.run_name:
        kwargs["run_name"] = args.run_name
    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed
    return kwargs


def lora_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    # "all-linear" excludes lm_head in PEFT, which keeps Liger usable.
    return dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, target_modules="all-linear",
                task_type="CAUSAL_LM", bias="none")


# ------------------------------------------------------------------- data


def load_task_split(args: argparse.Namespace, log: Callable[[str], None] = print) -> tuple[list, list, dict]:
    """(train tasks, val tasks, report). Val is held out BEFORE the prefilter
    so it measures the whole distribution, solved and dead cells included."""
    from tau_forge.episodes.generate import read_jsonl
    from tau_forge.train.task_pool import format_report, prefilter_tasks

    pool = read_jsonl(args.tasks)
    if args.templates:
        keep = set(args.templates.split(","))
        pool = [t for t in pool if t.template in keep]
    report: dict[str, Any] = {"pool": len(pool)}
    if args.val_tasks:
        val = read_jsonl(args.val_tasks)
        val_ids = {t.id for t in val}
        train = [t for t in pool if t.id not in val_ids]
    elif args.val_holdout > 0:
        rng = random.Random(f"val:{args.seed}")
        idx = set(rng.sample(range(len(pool)), min(args.val_holdout, max(0, len(pool) - 1))))
        val = [t for i, t in enumerate(pool) if i in idx]
        train = [t for i, t in enumerate(pool) if i not in idx]
    else:
        val, train = [], list(pool)
    if args.prefilter_audit:
        train, pre = prefilter_tasks(
            train, args.prefilter_audit, keep_dead_fraction=args.keep_dead_fraction,
            keep_solved_fraction=args.keep_solved_fraction, drop_flat_other=args.drop_flat_other, seed=args.seed,
        )
        report["prefilter"] = pre
        log(f"[grpo_episodes] prefilter {args.prefilter_audit}:\n{format_report(pre)}")
    report.update(train=len(train), val=len(val))
    if not train:
        raise ValueError("no training tasks left")
    return train, val, report


def task_rows(tasks: Sequence[Any]) -> list[dict[str, str]]:
    return [{"prompt": json.dumps(t.to_dict())} for t in tasks]


def group_seed(task_id: str, step: int, seed: int = 0) -> int:
    """One scripted-user seed per (task, step): every episode of a group
    meets the same user, and a task met again later meets a different one."""
    return zlib.crc32(f"{seed}:{task_id}:{step}".encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------- rollout


def episode_reward_func(prompts=None, completions=None, episode_reward=None, **kwargs) -> list[Optional[float]]:
    """Passes `score_episode`'s reward through (None = harness crash, which
    TRL treats as unscorable: no advantage, out of the group baseline)."""
    return [None if r is None else float(r) for r in episode_reward]


def make_sampling_params(args: argparse.Namespace, tokenizer: Any) -> Any:
    from vllm import SamplingParams

    from tau_forge.train.episode_rollout import IM_END, special_token_id

    return SamplingParams(
        n=1,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # -1 = disabled, exactly what scripts/episode_audit.py sends, so the
        # audit's variance numbers describe this sampler.
        top_k=args.top_k if args.top_k > 0 else -1,
        logprobs=0,  # the sampled token's logprob only
        stop_token_ids=[special_token_id(tokenizer, IM_END)],
    )


def _sampled_logprob(entry_map: Any, token_id: int) -> Optional[float]:
    entry = entry_map.get(token_id) if hasattr(entry_map, "get") else None
    if entry is None:  # logprobs=0 returns just the sampled token
        entry = sorted(entry_map.values(), key=lambda e: getattr(e, "rank", 0) or 0)[0]
    v = float(entry.logprob)
    return None if math.isnan(v) else v


def vllm_results(outputs: Sequence[Any]) -> list[tuple]:
    """vLLM RequestOutputs -> episode_rollout engine results."""
    res = []
    for o in outputs:
        c = o.outputs[0]
        ids = list(c.token_ids)
        lps = None if c.logprobs is None else [_sampled_logprob(d, t) for t, d in zip(ids, c.logprobs)]
        res.append((ids, c.text, lps, c.finish_reason or "stop"))
    return res


def make_vllm_engine(llm: Any, sampling_params: Any) -> Callable[[list[list[int]]], list[tuple]]:
    def engine(prompt_ids: list[list[int]]) -> list[tuple]:
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params=sampling_params, use_tqdm=False)
        return vllm_results(outputs)

    return engine


def _log_metrics(trainer: Any, metrics: dict[str, float]) -> None:
    """Into TRL's per-step metric buffer: averaged and logged at the next
    `trainer.log` (every step with logging_steps=1)."""
    store = getattr(trainer, "_metrics", None)
    if store is None:
        return
    mode = "train" if getattr(getattr(trainer, "model", None), "training", True) else "eval"
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            store[mode][k].append(float(v))


TRL_KEYS = ("prompt_ids", "completion_ids", "logprobs", "env_mask", "episode_reward", "episode_success", "end_reason")


def make_rollout_func(
    args: argparse.Namespace,
    tokenizer: Any,
    tools: list,
    system_message: Optional[dict[str, str]],
    *,
    sampling_params: Any = None,
) -> Callable[[list[str], Any], dict[str, Any]]:
    from tau_forge.episodes.task import EpisodeTask
    from tau_forge.train.episode_rollout import rollout_episodes

    state = {"sp": sampling_params}

    def rollout_func(prompts: list[str], trainer: Any) -> dict[str, Any]:
        tasks = [EpisodeTask.from_dict(json.loads(p)) for p in prompts]
        if state["sp"] is None:
            state["sp"] = make_sampling_params(args, tokenizer)
        engine = make_vllm_engine(trainer.vllm_generation.llm, state["sp"])
        step = int(trainer.state.global_step)
        res = rollout_episodes(
            tasks, tokenizer, engine,
            tools=tools, system_message=system_message,
            max_inflight=args.max_inflight, max_turns=args.max_turns, max_calls=args.max_calls,
            max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
            user_seeds=lambda i, t: group_seed(t.id, step, args.seed),
            mask_flat_groups=args.mask_flat_groups, group_size=args.num_generations,
            require_logprobs=True,
        )
        stats = res["stats"]
        metrics = {f"episodes/{k}": v for k, v in stats.items() if not isinstance(v, dict)}
        for reason, n in stats["end_reasons"].items():
            metrics[f"episodes/end/{reason}"] = n / max(1, stats["n_episodes"])
        _log_metrics(trainer, metrics)
        return {k: res[k] for k in TRL_KEYS}

    return rollout_func


def run_validation(
    trainer: Any,
    val_tasks: Sequence[Any],
    tokenizer: Any,
    engine: Callable,
    args: argparse.Namespace,
    tools: list,
    system_message: Optional[dict[str, str]],
) -> dict[str, Any]:
    """Score this rank's shard of `val_tasks` (user seed 0, one episode each),
    gather across ranks, return the summary."""
    from tau_forge.train.episode_rollout import rollout_episodes, summarize_scores

    acc = getattr(trainer, "accelerator", None)
    rank, world = (acc.process_index, acc.num_processes) if acc is not None else (0, 1)
    shard = list(val_tasks[rank::world])
    rows: list = []
    if shard:
        res = rollout_episodes(
            shard, tokenizer, engine, tools=tools, system_message=system_message,
            max_inflight=args.max_inflight, max_turns=args.max_turns, max_calls=args.max_calls,
            max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
        )
        rows = [(t.template, r, s) for t, r, s in zip(shard, res["episode_reward"], res["episode_success"])]
    if acc is not None and world > 1:
        from accelerate.utils import gather_object

        rows = gather_object(rows)
    return summarize_scores(rows)


def make_val_callback(args, val_tasks, tokenizer, tools, system_message, holder: dict[str, Any]):
    from transformers import TrainerCallback

    out_path = Path(args.output_dir) / "val_log.jsonl"

    class SyntheticValCallback(TrainerCallback):
        def _run(self, step: int) -> None:
            trainer = holder["trainer"]
            if trainer.state.global_step != getattr(trainer, "_last_loaded_step", None):
                trainer.vllm_generation.sync_weights()
                trainer._last_loaded_step = trainer.state.global_step
            sp = holder.get("sampling_params") or make_sampling_params(args, tokenizer)
            holder["sampling_params"] = sp
            summary = run_validation(
                trainer, val_tasks, tokenizer, make_vllm_engine(trainer.vllm_generation.llm, sp),
                args, tools, system_message,
            )
            metrics = {"val/mean_reward": summary["overall"]["mean_reward"],
                       "val/success_rate": summary["overall"]["success_rate"]}
            for t, a in summary["per_template"].items():
                metrics[f"val/{t}/mean_reward"] = a["mean_reward"]
            _log_metrics(trainer, metrics)
            if trainer.accelerator.is_main_process:
                print(f"[grpo_episodes] val @ step {step}: {json.dumps(summary['overall'])}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("a") as f:
                    f.write(json.dumps({"step": step, **summary}) + "\n")

        def on_train_begin(self, a, state, control, **kw):
            if args.val_at_start:
                self._run(0)

        def on_step_end(self, a, state, control, **kw):
            if state.global_step % args.val_every == 0 or state.global_step == state.max_steps:
                self._run(state.global_step)

    return SyntheticValCallback()


PROMPT_KEYS = ("prompt_ids", "prompt_mask")


def trim_padding_columns(inputs: dict[str, Any]) -> dict[str, Any]:
    """Drop the padding columns a training micro-batch does not use.

    TRL 1.12 pads the whole per-rank generation batch (64 episodes) to its
    longest prompt (left) and longest completion (right), then slices it into
    micro-batches of `per_device_train_batch_size` rows WITHOUT re-trimming
    (`_prepare_inputs` -> `split_tensor_dict`). With one 7-14k-token episode
    per micro-batch, every forward/backward would run at the batch maximum.
    This keeps prompt columns from the first one any row attends to and
    completion columns up to the last one any row attends to (at least one,
    so `logits_to_keep` never becomes 0 = "all"). Masked columns contribute
    nothing to any loss sum, and DAPO's denominator (`num_items_in_batch`) is
    computed before the split, so the loss is unchanged.

    Completion-dimension tensors are every 2-D value whose second dimension
    equals the completion width (completion ids/mask, tool_mask, old/ref/
    sampling logps, token-level IS ratios). Any other 2-D width -- e.g. a
    multimodal field spanning prompt+completion -- means an input layout this
    function does not know, and the batch is returned untouched. Works on
    torch tensors and numpy arrays alike."""
    pm, cm = inputs.get("prompt_mask"), inputs.get("completion_mask")
    if getattr(pm, "ndim", 0) != 2 or getattr(cm, "ndim", 0) != 2:
        return inputs
    p_width, c_width = pm.shape[1], cm.shape[1]
    for k, v in inputs.items():
        if k not in PROMPT_KEYS and getattr(v, "ndim", 0) >= 2 and v.shape[1] not in (c_width, 1):
            return inputs
    p_any = (pm.sum(0) > 0).tolist()
    left = next((k for k, used in enumerate(p_any) if used), max(0, p_width - 1))
    c_any = (cm.sum(0) > 0).tolist()
    right = max((k + 1 for k, used in enumerate(c_any) if used), default=1)
    if left == 0 and right == c_width:
        return inputs
    out = {}
    for k, v in inputs.items():
        if k in PROMPT_KEYS:
            out[k] = v[:, left:]
        elif getattr(v, "ndim", 0) >= 2 and v.shape[1] == c_width:
            out[k] = v[:, :right]
        else:
            out[k] = v
    return out


def system_message() -> dict[str, str]:
    from tau_forge.train.dataset import _default_policy_text, _system_message

    return _system_message(_default_policy_text())


def describe(args: argparse.Namespace, report: dict[str, Any]) -> str:
    layout = batch_layout(args)
    return (
        f"[grpo_episodes] tasks: pool={report['pool']} train={report['train']} val={report['val']}\n"
        f"[grpo_episodes] batch: P={args.prompts_per_step} x G={args.num_generations} = {layout['episodes_per_step']} "
        f"episodes/step over {args.num_processes} rank(s), grad_accum={layout['gradient_accumulation_steps']}, "
        f"whole groups per rank: {bool(layout['whole_groups_per_rank'])}\n"
        f"[grpo_episodes] {'LoRA r=' + str(args.lora_r) if args.lora else 'full-param'} lr={args.learning_rate} "
        f"wd={args.weight_decay} beta={args.beta} eps={args.epsilon}/{args.epsilon_high} loss={args.loss_type} "
        f"scale_rewards={args.scale_rewards} steps={args.max_steps}\n"
        f"[grpo_episodes] sampling: temperature={args.temperature} top_p={args.top_p} "
        f"top_k={args.top_k if args.top_k > 0 else 'disabled'} max_new_tokens={args.max_new_tokens} "
        f"max_turns={args.max_turns} max_inflight={args.max_inflight} mask_flat_groups={args.mask_flat_groups}"
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    train_tasks, val_tasks, report = load_task_split(args)
    print(describe(args, report))
    if not batch_layout(args)["whole_groups_per_rank"]:
        print("[grpo_episodes] WARNING: groups split across ranks; flat-group masking only sees whole groups.")
    if args.dry_run:
        print(json.dumps(build_config_kwargs(args), indent=1, default=str))
        print("[grpo_episodes] --dry-run: exiting before torch import.")
        return

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.episode_rollout import IM_END, special_token_id
    from tau_forge.train.grpo_train import filter_supported

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id != special_token_id(tokenizer, IM_END):
        raise ValueError(
            f"tokenizer eos {tokenizer.eos_token_id} is not <|im_end|>: mask_truncated_completions would mask "
            "every episode (sequences end at <|im_end|>)"
        )
    tools = RetailEnv().all_openai_schemas()
    sys_msg = system_message()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = GRPOConfig(**filter_supported(build_config_kwargs(args), GRPOConfig))
    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(**lora_config_kwargs(args))
    holder: dict[str, Any] = {}
    callbacks = []
    if val_tasks and args.val_every > 0:
        callbacks.append(make_val_callback(args, val_tasks, tokenizer, tools, sys_msg, holder))

    class EpisodeGRPOTrainer(GRPOTrainer):
        def _prepare_inputs(self, generation_batch):
            inputs = super()._prepare_inputs(generation_batch)
            if args.trim_padding and self.model.training:
                inputs = trim_padding_columns(inputs)
            return inputs

    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    trainer = EpisodeGRPOTrainer(
        model=args.model,
        reward_funcs=[episode_reward_func],
        args=config,
        train_dataset=Dataset.from_list(task_rows(train_tasks)),
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=make_rollout_func(args, tokenizer, tools, sys_msg),
        callbacks=callbacks,
    )
    holder["trainer"] = trainer
    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final"))
    print(f"[grpo_episodes] done; final checkpoint in {Path(args.output_dir) / 'final'}")


if __name__ == "__main__":
    main()
