"""Phase 7 GRPO training entrypoint -- full-parameter RLVR, not LoRA (see
docs/phase7_aws_setup.md, "Sizing: memory", for why). Trains
`Qwen3-4B-Instruct-2507` against the Phase 2 synthetic scenarios
(`tau_forge.train.dataset`) using `tau_forge.reward.reward()` as the reward
signal (`tau_forge.train.reward_adapter`), via TRL's `GRPOTrainer`.

Only runs on a real GPU box -- imports `torch`/`transformers`/`trl`/`datasets`,
which are the `train` extra (`uv sync --extra train`), deliberately not part
of the default install. Everything this module calls into
(`tau_forge.train.dataset`, `.completion_parsing`, `.reward_adapter`,
`.shaping`, `.curriculum`) is torch-free and already unit-tested against the
real 541 scenarios in `tests/test_train_pipeline.py` -- run those first if
anything here misbehaves, to rule out the data/reward plumbing before
suspecting the GRPOTrainer wiring.

Usage (see docs/phase7_aws_setup.md for the full launch sequence):
    accelerate launch --config_file infra/accelerate_zero2.yaml \\
        -m tau_forge.train.grpo_train --smoke-test

`--smoke-test` caps `max_steps` at 100 and points `output_dir` at a
`smoke_test_run/` subdirectory -- this is the Phase 6-flagged "50-100 step GRPO
smoke test", meant to run BEFORE the full job to validate the
generation->training->reward loop end-to-end and produce a real measured
seconds/rollout number (see docs/phase7_aws_setup.md, "Timing estimate") in
place of the first-principles estimate there. Do not skip straight to the full
run without it.

`--dry-run` does the whole dataset/mixture/prompt-length preflight and prints
what would be trained, without importing torch or touching a GPU. Run it before
every launch -- it is the check that catches a truncating `--max-prompt-length`
or a mixture that accidentally deleted a category.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "trained" / "phase7_run"
SMOKE_TEST_OUTPUT_DIR = REPO_ROOT / "data" / "trained" / "smoke_test_run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--data-glob",
        default=None,
        help="Glob for synthetic scenario JSON files. Defaults to "
        "tau_forge.train.dataset.DEFAULT_DATA_GLOB (all of data/synthetic/raw/). "
        "Held-out data policy: never point this at data/tau2/domains/retail/tasks.json.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-generations", type=int, default=16, help="GRPO group size per prompt.")
    p.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Must be a multiple of --num-generations divided across devices; tune down first if you hit OOM.",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Raised from 1: with --num-generations 16 and an effective batch of 32 across 4 GPUs, "
        "grad-accum 1 means each optimizer step sees only TWO unique prompts. The group size buys a "
        "good advantage estimate *within* a prompt; only accumulation buys prompt diversity *across* "
        "the update, and two prompts per step is a very noisy gradient on a 541-prompt corpus.",
    )
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL coefficient against the frozen reference policy. TRL defaults this to 0.0 "
        "(off) -- deliberately overridden here; see docs/phase7_aws_setup.md's methodology "
        "section on overfitting/reward-hacking exposure from full-parameter capacity on a "
        "small, repeatedly-seen prompt set. Note the tradeoff runs the other way too: KL to "
        "the reference policy is a ceiling on how far the policy can move, so it caps the "
        "Phase 8 improvement this run can produce. If maximum measured gain is the goal and "
        "--val-fraction checkpoint selection is catching overfitting, 0.0-0.01 is the setting "
        "to try; keep 0.04 when a conservative, obviously-safe run matters more.",
    )
    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Tokens of prompt kept. DEFAULT IS NONE (no truncation) and should stay that way: "
        "the rendered retail prompt is ~5-7k tokens (tau2 system prompt + full retail policy + "
        "16 tool schemas), and TRL truncates a prompt by keeping its LAST max_prompt_length "
        "tokens -- so any value below the real length silently deletes the tool definitions and "
        "the policy the model is being graded on. A preflight check refuses to run if the value "
        "given would truncate; pass --allow-prompt-truncation to override it deliberately.",
    )
    p.add_argument(
        "--allow-prompt-truncation",
        action="store_true",
        help="Override the preflight check that refuses a truncating --max-prompt-length.",
    )
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Exposed so a zero_shot_baseline variance audit can be run with the SAME sampling "
        "settings training will use. A variance measurement taken under different nucleus/top-k "
        "settings than the trainer's is not a measurement of the trainer's variance.",
    )
    p.add_argument("--top-k", type=int, default=None, help="See --top-p. None = no top-k filtering.")
    p.add_argument("--num-train-epochs", type=float, default=4.0)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument(
        "--scale-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to divide the advantage by the group's reward std (vanilla GRPO) or not "
        "(Dr. GRPO). DEFAULT OFF, deliberately. Dividing by std rescales every group to unit "
        "variance, so a group whose sixteen samples nearly agree gets its remaining noise "
        "amplified to the same magnitude as a group with real signal -- exactly backwards on a "
        "corpus measured at ~72%% near-degenerate groups. With this off, a low-variance group "
        "contributes in proportion to how much its rewards actually differ.",
    )
    p.add_argument(
        "--loss-type",
        default="dr_grpo",
        help="TRL GRPO loss variant. 'dr_grpo' removes the length normalization that otherwise "
        "biases the gradient toward longer completions -- irrelevant for a one-tool-call answer "
        "except that it quietly pays the policy to pad. Pass 'grpo' for the original formulation.",
    )
    p.add_argument(
        "--shaping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add tau_forge.train.shaping as a second reward function: bounded partial credit "
        "(<=0.15, strictly under reward()'s 0.2 tier) inside the flat 0.0 wrong-tool floor. This "
        "is the direct fix for zero-variance-at-0.0 groups -- see that module's docstring.",
    )
    p.add_argument(
        "--penalize-multi-call",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Within --shaping: subtract 0.05 for a completion emitting more than one <tool_call> "
        "block. Violates the retail policy's one-call-per-turn rule and only the first is parsed.",
    )
    p.add_argument(
        "--category-mix",
        default=None,
        help="Rebalance the training corpus, e.g. 'happy_path=0.36,requires_earlier_context=0.36,"
        "policy_violation=0.15,ambiguous=0.08,out_of_scope=0.05'. Pass 'real' for "
        "curriculum.REAL_TASK_ALIGNED_MIX, 'uniform' for the corpus as generated. Default (None) "
        "leaves the corpus untouched -- which means 53%% of the training signal rewards NOT acting; "
        "see tau_forge/train/curriculum.py for why that is a poor match to the benchmark.",
    )
    p.add_argument(
        "--exclude-zero-variance-from",
        default=None,
        help="Path to a zero_shot_baseline output JSON. Scenarios whose reward was constant across "
        "every sample there are dropped from training (they produce no gradient). Cold starts "
        "(constant at 0.0) only, unless --exclude-solved is also passed.",
    )
    p.add_argument(
        "--exclude-solved",
        action="store_true",
        help="Also drop scenarios that were constant at 1.0. Off by default: those are cheap "
        "regression insurance, and a scenario the base model solves can stop being solved mid-run.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of the (post-mixture) synthetic corpus held out for checkpoint selection, "
        "stratified across all 30 category__theme cells. Uses synthetic data rather than any of "
        "the 114 real tasks, per the README's held-out data policy: selecting a checkpoint by a "
        "real task's score steers weights by it just as surely as training on it does.",
    )
    p.add_argument("--curriculum-seed", type=int, default=0)
    p.add_argument(
        "--deepspeed",
        default=str(REPO_ROOT / "infra" / "ds_zero2.json"),
        help="Path to the DeepSpeed ZeRO-2 config. Pass '' to disable (single-GPU only).",
    )
    p.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for rollout generation instead of plain HF generate(). Faster but more "
        "moving parts to get right on a first run -- off by default so the smoke test's first "
        "job is validating correctness, not chasing a vLLM/DeepSpeed GPU-memory-sharing issue. "
        "Turn on once the smoke test passes without it.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Cap max_steps at 100 (if --max-steps not given) and default output-dir to "
        "data/trained/smoke_test_run/. Run this before the full job -- see module docstring.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report the dataset (mixture, exclusions, split, prompt-token distribution) "
        "and exit before importing torch. Needs a tokenizer download but no GPU.",
    )
    return p.parse_args(argv)


def resolve_mix(spec: str | None) -> dict[str, float] | None:
    from tau_forge.train.curriculum import REAL_TASK_ALIGNED_MIX, UNIFORM_MIX, parse_mix

    if spec is None:
        return None
    if spec == "real":
        return REAL_TASK_ALIGNED_MIX
    if spec == "uniform":
        return UNIFORM_MIX
    return parse_mix(spec)


def build_examples_for_run(args: argparse.Namespace):
    """Load -> exclude dead scenarios -> rebalance -> split. Returns
    `(train_examples, val_examples)`. Torch-free, so `--dry-run` can call it."""
    from tau_forge.train.curriculum import build_training_sets, load_zero_variance_ids, summarize
    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples

    examples = build_examples(data_glob=args.data_glob or DEFAULT_DATA_GLOB)
    print(f"[grpo_train] corpus as loaded: {json.dumps(summarize(examples))}")

    exclude: set[str] = set()
    if args.exclude_zero_variance_from:
        exclude = load_zero_variance_ids(
            args.exclude_zero_variance_from, include_solved=args.exclude_solved
        )
        print(
            f"[grpo_train] dropping {len(exclude)} zero-variance scenarios measured in "
            f"{args.exclude_zero_variance_from} (include_solved={args.exclude_solved})."
        )

    train_examples, val_examples = build_training_sets(
        examples,
        mix=resolve_mix(args.category_mix),
        exclude=exclude,
        val_fraction=args.val_fraction,
        seed=args.curriculum_seed,
    )
    print(f"[grpo_train] train split: {json.dumps(summarize(train_examples))}")
    if val_examples:
        print(f"[grpo_train] val split:   {json.dumps(summarize(val_examples))}")
    if not train_examples:
        raise ValueError("Training split is empty -- check --category-mix / --exclude-* filters.")
    return train_examples, val_examples


def check_prompt_lengths(rows: list[dict], tokenizer, args: argparse.Namespace) -> int:
    """Measure real prompt token lengths and refuse to train on truncated ones.

    This exists because the failure it catches is silent: TRL keeps the *last*
    `max_prompt_length` tokens of a prompt, so an over-tight value does not
    error -- it trains the policy on prompts whose system message, retail
    policy and tool schemas have been cut away, against a reward function that
    still grades as if they were there. Measured, not assumed: the check
    tokenizes the actual rendered prompts."""
    lengths = sorted(len(tokenizer(row["prompt"])["input_ids"]) for row in rows)
    longest = lengths[-1]
    print(
        f"[grpo_train] prompt tokens: min={lengths[0]} median={lengths[len(lengths) // 2]} "
        f"max={longest} (n={len(lengths)})"
    )
    limit = args.max_prompt_length
    if limit is None:
        print("[grpo_train] max_prompt_length=None -- no prompt truncation.")
        return longest
    if limit < longest:
        message = (
            f"--max-prompt-length {limit} would truncate {sum(1 for n in lengths if n > limit)} of "
            f"{len(lengths)} prompts (longest is {longest} tokens). TRL keeps the LAST {limit} "
            "tokens, which discards the system prompt, the retail policy and the tool schemas. "
            "Raise it above the longest prompt, or drop the flag entirely."
        )
        if not args.allow_prompt_truncation:
            raise ValueError(message + " Pass --allow-prompt-truncation to override.")
        print(f"[grpo_train] WARNING (overridden): {message}")
    return longest


def build_config_kwargs(args: argparse.Namespace, output_dir: Path, max_steps: int, bf16: bool) -> dict:
    kwargs = dict(
        output_dir=str(output_dir),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,  # reward_func needs the extra columns, see reward_adapter.py
        logging_steps=1,
        report_to=["none"],
        use_vllm=args.use_vllm,
    )
    if max_steps > 0:
        kwargs["max_steps"] = max_steps
    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed
    if args.val_fraction > 0:
        # Checkpoint by best held-out reward rather than last step -- the
        # overfitting mitigation docs/phase7_aws_setup.md asks for, and the
        # thing that makes a lower --beta safe to try.
        kwargs.update(
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_reward",
            greater_is_better=True,
            save_total_limit=3,
        )
    return kwargs


def filter_supported(kwargs: dict, config_cls) -> dict:
    """Drop kwargs the installed TRL's GRPOConfig does not accept, loudly.

    `scale_rewards`, `loss_type`, `top_k` and friends landed in TRL at
    different versions, and pyproject's floor (`trl>=0.16`) is deliberately not
    a hard pin. Silently passing an unknown kwarg raises deep inside a
    dataclass constructor with an unhelpful message; silently *dropping* one
    would mean a run quietly not using the setting its log says it used. So:
    drop, and say so."""
    import dataclasses

    accepted = {f.name for f in dataclasses.fields(config_cls)}
    supported = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        print(
            f"[grpo_train] WARNING: installed TRL's GRPOConfig does not accept {dropped} -- "
            "these settings are NOT in effect. Upgrade TRL if you need them."
        )
    return supported


def main() -> None:
    args = parse_args()

    train_examples, val_examples = build_examples_for_run(args)

    from transformers import AutoTokenizer

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.dataset import to_hf_rows

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True
    )
    tools = RetailEnv().all_openai_schemas()

    train_rows = to_hf_rows(train_examples, apply_chat_template, tools)
    val_rows = to_hf_rows(val_examples, apply_chat_template, tools) if val_examples else []
    check_prompt_lengths(train_rows + val_rows, tokenizer, args)

    if args.dry_run:
        print("[grpo_train] --dry-run: preflight complete, exiting before torch import.")
        return

    # Deferred: keeps --help, --dry-run and the dataset-only path usable
    # without the GPU training stack installed.
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from tau_forge.train.reward_adapter import grpo_reward_func
    from tau_forge.train.shaping import make_grpo_shaping_func

    max_steps = args.max_steps if args.max_steps is not None else (100 if args.smoke_test else -1)
    output_dir = Path(args.output_dir) if args.output_dir else (
        SMOKE_TEST_OUTPUT_DIR if args.smoke_test else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(val_rows) if val_rows else None
    print(f"[grpo_train] {len(train_dataset)} training examples, {len(val_rows)} validation examples.")

    reward_funcs = [grpo_reward_func]
    if args.shaping:
        reward_funcs.append(make_grpo_shaping_func(penalize_multi_call=args.penalize_multi_call))
        print("[grpo_train] shaping reward enabled (bounded partial credit inside the 0.0 floor).")

    config_kwargs = build_config_kwargs(
        args, output_dir, max_steps, bf16=torch.cuda.is_bf16_supported()
    )
    training_args = GRPOConfig(**filter_supported(config_kwargs, GRPOConfig))

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(f"[grpo_train] Done. Checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
