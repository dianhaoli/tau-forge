"""Phase 7 GRPO training entrypoint -- full-parameter RLVR, not LoRA (see
docs/phase7_aws_setup.md, "Sizing: memory", for why). Trains
`Qwen3-4B-Instruct-2507` against the Phase 2 synthetic scenarios
(`tau_forge.train.dataset`) using `tau_forge.reward.reward()` as the reward
signal (`tau_forge.train.reward_adapter`), via TRL's `GRPOTrainer`.

Only runs on a real GPU box -- imports `torch`/`transformers`/`trl`/`datasets`,
which are the `train` extra (`uv sync --extra train`), deliberately not part
of the default install. Everything this module calls into
(`tau_forge.train.dataset`, `.completion_parsing`, `.reward_adapter`) is
torch-free and already unit-tested against the real 541 scenarios in
`tests/test_train_pipeline.py` -- run those first if anything here misbehaves,
to rule out the data/reward plumbing before suspecting the GRPOTrainer wiring.

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
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "trained" / "phase7_run"
SMOKE_TEST_OUTPUT_DIR = REPO_ROOT / "data" / "trained" / "smoke_test_run"


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL coefficient against the frozen reference policy. TRL defaults this to 0.0 "
        "(off) -- deliberately overridden here; see docs/phase7_aws_setup.md's methodology "
        "section on overfitting/reward-hacking exposure from full-parameter capacity on a "
        "small, repeatedly-seen prompt set. Tune down only with a specific reason.",
    )
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-train-epochs", type=float, default=4.0)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=25)
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
    return p.parse_args()


def build_dataset(data_glob: str | None, apply_chat_template, tools: list[dict]):
    """Returns a `datasets.Dataset` with the columns GRPOTrainer needs (see
    `tau_forge.train.dataset.to_hf_rows`). Imports `datasets` lazily so this
    module can still be imported (e.g. for --help) without the train extra."""
    from datasets import Dataset

    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples, to_hf_rows

    examples = build_examples(data_glob=data_glob or DEFAULT_DATA_GLOB)
    rows = to_hf_rows(examples, apply_chat_template, tools)
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()

    # Deliberately deferred: keeps `--help` and the dataset-only path usable
    # without the GPU training stack installed.
    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.reward_adapter import grpo_reward_func

    max_steps = args.max_steps if args.max_steps is not None else (100 if args.smoke_test else -1)
    output_dir = Path(args.output_dir) if args.output_dir else (
        SMOKE_TEST_OUTPUT_DIR if args.smoke_test else DEFAULT_OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True
    )
    tools = RetailEnv().all_openai_schemas()

    dataset = build_dataset(args.data_glob, apply_chat_template, tools)
    print(f"[grpo_train] {len(dataset)} training examples loaded.")

    config_kwargs = dict(
        output_dir=str(output_dir),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        remove_unused_columns=False,  # reward_func needs the extra columns, see reward_adapter.py
        logging_steps=1,
        report_to=["none"],
        use_vllm=args.use_vllm,
    )
    if max_steps > 0:
        config_kwargs["max_steps"] = max_steps
    if args.deepspeed:
        config_kwargs["deepspeed"] = args.deepspeed

    training_args = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=grpo_reward_func,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(f"[grpo_train] Done. Checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
