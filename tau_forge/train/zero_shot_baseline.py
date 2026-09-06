"""Zero-shot baseline pass: run the base policy model over synthetic scenarios
with NO training, and report the reward distribution.

This is step 4 in docs/phase7_aws_setup.md's "Next steps" -- run BEFORE writing
or launching any training, for two reasons:
  1. It's the missing Phase 3-stage-4 difficulty signal. GRPO's advantage
     estimate needs within-group reward variance; a scenario the base model
     already gets right (or wrong) every time contributes ~zero training
     signal. A reward histogram bunched at the extremes (many 0.0s and 1.0s,
     few in between) is the warning sign to look for.
  2. It's the pre-training baseline every later checkpoint should be compared
     against -- without it, "did training help" has nothing to measure from.

Cheap and inference-only (no gradients, no optimizer, no reference model) --
run this first, before the smoke test, before the full job.

Supports two generation backends: plain HF `.generate()` (default, no extra
setup) or vLLM (`--use-vllm`, much faster, same dependency already installed
via `uv sync --extra train`) -- useful for a fast targeted re-run over a
scenario subset (e.g. the ids that came back zero-variance on a first pass)
at a higher temperature/sample count.

Usage:
    python -m tau_forge.train.zero_shot_baseline --model Qwen/Qwen3-4B-Instruct-2507

    # Fast, targeted re-run: only the scenarios listed in ids.txt (one id per
    # line), more samples, hotter sampling, and the actual completion text
    # saved (not just scores) so failures can be inspected directly:
    python -m tau_forge.train.zero_shot_baseline --use-vllm \
        --scenario-ids-file ids.txt --samples-per-scenario 16 \
        --temperature 1.3 --save-completions
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "trained" / "zero_shot_baseline.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--data-glob", default=None)
    p.add_argument("--category", default=None, help="Only run scenarios whose id starts with this category (e.g. 'out_of_scope').")
    p.add_argument("--scenario-ids-file", default=None, help="Path to a text file of scenario ids (one per line) to restrict the run to.")
    p.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Which curriculum split to score. 'all' (default) is the whole corpus -- right for "
        "a variance audit, wrong for a before/after. For a synthetic-data baseline you intend to "
        "re-measure after training, use --split val: it reproduces exactly the held-out slice "
        "grpo_train carves off, so the second measurement is on scenarios the run never trained "
        "on. Pass the SAME --val-fraction, --category-mix and --curriculum-seed here as you pass "
        "to grpo_train, or the two commands compute different splits and the comparison is void.",
    )
    p.add_argument("--val-fraction", type=float, default=0.1, help="See --split. Must match grpo_train's.")
    p.add_argument("--category-mix", default=None, help="See --split. Must match grpo_train's. 'real', 'uniform', or an explicit spec.")
    p.add_argument("--curriculum-seed", type=int, default=0, help="See --split. Must match grpo_train's.")
    p.add_argument("--samples-per-scenario", type=int, default=4, help="Repeats per scenario, for a rough within-scenario variance read -- not a full GRPO group, just enough to see if a scenario is deterministic.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling cutoff. MUST match what the trainer will use (grpo_train's "
        "--top-p, default 1.0), or this run measures the variance of a different sampler than "
        "the one GRPO will actually draw its groups with. Note the plain-HF backend otherwise "
        "inherits Qwen's shipped generation_config (top_p=0.8, top_k=20), which suppresses "
        "exactly the variance being measured -- so this is passed explicitly on both backends.",
    )
    p.add_argument("--top-k", type=int, default=0, help="0 / -1 = disabled. See --top-p.")
    p.add_argument(
        "--with-shaping",
        action="store_true",
        help="Also score each completion with tau_forge.train.shaping's auxiliary term and report "
        "the variance picture under reward()+shaping -- i.e. what GRPO will actually see when "
        "grpo_train runs with --shaping (its default). Run the audit both ways: the difference "
        "between the two zero-variance fractions is exactly how many dead groups the shaping "
        "term revives.",
    )
    p.add_argument("--batch-size", type=int, default=8, help="Ignored when --use-vllm is set (vLLM batches internally).")
    p.add_argument("--use-vllm", action="store_true", help="Generate with vLLM instead of plain HF .generate() -- much faster, same dependency already installed.")
    p.add_argument("--max-model-len", type=int, default=8192, help="vLLM only. Caps the KV cache to this many tokens instead of the model's full native context (Qwen3's is 262144, which needs far more KV cache memory than a single GPU has). Must exceed the longest prompt (the retail system prompt + policy text + tool schemas can run several thousand tokens) plus --max-new-tokens, or vLLM rejects that request outright.")
    p.add_argument("--save-completions", action="store_true", help="Include raw completion text per sample in the output JSON, not just scores. Off by default since it makes the output much larger.")
    p.add_argument(
        "--score-workers",
        type=int,
        default=0,
        help="Processes to score completions across. 0 (default) picks one per available "
        "core, capped at 16. Use 1 to score in-process. Scoring is pure CPU and, on a "
        "right-tool completion, deep-copies the 2.8MB retail db -- at n=16 over the whole "
        "corpus that is thousands of seconds of silence after vLLM has already printed its "
        "shutdown banner. Each worker holds its own db, so ~100MB of RSS per worker.",
    )
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args(argv)


def check_context_budget(prompts: list[str], tokenizer, args: argparse.Namespace) -> int:
    """Confirm the longest prompt plus its completion fits the KV-cache budget.

    This is a *memory* check, and it is the only length knob on this script.
    It does not shorten anything: `--max-model-len` is the ceiling vLLM sizes
    its KV cache to, and a request exceeding it is rejected outright rather
    than trimmed. Catching that here costs a tokenizer pass; catching it after
    vLLM has loaded the weights costs the model load.

    Do not confuse this with `grpo_train --max-prompt-length`, which really
    does silently delete the front of the prompt. See docs/phase7_aws_setup.md,
    "Three length knobs".
    """
    lengths = sorted(len(tokenizer(prompt)["input_ids"]) for prompt in prompts)
    needed = lengths[-1] + args.max_new_tokens
    print(
        f"[zero_shot_baseline] prompt tokens: min={lengths[0]} "
        f"median={lengths[len(lengths) // 2]} max={lengths[-1]}; "
        f"longest prompt + --max-new-tokens = {needed}"
    )
    if args.use_vllm and needed > args.max_model_len:
        raise ValueError(
            f"--max-model-len {args.max_model_len} is below the {needed} tokens the longest "
            f"prompt plus --max-new-tokens {args.max_new_tokens} needs; vLLM would reject those "
            f"requests. Raise --max-model-len to at least {needed}, or lower --max-new-tokens. "
            "Do NOT try to solve this by shortening the prompt -- it is the retail policy and the "
            "16 tool schemas, and the reward function grades as if the model saw all of them."
        )
    return lengths[-1]


def _load_scenario_ids(path: str) -> set[str]:
    ids = {line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")}
    if not ids:
        raise ValueError(f"No scenario ids found in {path}")
    return ids


def _generate_hf(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    import torch

    completions: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                # Explicit, not inherited: Qwen3 ships generation_config
                # top_p=0.8/top_k=20, which would quietly narrow the very
                # distribution this run exists to measure.
                top_p=args.top_p,
                top_k=args.top_k if args.top_k else 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completions.extend(
            tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        del inputs, out
        torch.cuda.empty_cache()
        print(f"[zero_shot_baseline] {min(start + args.batch_size, len(prompts))}/{len(prompts)} generations done")
    return completions


def _generate_vllm(args: argparse.Namespace, prompts: list[str], samples_per_prompt: int) -> list[str]:
    """Returns `len(prompts) * samples_per_prompt` completions, prompt-major
    (all samples for prompt 0, then prompt 1, ...).

    Uses vLLM's own `n=` rather than repeating each prompt in the request list:
    same sampling, but the ~6k-token prompt is prefilled once per scenario
    instead of once per sample. At n=16 over 541 scenarios that is the
    difference between 541 and 8656 prefills of an identical prompt."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        n=samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k else -1,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)
    completions: list[str] = []
    for o in outputs:
        texts = [c.text for c in o.outputs]
        if len(texts) != samples_per_prompt:  # vLLM can return fewer on a stop edge case
            texts = (texts + [""] * samples_per_prompt)[:samples_per_prompt]
        completions.extend(texts)
    return completions


# One `RetailEnv` per worker process. Shaping only ever asks it schema
# questions, so a single instance is reusable for the whole chunk -- and
# building one per completion would cost more than the shaping itself.
_worker_shaping_env = None


def _score_scenario(
    payload: tuple[Optional[str], str, list[str], bool, bool]
) -> tuple[list[float], list[float]]:
    """Score every sample of one scenario. Runs in a pool worker, so it takes
    plain picklable data and imports its own dependencies.

    A scenario is the right unit of work to hand a worker: its samples share a
    gold action, so `reward_adapter`'s gold cache hits for all but the first,
    and each worker pays the one-off `db.json` parse once rather than per call.
    """
    global _worker_shaping_env
    expected_name, expected_args_json, completions, with_shaping, penalize_multi_call = payload

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.reward_adapter import score_completion
    from tau_forge.train.shaping import shaping_score

    expected_args = json.loads(expected_args_json)
    if with_shaping and _worker_shaping_env is None:
        _worker_shaping_env = RetailEnv()

    scores: list[float] = []
    shaped: list[float] = []
    for completion in completions:
        score = score_completion(completion, expected_name, expected_args)
        scores.append(score)
        if with_shaping:
            shaped.append(
                score
                + shaping_score(
                    completion,
                    expected_name,
                    expected_args,
                    _worker_shaping_env,
                    penalize_multi_call,
                )
            )
    return scores, shaped


def resolve_score_workers(requested: int) -> int:
    """`--score-workers 0` means one per core, capped. The cap is there because
    past ~16 the per-worker db parse and the parent's result handling start
    costing more than the extra parallelism buys."""
    if requested > 0:
        return requested
    return max(1, min(16, os.cpu_count() or 1))


def main() -> None:
    args = parse_args()

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples, to_hf_rows
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True
    )
    tools = RetailEnv().all_openai_schemas()

    examples = build_examples(data_glob=args.data_glob or DEFAULT_DATA_GLOB)
    rows = to_hf_rows(examples, apply_chat_template, tools)

    if args.split != "all":
        from tau_forge.train.curriculum import build_training_sets
        from tau_forge.train.grpo_train import resolve_mix

        train_examples, val_examples = build_training_sets(
            examples,
            mix=resolve_mix(args.category_mix),
            val_fraction=args.val_fraction,
            seed=args.curriculum_seed,
        )
        keep = {e.id for e in (val_examples if args.split == "val" else train_examples)}
        rows = [r for r in rows if r["id"] in keep]
        print(
            f"[zero_shot_baseline] --split {args.split}: {len(rows)} scenarios "
            f"(val_fraction={args.val_fraction}, mix={args.category_mix}, seed={args.curriculum_seed}). "
            "Re-run with these exact values after training to compare like with like."
        )

    if args.category:
        rows = [r for r in rows if r["id"].startswith(args.category)]
    if args.scenario_ids_file:
        ids = _load_scenario_ids(args.scenario_ids_file)
        rows = [r for r in rows if r["id"] in ids]
        missing = ids - {r["id"] for r in rows}
        if missing:
            print(f"[zero_shot_baseline] warning: {len(missing)} requested ids not found in dataset: {sorted(missing)[:5]}...")
    if not rows:
        raise ValueError("No scenarios left after filtering -- check --category/--scenario-ids-file.")

    print(f"[zero_shot_baseline] {len(rows)} scenarios, {args.samples_per_scenario} samples each "
          f"= {len(rows) * args.samples_per_scenario} total generations.")

    check_context_budget([row["prompt"] for row in rows], tokenizer, args)

    # Prompt-major ordering in both backends: all samples of scenario 0, then
    # scenario 1, ... -- so `all_meta` lines up with `completions` either way.
    all_meta = [row for row in rows for _ in range(args.samples_per_scenario)]

    if args.use_vllm:
        completions = _generate_vllm(args, [row["prompt"] for row in rows], args.samples_per_scenario)
    else:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        completions = _generate_hf(
            model,
            tokenizer,
            [row["prompt"] for row in rows for _ in range(args.samples_per_scenario)],
            args,
        )

    per_scenario_scores: dict[str, list[float]] = {row["id"]: [] for row in rows}
    per_scenario_shaped: dict[str, list[float]] = {row["id"]: [] for row in rows}
    per_scenario_completions: dict[str, list[dict]] = {row["id"]: [] for row in rows}

    # Scoring is pure CPU, and on a right-tool completion `reward()` deep-copies
    # the 2.8MB db inside `execute_against`. At n=16 over the full corpus that
    # runs into thousands of seconds -- all of it after vLLM has torn its engine
    # down and printed its shutdown banner, so it reads exactly like a hung
    # process. Split it across processes and report progress as chunks land.
    n = args.samples_per_scenario
    per_row_completions = [completions[i * n : (i + 1) * n] for i in range(len(rows))]
    payloads = [
        (
            row["expected_tool_name"],
            row["expected_tool_arguments_json"],
            row_completions,
            args.with_shaping,
            True,
        )
        for row, row_completions in zip(rows, per_row_completions)
    ]

    workers = resolve_score_workers(args.score_workers)
    total = len(all_meta)
    report_every = max(1, len(rows) // 20)
    scoring_started = time.perf_counter()
    print(
        f"[zero_shot_baseline] scoring {total} completions across {workers} "
        f"worker{'' if workers == 1 else 's'}",
        flush=True,
    )

    def _progress(done_rows: int) -> None:
        if done_rows % report_every and done_rows != len(rows):
            return
        elapsed = time.perf_counter() - scoring_started
        rate = done_rows / elapsed if elapsed else 0.0
        remaining = (len(rows) - done_rows) / rate if rate else 0.0
        print(
            f"[zero_shot_baseline] scoring {done_rows}/{len(rows)} scenarios "
            f"({done_rows / len(rows):.0%}), ~{remaining / 60:.1f} min left",
            flush=True,
        )

    if workers == 1:
        results = (_score_scenario(payload) for payload in payloads)
    else:
        # One scenario per dispatch. Scenarios cost wildly different amounts
        # -- a wrong-tool group exits before executing anything, a right-tool
        # group runs 16 db diffs -- and the expensive ones cluster, so batching
        # them would leave workers idle while one chews through a slow chunk.
        # The IPC per dispatch is microseconds against that.
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_score_scenario, payloads, chunksize=1)

    try:
        for done, (row, row_completions, (scores, shaped_scores)) in enumerate(
            zip(rows, per_row_completions, results), start=1
        ):
            per_scenario_scores[row["id"]] = scores
            if args.with_shaping:
                per_scenario_shaped[row["id"]] = shaped_scores
            if args.save_completions:
                per_scenario_completions[row["id"]] = [
                    {
                        "score": score,
                        "shaped_score": shaped_scores[i] if args.with_shaping else score,
                        "completion": completion,
                    }
                    for i, (score, completion) in enumerate(zip(scores, row_completions))
                ]
            _progress(done)
    finally:
        if workers != 1:
            pool.shutdown()

    # Difficulty read: for each scenario, is the reward across samples
    # ~constant (near-zero variance -> ~zero GRPO training signal for it)?
    def _zero_variance(table: dict[str, list[float]]) -> list[tuple[str, float]]:
        return [
            (scenario_id, scores[0])
            for scenario_id, scores in table.items()
            if scores and len(set(round(s, 3) for s in scores)) == 1
        ]

    zero_variance_scenarios = _zero_variance(per_scenario_scores)

    all_scores = [s for scores in per_scenario_scores.values() for s in scores]
    histogram = Counter(round(s, 1) for s in all_scores)

    summary = {
        "model": args.model,
        # Recorded so a later run can be checked for split identity rather than
        # assumed to match -- a before/after across two different splits is not
        # a comparison, and nothing else in the output would reveal it.
        "split": args.split,
        "val_fraction": args.val_fraction,
        "category_mix": args.category_mix,
        "curriculum_seed": args.curriculum_seed,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n_scenarios": len(rows),
        "samples_per_scenario": args.samples_per_scenario,
        "temperature": args.temperature,
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "score_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "zero_variance_scenario_count": len(zero_variance_scenarios),
        "zero_variance_scenario_fraction": len(zero_variance_scenarios) / len(rows) if rows else 0.0,
        "per_scenario_scores": per_scenario_scores,
        # Lets scripts/bucket_analysis.py separate scenarios that *can* score
        # between 0 and 1 from ones that are structurally binary (gold is
        # silence), which no sampling or shaping change can ever make
        # middle-difficulty.
        "per_scenario_expected_tool_name": {row["id"]: row["expected_tool_name"] for row in rows},
    }
    if args.with_shaping:
        shaped_dead = _zero_variance(per_scenario_shaped)
        shaped_all = [s for scores in per_scenario_shaped.values() for s in scores]
        summary["shaping"] = {
            "mean_shaped_score": sum(shaped_all) / len(shaped_all) if shaped_all else 0.0,
            "zero_variance_scenario_count": len(shaped_dead),
            "zero_variance_scenario_fraction": len(shaped_dead) / len(rows) if rows else 0.0,
            "scenarios_revived_by_shaping": len(zero_variance_scenarios) - len(shaped_dead),
        }
        summary["per_scenario_shaped_scores"] = per_scenario_shaped
    if args.save_completions:
        summary["per_scenario_completions"] = per_scenario_completions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[zero_shot_baseline] mean score: {summary['mean_score']:.3f}")
    print(f"[zero_shot_baseline] histogram: {summary['score_histogram']}")
    print(
        f"[zero_shot_baseline] {summary['zero_variance_scenario_count']}/{len(rows)} "
        f"({summary['zero_variance_scenario_fraction']:.1%}) scenarios show zero variance "
        f"across {args.samples_per_scenario} samples -- these will contribute ~no GRPO "
        f"training signal at this group size. A high fraction here is the cue to revisit "
        f"scenario difficulty/curriculum before the full run, per docs/phase7_aws_setup.md."
    )
    if args.with_shaping:
        sh = summary["shaping"]
        print(
            f"[zero_shot_baseline] with shaping: mean {sh['mean_shaped_score']:.3f}, "
            f"{sh['zero_variance_scenario_count']}/{len(rows)} "
            f"({sh['zero_variance_scenario_fraction']:.1%}) still zero-variance -- "
            f"{sh['scenarios_revived_by_shaping']} scenarios gained a usable gradient."
        )
    print(f"[zero_shot_baseline] Full detail written to {output_path}")


if __name__ == "__main__":
    main()
