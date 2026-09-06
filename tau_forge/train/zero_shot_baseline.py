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
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "trained" / "zero_shot_baseline.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--data-glob", default=None)
    p.add_argument("--category", default=None, help="Only run scenarios whose id starts with this category (e.g. 'out_of_scope').")
    p.add_argument("--scenario-ids-file", default=None, help="Path to a text file of scenario ids (one per line) to restrict the run to.")
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
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


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


def main() -> None:
    args = parse_args()

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples, to_hf_rows
    from tau_forge.train.reward_adapter import score_completion
    from tau_forge.train.shaping import shaping_score
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

    shaping_env = None
    if args.with_shaping:
        shaping_env = RetailEnv()

    per_scenario_scores: dict[str, list[float]] = {row["id"]: [] for row in rows}
    per_scenario_shaped: dict[str, list[float]] = {row["id"]: [] for row in rows}
    per_scenario_completions: dict[str, list[dict]] = {row["id"]: [] for row in rows}
    for row, completion in zip(all_meta, completions):
        expected_args = json.loads(row["expected_tool_arguments_json"])
        score = score_completion(completion, row["expected_tool_name"], expected_args)
        per_scenario_scores[row["id"]].append(score)
        shaped = score
        if shaping_env is not None:
            shaped = score + shaping_score(
                completion, row["expected_tool_name"], expected_args, shaping_env
            )
            per_scenario_shaped[row["id"]].append(shaped)
        if args.save_completions:
            per_scenario_completions[row["id"]].append(
                {"score": score, "shaped_score": shaped, "completion": completion}
            )

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
