"""Zero-shot baseline pass: run the base policy model over all 541 synthetic
scenarios with NO training, and report the reward distribution.

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

Usage:
    python -m tau_forge.train.zero_shot_baseline --model Qwen/Qwen3-4B-Instruct-2507
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
    p.add_argument("--samples-per-scenario", type=int, default=4, help="Repeats per scenario, for a rough within-scenario variance read -- not a full GRPO group, just enough to see if a scenario is deterministic.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tau_forge.envs.retail import RetailEnv
    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples, to_hf_rows
    from tau_forge.train.reward_adapter import score_completion

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, tokenize=False, add_generation_prompt=True
    )
    tools = RetailEnv().all_openai_schemas()

    examples = build_examples(data_glob=args.data_glob or DEFAULT_DATA_GLOB)
    rows = to_hf_rows(examples, apply_chat_template, tools)
    print(f"[zero_shot_baseline] {len(rows)} scenarios, {args.samples_per_scenario} samples each "
          f"= {len(rows) * args.samples_per_scenario} total generations.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    per_scenario_scores: dict[str, list[float]] = {row["id"]: [] for row in rows}
    reason_counts: Counter[str] = Counter()

    all_prompts = [row["prompt"] for row in rows for _ in range(args.samples_per_scenario)]
    all_meta = [row for row in rows for _ in range(args.samples_per_scenario)]

    for start in range(0, len(all_prompts), args.batch_size):
        batch_prompts = all_prompts[start : start + args.batch_size]
        batch_meta = all_meta[start : start + args.batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completions = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        for row, completion in zip(batch_meta, completions):
            expected_args = json.loads(row["expected_tool_arguments_json"])
            score = score_completion(completion, row["expected_tool_name"], expected_args)
            per_scenario_scores[row["id"]].append(score)

        print(f"[zero_shot_baseline] {min(start + args.batch_size, len(all_prompts))}/{len(all_prompts)} generations done")

    # Difficulty read: for each scenario, is the reward across samples
    # ~constant (near-zero variance -> ~zero GRPO training signal for it)?
    zero_variance_scenarios = []
    for scenario_id, scores in per_scenario_scores.items():
        if len(set(round(s, 3) for s in scores)) == 1:
            zero_variance_scenarios.append((scenario_id, scores[0]))

    all_scores = [s for scores in per_scenario_scores.values() for s in scores]
    histogram = Counter(round(s, 1) for s in all_scores)

    summary = {
        "model": args.model,
        "n_scenarios": len(rows),
        "samples_per_scenario": args.samples_per_scenario,
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "score_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "zero_variance_scenario_count": len(zero_variance_scenarios),
        "zero_variance_scenario_fraction": len(zero_variance_scenarios) / len(rows) if rows else 0.0,
        "per_scenario_scores": per_scenario_scores,
    }

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
    print(f"[zero_shot_baseline] Full detail written to {output_path}")


if __name__ == "__main__":
    main()
