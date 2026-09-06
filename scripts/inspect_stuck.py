"""Diagnose *why* a band of scenarios is flat, by re-grading its saved
completions and reporting the reward reasons behind the score.

The scorecard says how much of the corpus produces no gradient. It does not say
what the model is doing wrong, and the two dead bands need opposite fixes:

  cold_start (flat 0.0)     the model never reaches the right tool
  stuck_partial (flat 0.2)  it reaches the right tool every time and the call
                            is rejected before execution -- schema-invalid
                            arguments, hallucinated arguments, or an expected
                            failure that succeeded

A band flat at 0.2 is the more tractable of the two: the policy is one argument
away from scoring, and shaping can only be designed for it once the actual
argument errors are known. That is what this prints -- which reason fires, and
for schema failures, which arguments are missing, extra or wrong-typed.

Requires an audit run with --save-completions (the completions are what get
re-graded; the scores alone can't say why).

Usage:
    python scripts/inspect_stuck.py data/trained/audit_n16.json
    python scripts/inspect_stuck.py data/trained/audit_n16.json --score 0.0
    python scripts/inspect_stuck.py data/trained/audit_n16.json --examples 10
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("audit", help="A zero_shot_baseline output JSON, run with --save-completions.")
    p.add_argument(
        "--score",
        type=float,
        default=0.2,
        help="Which flat band to inspect. 0.2 (default) is the schema-invalid tier; "
        "0.0 is cold start; 1.0 is already solved.",
    )
    p.add_argument(
        "--tolerance", type=float, default=0.001, help="How close to --score counts as that band."
    )
    p.add_argument("--examples", type=int, default=6, help="Completions to print verbatim.")
    p.add_argument(
        "--category", help="Restrict to scenario ids starting with this, e.g. happy_path."
    )
    return p.parse_args(argv)


def flat_at(scores: list[float], target: float, tolerance: float) -> bool:
    return bool(scores) and all(abs(s - target) <= tolerance for s in scores)


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.audit).read_text())

    completions = data.get("per_scenario_completions")
    if not completions:
        raise SystemExit(
            "This audit has no saved completions -- re-run it with --save-completions. "
            "Scores alone cannot say why a band is flat."
        )

    from tau_forge.reward.reward import Action, reward
    from tau_forge.train.completion_parsing import parse_completion
    from tau_forge.train.reward_adapter import _get_shared_db

    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples

    gold = {e.id: e for e in build_examples(data_glob=DEFAULT_DATA_GLOB)}
    db = _get_shared_db()

    scores_by_id = data["per_scenario_scores"]
    band = [
        sid
        for sid, scores in scores_by_id.items()
        if flat_at(scores, args.score, args.tolerance)
        and (not args.category or sid.startswith(args.category))
    ]
    if not band:
        raise SystemExit(f"No scenarios flat at {args.score}.")

    print(f"{len(band)} scenarios flat at {args.score}", end="")
    if args.category:
        print(f" in {args.category}", end="")
    print(f", of {len(scores_by_id)} scored.\n")

    print("by category:")
    for cat, n in Counter(sid.split("__")[0] for sid in band).most_common():
        print(f"  {n:4}  {cat}")

    print("\nby gold tool:")
    for tool, n in Counter(
        str(gold[sid].expected_tool_name) for sid in band if sid in gold
    ).most_common(10):
        print(f"  {n:4}  {tool}")

    reasons: Counter = Counter()
    missing_args: Counter = Counter()
    extra_args: Counter = Counter()
    type_errors: Counter = Counter()
    predicted_tools: Counter = Counter()
    right_tool = wrong_tool = 0
    examples: list[tuple[str, str, str, dict]] = []

    for sid in band:
        example = gold.get(sid)
        if example is None:
            continue
        expected = Action(
            tool_name=example.expected_tool_name, tool_input=example.expected_tool_arguments
        )
        for record in completions[sid]:
            text = record["completion"]
            name, parsed = parse_completion(text)
            predicted_tools[str(name)] += 1
            if name == example.expected_tool_name:
                right_tool += 1
            else:
                wrong_tool += 1

            breakdown = reward(Action(tool_name=name, tool_input=parsed), expected, db)
            reasons[breakdown.reason] += 1

            detail = breakdown.detail or {}
            for arg in detail.get("extra_args") or []:
                extra_args[arg] += 1
            # pydantic reports one line per rejected field; the field name is
            # the part worth counting, not the full message.
            error = detail.get("schema_error") or ""
            for field in re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\n", error, re.MULTILINE):
                if "missing" in error.lower():
                    missing_args[field] += 1
                else:
                    type_errors[field] += 1

            if len(examples) < args.examples:
                examples.append((sid, text, breakdown.reason, detail))

    print(f"\nof {right_tool + wrong_tool} completions in this band: "
          f"{right_tool} called the gold tool, {wrong_tool} did not.")

    print("\nreward reasons:")
    total = sum(reasons.values()) or 1
    for reason, n in reasons.most_common():
        print(f"  {n:5}  {n / total:5.1%}  {reason}")

    for label, counter in (
        ("missing / rejected required arguments", missing_args),
        ("wrong-typed arguments", type_errors),
        ("hallucinated (extra) arguments", extra_args),
    ):
        if counter:
            print(f"\n{label}:")
            for arg, n in counter.most_common(10):
                print(f"  {n:5}  {arg}")

    print("\npredicted tools in this band:")
    for tool, n in predicted_tools.most_common(8):
        print(f"  {n:5}  {tool}")

    print(f"\n=== {len(examples)} example completions ===")
    for sid, text, reason, detail in examples:
        example = gold[sid]
        print(f"\n--- {sid}  [{reason}]")
        print(f"    gold: {example.expected_tool_name}({json.dumps(example.expected_tool_arguments)[:220]})")
        print(f"    said: {' | '.join(text.strip().splitlines())[:400]}")
        if detail.get("schema_error"):
            print(f"    schema_error: {' '.join(str(detail['schema_error']).split())[:300]}")
        if detail.get("extra_args"):
            print(f"    extra_args: {detail['extra_args']}")


if __name__ == "__main__":
    main()
