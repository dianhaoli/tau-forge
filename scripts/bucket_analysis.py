"""Bucket every scenario in a zero_shot_baseline output into cold-start /
already-solved / stuck-partial / has-variance, and report what would change
under the shaping reward -- to find *why* variance does or doesn't show up,
not just how much.

Run the audit with --with-shaping --save-completions and this reports both
columns side by side: how many groups are flat under reward() alone, and how
many are still flat once tau_forge.train.shaping's partial credit is added.
The difference is the number of dead groups shaping revives.

It also reports each scenario's **reward granularity** -- whether an
intermediate score is even reachable for it. A scenario whose gold action is
"no tool call" is graded 1.0 or 0.0 with nothing in between, for any completion
the model could possibly produce. No amount of temperature, group size or
shaping makes such a scenario middle-difficulty; only changing the corpus
mixture reduces how much of the run they consume.

Usage: python scripts/bucket_analysis.py data/trained/audit_sample_n16.json
"""

import json
import sys
from collections import Counter, defaultdict

BINARY = "binary (gold is silence: 0.0 or 1.0 only)"
COARSE = "coarse (READ tool: output match -> 1.0, else arg_match)"
GRADED = "graded (partial credit reachable)"


def granularity(expected_tool_name):
    """Which reward tiers this scenario can reach. Mirrors reward.reward()'s
    branch structure; imports tau2 only if it is available so the script still
    runs as a plain JSON report where it is not."""
    if expected_tool_name is None:
        return BINARY
    try:
        from tau_forge.envs.retail import RetailEnv
        from tau_forge.reward.reward import OUTPUT_DETERMINES_CORRECTNESS
    except ImportError:
        return GRADED
    if expected_tool_name in OUTPUT_DETERMINES_CORRECTNESS and not RetailEnv().tool_mutates_state(
        expected_tool_name
    ):
        return COARSE
    return GRADED


def bucket_of(scores):
    if len(set(round(s, 3) for s in scores)) > 1:
        return "has_variance"
    value = round(scores[0], 2)
    if value >= 1.0:
        return "already_solved (1.0)"
    if value <= 0.0:
        return "cold_start (0.0)"
    return f"stuck_partial ({value})"


def bucketize(table):
    buckets = defaultdict(list)
    for scenario_id, scores in table.items():
        if scores:
            buckets[bucket_of(scores)].append((scenario_id, scenario_id.split("__")[0], scores))
    return buckets


def report(title, buckets, total):
    print(f"\n=== {title} (of {total} scenarios) ===")
    for bucket, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(items):4} ({len(items) / total:5.1%})  {bucket}")
        for category, n in Counter(cat for _, cat, _ in items).most_common():
            print(f"           {n:4}  {category}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/trained/audit_sample_n16.json"
    data = json.load(open(path))
    raw = data["per_scenario_scores"]
    total = len(raw)
    shaped = data.get("per_scenario_shaped_scores")
    completions = data.get("per_scenario_completions")

    raw_buckets = bucketize(raw)
    report("reward() alone", raw_buckets, total)

    if shaped:
        shaped_buckets = bucketize(shaped)
        report("reward() + shaping -- what GRPO actually sees", shaped_buckets, total)
        dead_before = sum(len(v) for k, v in raw_buckets.items() if k != "has_variance")
        dead_after = sum(len(v) for k, v in shaped_buckets.items() if k != "has_variance")
        print(
            f"\n  flat groups: {dead_before} -> {dead_after} "
            f"({dead_before - dead_after} revived, {(dead_before - dead_after) / total:.1%} of the corpus)"
        )
    else:
        print("\nNo per_scenario_shaped_scores -- rerun the audit with --with-shaping to see")
        print("how many of these flat groups the shaping reward revives.")

    # What is structurally incapable of a middle score, regardless of anything.
    expected = data.get("per_scenario_expected_tool_name")
    if expected:
        print(f"\n=== Reward granularity (what is reachable at all) ===")
        counts = Counter(granularity(expected.get(sid)) for sid in raw)
        for kind, n in counts.most_common():
            print(f"  {n:4} ({n / total:5.1%})  {kind}")
        binary_flat = [
            sid
            for sid, scores in raw.items()
            if granularity(expected.get(sid)) == BINARY and len(set(round(s, 3) for s in scores)) == 1
        ]
        print(
            f"\n  {len(binary_flat)} of the flat groups are structurally binary -- no completion "
            "could have scored between 0 and 1.\n  Temperature, group size and shaping cannot help "
            "these. Only --category-mix reduces their share of the run."
        )

    if completions:
        print("\n=== Example completions ===")
        for bucket in ["has_variance", "cold_start (0.0)"]:
            print(f"\n--- {bucket} (up to 3 scenarios) ---")
            for sid, _, scores in raw_buckets.get(bucket, [])[:3]:
                print(f"\n  {sid}  scores={scores}")
                seen = {}
                for c in completions[sid]:
                    seen.setdefault(round(c["score"], 2), c["completion"])
                for value, text in seen.items():
                    print(f"    [score={value}] {text[:300].replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
