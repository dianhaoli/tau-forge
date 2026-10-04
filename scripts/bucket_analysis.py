"""Bucket every scenario in a zero_shot_baseline output into cold-start /
already-solved / stuck-partial / has-variance, and report what would change
under the shaping reward -- to find *why* variance does or doesn't show up,
not just how much.

Run the audit with --with-shaping --save-completions and this reports both
columns side by side: how many groups are flat under reward() alone, and how
many are still flat once tau_forge.train.shaping's partial credit is added.
The difference is the number of dead groups shaping revives.

"has_variance" is the old, permissive test -- any two samples differing at 3
decimals -- so fifteen 0.0s and one 0.02 count as alive. Each table is
therefore followed by the **effective** count (group std >= --min-std, default
0.05; see tau_forge/train/scorecard.py, "Effective variance"): that is the
number to read for how much of the run carries real gradient, and the gap is
the groups shaping only nudged.

When the audit recorded label-audit blocking labels (zero_shot_baseline does
so by default) -- or, for an older audit, from --label-audit -- the variance
numbers are also split into clean vs label-defective scenarios. A defective
gold makes every policy-following sample score the same 0, so a corpus-wide
fraction mixes two populations that call for different fixes.

It also reports each scenario's **reward granularity** -- whether an
intermediate score is even reachable for it. A scenario whose gold action is
"no tool call" is graded 1.0 or 0.0 with nothing in between, for any completion
the model could possibly produce. No amount of temperature, group size or
shaping makes such a scenario middle-difficulty; only changing the corpus
mixture reduces how much of the run they consume.

Usage: python scripts/bucket_analysis.py data/trained/audit_sample_n16.json [--min-std 0.05]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tau_forge.train.scorecard import (
    EFFECTIVE_MIN_STD,
    has_any_variance,
    variance_by_label_status,
    variance_summary,
)

BINARY = "binary (gold is silence: 0.0 or 1.0 only)"
COARSE = "coarse (READ tool: output match -> 1.0, else arg_match)"
GRADED = "graded (partial credit reachable)"

DEFAULT_LABEL_AUDIT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "label_audit.json"


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
    if has_any_variance(scores):
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


def effective_lines(table, min_std, blocking=None):
    """The effective-variance lines printed under each bucket table. Returned
    rather than printed so the numbers are testable."""
    s = variance_summary(table, min_std)
    lines = [
        f"  effective variance (group std >= {min_std}): {s['effective_variance_count']} "
        f"({s['effective_variance_fraction']:.1%}); has_variance above also counts "
        f"{s['micro_variance_count']} micro-variance groups below that."
    ]
    if blocking is not None:
        for status, stats in variance_by_label_status(table, blocking, min_std).items():
            lines.append(
                f"    {status:12} n={stats['n']:4}  zero-variance {stats['zero_variance_fraction']:6.1%}"
                f"  effective {stats['effective_variance_fraction']:6.1%}"
            )
    return lines


def resolve_blocking(data, label_audit):
    """The audit's own record of blocking labels if it has one, else the
    label audit file (for audits taken before they were recorded), else None."""
    recorded = data.get("per_scenario_blocking_labels")
    if recorded is not None:
        return recorded, "recorded in the audit"
    if label_audit and Path(label_audit).exists():
        from tau_forge.train.curriculum import load_blocking_labels

        return load_blocking_labels(label_audit), f"{label_audit} (not recorded in the audit)"
    return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audit", nargs="?", default="data/trained/audit_sample_n16.json")
    p.add_argument("--min-std", type=float, default=EFFECTIVE_MIN_STD)
    p.add_argument(
        "--label-audit",
        default=str(DEFAULT_LABEL_AUDIT),
        help="Fallback source of blocking labels for an audit that did not record them. '' to skip.",
    )
    args = p.parse_args(argv)

    data = json.load(open(args.audit))
    raw = data["per_scenario_scores"]
    total = len(raw)
    shaped = data.get("per_scenario_shaped_scores")
    completions = data.get("per_scenario_completions")
    blocking, blocking_source = resolve_blocking(data, args.label_audit)
    if blocking is not None:
        n_defect = sum(1 for sid in raw if blocking.get(sid))
        print(f"label-audit blocking labels from {blocking_source}: {n_defect} of {total} scored scenarios defective")

    raw_buckets = bucketize(raw)
    report("reward() alone", raw_buckets, total)
    print("\n".join(effective_lines(raw, args.min_std, blocking)))

    if shaped:
        shaped_buckets = bucketize(shaped)
        report("reward() + shaping -- what GRPO actually sees", shaped_buckets, total)
        print("\n".join(effective_lines(shaped, args.min_std, blocking)))
        dead_before = sum(len(v) for k, v in raw_buckets.items() if k != "has_variance")
        dead_after = sum(len(v) for k, v in shaped_buckets.items() if k != "has_variance")
        eff_before = variance_summary(raw, args.min_std)["effective_variance_count"]
        eff_after = variance_summary(shaped, args.min_std)["effective_variance_count"]
        print(
            f"\n  flat groups: {dead_before} -> {dead_after} "
            f"({dead_before - dead_after} revived, {(dead_before - dead_after) / total:.1%} of the corpus)"
        )
        print(
            f"  effectively varying: {eff_before} -> {eff_after} "
            f"({eff_after - eff_before} revived past std {args.min_std})"
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
            if granularity(expected.get(sid)) == BINARY and not has_any_variance(scores)
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
