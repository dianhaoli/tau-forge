"""Bucket every scenario in a zero-shot baseline output (run with
--save-completions) into cold-start / already-solved / stuck-partial /
has-variance, broken down by category, with example completions from each
bucket -- to find *why* variance does or doesn't show up, not just how much.

Usage: python bucket_analysis.py data/trained/audit_sample_n16.json
"""
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "data/trained/audit_sample_n16.json"
d = json.load(open(path))

has_completions = "per_scenario_completions" in d

buckets = defaultdict(list)  # bucket_name -> [scenario_id, ...]
for sid, scores in d["per_scenario_scores"].items():
    rounded = set(round(s, 3) for s in scores)
    category = sid.split("__")[0]
    if len(rounded) == 1:
        val = round(scores[0], 2)
        if val == 1.0:
            bucket = "already_solved (1.0)"
        elif val == 0.0:
            bucket = "cold_start (0.0)"
        else:
            bucket = f"stuck_partial ({val})"
    else:
        bucket = "has_variance"
    buckets[bucket].append((sid, category, scores))

print(f"=== Bucket sizes (of {len(d['per_scenario_scores'])} scenarios) ===")
for bucket, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    print(f"  {bucket}: {len(items)}")

print()
print("=== By category within each bucket ===")
for bucket, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    cat_counts = Counter(cat for _, cat, _ in items)
    print(f"  {bucket}:")
    for cat, n in cat_counts.most_common():
        print(f"    {cat}: {n}")

if has_completions:
    print()
    print("=== Example completions ===")
    for bucket in ["has_variance", "cold_start (0.0)"]:
        items = buckets.get(bucket, [])
        print(f"\n--- {bucket} (showing up to 3 scenarios) ---")
        for sid, category, scores in items[:3]:
            print(f"\n  {sid}  scores={scores}")
            comps = d["per_scenario_completions"][sid]
            # show one completion per distinct score value to see the contrast
            seen_scores = {}
            for c in comps:
                seen_scores.setdefault(round(c["score"], 2), c["completion"])
            for score_val, comp in seen_scores.items():
                print(f"    [score={score_val}] {comp[:300].replace(chr(10), ' | ')}")
else:
    print()
    print("No per_scenario_completions in this file -- rerun with --save-completions to see example text.")
