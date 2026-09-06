"""Stratified sample of scenario ids for a faster n=16 audit -- takes N per
category/theme cell so every one of the 30 cells is represented, instead of
random sampling that could skip a whole cell by chance.

Usage: python sample_audit_ids.py [--per-cell N] [--skip-category out_of_scope]
Writes audit_sample_ids.txt in the current directory.
"""
import argparse
import glob
import json
import random
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--per-cell", type=int, default=5, help="How many scenarios to sample per category__theme cell.")
p.add_argument("--skip-category", action="append", default=[], help="Category to exclude entirely (repeatable).")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--output", default="audit_sample_ids.txt")
args = p.parse_args()

random.seed(args.seed)
ids = []
for path in sorted(glob.glob("data/synthetic/raw/*.json")):
    stem = Path(path).stem
    category = stem.split("__", 1)[0]
    if category in args.skip_category:
        continue
    scenarios = json.loads(Path(path).read_text())
    sample = random.sample(scenarios, min(args.per_cell, len(scenarios)))
    ids.extend(s["id"] for s in sample)

Path(args.output).write_text("\n".join(ids) + "\n")
print(f"{len(ids)} ids written to {args.output} "
      f"({args.per_cell}/cell, skipping {args.skip_category or 'nothing'})")
