"""Pull the per-step metric dicts that TRL prints to stdout out of a training log.

    python scripts/extract_train_metrics.py RUN_DIR/train.log OUT/metrics.jsonl
"""
import ast
import json
import re
import sys

pat = re.compile(r"\{'loss'.*?\}")
rows = []
for line in open(sys.argv[1], errors="replace"):
    for m in pat.finditer(line):
        d = ast.literal_eval(m.group(0))
        rows.append({k: (float(v) if isinstance(v, str) and re.fullmatch(r"[-+0-9.eEinfa]+", v) else v) for k, v in d.items()})
with open(sys.argv[2], "w") as f:
    for i, d in enumerate(rows, 1):
        f.write(json.dumps({"step": i, **d}) + "\n")
print(f"{len(rows)} steps -> {sys.argv[2]}")
