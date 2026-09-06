#!/usr/bin/env bash
#
# Run every diagnostic over one audit JSON and collect the whole thing into a
# single committable text file.
#
# The audit JSONs themselves are gitignored -- with --save-completions they run
# to hundreds of megabytes -- but the derived report is a few kilobytes and is
# the thing worth keeping and sharing. Writing it to a file rather than the
# terminal also survives a lost tmux scrollback, which is the usual way these
# numbers get read once and then disappear.
#
# Usage:
#   bash scripts/diagnose.sh                              # defaults below
#   bash scripts/diagnose.sh data/trained/audit_n16.json
#   bash scripts/diagnose.sh <audit.json> <report.txt>
#
# Then, to hand the report to someone (or to a Claude session) without pasting
# a screenful into a terminal:
#   git add data/reports && git commit -m "Audit report" && git push
set -euo pipefail

cd "$(dirname "$0")/.."

AUDIT="${1:-data/trained/audit_n16.json}"
OUT="${2:-data/reports/diagnosis.txt}"

if [ ! -f "$AUDIT" ]; then
    echo "No audit at $AUDIT -- pass the path as the first argument." >&2
    exit 1
fi

# tau2's registry dumps its whole domain/agent/task-set inventory through
# loguru at DEBUG on every import. That is four screens of JSON per `uv run`
# here, straight into the middle of the report.
export LOGURU_LEVEL="${LOGURU_LEVEL:-WARNING}"

mkdir -p "$(dirname "$OUT")"
MIX_FILE="$(dirname "$OUT")/recommended_mix.txt"
DEAD_FILE="$(dirname "$OUT")/dead_scenario_ids.txt"

# Everything below writes to $OUT and to the terminal both, so a run you *can*
# watch still shows progress.
{
    echo "=============================================================="
    echo "audit:     $AUDIT"
    echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "commit:    $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "=============================================================="

    echo
    echo "### run settings and headline numbers (from the audit JSON) ###"
    echo
    # Pulled straight out of the JSON rather than re-derived, so the report
    # records the flags the audit actually ran under -- split, mixture and seed
    # decide which scenarios were scored, and a comparison across two reports
    # is void if they differ.
    uv run python - "$AUDIT" <<'PY'
import json, sys

audit = json.load(open(sys.argv[1]))
skip = {
    "per_scenario_scores",
    "per_scenario_shaped_scores",
    "per_scenario_completions",
    "per_scenario_expected_tool_name",
    "score_histogram",
    "zero_variance_scenarios",
}
for key, value in audit.items():
    if key not in skip:
        print(f"{key}: {value}")

hist = audit.get("score_histogram") or {}
if hist:
    total = sum(hist.values()) or 1
    print("\nscore histogram (reward() alone):")
    for bucket, count in sorted(hist.items(), key=lambda kv: float(kv[0])):
        bar = "#" * round(60 * count / max(hist.values()))
        print(f"  {float(bucket):>4.1f}  {count:>6}  {count / total:>5.1%}  {bar}")
PY

    echo
    echo "### scorecard: per-cell yield x headroom, benchmark-weighted ###"
    echo
    uv run python scripts/data_scorecard.py "$AUDIT"

    echo
    echo "### scorecard: same table, unweighted ###"
    echo
    uv run python scripts/data_scorecard.py "$AUDIT" --relevance uniform

    echo
    echo "### scorecard: reward() alone, no shaping ###"
    echo
    uv run python scripts/data_scorecard.py "$AUDIT" --raw

    echo
    echo "### buckets: flat vs varying, before and after shaping ###"
    echo
    uv run python scripts/bucket_analysis.py "$AUDIT"
} 2>&1 | tee "$OUT"

# The two machine-readable side outputs, written separately so they can be fed
# straight back into grpo_train rather than retyped out of the report.
uv run python scripts/data_scorecard.py "$AUDIT" --emit-mix > "$MIX_FILE"
uv run python scripts/data_scorecard.py "$AUDIT" --emit-dead-ids "$DEAD_FILE" > /dev/null

echo
echo "wrote:"
echo "  $OUT             (the full report)"
echo "  $MIX_FILE   (--category-mix string)"
echo "  $DEAD_FILE  (--exclude-zero-variance-from list)"
echo
echo "to share it:"
echo "  git add $(dirname "$OUT") && git commit -m 'Audit report' && git push"
