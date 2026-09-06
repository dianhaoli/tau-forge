"""Which of the 30 scenario cells are actually earning their rollouts, and what
the corpus mixture should be as a result.

Reads a `zero_shot_baseline` output (run it with --with-shaping, so the numbers
reflect what GRPO will really see) and reports, per category__theme cell:

  yield     fraction of scenarios whose group reward varied -> can move weights
  headroom  1 - mean score -> how much reward is still winnable there
  signal    yield x headroom -> rank cells by this

Then it recommends a --category-mix, as measured signal times a benchmark
relevance prior. Both factors print separately so a surprising recommendation
can be traced to whichever one drove it.

Usage:
    python scripts/data_scorecard.py data/trained/zero_shot_baseline.json
    python scripts/data_scorecard.py <audit.json> --emit-mix
    python scripts/data_scorecard.py <audit.json> --emit-dead-ids dead.txt
    python scripts/data_scorecard.py <audit.json> --relevance uniform
"""

import argparse
import sys

from tau_forge.train.scorecard import (
    BENCHMARK_RELEVANCE,
    LOW_YIELD,
    UNIFORM_RELEVANCE,
    category_signal,
    dead_scenario_ids,
    load_scores,
    recommend_mix,
    score_cells,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audit", help="A zero_shot_baseline output JSON.")
    p.add_argument("--relevance", choices=["benchmark", "uniform"], default="benchmark")
    p.add_argument("--raw", action="store_true", help="Use reward() alone even if shaped scores exist.")
    p.add_argument("--emit-mix", action="store_true", help="Print only the --category-mix string, for piping.")
    p.add_argument("--emit-dead-ids", metavar="PATH", help="Write cold-start scenario ids to a file.")
    p.add_argument("--include-solved", action="store_true", help="With --emit-dead-ids, also list flat-1.0 and flat-partial scenarios.")
    args = p.parse_args()

    scores, basis = load_scores(args.audit, prefer_shaped=not args.raw)
    cells = score_cells(scores)
    relevance = BENCHMARK_RELEVANCE if args.relevance == "benchmark" else UNIFORM_RELEVANCE
    mix = recommend_mix(cells, relevance=relevance)

    if args.emit_mix:
        print(",".join(f"{c}={v}" for c, v in sorted(mix.items())))
        return

    total = sum(c.n for c in cells)
    print(f"Scored {total} scenarios in {len(cells)} cells, on {basis}.\n")

    header = f"{'cell':52} {'n':>4} {'mean':>6} {'yield':>6} {'head':>6} {'signal':>7}  usable/cold/solved/stuck"
    print(header)
    print("-" * len(header))
    for cell in sorted(cells, key=lambda c: -c.signal):
        flag = "  <- low yield" if cell.yield_ < LOW_YIELD else ""
        print(
            f"{cell.cell:52} {cell.n:4} {cell.mean_score:6.3f} {cell.yield_:6.2f} "
            f"{cell.headroom:6.2f} {cell.signal:7.3f}  "
            f"{cell.n_usable}/{cell.n_cold}/{cell.n_solved}/{cell.n_stuck}{flag}"
        )

    print("\nPer category:")
    signal = category_signal(cells)
    print(f"  {'category':28} {'signal':>7} {'relevance':>10} {'recommended share':>18}")
    for category in sorted(mix, key=lambda c: -mix[c]):
        print(f"  {category:28} {signal.get(category, 0.0):7.3f} {relevance.get(category, 0.0):10.2f} {mix[category]:17.1%}")

    print("\nRecommended mixture:")
    print("  --category-mix " + ",".join(f"{c}={v}" for c, v in sorted(mix.items())))

    cold = dead_scenario_ids(scores, include_solved=False)
    print(f"\n{len(cold)} cold-start scenarios ({len(cold) / total:.1%}) produce no gradient at any")
    print("temperature or group size. Shaping is the first thing to try on them; what survives")
    print("that needs prompting or an SFT warm-start, not more sampling.")

    if args.emit_dead_ids:
        ids = dead_scenario_ids(scores, include_solved=args.include_solved)
        with open(args.emit_dead_ids, "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"\nWrote {len(ids)} ids to {args.emit_dead_ids} (feed to --exclude-zero-variance-from's")
        print("source audit, or use directly with sample_audit_ids-style filtering).")


if __name__ == "__main__":
    sys.exit(main())
