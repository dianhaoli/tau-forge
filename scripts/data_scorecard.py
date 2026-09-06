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
from collections import Counter

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


# Gold actions that do not solve the customer's problem with a retail tool.
# `None` is "stay silent and ask"; `transfer_to_human_agents` is "escalate".
# Both are correct behaviours the model has to keep, and both are the wrong
# thing to spend a training run on: 110 of the 114 real tau2 retail tasks are
# solved by executing domain tools, so a corpus weighted toward these teaches
# the one behaviour most likely to lower the benchmark score.
NON_SOLVING_GOLD = {None, "transfer_to_human_agents"}


def _composition(examples) -> tuple[int, int]:
    non_solving = sum(1 for e in examples if e.expected_tool_name in NON_SOLVING_GOLD)
    return len(examples), non_solving


def simulate_mix(args, scores: dict[str, list[float]], basis: str) -> None:
    """Report what a candidate mixture would actually produce, measured against
    this audit's per-scenario scores rather than assumed from category labels."""
    from tau_forge.train.curriculum import apply_mixture
    from tau_forge.train.dataset import DEFAULT_DATA_GLOB, build_examples
    from tau_forge.train.grpo_train import resolve_mix
    from tau_forge.train.scorecard import classify

    corpus = build_examples(data_glob=DEFAULT_DATA_GLOB)
    # resolve_mix handles both the built-in names and a literal spec.
    mix = resolve_mix(args.simulate_mix)

    pool = corpus
    if args.simulate_exclude_solved:
        pool = [e for e in corpus if classify(scores.get(e.id, [])) != "already_solved"]
    kept = apply_mixture(pool, mix, seed=0)

    def report(label: str, examples) -> dict[str, int]:
        n, non_solving = _composition(examples)
        buckets = Counter(classify(scores[e.id]) for e in examples if e.id in scores)
        usable = buckets.get("usable", 0)
        print(f"\n{label}: {n} scenarios")
        print(f"  gradient-carrying          {usable:4} ({usable / n:5.1%})")
        for bucket in ("cold_start", "already_solved", "stuck_partial"):
            count = buckets.get(bucket, 0)
            print(f"  {bucket:26} {count:4} ({count / n:5.1%})")
        print(f"  gold is silence or escalate{non_solving:4} ({non_solving / n:5.1%})")
        return buckets

    print(f"Simulating --category-mix on {basis}.")
    print(f"  {','.join(f'{c}={v:g}' for c, v in sorted(mix.items()))}")
    if args.simulate_exclude_solved:
        print(f"  with --exclude-solved: {len(corpus) - len(pool)} flat-1.0 scenarios dropped first")

    report("corpus as generated", corpus)
    after = report("after the mixture", kept)

    print("\nper category, after the mixture:")
    per_cat = Counter(e.category for e in kept)
    for category, n in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        in_cat = [e for e in kept if e.category == category]
        usable = sum(1 for e in in_cat if e.id in scores and classify(scores[e.id]) == "usable")
        print(f"  {category:28} {n:4} ({n / len(kept):5.1%})   {usable:4} carry gradient")

    steps = after.get("usable", 0)
    print(
        f"\n{steps} of {len(kept)} scenarios ({steps / len(kept):.1%}) would produce a gradient. "
        "The rest occupy a slot in a step and contribute nothing."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audit", help="A zero_shot_baseline output JSON.")
    p.add_argument("--relevance", choices=["benchmark", "uniform"], default="benchmark")
    p.add_argument("--raw", action="store_true", help="Use reward() alone even if shaped scores exist.")
    p.add_argument("--emit-mix", action="store_true", help="Print only the --category-mix string, for piping.")
    p.add_argument("--emit-dead-ids", metavar="PATH", help="Write cold-start scenario ids to a file.")
    p.add_argument("--include-solved", action="store_true", help="With --emit-dead-ids, also list flat-1.0 and flat-partial scenarios.")
    p.add_argument(
        "--simulate-mix",
        metavar="SPEC",
        help="Score a candidate --category-mix against this audit instead of recommending one: "
        "apply the mixture to the corpus, then report what the resulting training set would "
        "actually contain -- how many groups carry gradient, and what share of it teaches the "
        "model not to solve the task. Accepts the same spec grpo_train takes, or the name of a "
        "built-in mix. Repeatable, comma-separated: happy_path=0.36,...",
    )
    p.add_argument(
        "--simulate-exclude-solved",
        action="store_true",
        help="With --simulate-mix, drop scenarios flat at 1.0 before applying the mixture, "
        "the way grpo_train --exclude-solved does.",
    )
    args = p.parse_args()

    scores, basis = load_scores(args.audit, prefer_shaped=not args.raw)
    cells = score_cells(scores)
    relevance = BENCHMARK_RELEVANCE if args.relevance == "benchmark" else UNIFORM_RELEVANCE
    mix = recommend_mix(cells, relevance=relevance)

    if args.simulate_mix:
        simulate_mix(args, scores, basis)
        return

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
