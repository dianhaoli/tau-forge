"""Which of the 30 scenario cells are actually earning their rollouts, and what
the corpus mixture should be as a result.

Reads a `zero_shot_baseline` output (run it with --with-shaping, so the numbers
reflect what GRPO will really see) and reports, per category__theme cell:

  yield     fraction of scenarios whose group reward std >= --min-std (0.05)
            -> can move weights. "any" is the older varied-at-all fraction;
            the gap is micro-variance groups (e.g. one shaping-only 0.02)
  headroom  1 - mean score -> how much reward is still winnable there
  signal    yield x headroom -> rank cells by this

Then it recommends a --category-mix, as measured signal times a benchmark
relevance prior. Both factors print separately so a surprising recommendation
can be traced to whichever one drove it.

Label-audit defects (--label-audit, 222 of 541 as committed) are dropped
before the cells are scored and the mix is recommended, as grpo_train drops
them before training: a mixture driven by scenarios the trainer never sees is
not a recommendation for the trainer. On an audit where the defects sat flat
at 0, scoring them in moved the recommendation by up to 9 points
(policy_violation 0.150 -> 0.237). --keep-label-defects scores the whole
audit, for a corpus trained with grpo_train --keep-label-defects; without it,
the full-audit recommendation still prints under the real one, for
comparison. When blocking labels are available (recorded in the audit by
zero_shot_baseline, else read from --label-audit) it also reports zero- and
effective-variance separately for clean vs label-defective scenarios.

Usage:
    python scripts/data_scorecard.py data/trained/zero_shot_baseline.json
    python scripts/data_scorecard.py <audit.json> --emit-mix
    python scripts/data_scorecard.py <audit.json> --emit-dead-ids dead.txt
    python scripts/data_scorecard.py <audit.json> --relevance uniform
    python scripts/data_scorecard.py <audit.json> --keep-label-defects
"""

import argparse
import sys
from collections import Counter

from tau_forge.train.curriculum import DEFAULT_LABEL_AUDIT, load_blocking_labels, load_label_defect_ids
from tau_forge.train.scorecard import (
    BENCHMARK_RELEVANCE,
    EFFECTIVE_MIN_STD,
    LOW_YIELD,
    UNIFORM_RELEVANCE,
    category_signal,
    dead_scenario_ids,
    has_effective_variance,
    load_blocking_labels as load_recorded_blocking,
    load_scores,
    recommend_mix,
    score_cells,
    variance_by_label_status,
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


def trainer_label_defects(args) -> set[str]:
    """The ids grpo_train drops by default given the same --label-audit /
    --keep-label-defects: read from the label audit itself, not from labels
    the audit recorded, because the label audit is what the trainer reads."""
    if not args.label_audit or args.keep_label_defects:
        return set()
    return load_label_defect_ids(args.label_audit)


def split_scored(
    scores: dict[str, list[float]], defects: set[str]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """`(trained_on, left_out)`: the scores grpo_train would see vs the label
    defects it drops. `left_out` is empty when nothing is dropped."""
    kept = {sid: v for sid, v in scores.items() if sid not in defects}
    return kept, {sid: v for sid, v in scores.items() if sid in defects}


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
    # grpo_train drops label-audit defects by default, so the simulation does
    # too -- otherwise it scores a corpus the trainer never sees.
    pool = [e for e in pool if e.id not in trainer_label_defects(args)]
    n_defects_dropped = len(corpus) - len(pool)
    if args.simulate_exclude_solved:
        pool = [e for e in pool if classify(scores.get(e.id, [])) != "already_solved"]
    kept = apply_mixture(pool, mix, seed=0)

    def effective(e) -> bool:
        return e.id in scores and has_effective_variance(scores[e.id], args.min_std)

    def report(label: str, examples) -> dict[str, int]:
        n, non_solving = _composition(examples)
        buckets = Counter(classify(scores[e.id]) for e in examples if e.id in scores)
        buckets["effective"] = sum(1 for e in examples if effective(e))
        usable = buckets.get("usable", 0)
        print(f"\n{label}: {n} scenarios")
        print(f"  {'gradient-carrying':26} {buckets['effective']:4} ({buckets['effective'] / n:5.1%})  std >= {args.min_std}")
        print(f"  {'varied at all':26} {usable:4} ({usable / n:5.1%})")
        for bucket in ("cold_start", "already_solved", "stuck_partial"):
            count = buckets.get(bucket, 0)
            print(f"  {bucket:26} {count:4} ({count / n:5.1%})")
        print(f"  gold is silence or escalate{non_solving:4} ({non_solving / n:5.1%})")
        return buckets

    print(f"Simulating --category-mix on {basis}.")
    print(f"  {','.join(f'{c}={v:g}' for c, v in sorted(mix.items()))}")
    if n_defects_dropped:
        print(f"  {n_defects_dropped} label-audit defects dropped first, as grpo_train does by default")
    if args.simulate_exclude_solved:
        n_solved = len(corpus) - n_defects_dropped - len(pool)
        print(f"  with --exclude-solved: {n_solved} flat-1.0 scenarios dropped first")

    report("corpus as generated", corpus)
    after = report("after the mixture", kept)

    print("\nper category, after the mixture:")
    per_cat = Counter(e.category for e in kept)
    for category, n in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        in_cat = [e for e in kept if e.category == category]
        usable = sum(1 for e in in_cat if effective(e))
        print(f"  {category:28} {n:4} ({n / len(kept):5.1%})   {usable:4} carry gradient")

    steps = after.get("effective", 0)
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
    p.add_argument("--include-solved", action="store_true", help="With --emit-dead-ids, also list solved scenarios (std < --min-std, mean within --min-std of 1.0), as grpo_train --exclude-solved does.")
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
    p.add_argument(
        "--keep-label-defects",
        "--simulate-keep-label-defects",
        dest="keep_label_defects",
        action="store_true",
        help="Score, recommend and simulate over the label-audit defects too, instead of dropping "
        "them the way grpo_train does by default. The counterpart of grpo_train "
        "--keep-label-defects; pass it here when you pass it there.",
    )
    p.add_argument(
        "--min-std",
        type=float,
        default=EFFECTIVE_MIN_STD,
        help="Group reward std at or above which a scenario counts toward yield. See "
        "tau_forge/train/scorecard.py, 'Effective variance'.",
    )
    p.add_argument(
        "--label-audit",
        default=str(DEFAULT_LABEL_AUDIT),
        help="Label audit whose 'blocking' scenarios are dropped before scoring, as grpo_train's "
        "--label-audit (same default). Also the blocking labels for the clean vs label-defect "
        "breakdown when the audit did not record per_scenario_blocking_labels. '' to drop nothing "
        "and skip the breakdown.",
    )
    args = p.parse_args()

    all_scores, basis = load_scores(args.audit, prefer_shaped=not args.raw)
    relevance = BENCHMARK_RELEVANCE if args.relevance == "benchmark" else UNIFORM_RELEVANCE

    if args.simulate_mix:
        simulate_mix(args, all_scores, basis)
        return

    scores, dropped = split_scored(all_scores, trainer_label_defects(args))
    cells = score_cells(scores, min_std=args.min_std)
    mix = recommend_mix(cells, relevance=relevance)

    if args.emit_mix:
        print(",".join(f"{c}={v}" for c, v in sorted(mix.items())))
        return

    total = sum(c.n for c in cells)
    print(f"Scored {total} scenarios in {len(cells)} cells, on {basis}.")
    if dropped:
        print(
            f"{len(dropped)} label-audit defects ({args.label_audit}) left out before scoring, as "
            "grpo_train drops them by default; --keep-label-defects to score them too."
        )
    print()

    print(f"yield = group std >= {args.min_std}; any = varied at all (the pre-effective-variance yield).\n")
    header = (
        f"{'cell':52} {'n':>4} {'mean':>6} {'yield':>6} {'any':>5} {'head':>6} {'signal':>7}"
        "  usable/cold/solved/stuck"
    )
    print(header)
    print("-" * len(header))
    for cell in sorted(cells, key=lambda c: -c.signal):
        flag = "  <- low yield" if cell.yield_ < LOW_YIELD else ""
        print(
            f"{cell.cell:52} {cell.n:4} {cell.mean_score:6.3f} {cell.yield_:6.2f} "
            f"{cell.yield_any_variance:5.2f} {cell.headroom:6.2f} {cell.signal:7.3f}  "
            f"{cell.n_usable}/{cell.n_cold}/{cell.n_solved}/{cell.n_stuck}{flag}"
        )
    n_eff = sum(c.n_effective for c in cells)
    n_any = sum(c.n_usable for c in cells)
    print(
        f"\nOverall yield: {n_eff}/{total} ({n_eff / total:.1%}) effective, "
        f"{n_any}/{total} ({n_any / total:.1%}) varied at all."
    )

    # Over the whole audit, defects included: this is a statement about the
    # audit, not a recommendation for the trainer.
    blocking = load_recorded_blocking(args.audit)
    blocking_source = "recorded in the audit"
    if blocking is None and args.label_audit:
        blocking, blocking_source = load_blocking_labels(args.label_audit), args.label_audit
    if blocking is not None:
        print(f"\nClean vs label-defective scenarios (blocking labels {blocking_source}):")
        for status, stats in variance_by_label_status(all_scores, blocking, args.min_std).items():
            print(
                f"  {status:12} n={stats['n']:4}  zero-variance {stats['zero_variance_fraction']:6.1%}  "
                f"effective {stats['effective_variance_fraction']:6.1%}"
            )

    print("\nPer category:")
    signal = category_signal(cells)
    print(f"  {'category':28} {'signal':>7} {'relevance':>10} {'recommended share':>18}")
    for category in sorted(mix, key=lambda c: -mix[c]):
        print(f"  {category:28} {signal.get(category, 0.0):7.3f} {relevance.get(category, 0.0):10.2f} {mix[category]:17.1%}")

    print("\nRecommended mixture:")
    print("  --category-mix " + ",".join(f"{c}={v}" for c, v in sorted(mix.items())))
    if dropped:
        # The secondary view: what the full audit, defects included, would
        # have recommended. A large gap means the defects were steering it.
        full = recommend_mix(score_cells(all_scores, min_std=args.min_std), relevance=relevance)
        print(f"  over all {len(all_scores)} scored, label defects included (for comparison only):")
        print("  " + ", ".join(f"{c}={full[c]:.1%}" for c in sorted(full)))

    cold = dead_scenario_ids(scores, include_solved=False, min_std=args.min_std)
    print(f"\n{len(cold)} unsolved scenarios ({len(cold) / total:.1%}) have group std < {args.min_std}:")
    print("no usable gradient at this temperature and group size, whatever value they sit at.")
    print("Shaping is the first thing to try on them; what survives that needs prompting or an")
    print("SFT warm-start, not more sampling.")

    if args.emit_dead_ids:
        ids = dead_scenario_ids(scores, include_solved=args.include_solved, min_std=args.min_std)
        with open(args.emit_dead_ids, "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"\nWrote {len(ids)} ids to {args.emit_dead_ids} (pass the file to grpo_train")
        print("--exclude-zero-variance-from as-is: it accepts a plain id list as well as an audit JSON).")


if __name__ == "__main__":
    sys.exit(main())
