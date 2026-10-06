"""Score the scripted user's NLU (`tau_forge.episodes.nlu`) against the gold
semantic labels of 806 real Qwen3-4B agent turns.

  python scripts/nlu_eval.py                  # train + held-out tables
  python scripts/nlu_eval.py --split train --confusions confirmation_request
  python scripts/nlu_eval.py --by-template

Split: turn index (line number in turns.jsonl) % 5 == 0 is the held-out test
split. The rules were tuned on the train split only; `tests/test_nlu_gold.py`
asserts held-out thresholds.

Decision-critical fields (and their targets): confirmation_request (>= 0.97),
refusal exact (>= 0.96), info_requests exact set (>= 0.92), names_target when
gold has a proposed_action (>= 0.95), details_match_task when gold is non-null
(>= 0.93).

The NLU state a turn was said in (late correction / fallback already given) is
rebuilt from the saved episode transcripts the turns were taken from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tau_forge.episodes.nlu import NLUState, RulesAnalyzer  # noqa: E402
from tau_forge.episodes.task import ORDER_ID_RE  # noqa: E402

GOLD_DIR = REPO_ROOT / "data" / "episodes" / "nlu_gold"
TRANSCRIPTS = [REPO_ROOT / "data" / "trained" / "episode_transcripts_subset.json",
               REPO_ROOT / "data" / "trained" / "episode_foreign_check.json"]
FIELDS = ("confirmation_request", "refusal", "info_requests", "names_target", "names_target#", "details_match_task",
          "details_match_task*",
          "identity_target", "action", "names_wrong_target", "constraint_statement", "offers_other_help_only",
          "claims_action_done", "narration_only")
TARGETS = {"confirmation_request": 0.97, "refusal": 0.96, "info_requests": 0.92, "names_target": 0.95,
           "details_match_task": 0.93}
# details_match_task* = details_match_task without the label-ambiguous return recaps (see `label_ambiguous_dm`);
# names_target# = names_target on turns that name an order id (the labels split on product-only exchange turns)


def load_gold() -> list[dict[str, Any]]:
    turns = [json.loads(line) for line in open(GOLD_DIR / "turns.jsonl")]
    labels = {json.loads(line)["turn_id"]: json.loads(line) for line in open(GOLD_DIR / "labels.jsonl")}
    users = _user_histories()
    out = []
    for i, t in enumerate(turns):
        t = dict(t)
        t["index"] = i
        t["split"] = "test" if i % 5 == 0 else "train"
        t["label"] = labels[t["turn_id"]]
        t["task"] = dict(t["task"], template=t["template"])
        t["state"] = _state(t, users)
        out.append(t)
    return out


def _user_histories() -> dict[tuple[str, int], list[dict[str, Any]]]:
    hist = {}
    for path in TRANSCRIPTS:
        if not path.exists():
            continue
        d = json.load(open(path))
        for pt in d["per_task"]:
            for s, msgs in enumerate(pt["transcripts"]):
                hist[(pt["id"], s)] = [m for m in msgs if m["role"] != "system"]
    return hist


def _state(t: dict[str, Any], hist) -> NLUState:
    tid, s, m = t["turn_id"].split("/")
    msgs = hist.get((tid, int(s[1:])))
    prof = t["task"].get("profile_lines") or {}
    corr = prof.get("correction") or []
    fb = prof.get("fallback") or []
    said = [t["previous_user_message"]]
    if msgs is not None:
        said = [x.get("content") or "" for x in msgs[: int(m[1:])] if x["role"] == "user"]
    return NLUState(
        correction_used=any(any(c in u for c in corr) for u in said),
        fallback_used=any(any(f in u for f in fb) for u in said),
    )


def label_ambiguous_dm(t: dict[str, Any]) -> bool:
    """Turns whose details_match_task label convention the adjudicated labels
    split on: a return recap, before the fallback, that names the refund method
    the user asked for (30 labelled False = against the gold write, 23 True =
    against the user's wish, on train). Defined from the turn text and state
    only, never from the label."""
    if t["template"] != "return_fallback" or t["state"].fallback_used:
        return False
    from tau_forge.episodes.nlu import _resolve_pm_mentions, clean, facts_from

    f = facts_from(t["task"])
    return any(m.pm == f.asked_pm for m in _resolve_pm_mentions(clean(t["agent_text"]), f))


def field_values(label: dict[str, Any]) -> dict[str, Any]:
    pa = label.get("proposed_action")
    return {
        "confirmation_request": bool(label["confirmation_request"]),
        "refusal": label["refusal"],
        "info_requests": tuple(sorted(label["info_requests"])),
        "identity_target": label["identity_target"],
        "action": pa["action"] if pa else None,
        "names_target": pa["names_target"] if pa else None,
        "names_target#": pa["names_target"] if pa else None,
        "names_wrong_target": pa["names_wrong_target"] if pa else None,
        "details_match_task": pa["details_match_task"] if pa else None,
        "details_match_task*": pa["details_match_task"] if pa else None,
        "constraint_statement": bool(label["constraint_statement"]),
        "offers_other_help_only": bool(label["offers_other_help_only"]),
        "claims_action_done": bool(label["claims_action_done"]),
        "narration_only": bool(label["narration_only"]),
    }


def evaluate(rows: list[dict[str, Any]], analyze: Optional[Callable] = None) -> dict[str, Any]:
    """Per-field accuracy. names_target is scored only where gold has a
    proposed_action; details_match_task only where gold's is non-null."""
    an = analyze or RulesAnalyzer().analyze
    stats: dict[str, Counter] = defaultdict(Counter)
    confusions: dict[str, list] = defaultdict(list)
    by_template: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for t in rows:
        sem = an(t["agent_text"], t["task"], t["state"])
        gold = field_values(t["label"])
        pred = field_values(sem.to_label())
        for f in FIELDS:
            if f in ("names_target", "names_target#", "names_wrong_target") and t["label"]["proposed_action"] is None:
                continue
            if f == "names_target#" and not ORDER_ID_RE.search(t["agent_text"]):
                continue
            if f.startswith("details_match_task") and gold[f] is None:
                continue
            if f == "details_match_task*" and label_ambiguous_dm(t):
                continue
            ok = gold[f] == pred[f]
            stats[f]["n"] += 1
            stats[f]["ok"] += ok
            by_template[t["template"]][f]["n"] += 1
            by_template[t["template"]][f]["ok"] += ok
            if not ok:
                confusions[f].append({"index": t["index"], "turn_id": t["turn_id"], "gold": gold[f], "pred": pred[f],
                                      "text": t["agent_text"]})
    acc = {f: (stats[f]["ok"] / stats[f]["n"] if stats[f]["n"] else 1.0) for f in FIELDS}
    return {"acc": acc, "n": {f: stats[f]["n"] for f in FIELDS}, "confusions": confusions,
            "by_template": {k: {f: (v[f]["ok"] / v[f]["n"] if v[f]["n"] else None, v[f]["n"]) for f in FIELDS}
                            for k, v in by_template.items()}}


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", choices=["train", "test", "all", "both"], default="both")
    p.add_argument("--confusions", default="", help="comma-separated fields whose errors to print")
    p.add_argument("--max-confusions", type=int, default=40)
    p.add_argument("--by-template", action="store_true")
    args = p.parse_args(argv)
    rows = load_gold()
    splits = {"train": [r for r in rows if r["split"] == "train"], "test": [r for r in rows if r["split"] == "test"],
              "all": rows}
    shown = ["train", "test"] if args.split == "both" else [args.split]
    results = {s: evaluate(splits[s]) for s in shown}
    print(f"{'field':24s} " + " ".join(f"{s:>16s}" for s in shown) + "   target")
    for f in FIELDS:
        cells = " ".join(f"{results[s]['acc'][f]:9.3f} (n={results[s]['n'][f]:3d})" for s in shown)
        print(f"{f:24s} {cells}   {TARGETS.get(f, '')}")
    if args.by_template:
        for s in shown:
            print(f"\n[{s}] per template")
            for tmpl, d in sorted(results[s]["by_template"].items()):
                print(f"  {tmpl:22s} " + " ".join(f"{f[:10]}={a:.2f}/{n}" for f, (a, n) in d.items() if a is not None
                                                   and f in TARGETS))
    for f in [x for x in args.confusions.split(",") if x]:
        for s in shown:
            if s == "test" and args.split == "both":
                continue  # never look at held-out errors while tuning
            print(f"\n=== confusions [{s}] {f}: {len(results[s]['confusions'][f])}")
            for c in results[s]["confusions"][f][: args.max_confusions]:
                print(f"#{c['index']} gold={c['gold']} pred={c['pred']} | {c['text'][:600]!r}")
    return results


if __name__ == "__main__":
    main()
