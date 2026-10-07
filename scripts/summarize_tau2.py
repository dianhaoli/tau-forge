"""Aggregate a tau2 results JSON into the summary.json fields used for the
baseline (pass^k, terminations, final user token, confirm+STOP, user tokens,
wall clock). Prints aggregates only; never prints task or message content.

    python scripts/summarize_tau2.py data/simulations/<run>.json
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from math import comb

STOP, TRANSFER = "###STOP###", "###TRANSFER###"
CONFIRM = re.compile(r"\b(yes|proceed|go ahead)\b", re.I)
THANKS = re.compile(r"\bthank", re.I)


def summarize(path: str) -> dict:
    d = json.load(open(path))
    sims = d["simulations"]
    by_task = defaultdict(list)
    term, final_tok = Counter(), Counter()
    confirm_stop = user_tokens = 0
    for s in sims:
        reward = (s.get("reward_info") or {}).get("reward") or 0.0
        by_task[s["task_id"]].append(reward >= 1 - 1e-6)
        term[s["termination_reason"]] += 1
        users = [m for m in s["messages"] if m["role"] == "user"]
        for m in users:
            u = m.get("usage") or {}
            user_tokens += u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
        last = (users[-1].get("content") or "") if users else ""
        final_tok["STOP" if STOP in last else "TRANSFER" if TRANSFER in last else "none"] += 1
        if STOP in last and CONFIRM.search(last) and not THANKS.search(last):
            confirm_stop += 1
    n = min(len(v) for v in by_task.values())
    pass_k = {f"pass^{k}": round(sum(comb(sum(v), k) / comb(len(v), k) for v in by_task.values()) / len(by_task), 4)
              for k in range(1, n + 1)}
    starts = [datetime.fromisoformat(s["start_time"]) for s in sims]
    ends = [datetime.fromisoformat(s["end_time"]) for s in sims]
    return {
        "n_tasks": len(by_task),
        "num_trials": n,
        "n_sims": len(sims),
        **pass_k,
        "termination": dict(term),
        "final_user_token": {k: v for k, v in final_tok.items() if v},
        "confirm_and_stop_count": confirm_stop,
        "luna_tokens_total": user_tokens,
        "mean_conversation_minutes": round(sum(s["duration"] for s in sims) / len(sims) / 60, 2),
        "wall_clock_minutes": round((max(ends) - min(starts)).total_seconds() / 60, 1),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))
