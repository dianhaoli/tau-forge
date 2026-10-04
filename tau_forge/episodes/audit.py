"""The episode variance audit loop, generator-agnostic and torch-free.

`scripts/episode_audit.py` wires this to vLLM on a GPU box; the tests wire it
to scripted fakes. Everything between "a batch of conversations needs its next
assistant turn" and "here are the per-task reward lists" lives here, so the
part that cannot run without a GPU is reduced to one call:
`engine(prompts) -> [(text, finish_reason)]`.

Batching. Each turn, every still-active episode (n_samples per task, all
tasks) contributes one request, and the generator answers the whole batch in
one call -- for vLLM one `LLM.generate`, with prefix caching absorbing the
~6k-token system prompt + tool schemas every request shares. Episodes end at
different turns, so batches shrink as the audit proceeds.

All samples of a task share scripted-user seed 0: the user then answers
identical agent text identically, so the spread of a task's rewards is the
policy's alone -- the quantity GRPO's group advantage is computed from.

Effective variance is std >= 0.05 per task (population std over its samples),
the bucket the design doc recommends over a 1e-3 range test: a group whose
rewards differ only by a shaping crumb barely moves the policy.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from tau_forge.episodes.reward import score_episode
from tau_forge.episodes.runner import Episode
from tau_forge.episodes.task import EpisodeTask

EFFECTIVE_STD = 0.05


@dataclass
class Generation:
    text: str
    # "stop" (complete turn), "length" (hit the per-turn token cap: the
    # episode ends as truncated), or "context" (the prompt itself no longer
    # fits the model's context: the episode ends without a turn).
    finish_reason: str = "stop"


@dataclass
class EpisodeRequest:
    key: tuple[int, int]  # (task index, sample index)
    task: EpisodeTask
    messages: list[dict[str, Any]]


BatchGenerator = Callable[[list[EpisodeRequest]], list[Generation]]


class ChatTemplatePolicy:
    """Renders each request with the tokenizer's chat template + the 16 tau2
    tool schemas, drops requests whose prompt + `max_new_tokens` would exceed
    `max_model_len` (vLLM rejects those outright), and sends the rest to
    `engine` in one batch. All model-specific pieces are injected."""

    def __init__(
        self,
        engine: Callable[[list[str]], list[tuple[str, str]]],
        apply_chat_template: Callable[..., str],
        tools: list[dict[str, Any]],
        count_tokens: Callable[[str], int],
        max_new_tokens: int,
        max_model_len: int,
    ):
        self.engine = engine
        self.apply_chat_template = apply_chat_template
        self.tools = tools
        self.count_tokens = count_tokens
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.max_prompt_tokens = 0
        self.n_context_overflows = 0

    def __call__(self, requests: list[EpisodeRequest]) -> list[Generation]:
        prompts = [self.apply_chat_template(r.messages, tools=self.tools) for r in requests]
        lengths = [self.count_tokens(p) for p in prompts]
        fits = [n + self.max_new_tokens <= self.max_model_len for n in lengths]
        self.max_prompt_tokens = max([self.max_prompt_tokens, *lengths])
        self.n_context_overflows += fits.count(False)
        outs = iter(self.engine([p for p, ok in zip(prompts, fits) if ok]) if any(fits) else [])
        return [Generation(*next(outs)) if ok else Generation("", "context") for ok in fits]


def run_audit(
    tasks: Sequence[EpisodeTask],
    generate: BatchGenerator,
    *,
    n_samples: int,
    system_message: Optional[dict[str, str]] = None,
    max_turns: int = 30,
    max_calls: int = 30,
    keep_transcripts: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    episodes = {
        (i, s): Episode(task, system_message=system_message, max_turns=max_turns, max_calls=max_calls)
        for i, task in enumerate(tasks)
        for s in range(n_samples)
    }
    turn = 0
    while active := [k for k, ep in episodes.items() if not ep.done]:
        requests = [EpisodeRequest(k, episodes[k].task, episodes[k].messages) for k in active]
        generations = generate(requests)
        if len(generations) != len(requests):
            raise RuntimeError(f"generator returned {len(generations)} turns for {len(requests)} requests")
        for k, g in zip(active, generations):
            episodes[k].step(g.text, g.finish_reason)
        turn += 1
        if progress:
            progress(turn, len(active))

    per_task = []
    for i, task in enumerate(tasks):
        scored = [(episodes[(i, s)], score_episode(task, episodes[(i, s)].result())) for s in range(n_samples)]
        rec: dict[str, Any] = {
            "id": task.id,
            "template": task.template,
            "difficulty": task.difficulty,
            "rewards": [r.reward for _, r in scored],
            "successes": [r.success for _, r in scored],
            "end_reasons": [ep.end_reason for ep, _ in scored],
            "n_calls": [ep.log.n_calls for ep, _ in scored],
            "n_assistant_turns": [ep.log.n_assistant_turns for ep, _ in scored],
        }
        if keep_transcripts:
            rec["transcripts"] = [ep.messages for ep, _ in scored]
            rec["reward_breakdowns"] = [r.to_dict() for _, r in scored]
        per_task.append(rec)
    return {"n_turns": turn, "per_task": per_task, "summary": summarize(per_task)}


def _task_stats(rec: dict[str, Any]) -> dict[str, Any]:
    rewards = rec["rewards"]
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return {
        "mean": statistics.fmean(rewards),
        "std": std,
        "success_rate": sum(rec["successes"]) / len(rec["successes"]),
        "effective_variance": std >= EFFECTIVE_STD,
        "flat": len(set(rewards)) == 1,
    }


def _aggregate(recs: list[dict[str, Any]]) -> dict[str, Any]:
    stats = [_task_stats(r) for r in recs]
    n = len(stats)
    return {
        "n_tasks": n,
        "n_episodes": sum(len(r["rewards"]) for r in recs),
        "mean_reward": round(statistics.fmean(s["mean"] for s in stats), 4),
        "success_rate": round(statistics.fmean(s["success_rate"] for s in stats), 4),
        "effective_variance_fraction": round(sum(s["effective_variance"] for s in stats) / n, 4),
        "flat_fraction": round(sum(s["flat"] for s in stats) / n, 4),
        "end_reasons": dict(Counter(e for r in recs for e in r["end_reasons"])),
    }


def summarize(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall, per-template and per-knob (template -> knob -> value)
    aggregates. Knobs are the task's scalar, non-float difficulty entries
    (floats like a price difference would make one bucket per task)."""
    if not per_task:
        return {"overall": None, "per_template": {}, "per_knob": {}}
    by_template: dict[str, list] = defaultdict(list)
    by_knob: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in per_task:
        by_template[rec["template"]].append(rec)
        for knob, value in rec["difficulty"].items():
            if isinstance(value, (bool, int, str)):
                by_knob[rec["template"]][knob][str(value)].append(rec)
    return {
        "overall": _aggregate(per_task),
        "per_template": {t: _aggregate(r) for t, r in sorted(by_template.items())},
        "per_knob": {
            t: {k: {v: _aggregate(r) for v, r in sorted(vals.items())} for k, vals in sorted(knobs.items())}
            for t, knobs in sorted(by_knob.items())
        },
    }
