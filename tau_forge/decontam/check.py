"""Phase 5: decontamination check -- do any of the 541 synthetic scenarios
retell the same story as one of the 114 real tau2-bench retail tasks?

This is the first and only place in the project (besides Phase 6's train-task
data prep, which is a separate sanctioned exception) where
`third_party/tau2-bench/data/tau2/domains/retail/tasks.json`'s actual content
gets read -- and only for this isolated comparison. Nothing read here may be
fed back into `tau_forge.gen.prompt_template` or any other generation-facing
code; this module has no import of, and is never imported by, `tau_forge.gen`.

**What "contamination" means here.** The synthetic scenarios and the real
tasks draw from the *same* shared `db.json` (1000 orders, 500 users, 50
products). A synthetic scenario mentioning order `#W2378156` or reusing a
product id a real task also uses is normal shared-inventory overlap, not
evidence of copying -- so nothing here treats id/value overlap as a
similarity signal. What's actually checked is *narrative*-level duplication:
does a synthetic scenario tell essentially the same story (same situation,
same reasoning, same resolution) as a real task, not just touch the same
database records. Concretely:

- **Narrative text similarity** compares a synthetic scenario's dialogue
  (`prior_turns` + `user_message`) against a real task's `reason_for_call`
  -- the free-text narrative substance of `user_scenario`, not the whole
  templated `UserScenario` object (which is mostly constant boilerplate
  headers, persona voice notes, and the `known_info`/`unknown_info` split
  that's about what the *simulated user* is allowed to say, not the story
  itself).
- **Tool-call shape** compares which *tools* a synthetic scenario's
  `expected_tool_calls` uses against which tools the real task's gold
  assistant actions use, and (only when the same tool appears in both) the
  *argument key sets* used -- never argument *values*, since values are
  where shared-inventory ids would leak in and produce a spurious signal.

**Similarity measure.** TF-IDF (word unigrams + bigrams) cosine similarity,
computed with plain `numpy` (already present transitively via `tau2`, so this
adds no new dependency). No embedding model/API is available in this
session, and difflib's character-level `SequenceMatcher` is a poor fit for
narrative-level "same story, different words" duplication across two
structurally different text registers (synthetic first/second-person dialogue
vs. real third-person "reason for call" narration) -- TF-IDF cosine over word
n-grams is the standard lexical middle ground for this and needs nothing
heavier.

**Threshold.** There is no meaningful *absolute* cosine cutoff across two
registers this different -- even genuinely similar narratives (same kind of
request, different specifics) tend to score low in absolute terms simply
because the phrasing conventions differ. So the threshold is computed from
this run's own distribution: for each synthetic scenario, take its best
(max) similarity against any real task; flag scenarios whose best match is a
statistical outlier relative to that distribution, at `mean + 3*std` (a
standard 3-sigma outlier rule). See README's Phase 5 section for the actual
numbers this produced and the spot-check verdict.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tau2.data_model.tasks import Task
from tau2.domains.retail.environment import get_tasks, get_tasks_split

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "synthetic" / "raw"
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "decontam" / "decontam_report.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_EXCERPT_LEN = 300


def tokenize(text: str) -> list[str]:
    """Lowercased word unigrams + adjacent-word bigrams. Bigrams let
    "return the" vs "the return" resolve differently and catch some
    reordering-invariant phrase-level overlap unigrams alone would miss."""
    words = _TOKEN_RE.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def build_tfidf_matrix(docs_tokens: list[list[str]]) -> np.ndarray:
    """Row-normalized TF-IDF matrix (L2 norm), one row per document, fit
    fresh on exactly the corpus passed in (so IDF reflects only this run's
    documents, not any external corpus statistics)."""
    vocab: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            vocab.setdefault(t, len(vocab))

    n_docs = len(docs_tokens)
    df = np.zeros(len(vocab), dtype=np.float64)
    for toks in docs_tokens:
        for t in set(toks):
            df[vocab[t]] += 1
    idf = np.log((1 + n_docs) / (1 + df)) + 1

    mat = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for i, toks in enumerate(docs_tokens):
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            mat[i, vocab[t]] = c * idf[vocab[t]]

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def render_synthetic_narrative(scenario: dict[str, Any]) -> str:
    parts = [turn.get("content", "") for turn in scenario.get("prior_turns", [])]
    parts.append(scenario.get("user_message", ""))
    return "\n".join(p for p in parts if p)


def render_real_narrative(task: Task) -> str:
    return task.user_scenario.instructions.reason_for_call


def tool_shape_score(expected_tool_calls: list[dict[str, Any]], real_task: Task) -> float | None:
    """Compare which tools a synthetic scenario's `expected_tool_calls` uses
    against a real task's gold assistant-action tool names, plus argument
    *key* overlap (never values -- see module docstring) when the same tool
    appears in both. Returns None when the synthetic scenario has no
    `expected_tool_calls` (ambiguous/policy_violation-with-no-call/
    out_of_scope-transfer scenarios have nothing to compare shape against)."""
    if not expected_tool_calls:
        return None

    synth_tools = {c["name"] for c in expected_tool_calls}
    real_actions = [a for a in (real_task.evaluation_criteria.actions or []) if a.requestor == "assistant"]
    real_tools = {a.name for a in real_actions}

    union = synth_tools | real_tools
    tool_jaccard = len(synth_tools & real_tools) / len(union) if union else 0.0

    shared = synth_tools & real_tools
    if not shared:
        return tool_jaccard

    key_jaccards = []
    for tool in shared:
        synth_keys = {k for c in expected_tool_calls if c["name"] == tool for k in c.get("arguments", {})}
        real_keys = {k for a in real_actions if a.name == tool for k in (a.arguments or {})}
        key_union = synth_keys | real_keys
        key_jaccards.append(len(synth_keys & real_keys) / len(key_union) if key_union else 0.0)

    return 0.5 * tool_jaccard + 0.5 * (sum(key_jaccards) / len(key_jaccards))


@dataclass
class SplitCheck:
    train_count: int
    test_count: int
    base_count: int
    train_test_disjoint: bool
    train_union_test_equals_base: bool

    @property
    def ok(self) -> bool:
        return (
            self.train_count == 74
            and self.test_count == 40
            and self.base_count == 114
            and self.train_test_disjoint
            and self.train_union_test_equals_base
        )


def check_split() -> SplitCheck:
    """Cheap re-confirmation of the split-level facts README already
    documents (train 74 / test 40 / base 114, no overlap) for the currently
    pinned tau2-bench commit, before doing the real (expensive) comparison."""
    splits = get_tasks_split()
    train_ids = set(splits["train"])
    test_ids = set(splits["test"])
    base_ids = set(splits["base"])
    return SplitCheck(
        train_count=len(train_ids),
        test_count=len(test_ids),
        base_count=len(base_ids),
        train_test_disjoint=train_ids.isdisjoint(test_ids),
        train_union_test_equals_base=(train_ids | test_ids) == base_ids,
    )


@dataclass
class FlaggedPair:
    synthetic_id: str
    real_task_id: str
    text_similarity: float
    tool_shape_score: float | None
    synthetic_excerpt: str
    real_excerpt: str

    def to_json(self) -> dict[str, Any]:
        return {
            "synthetic_id": self.synthetic_id,
            "real_task_id": self.real_task_id,
            "text_similarity": round(self.text_similarity, 4),
            "tool_shape_score": None if self.tool_shape_score is None else round(self.tool_shape_score, 4),
            "synthetic_excerpt": self.synthetic_excerpt,
            "real_excerpt": self.real_excerpt,
        }


def _excerpt(text: str, n: int = _EXCERPT_LEN) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def load_all_synthetic_scenarios(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        scenarios.extend(json.loads(path.read_text()))
    return scenarios


@dataclass
class DecontamReport:
    split_check: SplitCheck
    n_real_tasks: int
    n_synthetic_scenarios: int
    method: str
    threshold: float
    threshold_mean: float
    threshold_std: float
    flagged: list[FlaggedPair] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "split_check": {
                "train_count": self.split_check.train_count,
                "test_count": self.split_check.test_count,
                "base_count": self.split_check.base_count,
                "train_test_disjoint": self.split_check.train_test_disjoint,
                "train_union_test_equals_base": self.split_check.train_union_test_equals_base,
                "ok": self.split_check.ok,
            },
            "n_real_tasks": self.n_real_tasks,
            "n_synthetic_scenarios": self.n_synthetic_scenarios,
            "method": self.method,
            "threshold": round(self.threshold, 4),
            "threshold_derivation": {
                "rule": "mean + 3*std of each synthetic scenario's best (max) similarity to any real task",
                "mean": round(self.threshold_mean, 4),
                "std": round(self.threshold_std, 4),
            },
            "n_flagged": len(self.flagged),
            "flagged_pairs": [p.to_json() for p in sorted(self.flagged, key=lambda p: -p.text_similarity)],
        }


def run_decontam_check(
    real_tasks: list[Task] | None = None,
    synthetic_scenarios: list[dict[str, Any]] | None = None,
    similarity_threshold: float | None = None,
) -> DecontamReport:
    """`similarity_threshold`, when given, overrides the data-driven
    mean + 3*std threshold (mean/std are still computed and reported for
    context). Useful for tests exercising the flagging logic on corpora too
    small for a 3-sigma rule to be meaningful; the default script run never
    passes this."""
    split_check = check_split()

    if real_tasks is None:
        real_tasks = get_tasks(None)  # full base 114, not just train -- see README's held-out policy
    if synthetic_scenarios is None:
        synthetic_scenarios = load_all_synthetic_scenarios()

    real_texts = [render_real_narrative(t) for t in real_tasks]
    synth_texts = [render_synthetic_narrative(s) for s in synthetic_scenarios]

    all_tokens = [tokenize(t) for t in synth_texts] + [tokenize(t) for t in real_texts]
    mat = build_tfidf_matrix(all_tokens)
    synth_mat = mat[: len(synth_texts)]
    real_mat = mat[len(synth_texts) :]

    sims = synth_mat @ real_mat.T  # (n_synthetic, n_real), both rows L2-normalized -> cosine similarity
    best_real_idx = sims.argmax(axis=1)
    best_sims = sims[np.arange(len(synth_texts)), best_real_idx]

    mean, std = float(best_sims.mean()), float(best_sims.std())
    threshold = similarity_threshold if similarity_threshold is not None else mean + 3 * std

    flagged: list[FlaggedPair] = []
    for i, scenario in enumerate(synthetic_scenarios):
        score = float(best_sims[i])
        if score < threshold:
            continue
        real_task = real_tasks[best_real_idx[i]]
        flagged.append(
            FlaggedPair(
                synthetic_id=scenario.get("id", f"<index {i}>"),
                real_task_id=real_task.id,
                text_similarity=score,
                tool_shape_score=tool_shape_score(scenario.get("expected_tool_calls", []), real_task),
                synthetic_excerpt=_excerpt(synth_texts[i]),
                real_excerpt=_excerpt(real_texts[best_real_idx[i]]),
            )
        )

    return DecontamReport(
        split_check=split_check,
        n_real_tasks=len(real_tasks),
        n_synthetic_scenarios=len(synthetic_scenarios),
        method="tfidf_unigram_bigram_cosine",
        threshold=threshold,
        threshold_mean=mean,
        threshold_std=std,
        flagged=flagged,
    )


def main() -> int:
    report = run_decontam_check()
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT_PATH.write_text(json.dumps(report.to_json(), indent=2) + "\n")

    sc = report.split_check
    print(f"split check: train={sc.train_count} test={sc.test_count} base={sc.base_count} "
          f"disjoint={sc.train_test_disjoint} union_eq_base={sc.train_union_test_equals_base} "
          f"({'OK' if sc.ok else 'MISMATCH'})")
    print(f"compared {report.n_synthetic_scenarios} synthetic scenarios against {report.n_real_tasks} real tasks")
    print(f"threshold: {report.threshold:.4f} (mean {report.threshold_mean:.4f} + 3*std {report.threshold_std:.4f})")
    print(f"flagged: {len(report.flagged)} pair(s) -- see {DEFAULT_REPORT_PATH}")
    for p in sorted(report.flagged, key=lambda p: -p.text_similarity):
        print(f"  {p.text_similarity:.4f}  {p.synthetic_id}  <->  real task {p.real_task_id}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
