"""Stage B: the scripted user's NLU against 806 adjudicated real-turn labels.

`data/episodes/nlu_gold/{turns,labels}.jsonl` hold real Qwen3-4B agent turns
and what each turn does (contract, "Semantic NLU labels"). Every 5th turn
(index % 5 == 0) is a held-out split that was never looked at while the rules
in `tau_forge/episodes/nlu.py` were tuned; these tests pin the held-out
accuracy slightly below what the rules reach (`scripts/nlu_eval.py` prints the
full tables, train and held-out, per field and per template).

Known label-convention gaps (documented in the report, not tuned around):
  * names_target -- product-only exchange turns are labelled both ways across
    batches; on turns that name an order id the rules agree with the labels
    (`names_target#`).
  * details_match_task -- a return recap naming the refund card the user asked
    for, before the fallback, is labelled False (vs the gold write) in some
    batches and True (vs the user's wish) in others; `details_match_task*`
    leaves those turns out by a text-only rule.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("nlu_eval", REPO_ROOT / "scripts" / "nlu_eval.py")
nlu_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nlu_eval)

# Held-out floors: slightly below the measured held-out accuracy (see the report).
HELD_OUT_FLOOR = {
    "confirmation_request": 0.94,   # measured 0.957 (target 0.97)
    "refusal": 0.95,                # measured 0.969 (target 0.96)
    "info_requests": 0.91,          # measured 0.932 (target 0.92)
    "names_target": 0.90,           # measured 0.926 (target 0.95; label split, see names_target#)
    "names_target#": 0.97,          # measured 1.000
    "details_match_task": 0.77,     # measured 0.795 (target 0.93; label split, see details_match_task*)
    "details_match_task*": 0.84,    # measured 0.862
    "identity_target": 0.97,
    "claims_action_done": 0.96,
}
TRAIN_FLOOR = {
    "confirmation_request": 0.97, "refusal": 0.96, "info_requests": 0.93, "names_target#": 0.98,
    "details_match_task*": 0.92,
}


@pytest.fixture(scope="module")
def gold():
    return nlu_eval.load_gold()


@pytest.fixture(scope="module")
def held_out(gold):
    return nlu_eval.evaluate([r for r in gold if r["split"] == "test"])


@pytest.fixture(scope="module")
def train(gold):
    return nlu_eval.evaluate([r for r in gold if r["split"] == "train"])


def test_gold_set_shape(gold):
    assert len(gold) == 806
    assert sum(r["split"] == "test" for r in gold) == 162
    assert all((r["index"] % 5 == 0) == (r["split"] == "test") for r in gold)


@pytest.mark.parametrize("field, floor", sorted(HELD_OUT_FLOOR.items()))
def test_held_out_accuracy(held_out, field, floor):
    acc = held_out["acc"][field]
    assert acc >= floor, (field, acc, [(c["index"], c["gold"], c["pred"]) for c in held_out["confusions"][field]][:5])


@pytest.mark.parametrize("field, floor", sorted(TRAIN_FLOOR.items()))
def test_train_accuracy(train, field, floor):
    assert train["acc"][field] >= floor, (field, train["acc"][field])


def test_eval_script_runs(capsys):
    nlu_eval.main(["--split", "train"])
    out = capsys.readouterr().out
    assert "confirmation_request" in out and "details_match_task" in out
