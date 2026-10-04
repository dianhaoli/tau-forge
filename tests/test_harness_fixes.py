"""Tests for the harness fixes from the zero-variance audit: the top_k
sentinel (H1), shaped-score exclusion (H2), effective variance (H3), and
label-defect exclusion with split parity (H7). Torch-free, like
tests/test_grpo_signal.py; the one TRL check skips without the `train` extra."""

import importlib.util
import json
from pathlib import Path

import pytest

from tau_forge.train import curriculum, scorecard
from tau_forge.train.dataset import build_examples
from tau_forge.train.grpo_train import (
    TOP_K_DISABLED,
    build_config_kwargs,
    build_examples_for_run,
    describe_sampling,
    normalize_top_k,
    parse_args,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return build_examples()


def _write_audit(path, raw, shaped=None):
    data = {"per_scenario_scores": raw}
    if shaped is not None:
        data["per_scenario_shaped_scores"] = shaped
    path.write_text(json.dumps(data))
    return path


# --------------------------------------------------------------------------
# H1: top_k must reach TRL as an int meaning "disabled", never None
# --------------------------------------------------------------------------


def test_trainer_top_k_default_is_the_explicit_disabled_sentinel(tmp_path):
    """None used to flow into GRPOConfig: the HF path then back-filled Qwen's
    shipped top_k=20 and the vLLM path raised a TypeError."""
    args = parse_args([])
    assert args.top_k == TOP_K_DISABLED == 0
    kwargs = build_config_kwargs(args, tmp_path, max_steps=-1, bf16=False)
    assert kwargs["top_k"] == 0 and isinstance(kwargs["top_k"], int)


def test_every_disabled_spelling_normalizes_to_zero(tmp_path):
    """-1 is vLLM's legacy 'disabled' but raises in transformers'
    TopKLogitsWarper, so it must never reach either backend as-is."""
    assert normalize_top_k(None) == 0
    assert normalize_top_k(-1) == 0
    assert normalize_top_k(0) == 0
    assert normalize_top_k(20) == 20
    kwargs = build_config_kwargs(parse_args(["--top-k", "-1"]), tmp_path, max_steps=-1, bf16=False)
    assert kwargs["top_k"] == 0


def test_baseline_and_trainer_sample_with_the_same_top_k():
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args

    assert normalize_top_k(baseline_args([]).top_k) == normalize_top_k(parse_args([]).top_k) == 0
    assert normalize_top_k(baseline_args(["--top-k", "-1"]).top_k) == 0


def test_sampling_is_recorded_in_the_run_log():
    assert describe_sampling(parse_args([])) == "sampling: temperature=1.0 top_p=1.0 top_k=0 (disabled)"
    assert describe_sampling(parse_args(["--top-k", "40"])).endswith("top_k=40")


def test_installed_trl_agrees_zero_disables_top_k():
    """Guards the sentinel against a TRL upgrade changing its meaning. Only
    runs with the `train` extra installed."""
    trl = pytest.importorskip("trl")
    import dataclasses

    (field,) = [f for f in dataclasses.fields(trl.GRPOConfig) if f.name == "top_k"]
    assert field.default == TOP_K_DISABLED


# --------------------------------------------------------------------------
# H2: --exclude-zero-variance-from judges on what GRPO sees
# --------------------------------------------------------------------------


def test_zero_variance_exclusion_prefers_shaped_scores(tmp_path):
    """Both groups are flat 0.0 under raw reward(); shaping revives one. The
    old behavior dropped both -- i.e. exactly the group shaping rescued."""
    path = _write_audit(
        tmp_path / "a.json",
        raw={"revived": [0.0] * 16, "dead": [0.0] * 16},
        shaped={"revived": [0.0] * 14 + [0.1, 0.15], "dead": [0.0] * 16},
    )
    assert curriculum.load_zero_variance_ids(path) == {"dead"}
    assert curriculum.load_zero_variance_ids(path, prefer_shaped=False) == {"revived", "dead"}
    _, basis = curriculum.zero_variance_ids_with_basis(path)
    assert "shaping" in basis


def test_zero_variance_exclusion_falls_back_to_raw_without_shaped_scores(tmp_path):
    path = _write_audit(tmp_path / "a.json", raw={"dead": [0.0] * 4, "live": [0.0, 1.0, 0.0, 1.0]})
    ids, basis = curriculum.zero_variance_ids_with_basis(path)
    assert ids == {"dead"} and basis == "reward() alone"


def test_zero_variance_exclusion_accepts_an_emitted_id_list(tmp_path):
    """The runbook calls data_scorecard's dead_scenario_ids.txt 'ready for
    --exclude-zero-variance-from'; it used to crash on json.loads."""
    path = tmp_path / "dead.txt"
    path.write_text("# emitted\nhappy_path__x__001\n\nambiguous__y__002\n")
    ids, basis = curriculum.zero_variance_ids_with_basis(path)
    assert ids == {"happy_path__x__001", "ambiguous__y__002"} and basis == "plain id list"


def test_trainer_raw_flag_forces_reward_alone(tmp_path, corpus):
    first, second = corpus[0].id, corpus[1].id
    raw = {e.id: [0.0, 1.0] for e in corpus}
    raw[first] = raw[second] = [0.0, 0.0]
    shaped = dict(raw)
    shaped[first] = [0.0, 0.15]
    audit = _write_audit(tmp_path / "a.json", raw, shaped)
    flags = ["--keep-label-defects", "--val-fraction", "0", "--exclude-zero-variance-from", str(audit)]

    train, _ = build_examples_for_run(parse_args(flags))
    assert second not in {e.id for e in train} and first in {e.id for e in train}

    train_raw, _ = build_examples_for_run(parse_args(flags + ["--exclude-zero-variance-raw"]))
    assert not ({first, second} & {e.id for e in train_raw})


# --------------------------------------------------------------------------
# H3: effective variance -- group std, not "any two samples differ"
# --------------------------------------------------------------------------


def test_group_std_is_the_sample_std_trl_logs():
    """TRL's nanstd applies Bessel's correction; with n=16, one outlier of x
    gives exactly x/4 -- which is what pins the 0.05 default to the 0.2 tier."""
    assert scorecard.group_std([0.0] * 15 + [1.0]) == pytest.approx(0.25)
    assert scorecard.group_std([0.0] * 15 + [0.2]) == pytest.approx(0.05)
    assert scorecard.group_std([0.5]) == 0.0


@pytest.mark.parametrize(
    "scores, effective",
    [
        ([0.0] * 15 + [0.02], False),  # shaping micro-credit: 'alive' under the old test
        ([0.0] * 15 + [0.15], False),  # one max shaping outlier
        ([0.0] * 14 + [0.15, 0.15], True),  # repeated partial credit
        ([0.0] * 15 + [0.2], True),  # one sample at reward()'s smallest tier
        ([0.0] * 8 + [1.0] * 8, True),
        ([0.0] * 16, False),
    ],
)
def test_effective_variance_threshold(scores, effective):
    assert scorecard.has_effective_variance(scores) is effective


def test_effective_threshold_is_configurable():
    group = [0.0] * 14 + [0.15, 0.15]
    assert scorecard.has_effective_variance(group, min_std=0.05)
    assert not scorecard.has_effective_variance(group, min_std=0.1)


def test_variance_summary_reports_both_fractions_and_the_gap():
    table = {
        "a": [0.0] * 16,  # zero variance
        "b": [0.0] * 15 + [0.02],  # micro-variance
        "c": [0.0] * 8 + [1.0] * 8,  # effective
        "d": [1.0] * 16,  # zero variance
    }
    s = scorecard.variance_summary(table)
    assert s["zero_variance_count"] == 2 and s["zero_variance_fraction"] == 0.5
    assert s["effective_variance_count"] == 1 and s["effective_variance_fraction"] == 0.25
    assert s["micro_variance_count"] == 1


def test_scorecard_yield_uses_effective_variance_and_keeps_the_old_number():
    scores = {
        "happy_path__t__0": [0.0, 0.0, 0.0, 0.02],
        "happy_path__t__1": [0.0, 1.0, 0.0, 1.0],
    }
    (cell,) = scorecard.score_cells(scores)
    assert cell.yield_ == 0.5
    assert cell.yield_any_variance == 1.0
    d = cell.as_dict()
    assert d["yield"] == 0.5 and d["yield_any_variance"] == 1.0 and d["effective"] == 1
    (loose,) = scorecard.score_cells(scores, min_std=0.001)
    assert loose.yield_ == 1.0


def test_variance_by_label_status_splits_clean_from_defective():
    table = {"clean": [0.0, 1.0], "bad": [0.0, 0.0], "uncovered": [1.0, 1.0]}
    blocking = {"clean": [], "bad": ["confirmation_missing"]}
    split = scorecard.variance_by_label_status(table, blocking)
    assert split["clean"]["n"] == 2 and split["clean"]["effective_variance_count"] == 1
    assert split["label_defect"]["n"] == 1 and split["label_defect"]["zero_variance_fraction"] == 1.0


def test_baseline_variance_report_sits_next_to_the_zero_variance_fraction():
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args, variance_report

    assert baseline_args([]).min_std == scorecard.EFFECTIVE_MIN_STD
    assert baseline_args(["--min-std", "0.1"]).min_std == 0.1
    table = {"a": [0.0] * 15 + [0.02], "b": [0.0] * 8 + [1.0] * 8}
    report = variance_report(table, {"a": ["auth_missing"], "b": []}, 0.05)
    assert report["effective_variance_scenario_count"] == 1
    assert report["effective_variance_scenario_fraction"] == 0.5
    assert report["micro_variance_scenario_count"] == 1
    assert report["variance_by_label_status"]["label_defect"]["effective_variance_count"] == 0
    assert "variance_by_label_status" not in variance_report(table, None, 0.05)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bucket_analysis_reports_effective_variance_and_label_split():
    bucket_analysis = _load_script("bucket_analysis")
    table = {"a": [0.0] * 15 + [0.02], "b": [0.0] * 8 + [1.0] * 8, "c": [0.0] * 16}
    assert bucket_analysis.bucket_of(table["a"]) == "has_variance"  # old bucket unchanged
    lines = bucket_analysis.effective_lines(table, 0.05, {"a": [], "b": [], "c": ["auth_missing"]})
    assert "1 (33.3%)" in lines[0] and "1 micro-variance" in lines[0]
    assert any(line.strip().startswith("label_defect") for line in lines)


def test_bucket_analysis_prefers_the_audits_recorded_blocking_labels(tmp_path):
    bucket_analysis = _load_script("bucket_analysis")
    recorded = {"per_scenario_blocking_labels": {"x": ["auth_missing"]}}
    blocking, source = bucket_analysis.resolve_blocking(recorded, str(curriculum.DEFAULT_LABEL_AUDIT))
    assert blocking == {"x": ["auth_missing"]} and "recorded" in source
    fallback, _ = bucket_analysis.resolve_blocking({}, str(curriculum.DEFAULT_LABEL_AUDIT))
    assert len([sid for sid, b in fallback.items() if b]) == 222
    assert bucket_analysis.resolve_blocking({}, "") == (None, None)


# --------------------------------------------------------------------------
# Label-defect exclusion
# --------------------------------------------------------------------------


def test_load_label_defect_ids_reads_the_blocking_list(corpus):
    defects = curriculum.load_label_defect_ids()
    assert len(defects) == 222
    assert defects <= {e.id for e in corpus}
    blocking = curriculum.load_blocking_labels()
    assert all(blocking[sid] for sid in defects)
    assert len(blocking) == 541


def test_label_defect_loader_on_a_small_file(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "scenarios": {
                    "a": {"blocking": ["gold_ids_unreachable:item_ids"], "labels": []},
                    "b": {"blocking": [], "labels": ["mixed"]},
                }
            }
        )
    )
    assert curriculum.load_label_defect_ids(path) == {"a"}
    assert curriculum.blocking_category("gold_ids_unreachable:item_ids,payment_method_id") == "gold_ids_unreachable"


def test_trainer_drops_label_defects_by_default(corpus, capsys):
    defects = curriculum.load_label_defect_ids()
    train, val = build_examples_for_run(parse_args([]))
    kept = {e.id for e in train + val}
    assert not (kept & defects)
    assert len(kept) == len(corpus) - 222

    out = capsys.readouterr().out
    assert "dropping them" in out
    assert "label_defect" in out and "remaining" in out
    total_line = next(line for line in out.splitlines() if "TOTAL" in line)
    assert total_line.split()[-4:] == ["541", "222", "0", "319"]
    assert "confirmation_missing=" in out


def test_trainer_keeps_label_defects_on_request(corpus):
    train, val = build_examples_for_run(parse_args(["--keep-label-defects"]))
    assert len(train) + len(val) == len(corpus)
    train, val = build_examples_for_run(parse_args(["--label-audit", ""]))
    assert len(train) + len(val) == len(corpus)


def test_a_scenario_both_defective_and_dead_is_counted_once(corpus):
    defects = sorted(curriculum.load_label_defect_ids())
    clean = sorted({e.id for e in corpus} - set(defects))
    ex = curriculum.Exclusions(
        label_defects=set(defects), zero_variance={defects[0], clean[0]}, blocking=curriculum.load_blocking_labels()
    )
    total_line = next(line for line in curriculum.exclusion_table(corpus, ex) if line.startswith("TOTAL"))
    assert total_line.split()[1:] == ["541", "222", "1", "318"]
    assert ex.as_dict()["n_excluded"] == 223


def test_exclusion_fingerprint_tracks_the_excluded_set():
    a = curriculum.Exclusions(label_defects={"x", "y"})
    b = curriculum.Exclusions(label_defects={"x"}, zero_variance={"y"})
    c = curriculum.Exclusions(label_defects={"x"})
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


def test_baseline_split_all_scores_everything_but_records_blocking_labels(corpus):
    from tau_forge.train.zero_shot_baseline import (
        load_scenario_blocking,
        parse_args as baseline_args,
        select_split,
    )

    args = baseline_args([])
    assert select_split(corpus, args) == (None, None)
    blocking = load_scenario_blocking(args, [e.id for e in corpus])
    assert len(blocking) == len(corpus)
    assert sum(1 for labels in blocking.values() if labels) == 222
    assert load_scenario_blocking(baseline_args(["--label-audit", ""]), ["x"]) is None


def test_baseline_split_val_applies_the_trainers_exclusions(corpus):
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args, select_split

    keep, exclusions = select_split(corpus, baseline_args(["--split", "val"]))
    assert keep and not (keep & curriculum.load_label_defect_ids())
    assert exclusions.as_dict()["n_label_defects_excluded"] == 222


def test_baseline_parser_shares_the_trainers_exclusion_flags():
    """A flag on one parser but not the other is a split mismatch waiting to
    happen; add_exclusion_args is shared so they cannot drift."""
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args

    trainer, baseline = vars(parse_args([])), vars(baseline_args([]))
    for flag in (
        "label_audit",
        "keep_label_defects",
        "exclude_zero_variance_from",
        "exclude_zero_variance_raw",
        "exclude_solved",
    ):
        assert trainer[flag] == baseline[flag], flag


def test_baseline_parser_can_format_its_help():
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args

    with pytest.raises(SystemExit) as exit_info:
        baseline_args(["--help"])
    assert exit_info.value.code == 0
