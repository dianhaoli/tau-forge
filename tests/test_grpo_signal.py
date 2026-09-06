"""Tests for the pieces added to raise the fraction of the corpus that
produces a usable GRPO gradient: the shaping reward, the curriculum/mixture
filter, and grpo_train's torch-free preflight. Runs without a GPU or the
`train` extra, like tests/test_train_pipeline.py."""

import json

import pytest

from tau_forge.envs.retail import RetailEnv
from tau_forge.reward.reward import Action, reward
from tau_forge.train import curriculum, shaping
from tau_forge.train.dataset import build_examples
from tau_forge.train.grpo_train import build_config_kwargs, check_prompt_lengths, parse_args
from tau_forge.train.reward_adapter import _get_shared_db

CALL = '<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'


def _call(name, args=None):
    return CALL.format(name=name, args=json.dumps(args or {}))


@pytest.fixture(scope="module")
def env():
    return RetailEnv()


# --------------------------------------------------------------------------
# shaping: the ordering guarantee is the whole safety argument for this module
# --------------------------------------------------------------------------


def test_shaping_never_lets_a_wrong_tool_outscore_a_right_one(env):
    """reward() floors a right-tool call at 0.2 (schema-invalid) and 0.3
    (schema-valid). Shaping caps at 0.15, so wrong-tool + full shaping still
    loses to the worst right-tool outcome. Asserted, not just documented."""
    assert shaping.WRONG_TOOL_CEILING < 0.2
    components = (
        shaping.PARSEABLE_CALL
        + shaping.REAL_TOOL
        + shaping.SCHEMA_VALID
        + shaping.SAME_TOOL_CLASS
        + shaping.RIGHT_TARGET_RECORD
    )
    assert components == pytest.approx(shaping.WRONG_TOOL_CEILING)


def test_shaping_is_graded_across_wrong_answers(env):
    """The point of the module: four completions that reward() all scores a
    flat 0.0 must come out distinguishable, and ordered by how close they are."""
    expected_name = "cancel_pending_order"
    expected_args = {"order_id": "#W0000001", "reason": "no longer needed"}

    garbage = shaping.shaping_score("<tool_call>not json</tool_call>", expected_name, expected_args, env)
    fake_tool = shaping.shaping_score(_call("nonexistent_tool"), expected_name, expected_args, env)
    wrong_class = shaping.shaping_score(
        _call("get_order_details", {"order_id": "#W9999999"}), expected_name, expected_args, env
    )
    right_record = shaping.shaping_score(
        _call("get_order_details", {"order_id": "#W0000001"}), expected_name, expected_args, env
    )

    db = _get_shared_db()
    for completion in (garbage, fake_tool, wrong_class, right_record):
        assert completion <= shaping.WRONG_TOOL_CEILING

    # reward() itself cannot tell any of these apart -- that is the problem.
    flat = {
        reward(Action("__malformed_tool_call__"), Action(expected_name, expected_args), db).score,
        reward(Action("nonexistent_tool"), Action(expected_name, expected_args), db).score,
        reward(
            Action("get_order_details", {"order_id": "#W0000001"}),
            Action(expected_name, expected_args),
            db,
        ).score,
    }
    assert flat == {0.0}

    # ...and shaping can, strictly monotonically in closeness.
    assert garbage < fake_tool < wrong_class < right_record


def test_shaping_stays_out_of_the_way_where_reward_already_grades(env):
    # Gold is silence: crediting a call here would teach the policy to act.
    assert shaping.shaping_score(_call("get_user_details", {"user_id": "u"}), None, {}, env) == 0.0
    # Policy stayed silent when a call was needed: nothing to grade.
    assert shaping.shaping_score("I'm sorry, could you clarify?", "cancel_pending_order", {}, env) == 0.0
    # Right tool: reward() already resolves this in its 0.2-1.0 band.
    assert (
        shaping.shaping_score(
            _call("cancel_pending_order", {"order_id": "#W1"}), "cancel_pending_order", {}, env
        )
        == 0.0
    )


def test_multi_call_penalty_applies_even_to_an_otherwise_correct_turn(env):
    two = _call("cancel_pending_order", {"order_id": "#W1"}) + _call("get_user_details", {"user_id": "u"})
    assert shaping.count_tool_call_blocks(two) == 2
    assert shaping.shaping_score(two, "cancel_pending_order", {}, env) == pytest.approx(
        -shaping.MULTI_CALL_PENALTY
    )
    assert shaping.shaping_score(two, "cancel_pending_order", {}, env, penalize_multi_call=False) == 0.0


def test_id_values_flattens_list_valued_id_arguments():
    assert shaping._id_values({"item_ids": ["1", "2"], "order_id": "#W1", "reason": "x"}) == {
        "1",
        "2",
        "#W1",
    }


def test_grpo_shaping_func_matches_trl_reward_signature():
    func = shaping.make_grpo_shaping_func()
    out = func(
        prompts=["p", "p"],
        completions=[_call("get_order_details", {"order_id": "#W0000001"}), "just a message"],
        expected_tool_name=["cancel_pending_order", "cancel_pending_order"],
        expected_tool_arguments_json=['{"order_id": "#W0000001"}', "{}"],
    )
    assert len(out) == 2 and out[0] > 0.0 and out[1] == 0.0


# --------------------------------------------------------------------------
# curriculum
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_examples():
    return build_examples()


def test_corpus_as_generated_is_majority_non_acting(all_examples):
    """The finding that motivates REAL_TASK_ALIGNED_MIX: over half the
    training signal is spent on withholding a call or escalating out of the
    domain, against a benchmark where ~3.5% of tasks need escalation."""
    stats = curriculum.summarize(all_examples)
    assert stats["n"] == 541
    assert stats["no_call_fraction"] == pytest.approx(182 / 541, abs=1e-6)
    assert stats["non_acting_fraction"] > 0.5


def test_real_task_aligned_mix_flips_that_majority(all_examples):
    mixed = curriculum.apply_mixture(all_examples, curriculum.REAL_TASK_ALIGNED_MIX, seed=0)
    stats = curriculum.summarize(mixed)
    assert stats["non_acting_fraction"] < 0.3
    assert stats["n_cells"] == 30, "downsampling must not delete a whole category__theme cell"
    for category, share in curriculum.REAL_TASK_ALIGNED_MIX.items():
        actual = stats["by_category"][category] / stats["n"]
        assert actual == pytest.approx(share, abs=0.02)


def test_apply_mixture_is_deterministic_and_never_upsamples(all_examples):
    a = curriculum.apply_mixture(all_examples, curriculum.REAL_TASK_ALIGNED_MIX, seed=3)
    b = curriculum.apply_mixture(all_examples, curriculum.REAL_TASK_ALIGNED_MIX, seed=3)
    assert [e.id for e in a] == [e.id for e in b]
    assert len({e.id for e in a}) == len(a)
    available = curriculum.summarize(all_examples)["by_category"]
    for category, count in curriculum.summarize(a)["by_category"].items():
        assert count <= available[category]


def test_train_val_split_is_disjoint_and_covers_every_cell(all_examples):
    train, val = curriculum.train_val_split(all_examples, val_fraction=0.1, seed=0)
    assert len(train) + len(val) == len(all_examples)
    assert not ({e.id for e in train} & {e.id for e in val})
    assert curriculum.summarize(val)["n_cells"] == 30


def test_load_zero_variance_ids_separates_cold_start_from_solved(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "per_scenario_scores": {
                    "cold": [0.0, 0.0, 0.0],
                    "solved": [1.0, 1.0, 1.0],
                    "stuck": [0.3, 0.3, 0.3],
                    "live": [0.0, 1.0, 0.3],
                }
            }
        )
    )
    assert curriculum.load_zero_variance_ids(path) == {"cold"}
    assert curriculum.load_zero_variance_ids(path, include_solved=True) == {"cold", "solved", "stuck"}


def test_build_training_sets_filters_then_mixes_then_splits(all_examples):
    drop = {e.id for e in all_examples if e.category == "out_of_scope"}
    train, val = curriculum.build_training_sets(
        all_examples, mix=curriculum.REAL_TASK_ALIGNED_MIX, exclude=drop, val_fraction=0.1
    )
    combined = train + val
    assert not any(e.category == "out_of_scope" for e in combined)
    assert curriculum.summarize(combined)["non_acting_fraction"] < 0.3


# --------------------------------------------------------------------------
# grpo_train preflight
# --------------------------------------------------------------------------


class _FakeTokenizer:
    def __call__(self, text):
        return {"input_ids": list(range(len(text) // 4))}


def test_preflight_refuses_a_truncating_max_prompt_length():
    """The bug this guard exists for: every rendered retail prompt is ~5-7k
    tokens and the old default was 2048, which TRL applies by keeping the LAST
    2048 tokens -- silently deleting the system prompt, the retail policy and
    all 16 tool schemas from every single training example."""
    rows = [{"prompt": "x" * 24000}]
    args = parse_args(["--max-prompt-length", "2048"])
    with pytest.raises(ValueError, match="would truncate"):
        check_prompt_lengths(rows, _FakeTokenizer(), args)

    override = parse_args(["--max-prompt-length", "2048", "--allow-prompt-truncation"])
    assert check_prompt_lengths(rows, _FakeTokenizer(), override) == 6000


def test_preflight_default_does_not_truncate():
    args = parse_args([])
    assert args.max_prompt_length is None
    assert check_prompt_lengths([{"prompt": "x" * 24000}], _FakeTokenizer(), args) == 6000


def test_defaults_target_gradient_quality_not_trl_defaults():
    args = parse_args([])
    assert args.scale_rewards is False, "dividing by group std amplifies near-degenerate groups"
    assert args.loss_type == "dr_grpo"
    assert args.shaping is True
    assert args.gradient_accumulation_steps > 1, "grad-accum 1 means 2 unique prompts per update"


def test_filter_supported_drops_unknown_config_keys():
    import dataclasses

    from tau_forge.train.grpo_train import filter_supported

    @dataclasses.dataclass
    class OldConfig:
        output_dir: str = ""
        temperature: float = 1.0

    out = filter_supported({"output_dir": "x", "temperature": 0.9, "scale_rewards": False}, OldConfig)
    assert out == {"output_dir": "x", "temperature": 0.9}


def test_build_config_kwargs_enables_held_out_checkpoint_selection(tmp_path):
    args = parse_args(["--val-fraction", "0.1"])
    kwargs = build_config_kwargs(args, tmp_path, max_steps=-1, bf16=True)
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_reward"
    assert "max_steps" not in kwargs

    off = build_config_kwargs(parse_args(["--val-fraction", "0"]), tmp_path, max_steps=50, bf16=False)
    assert "load_best_model_at_end" not in off
    assert off["max_steps"] == 50


def test_resolve_mix_accepts_named_and_inline_specs():
    from tau_forge.train.grpo_train import resolve_mix

    assert resolve_mix(None) is None
    assert resolve_mix("real") == curriculum.REAL_TASK_ALIGNED_MIX
    assert resolve_mix("happy_path=0.5,ambiguous=0.5") == {"happy_path": 0.5, "ambiguous": 0.5}


def test_every_parser_can_format_its_help():
    """argparse %-formats help strings, so a literal '%' in help text raises
    only when --help is actually rendered -- which no other test does. Caught
    a real ValueError from '~72%' and '53%' in two help strings."""
    from tau_forge.eval.run_tau2 import parse_args as eval_parse_args
    from tau_forge.train.grpo_train import parse_args as train_parse_args

    for parse in (train_parse_args, eval_parse_args):
        with pytest.raises(SystemExit) as exit_info:
            parse(["--help"])
        assert exit_info.value.code == 0


# --------------------------------------------------------------------------
# scorecard: measured per-cell diagnosis
# --------------------------------------------------------------------------


def _audit(**cells):
    """Build a per_scenario_scores table: cells maps 'category__theme' to a
    list of per-scenario score lists."""
    out = {}
    for cell, groups in cells.items():
        for i, scores in enumerate(groups):
            out[f"{cell}__{i}"] = scores
    return out


def test_classify_separates_the_four_group_shapes():
    from tau_forge.train.scorecard import classify

    assert classify([0.0, 0.3, 1.0]) == "usable"
    assert classify([0.0, 0.0, 0.0]) == "cold_start"
    assert classify([1.0, 1.0, 1.0]) == "already_solved"
    assert classify([0.3, 0.3, 0.3]) == "stuck_partial"


def test_signal_ranks_a_varying_hard_cell_above_a_varying_easy_one():
    """The whole point of headroom: two cells can both vary, and the one with
    reward left on the table is where compute converts into improvement."""
    from tau_forge.train.scorecard import score_cells

    scores = _audit(
        happy_path__hard=[[0.0, 0.3], [0.0, 0.6]],
        happy_path__easy=[[0.95, 1.0], [0.9, 1.0]],
    )
    by_cell = {c.cell: c for c in score_cells(scores)}
    hard, easy = by_cell["happy_path__hard"], by_cell["happy_path__easy"]
    assert hard.yield_ == easy.yield_ == 1.0
    assert hard.headroom > easy.headroom
    assert hard.signal > easy.signal


def test_a_fully_flat_cell_has_zero_signal_however_wrong_it_is():
    from tau_forge.train.scorecard import score_cells

    (cell,) = score_cells(_audit(out_of_scope__dead=[[0.0] * 4, [0.0] * 4]))
    assert cell.headroom == 1.0
    assert cell.yield_ == 0.0
    assert cell.signal == 0.0, "no gradient means no value, no matter how much headroom"
    assert cell.n_cold == 2


def test_recommend_mix_downweights_a_measured_dead_category():
    from tau_forge.train.scorecard import recommend_mix, score_cells

    cells = score_cells(
        _audit(
            happy_path__a=[[0.0, 0.6], [0.2, 0.9]],
            requires_earlier_context__a=[[0.0, 0.5], [0.1, 0.8]],
            out_of_scope__a=[[0.0] * 4, [0.0] * 4],
        )
    )
    mix = recommend_mix(cells)
    assert mix["out_of_scope"] < mix["happy_path"]
    assert mix["out_of_scope"] > 0.0, "a floor keeps guardrail behavior represented"
    assert sum(mix.values()) == pytest.approx(1.0, abs=1e-3)


def test_relevance_prior_can_override_raw_measured_signal():
    """out_of_scope is hard for the base model, so measured signal alone would
    over-invest in a shape only 3.5% of real tasks have."""
    from tau_forge.train.scorecard import UNIFORM_RELEVANCE, recommend_mix, score_cells

    cells = score_cells(
        _audit(
            out_of_scope__a=[[0.0, 0.9], [0.0, 0.8]],
            happy_path__a=[[0.8, 0.9], [0.85, 0.95]],
        )
    )
    benchmark = recommend_mix(cells)
    uniform = recommend_mix(cells, relevance=UNIFORM_RELEVANCE)
    assert uniform["out_of_scope"] > benchmark["out_of_scope"]


def test_recommend_mix_falls_back_to_relevance_when_nothing_varies():
    from tau_forge.train.scorecard import recommend_mix, score_cells

    cells = score_cells(_audit(happy_path__a=[[0.0] * 3], ambiguous__a=[[0.0] * 3]))
    mix = recommend_mix(cells)
    assert sum(mix.values()) == pytest.approx(1.0, abs=1e-3)
    assert mix["happy_path"] > mix["ambiguous"]


def test_load_scores_prefers_shaped_when_present(tmp_path):
    from tau_forge.train.scorecard import load_scores

    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "per_scenario_scores": {"x__y__1": [0.0, 0.0]},
                "per_scenario_shaped_scores": {"x__y__1": [0.0, 0.1]},
            }
        )
    )
    scores, basis = load_scores(path)
    assert scores["x__y__1"] == [0.0, 0.1] and "shaping" in basis
    raw, basis_raw = load_scores(path, prefer_shaped=False)
    assert raw["x__y__1"] == [0.0, 0.0] and "alone" in basis_raw


def test_emitted_mix_is_accepted_by_grpo_train():
    """The scorecard's output has to be pasteable straight into the trainer."""
    from tau_forge.train.grpo_train import resolve_mix
    from tau_forge.train.scorecard import recommend_mix, score_cells

    cells = score_cells(_audit(happy_path__a=[[0.0, 0.6]], ambiguous__a=[[1.0, 1.0]]))
    mix = recommend_mix(cells)
    spec = ",".join(f"{c}={v}" for c, v in sorted(mix.items()))
    assert resolve_mix(spec) == pytest.approx(mix)


def test_recommended_mix_applies_cleanly_to_the_real_corpus():
    from tau_forge.train.scorecard import recommend_mix, score_cells

    cells = score_cells(
        _audit(
            **{
                f"{category}__{theme}": [[0.0, 0.5], [0.2, 0.8]]
                for category in curriculum.REAL_TASK_ALIGNED_MIX
                for theme in ("identity_and_order_lookup",)
            }
        )
    )
    mix = recommend_mix(cells)
    mixed = curriculum.apply_mixture(build_examples(), mix, seed=0)
    assert mixed and curriculum.summarize(mixed)["n_cells"] == 30


# --------------------------------------------------------------------------
# split selection: a synthetic before/after has to be measured on held-out data
# --------------------------------------------------------------------------


def test_baseline_and_trainer_compute_the_same_split_from_the_same_flags():
    """The whole contract behind `zero_shot_baseline --split val`: given equal
    --val-fraction / --category-mix / --curriculum-seed, the ids it scores are
    exactly the ones grpo_train holds out. If these ever diverge, a synthetic
    before/after silently becomes a train-on-test measurement."""
    from tau_forge.train.grpo_train import build_examples_for_run, parse_args as train_args

    examples = build_examples()
    args = train_args(["--val-fraction", "0.1", "--category-mix", "real", "--curriculum-seed", "3"])
    trainer_train, trainer_val = build_examples_for_run(args)

    baseline_train, baseline_val = curriculum.build_training_sets(
        examples,
        mix=curriculum.REAL_TASK_ALIGNED_MIX,
        val_fraction=0.1,
        seed=3,
    )
    assert {e.id for e in baseline_val} == {e.id for e in trainer_val}
    assert {e.id for e in baseline_train} == {e.id for e in trainer_train}
    assert baseline_val, "an empty held-out split would make the comparison vacuous"


def test_val_split_is_disjoint_from_what_training_sees():
    examples = build_examples()
    train, val = curriculum.build_training_sets(examples, val_fraction=0.1, seed=0)
    assert not ({e.id for e in train} & {e.id for e in val})


def test_baseline_parser_defaults_match_the_trainer_defaults():
    """Divergent defaults are the likeliest way these two commands end up
    computing different splits without anyone noticing, since the split is
    silent -- nothing in either output announces which scenarios were used."""
    from tau_forge.train.grpo_train import parse_args as train_args
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args

    trainer = train_args([])
    baseline = baseline_args([])
    for field in ("val_fraction", "curriculum_seed", "category_mix"):
        assert getattr(trainer, field) == getattr(baseline, field), field


def test_baseline_split_flag_defaults_to_the_whole_corpus():
    """--split all is right for a variance audit and wrong for a before/after;
    defaulting to the audit keeps the existing workflow unchanged."""
    from tau_forge.train.zero_shot_baseline import parse_args as baseline_args

    assert baseline_args([]).split == "all"
    assert baseline_args(["--split", "val"]).split == "val"


# ---------------------------------------------------------------------------
# Scoring throughput. These exist because the audit's scoring phase is silent,
# runs after vLLM has printed its shutdown banner, and at n=16 over the corpus
# takes long enough to look like a hung process. Both speedups below are only
# safe because they change *when* work happens, never what it computes -- so
# each test pins the results, not the timing.
# ---------------------------------------------------------------------------

_CALL = '<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'


def _completions_for(order_id: str) -> list[str]:
    """A spread that hits every branch of `reward()` that matters here: exact
    outcome match, right tool with a graded field wrong, right tool on the
    wrong record, a different tool, and no call at all."""
    import json as _json

    return [
        _CALL.format(
            name="cancel_pending_order",
            args=_json.dumps({"order_id": order_id, "reason": "ordered by mistake"}),
        ),
        _CALL.format(
            name="cancel_pending_order",
            args=_json.dumps({"order_id": order_id, "reason": "no longer needed"}),
        ),
        _CALL.format(
            name="cancel_pending_order",
            args=_json.dumps({"order_id": "#W2974929", "reason": "ordered by mistake"}),
        ),
        _CALL.format(name="get_order_details", args=_json.dumps({"order_id": order_id})),
        "Sure, let me take a look at that for you.",
    ]


def test_precomputed_gold_outcome_grades_identically():
    """`reward(..., gold_outcome=...)` is a pure caching hook. If it ever
    disagrees with recomputing gold, every cached score in a run is wrong."""
    import json

    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    from tau_forge.envs.retail import execute_against
    from tau_forge.reward.reward import Action, reward
    from tau_forge.train.completion_parsing import parse_completion

    db = RetailDB.load(RETAIL_DB_PATH)
    gold = Action(
        tool_name="cancel_pending_order",
        tool_input={"order_id": "#W5918442", "reason": "ordered by mistake"},
    )
    precomputed = execute_against(db, gold.tool_name, gold.tool_input)
    before = db.model_dump(mode="json")

    for text in _completions_for("#W5918442"):
        name, args = parse_completion(text)
        rollout = Action(tool_name=name, tool_input=args)
        fresh = reward(rollout, gold, db)
        cached = reward(rollout, gold, db, gold_outcome=precomputed)
        assert (fresh.score, fresh.reason) == (cached.score, cached.reason), text

    assert db.model_dump(mode="json") == before, "reward() mutated db_state"


def test_gold_cache_is_transparent_and_bounded():
    """A warm cache must score a group exactly like a cold one -- the cached
    `RetailDB` is shared between calls, so a single mutation downstream would
    corrupt every later score in the run. The bound matters because each entry
    pins a whole db copy."""
    from tau_forge.train import reward_adapter

    expected = "cancel_pending_order"
    args = {"order_id": "#W5918442", "reason": "ordered by mistake"}
    texts = _completions_for("#W5918442") * 3

    reward_adapter._gold_cache.clear()
    cold = []
    for text in texts:
        reward_adapter._gold_cache.clear()
        cold.append(reward_adapter.score_completion(text, expected, args))

    reward_adapter._gold_cache.clear()
    warm = [reward_adapter.score_completion(text, expected, args) for text in texts]
    assert cold == warm

    for order_id in ("#W5918442", "#W2974929", "#W2631563", "#W5918442"):
        for text in _completions_for(order_id):
            reward_adapter.score_completion(
                text, expected, {"order_id": order_id, "reason": "ordered by mistake"}
            )
    assert len(reward_adapter._gold_cache) <= reward_adapter._GOLD_CACHE_SIZE


def test_gold_is_not_executed_for_a_wrong_tool_completion():
    """Gold execution is the expensive half of a scoring call and `reward()`
    never reaches it when the tool name is wrong. Populating the cache anyway
    would slow down the single most common completion on a cold-start policy."""
    from tau_forge.train import reward_adapter

    reward_adapter._gold_cache.clear()
    reward_adapter.score_completion(
        _CALL.format(name="get_order_details", args='{"order_id": "#W5918442"}'),
        "cancel_pending_order",
        {"order_id": "#W5918442", "reason": "ordered by mistake"},
    )
    reward_adapter.score_completion("no call at all", "cancel_pending_order", {})
    assert not reward_adapter._gold_cache


def test_pooled_scoring_matches_in_process_scoring():
    """`--score-workers` splits scoring by scenario across processes. Worker
    state (the shared db, the gold cache, the shaping env) is per-process, so
    this pins that none of it leaks into the numbers."""
    import json
    from concurrent.futures import ProcessPoolExecutor

    from tau_forge.train.zero_shot_baseline import _score_scenario

    payloads = [
        (
            "cancel_pending_order",
            json.dumps({"order_id": order_id, "reason": "ordered by mistake"}),
            _completions_for(order_id),
            True,
            True,
        )
        for order_id in ("#W5918442", "#W2974929", "#W2631563")
    ]

    serial = [_score_scenario(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=2) as pool:
        pooled = list(pool.map(_score_scenario, payloads, chunksize=1))
    assert serial == pooled


def test_score_worker_count_resolves_to_something_runnable():
    from tau_forge.train.zero_shot_baseline import resolve_score_workers

    assert resolve_score_workers(3) == 3
    assert 1 <= resolve_score_workers(0) <= 16
