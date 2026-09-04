"""tests/test_conditions.py — Pipeline (M8), blueprint §6.7/§20.12. Phase 5
gate file (blueprint §10 Phase 5): part of "test_conditions.py,
test_outcome_baseline.py, test_replay.py, test_experiment_controller.py
green."

Named to match the blueprint's §9 test map rows exactly where it names
them: test_same_history_across_conditions,
test_history_items_serialize_without_recursion,
test_compare_commits_shared_observation_once,
test_compare_history_snapshot_excludes_current_turn,
test_observation_raw_history_populated,
test_run_turn_and_compare_agree_on_history_before,
test_record_id_unique_and_comparison_id_shared,
test_compare_task_focused_never_computes_agent_state. Also covers
test_fallback_paths_produce_valid_record, which the blueprint's own test
map nominally assigns to test_schemas.py by name — filed here instead,
alongside the module (Pipeline) it actually exercises, per this project's
established, documented deviation (Phase 1-4 did the same for rows
nominally tied to an earlier-phase file).
"""

import pytest

from models.enums import Condition, EstimatorStatus, GeneratorStatus
from models.observation import Turn
from models.turn_record import TurnRecord

from tests.pipeline_fixtures import build_pipeline, default_goal_state, default_raw_appraisal

RUN_ID = "run-1"
BASELINE_PRIORITIES = {"autonomy": 0.6, "safety": 0.4}


def test_same_history_across_conditions(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path,
        extract_responses=[default_raw_appraisal()],
        generate_responses=["r-task", "r-cue", "r-dyn"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    results = pipeline.compare(
        RUN_ID, "hello", {}, g_t,
        [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC], rho=0.35,
    )
    observations = [record.observation for record in results.values()]
    # compare() builds O_t exactly once and shares it across every
    # condition's record — the strongest form of "identical raw history":
    # not merely equal, the SAME object.
    assert all(o is observations[0] for o in observations)
    assert all(o.raw_history == observations[0].raw_history for o in observations)


def test_history_items_serialize_without_recursion(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[default_raw_appraisal()] * 2, generate_responses=["r1", "r2"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    pipeline.run_turn(RUN_ID, "turn one", {}, Condition.DYNAMIC, 0.35, g_t)
    record2 = pipeline.run_turn(RUN_ID, "turn two", {}, Condition.DYNAMIC, 0.35, g_t)

    history_item = record2.observation.raw_history[0]
    assert isinstance(history_item, Turn)
    assert not hasattr(history_item, "task_context")  # Turn is non-recursive: no nested Observation
    # Round-trips through JSON without recursion blowing up or losing data.
    reloaded = Turn.model_validate_json(history_item.model_dump_json())
    assert reloaded == history_item


def test_compare_commits_shared_observation_once(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["a", "b", "c"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    assert len(pipeline._history) == 0
    pipeline.compare(
        RUN_ID, "hi", {}, g_t,
        [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC], rho=0.35,
    )
    # One compare() call over three conditions commits exactly one Turn,
    # never one per condition.
    assert len(pipeline._history) == 1


def test_compare_history_snapshot_excludes_current_turn(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["a", "b"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    results = pipeline.compare(RUN_ID, "hi", {}, g_t, [Condition.TASK_FOCUSED, Condition.DYNAMIC], rho=0.35)
    for record in results.values():
        turn_ids_seen = [t.turn_id for t in record.observation.raw_history]
        assert record.observation.turn_id not in turn_ids_seen


def test_observation_raw_history_populated(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[default_raw_appraisal()] * 2, generate_responses=["a", "b"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    record1 = pipeline.run_turn(RUN_ID, "turn one", {}, Condition.TASK_FOCUSED, 0.0, g_t)
    assert record1.observation.raw_history == []  # nothing before the first turn

    record2 = pipeline.run_turn(RUN_ID, "turn two", {}, Condition.TASK_FOCUSED, 0.0, g_t)
    assert len(record2.observation.raw_history) == 1
    assert record2.observation.raw_history[0].user_text == "turn one"


def test_run_turn_and_compare_agree_on_history_before(tmp_path):
    pipeline_a, _adapter_a, _logger_a = build_pipeline(
        tmp_path / "a", extract_responses=[default_raw_appraisal()] * 2, generate_responses=["t1", "t2"],
    )
    pipeline_b, _adapter_b, _logger_b = build_pipeline(
        tmp_path / "b", extract_responses=[default_raw_appraisal()] * 2, generate_responses=["t1", "t2"],
    )
    pipeline_a.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    pipeline_b.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    pipeline_a.run_turn(RUN_ID, "turn one", {}, Condition.TASK_FOCUSED, 0.0, g_t)
    pipeline_b.run_turn(RUN_ID, "turn one", {}, Condition.TASK_FOCUSED, 0.0, g_t)

    record_via_run_turn = pipeline_a.run_turn(RUN_ID, "turn two", {}, Condition.TASK_FOCUSED, 0.0, g_t)
    results_via_compare = pipeline_b.compare(RUN_ID, "turn two", {}, g_t, [Condition.TASK_FOCUSED], rho=0.0)
    record_via_compare = results_via_compare[Condition.TASK_FOCUSED]

    assert record_via_run_turn.observation.raw_history == record_via_compare.observation.raw_history


def test_record_id_unique_and_comparison_id_shared(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path,
        extract_responses=[default_raw_appraisal(), default_raw_appraisal()],
        generate_responses=["a", "b", "c", "d"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    run_turn_record = pipeline.run_turn(RUN_ID, "turn one", {}, Condition.TASK_FOCUSED, 0.0, g_t)
    assert run_turn_record.comparison_id is None

    results = pipeline.compare(
        RUN_ID, "turn two", {}, g_t,
        [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC], rho=0.35,
    )
    comparison_ids = {record.comparison_id for record in results.values()}
    assert len(comparison_ids) == 1
    assert None not in comparison_ids

    all_record_ids = {run_turn_record.record_id, *(record.record_id for record in results.values())}
    assert len(all_record_ids) == 4  # every record_id is globally unique


def test_compare_task_focused_never_computes_agent_state(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["a", "b"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    results = pipeline.compare(RUN_ID, "hi", {}, g_t, [Condition.TASK_FOCUSED, Condition.DYNAMIC], rho=0.35)
    tf_record = results[Condition.TASK_FOCUSED]
    assert tf_record.a_prev is None
    assert tf_record.a_star is None
    assert tf_record.a_t is None
    assert tf_record.rho == 0.0
    assert tf_record.state_delta is None


def test_fallback_paths_produce_valid_record(tmp_path):
    """Named to match the blueprint's §9 test map row exactly (nominally
    assigned to test_schemas.py — see this file's module docstring for why
    it lives here instead). Both the estimator's fallback path (two failed
    extractions) and the generator's fallback path (a TimeoutError) fire on
    the same turn; the resulting TurnRecord must still be complete and
    valid."""
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[None, None], generate_responses=[TimeoutError()],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    record = pipeline.run_turn(RUN_ID, "hi", {}, Condition.DYNAMIC, 0.35, g_t)

    assert isinstance(record, TurnRecord)
    assert record.estimator_status == EstimatorStatus.FALLBACK_LOW_CONFIDENCE
    assert record.generator_status == GeneratorStatus.FALLBACK_TEMPLATE
    assert record.response_text  # never silent
    assert record.appraisal is not None  # FALLBACK_APPRAISAL — still populated, never None here


def test_compare_rejects_duplicate_conditions(tmp_path):
    """Post-delivery fix: compare() previously accepted any list[Condition],
    including a repeated one (e.g. [DYNAMIC, DYNAMIC]) — the second
    occurrence's branch would read the A_prev the FIRST occurrence's own
    _commit_condition_state() call just wrote earlier in the SAME loop,
    silently contaminating a "condition" with a prior branch's own
    persistence output. Rejected eagerly instead."""
    pipeline, _adapter, _logger = build_pipeline(tmp_path)
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()
    with pytest.raises(ValueError, match="unique"):
        pipeline.compare(RUN_ID, "hi", {}, g_t, [Condition.DYNAMIC, Condition.DYNAMIC], rho=0.35)


def test_compare_rejects_empty_conditions(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(tmp_path)
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()
    with pytest.raises(ValueError, match="at least one condition"):
        pipeline.compare(RUN_ID, "hi", {}, g_t, [], rho=0.35)
