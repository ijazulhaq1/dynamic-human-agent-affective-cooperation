"""tests/test_experiment_controller.py — ExperimentController (M8),
blueprint §6.7/§12.1/§13.1. Phase 5 gate file (blueprint §10 Phase 5).

Named to match the blueprint's §9 test map rows exactly where it names
them: test_resolve_rho_zeroes_non_dynamic_conditions,
test_compare_delegates_to_pipeline, test_start_new_run_resets_all_state,
test_rho_override_logged_as_intervention.
"""

from models.enums import Condition
from services.experiment_controller import ExperimentController

from tests.pipeline_fixtures import build_pipeline, default_goal_state, default_raw_appraisal

BASELINE_PRIORITIES = {"autonomy": 0.6, "safety": 0.4}


def _build_controller(tmp_path, **kwargs):
    pipeline, adapter, logger = build_pipeline(tmp_path, **kwargs)
    controller = ExperimentController(pipeline, rho_dynamic=0.35)
    pipeline.outcome_baseline_store.elicit(controller.current_run_id, BASELINE_PRIORITIES)
    return controller, pipeline, adapter, logger


def test_resolve_rho_zeroes_non_dynamic_conditions(tmp_path):
    controller, _pipeline, _adapter, _logger = _build_controller(tmp_path)

    assert controller.resolve_rho(Condition.CURRENT_CUE, override=0.35) == 0.0
    assert controller.resolve_rho(Condition.TASK_FOCUSED, override=0.35) == 0.0
    assert controller.resolve_rho(Condition.DYNAMIC, override=0.35) == 0.35
    assert controller.resolve_rho(Condition.DYNAMIC) == 0.35  # falls back to rho_dynamic
    assert controller.resolve_rho(Condition.CURRENT_CUE) == 0.0
    assert controller.resolve_rho(Condition.TASK_FOCUSED) == 0.0


def test_compare_delegates_to_pipeline(tmp_path):
    controller, pipeline, _adapter, _logger = _build_controller(
        tmp_path,
        extract_responses=[default_raw_appraisal()],
        generate_responses=["a", "b", "c"],
    )
    calls = []
    original_compare = pipeline.compare

    def spy_compare(*args, **kwargs):
        calls.append((args, kwargs))
        return original_compare(*args, **kwargs)

    pipeline.compare = spy_compare

    g_t = default_goal_state()
    conditions = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]
    result = controller.compare("hi", {}, g_t, conditions)

    assert len(calls) == 1
    (run_id, _user_text, _task_context, _g_t, called_conditions, rho), kwargs = calls[0]
    assert run_id == controller.current_run_id
    assert called_conditions == conditions
    assert rho == 0.35  # resolve_rho(DYNAMIC) with no override -> rho_dynamic, resolved exactly once
    assert kwargs["rho_override"] is None
    assert set(result) == set(conditions)


def test_start_new_run_resets_all_state(tmp_path):
    controller, pipeline, _adapter, _logger = _build_controller(
        tmp_path, extract_responses=[default_raw_appraisal()] * 2, generate_responses=["a", "b"],
    )
    g_t = default_goal_state()
    controller.run_turn("hi", {}, Condition.DYNAMIC, g_t)
    assert len(pipeline._history) == 1
    assert pipeline._a_prev_by_condition  # DYNAMIC's A_1 was stored

    old_run_id = controller.current_run_id
    new_run_id = controller.start_new_run()

    assert new_run_id != old_run_id
    assert controller.current_run_id == new_run_id
    assert len(pipeline._history) == 0
    assert pipeline._a_prev_by_condition == {}

    # The turn counter really was reset — a fresh turn after start_new_run()
    # starts back at turn_id 1, not 3.
    pipeline.outcome_baseline_store.elicit(new_run_id, BASELINE_PRIORITIES)
    fresh_record = controller.run_turn("hi again", {}, Condition.DYNAMIC, g_t)
    assert fresh_record.turn_id == 1


def test_rho_override_logged_as_intervention(tmp_path):
    controller, _pipeline, _adapter, _logger = _build_controller(
        tmp_path,
        extract_responses=[default_raw_appraisal()] * 3,
        generate_responses=["a", "b", "c"],
    )
    g_t = default_goal_state()

    dynamic_record = controller.run_turn("hi", {}, Condition.DYNAMIC, g_t, rho_override=0.9)
    assert dynamic_record.interventions == [{"type": "rho_override", "value": 0.9}]

    task_focused_record = controller.run_turn("hi", {}, Condition.TASK_FOCUSED, g_t, rho_override=0.9)
    assert task_focused_record.interventions == []

    current_cue_record = controller.run_turn("hi", {}, Condition.CURRENT_CUE, g_t, rho_override=0.9)
    assert current_cue_record.interventions == []


def test_rho_override_logged_as_intervention_in_compare(tmp_path):
    controller, _pipeline, _adapter, _logger = _build_controller(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["a", "b", "c"],
    )
    g_t = default_goal_state()

    results = controller.compare(
        "hi", {}, g_t, [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC], rho_override=0.9,
    )
    assert results[Condition.DYNAMIC].interventions == [{"type": "rho_override", "value": 0.9}]
    assert results[Condition.TASK_FOCUSED].interventions == []
    assert results[Condition.CURRENT_CUE].interventions == []


def test_replay_current_cue_always_zeroes_rho(tmp_path):
    """Post-delivery fix, named to match the reviewer's exact suggested
    test name. ExperimentController.replay_turn() previously forwarded a
    caller's rho straight to Pipeline.replay_turn() unresolved — so
    `controller.replay_turn(turn_id, Condition.CURRENT_CUE, rho=0.9)` could
    make a replayed CURRENT_CUE turn persistent, breaking the frozen
    specification's "no override can ever make CURRENT_CUE persistent"
    guarantee. Fixed by resolving rho through resolve_rho() before
    delegating, exactly like run_turn()/compare() already do."""
    controller, pipeline, _adapter, _logger = _build_controller(
        tmp_path,
        extract_responses=[
            default_raw_appraisal(),
            default_raw_appraisal(goal_relevance=0.9, uncertainty=0.1, affect_intensity=0.9),
        ],
        generate_responses=["turn one", "turn two", "replay"],
    )
    g_t = default_goal_state()

    controller.run_turn("turn one", {}, Condition.CURRENT_CUE, g_t)
    controller.run_turn("turn two", {}, Condition.CURRENT_CUE, g_t)

    replayed = controller.replay_turn(turn_id=2, condition=Condition.CURRENT_CUE, rho=0.9)

    assert replayed.rho == 0.0
    # model_dump() comparison, not == — a_t is AgentState, a_star is
    # AgentTargetState; same convention as tests/test_transition.py.
    assert replayed.a_t.model_dump() == replayed.a_star.model_dump()


def test_elicit_outcome_baseline_routes_through_pipelines_store(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(tmp_path)
    controller = ExperimentController(pipeline, rho_dynamic=0.35)
    baseline = controller.elicit_outcome_baseline(BASELINE_PRIORITIES)
    assert baseline.run_id == controller.current_run_id
    assert pipeline.outcome_baseline_store.get(controller.current_run_id) is baseline
