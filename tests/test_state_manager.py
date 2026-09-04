"""tests/test_state_manager.py — services/state_manager.py's SessionState
(Phase 7). Not named in the blueprint's own test map (state_manager.py is
never assigned to any build phase at all there — see that module's own
docstring for the full documented inconsistency this project resolved by
building it in Phase 7); added per this project's established pattern of
testing every module it builds.
"""

from models.enums import Condition, GoalUpdateSource
from services.experiment_controller import ExperimentController
from services.goal_state_manager import GoalStateManager
from services.state_manager import SessionState

from tests.pipeline_fixtures import build_pipeline, default_goal_state, default_raw_appraisal

BASELINE_PRIORITIES = {"autonomy": 0.6, "safety": 0.4}


def _build_session(tmp_path, **kwargs):
    pipeline, _adapter, _logger = build_pipeline(tmp_path, **kwargs)
    controller = ExperimentController(pipeline, rho_dynamic=0.35)
    pipeline.outcome_baseline_store.elicit(controller.current_run_id, BASELINE_PRIORITIES)
    session = SessionState(controller, GoalStateManager(), default_goal_state())
    return session, controller, pipeline


def test_run_id_reflects_controller(tmp_path):
    session, controller, _pipeline = _build_session(tmp_path)
    assert session.run_id == controller.current_run_id


def test_apply_explicit_goal_update_goes_through_goal_state_manager(tmp_path):
    session, _controller, _pipeline = _build_session(tmp_path)
    original_version = session.goal_state.goal_version

    updated = session.apply_explicit_goal_update(GoalUpdateSource.HUMAN_EXPLICIT, stakes=0.9)

    assert updated is session.goal_state  # apply_explicit_goal_update both returns and stores the new state
    assert session.goal_state.stakes == 0.9
    assert session.goal_state.goal_version == original_version + 1
    assert session.goal_state.update_source == GoalUpdateSource.HUMAN_EXPLICIT


def test_apply_explicit_goal_update_never_mutates_the_original_object(tmp_path):
    session, _controller, _pipeline = _build_session(tmp_path)
    original = session.goal_state

    session.apply_explicit_goal_update(GoalUpdateSource.RESEARCHER_CONFIG, safety_risk=0.5)

    assert original.safety_risk != 0.5  # the object session.goal_state used to point at is untouched
    assert session.goal_state is not original


def test_record_comparison_and_history_for_condition(tmp_path):
    session, controller, _pipeline = _build_session(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["a", "b", "c"],
    )
    conditions = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]

    results = controller.compare("hi", {}, session.goal_state, conditions)
    session.record_comparison(results)

    assert session.latest_comparison() == results
    for condition in conditions:
        history = session.history_for_condition(condition)
        assert history == [results[condition]]


def test_history_for_condition_accumulates_across_multiple_comparisons(tmp_path):
    session, controller, _pipeline = _build_session(
        tmp_path,
        extract_responses=[default_raw_appraisal(), default_raw_appraisal()],
        generate_responses=["a1", "d1", "a2", "d2"],
    )
    conditions = [Condition.CURRENT_CUE, Condition.DYNAMIC]

    session.record_comparison(controller.compare("turn one", {}, session.goal_state, conditions))
    session.record_comparison(controller.compare("turn two", {}, session.goal_state, conditions))

    dynamic_history = session.history_for_condition(Condition.DYNAMIC)
    assert len(dynamic_history) == 2
    assert dynamic_history[0].turn_id == 1
    assert dynamic_history[1].turn_id == 2


def test_record_turn_is_kept_separately_from_comparisons(tmp_path):
    session, controller, _pipeline = _build_session(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["solo"],
    )
    record = controller.run_turn("hi", {}, Condition.DYNAMIC, session.goal_state)
    session.record_turn(record)

    assert session.single_turn_records == [record]
    assert session.comparisons == []
    # run_turn() output never appears in any condition's comparison history.
    assert session.history_for_condition(Condition.DYNAMIC) == []


def test_reset_delegates_to_start_new_run_and_clears_own_bookkeeping(tmp_path):
    session, controller, pipeline = _build_session(
        tmp_path,
        extract_responses=[default_raw_appraisal(), default_raw_appraisal()],
        generate_responses=["a", "b", "c", "fresh"],
    )
    conditions = [Condition.TASK_FOCUSED, Condition.CURRENT_CUE, Condition.DYNAMIC]
    session.record_comparison(controller.compare("hi", {}, session.goal_state, conditions))
    old_run_id = session.run_id

    fresh_goal_state = default_goal_state(objective="a fresh scenario")
    new_run_id = session.reset(fresh_goal_state)

    assert new_run_id != old_run_id
    assert session.run_id == new_run_id
    assert session.comparisons == []
    assert session.single_turn_records == []
    assert session.goal_state is fresh_goal_state

    # The underlying Pipeline was really reset too (start_new_run()'s own
    # contract, not reimplemented here) — a fresh turn starts back at
    # turn_id 1, exactly like test_experiment_controller.py's own
    # test_start_new_run_resets_all_state already checks at the
    # ExperimentController layer directly.
    pipeline.outcome_baseline_store.elicit(new_run_id, BASELINE_PRIORITIES)
    fresh_record = controller.run_turn("hi again", {}, Condition.DYNAMIC, session.goal_state)
    assert fresh_record.turn_id == 1
