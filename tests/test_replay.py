"""tests/test_replay.py — Pipeline.replay_turn (M8), blueprint §6.7/§20.14.
Phase 5 gate file (blueprint §10 Phase 5).

Named to match the blueprint's §9 test map row exactly:
test_replay_preserves_original_and_history.
"""

import pytest

from models.enums import Condition

from tests.pipeline_fixtures import build_pipeline, default_goal_state, default_raw_appraisal

RUN_ID = "run-1"
BASELINE_PRIORITIES = {"autonomy": 0.6, "safety": 0.4}


def test_replay_preserves_original_and_history(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path,
        # Deliberately DIFFERENT H_t between the two turns, so turn 2's
        # a_star differs from its own a_prev (turn 1's a_t) — otherwise
        # persistence-blending two IDENTICAL values would produce the same
        # A_t at every rho, and this test's whole point (varying rho on
        # replay actually changes A_t) would pass for the wrong reason.
        extract_responses=[
            default_raw_appraisal(),
            default_raw_appraisal(goal_relevance=0.9, uncertainty=0.1, affect_intensity=0.9),
        ],
        generate_responses=["turn one response", "turn two response", "replayed response"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    # Two real DYNAMIC turns, so turn 2 has a real a_prev to persist against.
    pipeline.run_turn(RUN_ID, "turn one", {}, Condition.DYNAMIC, 0.35, g_t)
    original = pipeline.run_turn(RUN_ID, "turn two", {}, Condition.DYNAMIC, 0.35, g_t)

    history_before_replay = list(pipeline._history)
    a_prev_before_replay = dict(pipeline._a_prev_by_condition)

    replayed = pipeline.replay_turn(turn_id=2, condition=Condition.DYNAMIC, rho=0.9)

    # A different rho, applied to the SAME reused a_prev/a_star, produces a
    # different A_t than the original turn got with rho=0.35.
    assert replayed.rho == 0.9
    assert replayed.a_t != original.a_t
    # H_t/D_t/A*_t are REUSED, not recomputed.
    assert replayed.appraisal == original.appraisal
    assert replayed.derived_state == original.derived_state
    assert replayed.a_star == original.a_star
    assert replayed.a_prev == original.a_prev
    assert replayed.goal_state == original.goal_state
    assert replayed.replay_parent_turn_id == 2
    assert replayed.comparison_id is None
    assert replayed.record_id != original.record_id

    # The original run's committed state is untouched: same shared
    # history, same per-condition A_prev, and the original record's own
    # fields are unchanged.
    assert pipeline._history == history_before_replay
    assert pipeline._a_prev_by_condition == a_prev_before_replay
    assert original.rho == 0.35
    assert original.a_t == pipeline._a_prev_by_condition[Condition.DYNAMIC]


def test_replay_task_focused_never_computes_agent_state(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path, extract_responses=[default_raw_appraisal()], generate_responses=["r1", "replay"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    pipeline.run_turn(RUN_ID, "turn one", {}, Condition.TASK_FOCUSED, 0.0, g_t)
    replayed = pipeline.replay_turn(turn_id=1, condition=Condition.TASK_FOCUSED, rho=0.9)

    assert replayed.a_prev is None
    assert replayed.a_star is None
    assert replayed.a_t is None
    assert replayed.rho == 0.0  # a supplied rho is silently zeroed, exactly like resolve_rho does


def test_replay_missing_turn_raises(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(tmp_path)
    with pytest.raises(KeyError, match="turn_id=99"):
        pipeline.replay_turn(turn_id=99, condition=Condition.DYNAMIC)


def test_replay_current_cue_always_zeroes_rho(tmp_path):
    """Post-delivery fix: Pipeline.replay_turn() previously only forced
    rho=0.0 for TASK_FOCUSED (and, incidentally, whenever a_star was None).
    CURRENT_CUE — which the frozen specification requires to stay at rho=0
    exactly as strictly as TASK_FOCUSED stays at a_t=None, "no override can
    ever make CURRENT_CUE persistent" — fell through to the else branch
    and applied whatever rho a caller supplied. This asserts the fix at the
    Pipeline layer directly (see test_experiment_controller.py for the
    matching ExperimentController-layer assertion)."""
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path,
        extract_responses=[
            default_raw_appraisal(),
            default_raw_appraisal(goal_relevance=0.9, uncertainty=0.1, affect_intensity=0.9),
        ],
        generate_responses=["turn one", "turn two", "replay"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    pipeline.run_turn(RUN_ID, "turn one", {}, Condition.CURRENT_CUE, 0.0, g_t)
    pipeline.run_turn(RUN_ID, "turn two", {}, Condition.CURRENT_CUE, 0.0, g_t)

    replayed = pipeline.replay_turn(turn_id=2, condition=Condition.CURRENT_CUE, rho=0.9)

    assert replayed.rho == 0.0  # the supplied rho=0.9 must NOT survive
    # rho=0 collapses persistence to the target exactly (compared via
    # model_dump(): a_t is an AgentState, a_star an AgentTargetState — same
    # field values, different classes, so == would compare unequal on type
    # alone; tests/test_transition.py established this same convention).
    assert replayed.a_t.model_dump() == replayed.a_star.model_dump()


def test_replay_is_persisted_but_does_not_mutate_committed_state(tmp_path):
    """Post-delivery fix: replay_turn() constructed a TurnRecord and
    returned it, but never called TurnLogger.persist() — so a replay left
    no audit trail, contradicting logger.py's own "replay linkage"
    description (§6.8). Persisting is additive (a new, separately-
    identified record), not a mutation of the run's own committed state —
    shared history, per-condition A_prev, and the (turn_id, condition)
    replay-source cache (which must keep pointing at the ORIGINAL record,
    so a second replay of the same turn always replays the original, never
    a replay-of-a-replay) all still must not move."""
    pipeline, _adapter, logger = build_pipeline(
        tmp_path,
        extract_responses=[default_raw_appraisal()],
        generate_responses=["turn one", "replay", "replay again"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    original = pipeline.run_turn(RUN_ID, "turn one", {}, Condition.DYNAMIC, 0.35, g_t)

    records_before = len(logger.records)
    history_before = list(pipeline._history)
    state_before = dict(pipeline._a_prev_by_condition)

    replayed = pipeline.replay_turn(turn_id=1, condition=Condition.DYNAMIC, rho=0.9)

    assert len(logger.records) == records_before + 1
    assert logger.records[-1].record_id == replayed.record_id
    assert replayed.replay_parent_turn_id == original.turn_id

    assert pipeline._history == history_before
    assert pipeline._a_prev_by_condition == state_before
    # A second replay of the same turn still replays the ORIGINAL, not the
    # first replay — the cache entry was never overwritten.
    replayed_again = pipeline.replay_turn(turn_id=1, condition=Condition.DYNAMIC, rho=0.9)
    assert replayed_again.a_star == original.a_star
    assert replayed_again.record_id != replayed.record_id


def test_replay_without_rho_reuses_original_rho(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path,
        extract_responses=[default_raw_appraisal(), default_raw_appraisal()],
        generate_responses=["turn one", "turn two", "replay"],
    )
    pipeline.outcome_baseline_store.elicit(RUN_ID, BASELINE_PRIORITIES)
    g_t = default_goal_state()

    pipeline.run_turn(RUN_ID, "turn one", {}, Condition.DYNAMIC, 0.35, g_t)
    original = pipeline.run_turn(RUN_ID, "turn two", {}, Condition.DYNAMIC, 0.35, g_t)

    replayed = pipeline.replay_turn(turn_id=2, condition=Condition.DYNAMIC)  # rho omitted
    assert replayed.rho == original.rho == 0.35
    assert replayed.a_t == original.a_t  # same rho, same reused a_prev/a_star -> same A_t
