"""tests/test_outcome_baseline.py — OutcomeBaselineStore, blueprint
§6/§20.4/§20.19. Phase 5 gate file (blueprint §10 Phase 5): part of
"test_conditions.py, test_outcome_baseline.py, test_replay.py,
test_experiment_controller.py green."

Named to match the blueprint's §9 test map row exactly:
test_baseline_immutable_across_goal_updates. OutcomeBaseline's own
model-level immutability (frozen + MappingProxyType, rejecting both field
reassignment and in-place dict mutation) is already covered in
test_schemas.py (Phase 0) — this file instead covers the SERVICE-level
integration guarantee that gives that immutability meaning across a run:
GoalState.value_priorities can change turn to turn (via GoalStateManager),
but the OutcomeBaseline elicited before turn 1 — and every TurnRecord's
own copy of it — never does.
"""

import pytest

from models.enums import Condition, GoalUpdateSource
from services.goal_state_manager import GoalStateManager
from services.outcome_baseline import OutcomeBaselineStore

from tests.pipeline_fixtures import build_pipeline, default_goal_state, default_raw_appraisal


def test_baseline_immutable_across_goal_updates(tmp_path):
    pipeline, _adapter, _logger = build_pipeline(
        tmp_path,
        extract_responses=[default_raw_appraisal()] * 3,
        generate_responses=["ok"] * 3,
    )
    run_id = "run-1"
    baseline = pipeline.outcome_baseline_store.elicit(run_id, {"autonomy": 0.6, "safety": 0.4})

    manager = GoalStateManager()
    g_t = default_goal_state()

    records = []
    for turn in range(3):
        record = pipeline.run_turn(run_id, f"turn {turn}", {}, Condition.TASK_FOCUSED, 0.0, g_t)
        records.append(record)
        # G_t's OWN value_priorities change every turn — this must NOT
        # move OutcomeBaseline, which is a separate, frozen object.
        g_t = manager.update(
            g_t, GoalUpdateSource.HUMAN_EXPLICIT,
            value_priorities={"autonomy": 0.6 - 0.1 * (turn + 1), "safety": 0.4 + 0.1 * (turn + 1)},
        )

    # GoalState.value_priorities really did change turn to turn...
    assert records[0].goal_state.value_priorities != records[1].goal_state.value_priorities
    assert records[1].goal_state.value_priorities != records[2].goal_state.value_priorities

    # ...but every record's outcome_baseline_value_priorities is the SAME,
    # frozen-at-elicitation dict, unaffected by any of those updates.
    for record in records:
        assert record.outcome_baseline_value_priorities == {"autonomy": 0.6, "safety": 0.4}
        assert record.outcome_baseline_value_priorities == dict(baseline.value_priorities)

    # And the store's own OutcomeBaseline object is still the original.
    assert pipeline.outcome_baseline_store.get(run_id) is baseline


def test_elicit_twice_for_same_run_id_raises():
    store = OutcomeBaselineStore()
    store.elicit("run-1", {"autonomy": 0.5})
    with pytest.raises(ValueError, match="run-1"):
        store.elicit("run-1", {"autonomy": 0.9})


def test_get_without_elicit_raises():
    store = OutcomeBaselineStore()
    with pytest.raises(KeyError, match="run-1"):
        store.get("run-1")


def test_elicit_returns_frozen_baseline_matching_run_id():
    store = OutcomeBaselineStore()
    baseline = store.elicit("run-1", {"autonomy": 0.5, "safety": 0.5})
    assert baseline.run_id == "run-1"
    assert dict(baseline.value_priorities) == {"autonomy": 0.5, "safety": 0.5}
    assert store.get("run-1") is baseline
