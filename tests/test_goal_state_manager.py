"""tests/test_goal_state_manager.py — M3 (GoalStateManager), blueprint
§6.3/§8.1/§20.4. Not itself a Phase 5 gate file (the build-order table
names test_conditions.py/test_outcome_baseline.py/test_replay.py/
test_experiment_controller.py) but written per this project's own
established convention of one test file per service module.
"""

import pytest
from pydantic import ValidationError

from models.enums import GoalUpdateSource
from models.goal_state import GoalState
from services.goal_state_manager import GoalStateManager


def _goal_state(**overrides) -> GoalState:
    defaults = dict(
        objective="decide whether to proceed",
        value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5,
        task_constraints={},
        autonomy_weight=0.5,
        safety_risk=0.1,
    )
    defaults.update(overrides)
    return GoalState(**defaults)


def test_update_delegates_to_with_explicit_update():
    manager = GoalStateManager()
    g_t = _goal_state()
    updated = manager.update(g_t, GoalUpdateSource.HUMAN_EXPLICIT, safety_risk=0.9)

    assert updated.safety_risk == 0.9
    assert updated.goal_version == g_t.goal_version + 1
    assert updated.update_source == GoalUpdateSource.HUMAN_EXPLICIT
    # non-mutating — the original object is untouched
    assert g_t.safety_risk == 0.1
    assert g_t.goal_version == 1


@pytest.mark.parametrize("source", list(GoalUpdateSource))
def test_update_accepts_every_goal_update_source(source):
    """Documents the judgment call in services/goal_state_manager.py's
    module docstring: update() does not restrict which GoalUpdateSource a
    caller passes, since nothing in the blueprint singles any of the three
    out as illegitimate specifically through this method."""
    manager = GoalStateManager()
    g_t = _goal_state()
    updated = manager.update(g_t, source, stakes=0.9)
    assert updated.update_source == source
    assert updated.stakes == 0.9


def test_update_rejects_out_of_range_change_value():
    """update() goes through GoalState.with_explicit_update(), which
    revalidates every changed field via model_validate() rather than
    model_copy(update=...) — an out-of-range change must fail fast, not
    silently produce an invalid GoalState."""
    manager = GoalStateManager()
    g_t = _goal_state()
    with pytest.raises(ValidationError):
        manager.update(g_t, GoalUpdateSource.TASK_EVENT, safety_risk=5.0)
