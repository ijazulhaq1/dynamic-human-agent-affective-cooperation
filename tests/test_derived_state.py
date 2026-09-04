"""tests/test_derived_state.py — M4 (derive_interaction_state), blueprint §6.3/§20.5.

Phase 1 gate (blueprint §9 test map, §10 Phase 1). Every d_goal/d_amb fixture
below is hand-calculated against the exact §20.5 formula, not asserted
against the code's own output, per the interview-prep decision to test
Phase 1 against small synthetic fixtures rather than the (still unwritten)
real interview demo fixture — that belongs to Phase 6 only.
"""

from datetime import datetime, timezone

import pytest

from models.goal_state import GoalState
from models.observation import Observation
from services.derived_features import ScenarioConfigError, derive_interaction_state


def _goal_state(**overrides) -> GoalState:
    defaults = dict(
        objective="decide whether to proceed",
        value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5,
        task_constraints={},
        autonomy_weight=0.5,
        safety_risk=0.2,
    )
    defaults.update(overrides)
    return GoalState(**defaults)


def _observation(task_context: dict) -> Observation:
    return Observation(
        turn_id=1,
        timestamp=datetime.now(timezone.utc),
        user_text="x",
        task_context=task_context,
    )


# ---------- d_goal: structured-formula fixtures ----------


def test_goal_conflict_zero_when_impact_is_purely_supportive():
    """value_priorities={autonomy:.6, safety:.4}, impacts={autonomy:1.0}.
    w_autonomy=.6 (already normalized, weights sum to 1). P = .6*1.0 = .6,
    N = 0 (impacts has no negative-support term). min(P,N)=0 -> d_goal=0."""
    g_t = _goal_state(value_priorities={"autonomy": 0.6, "safety": 0.4})
    o_t = _observation({"options": {"A": {"autonomy": 1.0}}, "focal_option_id": "A"})
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.goal_conflict == pytest.approx(0.0)
    assert d_t.goal_conflict_source == "structured_formula"


def test_goal_conflict_maximal_when_impact_is_fully_opposed():
    """value_priorities={autonomy:.5, safety:.5}, impacts={autonomy:1.0, safety:-1.0}.
    w_autonomy=w_safety=.5. P = .5*1.0 + .5*0 = .5. N = .5*0 + .5*1.0 = .5.
    min(P,N)=.5 -> d_goal = clip01(2*.5) = 1.0 (the 'high conflict' fixture)."""
    g_t = _goal_state(value_priorities={"autonomy": 0.5, "safety": 0.5})
    o_t = _observation(
        {"options": {"A": {"autonomy": 1.0, "safety": -1.0}}, "focal_option_id": "A"}
    )
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.goal_conflict == pytest.approx(1.0)


def test_goal_conflict_moderate_partial_opposition():
    """value_priorities={autonomy:.7, safety:.3}, impacts={autonomy:.8, safety:-.5}.
    P = .7*.8 + .3*0 = .56. N = .7*0 + .3*.5 = .15. min(P,N)=.15 ->
    d_goal = clip01(2*.15) = .30."""
    g_t = _goal_state(value_priorities={"autonomy": 0.7, "safety": 0.3})
    o_t = _observation(
        {"options": {"A": {"autonomy": 0.8, "safety": -0.5}}, "focal_option_id": "A"}
    )
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.goal_conflict == pytest.approx(0.30)


def test_option_impact_above_one_fails_fast():
    """q_j must be normalized to [-1,1] (§20.5). An out-of-range impact must
    be rejected before the d_goal formula runs — not silently repaired by
    the formula's own clip01(), which would produce an indistinguishable-
    from-legitimate goal_conflict=1.0 for scenario-config nonsense like
    {"a": 2.0, "b": -2.0} (implementation-review fix)."""
    g_t = _goal_state(value_priorities={"a": 0.5, "b": 0.5})
    o_t = _observation({"options": {"A": {"a": 2.0, "b": -2.0}}, "focal_option_id": "A"})
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)


def test_option_impact_below_minus_one_fails_fast():
    g_t = _goal_state(value_priorities={"autonomy": 1.0})
    o_t = _observation({"options": {"A": {"autonomy": -1.5}}, "focal_option_id": "A"})
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)


def test_non_numeric_option_impact_fails_fast():
    g_t = _goal_state(value_priorities={"autonomy": 1.0})
    o_t = _observation({"options": {"A": {"autonomy": "high"}}, "focal_option_id": "A"})
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)


def test_goal_conflict_normalizes_weights_that_do_not_sum_to_one():
    """value_priorities={autonomy:.2, safety:.1} (sums to .3, not 1) must be
    normalized before use: w_autonomy=2/3, w_safety=1/3. impacts={autonomy:-1.0,
    safety:1.0}. P = 2/3*0 + 1/3*1.0 = 1/3. N = 2/3*1.0 + 1/3*0 = 2/3.
    min(P,N)=1/3 -> d_goal = clip01(2/3) = 2/3."""
    g_t = _goal_state(value_priorities={"autonomy": 0.2, "safety": 0.1})
    o_t = _observation(
        {"options": {"A": {"autonomy": -1.0, "safety": 1.0}}, "focal_option_id": "A"}
    )
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.goal_conflict == pytest.approx(2 / 3)


def test_goal_conflict_uses_task_rule_override_when_no_options_given():
    g_t = _goal_state()
    o_t = _observation({"d_goal_override": 0.42})
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.goal_conflict == pytest.approx(0.42)
    assert d_t.goal_conflict_source == "task_rule"


def test_goal_conflict_defaults_to_zero_with_no_options_and_no_override():
    g_t = _goal_state()
    o_t = _observation({})
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.goal_conflict == pytest.approx(0.0)
    assert d_t.goal_conflict_source == "task_rule"


# ---------- d_amb: required-evidence-resolution fixtures ----------


def test_evidence_ambiguity_zero_when_all_evidence_resolved():
    g_t = _goal_state()
    o_t = _observation(
        {"required_evidence": [{"status": "resolved"}, {"status": "resolved"}]}
    )
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.evidence_ambiguity == pytest.approx(0.0)
    assert d_t.ambiguity_source == "required_evidence_formula"


def test_evidence_ambiguity_partial_when_some_evidence_missing():
    g_t = _goal_state()
    o_t = _observation(
        {
            "required_evidence": [
                {"status": "resolved"},
                {"status": "missing"},
                {"status": "resolved"},
                {"status": "resolved"},
            ]
        }
    )
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.evidence_ambiguity == pytest.approx(0.25)


def test_evidence_ambiguity_high_when_evidence_is_contradictory():
    g_t = _goal_state()
    o_t = _observation(
        {
            "required_evidence": [
                {"status": "contradictory"},
                {"status": "contradictory"},
                {"status": "resolved"},
            ]
        }
    )
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.evidence_ambiguity == pytest.approx(2 / 3)


def test_evidence_ambiguity_uses_task_rule_override_when_no_required_evidence():
    g_t = _goal_state()
    o_t = _observation({"d_amb_override": 0.55})
    d_t = derive_interaction_state(o_t, g_t)
    assert d_t.evidence_ambiguity == pytest.approx(0.55)
    assert d_t.ambiguity_source == "task_rule"


def test_unknown_evidence_status_fails_fast_rather_than_counting_as_resolved():
    """A typo'd status must not silently count as resolved — {"status": "typo"}
    previously produced evidence_ambiguity=0.0 with no error
    (implementation-review fix)."""
    g_t = _goal_state()
    o_t = _observation({"required_evidence": [{"status": "typo"}]})
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)


# ---------- D_t never reads H_t (structural check) ----------


def test_derive_interaction_state_signature_has_no_h_t_parameter():
    """Checkable structurally, not just by convention (blueprint §6.3 note):
    the bug fixed two blueprint review rounds ago — c_t/H_t feeding D_t —
    cannot be reintroduced by accident if the signature itself has nowhere
    to put h_t."""
    import inspect

    params = inspect.signature(derive_interaction_state).parameters
    assert "h_t" not in params
    assert list(params) == ["o_t", "g_t"]


# ---------- ScenarioConfigError paths ----------


def test_unknown_focal_option_id_raises_scenario_config_error():
    g_t = _goal_state()
    o_t = _observation({"options": {"A": {"autonomy": 1.0}}, "focal_option_id": "B"})
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)


def test_option_impacting_unweighted_value_raises_scenario_config_error():
    g_t = _goal_state(value_priorities={"autonomy": 1.0})
    o_t = _observation(
        {"options": {"A": {"autonomy": 0.5, "reputation": 0.5}}, "focal_option_id": "A"}
    )
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)


def test_zero_weight_value_priorities_raises_scenario_config_error():
    """A goal state that weights nothing cannot support normalization
    (division by zero) — surfaced loudly as ScenarioConfigError, never a
    silent NaN or ZeroDivisionError mid-demo."""
    g_t = _goal_state(value_priorities={"autonomy": 0.0, "safety": 0.0})
    o_t = _observation({"options": {"A": {"autonomy": 1.0}}, "focal_option_id": "A"})
    with pytest.raises(ScenarioConfigError):
        derive_interaction_state(o_t, g_t)
