"""tests/test_transition.py — M5 (StateTransitionEngine), blueprint §6.4/§9.1/§20.6.

Phase 1 gate (blueprint §9 test map, §10 Phase 1). compute_target's fixture
is hand-calculated against the exact §20.6 weighted-sum formula using the
real config/default.yaml weights (loaded here, not re-typed as a second copy)
so a drift between the config file and the formula would fail this test —
not against the still-unwritten Phase 6 interview demo fixture.
"""

from pathlib import Path

import pytest
import yaml

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.enums import EvidenceStrength
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from services.state_transition import LinearPersistenceTransition

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _load_transition_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["transition"]


def _human_appraisal(**overrides) -> HumanAppraisal:
    defaults = dict(
        goal_relevance=0.8,
        goal_congruence=0.0,
        uncertainty=0.6,
        perceived_control=0.5,
        agency=0.5,
        affect_intensity=0.4,
        evidence_strength=EvidenceStrength.STRONG_INDIRECT,
    )
    defaults.update(overrides)
    return HumanAppraisal(**defaults)


def _goal_state(**overrides) -> GoalState:
    defaults = dict(
        objective="decide whether to proceed",
        value_priorities={},
        stakes=0.5,
        task_constraints={},
        autonomy_weight=0.5,
        safety_risk=0.2,
    )
    defaults.update(overrides)
    return GoalState(**defaults)


def _derived_state(**overrides) -> DerivedInteractionState:
    defaults = dict(
        goal_conflict=0.1,
        evidence_ambiguity=0.3,
        goal_conflict_source="task_rule",
        ambiguity_source="task_rule",
    )
    defaults.update(overrides)
    return DerivedInteractionState(**defaults)


@pytest.fixture(scope="module")
def transition_config() -> dict:
    return _load_transition_config()


@pytest.fixture()
def engine(transition_config) -> LinearPersistenceTransition:
    return LinearPersistenceTransition(weights=transition_config["weights"])


# ---------- compute_target: hand-calculated against config/default.yaml's weights ----------


def test_compute_target_matches_hand_calculated_fixture(engine):
    """h_t: h_rel=.8, h_unc=.6, h_int=.4. g_t: stakes=.5, safety_risk=.2.
    d_t: d_amb=.3, d_goal=.1. With the frozen weights (a_mot: .60/.40;
    a_info: .45/.30/.25; a_intv: .35/.30/.20/.15/-.35):

    a_mot  = .60*.8 + .40*.5                                = .68
    a_info = .45*.6 + .30*.3 + .25*.5                        = .485
    a_intv = .35*.68 + .30*.485 + .20*.1 + .15*.4 - .35*.2   = .3935
    """
    h_t = _human_appraisal(goal_relevance=0.8, uncertainty=0.6, affect_intensity=0.4)
    g_t = _goal_state(stakes=0.5, safety_risk=0.2)
    d_t = _derived_state(evidence_ambiguity=0.3, goal_conflict=0.1)

    a_star = engine.compute_target(h_t, g_t, d_t)

    assert a_star.motivational_priority == pytest.approx(0.68)
    assert a_star.decision_information_priority == pytest.approx(0.485)
    assert a_star.intervention_readiness == pytest.approx(0.3935)


def test_compute_target_clips_intervention_readiness_when_safety_dominates(engine):
    """a*_intv is the only term that can leave [0,1] on its own (safety_risk
    is its sole negative-weighted term). Drive it below 0 before clipping:
    a_mot=0, a_info=0, d_goal=0, h_int=0, safety_risk=1.0 ->
    raw a_intv = -0.35*1.0 = -.35, clipped to 0."""
    h_t = _human_appraisal(goal_relevance=0.0, uncertainty=0.0, affect_intensity=0.0)
    g_t = _goal_state(stakes=0.0, safety_risk=1.0)
    d_t = _derived_state(evidence_ambiguity=0.0, goal_conflict=0.0)

    a_star = engine.compute_target(h_t, g_t, d_t)

    assert a_star.motivational_priority == pytest.approx(0.0)
    assert a_star.decision_information_priority == pytest.approx(0.0)
    assert a_star.intervention_readiness == pytest.approx(0.0)


# ---------- apply(): A_1 = A*_1 on the first affect-enabled turn ----------


def test_apply_returns_target_unchanged_when_a_prev_is_none(engine):
    a_star = AgentTargetState(
        motivational_priority=0.68, decision_information_priority=0.485, intervention_readiness=0.3935
    )
    a_t = engine.apply(a_prev=None, a_star=a_star, rho=0.35, is_first_affect_enabled_turn=False)
    assert a_t.model_dump() == a_star.model_dump()


def test_apply_returns_target_unchanged_on_first_affect_enabled_turn_even_with_a_prev(engine):
    """is_first_affect_enabled_turn short-circuits the persistence formula
    even when a_prev is (unexpectedly) present — no arbitrary A_0 (§20.6)."""
    a_prev = AgentState(motivational_priority=0.9, decision_information_priority=0.9, intervention_readiness=0.9)
    a_star = AgentTargetState(
        motivational_priority=0.2, decision_information_priority=0.3, intervention_readiness=0.4
    )
    a_t = engine.apply(a_prev=a_prev, a_star=a_star, rho=0.35, is_first_affect_enabled_turn=True)
    assert a_t.model_dump() == a_star.model_dump()


# ---------- apply(): persistence formula, hand-calculated ----------


def test_apply_persistence_formula_hand_calculated(engine, transition_config):
    """a_prev=(.5,.4,.3), a_star=(.68,.485,.3935), rho=.35 (config's
    rho_dynamic):
    a_t.mot  = .35*.5  + .65*.68   = .617
    a_t.info = .35*.4  + .65*.485  = .45525
    a_t.intv = .35*.3  + .65*.3935 = .360775
    """
    rho = transition_config["rho_dynamic"]
    assert rho == pytest.approx(0.35)

    a_prev = AgentState(motivational_priority=0.5, decision_information_priority=0.4, intervention_readiness=0.3)
    a_star = AgentTargetState(
        motivational_priority=0.68, decision_information_priority=0.485, intervention_readiness=0.3935
    )

    a_t = engine.apply(a_prev=a_prev, a_star=a_star, rho=rho, is_first_affect_enabled_turn=False)

    assert a_t.motivational_priority == pytest.approx(0.617)
    assert a_t.decision_information_priority == pytest.approx(0.45525)
    assert a_t.intervention_readiness == pytest.approx(0.360775)


def test_apply_at_rho_zero_collapses_to_target_current_cue(engine):
    """CURRENT_CUE is LinearPersistenceTransition at rho=0, never a separate
    branch: the weighted average with rho=0 reduces to A_t = A*_t exactly,
    through the SAME code path as DYNAMIC's rho=.35 case above."""
    a_prev = AgentState(motivational_priority=0.9, decision_information_priority=0.1, intervention_readiness=0.5)
    a_star = AgentTargetState(
        motivational_priority=0.2, decision_information_priority=0.7, intervention_readiness=0.3
    )
    a_t = engine.apply(a_prev=a_prev, a_star=a_star, rho=0.0, is_first_affect_enabled_turn=False)
    assert a_t.model_dump() == a_star.model_dump()


def test_apply_persistence_isolated_to_agent_state(engine):
    """rho changes A_t only — it has no way to reach back into H_t/G_t/D_t/O_t:
    apply()'s signature takes only a_prev, a_star, rho and the first-turn
    flag, so there is nothing upstream it could mutate even in principle."""
    import inspect

    params = list(inspect.signature(engine.apply).parameters)
    assert params == ["a_prev", "a_star", "rho", "is_first_affect_enabled_turn"]


# ---------- explicit rho in [0,1] validation (ordinary defensive coding) ----------


@pytest.mark.parametrize("bad_rho", [-0.01, 1.01, 2.0, -1.0])
def test_apply_rejects_rho_outside_unit_interval(engine, bad_rho):
    a_prev = AgentState(motivational_priority=0.5, decision_information_priority=0.5, intervention_readiness=0.5)
    a_star = AgentTargetState(
        motivational_priority=0.5, decision_information_priority=0.5, intervention_readiness=0.5
    )
    with pytest.raises(ValueError):
        engine.apply(a_prev=a_prev, a_star=a_star, rho=bad_rho, is_first_affect_enabled_turn=False)


@pytest.mark.parametrize("boundary_rho", [0.0, 1.0])
def test_apply_accepts_rho_at_unit_interval_boundaries(engine, boundary_rho):
    a_prev = AgentState(motivational_priority=0.2, decision_information_priority=0.3, intervention_readiness=0.4)
    a_star = AgentTargetState(
        motivational_priority=0.8, decision_information_priority=0.7, intervention_readiness=0.6
    )
    a_t = engine.apply(a_prev=a_prev, a_star=a_star, rho=boundary_rho, is_first_affect_enabled_turn=False)
    assert 0.0 <= a_t.motivational_priority <= 1.0
