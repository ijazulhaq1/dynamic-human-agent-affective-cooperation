"""tests/test_state_relevance.py — state_could/did_influence_policy, blueprint
§6.5/§20.7. Phase 2 gate, together with test_policy.py and
test_policy_task_focused.py.
"""

import pytest

from models.enums import Condition, Policy
from services.policy_engine import PolicyEngine, build_affect_rules
from services.policy_engine_task_focused import build_task_focused_rules
from tests.policy_fixtures import agent_state, agent_target_state, derived_state, goal_state, human_appraisal, policy_config


@pytest.fixture(scope="module")
def cfg():
    return policy_config()


@pytest.fixture()
def engine(cfg):
    return PolicyEngine(config=cfg, affect_rules=build_affect_rules(cfg), tf_rules=build_task_focused_rules(cfg))


def test_task_focused_flags_always_false(engine):
    """Both flags hard-set to False for every TASK_FOCUSED turn (§20.7,
    §20.17) regardless of what would happen if this same G_t/D_t were run
    through the affect-enabled table."""
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.state_could_influence_policy is False
    assert p_t.state_did_influence_policy is False


def test_could_is_false_when_safety_gate_fires_first(engine):
    """R_SAFETY (reads_a_t=False) fires before evaluation ever reaches
    R_REDIRECT -> could=False, and did stays False even with a_star present
    (no counterfactual is computed when could=False)."""
    h_t = human_appraisal()
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state()
    p_t = engine.select(
        Condition.DYNAMIC, h_t, c_t=0.8, g_t=g_t, d_t=d_t,
        a_t=agent_state(), a_star=agent_target_state(intervention_readiness=1.0, decision_information_priority=1.0),
    )
    assert p_t.state_could_influence_policy is False
    assert p_t.state_did_influence_policy is False


def test_could_is_true_once_evaluation_reaches_redirect(engine):
    """safety_risk/c_t/perceived_control all clear their own gates ->
    evaluation reaches R_REDIRECT (reads_a_t=True) -> could=True, regardless
    of whether R_REDIRECT's own condition ends up holding."""
    h_t = human_appraisal(affect_intensity=0.3, perceived_control=0.6)  # h_int below redirect gate on purpose
    g_t = goal_state(safety_risk=0.2)
    d_t = derived_state()
    p_t = engine.select(
        Condition.DYNAMIC, h_t, c_t=0.65, g_t=g_t, d_t=d_t,
        a_t=agent_state(intervention_readiness=0.1, decision_information_priority=0.1), a_star=None,
    )
    assert p_t.state_could_influence_policy is True
    assert p_t.primary != Policy.REDIRECT   # h_int too low for REDIRECT itself to fire


def test_did_is_false_when_a_star_is_none_even_if_could_is_true(engine):
    """could=True alone is not enough to compute did — without a_star there
    is nothing to run the counterfactual against, so did stays False."""
    h_t = human_appraisal(affect_intensity=0.3, perceived_control=0.6)
    g_t = goal_state(safety_risk=0.2)
    d_t = derived_state()
    p_t = engine.select(
        Condition.DYNAMIC, h_t, c_t=0.65, g_t=g_t, d_t=d_t,
        a_t=agent_state(intervention_readiness=0.1, decision_information_priority=0.1), a_star=None,
    )
    assert p_t.state_could_influence_policy is True
    assert p_t.state_did_influence_policy is False


def test_did_is_false_when_counterfactual_agrees_with_actual(engine):
    """could=True, a_star given, but neither A_t nor A*_t is enough to flip
    the outcome away from R_ACK (h_int is below R_REDIRECT's gate for both,
    since h_t is shared and unaffected by the a_t <-> a_star swap) ->
    did=False."""
    h_t = human_appraisal(affect_intensity=0.3, perceived_control=0.6, goal_relevance=0.3)
    g_t = goal_state(safety_risk=0.2)
    d_t = derived_state()
    a_t = agent_state(intervention_readiness=0.1, decision_information_priority=0.1)
    a_star = agent_target_state(intervention_readiness=0.5, decision_information_priority=0.3)
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.65, g_t=g_t, d_t=d_t, a_t=a_t, a_star=a_star)
    assert p_t.state_could_influence_policy is True
    assert p_t.state_did_influence_policy is False


def test_could_and_did_true_when_counterfactual_flips_the_outcome(engine):
    """could=True, and swapping in A*_t (vs. the actual A_t) changes which
    rule fires: actual A_t is too weak for R_REDIRECT and falls through to
    R_ACK; A*_t clears every R_REDIRECT gate -> did=True."""
    h_t = human_appraisal(affect_intensity=0.8, perceived_control=0.6, goal_relevance=0.7)
    g_t = goal_state(safety_risk=0.2)
    d_t = derived_state()
    a_t = agent_state(intervention_readiness=0.3, decision_information_priority=0.3)
    a_star = agent_target_state(intervention_readiness=0.8, decision_information_priority=0.7)

    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.65, g_t=g_t, d_t=d_t, a_t=a_t, a_star=a_star)

    assert p_t.state_could_influence_policy is True
    assert p_t.state_did_influence_policy is True
    # the ACTUAL (non-counterfactual) result is what gets returned, not the counterfactual:
    assert p_t.primary == Policy.ACKNOWLEDGE


def test_counterfactual_probe_does_not_mutate_inputs_or_leak_into_result(engine):
    """The counterfactual context is a fresh dataclasses.replace() copy
    (services/policy_engine.py) — running it must not mutate a_t/a_star, and
    a repeat call with the same inputs must be deterministic (no hidden
    state carried between calls)."""
    h_t = human_appraisal(affect_intensity=0.8, perceived_control=0.6, goal_relevance=0.7)
    g_t = goal_state(safety_risk=0.2)
    d_t = derived_state()
    a_t = agent_state(intervention_readiness=0.3, decision_information_priority=0.3)
    a_star = agent_target_state(intervention_readiness=0.8, decision_information_priority=0.7)
    a_t_before, a_star_before = a_t.model_dump(), a_star.model_dump()

    p_first = engine.select(Condition.DYNAMIC, h_t, c_t=0.65, g_t=g_t, d_t=d_t, a_t=a_t, a_star=a_star)
    p_second = engine.select(Condition.DYNAMIC, h_t, c_t=0.65, g_t=g_t, d_t=d_t, a_t=a_t, a_star=a_star)

    assert a_t.model_dump() == a_t_before
    assert a_star.model_dump() == a_star_before
    assert p_first.model_dump() == p_second.model_dump()
    # the returned PolicyState reflects the ACTUAL a_t, never the counterfactual a_star:
    assert p_first.primary == Policy.ACKNOWLEDGE
