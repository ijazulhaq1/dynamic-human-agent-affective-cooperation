"""tests/test_policy_task_focused.py — TASK_FOCUSED reduced rule table,
blueprint §6.5/§20.11.1. Phase 2 gate, together with test_policy.py and
test_state_relevance.py.
"""

import pytest

from models.enums import Condition, Policy, RationaleCode
from services.policy_engine import PolicyEngine, build_affect_rules
from services.policy_engine_task_focused import build_task_focused_rules
from tests.policy_fixtures import (
    agent_state,
    agent_target_state,
    derived_state,
    goal_state,
    human_appraisal,
    load_policy_rules_yaml,
    policy_config,
)


@pytest.fixture(scope="module")
def cfg():
    return policy_config()


@pytest.fixture()
def engine(cfg):
    return PolicyEngine(config=cfg, affect_rules=build_affect_rules(cfg), tf_rules=build_task_focused_rules(cfg))


def test_task_focused_safety_gate(engine):
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state()
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.DEFER
    assert p_t.rationale_code == RationaleCode.SAFETY_OVERRIDE
    assert p_t.triggered_rules == ["TF_SAFETY"]


def test_task_focused_clarify_on_high_ambiguity(engine):
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state(evidence_ambiguity=0.9, goal_conflict=0.1)
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.CLARIFY
    assert p_t.rationale_code == RationaleCode.CLARIFY_NEEDED
    assert p_t.triggered_rules == ["TF_TASK_CLARIFY"]


def test_task_focused_clarify_on_high_goal_conflict(engine):
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state(evidence_ambiguity=0.1, goal_conflict=0.9)
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.CLARIFY
    assert p_t.triggered_rules == ["TF_TASK_CLARIFY"]


def test_task_focused_informs_when_task_requires_evidence(engine):
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state(evidence_ambiguity=0.1, goal_conflict=0.1, ambiguity_source="required_evidence_formula")
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.INFORM
    assert p_t.rationale_code == RationaleCode.INFORM_NEEDED
    assert p_t.triggered_rules == ["TF_INFORM"]


def test_no_undue_influence_always_attached_for_task_focused(engine):
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert "NO_UNDUE_INFLUENCE" in p_t.hard_constraints


def test_task_focused_safety_secondary_requires_real_evidence_structure(engine):
    """Same implementation-review fix as R_SAFETY: low evidence_ambiguity
    alone (with no required_evidence structure represented) must not read as
    "evidence exists" for TF_SAFETY's secondary either."""
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state(evidence_ambiguity=0.0, ambiguity_source="task_rule")
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.DEFER
    assert p_t.secondary == Policy.CLARIFY


def test_task_focused_default_rule(engine):
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state(evidence_ambiguity=0.1, goal_conflict=0.1)
    p_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.INFORM
    assert p_t.rationale_code == RationaleCode.MINIMAL_SUPPORT
    assert p_t.triggered_rules == ["TF_DEFAULT"]


# ---------- TASK_FOCUSED is reproducible from G_t + permitted D_t only ----------


def test_task_focused_ignores_h_t_c_t_a_t(engine):
    """Same G_t/D_t, wildly different H_t/c_t/A_t/A*_t (including None vs.
    populated) -> identical PolicyState. This is checkable at the
    PolicyEngine layer regardless of whether the caller (Pipeline, a later
    phase) happened to compute H_t/c_t for logging: select() never reads
    them on the TASK_FOCUSED branch, by construction of its own Ctx."""
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state(evidence_ambiguity=0.1, goal_conflict=0.1)

    p_minimal = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    p_with_everything = engine.select(
        Condition.TASK_FOCUSED,
        h_t=human_appraisal(affect_intensity=1.0, uncertainty=1.0, perceived_control=0.0),
        c_t=0.05,
        g_t=g_t,
        d_t=d_t,
        a_t=agent_state(intervention_readiness=1.0, decision_information_priority=1.0),
        a_star=agent_target_state(intervention_readiness=1.0, decision_information_priority=1.0),
    )
    assert p_minimal.model_dump() == p_with_everything.model_dump()


def test_task_focused_computes_but_does_not_read_h_t(engine):
    """Equivalent framing of the same guarantee, named to match the
    blueprint's §9 test map row: a caller that DID compute H_t/c_t (e.g. for
    dashboard/TurnRecord logging) gets exactly the same policy as one that
    didn't — proof PolicyEngineTaskFocused never reads what was computed."""
    g_t = goal_state(safety_risk=0.9)   # TF_SAFETY case, distinct from the default-rule case above
    d_t = derived_state()
    computed_h_t = human_appraisal(affect_intensity=0.9, uncertainty=0.9)
    p_without_h_t = engine.select(Condition.TASK_FOCUSED, h_t=None, c_t=None, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    p_with_computed_h_t = engine.select(
        Condition.TASK_FOCUSED, h_t=computed_h_t, c_t=0.85, g_t=g_t, d_t=d_t, a_t=None, a_star=None
    )
    assert p_without_h_t.model_dump() == p_with_computed_h_t.model_dump()


def test_task_focused_rule_table_matches_policy_rules_yaml(cfg):
    yaml_rows = load_policy_rules_yaml()["task_focused"]
    rules = build_task_focused_rules(cfg)
    assert [row["id"] for row in yaml_rows] == [rule.id for rule in rules]
