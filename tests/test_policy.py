"""tests/test_policy.py — M6 affect-enabled rule table, blueprint §6.5/§20.7.

Phase 2 gate (blueprint §9 test map, §10 Phase 2), together with
test_policy_task_focused.py and test_state_relevance.py.
"""

import pytest

from models.enums import Condition, Policy, RationaleCode
from services.policy_engine import PolicyEngine, build_affect_rules
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
    return PolicyEngine(config=cfg, affect_rules=build_affect_rules(cfg), tf_rules=[])


def test_safety_gate_always_wins(engine):
    """safety_risk >= .70 must win even when every other rule's condition
    would also independently fire (low c_t, low perceived_control) — R_SAFETY
    is priority 1."""
    h_t = human_appraisal(perceived_control=0.1)   # would also trigger R_LOW_CTRL
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state()
    p_t = engine.select(
        Condition.DYNAMIC, h_t, c_t=0.2, g_t=g_t, d_t=d_t,   # c_t=0.2 would also trigger R_LOW_CONF
        a_t=agent_state(), a_star=agent_target_state(),
    )
    assert p_t.primary == Policy.DEFER
    assert p_t.rationale_code == RationaleCode.SAFETY_OVERRIDE
    assert p_t.triggered_rules == ["R_SAFETY"]


def test_safety_gate_secondary_is_inform_when_required_evidence_is_resolved(engine):
    """Real required_evidence structure (ambiguity_source=
    "required_evidence_formula"), sufficiently resolved -> INFORM is
    allowed."""
    h_t = human_appraisal()
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state(evidence_ambiguity=0.25, ambiguity_source="required_evidence_formula")
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.8, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.DEFER
    assert p_t.secondary == Policy.INFORM


def test_safety_gate_secondary_is_clarify_when_required_evidence_is_unresolved(engine):
    h_t = human_appraisal()
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state(evidence_ambiguity=0.9, ambiguity_source="required_evidence_formula")
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.8, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.DEFER
    assert p_t.secondary == Policy.CLARIFY


def test_safety_gate_secondary_is_clarify_when_no_evidence_structure_represented(engine):
    """implementation-review fix: low evidence_ambiguity alone must NOT be
    read as "evidence exists". A D_t with no required_evidence at all
    (ambiguity_source="task_rule", the no-signal default, evidence_ambiguity
    left at 0.0) is the absence of the question being asked, not proof a
    safety concern was addressed — the secondary must fall back to CLARIFY,
    not silently become INFORM merely because d_amb happens to be low."""
    h_t = human_appraisal()
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state(evidence_ambiguity=0.0, ambiguity_source="task_rule")
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.8, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.DEFER
    assert p_t.secondary == Policy.CLARIFY


def test_safety_gate_secondary_is_clarify_when_evidence_ambiguous(engine):
    h_t = human_appraisal()
    g_t = goal_state(safety_risk=0.9)
    d_t = derived_state(evidence_ambiguity=0.9, ambiguity_source="required_evidence_formula")
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.8, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.DEFER
    assert p_t.secondary == Policy.CLARIFY


def test_low_confidence_gate_blocks_affect_assertions(engine):
    """c_t < .50 fires R_LOW_CONF even when perceived_control is also low
    (which would independently trigger R_LOW_CTRL, priority 3) — R_LOW_CONF
    is priority 2, safety_risk stays below its gate."""
    h_t = human_appraisal(perceived_control=0.1)
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.3, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.CLARIFY
    assert p_t.secondary == Policy.INFORM
    assert p_t.rationale_code == RationaleCode.LOW_CONFIDENCE
    assert p_t.triggered_rules == ["R_LOW_CONF"]


def test_low_control_allows_acknowledge_secondary(engine):
    """h_ctrl < .40 with c_t/safety_risk both clear of their own gates, and
    h_t independently R_ACK-eligible -> primary INFORM (unc/amb both low),
    secondary ACKNOWLEDGE ("if independently eligible")."""
    h_t = human_appraisal(
        perceived_control=0.2, uncertainty=0.1, affect_intensity=0.8, goal_relevance=0.8
    )
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state(evidence_ambiguity=0.1, goal_conflict=0.1)
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.INFORM
    assert p_t.secondary == Policy.ACKNOWLEDGE
    assert p_t.rationale_code == RationaleCode.LOW_CONTROL


def test_low_control_primary_is_clarify_when_uncertain(engine):
    h_t = human_appraisal(perceived_control=0.2, uncertainty=0.9)
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=None, a_star=None)
    assert p_t.primary == Policy.CLARIFY
    assert p_t.rationale_code == RationaleCode.LOW_CONTROL


def test_redirect_requires_all_five_conditions(engine):
    h_t = human_appraisal(affect_intensity=0.8, perceived_control=0.7)
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    a_t = agent_state(intervention_readiness=0.8, decision_information_priority=0.8)
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=a_t, a_star=None)
    assert p_t.primary == Policy.REDIRECT
    assert p_t.rationale_code == RationaleCode.REDIRECT_ELIGIBLE
    assert p_t.triggered_rules == ["R_REDIRECT"]


def test_redirect_does_not_fire_when_a_intv_below_gate(engine):
    """Same H_t/G_t/D_t/c_t as the passing REDIRECT fixture, but
    intervention_readiness just below its gate -> falls through instead."""
    h_t = human_appraisal(affect_intensity=0.8, perceived_control=0.7)
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    a_t = agent_state(intervention_readiness=0.1, decision_information_priority=0.1)
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=a_t, a_star=None)
    assert p_t.primary != Policy.REDIRECT


def test_default_rule_fires_when_nothing_else_matches(engine):
    h_t = human_appraisal()
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    a_t = agent_state()
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=a_t, a_star=None)
    assert p_t.primary == Policy.INFORM
    assert p_t.secondary is None
    assert p_t.rationale_code == RationaleCode.MINIMAL_SUPPORT
    assert p_t.triggered_rules == ["R_DEFAULT"]


# ---------- NO_UNDUE_INFLUENCE is a hard, unconditional safeguard ----------


def test_no_undue_influence_always_attached_regardless_of_autonomy(engine):
    """§20.7 preamble: NO_UNDUE_INFLUENCE is always attached, not data-
    dependent — checked across both a low- and a high-autonomy_weight
    GoalState (AUTONOMY_HIGH is the only conditional hard constraint;
    NO_UNDUE_INFLUENCE must appear either way)."""
    h_t = human_appraisal()
    d_t = derived_state()
    for autonomy_weight in (0.1, 0.95):
        g_t = goal_state(safety_risk=0.1, autonomy_weight=autonomy_weight)
        p_t = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=agent_state(), a_star=None)
        assert "NO_UNDUE_INFLUENCE" in p_t.hard_constraints


# ---------- output shape: exactly one primary, <= one compatible secondary ----------


@pytest.mark.parametrize(
    "h_kwargs,g_kwargs,d_kwargs,c_t,a_kwargs",
    [
        (dict(), dict(safety_risk=0.9), dict(), 0.8, dict()),
        (dict(), dict(safety_risk=0.1), dict(), 0.2, dict()),
        (dict(perceived_control=0.1), dict(safety_risk=0.1), dict(), 0.7, dict()),
        (
            dict(affect_intensity=0.8, perceived_control=0.7),
            dict(safety_risk=0.1),
            dict(),
            0.7,
            dict(intervention_readiness=0.8, decision_information_priority=0.8),
        ),
        (dict(uncertainty=0.9), dict(safety_risk=0.1), dict(), 0.7, dict()),
        (dict(), dict(safety_risk=0.1), dict(), 0.7, dict()),
    ],
)
def test_policy_output_shape(engine, h_kwargs, g_kwargs, d_kwargs, c_t, a_kwargs):
    h_t = human_appraisal(**h_kwargs)
    g_t = goal_state(**g_kwargs)
    d_t = derived_state(**d_kwargs)
    a_t = agent_state(**a_kwargs)
    p_t = engine.select(Condition.DYNAMIC, h_t, c_t=c_t, g_t=g_t, d_t=d_t, a_t=a_t, a_star=None)
    assert isinstance(p_t.primary, Policy)
    assert p_t.secondary is None or isinstance(p_t.secondary, Policy)
    if p_t.secondary is not None:
        assert p_t.secondary != p_t.primary
    assert len(p_t.triggered_rules) == 1


def test_current_cue_and_dynamic_conditions_select_identically(engine):
    """PolicyEngine.select() branches only on TASK_FOCUSED vs not — rho and
    the CURRENT_CUE/DYNAMIC distinction live entirely upstream (in how A_t
    was computed), so for the same resolved inputs both conditions must
    produce the same PolicyState."""
    h_t = human_appraisal(affect_intensity=0.8, perceived_control=0.7)
    g_t = goal_state(safety_risk=0.1)
    d_t = derived_state()
    a_t = agent_state(intervention_readiness=0.8, decision_information_priority=0.8)
    p_dynamic = engine.select(Condition.DYNAMIC, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=a_t, a_star=None)
    p_current_cue = engine.select(Condition.CURRENT_CUE, h_t, c_t=0.7, g_t=g_t, d_t=d_t, a_t=a_t, a_star=None)
    assert p_dynamic.model_dump() == p_current_cue.model_dump()


# ---------- config/policy_rules.yaml stays in sync with the executable Rule list ----------


def test_affect_rule_table_matches_policy_rules_yaml(cfg):
    """policy_rules.yaml documents the same priority-ordered rows as
    build_affect_rules() — this is the sync check its own header comment
    promises ('a later-phase test, test_policy.py, checks this file and the
    code stay in sync')."""
    yaml_rows = load_policy_rules_yaml()["affect_enabled"]
    rules = build_affect_rules(cfg)
    assert [row["id"] for row in yaml_rows] == [rule.id for rule in rules]
    assert [row["reads_a_t"] for row in yaml_rows] == [rule.reads_a_t for rule in rules]
