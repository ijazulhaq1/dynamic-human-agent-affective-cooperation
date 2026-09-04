"""Shared construction helpers for test_policy.py, test_policy_task_focused.py
and test_state_relevance.py — all three exercise the same PolicyEngine, so a
single set of builders (rather than three drifting copies) keeps their
default fixture values in sync. Not itself a test module (no test_ prefix).
"""

from pathlib import Path

import yaml

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.enums import EvidenceStrength
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from services.policy_engine import PolicyConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
POLICY_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "policy_rules.yaml"


def load_default_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_policy_rules_yaml() -> dict:
    with open(POLICY_RULES_PATH) as f:
        return yaml.safe_load(f)


def policy_config() -> PolicyConfig:
    return PolicyConfig.from_mapping(load_default_config())


def human_appraisal(**overrides) -> HumanAppraisal:
    defaults = dict(
        goal_relevance=0.3,
        goal_congruence=0.0,
        uncertainty=0.2,
        perceived_control=0.7,
        agency=0.5,
        affect_intensity=0.2,
        evidence_strength=EvidenceStrength.WEAK_INDIRECT,
    )
    defaults.update(overrides)
    return HumanAppraisal(**defaults)


def goal_state(**overrides) -> GoalState:
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


def derived_state(**overrides) -> DerivedInteractionState:
    defaults = dict(
        goal_conflict=0.1,
        evidence_ambiguity=0.1,
        goal_conflict_source="task_rule",
        ambiguity_source="task_rule",
    )
    defaults.update(overrides)
    return DerivedInteractionState(**defaults)


def agent_state(**overrides) -> AgentState:
    defaults = dict(motivational_priority=0.3, decision_information_priority=0.3, intervention_readiness=0.3)
    defaults.update(overrides)
    return AgentState(**defaults)


def agent_target_state(**overrides) -> AgentTargetState:
    defaults = dict(motivational_priority=0.3, decision_information_priority=0.3, intervention_readiness=0.3)
    defaults.update(overrides)
    return AgentTargetState(**defaults)
