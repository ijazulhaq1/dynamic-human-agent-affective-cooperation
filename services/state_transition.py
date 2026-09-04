"""M5 — StateTransitionEngine. Blueprint §6.4, spec §9.1/§20.6.

A small Protocol plus the frozen default implementation, matching the
"alternative-dynamics hook" requirement (§20.17): later decay/reset/
hysteresis models are new classes implementing the same interface, never
edits to PolicyEngine or Pipeline.

rho is never a field on AgentTargetState/AgentState (models/agent_state.py)
— it is supplied by the caller (ExperimentController.resolve_rho, a later
phase) on every call to apply(). CURRENT_CUE is this same class at rho=0,
never a separate branch: the "is_first_affect_enabled_turn or a_prev is
None" guard and the weighted-average line both apply uniformly regardless
of condition.
"""

from __future__ import annotations

from typing import Protocol

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.goal_state import GoalState
from models.human_state import HumanAppraisal


def clip01(value: float) -> float:
    """Clip to the closed [0,1] interval. Only a*_intv's formula can leave
    [0,1] on its own (its safety_risk term is the sole negative-weighted
    term, §20.6) — a*_mot and a*_info are convex combinations of already-
    bounded [0,1] inputs and cannot exceed the interval, but clip01 is
    applied to all three uniformly rather than relying on that being true
    forever as the weights evolve."""
    return max(0.0, min(1.0, value))


class TransitionModel(Protocol):
    def compute_target(
        self, h_t: HumanAppraisal, g_t: GoalState, d_t: DerivedInteractionState
    ) -> AgentTargetState: ...

    def apply(
        self,
        a_prev: AgentState | None,
        a_star: AgentTargetState,
        rho: float,
        is_first_affect_enabled_turn: bool,
    ) -> AgentState: ...


class LinearPersistenceTransition:
    """Default v1 model, §20.6. Weights are injected (read from
    config/default.yaml's transition.weights, §20.10) rather than hardcoded
    here — the same single-source-of-truth discipline AppraisalEstimator
    applies to confidence_map, so a weight can never drift between the
    config file and this class's formulas.

    All three target equations are self-bounded except a*_intv (its
    safety_risk term is the only negative-weighted term), so clip01() is
    load-bearing only there in principle, though it is applied uniformly.
    """

    def __init__(self, weights: dict[str, dict[str, float]]) -> None:
        self._w = weights

    def compute_target(
        self, h_t: HumanAppraisal, g_t: GoalState, d_t: DerivedInteractionState
    ) -> AgentTargetState:
        w_mot, w_info, w_intv = self._w["a_mot"], self._w["a_info"], self._w["a_intv"]

        a_mot = clip01(
            w_mot["h_rel"] * h_t.goal_relevance + w_mot["stakes"] * g_t.stakes
        )
        a_info = clip01(
            w_info["h_unc"] * h_t.uncertainty
            + w_info["d_amb"] * d_t.evidence_ambiguity
            + w_info["stakes"] * g_t.stakes
        )
        a_intv = clip01(
            w_intv["a_mot"] * a_mot
            + w_intv["a_info"] * a_info
            + w_intv["d_goal"] * d_t.goal_conflict
            + w_intv["h_int"] * h_t.affect_intensity
            + w_intv["safety_risk"] * g_t.safety_risk
        )
        return AgentTargetState(
            motivational_priority=a_mot,
            decision_information_priority=a_info,
            intervention_readiness=a_intv,
        )

    def apply(
        self,
        a_prev: AgentState | None,
        a_star: AgentTargetState,
        rho: float,
        is_first_affect_enabled_turn: bool,
    ) -> AgentState:
        if not 0.0 <= rho <= 1.0:
            # Ordinary defensive coding validation (not a scientific/architectural
            # decision — cf. the equivalent Pydantic 0<=rho<=1 check on TurnRecord).
            # rho is a plain float parameter here, not a pydantic field, so this
            # runtime check is what stands in for that guarantee until
            # ExperimentController.resolve_rho (Phase 5) is the sole caller.
            raise ValueError(f"rho must be in [0,1], got {rho}")
        if is_first_affect_enabled_turn or a_prev is None:
            return AgentState(**a_star.model_dump())  # A_1 = A*_1, no arbitrary A_0 (§20.6)
        return AgentState(
            motivational_priority=rho * a_prev.motivational_priority
            + (1 - rho) * a_star.motivational_priority,
            decision_information_priority=rho * a_prev.decision_information_priority
            + (1 - rho) * a_star.decision_information_priority,
            intervention_readiness=rho * a_prev.intervention_readiness
            + (1 - rho) * a_star.intervention_readiness,
        )
        # rho=0 (CURRENT_CUE) collapses this to A_t = A*_t through the SAME code
        # path — CURRENT_CUE is this class at rho=0, never a separate branch.
