"""TurnRecord — §12.3, §20.13 of the frozen specification, §5.7 of the Implementation Blueprint."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.enums import Condition, EstimatorStatus, EvidenceStrength, GeneratorStatus
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.observation import Observation
from models.policy import PolicyState


class TurnRecord(BaseModel):
    """Every field from §20.13, plus record_id/comparison_id (implementation-
    review fix: (run_id, turn_id) is NOT unique once compare() writes one
    record per condition for a single turn_id) and the two state-relevance
    flags nested inside PolicyState rather than duplicated at the top level."""

    record_id: str                          # globally unique (uuid4) — the real primary key
    comparison_id: str | None = None        # shared by every record from the same compare() call;
                                             # None for a plain run_turn() record

    run_id: str
    turn_id: int = Field(ge=1)
    timestamp: datetime
    condition: Condition

    model_id: str
    model_version: str | None = None
    config_hash: str

    outcome_baseline_value_priorities: dict[str, float]   # copied from OutcomeBaseline, read-only

    observation: Observation
    appraisal: HumanAppraisal | None        # always populated, TASK_FOCUSED included (§6.7 fix);
                                             # None only on an unrecoverable estimator error
    evidence_strength: EvidenceStrength | None = None
    c_t: float | None = Field(default=None, ge=0, le=1)

    goal_state: GoalState
    derived_state: DerivedInteractionState

    a_prev: AgentState | None = None
    a_star: AgentTargetState | None = None
    a_t: AgentState | None = None
    rho: float = Field(ge=0, le=1)
    state_delta: dict[str, float] | None = None

    policy: PolicyState                     # carries both state-relevance flags
    response_text: str

    estimator_status: EstimatorStatus
    generator_status: GeneratorStatus
    estimator_latency_ms: float = Field(ge=0)
    generator_latency_ms: float = Field(ge=0)

    interventions: list[dict[str, Any]] = Field(default_factory=list)
    observed_choice: str | None = None
    observed_confidence: float | None = Field(default=None, ge=0, le=1)
    error_messages: list[str] = Field(default_factory=list)
    replay_parent_turn_id: int | None = None
