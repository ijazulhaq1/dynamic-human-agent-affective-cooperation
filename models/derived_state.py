"""D_t — §8.2, §20.2 of the frozen specification, §5.4 of the Implementation Blueprint."""

from pydantic import BaseModel, Field


class DerivedInteractionState(BaseModel):
    """D_t. Derived from O_t and G_t only — NEVER reads H_t (checkable
    structurally in services.derived_features.derive_interaction_state, a
    later phase, whose signature has no h_t parameter at all)."""

    goal_conflict: float = Field(ge=0, le=1)         # d_goal
    evidence_ambiguity: float = Field(ge=0, le=1)    # d_amb
    goal_conflict_source: str                         # "structured_formula" | "task_rule" | "researcher_config"
    ambiguity_source: str                              # "required_evidence_formula" | "task_rule" | "researcher_config"
