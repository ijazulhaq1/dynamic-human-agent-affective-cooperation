"""A*_t / A_t — §9, §20.2 of the frozen specification, §5.5 of the Implementation Blueprint."""

from pydantic import BaseModel, Field


class AgentTargetState(BaseModel):
    """A*_t — the target state a transition model computes each turn, before persistence."""

    motivational_priority: float = Field(ge=0, le=1)             # a_mot
    decision_information_priority: float = Field(ge=0, le=1)     # a_info
    intervention_readiness: float = Field(ge=0, le=1)            # a_intv


class AgentState(AgentTargetState):
    """A_t — same three fields as A*_t, after the persistence transition is applied."""

    pass


# rho is NEVER a field on either class — it lives only in transition config
# (config/default.yaml's transition.rho_dynamic), per Appendix A: "rho ...
# stored in transition/config, not inside A_t".
