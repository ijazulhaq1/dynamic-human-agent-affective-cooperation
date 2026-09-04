"""P_t — §20.2, §20.7 of the frozen specification, §5.6 of the Implementation Blueprint."""

from pydantic import BaseModel, Field

from models.enums import Policy, RationaleCode


class PolicyState(BaseModel):
    """P_t.

    For every TASK_FOCUSED turn both state-relevance flags are hard-set to
    False at construction (§20.7, §20.17) — PolicyEngineTaskFocused (a later
    phase) never computes a counterfactual, it simply returns False, False.
    """

    primary: Policy
    secondary: Policy | None = None
    triggered_rules: list[str] = Field(default_factory=list)     # rule IDs only, e.g. ["R_REDIRECT"]
    hard_constraints: list[str] = Field(default_factory=list)    # e.g. ["AUTONOMY_HIGH", "NO_UNDUE_INFLUENCE"]
    rationale_code: RationaleCode
    policy_version: str = "v1.5"
    state_could_influence_policy: bool     # §20.7 instrumentation
    state_did_influence_policy: bool       # §20.7 counterfactual check
