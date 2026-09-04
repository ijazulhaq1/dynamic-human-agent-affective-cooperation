"""H_t — §7, §7.2, §20.2 of the frozen specification, §5.2 of the Implementation Blueprint."""

from pydantic import BaseModel, Field

from models.enums import EvidenceStrength


class HumanAppraisal(BaseModel):
    """H_t.

    c_t is intentionally NOT a field on this class — it is derived
    (confidence_map[evidence_strength]) and passed as a sibling value beside
    H_t, never stored inside it (Appendix A; H_t and c_t are parallel outputs
    of one estimation process, §19.1).
    """

    goal_relevance: float = Field(ge=0, le=1)      # h_rel — causal, feeds a*_mot (§20.6)
    goal_congruence: float = Field(ge=-1, le=1)     # h_cong — non-causal descriptor (§7.2); logged only
    uncertainty: float = Field(ge=0, le=1)          # h_unc — causal, feeds a*_info and R_CLARIFY
    perceived_control: float = Field(ge=0, le=1)    # h_ctrl — policy modifier only (R_LOW_CTRL); never in F
    agency: float = Field(ge=0, le=1)               # h_agency — non-causal descriptor (§7.2); logged only
    affect_intensity: float = Field(ge=0, le=1)     # h_int — causal, feeds a*_intv and R_ACK/R_REDIRECT
    possible_affect: str | None = None              # secondary, non-causal; hedged by c_t before user-facing use
    evidence_tags: list[str] = Field(default_factory=list)   # observable cues only, no hidden chain-of-thought
    evidence_strength: EvidenceStrength             # source for the deterministic c_t mapping
