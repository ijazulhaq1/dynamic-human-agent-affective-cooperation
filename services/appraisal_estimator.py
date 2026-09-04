"""M2 — AppraisalEstimator. Blueprint §6.2, spec §7.1/§20.3.

estimate() is a literal transcription of the blueprint's own pseudocode: no
try/except around the LLM call, only a parsed-is-None branch, because
LLMAdapter implementations are contractually required never to raise (see
llm/adapter.py's module docstring) — a transport failure and malformed JSON
are the same "None" outcome from this class's point of view.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from llm.adapter import LLMAdapter
from models.enums import EstimatorStatus, EvidenceStrength
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.observation import Observation


def confidence_map_from_config(config: dict) -> dict[EvidenceStrength, float]:
    """config is the full parsed config/default.yaml (or an equivalent
    mapping) — reads confidence_map verbatim, never a second hardcoded copy
    of these five numbers."""
    return {EvidenceStrength(key): value for key, value in config["confidence_map"].items()}


class AppraisalEstimator:
    FALLBACK_APPRAISAL = HumanAppraisal(
        goal_relevance=0.0,
        goal_congruence=0.0,
        uncertainty=1.0,
        perceived_control=0.5,
        agency=0.5,
        affect_intensity=0.0,
        possible_affect=None,
        evidence_tags=["ESTIMATOR_FAILURE"],
        evidence_strength=EvidenceStrength.INSUFFICIENT,
    )
    # Exact, deterministic fallback vector (implementation-review fix — was previously
    # an underspecified 'uncertainty_safe_fallback()'). It assigns zero to the UNSUPPORTED
    # relevance and affect-intensity signals (h_rel, h_int) so a failed extraction never
    # asserts unearned motivational/intervention pressure; MAXIMAL uncertainty (h_unc=1.0)
    # to represent the estimator failure itself, which deliberately RAISES a*_info/R_CLARIFY
    # pressure rather than lowering it; and neutral 0.5 for h_ctrl/h_agency rather than a
    # guessed value. So this does not uniformly pull A*_t toward its floor — it avoids
    # inventing affective evidence while still increasing uncertainty-driven clarification
    # pressure. A class-level constant, never recomputed, identical across every failure
    # and every test run.

    def __init__(self, llm_adapter: LLMAdapter, confidence_map: dict[EvidenceStrength, float]):
        self._llm = llm_adapter
        self._confidence_map = confidence_map

    def estimate(
        self, o_t: Observation, g_t: GoalState
    ) -> tuple[HumanAppraisal, EvidenceStrength, float, EstimatorStatus]:
        """§7.1 / §20.3. Returns (H_t, evidence_strength, c_t, status).
        c_t = self._confidence_map[evidence_strength] — NEVER a function of H_t's own
        field values (H_t and c_t are parallel outputs of one process, §19.1)."""
        raw = self._llm.extract_appraisal(o_t, g_t)          # structured JSON only, no free prose
        parsed = self._validate(raw)
        retried = False
        if parsed is None:
            retried = True
            raw = self._llm.extract_appraisal(o_t, g_t, repair=True)   # one retry (§20.3)
            parsed = self._validate(raw)
        if parsed is None:
            # .model_copy(deep=True), not FALLBACK_APPRAISAL itself (implementation-review
            # fix): HumanAppraisal is a plain mutable BaseModel, so returning the class-level
            # constant directly would hand every caller the SAME object — a later
            # `h_t.uncertainty = 0.2` on one turn's fallback would silently corrupt the
            # "exact, deterministic fallback vector" for every other turn/run that hit the
            # fallback path afterward. The blueprint requires the VALUES be an exact constant,
            # not that every call share one mutable instance.
            fallback = self.FALLBACK_APPRAISAL.model_copy(deep=True)
            return (fallback, EvidenceStrength.INSUFFICIENT,
                    self._confidence_map[EvidenceStrength.INSUFFICIENT],
                    EstimatorStatus.FALLBACK_LOW_CONFIDENCE)
        h_t, evidence_strength = parsed
        status = EstimatorStatus.RETRY_OK if retried else EstimatorStatus.OK
        return h_t, evidence_strength, self._confidence_map[evidence_strength], status

    def _validate(self, raw: dict[str, Any] | None) -> tuple[HumanAppraisal, EvidenceStrength] | None:
        """Not given a body in the blueprint's pseudocode — only its
        None-vs-(h_t, evidence_strength) usage is shown (§7.1/§20.3).
        Implemented here as a direct attempt to construct HumanAppraisal
        from raw's own field names (§5.2's H_t table — no second, differently
        -named JSON schema is invented). Anything that keeps this from
        producing a valid HumanAppraisal — a non-dict raw (including the
        None a well-behaved LLMAdapter returns for a transport failure),
        a missing field, a value outside its Pydantic range, or an
        evidence_strength string that isn't a real EvidenceStrength member —
        is treated identically: return None, which the caller reads as
        'try the repair retry, then fall back'."""
        if not isinstance(raw, dict):
            return None
        try:
            evidence_strength = EvidenceStrength(raw["evidence_strength"])
            h_t = HumanAppraisal(
                goal_relevance=raw["goal_relevance"],
                goal_congruence=raw["goal_congruence"],
                uncertainty=raw["uncertainty"],
                perceived_control=raw["perceived_control"],
                agency=raw["agency"],
                affect_intensity=raw["affect_intensity"],
                possible_affect=raw.get("possible_affect"),
                evidence_tags=raw.get("evidence_tags", []),
                evidence_strength=evidence_strength,
            )
        except (KeyError, TypeError, ValueError, ValidationError):
            return None
        return h_t, evidence_strength
