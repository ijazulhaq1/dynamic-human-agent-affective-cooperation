"""M7 — ResponseGenerator. Blueprint §6.6, spec §20.9.

generate() is a literal transcription of the blueprint's own pseudocode.
§20.9's causal-isolation rule: A_t -> P_t -> R_t, never A_t -> R_t directly.
Numeric A_t values are never passed here — this method's own signature has
no a_t/a_star parameter at all, and Pipeline (a later phase) does not pass
them at either of its two call sites (§6.7) — "the caller does not even
have access to raw A_t at this call site by construction" (blueprint §6.6
docstring). The only channel by which A_t's influence can reach R_t is the
qualitative, non-numeric cue in QUALITATIVE_CUES, and only when
p_t.state_did_influence_policy is True.
"""

from __future__ import annotations

from typing import Any

from llm.adapter import GenerationContract, GeneratorError, LLMAdapter
from llm.fallback import FallbackTemplates
from models.enums import GeneratorStatus, RationaleCode
from models.goal_state import GoalState
from models.observation import Observation, Turn
from models.policy import PolicyState

# Judgment call (documented gap, flagged for review): the blueprint shows
# only one fragmentary example ("engagement has grown across the
# conversation") — no dict is ever given. Short, non-numeric, natural-
# language descriptions keyed by RationaleCode, so a "bounded A_t summary"
# can never regress into leaking a_mot/a_info/a_intv floats into the prompt
# (§6.6 note). Structurally, only REDIRECT_ELIGIBLE, CLARIFY_NEEDED,
# INFORM_NEEDED, ACKNOWLEDGE_ELIGIBLE and MINIMAL_SUPPORT can ever appear on
# a PolicyState with state_did_influence_policy=True: could_influence can
# only become True once rule evaluation passes R_SAFETY/R_LOW_CONF/
# R_LOW_CTRL without firing (services/policy_engine.py's
# _reached_a_t_sensitive_rule) — i.e. reaches R_REDIRECT — so
# SAFETY_OVERRIDE/LOW_CONFIDENCE/LOW_CONTROL (the rationale codes for the
# three rules that fire BEFORE that point) can never co-occur with did=True.
# All eight RationaleCode members get an entry anyway, defensively, so a
# KeyError here is impossible regardless of how could/did are computed
# upstream. This exact wording is authored for this prototype, not
# transcribed from any spec — a researcher should review/replace it before
# a real study.
QUALITATIVE_CUES: dict[RationaleCode, str] = {
    RationaleCode.SAFETY_OVERRIDE: "a safety concern has been prioritized in this response",
    RationaleCode.LOW_CONFIDENCE: "there wasn't enough confidence in reading the situation to act on it further",
    RationaleCode.LOW_CONTROL: "your sense of control over this decision seems limited right now",
    RationaleCode.REDIRECT_ELIGIBLE: (
        "engagement and readiness for a more active suggestion have grown across the conversation"
    ),
    RationaleCode.CLARIFY_NEEDED: "more clarity seems needed before going further",
    RationaleCode.INFORM_NEEDED: "some additional information seems useful at this point",
    RationaleCode.ACKNOWLEDGE_ELIGIBLE: "your engagement with this has been acknowledged",
    RationaleCode.MINIMAL_SUPPORT: "a light touch seemed appropriate here",
}

# Judgment call (documented gap, flagged for review): _relevant/
# _relevant_goal_fields are named only at their call site in the blueprint
# (§6.6) — no body or field-selection rule is ever shown. The filter below
# is deliberately narrow: it strips only the keys that are
# services.derived_features's OWN computation inputs (structured-formula
# overrides and the raw required_evidence list) — data meant for D_t's
# arithmetic, not natural-language content a response might need — and
# passes everything else in task_context through unchanged (options,
# focal_option_id, task_event, event_flags, any scenario-specific key).
_DERIVED_FEATURE_INTERNAL_KEYS = frozenset({"d_goal_override", "d_amb_override", "required_evidence"})


def _relevant(task_context: dict[str, Any], g_t: GoalState) -> dict[str, Any]:
    """g_t is accepted to match the blueprint's exact call-site signature
    (_relevant(o_t.task_context, g_t)) but this implementation's filter does
    not vary by its content — no field-selection spec ties task_context
    relevance to GoalState content anywhere in the blueprint. Threaded
    through unused rather than dropped, so a later phase/researcher can
    extend this (e.g. restricting task_context further to keys touching
    g_t.value_priorities) without changing the call site."""
    return {key: value for key, value in task_context.items() if key not in _DERIVED_FEATURE_INTERNAL_KEYS}


def _relevant_goal_fields(g_t: GoalState) -> dict[str, Any]:
    """Same judgment call as _relevant: goal_version/update_source are
    provenance bookkeeping (§5.3), not content a generation prompt needs;
    no_undue_influence is a structural constant already carried separately
    as a hard_constraints flag (never varying, always True — see
    models/goal_state.py), not free-form goal content. Everything else
    passes through."""
    return {
        "objective": g_t.objective,
        "value_priorities": dict(g_t.value_priorities),
        "stakes": g_t.stakes,
        "task_constraints": g_t.task_constraints,
        "autonomy_weight": g_t.autonomy_weight,
        "safety_risk": g_t.safety_risk,
    }


class ResponseGenerator:
    def __init__(self, llm_adapter: LLMAdapter, fallback: FallbackTemplates):
        self._llm, self._fallback = llm_adapter, fallback

    def generate(
        self, o_t: Observation, g_t: GoalState, raw_history: tuple[Turn, ...], p_t: PolicyState
    ) -> tuple[str, GeneratorStatus]:
        """§20.9 causal-isolation rule: A_t → P_t → R_t, never A_t → R_t directly.
        Numeric A_t values are NEVER passed here — the caller (Pipeline) does not even
        have access to raw A_t at this call site by construction."""
        state_cue = None
        if p_t.state_did_influence_policy:
            state_cue = QUALITATIVE_CUES[p_t.rationale_code]
        contract = GenerationContract(
            user_text=o_t.user_text, task_context=_relevant(o_t.task_context, g_t),
            raw_history=raw_history, goal_context=_relevant_goal_fields(g_t),
            policy=p_t.primary, secondary_policy=p_t.secondary,
            hard_constraints=p_t.hard_constraints, qualitative_state_cue=state_cue,
        )
        try:
            return self._llm.generate(contract), GeneratorStatus.OK
        except (TimeoutError, GeneratorError):
            return self._fallback.render(p_t, o_t.task_context), GeneratorStatus.FALLBACK_TEMPLATE
