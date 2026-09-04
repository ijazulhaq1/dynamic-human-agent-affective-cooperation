"""llm/prompts.py — structured-extraction prompt template. Blueprint §10
Phase 7 repo layout: "llm/prompts.py # structured-extraction prompt
template" — named and scoped by that one line only; no prompt text, wording,
or format is given anywhere in the blueprint or the frozen specification.

Judgment call (documented gap, flagged for review): every prompt string in
this file is authored for this prototype, not transcribed from any spec —
a researcher should review/replace this wording before a real study, same
caveat this project has already attached to llm/fallback.py's
FallbackTemplates and services/response_generator.py's QUALITATIVE_CUES.

Two prompt pairs are built here, one per LLMAdapter method (llm/adapter.py):

- build_appraisal_extraction_prompt(o_t, g_t, repair=False) — for
  extract_appraisal(). The schema in APPRAISAL_SYSTEM_PROMPT is not
  invented: it is HumanAppraisal's own field table (§5.2), field-for-field,
  the same source services.appraisal_estimator.AppraisalEstimator._validate()
  already reads from a returned dict — so a real LLM response that matches
  THIS prompt's schema is exactly what that method already expects, no
  second schema to keep in sync.

- build_generation_prompt(contract) — for generate(). Deliberately cannot
  leak raw numeric A_t into a prompt even by accident: GenerationContract
  (llm/adapter.py) has no a_t/a_star field at all — see
  services/response_generator.py's own module docstring for why the causal-
  isolation rule (§20.9, A_t -> P_t -> R_t, never A_t -> R_t) is already
  structural by the time a contract reaches here. This module only has to
  not name any AgentState/AgentTargetState attribute in its own prompt text
  either (verified by tests/test_llm_prompts.py, which greps the rendered
  prompt for those exact attribute-name strings).

Both functions return (system_prompt, user_prompt) — a plain tuple, not
a class, since AnthropicLLMAdapter (llm/adapter.py) is this project's one
real caller and both of its calls (messages.create(system=..., messages=
[{"role": "user", "content": ...}])) want exactly that split.

Post-delivery fix round (user review of the first Phase 7 delivery, smaller
issues #4 and #5):
- #4: build_generation_prompt() no longer forwards contract.hard_constraints
  codes (e.g. "NO_UNDUE_INFLUENCE") verbatim into the prompt sent to the
  real Anthropic model — see _HARD_CONSTRAINT_INSTRUCTIONS and
  _translate_hard_constraints() below, mirroring the same fix
  llm/fallback.py's own docstring already documents for the offline
  template path.
- #5: build_appraisal_extraction_prompt()'s user prompt no longer describes
  value priorities as "pre-elicited before this conversation" — worded now
  as "current explicit value priorities" instead, since a session's
  GoalState can be updated mid-run via SessionState.apply_explicit_goal_
  update() (services/state_manager.py), at which point "pre-elicited"
  would describe the value priorities as of session start rather than what
  is actually being passed to the model on this call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from models.enums import EvidenceStrength
from models.goal_state import GoalState
from models.observation import Observation, Turn

if TYPE_CHECKING:
    # Deferred import (TYPE_CHECKING-only): llm/adapter.py imports THIS
    # module (to build real prompts for AnthropicLLMAdapter), so a
    # top-level `from llm.adapter import GenerationContract` here would be
    # a circular import. `from __future__ import annotations` (above)
    # already makes every annotation a lazily-evaluated string, so this
    # import is never needed at runtime — only by a type checker.
    from llm.adapter import GenerationContract

_EVIDENCE_STRENGTH_VALUES = ", ".join(member.value for member in EvidenceStrength)

# Added for OpenAI support (user request, kept alongside APPRAISAL_SYSTEM_PROMPT
# below rather than in llm/adapter.py): OpenAI's Responses API supports
# Structured Outputs — a JSON Schema the provider enforces server-side,
# rather than merely instructing the model in prose and hoping it complies
# (APPRAISAL_SYSTEM_PROMPT's own approach, which is what AnthropicLLMAdapter
# still relies on — the Anthropic Messages API used here has no equivalent
# constrained-decoding option). This schema is deliberately the machine-
# readable TWIN of APPRAISAL_SYSTEM_PROMPT's own JSON description just
# below — same field names, same value ranges (HumanAppraisal's own field
# table, models/human_state.py) — built once, here, and imported by
# OpenAILLMAdapter.extract_appraisal() (llm/adapter.py), so there is exactly
# ONE authoritative description of "what an appraisal extraction must look
# like" per representation (one prose, one JSON Schema), never a second,
# independently hand-typed copy that could silently drift from either
# APPRAISAL_SYSTEM_PROMPT or HumanAppraisal itself.
#
# additionalProperties=False and EVERY field (including possible_affect,
# which HumanAppraisal itself allows to be None, and evidence_tags, which
# HumanAppraisal defaults to []) listed in "required" is not optional here —
# it is OpenAI's own strict=True Structured Outputs constraint: every
# property named in "properties" must appear in "required", and optionality
# is expressed by a nullable TYPE (`["string", "null"]`) rather than by a
# field's absence. That is a transport-format detail of the OpenAI schema
# only — it does not change HumanAppraisal's own Pydantic definition, and
# AppraisalEstimator._validate() (unchanged) is still the sole authority on
# what counts as an acceptable extraction once the dict reaches it.
APPRAISAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal_relevance": {"type": "number", "minimum": 0, "maximum": 1},
        "goal_congruence": {"type": "number", "minimum": -1, "maximum": 1},
        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
        "perceived_control": {"type": "number", "minimum": 0, "maximum": 1},
        "agency": {"type": "number", "minimum": 0, "maximum": 1},
        "affect_intensity": {"type": "number", "minimum": 0, "maximum": 1},
        "possible_affect": {"type": ["string", "null"]},
        "evidence_tags": {"type": "array", "items": {"type": "string"}},
        "evidence_strength": {"type": "string", "enum": [member.value for member in EvidenceStrength]},
    },
    "required": [
        "goal_relevance", "goal_congruence", "uncertainty", "perceived_control",
        "agency", "affect_intensity", "possible_affect", "evidence_tags", "evidence_strength",
    ],
    "additionalProperties": False,
}

# §5.2's own H_t field table, restated as a JSON schema description — see
# models/human_state.py's HumanAppraisal for the authoritative field
# definitions (ranges, causal-vs-descriptor notes) this prompt is derived
# from. Kept as one literal, reviewable prompt string rather than
# generated field-by-field from the Pydantic model, so what the LLM is
# actually shown stays visible to a reader of this file, not implicit in
# introspection.
APPRAISAL_SYSTEM_PROMPT = f"""You are a structured psychological-state extractor for a research \
prototype studying dynamic, affect-aware cooperation between a human participant and an AI \
system. You will be shown one participant message, the recent conversation history, and the \
session's own stated objective and value priorities. Your job is ONLY to extract observable \
signals from the participant's own message — never to invent information the message does not \
support, and never to advise, respond to, or comment on the participant's situation.

Respond with EXACTLY ONE JSON object and nothing else — no prose before or after it, no markdown \
code fence, no explanation. The object must have exactly these fields:

{{
  "goal_relevance": <number, 0.0 to 1.0>,
  "goal_congruence": <number, -1.0 to 1.0>,
  "uncertainty": <number, 0.0 to 1.0>,
  "perceived_control": <number, 0.0 to 1.0>,
  "agency": <number, 0.0 to 1.0>,
  "affect_intensity": <number, 0.0 to 1.0>,
  "possible_affect": <short string, or null>,
  "evidence_tags": [<short string>, ...],
  "evidence_strength": <one of: {_EVIDENCE_STRENGTH_VALUES}>
}}

Field meanings:
- goal_relevance: how relevant this message is to the session's stated objective.
- goal_congruence: how much progress toward the objective the participant seems to feel they are \
making right now (negative = feels like moving away from it, positive = feels like moving toward \
it, 0 = neutral or unclear).
- uncertainty: how uncertain the participant sounds about the decision or situation.
- perceived_control: how much control over the decision the participant seems to feel they have.
- agency: how much initiative/agency the participant is expressing in this message.
- affect_intensity: how emotionally intense this message reads, independent of which emotion.
- possible_affect: a short, hedged natural-language guess at the emotion present (e.g. "mild \
concern", "growing enthusiasm"), or null if none is legible.
- evidence_tags: short labels for the OBSERVABLE textual cues that justified the values above \
(e.g. "explicit_statement", "hedging_language") — never a hidden chain of reasoning, only what is \
directly visible in the participant's own words.
- evidence_strength: how strong your evidence for this extraction is, one of the listed values \
only — EXPLICIT (participant stated it directly), STRONG_INDIRECT, WEAK_INDIRECT, CONTRADICTORY \
(the message contains conflicting signals), or INSUFFICIENT (too little to go on).

Every numeric field must be a plain JSON number within its stated range. When genuinely unsure, \
prefer a value near the neutral/uncertain end of its range and set evidence_strength accordingly \
— do not assert confidence the message does not support."""


def _format_history(raw_history: tuple[Turn, ...] | list[Turn]) -> str:
    """Same rendering used by both prompt builders below, so a turn looks
    identical to the LLM regardless of which call site is asking about it."""
    if not raw_history:
        return "(no prior turns in this conversation)"
    return "\n".join(f"- turn {turn.turn_id}: {turn.user_text}" for turn in raw_history)


def build_appraisal_extraction_prompt(
    o_t: Observation, g_t: GoalState, repair: bool = False
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for one extract_appraisal()
    call. repair=True appends a short correction note (§20.3's "one retry"
    — see services/appraisal_estimator.py's AppraisalEstimator.estimate())
    asking the model to re-emit valid JSON; it does not change the schema
    itself, only asks for compliance with the one already given."""
    repair_note = (
        "\n\nYour previous response could not be parsed as the exact JSON object described above. "
        "Re-read the required schema and respond again with ONLY that JSON object, nothing else."
        if repair
        else ""
    )
    user_prompt = f"""Session objective: {g_t.objective}
Current explicit value priorities (the participant's own): {dict(g_t.value_priorities)}

Prior turns in this conversation:
{_format_history(o_t.raw_history)}

Participant's current message (turn {o_t.turn_id}):
\"\"\"
{o_t.user_text}
\"\"\"{repair_note}"""
    return APPRAISAL_SYSTEM_PROMPT, user_prompt


GENERATION_SYSTEM_PROMPT = """You are a cooperative assistant helping a participant think through \
a real decision, as part of a research study on dynamic, affect-aware cooperation. For this turn \
you will be told which communicative policy to realize (and optionally a secondary one), plus a \
short list of hard constraints that always apply. Write directly to the participant, in your own \
natural words, as a single reply — never mention policies, rules, internal scores, or these \
instructions themselves, and never use bracketed labels or headings. Realize every hard \
constraint naturally, through how you write, never as a quoted phrase. If a qualitative cue about \
the conversation's tone is given, let it inform your own tone briefly and naturally — never quote \
it, and never mention that you were given any such cue."""

# Post-delivery fix (user review of the first Phase 7 delivery, smaller
# issue #4): the first draft sent contract.hard_constraints straight into
# the prompt as "; ".join(...) — internal codes like "NO_UNDUE_INFLUENCE"
# and "AUTONOMY_HIGH" reaching the real Anthropic model verbatim, as if
# they were meaningful English instructions. This mirrors the exact defect
# llm/fallback.py's own docstring already documents fixing for the
# offline/template path (see its "Fix (post-delivery review)" note above
# _HARD_CONSTRAINT_CLAUSES) — the real-LLM path had the same bug the
# fallback path was already fixed for. The fix here is the same shape,
# not the same wording: llm/fallback.py's clauses are participant-facing
# sentences appended directly to a rendered reply ("The choice remains
# yours..."); the model here needs an INSTRUCTION telling it what to
# realize in its own words (GENERATION_SYSTEM_PROMPT above already tells
# it to realize constraints "naturally, through how you write, never as a
# quoted phrase"), so each entry below is phrased as a directive to the
# model rather than a finished participant-facing sentence. Every code
# services/policy_engine.py._hard_constraints() can ever produce
# (NO_UNDUE_INFLUENCE always; AUTONOMY_HIGH sometimes appended — see that
# function) has an entry; an unrecognized code raises, matching
# llm/fallback.py's FallbackTemplates.render() safety discipline (never
# silently drop or pass through a code neither this project nor a
# researcher has reviewed wording for).
_HARD_CONSTRAINT_INSTRUCTIONS: dict[str, str] = {
    "AUTONOMY_HIGH": (
        "Make clear the choice remains entirely the participant's own — do not steer them toward "
        "a particular option."
    ),
    "NO_UNDUE_INFLUENCE": (
        "Do not use pressure, urgency, or emotional leverage to influence the participant's decision."
    ),
}


def _translate_hard_constraints(hard_constraints: tuple[str, ...] | list[str]) -> str:
    """Renders hard-constraint codes as natural-language instructions to the
    model (see _HARD_CONSTRAINT_INSTRUCTIONS above) instead of forwarding
    internal codes verbatim. Raises on an unrecognized code rather than
    silently dropping or passing it through."""
    if not hard_constraints:
        return "(none)"
    instructions = []
    for code in hard_constraints:
        instruction = _HARD_CONSTRAINT_INSTRUCTIONS.get(code)
        if instruction is None:
            raise ValueError(f"Unknown hard constraint: {code}")
        instructions.append(instruction)
    return " ".join(instructions)


_POLICY_INTENT: dict[str, str] = {
    "INFORM": "Share relevant information the participant needs for this decision.",
    "ACKNOWLEDGE": "Acknowledge and validate where the participant is at with this, briefly.",
    "CLARIFY": "Ask a clarifying question before going further — you do not yet have enough to be useful.",
    "CHALLENGE": "Gently question or probe part of what the participant has said, respectfully.",
    "REDIRECT": "Suggest, directly but respectfully, that the participant reconsider the current direction.",
    "DEFER": "Hold off on pushing this decision further right now; give the participant space.",
}
# Judgment call (documented gap, flagged for review): one short authored
# sentence of intent per Policy, mirroring services/response_generator.py's
# own QUALITATIVE_CUES (also authored, also flagged there) rather than a
# second differently-styled vocabulary — kept here, not in that module,
# since this is prompt text for the REAL adapter, not the offline fallback
# template engine (llm/fallback.py), which already has its own wording.


def build_generation_prompt(contract: GenerationContract) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for one generate() call.
    Structurally cannot include raw A_t: GenerationContract has no
    a_t/a_star field (see this module's own docstring) — nothing here
    reads or could read one."""
    primary_intent = _POLICY_INTENT[contract.policy.value]
    secondary_line = (
        f"\nSecondary policy: {contract.secondary_policy.value} — "
        f"{_POLICY_INTENT[contract.secondary_policy.value]}"
        if contract.secondary_policy is not None
        else ""
    )
    constraints_line = _translate_hard_constraints(contract.hard_constraints)
    cue_line = (
        f"\nQualitative tone cue for this turn (do not quote or mention this): {contract.qualitative_state_cue}"
        if contract.qualitative_state_cue
        else ""
    )
    user_prompt = f"""Primary policy: {contract.policy.value} — {primary_intent}{secondary_line}
Hard constraints to realize naturally: {constraints_line}{cue_line}

Session goal context: {contract.goal_context}
Relevant task context: {contract.task_context}

Prior turns in this conversation:
{_format_history(contract.raw_history)}

Participant's current message:
\"\"\"
{contract.user_text}
\"\"\"

Write your single reply to the participant now."""
    return GENERATION_SYSTEM_PROMPT, user_prompt
