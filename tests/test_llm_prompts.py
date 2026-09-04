"""tests/test_llm_prompts.py — llm/prompts.py (Phase 7). Not named in the
blueprint's own test map (prompts.py is named only once, in the repo layout
line "llm/prompts.py # structured-extraction prompt template" — no test file
is ever assigned to it either); added per this project's own established
pattern of covering every authored-content module with tests, same as
test_generator_isolation.py covering the equally-authored QUALITATIVE_CUES/
FallbackTemplates content in Phase 4.

Two concerns: (1) the appraisal-extraction prompt's schema matches
HumanAppraisal's own field set exactly, so a real LLM response following it
is exactly what AppraisalEstimator._validate() already expects; (2) the
generation prompt can never leak an AgentState/AgentTargetState field name —
the causal-isolation rule (§20.9) extended into the actual prompt text, not
just GenerationContract's own field set (already checked structurally by
its lack of an a_t/a_star field, but this test checks the STRING content
too, in case a future edit to this file's authored wording accidentally
names one).
"""

from datetime import datetime, timezone

from llm.adapter import GenerationContract
import pytest

from llm.prompts import (
    APPRAISAL_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    build_appraisal_extraction_prompt,
    build_generation_prompt,
)
from models.enums import EvidenceStrength, Policy
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.observation import Observation, Turn

# Every field name HumanAppraisal actually has (models/human_state.py) —
# the appraisal prompt's schema must mention every one of these, by name,
# so a compliant LLM response is directly constructible into a HumanAppraisal.
_HUMAN_APPRAISAL_FIELDS = list(HumanAppraisal.model_fields)

# AgentState/AgentTargetState's own field names — must NEVER appear in a
# generation prompt (see this module's own docstring).
_AGENT_STATE_FIELDS = ["motivational_priority", "decision_information_priority", "intervention_readiness"]


def _goal_state(**overrides) -> GoalState:
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


def _observation(raw_history=(), turn_id=1, user_text="hello") -> Observation:
    return Observation(
        turn_id=turn_id, timestamp=datetime.now(timezone.utc), user_text=user_text,
        task_context={}, raw_history=list(raw_history),
    )


def test_appraisal_system_prompt_names_every_human_appraisal_field():
    for field_name in _HUMAN_APPRAISAL_FIELDS:
        assert field_name in APPRAISAL_SYSTEM_PROMPT, f"missing field {field_name!r} in appraisal prompt schema"


def test_appraisal_system_prompt_lists_every_evidence_strength_value():
    for member in EvidenceStrength:
        assert member.value in APPRAISAL_SYSTEM_PROMPT


def test_build_appraisal_extraction_prompt_embeds_user_text_and_objective():
    g_t = _goal_state(objective="pick a job offer")
    o_t = _observation(user_text="I'm torn on the startup offer.")
    system, user = build_appraisal_extraction_prompt(o_t, g_t)

    assert system == APPRAISAL_SYSTEM_PROMPT  # system prompt is the fixed schema, never per-turn content
    assert "pick a job offer" in user
    assert "I'm torn on the startup offer." in user
    assert "no prior turns" in user.lower()


def test_build_appraisal_extraction_prompt_includes_history():
    g_t = _goal_state()
    history = (Turn(turn_id=1, user_text="first turn text"),)
    o_t = _observation(raw_history=history, turn_id=2, user_text="second turn text")
    _system, user = build_appraisal_extraction_prompt(o_t, g_t)

    assert "first turn text" in user
    assert "second turn text" in user


def test_build_appraisal_extraction_prompt_repair_flag_adds_correction_note():
    g_t = _goal_state()
    o_t = _observation()
    _system, user_no_repair = build_appraisal_extraction_prompt(o_t, g_t, repair=False)
    _system, user_repair = build_appraisal_extraction_prompt(o_t, g_t, repair=True)

    assert "could not be parsed" not in user_no_repair
    assert "could not be parsed" in user_repair


def test_generation_prompt_never_names_an_agent_state_field():
    contract = GenerationContract(
        user_text="hi", task_context={}, raw_history=(),
        goal_context={"objective": "decide", "value_priorities": {"a": 0.5}},
        policy=Policy.REDIRECT, secondary_policy=Policy.ACKNOWLEDGE,
        hard_constraints=["AUTONOMY_HIGH", "NO_UNDUE_INFLUENCE"],
        qualitative_state_cue="engagement has grown across the conversation",
    )
    system, user = build_generation_prompt(contract)
    combined = system + "\n" + user
    for field_name in _AGENT_STATE_FIELDS:
        assert field_name not in combined
    assert "a_star" not in combined
    assert "a_t" not in combined
    assert "rho" not in combined


def test_generation_prompt_embeds_policy_and_hard_constraints_and_cue():
    contract = GenerationContract(
        user_text="what should I do?", task_context={"focal_option_id": "startup_offer"},
        raw_history=(), goal_context={"objective": "decide"},
        policy=Policy.CLARIFY, secondary_policy=None,
        hard_constraints=["NO_UNDUE_INFLUENCE"],
        qualitative_state_cue=None,
    )
    system, user = build_generation_prompt(contract)

    assert system == GENERATION_SYSTEM_PROMPT
    assert "CLARIFY" in user
    # Post-delivery fix (smaller issue #4): the raw hard-constraint CODE must
    # never reach the prompt — only its translated natural-language
    # instruction to the model (see llm/prompts.py's
    # _HARD_CONSTRAINT_INSTRUCTIONS / _translate_hard_constraints()).
    assert "NO_UNDUE_INFLUENCE" not in user
    assert "pressure" in user.lower() and "influence the participant" in user.lower()
    assert "what should I do?" in user
    assert "Secondary policy" not in user  # None secondary_policy omits the whole line
    assert "Qualitative tone cue" not in user  # None cue omits the whole line


def test_generation_prompt_translates_autonomy_high_constraint():
    contract = GenerationContract(
        user_text="what should I do?", task_context={}, raw_history=(),
        goal_context={"objective": "decide"}, policy=Policy.INFORM, secondary_policy=None,
        hard_constraints=["AUTONOMY_HIGH"], qualitative_state_cue=None,
    )
    _system, user = build_generation_prompt(contract)

    assert "AUTONOMY_HIGH" not in user
    assert "participant's own" in user.lower()


def test_generation_prompt_raises_on_unrecognized_hard_constraint():
    contract = GenerationContract(
        user_text="ok", task_context={}, raw_history=(), goal_context={},
        policy=Policy.INFORM, secondary_policy=None,
        hard_constraints=["SOME_UNKNOWN_CODE"], qualitative_state_cue=None,
    )
    with pytest.raises(ValueError, match="SOME_UNKNOWN_CODE"):
        build_generation_prompt(contract)


def test_generation_prompt_omits_none_secondary_and_cue_lines_when_present():
    contract = GenerationContract(
        user_text="ok", task_context={}, raw_history=(), goal_context={},
        policy=Policy.INFORM, secondary_policy=Policy.ACKNOWLEDGE,
        hard_constraints=[], qualitative_state_cue="a light touch seemed appropriate here",
    )
    _system, user = build_generation_prompt(contract)

    assert "Secondary policy: ACKNOWLEDGE" in user
    assert "a light touch seemed appropriate here" in user
    assert "(none)" in user  # empty hard_constraints renders as an explicit "(none)", never a blank line
