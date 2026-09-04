"""tests/test_generator_isolation.py — M7 (ResponseGenerator) + fallback
templates, blueprint §6.6/§6.8/§20.9. Phase 4 gate (blueprint §9 test map,
§10 Phase 4): "test_generator_isolation.py green, incl. secondary/
hard_constraints in the fallback path."

Named to match the blueprint's §9 test map rows exactly where it names
them: test_generator_receives_no_numeric_state,
test_generator_receives_qualitative_cue_only,
test_fallback_template_includes_secondary_and_hard_constraints.
"""

import inspect
from datetime import datetime, timezone

import pytest

from llm.adapter import GenerationContract, GeneratorError, MockLLMAdapter
from llm.fallback import _PRIMARY_TEMPLATES, FallbackTemplates
from models.enums import GeneratorStatus, Policy, RationaleCode
from models.goal_state import GoalState
from models.observation import Observation
from models.policy import PolicyState
from services.response_generator import QUALITATIVE_CUES, ResponseGenerator, _relevant, _relevant_goal_fields


def _goal_state(**overrides) -> GoalState:
    defaults = dict(
        objective="decide whether to proceed",
        value_priorities={"autonomy": 0.6, "safety": 0.4},
        stakes=0.5,
        task_constraints={"deadline": "EOD"},
        autonomy_weight=0.5,
        safety_risk=0.2,
    )
    defaults.update(overrides)
    return GoalState(**defaults)


def _observation(**overrides) -> Observation:
    defaults = dict(
        turn_id=1,
        timestamp=datetime.now(timezone.utc),
        user_text="I think option A is fine.",
        task_context={
            "options": {"A": {"autonomy": 0.5}},
            "focal_option_id": "A",
            "d_goal_override": 0.1,
            "required_evidence": [{"status": "resolved"}],
        },
    )
    defaults.update(overrides)
    return Observation(**defaults)


def _policy_state(**overrides) -> PolicyState:
    defaults = dict(
        primary=Policy.INFORM,
        secondary=None,
        triggered_rules=["R_DEFAULT"],
        hard_constraints=["NO_UNDUE_INFLUENCE"],
        rationale_code=RationaleCode.MINIMAL_SUPPORT,
        state_could_influence_policy=False,
        state_did_influence_policy=False,
    )
    defaults.update(overrides)
    return PolicyState(**defaults)


@pytest.fixture()
def generator_with_success():
    adapter = MockLLMAdapter(generate_responses=["a generated response"])
    return ResponseGenerator(adapter, FallbackTemplates()), adapter


# ---------- causal isolation: no numeric A_t reaches the generator ----------


def test_generator_receives_no_numeric_state(generator_with_success):
    """generate()'s own signature has no a_t/a_star parameter, and
    GenerationContract has no field that could carry one — the caller
    (Pipeline, a later phase) has no way to pass raw A_t here even if it
    tried, by construction."""
    params = list(inspect.signature(ResponseGenerator.generate).parameters)
    assert params == ["self", "o_t", "g_t", "raw_history", "p_t"]

    contract_fields = set(inspect.signature(GenerationContract.__init__).parameters) - {"self"}
    assert "a_t" not in contract_fields
    assert "a_star" not in contract_fields

    generator, adapter = generator_with_success
    p_t = _policy_state(state_could_influence_policy=False, state_did_influence_policy=False)
    generator.generate(_observation(), _goal_state(), raw_history=(), p_t=p_t)
    contract = adapter.generate_calls[0]
    assert not hasattr(contract, "a_t")
    assert not hasattr(contract, "a_star")
    assert contract.qualitative_state_cue is None   # did=False -> no cue at all


def test_generator_receives_qualitative_cue_only(generator_with_success):
    """When state_did_influence_policy=True, the ONLY trace of A_t/A*_t's
    influence reaching the contract is a short, non-numeric string from
    QUALITATIVE_CUES — never a float."""
    generator, adapter = generator_with_success
    p_t = _policy_state(
        rationale_code=RationaleCode.REDIRECT_ELIGIBLE,
        state_could_influence_policy=True,
        state_did_influence_policy=True,
    )
    generator.generate(_observation(), _goal_state(), raw_history=(), p_t=p_t)
    contract = adapter.generate_calls[0]
    assert contract.qualitative_state_cue == QUALITATIVE_CUES[RationaleCode.REDIRECT_ELIGIBLE]
    assert isinstance(contract.qualitative_state_cue, str)
    assert not any(char.isdigit() for char in contract.qualitative_state_cue)


def test_qualitative_cues_cover_every_rationale_code_and_contain_no_digits():
    assert set(QUALITATIVE_CUES) == set(RationaleCode)
    for code, cue in QUALITATIVE_CUES.items():
        assert isinstance(cue, str) and cue, f"{code} has no cue"
        assert not any(char.isdigit() for char in cue), f"{code}'s cue contains a digit: {cue!r}"


# ---------- generate(): success and both failure modes ----------


def test_generate_returns_ok_on_successful_llm_call(generator_with_success):
    generator, adapter = generator_with_success
    text, status = generator.generate(_observation(), _goal_state(), raw_history=(), p_t=_policy_state())
    assert text == "a generated response"
    assert status == GeneratorStatus.OK


def test_generate_falls_back_on_timeout_error():
    adapter = MockLLMAdapter(generate_responses=[TimeoutError()])
    generator = ResponseGenerator(adapter, FallbackTemplates())
    p_t = _policy_state(primary=Policy.DEFER, secondary=Policy.INFORM)
    text, status = generator.generate(_observation(), _goal_state(), raw_history=(), p_t=p_t)
    assert status == GeneratorStatus.FALLBACK_TEMPLATE
    assert text  # never silence


def test_generate_falls_back_on_generator_error():
    adapter = MockLLMAdapter(generate_responses=[GeneratorError("provider refused")])
    generator = ResponseGenerator(adapter, FallbackTemplates())
    text, status = generator.generate(_observation(), _goal_state(), raw_history=(), p_t=_policy_state())
    assert status == GeneratorStatus.FALLBACK_TEMPLATE
    assert text


def test_generate_does_not_catch_unrelated_exceptions():
    """Only TimeoutError/GeneratorError trigger the fallback (§6.6's exact
    except clause) — anything else must propagate, not be silently
    swallowed into a fallback response."""
    adapter = MockLLMAdapter(generate_responses=[ValueError("not a generation failure")])
    generator = ResponseGenerator(adapter, FallbackTemplates())
    with pytest.raises(ValueError):
        generator.generate(_observation(), _goal_state(), raw_history=(), p_t=_policy_state())


# ---------- fallback template: primary + secondary + hard_constraints together ----------


def test_fallback_template_includes_secondary_and_hard_constraints():
    """Named to match the blueprint's §9 test map row exactly: a DEFER+
    INFORM combination must not collapse to bare DEFER's template — the
    implementation-review fix §6.8 itself documents. Hard constraints must
    be realized as participant-facing behavioral language, never as the
    internal control-label strings (post-delivery review fix) — so this
    asserts on the realized clauses, not on "AUTONOMY_HIGH"/
    "NO_UNDUE_INFLUENCE" appearing literally in the text."""
    fallback = FallbackTemplates()
    p_t = _policy_state(
        primary=Policy.DEFER,
        secondary=Policy.INFORM,
        hard_constraints=["AUTONOMY_HIGH", "NO_UNDUE_INFLUENCE"],
    )
    text = fallback.render(p_t, task_context={})
    assert "hold off" in text.lower() or "defer" in text.lower()   # primary (DEFER) realized
    assert "information" in text.lower()                            # secondary (INFORM) realized
    assert "choice remains yours" in text.lower()                   # AUTONOMY_HIGH realized
    assert "won't push" in text.lower()                             # AUTONOMY_HIGH realized
    assert "won't use pressure" in text.lower()                     # NO_UNDUE_INFLUENCE realized
    assert "AUTONOMY_HIGH" not in text
    assert "NO_UNDUE_INFLUENCE" not in text


def test_fallback_rejects_unknown_hard_constraint():
    """render() is safety infrastructure for the LLM-failure path — a
    hard-constraint code with no participant-facing realization must fail
    loudly (here, in testing) rather than being silently dropped from the
    response a participant actually sees."""
    fallback = FallbackTemplates()
    p_t = _policy_state(primary=Policy.INFORM, secondary=None, hard_constraints=["SOME_NEW_CONSTRAINT"])
    with pytest.raises(ValueError, match="SOME_NEW_CONSTRAINT"):
        fallback.render(p_t, task_context={})


def test_fallback_template_omits_secondary_clause_when_none():
    fallback = FallbackTemplates()
    p_t = _policy_state(primary=Policy.ACKNOWLEDGE, secondary=None, hard_constraints=[])
    text = fallback.render(p_t, task_context={})
    assert "also" not in text.lower()   # every secondary clause is phrased as an addition
    assert text == _PRIMARY_TEMPLATES[Policy.ACKNOWLEDGE]


@pytest.mark.parametrize("primary", list(Policy))
@pytest.mark.parametrize("secondary", [None, *list(Policy)])
def test_fallback_render_never_raises_or_returns_empty(primary, secondary):
    if secondary == primary:
        pytest.skip("PolicyState doesn't forbid this, but it isn't a real rule output — not worth covering")
    fallback = FallbackTemplates()
    p_t = _policy_state(primary=primary, secondary=secondary, hard_constraints=["AUTONOMY_HIGH"])
    text = fallback.render(p_t, task_context={"anything": "goes"})
    assert isinstance(text, str) and text


# ---------- _relevant / _relevant_goal_fields filtering ----------


def test_relevant_strips_only_derived_feature_internal_keys():
    task_context = {
        "options": {"A": {"autonomy": 1.0}},
        "focal_option_id": "A",
        "task_event": {"kind": "reminder"},
        "d_goal_override": 0.2,
        "d_amb_override": 0.3,
        "required_evidence": [{"status": "resolved"}],
    }
    filtered = _relevant(task_context, _goal_state())
    assert filtered == {
        "options": {"A": {"autonomy": 1.0}},
        "focal_option_id": "A",
        "task_event": {"kind": "reminder"},
    }


def test_relevant_goal_fields_excludes_bookkeeping_and_includes_content():
    g_t = _goal_state()
    fields = _relevant_goal_fields(g_t)
    assert fields == {
        "objective": g_t.objective,
        "value_priorities": dict(g_t.value_priorities),
        "stakes": g_t.stakes,
        "task_constraints": g_t.task_constraints,
        "autonomy_weight": g_t.autonomy_weight,
        "safety_risk": g_t.safety_risk,
    }
    assert "goal_version" not in fields
    assert "update_source" not in fields
    assert "no_undue_influence" not in fields
