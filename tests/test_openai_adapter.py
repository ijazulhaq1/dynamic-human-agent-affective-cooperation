"""tests/test_openai_adapter.py — llm/adapter.py's OpenAILLMAdapter, added
at the user's own explicit request as a THIRD, optional backend alongside
Offline (OfflineDemoAdapter) and Anthropic (AnthropicLLMAdapter) — never a
replacement for either. Not named in the blueprint (no specific provider is
ever named there — see AnthropicLLMAdapter's own docstring); added per this
project's established pattern of testing every module it builds, mirroring
tests/test_anthropic_adapter.py's own structure and coverage almost line for
line so the two providers' adapters are held to the identical standard.

Every test here constructs OpenAILLMAdapter with a FAKE client double — no
real network access, API key, or the `openai` package's own runtime request
behavior is exercised. The fake double's `.responses.create(...)` returns an
object shaped like the real OpenAI Responses API's own response (an
`.output_text` convenience property — see _extract_openai_text() in
llm/adapter.py), so the SAME parsing code that would run against a real
response runs here too — only the network call itself is faked.

The user's own explicit non-negotiables, each with its own test below:
  - OpenAI must never compute D_t/A*_t/A_t/P_t, and no numeric A_t/A*_t
    field may ever reach it (test_generate_never_leaks_agent_state_fields).
  - Internal hard-constraint codes (e.g. NO_UNDUE_INFLUENCE) must already be
    translated to natural language before OpenAI ever sees them — the exact
    llm/prompts.py fix already applied for Anthropic in the Phase 7 fix
    round, reused unchanged here (test_generate_never_sends_raw_hard_
    constraint_codes).
  - No chain-of-thought is ever requested of the model
    (test_prompts_never_request_chain_of_thought).
  - Both providers receive the SAME conceptual prompt content — this is
    verified by construction (OpenAILLMAdapter calls the exact same
    llm/prompts.py functions AnthropicLLMAdapter calls; there is no second,
    OpenAI-specific prompt-building function anywhere in this codebase to
    even test against), and cross-checked here by asserting the `input`
    OpenAI receives contains the same session content (objective, user
    text) tests/test_anthropic_adapter.py already asserts for Anthropic.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import openai as openai_sdk
import pytest

from llm.adapter import GenerationContract, GeneratorError, OpenAILLMAdapter
from models.enums import Policy
from models.goal_state import GoalState
from models.observation import Observation


def _text_response(text: str) -> SimpleNamespace:
    """A fake OpenAI Responses API response — same `.output_text` shape
    _extract_openai_text() (llm/adapter.py) reads from a real one."""
    return SimpleNamespace(output_text=text)


class FakeOpenAIClient:
    """Records every call it receives and returns (or raises) whatever was
    scripted for it — the same scriptable-queue pattern
    tests/test_anthropic_adapter.py's own FakeAnthropicClient uses, at the
    same transport-double layer (OpenAILLMAdapter itself is the class under
    test, not this double)."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _fake_timeout_error() -> openai_sdk.APITimeoutError:
    """A REAL openai.APITimeoutError instance, not a stand-in — exercises
    _is_openai_timeout()'s provider-specific branch (llm/adapter.py), not
    just its builtin-TimeoutError branch. Built via __new__() rather than
    the real constructor (which requires an httpx request object from the
    SDK's own vendored transport library) so this test stays independent
    of that internal, version-specific detail — only real exception-class
    MEMBERSHIP is being exercised here, never any of its message/request
    attributes."""
    return openai_sdk.APITimeoutError.__new__(openai_sdk.APITimeoutError)


def _fake_auth_error() -> openai_sdk.AuthenticationError:
    """A REAL openai.AuthenticationError instance — exercises the
    "auth/provider exception -> None for extract_appraisal" requirement
    with the actual exception class a real invalid-key call would raise,
    built the same __new__()-based way as _fake_timeout_error() above, for
    the same reason."""
    return openai_sdk.AuthenticationError.__new__(openai_sdk.AuthenticationError)


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


def _observation() -> Observation:
    return Observation(turn_id=1, timestamp=datetime.now(timezone.utc), user_text="hi", task_context={})


VALID_APPRAISAL_JSON = """{
  "goal_relevance": 0.8, "goal_congruence": 0.1, "uncertainty": 0.3, "perceived_control": 0.5,
  "agency": 0.6, "affect_intensity": 0.4, "possible_affect": "mild concern",
  "evidence_tags": ["explicit_statement"], "evidence_strength": "EXPLICIT"
}"""


# ---------------------------------------------------------------------------
# extract_appraisal
# ---------------------------------------------------------------------------


def test_extract_appraisal_parses_valid_structured_response():
    """Successful structured appraisal extraction: a schema-compatible dict
    is returned, and the request actually used Structured Outputs
    (text.format.type == "json_schema", strict=True) against the SAME
    schema HumanAppraisal itself defines — not free-form JSON-in-prose."""
    client = FakeOpenAIClient(responses=[_text_response(VALID_APPRAISAL_JSON)])
    adapter = OpenAILLMAdapter(client, model="test-model")

    result = adapter.extract_appraisal(_observation(), _goal_state())

    assert result == {
        "goal_relevance": 0.8, "goal_congruence": 0.1, "uncertainty": 0.3, "perceived_control": 0.5,
        "agency": 0.6, "affect_intensity": 0.4, "possible_affect": "mild concern",
        "evidence_tags": ["explicit_statement"], "evidence_strength": "EXPLICIT",
    }
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert "decide whether to proceed" in call["input"]  # the goal_state's own objective
    text_format = call["text"]["format"]
    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    schema_fields = set(text_format["schema"]["properties"])
    assert schema_fields == {
        "goal_relevance", "goal_congruence", "uncertainty", "perceived_control",
        "agency", "affect_intensity", "possible_affect", "evidence_tags", "evidence_strength",
    }


def test_extract_appraisal_returns_none_on_malformed_json():
    client = FakeOpenAIClient(responses=[_text_response("not json at all")])
    adapter = OpenAILLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_returns_none_on_empty_output_text():
    """output_text is "" (the SDK's own documented empty-case return, not
    an exception) when a response carries no text content — must be
    treated as a failed extraction, not parsed as if it were valid."""
    client = FakeOpenAIClient(responses=[_text_response("")])
    adapter = OpenAILLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_returns_none_on_timeout():
    client = FakeOpenAIClient(responses=[_fake_timeout_error()])
    adapter = OpenAILLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_returns_none_on_auth_error():
    client = FakeOpenAIClient(responses=[_fake_auth_error()])
    adapter = OpenAILLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_never_raises_on_generic_transport_failure():
    """extract_appraisal's own documented contract (llm/adapter.py's module
    docstring, unchanged by this addition): ANY transport-level failure —
    not just the two OpenAI-specific exception classes exercised above — is
    reported the SAME way malformed JSON is, by returning None, never by
    raising."""
    client = FakeOpenAIClient(responses=[ConnectionError("network is down")])
    adapter = OpenAILLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_repair_flag_reaches_the_prompt():
    client = FakeOpenAIClient(
        responses=[_text_response(VALID_APPRAISAL_JSON), _text_response(VALID_APPRAISAL_JSON)]
    )
    adapter = OpenAILLMAdapter(client)

    adapter.extract_appraisal(_observation(), _goal_state(), repair=False)
    adapter.extract_appraisal(_observation(), _goal_state(), repair=True)

    prompt_no_repair = client.calls[0]["input"]
    prompt_repair = client.calls[1]["input"]
    assert "could not be parsed" not in prompt_no_repair
    assert "could not be parsed" in prompt_repair


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _generation_contract(**overrides) -> GenerationContract:
    defaults = dict(
        user_text="what should I do?", task_context={}, raw_history=(),
        goal_context={"objective": "decide"}, policy=Policy.CLARIFY, secondary_policy=None,
        hard_constraints=["NO_UNDUE_INFLUENCE"], qualitative_state_cue=None,
    )
    defaults.update(overrides)
    return GenerationContract(**defaults)


def test_generate_returns_text_on_success():
    client = FakeOpenAIClient(responses=[_text_response("Here's some information that's relevant.")])
    adapter = OpenAILLMAdapter(client, model="test-model", max_output_tokens=256, temperature=0.2)

    result = adapter.generate(_generation_contract())

    assert result == "Here's some information that's relevant."
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["max_output_tokens"] == 256
    assert client.calls[0]["temperature"] == 0.2
    # generate() never asks for Structured Outputs — there is no schema to
    # enforce for free-text generation, unlike extract_appraisal above.
    assert "text" not in client.calls[0]


def test_generate_raises_timeout_error_on_builtin_timeout():
    client = FakeOpenAIClient(responses=[TimeoutError("took too long")])
    adapter = OpenAILLMAdapter(client)

    with pytest.raises(TimeoutError):
        adapter.generate(_generation_contract())


def test_generate_raises_timeout_error_on_openai_api_timeout():
    client = FakeOpenAIClient(responses=[_fake_timeout_error()])
    adapter = OpenAILLMAdapter(client)

    with pytest.raises(TimeoutError):
        adapter.generate(_generation_contract())


def test_generate_raises_generator_error_on_other_failure():
    client = FakeOpenAIClient(responses=[ConnectionError("network is down")])
    adapter = OpenAILLMAdapter(client)

    with pytest.raises(GeneratorError):
        adapter.generate(_generation_contract())


def test_generate_raises_generator_error_on_empty_output_text():
    client = FakeOpenAIClient(responses=[_text_response("")])
    adapter = OpenAILLMAdapter(client)

    with pytest.raises(GeneratorError):
        adapter.generate(_generation_contract())


# ---------------------------------------------------------------------------
# Scientific-isolation guarantees (user's explicit non-negotiables)
# ---------------------------------------------------------------------------


def test_generate_never_leaks_agent_state_fields():
    """No numeric A_t/A*_t field, nor rho, ever reaches OpenAI — the same
    causal-isolation check tests/test_llm_prompts.py already runs on the
    shared prompt text itself, repeated here at the ADAPTER boundary so a
    future change to how OpenAILLMAdapter assembles its own request (as
    opposed to the prompt text) can't reintroduce a leak some other way
    (e.g. stuffing extra fields into `instructions` or `input` outside of
    build_generation_prompt())."""
    client = FakeOpenAIClient(responses=[_text_response("A reply.")])
    adapter = OpenAILLMAdapter(client)
    contract = _generation_contract(
        hard_constraints=["AUTONOMY_HIGH", "NO_UNDUE_INFLUENCE"],
        qualitative_state_cue="engagement has grown across the conversation",
    )

    adapter.generate(contract)

    sent = client.calls[0]["instructions"] + "\n" + client.calls[0]["input"]
    for field_name in ("motivational_priority", "decision_information_priority", "intervention_readiness"):
        assert field_name not in sent
    assert "a_star" not in sent
    assert "a_t" not in sent
    assert "rho" not in sent


def test_generate_never_sends_raw_hard_constraint_codes():
    """The Phase 7 fix round's hard-constraint translation (llm/prompts.py's
    _translate_hard_constraints(), originally fixed for Anthropic) is reused
    unchanged for OpenAI — reused, not reimplemented, since both adapters
    call the identical build_generation_prompt()."""
    client = FakeOpenAIClient(responses=[_text_response("A reply.")])
    adapter = OpenAILLMAdapter(client)
    contract = _generation_contract(hard_constraints=["NO_UNDUE_INFLUENCE"])

    adapter.generate(contract)

    sent_input = client.calls[0]["input"]
    assert "NO_UNDUE_INFLUENCE" not in sent_input
    assert "pressure" in sent_input.lower()


def test_prompts_never_request_chain_of_thought():
    """Neither the appraisal-extraction prompt nor the generation prompt —
    the SAME shared prompts both providers receive — ever asks the model to
    show its reasoning process; the appraisal prompt explicitly forbids it
    ("no hidden chain-of-thought", evidence_tags must be observable cues
    only)."""
    from llm.prompts import build_appraisal_extraction_prompt, build_generation_prompt

    appraisal_system, appraisal_user = build_appraisal_extraction_prompt(_observation(), _goal_state())
    generation_system, generation_user = build_generation_prompt(_generation_contract())

    forbidden_phrases = ["chain of thought", "chain-of-thought", "think step by step", "show your reasoning"]
    combined = "\n".join(
        [appraisal_system, appraisal_user, generation_system, generation_user]
    ).lower()
    for phrase in forbidden_phrases:
        assert phrase not in combined

    # The appraisal prompt affirmatively forbids a hidden reasoning chain
    # (not merely omits asking for one) — evidence_tags must be observable.
    assert "hidden chain of reasoning" in appraisal_system.lower()


def test_appraisal_json_schema_matches_human_appraisal_fields_exactly():
    """APPRAISAL_JSON_SCHEMA (llm/prompts.py) is the schema OpenAI's
    Structured Outputs enforces server-side — it must name exactly
    HumanAppraisal's own fields (models/human_state.py), no more, no
    fewer, and no renamed field, so a compliant OpenAI response is directly
    usable by AppraisalEstimator._validate() with zero translation."""
    from llm.prompts import APPRAISAL_JSON_SCHEMA
    from models.human_state import HumanAppraisal

    assert set(APPRAISAL_JSON_SCHEMA["properties"]) == set(HumanAppraisal.model_fields)
    assert set(APPRAISAL_JSON_SCHEMA["required"]) == set(HumanAppraisal.model_fields)
    assert APPRAISAL_JSON_SCHEMA["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Construction / environment wiring
# ---------------------------------------------------------------------------


def test_model_defaults_to_openai_model_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-from-env")
    client = FakeOpenAIClient(responses=[])
    adapter = OpenAILLMAdapter(client)

    assert adapter.model == "gpt-test-from-env"


def test_model_falls_back_to_default_when_no_env_or_arg(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = FakeOpenAIClient(responses=[])
    adapter = OpenAILLMAdapter(client)

    assert adapter.model == OpenAILLMAdapter.DEFAULT_MODEL


def test_explicit_model_argument_overrides_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-from-env")
    client = FakeOpenAIClient(responses=[])
    adapter = OpenAILLMAdapter(client, model="explicit-model")

    assert adapter.model == "explicit-model"


def test_constructor_builds_real_client_from_env_when_none_given(monkeypatch):
    """client=None (the default) must build a real openai.OpenAI() reading
    OPENAI_API_KEY from the environment itself — never a hardcoded key.
    Constructing the SDK client does not itself make a network call, so
    this is safe to exercise directly with a fake key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-not-real")
    adapter = OpenAILLMAdapter()

    assert isinstance(adapter._client, openai_sdk.OpenAI)  # noqa: SLF001 — verifying construction, not behavior


def test_from_env_raises_clear_error_without_openai_package(monkeypatch):
    """This sandbox DOES have the openai package installed, so this test
    simulates the "not installed" branch directly rather than actually
    uninstalling anything — same pattern
    tests/test_anthropic_adapter.py's own equivalent test uses."""
    import llm.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "openai", None)
    with pytest.raises(ImportError, match="openai"):
        adapter_module.OpenAILLMAdapter.from_env()


def test_constructor_raises_clear_error_without_openai_package_and_no_client(monkeypatch):
    """Same as from_env() above, but via the plain constructor's own
    client=None fallback path (this class's distinguishing feature versus
    AnthropicLLMAdapter, which has no such fallback in its constructor)."""
    import llm.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "openai", None)
    with pytest.raises(ImportError, match="openai"):
        adapter_module.OpenAILLMAdapter()


# ---------------------------------------------------------------------------
# Researcher-only provider diagnostics
# ---------------------------------------------------------------------------


def test_provider_failure_records_sanitized_last_error_and_success_clears_it():
    client = FakeOpenAIClient(
        responses=[
            ConnectionError("api_key=sk-supersecret123456789"),
            _text_response(VALID_APPRAISAL_JSON),
        ]
    )
    adapter = OpenAILLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None
    assert adapter.last_error is not None
    assert "ConnectionError" in adapter.last_error
    assert "supersecret" not in adapter.last_error
    assert "[REDACTED]" in adapter.last_error

    assert adapter.extract_appraisal(_observation(), _goal_state()) is not None
    assert adapter.last_error is None


def test_generation_failure_records_last_error():
    client = FakeOpenAIClient(responses=[ConnectionError("network is down")])
    adapter = OpenAILLMAdapter(client)

    with pytest.raises(GeneratorError):
        adapter.generate(_generation_contract())

    assert adapter.last_error == "ConnectionError: network is down"
