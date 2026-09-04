"""tests/test_anthropic_adapter.py — llm/adapter.py's AnthropicLLMAdapter
(Phase 7's "real llm/adapter.py backend", blueprint §10). Not named in the
blueprint's own test map (nothing there anticipates a specific provider —
see AnthropicLLMAdapter's own docstring for why Anthropic was the one
concrete backend built); added per this project's established pattern of
testing every module it builds.

Every test here constructs AnthropicLLMAdapter with a FAKE client double —
no real network access, API key, or the `anthropic` package's own runtime
behavior is exercised. The fake double's `.messages.create(...)` returns an
object shaped exactly like the real Anthropic SDK's own response
(`.content` = a list of blocks, each with `.type`/`.text`), so the SAME
parsing code (_extract_text/_parse_json_object in llm/adapter.py) that would
run against a real response runs here too — only the network call itself is
faked, not the response-shape contract.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm.adapter import AnthropicLLMAdapter, GenerationContract, GeneratorError
from models.enums import Policy
from models.goal_state import GoalState
from models.observation import Observation


def _text_response(text: str) -> SimpleNamespace:
    """A fake Anthropic Messages API response carrying one text block —
    same shape _extract_text() (llm/adapter.py) reads from a real one."""
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeAnthropicClient:
    """Records every call it receives and returns (or raises) whatever was
    scripted for it — MockLLMAdapter's own scriptable-queue pattern
    (llm/adapter.py), reused here at the transport-double layer instead of
    the LLMAdapter layer, since AnthropicLLMAdapter itself is the class
    under test this time."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self._responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


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


def test_extract_appraisal_parses_valid_json_response():
    client = FakeAnthropicClient(responses=[_text_response(VALID_APPRAISAL_JSON)])
    adapter = AnthropicLLMAdapter(client, model="test-model")

    result = adapter.extract_appraisal(_observation(), _goal_state())

    assert result == {
        "goal_relevance": 0.8, "goal_congruence": 0.1, "uncertainty": 0.3, "perceived_control": 0.5,
        "agency": 0.6, "affect_intensity": 0.4, "possible_affect": "mild concern",
        "evidence_tags": ["explicit_statement"], "evidence_strength": "EXPLICIT",
    }
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "test-model"
    sent_messages = client.calls[0]["messages"]
    assert len(sent_messages) == 1
    assert sent_messages[0]["role"] == "user"
    assert "decide whether to proceed" in sent_messages[0]["content"]  # the goal_state's own objective


def test_extract_appraisal_strips_markdown_code_fence():
    fenced = "```json\n" + VALID_APPRAISAL_JSON + "\n```"
    client = FakeAnthropicClient(responses=[_text_response(fenced)])
    adapter = AnthropicLLMAdapter(client)

    result = adapter.extract_appraisal(_observation(), _goal_state())
    assert result is not None
    assert result["evidence_strength"] == "EXPLICIT"


def test_extract_appraisal_returns_none_on_malformed_json():
    client = FakeAnthropicClient(responses=[_text_response("not json at all")])
    adapter = AnthropicLLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_returns_none_on_non_object_json():
    client = FakeAnthropicClient(responses=[_text_response("[1, 2, 3]")])
    adapter = AnthropicLLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_never_raises_on_transport_failure():
    """extract_appraisal's own documented contract (llm/adapter.py's module
    docstring): a transport-level failure is reported the SAME way
    malformed JSON is — by returning None — never by raising."""
    client = FakeAnthropicClient(responses=[ConnectionError("network is down")])
    adapter = AnthropicLLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_returns_none_on_empty_content():
    client = FakeAnthropicClient(responses=[SimpleNamespace(content=[])])
    adapter = AnthropicLLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None


def test_extract_appraisal_repair_flag_reaches_the_prompt():
    """repair=True should produce a DIFFERENT (longer, correction-noted)
    user prompt than repair=False — see llm/prompts.py's own repair note —
    confirmed here by inspecting what the fake client actually received."""
    client = FakeAnthropicClient(responses=[_text_response(VALID_APPRAISAL_JSON), _text_response(VALID_APPRAISAL_JSON)])
    adapter = AnthropicLLMAdapter(client)

    adapter.extract_appraisal(_observation(), _goal_state(), repair=False)
    adapter.extract_appraisal(_observation(), _goal_state(), repair=True)

    prompt_no_repair = client.calls[0]["messages"][0]["content"]
    prompt_repair = client.calls[1]["messages"][0]["content"]
    assert "could not be parsed" not in prompt_no_repair
    assert "could not be parsed" in prompt_repair


def _generation_contract() -> GenerationContract:
    return GenerationContract(
        user_text="hi", task_context={}, raw_history=(), goal_context={"objective": "decide"},
        policy=Policy.INFORM, secondary_policy=None, hard_constraints=[], qualitative_state_cue=None,
    )


def test_generate_returns_text_on_success():
    client = FakeAnthropicClient(responses=[_text_response("Here's some information that's relevant.")])
    adapter = AnthropicLLMAdapter(client, model="test-model", max_tokens=256, temperature=0.2)

    result = adapter.generate(_generation_contract())

    assert result == "Here's some information that's relevant."
    assert client.calls[0]["model"] == "test-model"
    assert client.calls[0]["max_tokens"] == 256
    assert client.calls[0]["temperature"] == 0.2


def test_generate_raises_timeout_error_on_builtin_timeout():
    client = FakeAnthropicClient(responses=[TimeoutError("took too long")])
    adapter = AnthropicLLMAdapter(client)

    with pytest.raises(TimeoutError):
        adapter.generate(_generation_contract())


def test_generate_raises_generator_error_on_other_failure():
    client = FakeAnthropicClient(responses=[ConnectionError("network is down")])
    adapter = AnthropicLLMAdapter(client)

    with pytest.raises(GeneratorError):
        adapter.generate(_generation_contract())


def test_generate_raises_generator_error_on_empty_response():
    client = FakeAnthropicClient(responses=[SimpleNamespace(content=[])])
    adapter = AnthropicLLMAdapter(client)

    with pytest.raises(GeneratorError):
        adapter.generate(_generation_contract())


def test_from_env_raises_clear_error_without_anthropic_package(monkeypatch):
    """This sandbox DOES have the anthropic package installed (Phase 7
    added it to requirements.txt), so this test simulates the "not
    installed" branch directly rather than actually uninstalling anything —
    monkeypatching the module-level soft-import flag llm.adapter itself
    uses to decide whether the real package is available."""
    import llm.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "anthropic", None)
    with pytest.raises(ImportError, match="anthropic"):
        adapter_module.AnthropicLLMAdapter.from_env()


def test_provider_failure_records_sanitized_last_error_and_success_clears_it():
    client = FakeAnthropicClient(
        responses=[
            ConnectionError("Authorization: Bearer sk-supersecret123456789"),
            _text_response(VALID_APPRAISAL_JSON),
        ]
    )
    adapter = AnthropicLLMAdapter(client)

    assert adapter.extract_appraisal(_observation(), _goal_state()) is None
    assert adapter.last_error is not None
    assert "ConnectionError" in adapter.last_error
    assert "supersecret" not in adapter.last_error
    assert "[REDACTED]" in adapter.last_error

    assert adapter.extract_appraisal(_observation(), _goal_state()) is not None
    assert adapter.last_error is None


def test_generation_failure_records_last_error():
    client = FakeAnthropicClient(responses=[ConnectionError("network is down")])
    adapter = AnthropicLLMAdapter(client)

    with pytest.raises(GeneratorError):
        adapter.generate(_generation_contract())

    assert adapter.last_error == "ConnectionError: network is down"
