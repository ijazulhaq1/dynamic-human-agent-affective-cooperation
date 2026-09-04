"""LLMAdapter — provider-agnostic call interface. Blueprint §6.8/§15.1.

"llm/adapter.py: one interface, at least two implementations — a real
API-backed adapter and a MockAdapter" (blueprint §6.8 prose). Declares both
methods the interface needs: extract_appraisal (used by
services.appraisal_estimator.AppraisalEstimator, Phase 3) and generate
(used by services.response_generator.ResponseGenerator, Phase 4).

Implementation-review note (documented gap, flagged for review): the
blueprint never specifies whether extract_appraisal should raise on a
genuine transport failure (network error, timeout) or fold that outcome
into the same "returns something _validate rejects" path it uses for
malformed JSON. AppraisalEstimator.estimate()'s own pseudocode (§7.1/§20.3)
has no try/except at all — only a `parsed is None` branch — so the contract
adopted here is that an LLMAdapter implementation must NEVER raise from
extract_appraisal: a transport failure is reported the same way malformed
JSON is, by returning None. This keeps estimate() a literal, unmodified
transcription of the blueprint's own code block rather than requiring
exception-handling the blueprint never shows. EstimatorStatus.ERROR is
correspondingly never produced by AppraisalEstimator itself (see that
module's docstring) — TurnRecord.appraisal's own note ("None only on an
unrecoverable estimator error") implies ERROR belongs to a higher layer
(Pipeline, Phase 5) catching something more catastrophic than this.

generate()'s failure contract is the DELIBERATE OPPOSITE of
extract_appraisal's, and this asymmetry is directly evidenced by the
blueprint's own pseudocode rather than invented: ResponseGenerator.generate()
(§6.6/§20.9) wraps self._llm.generate(contract) in
`except (TimeoutError, GeneratorError):` — an explicit try/except the
estimator's pseudocode never has. So generate() implementations ARE expected
to raise on failure (builtin TimeoutError for a timeout, GeneratorError
below for any other generation failure), and ResponseGenerator is the layer
that catches them and falls back to FallbackTemplates — never the adapter
swallowing the failure into a sentinel return value the way extract_appraisal
does.
"""

from __future__ import annotations

from typing import Any, Protocol

from models.enums import Policy
from models.goal_state import GoalState
from models.observation import Observation, Turn


class GeneratorError(Exception):
    """Raised by an LLMAdapter.generate() implementation to signal a
    non-timeout generation failure (a malformed/empty response, a provider-
    side error, content-policy refusal, etc.). Never defined in the
    blueprint itself — only referenced by name in ResponseGenerator.
    generate()'s except clause (§6.6). Homed here since llm/adapter.py is
    "the provider-agnostic call interface" file (repo-layout comment) and
    no other module is named for it."""


class GenerationContract:
    """The payload passed to LLMAdapter.generate() (§6.6/§20.9). The
    blueprint never gives this a class body or field table — only its
    seven constructor kwargs at the one call site inside
    ResponseGenerator.generate() are shown. Field types below are the most
    direct reading of that call site; see services/response_generator.py
    for how task_context/goal_context are filtered before reaching here.

    A plain class (not a Pydantic BaseModel) deliberately: unlike
    Observation/TurnRecord/etc., nothing here needs range validation or
    JSON (de)serialization — it exists only to cross the one in-process
    call from ResponseGenerator into an LLMAdapter implementation, and
    Turn/GoalState-derived dict content inside it is already validated
    upstream by the models that produced it.
    """

    def __init__(
        self,
        user_text: str,
        task_context: dict[str, Any],
        raw_history: tuple[Turn, ...],
        goal_context: dict[str, Any],
        policy: Policy,
        secondary_policy: Policy | None,
        hard_constraints: list[str],
        qualitative_state_cue: str | None,
    ) -> None:
        self.user_text = user_text
        self.task_context = task_context
        self.raw_history = raw_history
        self.goal_context = goal_context
        self.policy = policy
        self.secondary_policy = secondary_policy
        self.hard_constraints = hard_constraints
        self.qualitative_state_cue = qualitative_state_cue


class LLMAdapter(Protocol):
    def extract_appraisal(
        self, o_t: Observation, g_t: GoalState, repair: bool = False
    ) -> dict[str, Any] | None:
        """Structured JSON only, no free prose (§7.1). Returns a flat dict
        with exactly HumanAppraisal's own field names (goal_relevance,
        goal_congruence, uncertainty, perceived_control, agency,
        affect_intensity, possible_affect, evidence_tags, evidence_strength)
        — the raw JSON schema is not specified in the blueprint beyond H_t's
        own field table (§5.2), so this implementation uses that table
        directly rather than inventing a second, differently-named schema.

        Returns None whenever no usable structured output was obtained —
        malformed/unparseable JSON, a response missing required fields, AND
        any transport-level failure (network error, timeout): see the
        module docstring. Never raises.

        repair=True requests the one-retry "repair" attempt (§20.3) after a
        first attempt this adapter itself could not turn into a valid dict;
        it is the caller's (AppraisalEstimator's) responsibility to decide
        when a retry is warranted, not the adapter's.
        """
        ...

    def generate(self, contract: GenerationContract) -> str:
        """§6.6/§20.9. Returns the response text on success. Raises
        TimeoutError on a timeout, or GeneratorError for any other
        generation failure — see the module docstring for why this is the
        deliberate opposite of extract_appraisal's never-raises contract.
        ResponseGenerator is the layer that catches both and falls back to
        FallbackTemplates; this method itself must not catch or swallow
        them."""
        ...


class MockLLMAdapter:
    """Test double for LLMAdapter — Phase 3's own scriptable mock, not the
    demo-fixture-replay MockAdapter the blueprint's §6.8 prose describes for
    offline rehearsal (that one replays demo_fixture.yaml's frozen H_t/c_t/D_t
    values; that fixture doesn't exist in this repo yet — see Phase 6 — so it
    is not built here, per this project's standing decision not to fabricate
    its content early).

    Constructed with a queue of canned extract_appraisal responses (dict for
    a usable extraction, None for "no usable output" — malformed JSON or a
    transport failure, indistinguishable at this interface per the module
    docstring) and, separately, a queue of canned generate() outcomes (a str
    for a successful response, or an exception instance/class — TimeoutError
    or GeneratorError — to raise, matching generate()'s raises-on-failure
    contract, the opposite of extract_appraisal's). Each call pops the next
    scripted item off its own queue and is recorded (extract_appraisal calls
    as (o_t, g_t, repair); generate calls as the GenerationContract itself)
    so tests can assert exact call counts, repair-flag sequencing, and what
    a generator call actually saw.
    """

    def __init__(
        self,
        responses: list[dict[str, Any] | None] = (),
        generate_responses: list[str | BaseException | type[BaseException]] = (),
    ):
        self._responses = list(responses)
        self._generate_responses = list(generate_responses)
        self.calls: list[tuple[Observation, GoalState, bool]] = []
        self.generate_calls: list[GenerationContract] = []

    def extract_appraisal(
        self, o_t: Observation, g_t: GoalState, repair: bool = False
    ) -> dict[str, Any] | None:
        self.calls.append((o_t, g_t, repair))
        if not self._responses:
            raise AssertionError(
                "MockLLMAdapter.extract_appraisal called more times than it was scripted for "
                f"(had {len(self.calls) - 1} scripted responses, this is call #{len(self.calls)})"
            )
        return self._responses.pop(0)

    def generate(self, contract: GenerationContract) -> str:
        self.generate_calls.append(contract)
        if not self._generate_responses:
            raise AssertionError(
                "MockLLMAdapter.generate called more times than it was scripted for "
                f"(had {len(self.generate_calls) - 1} scripted responses, this is call #{len(self.generate_calls)})"
            )
        outcome = self._generate_responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome()
        return outcome
