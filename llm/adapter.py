"""LLMAdapter — provider-agnostic call interface. Blueprint §6.8/§15.1.

"llm/adapter.py: one interface, at least two implementations — a real
API-backed adapter and a MockAdapter" (blueprint §6.8 prose). Only
extract_appraisal (used by services.appraisal_estimator.AppraisalEstimator,
Phase 3) is declared on the Protocol here. generate() (used by
services.response_generator.ResponseGenerator, Phase 4) is added to this
same Protocol when Phase 4 introduces GenerationContract — not built ahead
of that phase's own scope.

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
"""

from __future__ import annotations

from typing import Any, Protocol

from models.goal_state import GoalState
from models.observation import Observation


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


class MockLLMAdapter:
    """Test double for LLMAdapter — Phase 3's own scriptable mock, not the
    demo-fixture-replay MockAdapter the blueprint's §6.8 prose describes for
    offline rehearsal (that one replays demo_fixture.yaml's frozen H_t/c_t/D_t
    values; that fixture doesn't exist in this repo yet — see Phase 6 — so it
    is not built here, per this project's standing decision not to fabricate
    its content early).

    Constructed with a queue of canned responses (dict for a usable
    extraction, None for "no usable output" — malformed JSON or a transport
    failure, indistinguishable at this interface per the module docstring).
    Each call to extract_appraisal pops the next response off the queue and
    records the call (o_t, g_t, repair) so tests can assert exact call
    counts and repair-flag sequencing.
    """

    def __init__(self, responses: list[dict[str, Any] | None]):
        self._responses = list(responses)
        self.calls: list[tuple[Observation, GoalState, bool]] = []

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
