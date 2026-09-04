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

import json
import os
from typing import Any, Protocol

from models.enums import Policy
from models.goal_state import GoalState
from models.observation import Observation, Turn

try:  # pragma: no cover — exercised implicitly whenever the optional dependency is present
    import anthropic
except ImportError:  # pragma: no cover — exercised only in an environment without the package
    anthropic = None  # AnthropicLLMAdapter.from_env() raises a clear error in this case;
    # AnthropicLLMAdapter itself (dependency-injected client) still works without the
    # real package installed, since every test constructs it with a fake client double.

try:  # pragma: no cover — exercised implicitly whenever the optional dependency is present
    import openai
except ImportError:  # pragma: no cover — exercised only in an environment without the package
    openai = None  # Same soft-import pattern as anthropic above: OpenAILLMAdapter's
    # constructor only needs the real package when it has to build its OWN client
    # (client=None); every test constructs it with a fake client double instead, so
    # importing llm.adapter never requires 'openai' to be installed.


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


class OfflineDemoAdapter:
    """Unlimited, deterministic offline/replay backend for app.py's Mock
    (offline/replay) UI mode (Phase 7) — NOT a test double, and deliberately
    NOT MockLLMAdapter above.

    Post-delivery fix (user review of the first Phase 7 delivery): app.py
    originally used MockLLMAdapter, scripted with exactly as many responses
    as the frozen demo scenario's own two-turn, three-condition run needs.
    MockLLMAdapter's queues are intentionally ONE-SHOT — many other test
    files in this repo rely on its "called more times than it was scripted
    for" AssertionError as a safety net catching an unexpected extra call —
    but that same behavior is wrong for a live UI a researcher can click an
    unbounded number of times: Pipeline.replay_turn() (§20.14) reuses a
    past turn's stored H_t/D_t/A*_t but ALWAYS re-invokes the generator
    (services/pipeline.py's own replay_turn(), built and frozen in Phase 5
    — not something this phase may change), so "run the demo, then replay
    any turn" already calls generate() more times than any finite queue
    scripted for exactly one demo run could ever cover. The exhaustion
    AssertionError would surface as a live crash mid-interview — directly
    contradicting the "offline/replay fallback" being the RELIABLE path.

    Fixed by building a second, purpose-specific adapter instead of
    changing MockLLMAdapter's own exhaustion behavior (which would weaken a
    safety net most of this repo's OTHER tests still want): OfflineDemoAdapter
    never raises "called more times than scripted" at all — it is
    unlimited by construction, so replay, a second demo run without a
    Reset, or any other repeated call sequence a researcher's own clicking
    might produce all stay deterministic and network-free indefinitely.

    - extract_appraisal(): looks up the fixture's own frozen H_t by
      o_t.turn_id in a dict built once at construction — deterministic and
      repeatable for any turn_id in the fixture, any number of times.
      Returns None for a turn_id outside the fixture (defensive; the
      frozen 2-turn demo scenario never triggers this) — extract_appraisal's
      own "None means no usable extraction" contract already makes
      AppraisalEstimator handle that safely (retry, then FALLBACK_APPRAISAL),
      never a crash.
    - generate(): GenerationContract carries no turn_id at all (see this
      module's own GenerationContract docstring), so there is no frozen
      per-turn text to "replay" the way extract_appraisal's H_t is replayed
      — this returns one deterministic, policy-derived placeholder string
      per call instead. Response WORDING was never part of what Phase 6
      analytically verifies (only D_t/A*_t/A_t/policy are); this class
      exists to keep the offline/replay path from crashing, not to author
      a second set of natural-language demo responses.
    """

    def __init__(self, raw_appraisal_by_turn_id: dict[int, dict[str, Any]]) -> None:
        self._raw_appraisal_by_turn_id = dict(raw_appraisal_by_turn_id)

    def extract_appraisal(
        self, o_t: Observation, g_t: GoalState, repair: bool = False
    ) -> dict[str, Any] | None:
        raw = self._raw_appraisal_by_turn_id.get(o_t.turn_id)
        return dict(raw) if raw is not None else None

    def generate(self, contract: GenerationContract) -> str:
        secondary = f" +{contract.secondary_policy.value}" if contract.secondary_policy is not None else ""
        return f"[offline demo response — policy={contract.policy.value}{secondary}, no network call made]"


def _is_timeout(exc: BaseException) -> bool:
    """Shared by AnthropicLLMAdapter.generate() below. Python's builtin
    TimeoutError always counts (matching MockLLMAdapter's own generate_
    responses contract above, and ResponseGenerator's except clause, which
    both already treat builtin TimeoutError as THE timeout signal) — plus,
    when the real `anthropic` package is installed, its own
    anthropic.APITimeoutError, so a genuine network-level timeout from the
    real API is classified the same way a scripted one is in tests."""
    if isinstance(exc, TimeoutError):
        return True
    if anthropic is not None and isinstance(exc, anthropic.APITimeoutError):
        return True
    return False


def _extract_text(response: Any) -> str:
    """Pulls the first text block out of an Anthropic Messages API
    response's own `.content` list (a list of content blocks, each with a
    `.type`; only "text" blocks carry `.text` — a real response could in
    principle include other block types this prototype never requests,
    e.g. tool_use, so this skips anything that isn't text rather than
    assuming content[0] is always text). Raises ValueError (caught by both
    of this adapter's own methods, see below) if no text block is present
    — an empty/non-text response is a generation failure, not a "" success."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if text:
                return text
    raise ValueError("Anthropic response contained no text content block")


def _is_openai_timeout(exc: BaseException) -> bool:
    """OpenAI counterpart to _is_timeout() above — same reasoning, same
    shape, kept as a separate function rather than generalizing _is_timeout
    itself so neither Anthropic's nor OpenAI's already-tested branch has to
    change to accommodate the other provider's exception type."""
    if isinstance(exc, TimeoutError):
        return True
    if openai is not None and isinstance(exc, openai.APITimeoutError):
        return True
    return False


def _extract_openai_text(response: Any) -> str:
    """Pulls the aggregated text out of an OpenAI Responses API response's
    own `.output_text` convenience property (the SDK's own documented
    accessor — see openai.types.responses.response.Response.output_text —
    which aggregates every `output_text` content block across the
    response's `output` list). That property returns "" rather than
    raising when no text content is present (e.g. every output item was a
    refusal, or the response was truncated before any text), so this
    function raises ValueError in that case instead — the same "empty
    response is a generation failure, not a '' success" contract
    _extract_text() (above) already enforces for Anthropic, so both
    providers' adapters fail the same way for the same underlying
    condition."""
    text = getattr(response, "output_text", None)
    if not text:
        raise ValueError("OpenAI response contained no output text")
    return text


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of one JSON object out of `text` — the shape
    build_appraisal_extraction_prompt() (llm/prompts.py) asks the model to
    return. Strips a markdown code fence if the model wrapped its answer in
    one despite being asked not to (```json ... ``` or ``` ... ```) — real
    models do this often enough that treating it as a hard failure would
    throw away perfectly good extractions. Returns None (never raises) for
    anything that isn't a JSON object after that — malformed JSON, a JSON
    array/string/number instead of an object, or empty text — matching
    extract_appraisal's own "None means no usable structured output"
    contract (this module's own docstring); AppraisalEstimator._validate()
    is the layer that checks individual field values/ranges, not this one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class AnthropicLLMAdapter:
    """The real, API-backed LLMAdapter implementation (§2's own technology-
    stack table: "LLM adapter... Provider-agnostic interface... Swappable
    API/local/mock backend"). Phase 3-6 only ever needed MockLLMAdapter;
    this is Phase 7's "real llm/adapter.py backend" (blueprint §10).

    Judgment call (documented gap, flagged for review): the blueprint
    specifies the ADAPTER INTERFACE exactly (extract_appraisal/generate's
    signatures and failure contracts — see LLMAdapter's own docstrings
    above) but never names a specific LLM provider anywhere — "provider-
    agnostic" is the whole point of the interface. Anthropic's Claude API
    was chosen here as the one concrete backend actually built, since this
    prototype's own development environment is Claude-based and the
    `anthropic` Python SDK's Messages API is a well-documented, directly
    testable target — not because the frozen specification names Anthropic
    anywhere. A researcher who wants a different provider (or a local
    model) can implement a second class satisfying the same LLMAdapter
    Protocol; nothing elsewhere in this codebase imports AnthropicLLMAdapter
    by name (AppraisalEstimator/ResponseGenerator only ever depend on the
    LLMAdapter Protocol), so swapping is a one-line change at whichever
    call site constructs the adapter (app.py, Phase 7's own entry point).

    Dependency-injected `client` (constructor parameter, not a global) so
    this class is unit-testable with a fake double — no real API key or
    network access required for tests (see tests/test_anthropic_adapter.py)
    — matching MockLLMAdapter's own zero-network testability above. `client`
    is expected to expose `.messages.create(model=..., max_tokens=...,
    temperature=..., system=..., messages=[...])` returning an object with
    a `.content` list of blocks (see _extract_text above) — exactly
    anthropic.Anthropic()'s own real shape, so a fake double in tests
    exercises the SAME parsing code a real response would.
    """

    # Judgment call (documented gap, flagged for review): a model identifier
    # is not, and cannot be, "frozen" the way config/default.yaml's numeric
    # thresholds are — provider model strings change over time independent
    # of anything in this codebase. This default is a reasonable current
    # choice, but a researcher should treat it as configuration, not a
    # constant: override it via the `model` constructor parameter or
    # from_env()'s own `model` argument, not by editing this line.
    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        """Public accessor for the configured model id — app.py (Phase 7)
        reads this to populate TurnRecord.model_id when constructing a
        Pipeline around this adapter, rather than reaching into the
        otherwise-private self._model directly."""
        return self._model

    @classmethod
    def from_env(cls, model: str = DEFAULT_MODEL, **kwargs: Any) -> "AnthropicLLMAdapter":
        """Constructs a REAL, network-calling adapter: `anthropic.Anthropic()`
        reads ANTHROPIC_API_KEY from the environment itself (the SDK's own
        documented behavior, not reimplemented here). Lazy-imported inside
        this method (module-level import above is already soft/optional) so
        importing llm.adapter — and constructing AnthropicLLMAdapter directly
        with an injected client, as every test in this repo does — never
        requires the `anthropic` package or any credential to be present;
        only this one specific entry point does."""
        if anthropic is None:
            raise ImportError(
                "AnthropicLLMAdapter.from_env() requires the 'anthropic' package "
                "('pip install anthropic') and an ANTHROPIC_API_KEY environment variable."
            )
        return cls(anthropic.Anthropic(), model=model, **kwargs)

    def extract_appraisal(
        self, o_t: Observation, g_t: GoalState, repair: bool = False
    ) -> dict[str, Any] | None:
        """Never raises (see LLMAdapter.extract_appraisal's own docstring
        above) — malformed JSON, a missing field (checked one layer up, by
        AppraisalEstimator._validate(), not here), AND any transport-level
        failure (network error, auth failure, rate limit, timeout) are all
        the same "None" outcome from this method's point of view, exactly
        as MockLLMAdapter's own responses queue already models with a bare
        `None` entry."""
        from llm.prompts import build_appraisal_extraction_prompt  # local import: see llm/prompts.py's

        # own module docstring for why prompts.py cannot import GenerationContract from this module
        # at the top level (circular import) — this module importing prompts.py locally, only inside
        # the two methods that need it, keeps the dependency one-directional at call time too, not
        # just in the type-checking-only annotation prompts.py uses for its own imports.
        system, user = build_appraisal_extraction_prompt(o_t, g_t, repair=repair)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = _extract_text(response)
        except Exception:
            return None
        return _parse_json_object(text)

    def generate(self, contract: GenerationContract) -> str:
        """Raises TimeoutError or GeneratorError on failure (the deliberate
        OPPOSITE of extract_appraisal's never-raises contract — see this
        module's own docstring for why). ResponseGenerator is the layer
        that catches both and falls back to FallbackTemplates; this method
        itself must not swallow either."""
        from llm.prompts import build_generation_prompt  # local import — see extract_appraisal's own note

        system, user = build_generation_prompt(contract)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = _extract_text(response)
        except Exception as exc:
            if _is_timeout(exc):
                raise TimeoutError(str(exc)) from exc
            raise GeneratorError(str(exc)) from exc
        return text


class OpenAILLMAdapter:
    """A second, optional real LLMAdapter implementation — OpenAI's
    Responses API — added alongside AnthropicLLMAdapter above, at the
    user's own request, as a live backend they can rehearse with using a
    key they already hold. Satisfies the exact same LLMAdapter Protocol
    (extract_appraisal/generate, same signatures, same failure contracts)
    as AnthropicLLMAdapter, and is otherwise a strict sibling of it, not a
    replacement: AnthropicLLMAdapter is unchanged, app.py's Mock/Offline
    path is unchanged, and NOTHING about H_t -> D_t -> A*_t -> A_t -> P_t
    changes — this class only ever sits at the same two boundaries
    AnthropicLLMAdapter already sat at (extract_appraisal, generate), and
    is invisible to Pipeline/AppraisalEstimator/ResponseGenerator/
    PolicyEngine/StateTransition, all of which depend on the LLMAdapter
    Protocol only, never on a concrete adapter class by name.

    Same-prompt guarantee (explicit user requirement — "do NOT create
    different scientific prompts for OpenAI and Anthropic"): both methods
    below call the exact same llm/prompts.py functions
    (build_appraisal_extraction_prompt/build_generation_prompt)
    AnthropicLLMAdapter calls, with no OpenAI-specific wording branch
    anywhere in this class or in prompts.py. The two providers differ only
    in TRANSPORT (Responses API vs. Messages API) and, for appraisal
    extraction, in HOW compliance with the shared schema is enforced (see
    extract_appraisal's own docstring below) — never in what is being
    asked.

    Dependency-injected `client` (matching AnthropicLLMAdapter's own
    testability — see tests/test_openai_adapter.py, which never touches
    the real network or the `openai` package's own runtime behavior).
    Unlike AnthropicLLMAdapter, `client` defaults to None: when the caller
    doesn't inject one, the constructor builds a real `openai.OpenAI()`
    itself, which reads OPENAI_API_KEY from the environment the same way
    `anthropic.Anthropic()` reads ANTHROPIC_API_KEY — this class has no
    equivalent of AnthropicLLMAdapter's separate from_env() classmethod
    for that reason; a from_env() is still provided below for symmetry and
    because app.py's own backend-construction call sites use it, but the
    plain constructor already covers the "no client, use the environment"
    case per the user's own explicit instruction. `model` defaults to None
    too, resolved to OPENAI_MODEL (environment) or DEFAULT_MODEL, in that
    order — never hardcoded, and never required to be passed at all.
    """

    # Judgment call (documented gap, flagged for review), same reasoning as
    # AnthropicLLMAdapter.DEFAULT_MODEL above: not "frozen" the way
    # config/default.yaml's numeric thresholds are — a reasonable current,
    # override-only default (gpt-4o-mini: cheap, fast, and supports
    # Structured Outputs with strict=True, which extract_appraisal below
    # requires). Override via the `model` constructor parameter, the
    # OPENAI_MODEL environment variable, or from_env()'s own `model`
    # argument — never by editing this line and treating it as load-bearing.
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        client: Any = None,
        model: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        if client is None:
            if openai is None:
                raise ImportError(
                    "OpenAILLMAdapter requires the 'openai' package ('pip install openai') "
                    "and an OPENAI_API_KEY environment variable when no client is injected."
                )
            client = openai.OpenAI()  # reads OPENAI_API_KEY from the environment itself —
            # the SDK's own documented behavior, not reimplemented here (never hardcode a key).
        self._client = client
        self._model = model or os.environ.get("OPENAI_MODEL") or self.DEFAULT_MODEL
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature

    @property
    def model(self) -> str:
        """Public accessor for the configured model id — app.py reads this
        to populate TurnRecord.model_id, exactly as it already does for
        AnthropicLLMAdapter.model."""
        return self._model

    @classmethod
    def from_env(cls, model: str | None = None, **kwargs: Any) -> "OpenAILLMAdapter":
        """Constructs a REAL, network-calling adapter, explicitly — kept
        for symmetry with AnthropicLLMAdapter.from_env() and because it is
        the clearer call site inside app.py, even though the plain
        constructor above already falls back to the same behavior when
        client=None. Lazy-imported nowhere here since the module-level
        `openai` soft-import above already covers it; only raises if the
        package truly isn't installed."""
        if openai is None:
            raise ImportError(
                "OpenAILLMAdapter.from_env() requires the 'openai' package ('pip install openai') "
                "and an OPENAI_API_KEY environment variable."
            )
        return cls(openai.OpenAI(), model=model, **kwargs)

    def extract_appraisal(
        self, o_t: Observation, g_t: GoalState, repair: bool = False
    ) -> dict[str, Any] | None:
        """Never raises — identical contract to AnthropicLLMAdapter.
        extract_appraisal above, and for the identical reason (this
        module's own docstring): a transport failure, an auth failure, a
        timeout, and a malformed/empty response are all the same "None"
        outcome from this method's point of view.

        Structured Outputs (JSON Schema, strict=True) rather than free-form
        JSON-in-prose (explicit user requirement): APPRAISAL_JSON_SCHEMA
        (llm/prompts.py) is built directly from HumanAppraisal's own field
        table — the SAME field names APPRAISAL_SYSTEM_PROMPT already
        describes in prose for Anthropic, not a second, independently
        authored schema that could drift from it. Structured Outputs makes
        the model's response schema-conformant BY CONSTRUCTION (the API
        rejects/repairs non-conformant output on OpenAI's own side for
        strict=True), so a successfully-returned response is always valid
        JSON shaped like the schema; this method still runs it through
        _parse_json_object() (shared with AnthropicLLMAdapter) rather than
        assuming that, since AppraisalEstimator._validate() one layer up is
        the actual authority on field-level acceptance (range checks,
        EvidenceStrength membership) — this method's job is only to get a
        parseable dict there or return None trying."""
        from llm.prompts import APPRAISAL_JSON_SCHEMA, build_appraisal_extraction_prompt

        system, user = build_appraisal_extraction_prompt(o_t, g_t, repair=repair)
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=system,
                input=user,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "human_appraisal",
                        "schema": APPRAISAL_JSON_SCHEMA,
                        "strict": True,
                    }
                },
            )
            text = _extract_openai_text(response)
        except Exception:
            return None
        return _parse_json_object(text)

    def generate(self, contract: GenerationContract) -> str:
        """Raises TimeoutError or GeneratorError on failure — identical
        contract to AnthropicLLMAdapter.generate above, for the identical
        reason (this module's own docstring). Plain-text output (no `text=`
        schema argument): unlike appraisal extraction, generation has no
        structured shape to enforce — §20.9's causal-isolation rule is
        already structural (GenerationContract carries no a_t/a_star field
        at all, see this module's own docstring, and build_generation_
        prompt()'s own docstring/tests), not something a JSON Schema could
        add or subtract from."""
        from llm.prompts import build_generation_prompt

        system, user = build_generation_prompt(contract)
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=system,
                input=user,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
            )
            text = _extract_openai_text(response)
        except Exception as exc:
            if _is_openai_timeout(exc):
                raise TimeoutError(str(exc)) from exc
            raise GeneratorError(str(exc)) from exc
        return text
