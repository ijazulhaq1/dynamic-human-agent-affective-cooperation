"""Pipeline — M8 orchestration. Blueprint §6.7, spec §20.12.

Implements the ten-step turn execution order (§7 of this blueprint, §20.12
of the frozen spec) exactly: ObservationBuilder.build [step 1] ->
[GoalStateManager.update happens OUTSIDE Pipeline, by the caller, before
g_t is passed in here — step 2 is explicitly "Never called from the
estimator or policy layer" per §7's own table, and Pipeline's own
run_turn()/compare() pseudocode always receives g_t as an already-current
parameter, never calls goal_manager itself; see __init__'s judgment-call
note] -> AppraisalEstimator.estimate [step 3] -> derive_interaction_state
[step 4] -> TransitionModel.compute_target [step 5] -> TransitionModel.
apply [step 6] -> PolicyEngine.select [step 7] -> ResponseGenerator.
generate [step 8] -> TurnLogger.persist [step 9] -> (UI renders the
returned TurnRecord, step 10, a later phase).

run_turn() and compare() are literal transcriptions of the blueprint's own
pseudocode (§6.7) field-for-field, including every implementation-review
fix already baked into that pseudocode: history_before snapshotted once,
before any per-condition work; the shared observation committed exactly
once, AFTER every condition's generator call, so the current turn can
never leak into its own raw_history; TASK_FOCUSED never computing A*_t/A_t
inside compare()'s loop; and rho_override logged as a TurnRecord.
intervention only on the one DYNAMIC record it actually changed anything
for. Two pieces the blueprint calls by name but never gives a body for —
_assemble_turn_record and replay_turn — are original to this
implementation; both are documented in detail at their own definitions
below, along with every additional constructor parameter this class needed
beyond the blueprint's own shown seven-argument signature.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid

from models.agent_state import AgentState, AgentTargetState
from models.derived_state import DerivedInteractionState
from models.enums import Condition, EstimatorStatus, EvidenceStrength, GeneratorStatus
from models.goal_state import GoalState
from models.human_state import HumanAppraisal
from models.observation import Observation, Turn
from models.policy import PolicyState
from models.turn_record import TurnRecord
from services.derived_features import derive_interaction_state
from services.observation_builder import ObservationBuilder
from services.outcome_baseline import OutcomeBaselineStore


def compute_config_hash(config: dict) -> str:
    """Judgment call (documented gap, flagged for review): §2's technology-
    stack table says only "config_hash is a required TurnRecord field" for
    "Reproducible conditions" — no hashing algorithm, input scope (one
    config dict vs. every loaded YAML file concatenated), or encoding is
    ever specified. Implemented as a SHA-256 hex digest of the config
    dict's own canonical (sort_keys=True) JSON serialization — deterministic
    across runs and processes for the same content, and changes if any
    threshold/weight/rule in that dict changes. The caller decides what to
    merge into the dict before calling (e.g. default.yaml alone, or
    default.yaml + policy_rules.yaml + conditions.yaml combined) — Pipeline
    itself is agnostic to that choice; it only ever receives the
    already-computed string (see Pipeline.__init__).

    REMINDER for Phase 6/7 integration (post-delivery note, not a Phase 5
    defect): this project's own Phase 5 tests currently hash default.yaml
    ALONE (tests/pipeline_fixtures.py's build_pipeline) — sufficient for
    Phase 5's own scope, since none of its tests vary policy_rules.yaml/
    conditions.yaml/demo_fixture.yaml content. When Phase 6/7 wires a real
    application entry point, config_hash MUST be computed over the
    complete runtime configuration — default.yaml + conditions.yaml +
    policy_rules.yaml + (once it exists) demo_fixture.yaml, merged into one
    dict before calling this function — or a policy_rules.yaml-only change
    between two runs would silently produce the SAME config_hash on two
    TurnRecords whose actual rule table differed, defeating the whole
    "reproducible conditions" purpose §2 gives this field. An integration
    test asserting that changing any one of those files changes the
    resulting hash belongs in Phase 6/7, not here."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Pipeline:
    def __init__(
        self,
        observation_builder: ObservationBuilder,
        estimator,
        transition,
        policy_engine,
        generator,
        goal_manager,
        logger,
        outcome_baseline_store: OutcomeBaselineStore,
        model_id: str,
        config_hash: str,
        model_version: str | None = None,
    ) -> None:
        """Judgment call (documented gap, flagged for review): the
        blueprint's own __init__ signature (§6.7) is (observation_builder,
        estimator, transition, policy_engine, generator, goal_manager,
        logger) — seven parameters, body shown only as "...". Three more
        are added here, each because a REQUIRED TurnRecord field (§5.7) has
        no other source anywhere in the given pseudocode:

        - outcome_baseline_store: populates outcome_baseline_value_
          priorities — see services/outcome_baseline.py's own module
          docstring for why a store, not a bare OutcomeBaseline, is needed
          (one Pipeline instance outlives many runs/run_ids, via reset()).
        - model_id: populates the required model_id field — which LLM/
          adapter identity produced this run's H_t/R_t. Nothing in §6.7's
          pseudocode ever sets it, and it cannot default to anything
          meaningful, since a silent default would misattribute every
          record in a run to the wrong model.
        - config_hash: populates the required config_hash field — see
          compute_config_hash() above; computed once by the caller and
          passed in as a plain string, so Pipeline itself never needs to
          know how config was loaded or merged.

        model_version is also threaded through (TurnRecord's own field is
        optional, `| None = None`) — Phase 5's MockLLMAdapter has no real
        version string, so this stays None for every test in this phase; a
        real adapter (Phase 7) would supply one.

        goal_manager is stored but, matching the blueprint's own turn-
        execution-flow table (§7 step 2, "GoalStateManager.update... never
        called from the estimator or policy layer"), is never called BY
        Pipeline itself — G_t is read/updated by the caller (a future UI/
        orchestration layer) BEFORE g_t is passed into run_turn()/
        compare(), exactly as that table's wording implies. Pipeline holds
        the reference only so callers have one object to construct
        (matching the blueprint's own constructor parameter) rather than
        wiring GoalStateManager somewhere else entirely.
        """
        self._observation_builder = observation_builder
        self._estimator = estimator
        self._transition = transition
        self._policy_engine = policy_engine
        self._generator = generator
        self._goal_manager = goal_manager
        self._logger = logger
        self.outcome_baseline_store = outcome_baseline_store
        self._model_id = model_id
        self._model_version = model_version
        self._config_hash = config_hash

        self._a_prev_by_condition: dict[Condition, AgentState] = {}
        self._history: list[Turn] = []
        # ONE shared list of Turn (non-recursive) — identical raw history is
        # structural, not a promise (§6.1, P7).

        # Judgment call (documented gap, flagged for review): the blueprint
        # never says how replay_turn() (§6.7, body not given — see that
        # method below) gets access to a past turn's stored H_t/c_t/D_t/
        # A*_t/A_t. Every completed run_turn()/compare() record is kept
        # here, keyed by (turn_id, condition) — a plain turn_id alone is
        # NOT enough, since compare() produces one record per condition for
        # the SAME turn_id — purely so replay_turn() has something to reuse
        # without needing a live query back into TurnLogger's JSONL file
        # (or the not-yet-built demo_fixture.yaml offline fallback the
        # blueprint's own replay_turn() docstring mentions — see that
        # method). Cleared by reset(), same as every other piece of
        # per-run state.
        self._records_by_turn_and_condition: dict[tuple[int, Condition], TurnRecord] = {}

    def reset(self) -> None:
        """Called only by ExperimentController.start_new_run()
        (implementation-review fix per the blueprint). Clears every piece
        of state a second rehearsal could otherwise inherit from the
        first: shared history, both conditions' stored A_prev,
        ObservationBuilder's turn counter, and (addition beyond the
        blueprint's own three-item list, for the same reason) this
        Pipeline's own turn_id/condition -> TurnRecord replay cache."""
        self._history.clear()
        self._a_prev_by_condition.clear()
        self._observation_builder.reset_turn_counter()
        self._records_by_turn_and_condition.clear()

    def _commit_shared_observation_once(self, o_t: Observation) -> None:
        """Appends exactly one Turn to the shared history, regardless of
        how many conditions the current call evaluates (§6.1, P7). Called
        once per run_turn() and once per compare() — never once per
        condition inside compare()'s loop. Always called AFTER every
        condition has read this turn's pre-turn history snapshot — never
        before, or the current turn would leak into its own generator
        call."""
        self._history.append(Turn.from_observation(o_t))

    def _commit_condition_state(
        self, condition: Condition, a_t: AgentState | None, p_t: PolicyState, r_t: str
    ) -> None:
        """Stores condition-specific A_t only, in self._a_prev_by_condition.
        P_t and R_t already live inside the TurnRecord persisted by the
        caller; this method never touches self._history."""
        if a_t is not None:
            self._a_prev_by_condition[condition] = a_t

    @staticmethod
    def _rho_for(condition: Condition, rho: float) -> float:
        """Same zero-for-non-DYNAMIC rule as ExperimentController.
        resolve_rho, applied per-condition inside one compare() call so a
        single caller-supplied rho can never leak into CURRENT_CUE or
        TASK_FOCUSED's branch of the same comparison."""
        return rho if condition is Condition.DYNAMIC else 0.0

    def run_turn(
        self,
        run_id: str,
        user_text: str,
        task_context: dict,
        condition: Condition,
        rho: float,
        g_t: GoalState,
        rho_override: float | None = None,
        observed_choice: str | None = None,
        observed_confidence: float | None = None,
    ) -> TurnRecord:
        """rho is trusted as already-resolved for this condition — callers
        go through ExperimentController.resolve_rho, never invent rho
        here. rho_override is passed through ONLY so it can be logged
        (§13.1) — it plays no role in computing anything.

        observed_choice/observed_confidence: not in the blueprint's own
        run_turn() signature (§6.7) — added here as optional trailing
        kwargs, forwarded straight to ObservationBuilder.build(), which
        has always accepted them (Phase 1) — TurnRecord has always had
        fields for them too (§5.7) — but nothing in the given pseudocode
        ever threads a value from a run_turn() caller through to the
        builder. Without this, both fields would be permanently
        unreachable through the one real entry point that produces
        TurnRecords one turn at a time.
        """
        history_before = tuple(self._history)  # pre-turn snapshot — the ONLY
        # history this turn's O_t/estimator/generator see

        o_t = self._observation_builder.build(  # step 1
            user_text, task_context, history_before,
            observed_choice=observed_choice, observed_confidence=observed_confidence,
        )

        t0 = time.perf_counter()
        h_t, evidence_strength, c_t, estimator_status = self._estimator.estimate(o_t, g_t)  # step 3
        estimator_latency_ms = (time.perf_counter() - t0) * 1000.0
        # H_t/c_t are now ALWAYS computed, TASK_FOCUSED included: dashboard
        # visibility for every condition; PolicyEngine still structurally
        # never reads h_t/c_t for TASK_FOCUSED (§6.5), so this cannot leak
        # into TASK_FOCUSED's policy output.

        d_t = derive_interaction_state(o_t, g_t)  # step 4

        a_star = a_t = a_prev = None
        if condition is not Condition.TASK_FOCUSED:
            a_star = self._transition.compute_target(h_t, g_t, d_t)  # step 5
            a_prev = self._a_prev_by_condition.get(condition)
            a_t = self._transition.apply(  # step 6
                a_prev, a_star, rho, is_first_affect_enabled_turn=a_prev is None
            )

        p_t = self._policy_engine.select(condition, h_t, c_t, g_t, d_t, a_t, a_star)  # step 7

        t1 = time.perf_counter()
        r_t, gen_status = self._generator.generate(o_t, g_t, history_before, p_t)  # step 8, pre-turn only
        generator_latency_ms = (time.perf_counter() - t1) * 1000.0

        interventions = (
            [{"type": "rho_override", "value": rho}]
            if (rho_override is not None and condition is Condition.DYNAMIC)
            else []
        )  # DYNAMIC only — resolve_rho already refuses the override for any other condition

        record = self._assemble_turn_record(
            record_id=str(uuid.uuid4()), comparison_id=None,
            run_id=run_id, o_t=o_t, h_t=h_t, evidence_strength=evidence_strength,
            c_t=c_t, g_t=g_t, d_t=d_t, a_prev=a_prev, a_star=a_star, a_t=a_t,
            rho=rho, p_t=p_t, r_t=r_t, estimator_status=estimator_status,
            gen_status=gen_status, condition=condition, interventions=interventions,
            estimator_latency_ms=estimator_latency_ms, generator_latency_ms=generator_latency_ms,
        )
        self._logger.persist(record)  # step 9
        self._commit_shared_observation_once(o_t)  # AFTER generation, so this
        # turn never appears in its own history_before
        self._commit_condition_state(condition, a_t, p_t, r_t)
        self._records_by_turn_and_condition[(o_t.turn_id, condition)] = record
        return record  # step 10 (UI renders it)

    def compare(
        self,
        run_id: str,
        user_text: str,
        task_context: dict,
        g_t: GoalState,
        conditions: list[Condition],
        rho: float,
        rho_override: float | None = None,
        observed_choice: str | None = None,
        observed_confidence: float | None = None,
    ) -> dict[Condition, TurnRecord]:
        """The live-demo primitive (§19.9/§20.15): H_t/c_t/D_t are computed
        ONCE and shared by every condition, TASK_FOCUSED included. A*_t is
        ALSO computed once — but only CURRENT_CUE/DYNAMIC ever apply the
        transition to it; TASK_FOCUSED gets a_prev=a_star=a_t=None and
        rho=0 in the loop below, exactly like run_turn()'s own branch. This
        is what makes CURRENT_CUE vs DYNAMIC a true replay of one turn
        rather than two independently-estimated runs — every condition
        below reads the SAME pre-turn history_before that run_turn() would
        have used for this same input.

        observed_choice/observed_confidence: same addition as run_turn(),
        for the same reason (see that method's docstring).

        Post-delivery fix: conditions must be non-empty and duplicate-free.
        Nothing in the blueprint's own pseudocode validates its `conditions`
        argument, but a repeated condition (e.g. [DYNAMIC, DYNAMIC]) would
        let the second occurrence's branch read the A_prev the FIRST
        occurrence's own _commit_condition_state() call just wrote inside
        this same loop — silently contaminating one "condition" with
        another turn's worth of persistence from itself, which is exactly
        the kind of cross-branch leakage the rest of this method's design
        (one shared history_before, one shared H_t/D_t/A*_t, independent
        per-condition A_prev) works to prevent. Rejected eagerly, before
        any of the loop's side effects run, rather than left as a caller
        error the UI is trusted never to make.
        """
        if not conditions:
            raise ValueError("compare() requires at least one condition")
        if len(conditions) != len(set(conditions)):
            raise ValueError(f"compare() conditions must be unique, got {conditions!r}")

        history_before = tuple(self._history)  # pre-turn snapshot, shared by every branch

        o_t = self._observation_builder.build(
            user_text, task_context, history_before,
            observed_choice=observed_choice, observed_confidence=observed_confidence,
        )

        t0 = time.perf_counter()
        h_t, evidence_strength, c_t, estimator_status = self._estimator.estimate(o_t, g_t)
        estimator_latency_ms = (time.perf_counter() - t0) * 1000.0

        d_t = derive_interaction_state(o_t, g_t)

        needs_a_star = any(c is not Condition.TASK_FOCUSED for c in conditions)
        a_star_shared = self._transition.compute_target(h_t, g_t, d_t) if needs_a_star else None
        # Skips a wasted compute_target() call when compare() is given ONLY
        # TASK_FOCUSED — never a correctness issue, since TASK_FOCUSED never
        # reads a_star_shared below regardless.

        comparison_id = str(uuid.uuid4())  # groups every record below as one comparison
        results: dict[Condition, TurnRecord] = {}

        for condition in conditions:
            if condition is Condition.TASK_FOCUSED:
                r, a_prev, a_star, a_t = 0.0, None, None, None  # never computed for TASK_FOCUSED (§9.1)
            else:
                r = self._rho_for(condition, rho)  # 0.0 unless DYNAMIC
                a_prev = self._a_prev_by_condition.get(condition)
                a_star = a_star_shared
                a_t = self._transition.apply(a_prev, a_star, r, is_first_affect_enabled_turn=a_prev is None)

            p_t = self._policy_engine.select(condition, h_t, c_t, g_t, d_t, a_t, a_star)

            t1 = time.perf_counter()
            r_t, gen_status = self._generator.generate(o_t, g_t, history_before, p_t)  # SAME pre-turn
            # snapshot as every other branch
            generator_latency_ms = (time.perf_counter() - t1) * 1000.0

            interventions = (
                [{"type": "rho_override", "value": r}]
                if (rho_override is not None and condition is Condition.DYNAMIC)
                else []
            )

            record = self._assemble_turn_record(
                record_id=str(uuid.uuid4()), comparison_id=comparison_id,
                run_id=run_id, o_t=o_t, h_t=h_t, evidence_strength=evidence_strength,
                c_t=c_t, g_t=g_t, d_t=d_t, a_prev=a_prev, a_star=a_star, a_t=a_t,
                rho=r, p_t=p_t, r_t=r_t, estimator_status=estimator_status,
                gen_status=gen_status, condition=condition, interventions=interventions,
                estimator_latency_ms=estimator_latency_ms, generator_latency_ms=generator_latency_ms,
            )
            self._logger.persist(record)
            self._commit_condition_state(condition, a_t, p_t, r_t)
            self._records_by_turn_and_condition[(o_t.turn_id, condition)] = record
            results[condition] = record

        self._commit_shared_observation_once(o_t)  # exactly once, AFTER every
        # condition has read history_before — never before the loop, or the
        # branches would see their own current turn
        return results

    def replay_turn(self, turn_id: int, condition: Condition, rho: float | None = None) -> TurnRecord:
        """§20.14: never mutates the original run, and never calls
        _commit_shared_observation_once or _commit_condition_state — a
        replay must not advance the committed state that the next real
        turn will read.

        Judgment call (documented gap, flagged for review): the blueprint
        gives this method a docstring only — "Reuses the stored H_t/c_t/
        D_t/A*_t for that turn_id (or, if the live estimator is
        unavailable, the frozen fallback fixture, §19.9)" — no body. The
        demo_fixture.yaml "frozen fallback fixture" branch it mentions does
        not exist yet in this repo (Phase 6, per this project's standing
        decision not to fabricate its content early — see README) and
        "live estimator... unavailable" has no defined detection trigger
        anywhere in the blueprint either, so that branch is NOT implemented
        here; this method has exactly one path — reusing the ORIGINAL
        record's own stored H_t/c_t/D_t/A*_t, sourced from the
        (turn_id, condition) -> TurnRecord cache this Pipeline keeps in
        self._records_by_turn_and_condition (populated by run_turn()/
        compare(), cleared by reset() — see __init__'s own note). A
        researcher who needs the offline-fallback branch before Phase 6
        exists should treat that as a scope decision, not silently expect
        this method to already cover it.

        "Reuses ... for that turn_id" is read literally: this replays the
        ORIGINAL condition's own H_t/c_t/D_t/A*_t (never a different
        condition's — condition is part of the cache key, not something
        this method translates), re-running only steps 6 onward
        (persistence apply, select, generate). rho is the one input this
        method's own signature lets a caller vary; replaying with a
        different rho and observing how A_t/P_t/R_t change downstream of
        the SAME upstream H_t/D_t/A*_t is what a replay with no new
        estimator input can usefully demonstrate — re-running the
        estimator itself against the original raw_history would not be a
        "replay" of the same evidence, it would be a fresh (and possibly
        different, since a scripted MockLLMAdapter's queue may already be
        consumed) estimation.

        Raises KeyError if no record exists for (turn_id, condition) —
        nothing to replay is a caller bug, not a case to default silently
        on. estimator_latency_ms is recorded as 0.0 (no estimator call
        happens during a replay); generator_latency_ms is measured for
        real, since generate() genuinely runs again.

        Post-delivery fix: a supplied rho is now run back through
        self._rho_for(condition, ...) — the SAME zero-for-non-DYNAMIC rule
        run_turn()/compare() apply via ExperimentController.resolve_rho —
        before it is ever used. Before this fix, a caller could do
        `pipeline.replay_turn(turn_id, Condition.CURRENT_CUE, rho=0.9)` and
        the replay would actually apply rho=0.9 to a CURRENT_CUE turn,
        silently turning a condition the frozen specification requires to
        stay at rho=0 into a partially-persistent one — exactly the same
        class of bug §6.7's own resolve_rho fix (condition checked before
        override) already exists to prevent on the run_turn()/compare()
        path, just not yet extended to replay. ExperimentController.
        replay_turn() now also resolves rho before calling this method, so
        the invariant is enforced at both layers, not only here.
        """
        key = (turn_id, condition)
        try:
            original = self._records_by_turn_and_condition[key]
        except KeyError:
            raise KeyError(f"No TurnRecord to replay for turn_id={turn_id}, condition={condition!r}") from None

        replay_rho = original.rho if rho is None else rho
        replay_rho = self._rho_for(condition, replay_rho)  # structural: only DYNAMIC may ever be nonzero

        if condition is Condition.TASK_FOCUSED or original.a_star is None:
            a_prev = a_star = a_t = None
            replay_rho = 0.0
        else:
            a_prev, a_star = original.a_prev, original.a_star
            a_t = self._transition.apply(a_prev, a_star, replay_rho, is_first_affect_enabled_turn=a_prev is None)

        p_t = self._policy_engine.select(
            condition, original.appraisal, original.c_t, original.goal_state,
            original.derived_state, a_t, a_star,
        )

        t1 = time.perf_counter()
        r_t, gen_status = self._generator.generate(
            original.observation, original.goal_state, tuple(original.observation.raw_history), p_t,
        )
        generator_latency_ms = (time.perf_counter() - t1) * 1000.0

        record = self._assemble_turn_record(
            record_id=str(uuid.uuid4()), comparison_id=None,
            run_id=original.run_id, o_t=original.observation, h_t=original.appraisal,
            evidence_strength=original.evidence_strength, c_t=original.c_t,
            g_t=original.goal_state, d_t=original.derived_state,
            a_prev=a_prev, a_star=a_star, a_t=a_t, rho=replay_rho, p_t=p_t, r_t=r_t,
            estimator_status=original.estimator_status, gen_status=gen_status,
            condition=condition, interventions=[],
            estimator_latency_ms=0.0, generator_latency_ms=generator_latency_ms,
            replay_parent_turn_id=turn_id,
        )
        # Post-delivery fix: replay records were previously constructed and
        # returned but never persisted. logger.py's own module docstring
        # (§6.8) describes TurnLogger as providing "replay linkage" — a
        # replay TurnRecord that's never written anywhere has nothing for
        # replay_parent_turn_id to link TO in the audit trail. Persisting
        # here does NOT violate "never mutates the original run" — that
        # rule (§20.14, this method's own opening docstring line) is about
        # the run's own COMMITTED STATE (_history, _a_prev_by_condition,
        # and — deliberately — the (turn_id, condition) replay-source
        # cache below, which keeps pointing at the ORIGINAL record, not
        # this replay, so a second replay of the same turn always replays
        # the original again, never a replay-of-a-replay). Writing a new,
        # separately-identified audit record is additive, not mutating.
        self._logger.persist(record)
        return record
        # Still no _commit_shared_observation_once / _commit_condition_state
        # / _records_by_turn_and_condition write here, by design — a replay
        # must not advance the state the next REAL turn will read, and must
        # not become replayable itself in place of the original.

    def _assemble_turn_record(
        self,
        *,
        record_id: str,
        comparison_id: str | None,
        run_id: str,
        o_t: Observation,
        h_t: HumanAppraisal,
        evidence_strength: EvidenceStrength | None,
        c_t: float | None,
        g_t: GoalState,
        d_t: DerivedInteractionState,
        a_prev: AgentState | None,
        a_star: AgentTargetState | None,
        a_t: AgentState | None,
        rho: float,
        p_t: PolicyState,
        r_t: str,
        estimator_status: EstimatorStatus,
        gen_status: GeneratorStatus,
        condition: Condition,
        interventions: list[dict],
        estimator_latency_ms: float,
        generator_latency_ms: float,
        replay_parent_turn_id: int | None = None,
    ) -> TurnRecord:
        """Assembles the complete TurnRecord for one condition's turn.
        Named and called exactly as the blueprint's own run_turn()/
        compare() pseudocode shows (§6.7) — record_id, comparison_id,
        run_id, o_t, h_t, evidence_strength, c_t, g_t, d_t, a_prev, a_star,
        a_t, rho, p_t, r_t, estimator_status, gen_status, condition,
        interventions — but this method's own BODY is never given anywhere
        in the blueprint, so this implementation designs it from the
        TurnRecord field table (§5.7) directly. Every field NOT among that
        call-site kwarg list is filled in here, each a documented judgment
        call:

        - turn_id, timestamp: read off o_t (Observation already carries
          both — ObservationBuilder's own job, Phase 1).
        - model_id, model_version, config_hash: not threaded through
          run_turn()/compare()'s pseudocode at all — sourced from this
          Pipeline instance's own constructor (see __init__'s judgment-call
          note for why they were added there).
        - outcome_baseline_value_priorities: looked up from
          self.outcome_baseline_store by run_id (see services/
          outcome_baseline.py) and copied out as a plain dict — "copied
          from OutcomeBaseline, read-only" (§5.7) is read literally: this
          TurnRecord gets its OWN dict, never a live reference into the
          frozen OutcomeBaseline's MappingProxyType.
        - appraisal: h_t is passed straight through as-is. §5.7 says this
          field is "None only on an unrecoverable estimator error" — but
          AppraisalEstimator.estimate() (Phase 3, this project's own
          implementation) never raises and never returns None; a total
          extraction failure returns FALLBACK_APPRAISAL instead (a real,
          non-None HumanAppraisal). So in THIS implementation h_t is always
          a real object by the time it reaches here, and appraisal is
          correspondingly always populated — the None case this field's
          type allows for is dead code given Phase 3's own documented
          contract, not something this method needs to construct.
        - estimator_latency_ms, generator_latency_ms: not in the
          pseudocode kwarg list either — measured by run_turn()/compare()/
          replay_turn() themselves (time.perf_counter() around the
          estimator/generator calls) and passed in here, since §2's
          technology-stack table lists both as required TurnRecord fields
          with no other source given for either.
        - state_delta: no formula is given anywhere in the blueprint for
          this field (only its type, `dict | None`, §5.7). Implemented as
          the per-field difference A_t - A_{t-1} (`a_t - a_prev`) across
          all three AgentState dimensions when both exist, else None.

          Precisely: this is the TURN-TO-TURN change in A_t (this turn's
          persisted state minus the PREVIOUS turn's persisted state for
          the same condition) — NOT the within-turn persistence
          contribution A_t - A*_t (this turn's persisted state minus this
          turn's OWN target). Those are different quantities (post-
          delivery correction — an earlier version of this docstring
          described state_delta loosely as "how much did persistence move
          A_t this turn," which reads as the latter, A_t - A*_t, and would
          mislead a later statistical analysis into computing the wrong
          thing from this field). If a future phase needs the persistence
          contribution itself, that is `a_t - a_star` at assembly time —
          NOT what this field currently holds.

          state_delta is None whenever a_prev or a_t is None: TASK_FOCUSED
          always; also the very first affect-enabled turn of a run, since
          there is no prior A_t to delta against even though A_1 = A*_1 is
          itself non-None. A researcher should confirm this reading
          against the frozen specification before relying on it for
          analysis.
        - observed_choice, observed_confidence: read off o_t (already
          collected by ObservationBuilder.build(), Phase 1 — run_turn()/
          compare() accept and forward them as optional trailing kwargs,
          an addition beyond the blueprint's own shown signatures — see
          those methods' docstrings).
        - error_messages: always [] in this implementation — nothing in
          Phase 5's own pseudocode ever appends to it; kept as an empty
          list per TurnRecord's own default rather than omitted.
        - replay_parent_turn_id: None for run_turn()/compare(); set by
          replay_turn() (see that method) to the original turn_id being
          replayed.
        """
        baseline = self.outcome_baseline_store.get(run_id)

        # state_delta = A_t - A_{t-1} (turn-to-turn change), NOT A_t - A*_t
        # (the within-turn persistence contribution) — see this method's own
        # docstring above for why that distinction matters.
        state_delta = None
        if a_prev is not None and a_t is not None:
            state_delta = {
                "motivational_priority": a_t.motivational_priority - a_prev.motivational_priority,
                "decision_information_priority": (
                    a_t.decision_information_priority - a_prev.decision_information_priority
                ),
                "intervention_readiness": a_t.intervention_readiness - a_prev.intervention_readiness,
            }

        return TurnRecord(
            record_id=record_id,
            comparison_id=comparison_id,
            run_id=run_id,
            turn_id=o_t.turn_id,
            timestamp=o_t.timestamp,
            condition=condition,
            model_id=self._model_id,
            model_version=self._model_version,
            config_hash=self._config_hash,
            outcome_baseline_value_priorities=dict(baseline.value_priorities),
            observation=o_t,
            appraisal=h_t,
            evidence_strength=evidence_strength,
            c_t=c_t,
            goal_state=g_t,
            derived_state=d_t,
            a_prev=a_prev,
            a_star=a_star,
            a_t=a_t,
            rho=rho,
            state_delta=state_delta,
            policy=p_t,
            response_text=r_t,
            estimator_status=estimator_status,
            generator_status=gen_status,
            estimator_latency_ms=estimator_latency_ms,
            generator_latency_ms=generator_latency_ms,
            interventions=interventions,
            observed_choice=o_t.observed_choice,
            observed_confidence=o_t.observed_confidence,
            error_messages=[],
            replay_parent_turn_id=replay_parent_turn_id,
        )
