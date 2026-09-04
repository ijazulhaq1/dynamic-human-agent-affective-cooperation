# Dynamic Affective Cooperation — prototype

Direct implementation of the frozen architecture (`04_Prototype_Specification_Dynamic_Affective_Cooperation_Frozen_PreImplementation.docx`)
per the Implementation Blueprint (`05_Implementation_Blueprint_Dynamic_Affective_Cooperation.docx`).
Built in the phased order the blueprint's §10 specifies — each phase's own
tests must be green before the next phase starts.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | `config/*.yaml`, `models/*.py` (all Pydantic schemas incl. `Turn`) | **done** — `tests/test_schemas.py`, 42/42 green |
| 1 | `services/observation_builder.py`, `derived_features.py`, `state_transition.py` | **done** — `tests/test_observation_builder.py`, `test_derived_state.py`, `test_transition.py`, 39/39 green |
| 2 | `services/policy_engine.py` + `policy_engine_task_focused.py` + Rule table | **done** — `tests/test_policy.py`, `test_policy_task_focused.py`, `test_state_relevance.py`, 119/119 green (whole suite) |
| 3 | `llm/adapter.py` (mock first), `appraisal_estimator.py` | **done** — `tests/test_appraisal_estimator.py`, 134/134 green (whole suite) |
| 4 | `response_generator.py` + fallback templates | **done** — `tests/test_generator_isolation.py`, 182 passed + 6 intentionally skipped (whole suite) |
| 5 | `pipeline.py`, `experiment_controller.py`, `goal_state_manager.py`, `outcome_baseline.py`, `logger.py` | **done** — `tests/test_conditions.py`, `test_outcome_baseline.py`, `test_replay.py`, `test_experiment_controller.py` (+ `test_goal_state_manager.py`, `test_logger.py`), 219 passed + 6 intentionally skipped (whole suite) |
| 6 | `demo_fixture.yaml` tuned to the analytical check, wired through `compare()` | not started |
| 7 | `ui/*` (Streamlit), real `llm/adapter.py` backend | not started |

## Running the tests

```bash
pip install -r requirements.txt
pytest -v
```

## Notes for whoever picks this up next

- Every model, threshold and rule here is a direct restatement of the frozen
  specification/blueprint — the section citation in each file's docstring is
  where to look if something here looks wrong; if code and spec disagree, the
  spec governs.
- `config/*.yaml` thresholds and weights are copied verbatim; nothing in
  `models/` or later phases should ever hardcode a duplicate of a number that
  belongs in these files.
- Phase 1's transition-formula tests do **not** depend on the real interview
  demo scenario. `test_transition.py` is tested against small, synthetic,
  hand-calculated fixtures (pick simple H_t/G_t/D_t values, work the
  transition equations by hand, assert the code matches) — that is better
  science than testing against one unverified fixture anyway, since the
  expected values are independently derivable and don't depend on transcribing
  anything from the specification first. `compute_target`'s fixture loads the
  real weights out of `config/default.yaml` rather than re-typing them, so a
  drift between the config file and the formula fails the test.
- `LinearPersistenceTransition` takes its weights via constructor injection
  (read from `config/default.yaml`'s `transition.weights`) rather than
  hardcoding them as a second copy, the same discipline `AppraisalEstimator`
  (a later phase) applies to `confidence_map` — the blueprint's own pseudocode
  shows the weights as literals for readability, but the config file is the
  single source of truth per its own header comment.
- `LinearPersistenceTransition.apply()` validates `0 <= rho <= 1` itself and
  raises `ValueError` outside that range — ordinary defensive coding, not a
  scientific/architectural decision (the same category as `TurnRecord`'s
  Pydantic `rho` field bound), needed here because `rho` is a plain float
  parameter until `ExperimentController.resolve_rho` (Phase 5) is the sole
  caller.
- `derive_interaction_state()` validates every focal-option impact q_j to
  [-1,1] and rejects a non-numeric impact, before the d_goal formula runs —
  not after, via clip01(). clip01() bounds a valid formula's output; it must
  never be relied on to silently repair invalid scenario input
  (`{"a": 2.0, "b": -2.0}` previously passed through and produced an
  indistinguishable-from-legitimate `goal_conflict=1.0`).
- `required_evidence[i].status` is restricted to a canonical vocabulary
  (`VALID_EVIDENCE_STATUSES` in `services/derived_features.py`: `resolved`,
  `missing`, `contradictory`, `unresolved`) rather than treating anything
  outside `{missing, contradictory, unresolved}` as resolved by default — a
  typo'd status (e.g. `"typo"`) now raises `ScenarioConfigError` instead of
  silently producing `evidence_ambiguity=0.0`. Use `"resolved"` (not
  `"confirmed"` or any other spelling) for a satisfied evidence requirement
  in every later phase and in Phase 6's demo fixture.
- **Phase 2 rule-table operationalizations were reviewed before freeze.**
  The blueprint's own §20.7/§20.11.1 table cells are compressed English, not
  exact formulas ("INFORM if evidence else CLARIFY", "CLARIFY if unc/amb else
  INFORM", "ACKNOWLEDGE if independently eligible/eligible", "INFORM if
  required", "task requires evidence"/"task requires factual evidence").
  `PolicyEngine.select()`'s own signature (h_t, c_t, g_t, d_t, a_t, a_star —
  no o_t/task_context) rules out reading `O_t.task_context` directly, so
  `services/policy_engine.py`'s module docstring resolves all five from G_t/D_t
  only: "evidence available" = `task_requires_evidence(ctx) AND
  D_t.evidence_ambiguity` below the `clarify.d_amb_min` gate — BOTH a real
  required_evidence structure was represented this turn AND it is
  sufficiently resolved (implementation-review fix: this originally checked
  evidence_ambiguity alone, which meant a turn with no required_evidence at
  all — evidence_ambiguity=0.0 via the task_rule/no-signal default — read as
  "evidence available" purely because ambiguity happened to be low; "no
  evidence was asked for" and "evidence exists and is resolved" are not the
  same construct, and R_SAFETY/TF_SAFETY's secondary now distinguishes them);
  "task requires evidence" = `D_t.ambiguity_source ==
  "required_evidence_formula"`; "unc/amb" reuses R_CLARIFY's own h_unc/d_amb
  gates; "eligible"/"independently eligible" = R_ACK's own condition
  evaluated on the same Ctx. These are documented, defensible choices, not
  verbatim from the frozen specification. **Reviewed before freeze**: this
  review is what caught the `evidence_available()` conflation documented
  immediately below (Phase 2's fix round) — the operationalizations above
  reflect the reviewed, post-fix state.
- `NO_UNDUE_INFLUENCE` is an unconditional hard constraint, always in
  `PolicyState.hard_constraints` (implementation-review fix: previously
  branched on `GoalState.no_undue_influence`, a plain bool a researcher could
  set `False` to make the safeguard silently disappear). `GoalState.
  no_undue_influence` is now `Literal[True]` — constructing a `GoalState`
  with `no_undue_influence=False` is a `ValidationError`, so the model and
  the policy layer tell the same story about this being a hard, always-on
  normative safeguard alongside safety and autonomy.
- `PolicyConfig` (in `services/policy_engine.py`) is constructor-injected from
  `config/default.yaml`'s `policy_thresholds` + `goal_state.autonomy_high_threshold`
  — same single-source-of-truth discipline as `LinearPersistenceTransition`'s
  weights and `AppraisalEstimator`'s `confidence_map`.
- `config/policy_rules.yaml`'s header comment promises "a later-phase test,
  test_policy.py, checks this file and the code stay in sync" — that's
  `test_affect_rule_table_matches_policy_rules_yaml` /
  `test_task_focused_rule_table_matches_policy_rules_yaml`, which assert rule
  ID order (and `reads_a_t` for the affect table) match the executable
  `build_affect_rules`/`build_task_focused_rules` lists exactly. If a rule is
  ever added/reordered in one, this fails fast rather than the two silently
  drifting apart.
- **Phase 3's two documented gaps were reviewed before freeze** (same
  treatment as Phase 2's rule-table calls). (1) The blueprint never says
  whether `LLMAdapter.extract_appraisal` should raise on a transport failure
  (network error/timeout) or fold that into the same outcome as malformed
  JSON — `AppraisalEstimator.estimate()`'s own pseudocode (§7.1/§20.3) has no
  try/except at all, only a `parsed is None` branch. The contract adopted in
  `llm/adapter.py`: an `LLMAdapter` implementation must **never raise** from
  `extract_appraisal` — a transport failure is reported by returning `None`,
  exactly like malformed JSON, so `estimate()` stays a literal transcription
  of the blueprint's own code rather than needing exception-handling the
  blueprint never shows. Correspondingly, `AppraisalEstimator` never returns
  `EstimatorStatus.ERROR` (that value exists on the enum but the estimator's
  pseudocode never returns it) — `TurnRecord.appraisal`'s own note ("None
  only on an unrecoverable estimator error") implies `ERROR` belongs to a
  higher layer (Pipeline, Phase 5) catching something more catastrophic than
  this class is designed to handle. Reviewed and accepted as-is; transport-
  specific exception handling can be revisited when the real API-backed
  adapter is built (Phase 7). (2) `_validate()`'s method body is never
  shown in the blueprint — only its `None`-vs-`(h_t, evidence_strength)`
  usage is. Implemented in `services/appraisal_estimator.py` as a direct
  attempt to build a `HumanAppraisal` from a raw dict using exactly H_t's own
  field names (§5.2's table) — no second, differently-named JSON schema is
  invented — rejecting (returning `None`) on a non-dict raw, a missing
  field, an out-of-range value, or an `evidence_strength` string that isn't
  a real `EvidenceStrength` member.
- `AppraisalEstimator.estimate()`'s fallback path returns
  `FALLBACK_APPRAISAL.model_copy(deep=True)`, never the class-level constant
  itself (implementation-review fix: `HumanAppraisal` is a plain mutable
  `BaseModel`, so returning the same object on every fallback would let one
  turn's `h_t.uncertainty = ...` corrupt the "exact, deterministic fallback
  vector" for every other turn/run that hits the fallback path afterward —
  the blueprint requires the fallback's VALUES be a constant, not that every
  caller share one mutable instance).
- `llm/adapter.py`'s `MockLLMAdapter` is this project's own scriptable test
  double (queues of canned responses + call recording for both
  `extract_appraisal` and, as of Phase 4, `generate`), **not** the
  demo-fixture-replay `MockAdapter` the blueprint's §6.8 prose describes for
  offline rehearsal — that one replays `demo_fixture.yaml`'s frozen
  H_t/c_t/D_t values, which don't exist in this repo yet (Phase 6).
- **Phase 4 documented implementation choices were reviewed before freeze.**
  The blueprint gives `ResponseGenerator.generate()`'s own body verbatim
  (§6.6/§20.9) but never shows `GenerationContract`'s field types,
  `QUALITATIVE_CUES`'s actual dict contents (only one fragmentary example,
  "engagement has grown across the conversation"), `FallbackTemplates`'s
  body or any per-Policy wording, `GeneratorError`'s definition,
  `_relevant()`/`_relevant_goal_fields()`'s bodies, or whether `generate()`
  is expected to raise (as opposed to `extract_appraisal`, which the
  blueprint's own Phase 3 pseudocode never wraps in try/except). The
  decisions below were reviewed and, except for one fix, accepted as-is;
  kept here as audit documentation:
  - `generate()` DOES raise on failure (`TimeoutError` or the new
    `GeneratorError`, defined in `llm/adapter.py`) — the deliberate
    opposite of `extract_appraisal`'s never-raises contract, directly
    evidenced by `ResponseGenerator.generate()`'s own explicit
    `except (TimeoutError, GeneratorError):` clause, which
    `AppraisalEstimator.estimate()` never has.
  - `GenerationContract` (`llm/adapter.py`) is a plain class, not a
    Pydantic `BaseModel` — nothing in it needs range validation or JSON
    round-tripping; it exists only to cross one in-process call.
  - `QUALITATIVE_CUES` (`services/response_generator.py`) is an
    authored-for-this-prototype dict covering all eight `RationaleCode`
    members (though only five — `REDIRECT_ELIGIBLE`, `CLARIFY_NEEDED`,
    `INFORM_NEEDED`, `ACKNOWLEDGE_ELIGIBLE`, `MINIMAL_SUPPORT` — can ever
    actually appear when `state_did_influence_policy=True`, per
    `PolicyEngine._reached_a_t_sensitive_rule`'s reachability). Every
    entry is non-numeric by construction; a test asserts no digit appears
    in any of them. **A researcher should review/replace this wording
    before a real study — it was not transcribed from any spec.**
  - `_relevant()`/`_relevant_goal_fields()` (`services/response_generator.py`)
    strip only known computation-internal keys
    (`d_goal_override`/`d_amb_override`/`required_evidence` from
    `task_context`; `goal_version`/`update_source`/`no_undue_influence`
    from `GoalState`) and pass everything else through unchanged.
    `_relevant()` keeps its `g_t` parameter (matching the blueprint's exact
    call-site signature) even though this implementation doesn't filter by
    its content — no field-selection rule ties task_context relevance to
    GoalState content anywhere in the blueprint.
  - `FallbackTemplates` (`llm/fallback.py`) wording (one primary template
    + one secondary clause per `Policy`, plus one realized clause per
    `hard_constraints` entry) is entirely authored for this prototype —
    the blueprint specifies only the structural requirement (primary +
    secondary + hard_constraints all represented, never silence). **Also
    needs a researcher's review before a real study.**
  - **Post-delivery fix**: the first version of `FallbackTemplates.render()`
    printed `hard_constraints` as a parenthetical of the raw internal
    codes — e.g. `"(AUTONOMY_HIGH, NO_UNDUE_INFLUENCE noted.)"` — straight
    into participant-facing text. Those are control labels
    (`services/policy_engine.py._hard_constraints()`), never meant to
    reach a participant, and this is the LLM-failure fallback path — the
    output that's supposed to be the most controlled and safety-checked
    the system produces. Fixed via `_HARD_CONSTRAINT_CLAUSES`, a dict
    mapping each known code to an authored natural-language clause
    (`"AUTONOMY_HIGH"` → "The choice remains yours, and I won't push you
    toward a particular option."; `"NO_UNDUE_INFLUENCE"` → "I won't use
    pressure or emotional leverage to influence your decision."). An
    unrecognized hard-constraint code now raises `ValueError` from
    `render()` rather than being silently dropped — fail loudly in testing
    rather than let a safety-relevant constraint go unrealized in a live
    run. Every code `_hard_constraints()` can currently produce
    (`NO_UNDUE_INFLUENCE` unconditionally, `AUTONOMY_HIGH` conditionally)
    has an entry; a third code would need a clause added here before it
    could reach `render()` without raising.
- **Phase 5 (`pipeline.py`, `experiment_controller.py`, `goal_state_manager.py`,
  `outcome_baseline.py`, `logger.py`) has the largest documented-gap surface of
  any phase so far — reviewed once already for internal consistency, but this
  is the first pass and everything below should be checked against the frozen
  specification directly, not just this blueprint's compressed pseudocode,
  before it is relied on for a real study.** The blueprint gives `Pipeline.
  run_turn()`/`compare()`/`ExperimentController`'s `resolve_rho`/`run_turn`/
  `compare`/`start_new_run` verbatim (§6.7), and this implementation
  transcribes all of that field-for-field, including every implementation-
  review fix already baked into the pseudocode itself (history_before
  snapshotted once before any per-condition work; the shared observation
  committed exactly once, after every condition's generator call; TASK_FOCUSED
  never computing A*_t/A_t inside `compare()`'s loop; `resolve_rho` checking
  condition before override; rho_override logged only on the one DYNAMIC
  record it actually changed anything for). Everything else in this phase —
  `GoalStateManager`, `OutcomeBaselineStore`, `TurnLogger`,
  `Pipeline._assemble_turn_record`, `Pipeline.replay_turn`, and three extra
  `Pipeline.__init__` parameters — has no body, and in some cases no class or
  method signature at all, anywhere in the blueprint. Decisions made, all
  documented in-code at their own definitions and summarized here:
  - `GoalStateManager` (`services/goal_state_manager.py`) is a thin,
    single-method wrapper (`update(g_t, source, **changes)`) around
    `GoalState.with_explicit_update()` — its only real job is being the ONE
    class every real-time G_t mutation is meant to go through. It does NOT
    restrict which of `GoalUpdateSource`'s three members a caller passes
    (see the module's own note on why) — a researcher should double-check
    this reading, since "no affective cue alone may call this" is the one
    invariant the blueprint's §6.3 prose is protecting, and this
    implementation currently protects it only by construction elsewhere
    (no estimator/transition/policy code references GoalStateManager at
    all), not by `update()` itself refusing anything.
  - `OutcomeBaselineStore` (`services/outcome_baseline.py`) is a per-run_id
    store with `elicit(run_id, value_priorities)` (raises `ValueError` on a
    second elicitation for the same run_id) and `get(run_id)` (raises
    `KeyError` if never elicited). Added because `TurnRecord.
    outcome_baseline_value_priorities` (§5.7) is REQUIRED on every record
    and nothing in the blueprint's own `Pipeline`/`ExperimentController`
    pseudocode threads `OutcomeBaseline` data anywhere — a caller (a test,
    or Phase 7's UI) must call `pipeline.outcome_baseline_store.elicit(...)`
    (or `ExperimentController.elicit_outcome_baseline(...)`, also an
    addition — see below) once per run, before that run's first turn, or
    every `run_turn()`/`compare()` call for that run_id raises `KeyError`.
  - `TurnLogger` (`services/logger.py`) writes one `record.model_dump_json()`
    line per `persist()` call to an append-only file, and also keeps an
    in-memory `self.records` list (a testing/audit convenience, not in the
    blueprint). The optional SQLite mirror the blueprint mentions is
    deliberately NOT built — no schema is given for it anywhere, same
    treatment this project already gave `demo_fixture.yaml`'s content.
  - `Pipeline.__init__` takes three parameters beyond the blueprint's own
    shown seven (`observation_builder, estimator, transition, policy_engine,
    generator, goal_manager, logger`): `outcome_baseline_store`, `model_id`,
    and `config_hash` (plus optional `model_version`) — each because a
    REQUIRED `TurnRecord` field (§5.7) has no other source anywhere in the
    given pseudocode. `goal_manager` is stored but never called by `Pipeline`
    itself, matching §7 step 2's own wording ("`GoalStateManager.update`...
    never called from the estimator or policy layer") — G_t is expected to
    be read/updated by the caller before being passed into `run_turn()`/
    `compare()` as `g_t`.
  - `compute_config_hash(config)` (`services/pipeline.py`) is a SHA-256 hex
    digest of `json.dumps(config, sort_keys=True)` — the blueprint's
    technology-stack table (§2) says only "`config_hash` is a required
    TurnRecord field" for "reproducible conditions," with no algorithm or
    input scope specified. The caller decides what to merge into the config
    dict before hashing (this repo's tests hash `default.yaml` alone).
  - `Pipeline._assemble_turn_record` — called with exactly the kwargs the
    blueprint's own `run_turn()`/`compare()` pseudocode shows
    (`record_id, comparison_id, run_id, o_t, h_t, evidence_strength, c_t,
    g_t, d_t, a_prev, a_star, a_t, rho, p_t, r_t, estimator_status,
    gen_status, condition, interventions`), but its BODY is never given.
    Every `TurnRecord` field not in that list is filled in and documented
    at the method's own definition: `turn_id`/`timestamp` come from `o_t`;
    `model_id`/`model_version`/`config_hash` come from the `Pipeline`
    instance; `outcome_baseline_value_priorities` is looked up from
    `outcome_baseline_store` by `run_id`; `appraisal` is always populated
    in this implementation (`AppraisalEstimator.estimate()` never returns
    `None`, so the "`None` only on an unrecoverable estimator error" case
    is dead code here); `estimator_latency_ms`/`generator_latency_ms` are
    measured with `time.perf_counter()` around the estimator/generator
    calls; `state_delta` (no formula given anywhere in the blueprint) is
    implemented as the per-field `a_t - a_prev` difference across all three
    `AgentState` dimensions, `None` whenever either side is `None`;
    `observed_choice`/`observed_confidence` are read off `o_t` (and
    threaded through as new optional trailing kwargs on `run_turn()`/
    `compare()`, since `ObservationBuilder.build()` has always accepted
    them but the blueprint's own `run_turn()`/`compare()` signatures never
    forward them); `error_messages` is always `[]`; `replay_parent_turn_id`
    is `None` except from `replay_turn()`. **`state_delta`'s formula in
    particular is this project's own invention and should be checked
    against the frozen specification.**
  - `Pipeline.replay_turn(turn_id, condition, rho=None)` has a docstring in
    the blueprint ("Reuses the stored H_t/c_t/D_t/A*_t for that turn_id (or,
    if the live estimator is unavailable, the frozen fallback fixture,
    §19.9)") but no body. This implementation has exactly one path: it
    reuses the ORIGINAL `(turn_id, condition)` record's own stored
    H_t/c_t/D_t/A*_t (kept in a `Pipeline`-internal cache populated by
    `run_turn()`/`compare()`, cleared by `reset()`) and re-runs only
    persistence-apply → policy-select → generate, optionally at a different
    `rho`. The `demo_fixture.yaml`-backed offline-fallback branch the
    blueprint's docstring mentions is **not implemented** — that fixture
    doesn't exist until Phase 6, and "live estimator unavailable" has no
    defined trigger anywhere in the blueprint either. `replay_turn()` never
    calls `_commit_shared_observation_once`/`_commit_condition_state` — a
    replay must not advance state the next real turn would read.
  - `ExperimentController.replay_turn` and `.elicit_outcome_baseline` are
    both additions with no explicit pseudocode line — added because §6.7's
    own opening sentence ("ExperimentController... is the only caller of
    Pipeline... UI code never calls the services directly") would otherwise
    be contradicted by a caller needing to reach `Pipeline.replay_turn()` or
    the outcome-baseline elicitation directly.
- **Phase 5 fix round (post-delivery review): two real bugs found and fixed
  in `replay_turn`, plus three smaller corrections.**
  1. **CURRENT_CUE replay could become persistent.** `Pipeline.replay_turn()`
     only forced `rho=0.0` for `TASK_FOCUSED`; for `CURRENT_CUE` it fell
     through to the `else` branch and applied whatever `rho` a caller
     supplied — so `pipeline.replay_turn(turn_id, Condition.CURRENT_CUE,
     rho=0.9)` genuinely applied `rho=0.9` to a CURRENT_CUE replay, directly
     violating the frozen specification's "no override can ever make
     CURRENT_CUE persistent" guarantee that `resolve_rho`'s own
     condition-before-override ordering exists to enforce everywhere else.
     Fixed at both layers: `Pipeline.replay_turn()` now runs its candidate
     rho through `self._rho_for(condition, ...)` — the same zero-for-
     non-DYNAMIC rule `compare()`'s loop already uses — before ever applying
     it, and `ExperimentController.replay_turn()` now resolves a supplied
     rho through `resolve_rho()` before delegating, matching `run_turn()`/
     `compare()`'s own contract. Covered by
     `test_replay_current_cue_always_zeroes_rho` in both
     `tests/test_replay.py` (Pipeline layer) and
     `tests/test_experiment_controller.py` (ExperimentController layer).
  2. **Replay records were never persisted.** `replay_turn()` built a new
     `TurnRecord` (with `replay_parent_turn_id` set) and returned it, but
     never called `TurnLogger.persist()` — so a replay left no audit trail,
     contradicting `logger.py`'s own "TurnRecord persistence + replay
     linkage" description (§6.8). Fixed by persisting before returning.
     This is additive, not a mutation of the run's committed state: shared
     history, per-condition A_prev, and the `(turn_id, condition)` replay-
     source cache (which still always points at the ORIGINAL record, so a
     second replay of the same turn replays the original again, never a
     replay-of-a-replay) are untouched. Covered by
     `test_replay_is_persisted_but_does_not_mutate_committed_state`.
  3. `state_delta`'s docstring previously described it loosely as "how much
     did persistence move A_t this turn," which reads as the within-turn
     persistence contribution `A_t - A*_t`. The actual formula
     (`a_t - a_prev`) computes the TURN-TO-TURN change `A_t - A_{t-1}` — a
     different quantity. The formula is unchanged (the blueprint gives none,
     so there is nothing to be "more correct" against); only the docstring
     is corrected, to prevent a later statistical analysis from reading the
     field as something it isn't.
  4. `compare()` now rejects a duplicate or empty `conditions` list
     (`ValueError`) before doing any work. Nothing in the blueprint's own
     pseudocode validates this argument, but `[DYNAMIC, DYNAMIC]` would let
     the second occurrence's branch read the A_prev the first occurrence's
     own `_commit_condition_state()` call just wrote earlier in the SAME
     loop — silently contaminating one "condition" with another turn's
     worth of persistence from itself. Recommended defensive correctness,
     not a specification violation, but cheap to make impossible.
  5. `compute_config_hash()`'s docstring now explicitly flags that Phase 5's
     own tests hash `default.yaml` alone (sufficient for this phase's own
     scope) and that Phase 6/7's real entry point MUST hash the complete
     runtime configuration (`default.yaml` + `conditions.yaml` +
     `policy_rules.yaml` + `demo_fixture.yaml` once it exists) or a
     `policy_rules.yaml`-only change between two runs could silently
     produce the same `config_hash` on TurnRecords whose actual rule table
     differed. No code change needed for Phase 5 itself.
  6. `GoalStateManager` accepting all three `GoalUpdateSource` values
     (including `RESEARCHER_CONFIG`) was reviewed and confirmed acceptable
     as-is — not changed.
  - Result: 219 passed + 6 intentionally skipped (up from 214 passed).
- Phase 6's `demo_fixture.yaml` is where the actual frozen demo-turn scenario
  content (Turn 1 / Turn 2 task context, value priorities, etc.) from the
  specification's §19.9/§20.15 belongs, and only there — it hasn't been
  transcribed into this repo yet, is not a dependency of any earlier phase,
  and should be pulled in and tuned to the analytical check when Phase 6 is
  actually reached.
