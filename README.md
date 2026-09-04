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
| 6 | `demo_fixture.yaml` tuned to the analytical check, wired through `compare()` | **done** — `tests/test_demo_fixture.py`, 225 passed + 6 intentionally skipped (whole suite) |
| 7 | `ui/*` (Streamlit), real `llm/adapter.py` backend (Anthropic + OpenAI) | **fix round reviewed and passed; OpenAI backend added post-approval, same status pending review** — `tests/test_llm_prompts.py`, `test_anthropic_adapter.py`, `test_openai_adapter.py`, `test_state_manager.py`, `test_app_smoke.py`, 285 passed + 6 intentionally skipped (whole suite, including Streamlit `AppTest` — up from 260 passed). Blueprint's own gate is a MANUAL researcher acceptance step — see the Phase 7 notes below, the "Phase 7 fix round" notes, and the "OpenAI support" notes for the full account of what changed and why. |

## Running the tests

```bash
pip install -r requirements.txt
pytest -v
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens with the **Offline** backend selected by default (no API key needed,
no network calls made) — click "Run demo_fixture.yaml (both turns, all three
conditions)" to rehearse the frozen, analytically-verified Phase 6 scenario
end to end. Switch to **OpenAI** or **Anthropic** in the sidebar (and set
`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`, or paste one into the sidebar field —
never displayed, stored, or logged in full) for live, free-text rehearsal
against a real model; the sidebar always shows which backend and model id is
currently active. Whichever live backend has an environment key present is
selected by default (Anthropic taking priority if both are set); with neither
key present, Offline stays the default. See "Phase 7" and "OpenAI support"
below for exactly what this delivery does and does not verify on its own.

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
- **Phase 6 (`config/demo_fixture.yaml`, `services/demo_fixture.py`) — the
  frozen specification does NOT contain a literal §19.9/§20.15 scenario to
  transcribe.** The note that used to sit here (through the Phase 5 delivery)
  assumed such a scenario existed somewhere and just hadn't been pulled in
  yet. That assumption was wrong, and it was corrected mid-project, not
  discovered by this implementation on its own — worth recording exactly how,
  since it changes what "frozen" means for every number in this fixture:
  - This sandbox never had a copy of
    `04_Prototype_Specification_Dynamic_Affective_Cooperation_Frozen_
    PreImplementation.docx` (the document this repo's own header, and the
    note above, both assumed existed). Only the Implementation Blueprint
    docx is present here, and it only *references* §19.9/§20.15 — it never
    reproduces their content. Four other architecture docx files available
    in this sandbox were also checked; none contain a literal scenario
    either.
  - The user was asked how to proceed and, after checking their own
    recovered copy of the actual frozen specification, confirmed directly:
    even that source document freezes only an **analytical contract** for
    this scenario, not literal values. The frozen contract is: 2-3 options;
    pre-elicited explicit `value_priorities`; structured option-to-value
    impacts; required-evidence fields; `stakes` in [0.75, 0.85];
    `autonomy_weight=0.90`; `safety_risk=0.20`; `no_undue_influence=true`;
    Turn 1 establishes `a*_intv >= 0.65`; Turn 2 satisfies `c_t>=0.60`,
    `h_ctrl>=0.40`, `h_int>=0.65`, `h_unc<0.70`, `d_amb<0.65`, `d_goal<0.65`,
    `safety_risk<0.70`; CURRENT_CUE must fail at least one A-dependent
    REDIRECT threshold at Turn 2; DYNAMIC at `rho=0.35` must clear
    `a_info>=0.55` and `a_intv>=0.60` and select REDIRECT. Option names,
    turn wording, and every concrete `value_priorities`/option-impact number
    are explicitly **not** pinned down by the frozen specification — the
    user's own instruction was to author those as implementation-level
    fixture content, "not by trial-and-error wording," and to verify the
    result by running it through the real Phase 1-5 code rather than
    asserting it by hand. **A researcher should treat `config/
    demo_fixture.yaml`'s scenario text/numbers as authored fixture content
    to review, not as a transcription of anything frozen — only the twelve
    analytical gates listed above, and the fact that this fixture provably
    satisfies every one of them, are the frozen/binding part.**
  - The scenario actually built (`config/demo_fixture.yaml`): a 2-turn,
    3-option career-decision fixture (`startup_offer` focal, plus
    `enterprise_offer`/`remote_offer` for structure only — only the focal
    option's impacts ever feed `d_goal`). It was **engineered analytically**,
    not guessed: Turn 1's `H_t` is chosen so `a*_intv` clears 0.65 with a
    healthy margin (0.71775); Turn 2's own target `A*_2` is engineered to
    fall JUST below the REDIRECT `a_info`/`a_intv` thresholds (0.435/0.552 —
    this is exactly what CURRENT_CUE experiences at Turn 2, since
    CURRENT_CUE always runs at `rho=0`, so its `A_t` equals `A*_2` exactly);
    DYNAMIC's `rho=0.35` blend of the still-high `A_1` and the just-under
    `A*_2` climbs back above both thresholds (0.560125/0.6100125). Every one
    of these numbers was computed by, and is re-verified at test time by,
    the real `derive_interaction_state`/`LinearPersistenceTransition`/
    `PolicyEngine` code — not copied out of a spreadsheet. **Wording
    correction (user review):** DYNAMIC's Turn-2 crossing margins over the
    REDIRECT thresholds are narrow by design (~+0.010 on both `a_info` and
    `a_intv`), not a "clean" or "healthy" margin — earlier drafts of this
    documentation used those words loosely. The narrowness is intentional
    (the scenario is engineered so DYNAMIC just barely crosses the line
    CURRENT_CUE stays just under) and is not a reliability concern, since
    the whole trajectory is fully deterministic — no randomness anywhere in
    Phase 1-5's code — so the same positive margin reproduces identically
    on every run (`test_demo_fixture_repeatable` checks exactly this).
    Turn 1's `a*_intv` margin (+0.068 over the 0.65 gate) is the one
    genuinely comfortable margin in this scenario.
  - `services/demo_fixture.py`: `DemoTurn`/`DemoFixture` are plain frozen
    dataclasses (same rationale as Phase 2's `Ctx`/`Rule`, Phase 4's
    `GenerationContract`). `load_demo_fixture()` parses and validates the
    YAML before returning: it schema-validates each stored `HumanAppraisal`
    (round-tripped through a real `HumanAppraisal(**...)` construction, so
    an out-of-range field raises pydantic's `ValidationError` immediately)
    and calls `validate_demo_fixture()` on the fully-parsed fixture, so a
    scenario-level inconsistency raises `ScenarioConfigError` at load time
    too — see the "Phase 6 fix round" entry below for why this wasn't true
    of the first Phase 6 delivery. `validate_demo_fixture()` remains public
    and separate from `load_demo_fixture()` so a deliberately-broken
    in-process fixture can still be validated directly, with no second YAML
    file on disk needed (see `test_focal_option_validation_fails_fast`).
    `validate_demo_fixture()` has a docstring in `services/
    derived_features.py`'s own `ScenarioConfigError` ("checked at
    fixture-load time... before any turn runs") but no body anywhere in the
    blueprint — implemented here by constructing a throwaway `Observation`
    per turn and calling the REAL `derive_interaction_state(o_t, g_t)`
    eagerly, so the exact same
    `ScenarioConfigError` a broken scenario would raise mid-demo is instead
    raised at load time, from ONE place, never duplicated logic that could
    drift from the real formula. `run_demo_fixture()` elicits the fixture's
    `OutcomeBaseline` once, then calls `ExperimentController.compare()` once
    per turn, in order — this literally is the "wired through `compare()`"
    requirement (blueprint §10 Phase 6's own line), and it is the exact same
    `compare()` a live demo turn would call; there is no separate fixture-
    replay code path.
  - `tests/test_demo_fixture.py` implements the three tests the user asked
    for by name (`test_demo_turn2_satisfies_redirect_eligibility_and_flips_
    policy`, `test_demo_fixture_repeatable`, `test_focal_option_validation_
    fails_fast`), plus two supporting tests
    (`test_load_demo_fixture_matches_frozen_analytical_contract` as cheap
    insurance against a future edit silently drifting outside the frozen
    numeric-gate contract). The first test computes the fixture's actual
    `D_t`/`A*_t`/`A_t`/policy through the real pipeline and asserts every one
    of the twelve frozen gates plus the actual REDIRECT-vs-not policy flip
    and the `state_could_influence_policy`/`state_did_influence_policy`
    causal-instrumentation split between CURRENT_CUE (`could=True,
    did=False`) and DYNAMIC (`could=True, did=True`) — not asserted by hand,
    computed live. The repeatability test runs the whole fixture through two
    fully independent `Pipeline`/`ExperimentController` instances and checks
    every `D_t`/`c_t`/`A*_t`/`A_t`/policy field comes back numerically
    identical, which it must, since nothing in Phase 1-5's code is
    stochastic.
  - Post-delivery correction, self-caught before delivery (not user-found):
    the first draft of `test_demo_turn2_satisfies_redirect_eligibility_and_
    flips_policy` checked DYNAMIC's Turn-2 `a_star.decision_information_
    priority`/`a_star.intervention_readiness` against the REDIRECT
    thresholds — but `compare()` computes ONE shared `a_star` per turn
    (`a_star_shared`, §6.7), so DYNAMIC's `a_star` is identical to
    CURRENT_CUE's, and would never clear the threshold either. The correct
    quantity to check for DYNAMIC is `a_t` (the persisted, `rho`-blended
    state) — `a_star` is the shared pre-persistence target every non-
    TASK_FOCUSED condition reads, `a_t` is what actually differs between
    CURRENT_CUE and DYNAMIC. Fixed before this delivery; the test now also
    asserts `dynamic_turn2.a_star == cue_turn2.a_star` explicitly, so this
    shared-target design is checked, not just relied on silently.
  - Result: 223 passed + 6 intentionally skipped (up from 219 passed).
- **Phase 6 fix round (post-delivery review): one real gap found and fixed
  in `load_demo_fixture()`, with two parts, plus a wording correction.**
  This is supposed to be the reliable, pre-validated offline interview
  fallback, so a malformed prepared fixture must fail at load time, not
  silently load and only surface a problem later, mid-demo.
  1. **`load_demo_fixture()` never actually validated.** The first draft
     parsed YAML into a `DemoFixture` and returned it without calling
     `validate_demo_fixture()` — validation only happened if a caller
     remembered to invoke that second function manually, contradicting the
     blueprint's own "`ScenarioConfigError`... checked at fixture-load time
     via `validate_demo_fixture()`" line (`services/derived_features.py`'s
     own docstring) and this module's own (previously aspirational, not
     actually enforced) claim to do so. Fixed: `load_demo_fixture()` now
     calls `validate_demo_fixture(fixture)` on the fully-parsed fixture
     before returning it — an unknown `focal_option_id` (or any other
     scenario-level inconsistency) now raises `ScenarioConfigError` from
     `load_demo_fixture()` itself. `validate_demo_fixture()` stays public
     and separate so a test can still exercise it directly against a
     deliberately-broken in-process fixture, with no second YAML file
     needed. Covered by
     `test_load_demo_fixture_fails_fast_on_invalid_focal_option`.
  2. **Stored appraisals were never schema-validated at load time.** Each
     turn's `appraisal` block was stored as a bare `dict(turn_raw[
     "appraisal"])` — an out-of-range value like `affect_intensity: 5.0`
     would load successfully and only be rejected later by
     `AppraisalEstimator._validate()`, during a live-feeling demo run, via
     its retry-then-fallback path. That is the right failure mode for a
     genuinely flaky live LLM; it is the wrong one for a frozen fixture.
     Fixed: `load_demo_fixture()` now constructs a real
     `HumanAppraisal(**turn_raw["appraisal"])` per turn (raising pydantic's
     `ValidationError` immediately on bad input) and stores
     `validated_appraisal.model_dump(mode="json")` — the same flat-dict
     shape `MockLLMAdapter`'s `extract_appraisal` queue and
     `AppraisalEstimator._validate()` both already expect, so nothing
     downstream of `DemoTurn.raw_appraisal` changed. Covered by
     `test_load_demo_fixture_rejects_invalid_appraisal`.
  3. Wording correction: this file previously described the scenario's
     margins over the frozen thresholds as uniformly "clean"/"healthy."
     DYNAMIC's Turn-2 crossing margins are narrow by design (~+0.010 on
     both `a_info` and `a_intv`) — corrected above, where those margins are
     first described, rather than restated a third time here.
  - Result: 225 passed + 6 intentionally skipped (up from 223 passed).
- **Phase 7 (`llm/prompts.py`, `llm/adapter.py`'s `AnthropicLLMAdapter`,
  `services/state_manager.py`, `ui/*.py`, `app.py`) is the largest
  authored-content phase in this project — everything below should be
  reviewed against the frozen specification directly, same caveat already
  attached to every earlier phase's authored wording.** Phase 7's own build-
  order line (§10) is short — "`ui/*` (Streamlit), real `llm/adapter.py`
  backend" — and its gate is explicitly a **manual researcher acceptance
  step**, not a pytest target: "Full live rehearsal × 10 (§16.2, §20.17);
  offline/replay fallback verified with network disabled." Two things worth
  saying plainly before the rest of this section: (1) this delivery cannot
  complete that gate on its own — it needs a real `ANTHROPIC_API_KEY` and a
  human interviewer/participant exchange for the ten-live-turn half, and a
  human actually disconnecting the network for the "verified with network
  disabled" half; (2) what this delivery DOES do is build everything needed
  to perform both halves, and get the automatable slice of the second half
  (the app launches and runs the frozen demo scenario end to end with zero
  network calls) actually verified by `tests/test_app_smoke.py`, using
  Streamlit's own headless `AppTest`. Run `streamlit run app.py` yourself
  (see "Running the app" above) to complete the parts only a human can.

  - **A genuine inconsistency inside the blueprint itself, resolved here
    (not merely an underspecified body — see `services/state_manager.py`'s
    own module docstring for the full reasoning):** §3's repo layout names
    `services/state_manager.py` ("session lifecycle, goal_version
    bookkeeping") but §10's own build-order table never assigns it to ANY
    phase — Phase 5's line names `pipeline.py`, `experiment_controller.py`,
    `goal_state_manager.py`, `outcome_baseline.py` and `logger.py`
    explicitly and omits `state_manager.py`, and no later line mentions it
    either. Two readings were considered: (a) it's redundant naming,
    already fully covered by `ExperimentController`/`GoalStateManager`; (b)
    it names a genuinely missing piece — `Pipeline`'s own docstring already
    assumes a "caller" that holds the current `G_t` across calls, but
    nothing built through Phase 6 ever needed to, since every test/demo-
    fixture caller just constructs a `GoalState` locally. Reading (b) was
    adopted: `SessionState` is that caller layer, built now because
    Streamlit's rerun-the-whole-script-every-interaction model is the
    FIRST real caller this codebase has that must persist "the current
    session" at all. **A researcher should confirm this reading against
    the frozen specification directly** — the blueprint's own §3/§10
    disagreement isn't resolved by anything else in it.
  - `llm/prompts.py` (structured-extraction prompt template, named but
    given no body anywhere in the blueprint — same gap `llm/fallback.py`'s
    `FallbackTemplates` and `services/response_generator.py`'s
    `QUALITATIVE_CUES` already had in earlier phases). Two prompt pairs:
    `build_appraisal_extraction_prompt()` (schema = `HumanAppraisal`'s own
    field table, field-for-field, so a compliant LLM response is exactly
    what `AppraisalEstimator._validate()` already expects — no second
    schema to keep in sync) and `build_generation_prompt()` (structurally
    cannot leak raw `A_t`, since `GenerationContract` has no `a_t`/`a_star`
    field at all — `tests/test_llm_prompts.py` additionally greps the
    rendered prompt text itself for every `AgentState` field name, in case
    a future wording edit ever names one by accident). **All prompt wording
    is authored for this prototype and should be reviewed/replaced by a
    researcher before a real study**, same standing caveat as every other
    authored-wording module in this codebase.
  - `llm/adapter.py`'s `AnthropicLLMAdapter` — the real, network-calling
    `LLMAdapter` implementation Phase 3-6 never needed (`MockLLMAdapter`
    covered every test/demo-fixture use through Phase 6). **Judgment call:
    the blueprint specifies the adapter INTERFACE exactly but never names a
    concrete LLM provider anywhere** — "provider-agnostic" is the whole
    point of the interface (§2's own technology-stack table). Anthropic's
    Claude API was chosen as the one concrete backend actually built, since
    this prototype's own development environment is Claude-based and the
    `anthropic` SDK's Messages API is a well-documented, directly-testable
    target — not because the frozen specification names Anthropic anywhere.
    Swapping providers is a one-line change at `app.py`'s own adapter-
    construction call sites, since `AppraisalEstimator`/`ResponseGenerator`
    only ever depend on the `LLMAdapter` Protocol, never on
    `AnthropicLLMAdapter` by name. Dependency-injected `client` constructor
    parameter (matching `MockLLMAdapter`'s own zero-network testability) —
    every test in `tests/test_anthropic_adapter.py` constructs one with a
    fake client double shaped exactly like the real SDK's own response
    object, so the SAME parsing code a real response would hit is what
    tests actually exercise; only the network call itself is faked.
    `DEFAULT_MODEL` is a plain configuration default, explicitly documented
    as NOT frozen the way `config/default.yaml`'s numeric thresholds are —
    provider model identifiers go stale over time independent of anything
    in this codebase; override it, don't treat the current value as load-
    bearing.
  - `ui/*.py` (`interaction_view.py`, `researcher_dashboard.py`,
    `experiment_controls.py`, `trajectory_view.py`) — one module per §8's
    own UI component-map row (four rows folded into `researcher_dashboard.py`,
    since all four read off ONE condition's own `TurnRecord` with no
    derivation of their own: Human/Goals panel, Agent State panel — rho
    shown separately, never inside `A_t`, per §20.16 — Policy panel, and
    Status bar). Every module is a pure, read-only renderer of data `app.py`
    already produced via `ExperimentController`; none of them call a
    service directly, matching this project's existing services-vs-UI
    separation. **Layout/wording is authored, not specified** — the
    blueprint gives each component's data source but no visual design.
  - `app.py` — the one real entry point. Wires a `Pipeline`/
    `ExperimentController`/`SessionState` per Streamlit session, offers two
    backend modes in the sidebar (documented at the top of the file: Mock
    = the "offline/replay fallback verified with network disabled" half of
    Phase 7's gate, Anthropic = the "full live rehearsal × 10" half), and
    is the first real caller of `compute_config_hash()` over ALL FOUR
    runtime YAML files together (`default.yaml` + `conditions.yaml` +
    `policy_rules.yaml` + `demo_fixture.yaml`) — the exact "REMINDER for
    Phase 6/7 integration" `services/pipeline.py`'s own
    `compute_config_hash()` docstring left open back in the Phase 5 fix
    round; Phase 0-6's own tests deliberately hashed `default.yaml` alone,
    sufficient for their own narrower scope per that same docstring.
  - Post-delivery corrections, self-caught before this delivery (not user-
    found): (1) `ui/researcher_dashboard.py`'s `st.table()` calls originally
    mixed types (float/bool/dict/enum) within one column, which Streamlit's
    own pyarrow-based table serializer cannot cleanly infer — it silently
    recovers by stringifying, but only after logging a full traceback per
    failed column on every rerun; fixed by `str()`-casting each such column
    before it reaches `st.table()`, which renders identically with no log
    spam. (2) The "New Run / Reset Demo" button originally called
    `SessionState.reset()` on the EXISTING session — correct for a live-only
    caller, but wrong for Mock mode specifically: Mock's adapter carries a
    one-shot, pre-scripted extract/generate queue that a same-object reset
    would leave already-exhausted, so a second demo-scenario run after
    Reset would hit `MockLLMAdapter`'s own "called more times than it was
    scripted for" `AssertionError`. Fixed by discarding the whole
    `SessionState` on reset and letting the normal init path rebuild a
    fresh adapter with a freshly-reloaded queue instead — correct for both
    backends, caught by `tests/test_app_smoke.py`'s own
    `test_demo_scenario_runs_again_cleanly_after_reset`, which specifically
    drives run → reset → run again. `SessionState.reset()` itself is
    unchanged and still correct for a caller with no such one-shot state to
    lose (a live-only session, a notebook, a future UI) — see its own
    docstring in `services/state_manager.py` for why `app.py` doesn't use
    it after this fix.
  - `requirements.txt` gained `streamlit>=1.63` and `anthropic>=1.3` —
    Phase 0-6 never needed either.
  - Result: 257 passed + 6 intentionally skipped (up from 225 passed).

- **Phase 7 fix round (independent review of the first Phase 7 delivery
  found three blockers and two smaller issues; all five are corrected
  below, verified by rerunning the full suite — 260 passed + 6 intentionally
  skipped, up from 257 passed, INCLUDING the Streamlit `AppTest`-dependent
  files, which the reviewer's own environment could not run — no network
  and no `streamlit` install there).** This round is targeted fixes, not a
  redesign, per the reviewer's own framing.

  1. **Blocker — the "one-variable intervention" control mutated `G_t`,
     which is the wrong intervention.** The first draft's
     `ui/experiment_controls.py` let a researcher change
     `GoalState.stakes`/`autonomy_weight`/`safety_risk` mid-run through
     `GoalStateManager` under the label "one-variable intervention."
     Changing `G_t` changes the mechanism upstream of `D_t`/`A*_t`/`A_t`,
     so it demonstrates a *different* experiment than the frozen
     acceptance criterion actually names — "changing `a_info`, `a_intv`,
     or `rho` can change `P_t` without changing user text or `H_t`." `rho`
     is the one variable this codebase already has a clean, non-mutating
     way to vary in isolation: `Pipeline.replay_turn()` (§20.14, frozen
     Phase 5 code, `services/pipeline.py`) reuses a turn's own stored
     `H_t`/`c_t`/`D_t`/`A*_t` and only re-runs persistence-apply →
     policy-select → generate at a different `rho` — never touching `G_t`,
     never mutating the original `TurnRecord`. **Fix:** `ui/experiment_
     controls.py` was rewritten to drop the `G_t`-mutating control
     entirely; `ExperimentControlsResult` no longer has an `intervention`
     field, only `rho_override` (for a brand-new live turn, logged on that
     turn's own `TurnRecord.interventions`) and `replay_request` (a
     `(turn_id, condition, rho_override)` tuple driving `replay_turn()` on
     a *past* turn, additively, without touching the original record).
     `app.py`'s intervention-handling block was removed to match.
     `SessionState.apply_explicit_goal_update()` itself is unchanged and
     still available to a programmatic caller for a genuinely different
     kind of intervention (an explicit human preference change) — it is
     simply no longer exposed under the "one-variable intervention" label,
     which is now reserved for the frozen `rho`-isolation demonstration.
     See `ui/experiment_controls.py`'s own module docstring for the full
     reasoning.
  2. **Blocker — offline replay crashed after a full demo run.** The first
     draft's Mock backend used `MockLLMAdapter` (a Phase 3 test double)
     with a `generate_responses` queue scripted for exactly one demo pass.
     `Pipeline.replay_turn()` *always* re-invokes the generator, even
     though it reuses the original record's own stored `H_t`/`c_t`/`D_t`/
     `A*_t` rather than re-estimating them — this is frozen Phase 5
     behavior, not something Phase 7 can change. So replaying any turn
     after the demo had already run through all three conditions (which
     itself consumes the queue) hit `MockLLMAdapter`'s own "called more
     times than it was scripted for" `AssertionError` — a test-safety-net
     behavior that is correct and load-bearing for `MockLLMAdapter`'s many
     other callers across the test suite, so weakening it was not an
     option. **Fix:** a new class, `OfflineDemoAdapter` (`llm/adapter.py`),
     deliberately separate from `MockLLMAdapter` — unlimited and
     deterministic rather than a one-shot scripted queue. It answers
     `extract_appraisal()` by turn_id from `demo_fixture.yaml`'s own frozen
     `raw_appraisal` values (exactly what the demo needs, any number of
     times) and `generate()` with a fixed, clearly-labeled placeholder
     string built from the contract's own policy/secondary-policy — never
     empty, never exhausted. `app.py`'s `_init_mock_session()` now builds
     one of these instead of a `MockLLMAdapter`. **Regression test added**
     per the reviewer's own exact request — "run demo -> replay Turn 2
     DYNAMIC -> no exception" — as
     `tests/test_app_smoke.py::test_replay_after_full_demo_does_not_raise`:
     loads the app, clicks "Run demo_fixture.yaml," then drives the
     "Replay a past turn" control to turn 2 / DYNAMIC and clicks "Replay,"
     asserting no exception at any step.
  3. **Blocker — replay results were shown as a transient text-only
     banner, not state/policy panels.** The first draft's replay handling
     rendered only an `st.success()` message with the regenerated response
     text. The reviewer's fix instruction: state/policy should be the
     *default* replay presentation, with response-text regeneration
     separately optional. Since `Pipeline.replay_turn()` always regenerates
     the response text as a side effect (see point 2 — that is frozen
     Phase 5 behavior, not optional at the `Pipeline` layer), "optional"
     is implemented at the *display* layer instead. **Fix:** `app.py`
     stores the `(original, replayed)` `TurnRecord` pair in
     `st.session_state["last_replay"]` (cleared whenever a fresh demo or
     live turn is recorded, or on reset) and a new function,
     `ui/researcher_dashboard.py`'s `render_replay_comparison(original,
     replayed)`, renders both records' Agent State and Policy panels
     side-by-side (reusing the exact same `_render_agent_state_panel()`/
     `_render_policy_panel()` helpers the live dashboard uses, so the two
     views are visually identical and directly comparable) with the
     regenerated response text tucked inside a collapsed-by-default
     `st.expander("Regenerated response text (optional)", ...)`. If the
     original record can't be located (e.g. session state was reset
     between recording and replay), only the replayed side renders, with
     a caption explaining why.
  4. **Smaller issue — raw hard-constraint codes reached the real model
     verbatim.** `build_generation_prompt()` (`llm/prompts.py`) originally
     did `"; ".join(contract.hard_constraints)` straight into the prompt
     sent to Anthropic, so a real model call could see literal internal
     codes like `NO_UNDUE_INFLUENCE` and `AUTONOMY_HIGH` — the exact defect
     `llm/fallback.py`'s own docstring already documents fixing for the
     offline template path, just not yet fixed for the real-LLM path.
     **Fix:** a new `_HARD_CONSTRAINT_INSTRUCTIONS` dict and
     `_translate_hard_constraints()` helper in `llm/prompts.py` translate
     every code `services/policy_engine.py._hard_constraints()` can ever
     produce into a natural-language *instruction to the model* (not a
     finished participant-facing sentence like `llm/fallback.py`'s own
     clauses — the model is told what to realize in its own words, per
     `GENERATION_SYSTEM_PROMPT`'s existing "never as a quoted phrase"
     instruction). An unrecognized code raises `ValueError`, matching
     `FallbackTemplates.render()`'s own safety discipline of never
     silently forwarding or dropping an un-reviewed code.
     `tests/test_llm_prompts.py`'s existing
     `test_generation_prompt_embeds_policy_and_hard_constraints_and_cue`
     (which asserted the raw code string appeared) was updated to assert
     the opposite plus the translated wording; two tests were added
     (`test_generation_prompt_translates_autonomy_high_constraint`,
     `test_generation_prompt_raises_on_unrecognized_hard_constraint`).
  5. **Smaller issue — "pre-elicited" wording was inaccurate once `G_t` can
     change mid-run.** `build_appraisal_extraction_prompt()`'s user prompt
     described value priorities as "the participant's own, pre-elicited
     before this conversation" — inaccurate once `SessionState.apply_
     explicit_goal_update()` has changed them mid-session, since
     "pre-elicited" implies session-start values specifically. **Fix:**
     reworded to "Current explicit value priorities (the participant's
     own)."

  All five fixes reviewed together against `llm/fallback.py`'s and
  `services/pipeline.py`'s already-frozen, already-tested behavior rather
  than introducing new mechanisms — every fix reuses or relabels existing,
  correct machinery (`Pipeline.replay_turn()` for #1, a sibling adapter
  class for #2 that doesn't touch `MockLLMAdapter`'s test-safety-net
  behavior, the existing dashboard panel helpers for #3, `llm/fallback.py`'s
  own already-fixed translation pattern for #4).
  - Result: 260 passed + 6 intentionally skipped (up from 257 passed),
    including `tests/test_app_smoke.py`'s Streamlit `AppTest`-based tests —
    rerun on a machine with `streamlit` installed and network available,
    since the reviewer's own environment has neither and could not run
    that file (confirmed via a fresh extraction of the delivered zip).
  - **Second-round fix (documentation only, no code/behavior change):** the
    reviewer caught that `services/state_manager.py`'s
    `apply_explicit_goal_update()` docstring still said `ui/experiment_
    controls.py`'s "one-variable intervention" control was "this method's
    one real caller" — true of the first Phase 7 delivery, stale after fix
    #1 above replaced that control's mechanism (the rewritten
    `ui/experiment_controls.py` never calls this method). Reworded to
    state plainly that this method has no caller in `ui/*.py` as of the
    fix round and remains available for a future/programmatic caller with
    a genuinely different kind of explicit `GoalState` update to make.
    Full suite rerun after this change: still 260 passed + 6 skipped (a
    docstring-only edit, as expected).

- **OpenAI support (`llm/adapter.py`'s `OpenAILLMAdapter`, `llm/prompts.py`'s
  `APPRAISAL_JSON_SCHEMA`, `app.py`'s three-way backend selector) — added
  post-approval, at the user's own explicit request, as a THIRD optional
  backend alongside Offline and Anthropic.** The user's own framing, quoted
  because it is the actual acceptance criterion for everything below: "The
  important rule is: do not change the scientific architecture at all.
  OpenAI should only replace Anthropic at the two LLM boundaries: (1)
  appraisal extraction (2) natural-language response generation. Everything
  else must remain unchanged: `H_t -> D_t -> A*_t -> A_t -> P_t`." Nothing
  in `Pipeline`, `AppraisalEstimator`, `StateTransition`, `PolicyEngine`,
  `rho` logic, replay semantics, `TurnRecord`, or `demo_fixture.yaml` was
  touched by this addition — `_build_pipeline()` (`app.py`) is the one and
  only place a `Pipeline` gets constructed, and it is byte-for-byte
  unmodified; only which `LLMAdapter` implementation gets handed to it
  differs per backend. The Turn-2 result the user asked to be re-verified
  after this addition (`CURRENT_CUE -> INFORM`, `DYNAMIC -> REDIRECT`) was
  checked directly against a live `_init_mock_session()` run and is
  unchanged.

  - `llm/adapter.py`'s `OpenAILLMAdapter` — a strict sibling of
    `AnthropicLLMAdapter`, not a replacement (`AnthropicLLMAdapter` itself
    is untouched). Implements the same `LLMAdapter` Protocol
    (`extract_appraisal`/`generate`, identical failure contracts: never
    raises from `extract_appraisal`, raises `TimeoutError`/`GeneratorError`
    from `generate` — see this class's own docstring for the full
    reasoning, unchanged from `AnthropicLLMAdapter`'s). Uses the OpenAI
    Responses API (`client.responses.create(...)`), not the older Chat
    Completions API, per the user's own explicit instruction. Constructor
    supports dependency injection (`client=None` — builds a real
    `openai.OpenAI()` itself, reading `OPENAI_API_KEY` from the environment,
    exactly as `anthropic.Anthropic()` already does for `ANTHROPIC_API_KEY`
    — never a hardcoded key) and a configurable model (`model=None` resolves
    to the `OPENAI_MODEL` environment variable, then `DEFAULT_MODEL =
    "gpt-4o-mini"` — same "override-only default, not frozen" caveat already
    attached to `AnthropicLLMAdapter.DEFAULT_MODEL`). Every test in
    `tests/test_openai_adapter.py` constructs it with a fake client double,
    exactly like `tests/test_anthropic_adapter.py` — no real network access,
    API key, or the `openai` package's own runtime request behavior is ever
    exercised by the test suite.
  - **Same prompts, no OpenAI-specific scientific wording** (explicit user
    requirement): `OpenAILLMAdapter.extract_appraisal()`/`generate()` call
    the EXACT SAME `build_appraisal_extraction_prompt()`/
    `build_generation_prompt()` functions (`llm/prompts.py`)
    `AnthropicLLMAdapter` already calls — there is no second, OpenAI-
    specific prompt-building function anywhere in this codebase. Both
    providers therefore automatically inherit every scientific safeguard
    already built into those shared functions, including the Phase 7 fix
    round's own hard-constraint translation (`_translate_hard_constraints()`)
    and "current explicit value priorities" wording — reused, not
    reimplemented.
  - **Structured Outputs for appraisal extraction, not free-form JSON-in-
    prose** (explicit user requirement): `APPRAISAL_JSON_SCHEMA`
    (`llm/prompts.py`) is a JSON Schema built directly from `HumanAppraisal`'s
    own field table (`models/human_state.py`) — the same field names
    `APPRAISAL_SYSTEM_PROMPT` already describes in prose for Anthropic, kept
    as one machine-readable twin rather than a second, independently typed
    schema that could drift from either. Sent via `text={"format": {"type":
    "json_schema", "name": "human_appraisal", "schema": APPRAISAL_JSON_SCHEMA,
    "strict": True}}`, so the API enforces schema conformance server-side.
    `additionalProperties: false` and every field (including the nullable
    `possible_affect` and the always-array `evidence_tags`) listed in
    `required` is OpenAI's own `strict=True` constraint, not a change to
    `HumanAppraisal`'s Pydantic definition — optionality is expressed via a
    nullable JSON type, not field absence. `generate()` uses plain-text
    output (no schema) since there is nothing to structurally enforce for
    free-text generation — the causal-isolation rule is already structural
    via `GenerationContract` having no `a_t`/`a_star` field at all, unchanged.
  - **Error behavior, identical contract to Anthropic:** `extract_appraisal()`
    catches every exception and returns `None` — timeout, auth failure,
    network failure, and a malformed/empty response are all the same "no
    usable extraction" outcome, letting `AppraisalEstimator`'s existing
    retry/fallback behavior handle it exactly as it already does for
    Anthropic. `generate()` maps a timeout (builtin `TimeoutError`, or a real
    `openai.APITimeoutError`, both checked by `_is_openai_timeout()`) to
    `TimeoutError`, and any other failure to `GeneratorError` — `_extract_
    openai_text()` also treats OpenAI's own documented `output_text == ""`
    empty-response case as a failure (raises internally, caught the same
    way), matching `_extract_text()`'s equivalent empty-content check for
    Anthropic. `ResponseGenerator`'s existing deterministic
    `FallbackTemplates` fallback is exercised identically regardless of
    which live backend raised.
  - `app.py`'s backend selector is now three-way — `BACKEND_OPTIONS =
    [OFFLINE_BACKEND, OPENAI_BACKEND, ANTHROPIC_BACKEND]` ("Offline" /
    "OpenAI" / "Anthropic", replacing Phase 7's first-delivery "Mock
    (offline/replay)" / "Anthropic (live)" labels). Whichever live backend
    has an environment key present is preselected (Anthropic given priority
    when both are present — a judgment call, since this combination wasn't
    previously reachable; no acceptance criterion orders the two), Offline
    otherwise. A sidebar caption always shows the active backend and model
    id (`_init_*_session()` now each return `(SessionState, model_id)`
    rather than `SessionState` alone). Neither API key field
    (`type="password"`) nor either environment-presence flag is ever written
    to `st.session_state`, logged, or included in `config_hash` — see this
    module's own docstring for the explicit "secrets and config_hash" note
    the user's requirement #8 asked for; `compute_config_hash()` is still
    called on `_load_full_runtime_config()`'s return value only (the four
    frozen `RUNTIME_CONFIG_FILES`), unchanged.
  - `_init_openai_session(api_key)` mirrors `_init_live_session(api_key)`
    field for field — `api_key=None` falls back to `openai.OpenAI()`'s own
    `OPENAI_API_KEY` environment lookup; a sidebar-entered key is passed
    through explicitly otherwise. Calls the same unmodified
    `_build_pipeline()` every other backend calls.
  - **Session/backend switching never reuses a Pipeline across providers**
    (explicit user requirement #7): the existing `st.session_state.get(
    "backend") != backend` guard already rebuilds a brand-new `SessionState`
    (and therefore a brand-new `Pipeline`/`ExperimentController`) on ANY
    backend change, not only an Offline-reset — this was true before this
    addition and needed no new logic, only the three-way `elif` chain
    calling the right `_init_*_session()`. `tests/test_app_smoke.py::
    test_backend_switch_does_not_contaminate_state` drives exactly this:
    run the Offline demo to completion, switch to OpenAI, and confirm the
    new session starts with zero turns (not the Offline session's leftover
    six-tab comparison view).
  - `requirements.txt` gained `openai>=1.0` — Phase 0-7 (Anthropic-only)
    never needed it.
  - `tests/test_openai_adapter.py` (new, 22 tests) mirrors `tests/
    test_anthropic_adapter.py`'s coverage almost line for line (successful
    structured extraction, malformed/empty response, timeout, transport
    failure, repair-flag propagation, successful/timeout/failure `generate()`
    paths, `from_env()`/constructor error paths) plus the user's own explicit
    non-negotiables as their own tests: no numeric `A_t`/`A*_t`/`rho` field
    ever reaches the request sent to OpenAI
    (`test_generate_never_leaks_agent_state_fields`), raw hard-constraint
    codes are never sent (`test_generate_never_sends_raw_hard_constraint_
    codes`), neither shared prompt ever requests chain-of-thought
    (`test_prompts_never_request_chain_of_thought`), and `APPRAISAL_JSON_
    SCHEMA` names exactly `HumanAppraisal`'s own fields, no more, no fewer
    (`test_appraisal_json_schema_matches_human_appraisal_fields_exactly`).
  - `tests/test_app_smoke.py` gained three OpenAI-specific tests (backend
    stops cleanly with no key; a full live turn runs end to end through a
    real `openai.OpenAI()` client with only the two LLM-boundary methods
    faked — "instantiated with a fake client/config path," per the user's
    own test requirement; backend switching doesn't leak state) and every
    existing test's backend-label assertions were updated for the new
    three-way selector. `test_app_loads_in_mock_backend_by_default_with_no_
    api_key` was renamed `test_app_loads_in_offline_backend_by_default_
    with_no_api_keys` and now guards against BOTH provider keys leaking in
    from the real environment, not just Anthropic's.
  - Result: 285 passed + 6 intentionally skipped (up from 260 passed).
    `tests/test_demo_fixture.py`'s own 6 tests (the analytical Turn-2
    `CURRENT_CUE -> INFORM` / `DYNAMIC -> REDIRECT` check among them) are
    untouched by this addition and still pass unmodified — direct evidence
    this addition changed no scientific result.
