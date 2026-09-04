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
| 3 | `llm/adapter.py` (mock first), `appraisal_estimator.py` | **done** — `tests/test_appraisal_estimator.py`, 133/133 green (whole suite) |
| 4 | `response_generator.py` + fallback templates | not started |
| 5 | `pipeline.py`, `experiment_controller.py`, `goal_state_manager.py`, `outcome_baseline.py`, `logger.py` | not started |
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
  verbatim from the frozen specification — please re-check them against the
  source spec (not just the blueprint's compressed table) before Phase 3
  wires PolicyEngine into the live estimator loop.
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
- `llm/adapter.py`'s `MockLLMAdapter` is Phase 3's own scriptable test
  double (a queue of canned raw responses + call recording), **not** the
  demo-fixture-replay `MockAdapter` the blueprint's §6.8 prose describes for
  offline rehearsal — that one replays `demo_fixture.yaml`'s frozen
  H_t/c_t/D_t values, which don't exist in this repo yet (Phase 6). The
  `LLMAdapter` Protocol currently declares only `extract_appraisal`;
  `generate()` (used by `ResponseGenerator`, Phase 4) is added to the same
  Protocol once `GenerationContract` exists, not built ahead of that phase.
- Phase 6's `demo_fixture.yaml` is where the actual frozen demo-turn scenario
  content (Turn 1 / Turn 2 task context, value priorities, etc.) from the
  specification's §19.9/§20.15 belongs, and only there — it hasn't been
  transcribed into this repo yet, is not a dependency of any earlier phase,
  and should be pulled in and tuned to the analytical check when Phase 6 is
  actually reached.
