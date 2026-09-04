# Dynamic Affective Cooperation — prototype

Direct implementation of the frozen architecture (`04_Prototype_Specification_Dynamic_Affective_Cooperation_Frozen_PreImplementation.docx`)
per the Implementation Blueprint (`05_Implementation_Blueprint_Dynamic_Affective_Cooperation.docx`).
Built in the phased order the blueprint's §10 specifies — each phase's own
tests must be green before the next phase starts.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | `config/*.yaml`, `models/*.py` (all Pydantic schemas incl. `Turn`) | **done** — `tests/test_schemas.py`, 42/42 green |
| 1 | `services/observation_builder.py`, `derived_features.py`, `state_transition.py` | not started |
| 2 | `services/policy_engine.py` + `policy_engine_task_focused.py` + Rule table | not started |
| 3 | `llm/adapter.py` (mock first), `appraisal_estimator.py` | not started |
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
- Phase 1's transition-formula smoke test does **not** depend on the real
  interview demo scenario. It should be tested against small, synthetic,
  hand-calculated fixtures (pick simple H_t/G_t/D_t values, work the
  transition equations by hand, assert the code matches) — that is better
  science than testing against one unverified fixture anyway, since the
  expected values are independently derivable and don't depend on transcribing
  anything from the specification first.
- Phase 6's `demo_fixture.yaml` is where the actual frozen demo-turn scenario
  content (Turn 1 / Turn 2 task context, value priorities, etc.) from the
  specification's §19.9/§20.15 belongs, and only there — it hasn't been
  transcribed into this repo yet, is not a dependency of any earlier phase,
  and should be pulled in and tuned to the analytical check when Phase 6 is
  actually reached.
