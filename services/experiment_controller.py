"""ExperimentController — M8. Blueprint §6.7, spec §12.1/§13.1.

Owns condition/rho/replay/intervention control and is the only caller of
Pipeline; UI code (a later phase) never calls Pipeline directly, and never
resolves rho itself. resolve_rho/run_turn/compare/start_new_run are
transcribed field-for-field from the blueprint's own pseudocode (§6.7),
including the two implementation-review fixes the blueprint's own text
calls out: resolve_rho checks condition BEFORE override (a real bug in an
earlier draft let resolve_rho(CURRENT_CUE, override=.35) return .35), and
start_new_run() actually resets Pipeline's state rather than expecting a
UI layer to clear session state by hand.
"""

from __future__ import annotations

import uuid

from models.enums import Condition
from models.goal_state import GoalState, OutcomeBaseline
from models.turn_record import TurnRecord
from services.pipeline import Pipeline


class ExperimentController:
    def __init__(self, pipeline: Pipeline, rho_dynamic: float) -> None:
        self._pipeline, self._rho_dynamic = pipeline, rho_dynamic
        self.current_run_id: str = self.start_new_run()

    def resolve_rho(self, condition: Condition, override: float | None = None) -> float:
        """Central, single-source rho enforcement. Pipeline itself no
        longer decides rho by condition — it just applies whatever float it
        is given — so without this method a caller could accidentally pass
        a nonzero rho to a CURRENT_CUE turn. The CONDITION check runs FIRST
        and is not overridable: only DYNAMIC can ever return a nonzero rho,
        override included. override is ONLY for an explicit, logged
        research/ablation probe of DYNAMIC's rho; the fact that one was
        used is recorded via TurnRecord.interventions on the DYNAMIC record
        only — see Pipeline.run_turn()/compare()."""
        if condition is not Condition.DYNAMIC:
            return 0.0  # TASK_FOCUSED and CURRENT_CUE: always exactly 0 — an
            # override CANNOT change this, by construction, not by convention
        if override is not None:
            return override
        return self._rho_dynamic

    def run_turn(
        self,
        user_text: str,
        task_context: dict,
        condition: Condition,
        g_t: GoalState,
        rho_override: float | None = None,
        observed_choice: str | None = None,
        observed_confidence: float | None = None,
    ) -> TurnRecord:
        rho = self.resolve_rho(condition, rho_override)
        return self._pipeline.run_turn(
            self.current_run_id, user_text, task_context, condition, rho, g_t,
            rho_override=rho_override,
            observed_choice=observed_choice, observed_confidence=observed_confidence,
        )

    def compare(
        self,
        user_text: str,
        task_context: dict,
        g_t: GoalState,
        conditions: list[Condition],
        rho_override: float | None = None,
        observed_choice: str | None = None,
        observed_confidence: float | None = None,
    ) -> dict[Condition, TurnRecord]:
        """UI talks only to ExperimentController, never to Pipeline
        directly. Resolves DYNAMIC's rho once via resolve_rho — an
        explicit rho_override applies identically to every condition
        passed to Pipeline.compare(), which still zeroes it for every
        non-DYNAMIC condition (see Pipeline._rho_for)."""
        rho = self.resolve_rho(Condition.DYNAMIC, rho_override)
        return self._pipeline.compare(
            self.current_run_id, user_text, task_context, g_t, conditions, rho,
            rho_override=rho_override,
            observed_choice=observed_choice, observed_confidence=observed_confidence,
        )

    def replay_turn(self, turn_id: int, condition: Condition, rho: float | None = None) -> TurnRecord:
        """Judgment call (documented gap, flagged for review): the
        blueprint places replay_turn() on Pipeline (§6.7) and never shows
        ExperimentController delegating to it the way it does for
        run_turn()/compare() — but §6.7's own opening line says
        ExperimentController "is the only caller of Pipeline... UI code
        never calls the services directly." A UI needing to trigger a
        replay with no path through ExperimentController would contradict
        that sentence, so this thin pass-through is added here for
        consistency with run_turn()/compare()'s own pattern, not from an
        explicit pseudocode line naming it.

        Post-delivery fix: a supplied rho is now resolved through
        resolve_rho() before being passed to Pipeline.replay_turn() —
        exactly the same condition-checked-before-override rule run_turn()/
        compare() already apply. Before this fix, ExperimentController
        forwarded a caller's rho to Pipeline unresolved, so
        `controller.replay_turn(turn_id, Condition.CURRENT_CUE, rho=0.9)`
        could make a replayed CURRENT_CUE turn persistent — exactly the bug
        resolve_rho exists to make structurally impossible on every other
        path. Pipeline.replay_turn() now also enforces this itself (see
        that method's own docstring), so the invariant holds even for a
        caller that reaches Pipeline directly, bypassing this class
        entirely — but this fix keeps ExperimentController's own contract
        ("resolves rho," matching run_turn()/compare()) consistent too.
        rho=None (no override — reuse the original turn's own rho) is left
        unresolved here and passed straight through: Pipeline.replay_turn()
        already reuses original.rho in that case, and that stored rho was
        itself resolved correctly when the original record was created."""
        if rho is not None:
            rho = self.resolve_rho(condition, rho)
        return self._pipeline.replay_turn(turn_id, condition, rho)

    def elicit_outcome_baseline(self, value_priorities: dict[str, float]) -> OutcomeBaseline:
        """Judgment call (documented gap, flagged for review): not shown in
        the blueprint's own ExperimentController pseudocode (§6.7) at all
        — added because OutcomeBaseline (§5.3/§20.4) is documented as
        "elicited before turn 1" of a run, and ExperimentController is
        where every other per-run action (resolve_rho, run_turn, compare,
        start_new_run) already lives; routing this through Pipeline
        directly instead would be the one exception to "...is the only
        caller of Pipeline" (§6.7's own opening line). Call once per run,
        after start_new_run() (including the implicit one inside
        __init__) and before that run's first run_turn()/compare() call —
        Pipeline._assemble_turn_record raises if a turn is attempted for a
        run_id with no elicited baseline (see services/outcome_baseline.py).
        """
        return self._pipeline.outcome_baseline_store.elicit(self.current_run_id, value_priorities)

    def start_new_run(self) -> str:
        """'New Run / Reset Demo'. Resets EVERYTHING a second rehearsal, or
        one accidental click mid-interview, could otherwise leak from the
        previous run: shared history, both conditions' stored A_prev, the
        turn counter, and the run_id itself. A UI layer (Phase 7) is meant
        to expose this as a single button — never clear Streamlit session
        state by hand instead. Does NOT elicit a new OutcomeBaseline for
        the new run_id (see elicit_outcome_baseline's own docstring) —
        that is left to the caller, since value_priorities for a fresh
        elicitation come from outside anything start_new_run() itself has
        access to."""
        run_id = str(uuid.uuid4())
        self._pipeline.reset()
        self.current_run_id = run_id
        return run_id
