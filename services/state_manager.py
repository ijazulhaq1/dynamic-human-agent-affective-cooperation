"""state_manager.py — SessionState. Blueprint §10 Phase 7 ("ui/* (Streamlit),
real llm/adapter.py backend").

Judgment call (documented gap, flagged for review — this one is an
inconsistency in the blueprint's OWN two sections, not merely an
underspecified body): §3's repo layout names this file — "state_manager.py
# session lifecycle, goal_version bookkeeping" — but §10's own build-order
table never assigns it to ANY phase; Phase 5's line names pipeline.py,
experiment_controller.py, goal_state_manager.py, outcome_baseline.py and
logger.py explicitly and does not mention state_manager.py, and no later
phase line mentions it either. Two readings were considered:

  (a) it is dead/redundant naming — "session lifecycle" already fully
      covered by ExperimentController.start_new_run()/.current_run_id, and
      "goal_version bookkeeping" already fully covered by GoalStateManager/
      GoalState.with_explicit_update() (both built in Phase 5) — so nothing
      new needs building here at all;
  (b) it names a genuinely missing piece: HumanAppraisal/GoalState/
      Pipeline's own docstrings are explicit that G_t is "read/updated by
      the caller... BEFORE g_t is passed into run_turn()/compare()"
      (services/pipeline.py's own __init__ docstring) — but nothing built
      through Phase 6 ever HELD "the current G_t for this session" anywhere,
      because every test/demo-fixture caller just constructs a GoalState
      locally and passes it straight in. Streamlit reruns its entire script
      top-to-bottom on every interaction (a framework fact, not a project
      choice), so the FIRST time this codebase has ANY caller that must
      persist "the current g_t" (and the current ExperimentController
      instance, and a UI-facing turn-by-turn record list) across repeated
      calls is exactly Phase 7's own UI. That is a real, previously-absent
      responsibility, not a duplicate of ExperimentController's run_id
      bookkeeping or GoalStateManager's pure update() method.

Reading (b) was adopted: SessionState below is the "caller" layer Pipeline's
own docstring already assumes exists but that no earlier phase needed to
build. It owns exactly three things a Streamlit session must keep across
reruns — the ExperimentController itself, the current GoalState (mutated
ONLY through GoalStateManager, never in place), and a UI-facing list of
past TurnRecords/comparisons for ui/trajectory_view.py — and delegates
every actual state-transition rule to the services already built in earlier
phases (ExperimentController.start_new_run(), GoalStateManager.update()) as
required by *this method's own docstring below* and by start_new_run()'s
own "never clear Streamlit session state by hand instead" line. A
researcher should confirm reading (b) against the frozen specification
directly before relying on it, since §3/§10's own inconsistency is not
resolved by anything in the blueprint itself.
"""

from __future__ import annotations

from typing import Any

from models.enums import Condition, GoalUpdateSource
from models.goal_state import GoalState
from models.turn_record import TurnRecord
from services.experiment_controller import ExperimentController
from services.goal_state_manager import GoalStateManager


class SessionState:
    """Owns everything one researcher's UI session must keep across
    Streamlit reruns. Not itself a Streamlit object — ui/*.py modules read
    and call this class; app.py is the only place that stores an instance
    in st.session_state (kept this way so SessionState stays importable and
    unit-testable — see tests/test_state_manager.py — without Streamlit
    installed or a script context running)."""

    def __init__(
        self,
        controller: ExperimentController,
        goal_manager: GoalStateManager,
        goal_state: GoalState,
    ) -> None:
        self.controller = controller
        self._goal_manager = goal_manager
        self.goal_state = goal_state
        # UI-facing history, kept independently of TurnLogger's own
        # append-only JSONL log (services/logger.py) — the UI wants a live,
        # in-memory view of THIS session's own turns without re-reading a
        # file, and this list is cleared on reset() (see below) while the
        # JSONL log itself is never truncated (logger.py's own contract).
        self.comparisons: list[dict[Condition, TurnRecord]] = []
        self.single_turn_records: list[TurnRecord] = []

    @property
    def run_id(self) -> str:
        return self.controller.current_run_id

    def apply_explicit_goal_update(self, source: GoalUpdateSource, **changes: Any) -> GoalState:
        """The ONLY sanctioned path a UI may use to change goal_state —
        goes through GoalStateManager (never GoalState.with_explicit_update()
        directly), matching this codebase's own single-entry-point rule
        (services/goal_state_manager.py's own docstring).

        This method remains the sanctioned path for genuine explicit
        GoalState updates from human/task/researcher sources. The Phase 7
        one-variable intervention UI does not use this method; that control
        is now correctly implemented as a rho-based replay/counterfactual
        intervention (see ui/experiment_controls.py's own module docstring,
        and services/pipeline.py's replay_turn(), §20.14) rather than a
        GoalState mutation. (Post-delivery fix, Phase 7 fix round: this
        docstring previously named ui/experiment_controls.py's
        "one-variable intervention" control as this method's one real
        caller — that was true of the first Phase 7 delivery, before the
        fix round replaced that control's mechanism; this method has no
        caller in ui/*.py as of the fix round, and is kept here for a
        future/programmatic caller with a genuinely different kind of goal-
        state intervention to make.)"""
        self.goal_state = self._goal_manager.update(self.goal_state, source, **changes)
        return self.goal_state

    def record_comparison(self, results: dict[Condition, TurnRecord]) -> None:
        """Called after ExperimentController.compare() — one entry per
        compare() call, each holding every condition's own TurnRecord for
        that turn (mirrors compare()'s own return shape exactly, so no
        reshaping happens here)."""
        self.comparisons.append(results)

    def record_turn(self, record: TurnRecord) -> None:
        """Called after ExperimentController.run_turn() — the single-
        condition path, kept in its own list rather than folded into
        comparisons (a plain TurnRecord is not a dict[Condition, TurnRecord],
        and conflating the two shapes would make history_for_condition()
        below silently drop single-condition turns or require every caller
        to know which shape it is looking at)."""
        self.single_turn_records.append(record)

    def history_for_condition(self, condition: Condition) -> list[TurnRecord]:
        """Turn-by-turn TurnRecord list for one condition, across every
        compare() call this session has made — ui/trajectory_view.py's own
        primary data source (§20.16's "turn-by-turn trend table/mini-
        chart"). Does not include single_turn_records: run_turn() is a
        single-condition rehearsal path, not part of any condition's own
        comparative trajectory."""
        return [comparison[condition] for comparison in self.comparisons if condition in comparison]

    def latest_comparison(self) -> dict[Condition, TurnRecord] | None:
        return self.comparisons[-1] if self.comparisons else None

    def reset(self, goal_state: GoalState) -> str:
        """Resets an EXISTING session in place: a fresh run_id (delegated
        entirely to ExperimentController.start_new_run() — per that
        method's own docstring, "never clear Streamlit session state by
        hand instead") plus this class's own UI-display bookkeeping
        (comparisons/single_turn_records cleared, goal_state replaced).
        Requires a fresh `goal_state` from the caller (mirrors
        start_new_run() itself not eliciting a new OutcomeBaseline — see
        its own docstring for the same reasoning): a fresh run may want a
        different scenario/G_t entirely, and nothing about resetting run/
        turn history implies which new goal_state to use.

        Post-delivery note: app.py's own "New Run / Reset Demo" button
        (§8's UI component map) does NOT call this method — it discards the
        whole SessionState and rebuilds one from scratch instead, because
        Mock-backend sessions carry a one-shot, pre-scripted LLM-adapter
        queue (see app.py's _init_mock_session() docstring) that resetting
        THIS object in place would leave empty. This method stays here,
        fully correct and independently tested (tests/test_state_manager.py),
        for a caller whose adapter has no such one-shot state to lose — a
        live-only session, a notebook, or a future UI — reusing the exact
        same ExperimentController/adapter rather than reconstructing them."""
        run_id = self.controller.start_new_run()
        self.comparisons.clear()
        self.single_turn_records.clear()
        self.goal_state = goal_state
        return run_id
