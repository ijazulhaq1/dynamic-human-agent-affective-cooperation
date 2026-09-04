"""OutcomeBaselineStore — services/outcome_baseline.py. Blueprint §6 repo
layout ("immutable pre-interaction value-priority store", §20.4/§20.19).
The model itself (OutcomeBaseline — frozen, MappingProxyType-protected
against even in-place dict mutation) already exists in
models/goal_state.py, built in Phase 0.

Judgment call (documented gap, flagged for review): the blueprint never
gives this module a class, method signature, or storage/lookup design of
any kind beyond that one-line repo-layout description and the model class
itself. Two things this design must satisfy, both taken directly from
elsewhere in the blueprint rather than invented: (1) TurnRecord.
outcome_baseline_value_priorities (§5.7) is a REQUIRED field ("copied from
OutcomeBaseline, read-only") on every single TurnRecord Pipeline (a
sibling module, this same phase) assembles, so something must let Pipeline
look up "the OutcomeBaseline for this run_id" at record-assembly time; (2)
OutcomeBaseline itself is "elicited before turn 1, frozen for the run's
lifetime" (§5.3) — i.e. exactly one per run_id, set once. OutcomeBaseline
Store below is a minimal per-run_id store satisfying both: elicit()
creates and stores the one OutcomeBaseline for a run_id (raising if that
run_id already has one, rather than silently overwriting it — the same
fail-fast-on-ambiguity discipline this project applied to Phase 1's
required_evidence status vocabulary), and get() looks it up (raising, not
defaulting, if turn-1 elicitation never happened for that run_id — a run's
OutcomeBaseline missing entirely is a setup bug that should surface
immediately, not silently produce a TurnRecord with a fabricated or empty
outcome_baseline_value_priorities).

services/pipeline.py's own Pipeline.__init__ takes an OutcomeBaselineStore
instance — documented there as an addition beyond the blueprint's shown
constructor signature, for the same reason: nothing in the given
run_turn()/compare() pseudocode threads OutcomeBaseline data into
_assemble_turn_record at all.
"""

from __future__ import annotations

from models.goal_state import OutcomeBaseline


class OutcomeBaselineStore:
    def __init__(self) -> None:
        self._by_run_id: dict[str, OutcomeBaseline] = {}

    def elicit(self, run_id: str, value_priorities: dict[str, float]) -> OutcomeBaseline:
        """Elicit and freeze the ONE OutcomeBaseline for this run_id. Must
        be called before turn 1 of a run — Pipeline._assemble_turn_record
        will raise via get() below if it wasn't. Raises ValueError on a
        second call for the same run_id: OutcomeBaseline is documented as
        elicited once, frozen for the run's lifetime, not re-elicitable."""
        if run_id in self._by_run_id:
            raise ValueError(
                f"OutcomeBaseline already elicited for run_id={run_id!r} — "
                "it is frozen for the run's lifetime and cannot be re-elicited."
            )
        baseline = OutcomeBaseline(run_id=run_id, value_priorities=value_priorities)
        self._by_run_id[run_id] = baseline
        return baseline

    def get(self, run_id: str) -> OutcomeBaseline:
        """Raises KeyError — never returns a default/fabricated baseline —
        if elicit() was never called for this run_id. A missing
        pre-interaction baseline is a setup bug, not a value to guess at."""
        try:
            return self._by_run_id[run_id]
        except KeyError:
            raise KeyError(
                f"No OutcomeBaseline elicited for run_id={run_id!r} — "
                "call OutcomeBaselineStore.elicit() before the run's first turn."
            ) from None
