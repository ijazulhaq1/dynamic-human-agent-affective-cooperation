"""GoalStateManager — M3. Blueprint §6.3, spec §8.1/§20.4.

The blueprint gives this module almost no body: "GoalStateManager
(services/goal_state_manager.py) only ever calls
GoalState.with_explicit_update(); it never mutates GoalState directly, so
'H_t silently rewrites G_t' is unrepresentable in this codebase, not merely
disallowed by convention" (§6.3), and the turn-execution-flow table (§7,
step 2) names its one method: "GoalStateManager.update (only from
explicit/task callers)... Never called from the estimator or policy
layer."

No class body, method signature beyond that name, or restriction on which
GoalUpdateSource values are legitimate is ever given. Implemented here as a
thin, single-method wrapper around GoalState.with_explicit_update() itself
(models/goal_state.py, already built and frozen in Phase 0) — its entire
value is being the ONE class every explicit/task-event caller in this
codebase is meant to go through, so a future caller wanting to mutate G_t
has exactly one method to call, never GoalState.with_explicit_update()
directly. That boundary is enforced by convention (no other module in this
codebase currently calls with_explicit_update() directly, and this
project's test suite would need to be extended if one started to), not by
this class refusing anything at runtime — see the judgment-call note below
for why `source` is deliberately NOT restricted further.

Judgment call (documented gap, flagged for review): "no affective cue alone
may call this" is the invariant the §6.3 docstring above protects. This is
already fully enforced elsewhere in the codebase without GoalStateManager's
help — AppraisalEstimator, StateTransitionEngine and PolicyEngine (M2/M5/M6)
none of them import or reference GoalStateManager or with_explicit_update()
at all, so there is no code path by which H_t's own computation could reach
a G_t mutation regardless of what GoalStateManager does internally. Given
that, update() below does not additionally restrict which GoalUpdateSource
value a caller passes — GoalUpdateSource's three members (HUMAN_EXPLICIT,
TASK_EVENT, RESEARCHER_CONFIG) are all documented elsewhere as legitimate
real update sources (§5.3's own field table; RESEARCHER_CONFIG plausibly
covers a live one-variable intervention, per the UI component map's
"experiment_controls.py... one-variable intervention", §8, not only initial
construction), and nothing in the blueprint singles any of them out as
illegitimate specifically through this method. A researcher should
double-check that reading against the frozen specification directly (not
just this blueprint's compressed table) before this matters for a real
study.
"""

from __future__ import annotations

from typing import Any

from models.enums import GoalUpdateSource
from models.goal_state import GoalState


class GoalStateManager:
    """M3 (§8.1, §20.4)."""

    def update(self, g_t: GoalState, source: GoalUpdateSource, **changes: Any) -> GoalState:
        """The sole sanctioned entry point for mutating G_t in this codebase
        (never call GoalState.with_explicit_update() directly from any
        other module). Returns a new GoalState; never mutates g_t in place
        — see GoalState.with_explicit_update()'s own docstring for why that
        goes through full model_validate() rather than
        model_copy(update=...)."""
        return g_t.with_explicit_update(source, **changes)
