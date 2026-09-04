"""ui/condition_labels.py — shared, participant-facing labels for the three
experimental conditions (Condition enum, models/enums.py).

Centralized here rather than duplicated inside experiment_controls.py/
interaction_view.py/researcher_dashboard.py/trajectory_view.py — every one
of those modules displays a condition name somewhere, and a hand-copied
string in each would drift the moment one of them is edited without the
others. Holds no logic beyond these two dicts.

CONDITION_BLURBS describes the controller-level difference between
conditions (what each one reads/ignores, and rho's role), not any specific
numeric threshold or rule — those stay in README.md and the frozen
specification.
"""

from __future__ import annotations

from models.enums import Condition

CONDITION_LABELS: dict[Condition, str] = {
    Condition.TASK_FOCUSED: "Task Focused",
    Condition.CURRENT_CUE: "Current Cue",
    Condition.DYNAMIC: "Dynamic",
}

CONDITION_BLURBS: dict[Condition, str] = {
    Condition.TASK_FOCUSED: (
        "Uses task context and goals without allowing an explicit affective state to influence policy."
    ),
    Condition.CURRENT_CUE: (
        "Responds to the current human appraisal without carrying the explicit agent affective "
        "state across turns (rho = 0)."
    ),
    Condition.DYNAMIC: (
        "Integrates the current appraisal with the previous agent state across turns (rho > 0)."
    ),
}
