"""FallbackTemplates — deterministic template responses per Policy.
Blueprint §6.8 (repo-layout comment: "llm/fallback.py: deterministic
template responses per Policy"), used by
services.response_generator.ResponseGenerator whenever GeneratorStatus is
FALLBACK_TEMPLATE — i.e. the live LLM call raised TimeoutError or
GeneratorError (see llm/adapter.py).

FallbackTemplates.render(policy_state, task_context) composes a
deterministic string from primary + secondary + hard_constraints TOGETHER —
never just primary alone, so a combination like DEFER+INFORM is not silently
collapsed to DEFER's template (§6.8 note — this was itself an
implementation-review fix the blueprint had already made before this
project reached Phase 4: an earlier draft's render(p_t.primary) signature
dropped secondary/hard_constraints exactly when the fallback path is under
the most stress). render() must never be silent — every Policy member has
both a primary template and a secondary clause below, so there is no
primary/secondary combination that falls through unhandled. It DOES raise
for one specific case: an unrecognized hard-constraint code (see below).

Implementation-review note (documented gap, flagged for review): the
blueprint gives no per-Policy wording spec at all — only the structural
requirement above. The template strings here are authored for this
prototype and should be reviewed/replaced by a researcher before a real
study; task_context is accepted per the documented
render(policy_state, task_context) signature but this implementation does
not vary wording by its content, since no scenario-specific fallback
wording spec is given either.

Fix (post-delivery review): hard_constraints must be REALIZED in
participant-facing language, not printed as internal control labels. The
first version of this file did `", ".join(policy_state.hard_constraints)`
straight into the response text, so a participant could see literal
strings like "(AUTONOMY_HIGH, NO_UNDUE_INFLUENCE noted.)" — an internal
enum-adjacent code, never meant to reach a study participant, leaking
through the one path (the LLM-failure fallback) that is supposed to be
the SAFEST, most controlled output the system produces. _HARD_CONSTRAINT_
CLAUSES below maps each known code to a natural-language clause instead.
An unrecognized code raises ValueError rather than being silently dropped:
since this fallback path only runs when the primary generation path has
already failed, a hard constraint silently going unrealized here — instead
of failing loudly in testing — is exactly the kind of safety-relevant gap
that should never reach a live run undetected.
"""

from __future__ import annotations

from typing import Any

from models.enums import Policy
from models.policy import PolicyState

_PRIMARY_TEMPLATES: dict[Policy, str] = {
    Policy.INFORM: "Here's some information that's relevant to this.",
    Policy.ACKNOWLEDGE: "I hear where you're at with this.",
    Policy.CLARIFY: "Could you help clarify a bit more before we go further?",
    Policy.CHALLENGE: "I'd like to gently question part of that before we continue.",
    Policy.REDIRECT: "Let's take a moment to reconsider the direction here.",
    Policy.DEFER: "I'm going to hold off on this for now.",
}

_SECONDARY_CLAUSES: dict[Policy, str] = {
    Policy.INFORM: " I can also share some relevant information.",
    Policy.ACKNOWLEDGE: " I also want to acknowledge where you're at with this.",
    Policy.CLARIFY: " It would also help to clarify a detail or two.",
    Policy.CHALLENGE: " I'd also gently push back on part of this.",
    Policy.REDIRECT: " I'd also suggest reconsidering the direction.",
    Policy.DEFER: " For now, I'm also holding off on this part.",
}

# Judgment call (documented gap, flagged for review): the blueprint defines
# AUTONOMY_HIGH and NO_UNDUE_INFLUENCE as policy/hard-constraint machinery
# (models/goal_state.py, services/policy_engine.py._hard_constraints) but
# never gives participant-facing wording for either. These two clauses are
# authored for this prototype and should be reviewed/replaced by a
# researcher before a real study. Every hard-constraint code currently
# ever produced by services/policy_engine.py._hard_constraints() has an
# entry here (see that function: the return list is always
# ["NO_UNDUE_INFLUENCE"], optionally with "AUTONOMY_HIGH" appended) — a
# third code would need a clause added here before it could ever reach a
# participant through this fallback path.
_HARD_CONSTRAINT_CLAUSES: dict[str, str] = {
    "AUTONOMY_HIGH": " The choice remains yours, and I won't push you toward a particular option.",
    "NO_UNDUE_INFLUENCE": " I won't use pressure or emotional leverage to influence your decision.",
}


class FallbackTemplates:
    def render(self, policy_state: PolicyState, task_context: dict[str, Any]) -> str:
        parts = [_PRIMARY_TEMPLATES[policy_state.primary]]
        if policy_state.secondary is not None:
            parts.append(_SECONDARY_CLAUSES[policy_state.secondary])
        for constraint in policy_state.hard_constraints:
            clause = _HARD_CONSTRAINT_CLAUSES.get(constraint)
            if clause is None:
                raise ValueError(f"Unknown hard constraint: {constraint}")
            parts.append(clause)
        return "".join(parts)
