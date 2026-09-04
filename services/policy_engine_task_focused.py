"""TASK_FOCUSED reduced rule table. Blueprint §6.5, spec §20.11.1.

Uses only G_t and the task-derived half of D_t; H_t, c_t, A_t are
structurally absent from every condition below — none of the predicates
imported from services.policy_engine that ARE used here (evidence_available,
task_requires_evidence) touch H_t/c_t/A_t, and PolicyEngine.select() forces
Ctx.h_t/c_t/a_t to None on the TASK_FOCUSED branch regardless of what is
passed to it, so this table has no way to read them even if a rule tried.
"""

from __future__ import annotations

from models.enums import Policy, RationaleCode
from services.policy_engine import PolicyConfig, Rule, evidence_available, task_requires_evidence


def build_task_focused_rules(cfg: PolicyConfig) -> list[Rule]:
    return [
        Rule(
            id="TF_SAFETY",
            condition=lambda ctx: ctx.g_t.safety_risk >= cfg.task_focused_safety_risk_gate,
            primary=Policy.DEFER,
            secondary=lambda ctx: Policy.INFORM if evidence_available(ctx, cfg) else Policy.CLARIFY,
            rationale=RationaleCode.SAFETY_OVERRIDE,
        ),
        Rule(
            id="TF_TASK_CLARIFY",
            condition=lambda ctx: (
                ctx.d_t.evidence_ambiguity >= cfg.task_focused_clarify_d_amb_min
                or ctx.d_t.goal_conflict >= cfg.task_focused_clarify_d_goal_min
            ),
            primary=Policy.CLARIFY,
            secondary=lambda ctx: Policy.INFORM if task_requires_evidence(ctx) else None,
            rationale=RationaleCode.CLARIFY_NEEDED,
        ),
        Rule(
            id="TF_INFORM",
            condition=lambda ctx: task_requires_evidence(ctx),
            primary=Policy.INFORM,
            secondary=None,
            rationale=RationaleCode.INFORM_NEEDED,
        ),
        Rule(
            id="TF_DEFAULT",
            condition=lambda ctx: True,
            primary=Policy.INFORM,
            secondary=None,
            rationale=RationaleCode.MINIMAL_SUPPORT,
        ),
    ]
