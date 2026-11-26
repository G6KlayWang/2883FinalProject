# PushT team operating rules

The user approved doc/agent_team_plan.md and the dataset-preparation milestone on 2026-11-26, followed by the ENCODER-001 implementation and bounded Arm A smoke described below. These are implementation authorizations, not another brainstorming approval gate. Read doc/research_proposal.md and doc/implementation_plan.md. Orchestrator: Herdr t1c (confirm the sender on future tasks). Model execution is limited to the explicitly authorized ENCODER-001 diagnostic scope; full encoder, pose-probe and Diffusion Forcing training remain unauthorized. No new nested agents. Do not touch the separate Capstone project or its agents/templates.

**ENCODER-001 authorization update (2026-11-26):** the user now authorizes encoder implementation and one bounded small-data Arm A GPU smoke test. The exact scope and limits in [encoder001-assignment.md](handoffs/encoder001-assignment.md) supersede the earlier blanket training prohibition for that task only. Necessary isolated setup and implementation tests are included. Full training and merge still require the user's later decision. The Server coordinator may use the existing Claude Chrome worker t2j for visible JupyterLab operations; no new nested agents.

Use Herdr hc for messages, status and waits. Report completion/blockers via `hc send t1c "<status and report path>"`. Write substantive evidence to the assigned report path, then remain available for follow-up. Never infer success from a command being launched. Do not put credentials, datasets, caches or checkpoints in Git or messages.

The Executor alone edits implementation/test files. The Reviewer checks code without changing it and writes only review reports. The GitHub manager owns Git mutations. The Server coordinator alone executes remote commands, including Git sync instructions from the GitHub manager. No one changes a running experiment's checkout.

## User-approved PR gate (2026-11-26)

The Claude session is Research Reviewer (existing terminal t1g, stable hc alias pusht-reviewer). A separate Codex PR Reviewer independently checks each PR before merge. Workflow: Executor → Research Reviewer → GitHub manager publishes/synchronizes → PR Reviewer checks live PR → orchestrator reports readiness → user decides merge. Both reviewers report findings to the orchestrator; neither merges. Approval is bound to the exact reviewed head/base and must be renewed after relevant changes. No automatic merge or draft-to-ready transition. The training approval gate remains separate and unchanged.
