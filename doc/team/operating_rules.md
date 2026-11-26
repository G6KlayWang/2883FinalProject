# PushT team operating rules

The user approved doc/agent_team_plan.md and the first dataset-preparation milestone on 2026-11-26. This is an implementation authorization, not another brainstorming approval gate. Read doc/research_proposal.md and doc/implementation_plan.md. Orchestrator: Herdr t1c (confirm the sender on future tasks). No encoder, pose-probe, or Diffusion Forcing training, including pilot training, is authorized. No nested agents. Do not touch the separate Capstone project or its agents/templates.

Use Herdr hc for messages, status and waits. Report completion/blockers via `hc send t1c "<status and report path>"`. Write substantive evidence to the assigned report path, then remain available for follow-up. Never infer success from a command being launched. Do not put credentials, datasets, caches or checkpoints in Git or messages.

The Executor alone edits implementation/test files. The Reviewer checks code without changing it and writes only review reports. The GitHub manager owns Git mutations. The Server coordinator alone executes remote commands, including Git sync instructions from the GitHub manager. No one changes a running experiment's checkout.
