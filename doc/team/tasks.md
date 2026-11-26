# PushT team task ledger

Plan: ../agent_team_plan.md (user-approved 2026-11-26).
Training authorization: NONE.

| Task | Owner | Status | Deliverable |
| --- | --- | --- | --- |
| TEAM-001 | Orchestrator | In progress | Four project-specific Herdr workers and role briefs |
| DATA-001-GIT | GitHub manager | Pending | Baseline, isolated dataset branch/worktree |
| DATA-001-PREFLIGHT | Server coordinator | Pending | Server/data/environment evidence |
| DATA-001-CODE | Executor | Pending | Extraction, schema inspection, deterministic splits, loader and tests |
| DATA-001-REVIEW | Reviewer | Pending | Research/code review and any revision loop |
| DATA-001-SYNC | GitHub manager + Server coordinator | Pending | PR, reviewed SHA, verified remote checkout |
| DATA-001-RUN | Server coordinator | Pending | Dataset schema/splits/loader verification; no training |

Ruling: use fixed split seed 42 unless existing verified metadata requires another seed; this is a reproducibility default, not hyperparameter selection.
Ruling: preserve all pre-existing unrelated files and Capstone agents/templates.
