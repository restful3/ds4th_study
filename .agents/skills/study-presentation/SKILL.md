---
name: study-presentation
description: Create, revise, review, and prepare paired long-form HTML reports and browser slide decks for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for a report, slides, 발표자료, or a session package for GitHub Pages.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. Those files are the canonical rules and workflow; do not duplicate or override them here.

Use the registry to resolve the current learning-material path. It may point to `source/` for an active study or `archive/` for a completed one. Keep the public path under `docs/studies/<study-slug>` unchanged across that lifecycle.

For a new session, read both `agent-support/templates/study-report/DESIGN.md` and `agent-support/templates/study-deck/DESIGN.md`, then scaffold it with `agent-support/scripts/new-presentation.py`. Unless the user explicitly requests a single artifact, create both `report.html` with `study-report-v1` and the `study-deck-v1` slide entrypoint at `index.html`. Do not hand-build a competing starter or overwrite an existing session directory. Edit only the generated snapshots for session-specific changes.

After creating or editing a report or deck, run:

```bash
python3 agent-support/scripts/build-index.py
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

Inspect both the rendered report and presentation in a browser when visual behavior matters. Prepare local changes by default; commit, push, open a PR, or change Pages settings only when the user explicitly requests that external action.
