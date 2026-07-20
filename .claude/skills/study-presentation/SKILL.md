---
name: study-presentation
description: Create, revise, review, and prepare paired long-form HTML reports and browser slide decks for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for a report, slides, 발표자료, or a session package for GitHub Pages.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. They are the shared source of truth for Claude Code and Codex.

Resolve study materials through the registry because completed books move from `source/` to `archive/`. Never move the public directory under `docs/studies/<study-slug>` when that happens.

For a new session, read `agent-support/templates/study-report/DESIGN.md` and `agent-support/templates/study-deck/DESIGN.md`, then run `agent-support/scripts/new-presentation.py`. Unless the user explicitly asks for one artifact, create both `report.html` with `study-report-v1` and the `study-deck-v1` slide entrypoint at `index.html`. Never overwrite an existing session directory.

Run the shared index builder and validator after every report or presentation change. Inspect both browser outputs. Prepare changes locally by default, and perform commits, pushes, PR creation, or Pages setting changes only when the user explicitly requests them.
