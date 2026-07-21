---
name: study-presentation
description: Create, revise, review, and prepare paired long-form HTML reports and browser slide decks for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for a report, slides, 발표자료, or a session package for GitHub Pages.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. They are the shared source of truth for Claude Code and Codex.

Resolve study materials through the registry because completed books move from `source/` to `archive/`. Never move the public directory under `docs/studies/<study-slug>` when that happens.

For a new session, read `agent-support/templates/STUDY_SESSION_BLUEPRINT.md` and both template `DESIGN.md` files completely, then inspect the Chapter 1 reference named by the blueprint and run `agent-support/scripts/new-presentation.py`. Unless the user explicitly asks for one artifact, create both `report.html` with `study-report-v1` and the `study-deck-v1` slide entrypoint at `index.html`. Never overwrite an existing session directory.

Treat the report as a long-form publication, preserve the blueprint's problem-to-decision logic, and create source-backed tables and newly composed SVG diagrams where they explain the chapter. Preserve automatic report/deck TOCs and the report image lightbox. Inspect both browser outputs at desktop and mobile widths, exercise image zoom and keyboard closing, and print the report to A4 PDF.

After creating or editing a report or deck, run:

```bash
python3 agent-support/scripts/build-index.py
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

Prepare changes locally by default, and perform commits, pushes, PR creation, or Pages setting changes only when the user explicitly requests them.
