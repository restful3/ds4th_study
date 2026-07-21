---
name: study-presentation
description: Create, revise, review, and prepare paired long-form HTML reports and browser slide decks for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for a report, slides, 발표자료, or a session package for GitHub Pages.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. They are the shared source of truth for Claude Code and Codex.

Resolve study materials through the registry because completed books move from `source/` to `archive/`. Never move the public directory under `docs/studies/<study-slug>` when that happens.

For a new session, read `agent-support/templates/STUDY_SESSION_BLUEPRINT.md` and both template `DESIGN.md` files completely, then inspect the Chapter 1 reference named by the blueprint and run `agent-support/scripts/new-presentation.py`. Unless the user explicitly asks for one artifact, create both `report.html` with `study-report-v1` and the `study-deck-v1` slide entrypoint at `index.html`. Never overwrite an existing session directory.

Work strictly in three phases: audit the raw chapter material; complete and validate the long-form report; only then derive the session-specific deck from that approved report. Do not write the real deck in parallel with an incomplete report. If a slide needs a new claim, example, or qualification, add it to the report and pass the report gate again before using it.

Treat the report as a long-form publication, preserve the blueprint's problem-to-decision logic, and create source-backed tables and newly composed SVG diagrams where they explain the chapter. Give report sections, tables, and figures stable IDs; mark visuals required in the deck with `data-deck-use="required"`. The report gate includes complete argument coverage, captions and sources, TOC/lightbox behavior, desktop/mobile rendering, and an inspected A4 PDF.

Build the deck from the report's narrative, claims, terms, tables, SVGs, and CSS relationships. Reuse report visuals when legible and otherwise adapt them faithfully for slide scale. Keep `data-report-source="report.html"` on the deck main element and valid `data-report-refs` on every slide, covering every report section and required visual. Preserve automatic report/deck TOCs and inspect both browser outputs at desktop and mobile widths.

After creating or editing a report or deck, run:

```bash
python3 agent-support/scripts/build-index.py
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

Prepare changes locally by default, and perform commits, pushes, PR creation, or Pages setting changes only when the user explicitly requests them.
