---
name: study-presentation
description: Create, revise, review, and prepare paired long-form HTML reports and browser slide decks for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for a report, slides, 발표자료, or a session package for GitHub Pages.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. Those files are the canonical rules and workflow; do not duplicate or override them here.

Use the registry to resolve the current learning-material path. It may point to `source/` for an active study or `archive/` for a completed one. Keep the public path under `docs/studies/<study-slug>` unchanged across that lifecycle.

For a new session, read `agent-support/templates/STUDY_SESSION_BLUEPRINT.md` and both template `DESIGN.md` files completely, then inspect the Chapter 1 reference named by the blueprint and scaffold with `agent-support/scripts/new-presentation.py`. Unless the user explicitly requests a single artifact, create both `report.html` with `study-report-v1` and the `study-deck-v1` slide entrypoint at `index.html`. Do not hand-build a competing starter or overwrite an existing session directory. Edit only the generated snapshots for session-specific changes.

Follow the pipeline in order: (A) audit the raw chapter material and record its claims, examples, evidence, terms, and visualizable relationships; (B) finish and render `report.html`; (C) only after the report gate passes, replace the scaffold deck content by deriving it from the report. Do not author session-specific slide content in parallel with an unfinished report. If a slide needs a claim absent from the report, update and revalidate the report first.

Treat the report as a long-form publication, not a slide transcript. Preserve the canonical ConnectBrick-derived report component hierarchy and the blueprint's problem-to-decision logic. Add source-backed tables and newly composed SVG diagrams when they materially explain the chapter. Give report sections, tables, and figures stable IDs, and mark report figures that the deck must carry with `data-deck-use="required"`. The report gate requires complete argument coverage, captions and sources, working TOC/lightbox, desktop/mobile rendering, and an inspected A4 PDF.

Derive the deck's narrative, claims, terminology, tables, SVGs, and CSS relationships from the approved report. Reuse a report visual directly when it is legible at slide scale; otherwise make a faithful slide-scale adaptation without changing its meaning. Keep `data-report-source="report.html"` on the deck main element and put valid `data-report-refs` on every slide. Preserve automatic report/deck TOCs and verify that every report section and every required figure is covered.

After creating or editing a report or deck, run:

```bash
python3 agent-support/scripts/build-index.py
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

Inspect both the rendered report and presentation in a browser when visual behavior matters. Prepare local changes by default; commit, push, open a PR, or change Pages settings only when the user explicitly requests that external action.
