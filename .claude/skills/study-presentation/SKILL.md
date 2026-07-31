---
name: study-presentation
description: Create, revise, review, and prepare paired long-form HTML reports and browser slide decks for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for a report, slides, 발표자료, or a session package for GitHub Pages.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. Those files are the shared source of truth for Claude Code and Codex; do not duplicate or override them here.

Use the registry to resolve the current learning-material path. It may point to `source/` for an active study or `archive/` for a completed one. Keep the public path under `docs/studies/<study-slug>` unchanged across that lifecycle.

For a new session, read `agent-support/templates/STUDY_SESSION_BLUEPRINT.md` and both template `DESIGN.md` files completely, then inspect the Chapter 1 reference named by the blueprint and scaffold with `agent-support/scripts/new-presentation.py`. Unless the user explicitly requests a single artifact, create both `report.html` with `study-report-v1` and the `study-deck-v1` slide entrypoint at `index.html`. Do not hand-build a competing starter or overwrite an existing session directory. Edit only the generated snapshots for session-specific changes.

Follow the pipeline in order: (A) audit the raw chapter material and record its claims, examples, evidence, terms, and visualizable relationships; (B) finish and render `report.html`; (C) only after the report gate passes, replace the scaffold deck content by deriving it from the report. Do not author session-specific slide content in parallel with an unfinished report. If a slide needs a claim absent from the report, update and revalidate the report first.

Preserve the source's claim strength and ownership while rewriting: do not turn “may,” “can,” a conditional mechanism, or “the authors observed” into “does,” “proves,” “guarantees,” or an unattributed general fact. Carry attribution and comparison conditions through section summaries, deck text, captions, and SVG labels. If the report deliberately proposes a more conservative decision rule or an operational extension, label it visibly as the report author's synthesis rather than the book's claim. Treat dated model demonstrations as historical snapshots, not current product benchmarks. Define recurring specialist terms at first use as well as in the glossary; a glossary alone does not make the preceding explanation beginner-friendly.

Write the report in its own voice. Attach attribution to source-dependent claims, observations, numbers, excerpts, execution results, and the reconstruction scope of a table or figure — not to the grammatical subject of a sentence that explains how a general technique works. Apply the delete-the-source test: if removing the source name preserves the sentence's truth, conditions, timeframe, claim ownership, and scope, make the technique the subject and keep the citation in the same or the next sentence. Keep the source as subject where the source itself is what the sentence describes — the authors' claims and recommendations, book-snapshot versus re-run comparisons, the spec version the book followed, and quirks or gaps in its code — and name it precisely (`the authors`, `the book's Listing N`, `the book's execution snapshot`) rather than "the textbook" generically. Never let the reordering promote an author's conditional observation into a universal fact. In figure and table notes, lead with how to read the visual and its interpretation limits, and place the reconstruction-and-source sentence last, unless a source-by-source contrast is itself the visual's axis. This is a human-review item; do not gate it with a string check.

Keep comparison rows at the same semantic level. Do not present a model's implementation behavior as a peer of formal categories without a visible group boundary and an explanation of the distinction. Likewise, distinguish a system category's stated decision authority from evidence that the interface and operating process actually preserve that authority: human-in-the-loop requires usable review, rejection, audit, and rollback paths, not a nominal approval click.

Treat the report as a long-form publication, not a slide transcript. Preserve the canonical ConnectBrick-derived report component hierarchy and the blueprint's problem-to-decision logic. Add source-backed tables and newly composed SVG diagrams when they materially explain the chapter. Give report sections, tables, and figures stable IDs, and mark report figures that the deck must carry with `data-deck-use="required"`. The report gate requires complete argument coverage, captions and sources, working TOC/lightbox, desktop/mobile rendering, and an inspected A4 PDF.

Derive the deck's narrative, claims, terminology, tables, SVGs, and CSS relationships from the approved report. Reuse a report visual directly when it is legible at slide scale; otherwise make a faithful slide-scale adaptation without changing its meaning. Keep `data-report-source="report.html"` on the deck main element and put valid `data-report-refs` on every slide. Preserve automatic report/deck TOCs and verify that every report section and every required figure is covered.

When revising either artifact, re-audit the paired report and every slide that cites a changed section, table, or figure. The deck must neither lag behind changed report claims nor introduce claims absent from the report. Update conditions, terminology, visible report-reference labels—including roadmap/body text—and trace metadata together, then pass both gates again.

After creating or editing a report or deck, run:

```bash
python3 agent-support/scripts/build-index.py
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

Inspect both the rendered report and presentation in a browser when visual behavior matters. For Korean long-form text, verify at the actual desktop, narrow-mobile, and print widths that line wrapping keeps words intact (`word-break: keep-all` with an overflow fallback where appropriate); a clean DOM is not evidence that Korean words are not splitting on screen. Prepare changes locally by default, and perform commits, pushes, PR creation, or Pages setting changes only when the user explicitly requests them.
