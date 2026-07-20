---
name: study-presentation
description: Create, revise, review, and prepare browser-based HTML presentations for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for slides or 발표자료, wants an existing presentation reviewed, or needs a deck prepared for the repository's GitHub Pages site.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. They are the shared source of truth for Claude Code and Codex.

Resolve study materials through the registry because completed books move from `source/` to `archive/`. Never move the public directory under `docs/studies/<study-slug>` when that happens.

For a new presentation, read `agent-support/templates/study-deck/DESIGN.md` and run `agent-support/scripts/new-presentation.py`. Use the canonical `study-deck-v1` template unless the user explicitly asks for another format, and never overwrite an existing presentation directory.

Run the shared index builder and validator after every presentation change. Prepare changes locally by default, and perform commits, pushes, PR creation, or Pages setting changes only when the user explicitly requests them.
