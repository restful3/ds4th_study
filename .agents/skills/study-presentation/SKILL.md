---
name: study-presentation
description: Create, revise, review, and prepare browser-based HTML presentations for ds4th study sessions. Use when a participant mentions an assigned study date or chapter, asks for slides or 발표자료, wants an existing presentation reviewed, or needs a deck prepared for the repository's GitHub Pages site.
---

# Study Presentation

Read `AGENTS.md`, `agent-support/studies.toml`, and `agent-support/procedures/study-presentation.md` completely before changing presentation files. Those files are the canonical rules and workflow; do not duplicate or override them here.

Use the registry to resolve the current learning-material path. It may point to `source/` for an active study or `archive/` for a completed one. Keep the public path under `docs/studies/<study-slug>` unchanged across that lifecycle.

For a new presentation, read `agent-support/templates/study-deck/DESIGN.md` and scaffold it with `agent-support/scripts/new-presentation.py`. Use the canonical `study-deck-v1` template unless the user explicitly requests another format. Do not hand-build a competing starter or overwrite an existing presentation directory. Edit only the generated deck snapshot for session-specific changes.

After creating or editing a deck, run:

```bash
python3 agent-support/scripts/build-index.py
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

Inspect the rendered presentation in a browser when visual behavior matters. Prepare local changes by default; commit, push, open a PR, or change Pages settings only when the user explicitly requests that external action.
