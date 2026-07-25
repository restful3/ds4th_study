---
name: study-materials
description: Set up a study textbook's runnable code and data, place upstream chapter sources under the matching book chapters, and build explainer-based chapter notebooks. Use when a participant mentions a new textbook, asks for 교재 코드 실행 환경, 챕터별 src 배치, 해설 노트북, data manifest, or says a chapter notebook fails verification.
---

# Study Materials

Read `AGENTS.md`, `agent-support/studies.toml`, `agent-support/procedures/study-materials.md`, and `agent-support/templates/study-materials/DESIGN.md` completely before touching study materials. They are the shared source of truth for Claude Code and Codex.

Resolve textbook paths through the registry because completed books move from `source/` to `archive/`. Each textbook carries its own `study.toml` and `.venv`; the shared tooling lives once in `agent-support/studykit/`. Never copy `studykit` into a textbook folder.

Use the CLIs rather than hand-rolling equivalents:

```bash
python3 "source/<교재>/setup_env.py"                                    # 환경 구축
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" --list
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" <chXX> [--dry-run|--embed]
python3 agent-support/scripts/study-verify.py "source/<교재>" [--lint|--no-urls|--execute]
```

Never guess the chapter-to-source mapping. Upstream repositories often keep MEAP numbering, so directory names and book chapters diverge and matching counts produces wrong placements. Confirm each mapping with distinctive keyword frequencies in the chapter's original markdown, and separate directories with no final-edition counterpart into `meap-only/`.

Treat `[mapping.listings.chXX]` in `study.toml` as authoritative for listing numbers. A single per-chapter offset cannot express a book that inserts a code-free listing mid-chapter, which shifts everything after it. Offsets differ per chapter, so never carry one chapter's value to another. Declare broken upstream listings — zero-byte or duplicated files — as `{ source = "explainer" }` and carry the explainer text instead, correcting any query that would mutate the graph.

Build notebooks only for chapters that have code in `src`. Generate the skeleton first, fill every `TODO(agent)` from the explainer, embed figures as notebook attachments, then declare `listing_coverage` for every book listing as `executed`, `substituted`, `documented-only`, or `optional` and set `status` to `complete`. Embed figures as attachments; remote `<img>` URLs and relative paths have both failed to render, and a 200 response does not prove a figure displays. Link in-repo targets with relative paths so the notebook works before anything is pushed.

Check external service requirements per chapter before promising execution. Neo4j editions and plugins can be mutually exclusive — `n10s` crashes on Enterprise while `seedUri` and `IS NODE KEY` require it — so plan container switching. Enterprise evaluation use is free, but the user must accept the licence themselves. Route OpenAI-dependent listings through the endpoint named by `[llm].env_file`, which is referenced from outside the repository so no secret is copied in, and override hard-coded model constants at runtime instead of editing upstream files.

Run the verification gate before reporting anything complete. `--lint` and the default completion gate differ; a `draft` notebook only needs lint, so without that distinction a partial notebook reports as passing.

```bash
python3 agent-support/scripts/study-verify.py "source/<교재>"
python3 -m unittest discover -s agent-support/tests
```

Prepare changes locally by default, and perform commits, pushes, or PR creation only when the user explicitly requests them.
