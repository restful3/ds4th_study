# Repository Guidelines

## Project Structure & Module Organization
Chapter-specific directories such as `10_8_komponentenbasiertes_denken/` combine Markdown translations, extracted page images (`_page_*.jpeg`), and the sliced PDF for that section. Mirror the numbering prefix `NN_` and keep German titles so every folder lines up with `metadata.json`, which tracks the page ranges inside `fundamentals-of-software-architecture-2.pdf`. The consolidated PDFs live in `chapters/`, while root-level companions (`Architectural_Essentials.pdf`, `info-graphic.png`, Korean briefings, and audio/video files) provide supplemental study material referenced throughout the notes.

## Build, Test, and Development Commands
- `markdownlint "**/*.md"` – lint headings, spacing, and fenced code before opening a PR; install via `npm i -g markdownlint-cli` if needed.
- `codespell --ignore-words=spellings.txt **/*.md` – catch accidental typos without flagging domain terms once you maintain the optional allow-list.
- `jq '.chapters[] | select(.file=="10_8_komponentenbasiertes_denken.pdf")' metadata.json` – verify that any renamed PDF or new excerpt stays synchronized with the metadata manifest.

## Coding Style & Naming Conventions
Write Markdown in English, German, or Korean exactly as in the source text, keeping level-one headings for chapter titles and two blank lines before horizontal rules. Use descriptive alt text for every embedded image file in a chapter folder and prefer relative links (`![](_page_1_Picture_2.jpeg)`). Preserve camelCase in technical identifiers quoted from the book, but default to sentence case elsewhere. New directories should follow the `NN_title-with-underscores` pattern to keep chronological ordering stable.

## Testing Guidelines
Treat linting and spell-checking as the primary automated “tests.” For visual verification, open the updated Markdown in a Markdown previewer to ensure figures render and that anchor links like `#page-2-0` still resolve. When touching `metadata.json`, rerun the `jq` query above and diff the output to confirm only the intended chapter metadata moved. If you add media, validate checksums and confirm the referenced filename matches the embedded link.

## Commit & Pull Request Guidelines
Use short, present-tense commit subjects prefixed with the chapter identifier (e.g., `18_event update summary`). Reference the affected files in the body and note whether assets or metadata were modified. Pull requests should describe the motivation, list translated chapters or sections, mention any tooling versions used, and attach before/after screenshots when visual assets change. Link to tracking issues or TODOs so downstream agents understand why the excerpt was edited.
