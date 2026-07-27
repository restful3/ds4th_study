---
name: localize-image-text
description: Localize English text embedded in instructional raster images into natural Korean while preserving diagrams, screenshots, identifiers, connectors, layout, and non-text pixels. Use for textbook figures, screenshots, infographics, charts, workflow diagrams, and other PNG/JPEG assets whose in-image labels or explanatory copy must be translated and visually quality-checked.
---

# Localize Image Text

Use deterministic mask-and-typeset compositing for diagrams and screenshots. Do not use
generative image editing when unchanged pixels, exact identifiers, or connector geometry matter.

## Workflow

1. Inventory every image consumed by the target document. Map each file to its figure number and
   caption; do not infer the mapping from filename order. For large chapters, create a review
   manifest from the source-language Markdown:

   ```bash
   python .agents/skills/localize-image-text/scripts/inventory.py \
     chapter.md --output /tmp/chapter-images.json
   ```

   The tool deliberately marks every caption-based mapping as
   `inferred-requires-visual-review`. Compare the resulting contact sheet with the source
   Markdown because missing captions, scanned listings, and multi-panel figures can defeat
   automatic grouping.
2. Inspect every source at original size. Classify each string:
   - translate prose, headings, UI labels, and human-readable entity names;
   - preserve IDs, code, property keys, relationship types, URLs, and person names unless the
     user explicitly asks otherwise;
   - retain a short original term in parentheses only when the Korean term would lose precision.
3. Create one JSON localization spec per image. Use `extends` and variables only after comparing
   the actual pixel geometry of every repeated layout. If boxes, labels, or crop boundaries moved,
   use a standalone spec for that sibling instead of relying on proportional scaling.
   Read [spec-format.md](references/spec-format.md) when authoring a new spec.
   When reviewed siblings share the same geometry and differ only in active/inactive text color,
   add a `group` key to base text operations and generate each sibling explicitly:

   ```bash
   python .agents/skills/localize-image-text/scripts/variant.py base.json \
     --output-spec sibling.json --source ../../sibling.jpg \
     --output-image ../sibling-ko.png --coordinate-space 1126x807 \
     --fill future=#b5b5b5
   ```

   The generated file remains a complete standalone spec. Do not use this tool until each
   sibling's actual render has been compared with the base geometry.
4. Render losslessly:

   ```bash
   uv run --with 'pillow>=11,<13' python \
     .agents/skills/localize-image-text/scripts/render.py path/to/spec.json
   ```

5. Build a contact sheet at the document's actual display width:

   ```bash
   uv run --with 'pillow>=11,<13' python \
     .agents/skills/localize-image-text/scripts/contact_sheet.py \
     --inventory /tmp/chapter-images.json \
     --width 900 --columns 1 --output /tmp/localized-review.png
   ```

6. Inspect the rendered pixels, not only the spec. Apply every gate in
   [quality-gates.md](references/quality-gates.md). Fix one defect class across sibling figures,
   rerender, and reinspect.
7. Preserve the source asset. Point only localized documents at the localized PNG. Keep an
   editable spec beside the outputs.

## Required invariants

- Keep output dimensions and aspect ratio identical to the source unless the user requests a
  redesign.
- Save localized raster outputs as PNG so unchanged source pixels are not recompressed.
- Make every changed pixel fall inside a declared operation region; the renderer enforces this.
- Clip typeset text to its declared box and fail when it cannot fit above `min_font_size`.
- Keep arrowheads, shafts, borders, node boundaries, and semantic anchors visible.
- Use a Korean font with verified glyph coverage. Prefer NanumGothic or Noto Sans CJK KR.
- Never claim completion before inspecting both original-size and final-display-size renders.
- For public repositories, follow the repository's copyright/publication policy before adding a
  near-identical translated derivative.

## Iteration discipline

After applying the skill to a real batch, record newly observed failure modes in
`references/quality-gates.md` and, when enforceable, add the check to `scripts/render.py`.
Validate the skill after every material update:

```bash
python3 /home/restful3/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  .agents/skills/localize-image-text
```
