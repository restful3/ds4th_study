# Visual quality gates

Inspect original-size outputs and a contact sheet rendered at the consuming document's width.

## Text

- Every intended English prose string is replaced; intentional code and IDs remain.
- Korean wording is natural in context and terminology matches the surrounding document.
- No missing Hangul glyphs, tofu boxes, clipped descenders, orphan punctuation, or residual source
  letters remain under a mask.
- Font size and weight preserve the source hierarchy. Sibling figures use the same translations,
  line breaks, sizes, and active/inactive colors.
- Translate reader-facing inactive or faded labels too. Preserve their gray emphasis level instead
  of leaving the English text merely because it is visually de-emphasized.
- Do not assume proportional scaling makes repeated templates identical. Inspect every sibling
  for source-font fragments at the right and bottom edges of masks; expand masks only within the
  containing shape or add a per-image cleanup operation.
- A generated repeated-layout variant must remain a complete standalone spec. Verify that every
  requested group override matched at least one operation, then inspect every sibling render;
  shared coordinates do not waive per-image visual review.
- Treat rotated source text as a quadrilateral footprint, not as the visible centerline alone.
  Inspect every corner of the erased area at high zoom for residual Latin glyph fragments.
- When a prose block or cropped source glyph reaches the canvas boundary, extend its cleanup mask
  exactly to that boundary. A one- to six-pixel uncovered strip can leave punctuation or the final
  Latin letter visible only at original size.
- Long Korean text has balanced line lengths. Do not solve overflow by shrinking below projector
  or document readability.

## Geometry

- Masks do not erase icon edges, node outlines, table borders, graph edges, arrow shafts, or
  arrowheads.
- Preserve label brackets, braces, divider lines, and other small framing marks at both sides of a
  text mask; these are easy to mistake for expendable source text.
- Connectors still touch the intended semantic anchor. Redraw a connector when its source text
  mask leaves a floating or head-only arrow.
- When a mask must overlap an original border or connector, prefer `restore_line` or
  `restore_bezier` to copy the exact source pixels back. Approximate redrawing is a last resort
  because it changes line weight, antialiasing, and curvature.
- Repeated boxes align to a common grid. No translated label protrudes into neighboring shapes.
- Rotated labels remain centered within their segment and do not cross boundary lines.
- For chart-axis localization, keep the mask inside the axis-title gutter. At final display size,
  verify that adjacent tick labels, decimal points, minus signs, gridlines, and plot marks are
  still intact; a narrow vertical title can sit only a few pixels from numeric ticks.
- Check target rings, selection halos, radio buttons, and other small circular markers after
  masking nearby labels. If a mask overlaps one, restore both the outer ring and the inner mark;
  use the ellipse operation's `outline` and `width` fields rather than a borderless fill.

## Fidelity

- Output dimensions equal source dimensions.
- Unchanged pixels outside declared operation regions are byte-for-byte equal in decoded RGB.
- Technical identifiers, relationship types, property keys, IDs, URLs, and names are unchanged
  unless translation is explicitly part of the figure's purpose.
- Localized PNGs are referenced only by localized documents; source-language documents retain the
  original assets.

## Batch audit

- Treat caption-derived inventory as untrusted until the original Markdown and rendered contact
  sheet agree. A missing caption can silently attach code scans or equations to the next figure.
- Inventory count equals localized output count.
- Every localized Markdown reference resolves to an existing file.
- Inspect all siblings when one repeated-template defect is found.
- After the contact-sheet pass, reopen dense, rotated, and edge-touching figures individually at
  original size; contact-sheet downscaling can hide single-letter remnants and broken thin lines.
- Rerun repository validators after updating consuming documents.
