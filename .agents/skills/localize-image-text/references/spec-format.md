# Localization spec format

Paths are resolved relative to the concrete spec file. Coordinates are expressed in
`coordinate_space`; the renderer scales them to the source dimensions.

```json
{
  "title": "Figure 1 Korean localization",
  "extends": "optional-template.json",
  "source": "../../source.jpg",
  "output": "../figure-1-ko.png",
  "coordinate_space": [1200, 700],
  "variables": {"inactive": "#9a9a9a"},
  "operations": [
    {"type": "rect", "box": [20, 20, 300, 100], "fill": "#ffffff"},
    {
      "type": "text",
      "box": [30, 25, 280, 90],
      "text": "자연스러운\\n한국어 설명",
      "font": "bold",
      "font_size": 24,
      "min_font_size": 18,
      "fill": "#111111",
      "align": "left",
      "valign": "top",
      "line_gap": 5
    }
  ]
}
```

## Fields

- `extends`: inherit another JSON spec. Child fields override parent fields. `${variable}` strings
  are substituted after merging. Use inheritance only when the repeated source images have matching
  pixel geometry; visual similarity alone is not sufficient.
- `source`, `output`: required on the concrete spec. Output must be PNG.
- `coordinate_space`: optional design width and height. Defaults to source dimensions.
- `variables`: string substitutions usable anywhere, for example `"fill": "${inactive}"`.
- `operations`: ordered drawing operations. Later operations appear on top.

## Operations

- `rect`: `box=[x,y,width,height]`, `fill`, optional `radius`.
- `ellipse`: `box`, `fill`, optional `outline` and `width`.
- `polygon`: `points=[[x,y], ...]`, `fill`.
- `text`: `box`, `text`, `font=regular|bold|/path/font.ttf`, `font_size`,
  `min_font_size`, `fill`, `align=left|center|right`, `valign=top|center|bottom`,
  optional `line_gap`, `rotation`, `stroke_fill`, and `stroke_width`.
- `line`: `points`, `fill`, `width`, optional `arrow_end` and `arrow_size`.
- `bezier`: four `points`, `fill`, `width`, optional `arrow_end` and `arrow_size`.
- `restore_line`: copy exact source pixels back along a polyline; accepts `points` and `width`.
- `restore_bezier`: copy exact source pixels back along a four-point Bézier path; accepts
  `points` and `width`. Use restore operations after a mask overlaps an original border or
  connector, and before drawing the translated text.

Colors may be CSS hex strings or sampled from the source:

```json
{"fill": {"sample": [200, 140], "radius": 3}}
```

Use sampled fills for flat colored nodes. Keep masks inset from antialiased borders.
