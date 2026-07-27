#!/usr/bin/env python3
"""Render deterministic in-image text localizations from JSON overlay specs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised by the CLI environment
    raise SystemExit(
        "Pillow is required. Run with: uv run --with 'pillow>=11,<13' python render.py ..."
    ) from exc


DEFAULT_FONTS = {
    "regular": (
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ),
    "bold": (
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ),
}
VARIABLE = re.compile(r"\$\{([a-zA-Z0-9_-]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--font-regular", type=Path)
    parser.add_argument("--font-bold", type=Path)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(parent)
    for key, value in child.items():
        if key == "variables":
            result[key] = {**result.get(key, {}), **value}
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def substitute(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return VARIABLE.sub(lambda match: str(variables[match.group(1)]), value)
    if isinstance(value, list):
        return [substitute(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, variables) for key, item in value.items()}
    return value


def load_spec(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic extends chain: {path}")
    seen.add(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    parent: dict[str, Any] = {}
    if "extends" in data:
        parent = load_spec(path.parent / data["extends"], seen)
    merged = deep_merge(parent, {key: value for key, value in data.items() if key != "extends"})
    return substitute(merged, merged.get("variables", {}))


def resolve_font(kind: str, overrides: dict[str, Path | None]) -> Path:
    if kind not in ("regular", "bold"):
        path = Path(kind)
        if path.is_file():
            return path
        raise FileNotFoundError(f"font not found: {kind}")
    override = overrides.get(kind)
    if override and override.is_file():
        return override
    for candidate in DEFAULT_FONTS[kind]:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(f"no Korean-capable {kind} font found")


def scaled_box(box: list[float], sx: float, sy: float) -> tuple[int, int, int, int]:
    x, y, width, height = box
    left = round(x * sx)
    top = round(y * sy)
    return left, top, left + round(width * sx), top + round(height * sy)


def scaled_points(points: list[list[float]], sx: float, sy: float) -> list[tuple[int, int]]:
    return [(round(x * sx), round(y * sy)) for x, y in points]


def sampled_color(source: Image.Image, value: Any, sx: float, sy: float) -> Any:
    if not isinstance(value, dict) or "sample" not in value:
        return value
    x, y = value["sample"]
    radius = int(value.get("radius", 0))
    px = round(x * sx)
    py = round(y * sy)
    colors = []
    for yy in range(max(0, py - radius), min(source.height, py + radius + 1)):
        for xx in range(max(0, px - radius), min(source.width, px + radius + 1)):
            colors.append(source.getpixel((xx, yy))[:3])
    return tuple(round(sum(channel) / len(colors)) for channel in zip(*colors))


def multiline_metrics(
    text: str,
    font_path: Path,
    size: int,
    line_gap: int,
    align: str,
    stroke_width: int,
) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    font = ImageFont.truetype(str(font_path), size=size)
    probe = Image.new("L", (4, 4))
    draw = ImageDraw.Draw(probe)
    bbox = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=line_gap,
        align=align,
        stroke_width=stroke_width,
    )
    return font, bbox


def fitted_text_layer(
    operation: dict[str, Any],
    width: int,
    height: int,
    scale: float,
    font_overrides: dict[str, Path | None],
) -> tuple[Image.Image, dict[str, Any]]:
    text = operation["text"]
    align = operation.get("align", "left")
    valign = operation.get("valign", "top")
    rotation = float(operation.get("rotation", 0))
    stroke_width = round(float(operation.get("stroke_width", 0)) * scale)
    font_path = resolve_font(operation.get("font", "regular"), font_overrides)
    start_size = max(1, round(float(operation["font_size"]) * scale))
    min_size = max(1, round(float(operation.get("min_font_size", operation["font_size"])) * scale))
    base_gap = float(operation.get("line_gap", max(1, operation["font_size"] * 0.2)))

    chosen: tuple[Image.Image, int, tuple[int, int, int, int]] | None = None
    for size in range(start_size, min_size - 1, -1):
        gap = max(0, round(base_gap * size / start_size))
        font, bbox = multiline_metrics(text, font_path, size, gap, align, stroke_width)
        text_width = max(1, math.ceil(bbox[2] - bbox[0]))
        text_height = max(1, math.ceil(bbox[3] - bbox[1]))
        raw = Image.new("RGBA", (text_width + stroke_width * 4, text_height + stroke_width * 4))
        draw = ImageDraw.Draw(raw)
        draw.multiline_text(
            (stroke_width * 2 - bbox[0], stroke_width * 2 - bbox[1]),
            text,
            font=font,
            fill=operation.get("fill", "#111111"),
            spacing=gap,
            align=align,
            stroke_width=stroke_width,
            stroke_fill=operation.get("stroke_fill", operation.get("fill", "#111111")),
        )
        rendered = raw.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
        if rendered.width <= width and rendered.height <= height:
            chosen = rendered, size, bbox
            break
    if chosen is None:
        raise ValueError(
            f"text does not fit {width}x{height} above min_font_size: {text!r}"
        )

    rendered, size, bbox = chosen
    if align == "left":
        x = 0
    elif align == "right":
        x = width - rendered.width
    else:
        x = (width - rendered.width) // 2
    if valign == "top":
        y = 0
    elif valign == "bottom":
        y = height - rendered.height
    else:
        y = (height - rendered.height) // 2
    layer = Image.new("RGBA", (width, height))
    layer.alpha_composite(rendered, (x, y))
    return layer, {
        "text": text,
        "font": str(font_path),
        "font_size": size,
        "target": [width, height],
        "rendered": [rendered.width, rendered.height],
        "rotation": rotation,
    }


def bezier_points(points: list[tuple[int, int]], steps: int = 96) -> list[tuple[int, int]]:
    p0, p1, p2, p3 = points
    result = []
    for index in range(steps + 1):
        t = index / steps
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        result.append((round(x), round(y)))
    return result


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    tail: tuple[int, int],
    tip: tuple[int, int],
    fill: Any,
    size: int,
) -> list[tuple[int, int]]:
    angle = math.atan2(tip[1] - tail[1], tip[0] - tail[0])
    left = (
        round(tip[0] - size * math.cos(angle - math.pi / 6)),
        round(tip[1] - size * math.sin(angle - math.pi / 6)),
    )
    right = (
        round(tip[0] - size * math.cos(angle + math.pi / 6)),
        round(tip[1] - size * math.sin(angle + math.pi / 6)),
    )
    polygon = [tip, left, right]
    draw.polygon(polygon, fill=fill)
    return polygon


def composite_antialiased_path(
    result: Image.Image,
    points: list[tuple[int, int]],
    fill: Any,
    width: int,
    arrow_end: bool,
    arrow_size: int,
    supersample: int = 4,
) -> None:
    """Draw a smooth path without changing the output canvas dimensions."""
    layer = Image.new(
        "RGBA",
        (result.width * supersample, result.height * supersample),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(layer)
    high_points = [(x * supersample, y * supersample) for x, y in points]
    draw.line(
        high_points,
        fill=fill,
        width=max(1, width * supersample),
        joint="curve",
    )
    if arrow_end:
        draw_arrow(
            draw,
            high_points[-2],
            high_points[-1],
            fill,
            arrow_size * supersample,
        )
    result.alpha_composite(
        layer.resize(result.size, resample=Image.Resampling.LANCZOS)
    )


def render_spec(
    spec_path: Path,
    font_overrides: dict[str, Path | None],
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = load_spec(spec_path)
    for field in ("source", "output", "operations"):
        if field not in spec:
            raise ValueError(f"{spec_path}: missing {field}")
    source_path = (spec_path.parent / spec["source"]).resolve()
    output_path = (spec_path.parent / spec["output"]).resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"localized output must be PNG: {output_path}")
    source = Image.open(source_path).convert("RGBA")
    result = source.copy()
    design_width, design_height = spec.get("coordinate_space", source.size)
    sx = source.width / design_width
    sy = source.height / design_height
    scale = min(sx, sy)
    allowed = Image.new("L", source.size)
    allowed_draw = ImageDraw.Draw(allowed)
    text_reports: list[dict[str, Any]] = []

    for operation in spec["operations"]:
        op_type = operation["type"]
        draw = ImageDraw.Draw(result)
        if op_type == "rect":
            box = scaled_box(operation["box"], sx, sy)
            fill = sampled_color(source, operation["fill"], sx, sy)
            radius = round(float(operation.get("radius", 0)) * scale)
            draw.rounded_rectangle(box, radius=radius, fill=fill)
            allowed_draw.rounded_rectangle(box, radius=radius, fill=255)
        elif op_type == "ellipse":
            box = scaled_box(operation["box"], sx, sy)
            fill = sampled_color(source, operation["fill"], sx, sy)
            outline_spec = operation.get("outline")
            outline = (
                sampled_color(source, outline_spec, sx, sy)
                if outline_spec is not None
                else None
            )
            width = max(1, round(float(operation.get("width", 1)) * scale))
            draw.ellipse(box, fill=fill, outline=outline, width=width)
            allowed_draw.ellipse(box, fill=255)
        elif op_type == "polygon":
            points = scaled_points(operation["points"], sx, sy)
            fill = sampled_color(source, operation["fill"], sx, sy)
            draw.polygon(points, fill=fill)
            allowed_draw.polygon(points, fill=255)
        elif op_type == "text":
            left, top, right, bottom = scaled_box(operation["box"], sx, sy)
            layer, report = fitted_text_layer(
                operation, right - left, bottom - top, scale, font_overrides
            )
            result.alpha_composite(layer, (left, top))
            allowed_draw.rectangle((left, top, right, bottom), fill=255)
            report["box"] = [left, top, right - left, bottom - top]
            text_reports.append(report)
        elif op_type in ("restore_line", "restore_bezier"):
            control = scaled_points(operation["points"], sx, sy)
            points = bezier_points(control) if op_type == "restore_bezier" else control
            width = max(1, round(float(operation.get("width", 5)) * scale))
            restore_mask = Image.new("L", source.size)
            restore_draw = ImageDraw.Draw(restore_mask)
            restore_draw.line(points, fill=255, width=width, joint="curve")
            radius = max(1, (width + 1) // 2)
            for endpoint_x, endpoint_y in (points[0], points[-1]):
                restore_draw.ellipse(
                    (
                        endpoint_x - radius,
                        endpoint_y - radius,
                        endpoint_x + radius,
                        endpoint_y + radius,
                    ),
                    fill=255,
                )
            result.paste(source, (0, 0), restore_mask)
            allowed_draw.bitmap((0, 0), restore_mask, fill=255)
        elif op_type in ("line", "bezier"):
            control = scaled_points(operation["points"], sx, sy)
            points = bezier_points(control) if op_type == "bezier" else control
            fill = sampled_color(source, operation.get("fill", "#111111"), sx, sy)
            width = max(1, round(float(operation.get("width", 1)) * scale))
            arrow_end = bool(operation.get("arrow_end"))
            arrow_size = max(3, round(float(operation.get("arrow_size", 9)) * scale))
            composite_antialiased_path(
                result,
                points,
                fill,
                width,
                arrow_end,
                arrow_size,
            )
            # Include the small Lanczos halo introduced by supersampled rendering.
            audit_width = width + max(10, round(10 * scale))
            allowed_draw.line(points, fill=255, width=audit_width, joint="curve")
            audit_radius = max(6, (audit_width + 1) // 2 + 3)
            for endpoint_x, endpoint_y in (points[0], points[-1]):
                allowed_draw.ellipse(
                    (
                        endpoint_x - audit_radius,
                        endpoint_y - audit_radius,
                        endpoint_x + audit_radius,
                        endpoint_y + audit_radius,
                    ),
                    fill=255,
                )
            if arrow_end:
                polygon = draw_arrow(
                    allowed_draw,
                    points[-2],
                    points[-1],
                    255,
                    arrow_size,
                )
                allowed_draw.line(
                    polygon + [polygon[0]],
                    fill=255,
                    width=max(8, round(8 * scale)),
                    joint="curve",
                )
        else:
            raise ValueError(f"{spec_path}: unsupported operation type {op_type}")

    difference = ImageChops.difference(source.convert("RGB"), result.convert("RGB"))
    difference.paste((0, 0, 0), mask=allowed)
    unexpected_bbox = difference.getbbox()
    if unexpected_bbox is not None:
        raise AssertionError(
            f"{spec_path}: pixels changed outside declared operation regions "
            f"at {unexpected_bbox}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path, format="PNG", optimize=True)
    with Image.open(output_path) as saved:
        if saved.size != source.size:
            raise AssertionError(f"{output_path}: output dimensions changed")
    return {
        "spec": str(spec_path),
        "source": str(source_path),
        "output": str(output_path),
        "dimensions": list(source.size),
        "operations": len(spec["operations"]),
        "texts": text_reports,
    }


def main() -> int:
    args = parse_args()
    overrides = {"regular": args.font_regular, "bold": args.font_bold}
    reports = []
    for spec_path in args.specs:
        report = render_spec(spec_path, overrides)
        reports.append(report)
        print(
            f"OK {Path(report['output']).name}: "
            f"{report['dimensions'][0]}x{report['dimensions'][1]}, "
            f"{report['operations']} operations, {len(report['texts'])} text blocks"
        )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
