#!/usr/bin/env python3
"""Build a labeled contact sheet at the target document display width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required. Run with: uv run --with 'pillow>=11,<13' python contact_sheet.py ..."
    ) from exc


FONT = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument(
        "--inventory",
        type=Path,
        help="inventory.py JSON; uses reviewed figure keys as labels",
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--gap", type=int, default=28)
    parser.add_argument("--label-height", type=int, default=34)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.columns < 1:
        raise SystemExit("--columns must be positive")
    if not args.images and not args.inventory:
        raise SystemExit("provide image paths or --inventory")
    cell_width = args.width
    images = []
    labeled_paths: list[tuple[Path, str]] = []
    if args.inventory:
        inventory_path = args.inventory.resolve()
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        document_dir = Path(inventory["document"]).parent
        for figure in inventory["figures"]:
            for item in figure["images"]:
                labeled_paths.append((document_dir / item["path"], item["key"]))
        for index, item in enumerate(inventory.get("unassigned", []), start=1):
            labeled_paths.append((document_dir / item["path"], f"unassigned-{index}"))
    else:
        labeled_paths = [(path, path.stem) for path in args.images]

    for path, label in labeled_paths:
        source = Image.open(path).convert("RGB")
        height = round(source.height * cell_width / source.width)
        resized = source.resize((cell_width, height), Image.Resampling.LANCZOS)
        images.append((path, label, resized))
    rows = (len(images) + args.columns - 1) // args.columns
    row_heights = []
    for row in range(rows):
        row_items = images[row * args.columns : (row + 1) * args.columns]
        row_heights.append(
            max(image.height for _, _, image in row_items) + args.label_height
        )
    sheet_width = args.columns * cell_width + (args.columns + 1) * args.gap
    sheet_height = sum(row_heights) + (rows + 1) * args.gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#f3f4f6")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 18)
    y = args.gap
    for row in range(rows):
        x = args.gap
        row_items = images[row * args.columns : (row + 1) * args.columns]
        for path, label, image in row_items:
            draw.text((x, y), label, font=font, fill="#111111")
            sheet.paste(image, (x, y + args.label_height))
            x += cell_width + args.gap
        y += row_heights[row] + args.gap
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, format="PNG", optimize=True)
    print(f"OK {args.output}: {sheet.width}x{sheet.height}, {len(images)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
