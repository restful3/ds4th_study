#!/usr/bin/env python3
"""Map Markdown image references to nearby Figure captions for manual review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FIGURE = re.compile(
    r"^\s*(?:\*\*)?Figure\s+(\d+\.\d+)\b(?:\*\*)?\s*(.*)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when referenced files are missing or images remain unassigned",
    )
    return parser.parse_args()


def panel_suffix(index: int) -> str:
    if index >= 26:
        raise ValueError("more than 26 panels for one figure are unsupported")
    return chr(ord("a") + index)


def main() -> int:
    args = parse_args()
    document = args.document.resolve()
    lines = document.read_text(encoding="utf-8").splitlines()
    pending: list[dict[str, object]] = []
    figures: list[dict[str, object]] = []
    warnings: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        for match in IMAGE.finditer(line):
            path = match.group(1).strip()
            if path.startswith(("http://", "https://", "data:")):
                warnings.append(
                    f"line {line_number}: remote/data image requires manual review: {path}"
                )
                exists = None
            else:
                exists = (document.parent / path).is_file()
            pending.append({"path": path, "line": line_number, "exists": exists})

        caption = FIGURE.match(line)
        if not caption:
            continue
        figure_number = caption.group(1)
        caption_text = caption.group(2).strip()
        if not pending:
            warnings.append(
                f"line {line_number}: Figure {figure_number} has no preceding pending image"
            )
            continue
        count = len(pending)
        images = []
        for index, image in enumerate(pending):
            suffix = panel_suffix(index) if count > 1 else ""
            images.append(
                {
                    **image,
                    "key": f"figure-{figure_number.replace('.', '-')}{suffix}",
                    "panel": suffix or None,
                }
            )
        figures.append(
            {
                "figure": figure_number,
                "caption": caption_text,
                "caption_line": line_number,
                "images": images,
                "mapping_status": "inferred-requires-visual-review",
            }
        )
        pending = []

    if pending:
        warnings.append(
            f"{len(pending)} image(s) remain after the final numbered Figure caption"
        )

    manifest = {
        "schema_version": 1,
        "document": str(document),
        "figures": figures,
        "unassigned": pending,
        "warnings": warnings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    image_count = sum(len(item["images"]) for item in figures)
    missing = [
        image
        for figure in figures
        for image in figure["images"]
        if image["exists"] is False
    ]
    print(
        f"OK {args.output}: {len(figures)} figures, {image_count} mapped images, "
        f"{len(pending)} unassigned, {len(missing)} missing"
    )
    for warning in warnings:
        print(f"WARNING {warning}")
    if args.check and (pending or missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
