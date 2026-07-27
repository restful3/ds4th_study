#!/usr/bin/env python3
"""Create a reviewed repeated-layout localization spec from a base spec."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a localization spec after visual geometry review and override "
            "grouped operation fills for a repeated-layout sibling."
        )
    )
    parser.add_argument("base", type=Path)
    parser.add_argument("--output-spec", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--title")
    parser.add_argument(
        "--coordinate-space",
        metavar="WIDTHxHEIGHT",
        help="Override the base coordinate space after sibling geometry review.",
    )
    parser.add_argument(
        "--fill",
        action="append",
        default=[],
        metavar="GROUP=COLOR",
        help="Override fill on every operation whose group matches GROUP.",
    )
    return parser.parse_args()


def parse_coordinate_space(value: str) -> list[int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            f"invalid coordinate space {value!r}; expected WIDTHxHEIGHT"
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("coordinate dimensions must be positive")
    return [width, height]


def parse_fill_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--fill must be GROUP=COLOR, got {item!r}")
        group, color = item.split("=", 1)
        group, color = group.strip(), color.strip()
        if not group or not color:
            raise SystemExit(f"--fill must be GROUP=COLOR, got {item!r}")
        overrides[group] = color
    return overrides


def main() -> int:
    args = parse_args()
    spec = copy.deepcopy(json.loads(args.base.read_text(encoding="utf-8")))
    overrides = parse_fill_overrides(args.fill)

    available_groups = {
        operation["group"]
        for operation in spec.get("operations", [])
        if operation.get("group")
    }
    missing = sorted(set(overrides) - available_groups)
    if missing:
        raise SystemExit(
            "fill override references unknown operation groups: " + ", ".join(missing)
        )

    spec["source"] = args.source
    spec["output"] = args.output_image
    if args.title:
        spec["title"] = args.title
    if args.coordinate_space:
        spec["coordinate_space"] = parse_coordinate_space(args.coordinate_space)

    for operation in spec.get("operations", []):
        group = operation.get("group")
        if group in overrides:
            operation["fill"] = overrides[group]

    args.output_spec.parent.mkdir(parents=True, exist_ok=True)
    args.output_spec.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"OK {args.output_spec}: "
        f"{len(spec.get('operations', []))} operations, "
        f"{len(overrides)} group overrides"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
