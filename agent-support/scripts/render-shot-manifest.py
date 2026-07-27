#!/usr/bin/env python3
"""Run guarded browser captures serially from a strict JSON manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = (
    Path.home()
    / ".local"
    / "libexec"
    / "ds4th-safe-browser-shot"
    / "safe-browser-shot.sh"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "browser-shots"
ALLOWED_KEYS = {
    "url",
    "output",
    "format",
    "width",
    "height",
    "virtual_time_budget",
    "timeout",
}


@dataclass(frozen=True)
class Capture:
    url: str
    output: Path
    output_format: str
    width: int
    height: int
    virtual_time_budget: int | None
    timeout: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def require_integer(
    item: dict[str, object], key: str, default: int | None, minimum: int, maximum: int
) -> int | None:
    value = item.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def load_manifest(path: Path, output_root: Path) -> list[Capture]:
    if not path.is_absolute():
        raise ValueError("--manifest must be an absolute path")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON manifest: {exc}") from exc

    if not isinstance(data, dict) or set(data) != {"version", "captures"}:
        raise ValueError("manifest must contain exactly version and captures")
    if data["version"] != 1:
        raise ValueError("manifest version must be 1")
    rows = data["captures"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("captures must be a list with 1..100 entries")

    root = output_root.resolve()
    if not root.is_dir():
        raise ValueError(f"output root must already exist: {root}")
    captures: list[Capture] = []
    seen_outputs: set[Path] = set()
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"capture {index} must be an object")
        unknown = sorted(set(item) - ALLOWED_KEYS)
        missing = sorted({"url", "output"} - set(item))
        if unknown or missing:
            raise ValueError(
                f"capture {index} has unknown keys {unknown} or missing keys {missing}"
            )

        url = item["url"]
        output_value = item["output"]
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"}:
            raise ValueError(f"capture {index} url must use http:// or https://")
        if not isinstance(output_value, str):
            raise ValueError(f"capture {index} output must be a string")
        output = Path(output_value)
        if not output.is_absolute():
            raise ValueError(f"capture {index} output must be absolute")
        output = output.resolve(strict=False)
        try:
            output.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"capture {index} output must stay under {root}: {output}"
            ) from exc
        if output in seen_outputs:
            raise ValueError(f"duplicate capture output: {output}")
        seen_outputs.add(output)
        if not output.parent.is_dir():
            raise ValueError(f"capture {index} output directory does not exist: {output.parent}")

        output_format = item.get("format", "png")
        if output_format not in {"png", "pdf"}:
            raise ValueError(f"capture {index} format must be png or pdf")
        if output.suffix.lower() != f".{output_format}":
            raise ValueError(
                f"capture {index} output extension must match format {output_format}"
            )

        width = require_integer(item, "width", 1600, 320, 4096)
        height = require_integer(item, "height", 900, 180, 32000)
        virtual_time_budget = require_integer(
            item, "virtual_time_budget", None, 1, 60000
        )
        timeout = require_integer(item, "timeout", 120, 5, 120)
        assert width is not None and height is not None and timeout is not None
        captures.append(
            Capture(
                url=url,
                output=output,
                output_format=output_format,
                width=width,
                height=height,
                virtual_time_budget=virtual_time_budget,
                timeout=timeout,
            )
        )
    return captures


def runner_command(runner: Path, capture: Capture) -> list[str]:
    command = [
        str(runner),
        "--format",
        capture.output_format,
        "--url",
        capture.url,
        "--output",
        str(capture.output),
        "--width",
        str(capture.width),
        "--height",
        str(capture.height),
        "--timeout",
        str(capture.timeout),
    ]
    if capture.virtual_time_budget is not None:
        command.extend(
            ["--virtual-time-budget", str(capture.virtual_time_budget)]
        )
    return command


def main() -> int:
    args = parse_args()
    try:
        runner = args.runner.resolve(strict=True)
        if not runner.is_file():
            raise ValueError(f"runner is not a regular file: {runner}")
        captures = load_manifest(args.manifest, args.output_root)
        for index, capture in enumerate(captures, start=1):
            print(
                f"[{index}/{len(captures)}] {capture.output_format} "
                f"{capture.width}x{capture.height} -> {capture.output}",
                flush=True,
            )
            result = subprocess.run(
                runner_command(runner, capture),
                check=False,
                text=True,
                capture_output=True,
            )
            if result.stdout:
                print(result.stdout.rstrip())
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            if result.returncode:
                print(
                    f"capture {index} failed with exit status {result.returncode}",
                    file=sys.stderr,
                )
                return result.returncode
        print(f"completed {len(captures)} guarded capture(s)")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
