#!/usr/bin/env python3
"""Validate an A4 study report PDF and render every page for visual QA."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


A4_WIDTH_POINTS = 595.28
A4_HEIGHT_POINTS = 841.89
PAGE_SIZE_RE = re.compile(
    r"(?:Page\s+\d+\s+size|Page size):\s*([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
    re.IGNORECASE,
)
PAGE_COUNT_RE = re.compile(r"^Pages:\s*([1-9][0-9]*)\s*$", re.MULTILINE)
MATH_RE = re.compile(r"\\\(.+?\\\)|\\\[.+?\\\]|\$\$.+?\$\$", re.DOTALL)
STALE_FOOTER_TEXT = (
    "Computer Use",
    "빅3 벤더",
    "오픈소스 & 자율형 에이전트",
    "실사용 사례 서베이",
    "이슈와 시사점",
    "기술 딥다이브 & 결론",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-html", type=Path)
    parser.add_argument("--dpi", type=int, default=96)
    return parser.parse_args()


def is_under_browser_shots(path: Path) -> bool:
    parts = path.resolve(strict=False).parts
    return any(
        parts[index : index + 2] == ("tmp", "browser-shots")
        for index in range(len(parts) - 1)
    )


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode:
        raise ValueError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def parse_page_count(pdfinfo_text: str) -> int:
    match = PAGE_COUNT_RE.search(pdfinfo_text)
    if not match:
        raise ValueError("pdfinfo output has no positive page count")
    return int(match.group(1))


def parse_page_size(pdfinfo_text: str) -> tuple[float, float]:
    match = PAGE_SIZE_RE.search(pdfinfo_text)
    if not match:
        raise ValueError("pdfinfo output has no page size")
    return float(match.group(1)), float(match.group(2))


def assert_a4_pages(pdf: Path, pages: int) -> None:
    for page in range(1, pages + 1):
        info = command_output(
            ["pdfinfo", "-f", str(page), "-l", str(page), str(pdf)]
        )
        width, height = parse_page_size(info)
        if abs(width - A4_WIDTH_POINTS) > 2 or abs(height - A4_HEIGHT_POINTS) > 2:
            raise ValueError(
                f"page {page} is not portrait A4: {width:.2f} x {height:.2f} pts"
            )


def verify_katex_and_stale_text(pdf: Path, source_html: Path | None) -> str:
    text = command_output(["pdftotext", str(pdf), "-"])
    stale = [value for value in STALE_FOOTER_TEXT if value in text]
    if stale:
        raise ValueError(f"PDF contains stale hard-coded footer text: {stale}")
    if source_html is None:
        return "source-not-provided"
    source = source_html.read_text(encoding="utf-8", errors="replace")
    expressions = MATH_RE.findall(source)
    if not expressions:
        return "not-applicable"
    leaked = [expression for expression in expressions if expression in text]
    if leaked:
        raise ValueError(
            "PDF still contains raw TeX delimiters; deferred KaTeX may not have rendered"
        )
    return f"raw-delimiters-absent:{len(expressions)}"


def render_pages(pdf: Path, output_dir: Path, dpi: int) -> list[Path]:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, mode=0o700)
    prefix = output_dir / "page"
    command_output(["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)])
    pages = sorted(output_dir.glob("page-*.png"))
    if not pages:
        raise ValueError("pdftoppm produced no page images")
    for page in pages:
        if page.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"invalid page PNG: {page}")
        page.chmod(0o600)
    return pages


def render_contact_sheet(pages: list[Path], output_dir: Path) -> Path | None:
    montage = shutil.which("montage")
    if montage is None:
        return None
    output = output_dir / "contact-sheet.png"
    command_output(
        [
            montage,
            *map(str, pages),
            "-thumbnail",
            "238x337",
            "-tile",
            "4x",
            "-geometry",
            "+12+18",
            "-background",
            "#d9dde3",
            str(output),
        ]
    )
    if output.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid contact sheet PNG: {output}")
    output.chmod(0o600)
    return output


def main() -> int:
    args = parse_args()
    try:
        for command in ("pdfinfo", "pdftoppm", "pdftotext"):
            if shutil.which(command) is None:
                raise ValueError(f"required command not found: {command}")
        pdf = args.pdf.resolve(strict=True)
        output_dir = args.output_dir.resolve(strict=False)
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError(f"--pdf must name a PDF file: {pdf}")
        if not is_under_browser_shots(pdf) or not is_under_browser_shots(output_dir):
            raise ValueError("PDF and page images must stay under tmp/browser-shots")
        if not 48 <= args.dpi <= 180:
            raise ValueError("--dpi must be between 48 and 180")
        source_html = args.source_html.resolve(strict=True) if args.source_html else None

        info = command_output(["pdfinfo", str(pdf)])
        pages = parse_page_count(info)
        assert_a4_pages(pdf, pages)
        katex = verify_katex_and_stale_text(pdf, source_html)
        rendered = render_pages(pdf, output_dir, args.dpi)
        if len(rendered) != pages:
            raise ValueError(
                f"rendered page count differs from PDF: {len(rendered)} != {pages}"
            )
        contact = render_contact_sheet(rendered, output_dir)
        print(f"status=ready")
        print(f"pages={pages}")
        print(f"page_size=A4-portrait")
        print(f"katex_check={katex}")
        print(f"page_images={output_dir}")
        print(f"contact_sheet={contact if contact else 'skipped:montage-missing'}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
