#!/usr/bin/env python3
"""Scaffold a raw-to-report-to-deck ds4th study session."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import tomllib
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "agent-support" / "studies.toml"
DEFAULT_SITE = REPO_ROOT / "docs"
DEFAULT_TEMPLATE = REPO_ROOT / "agent-support" / "templates" / "study-deck"
DEFAULT_REPORT_TEMPLATE = REPO_ROOT / "agent-support" / "templates" / "study-report"
SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
TOKEN_RE = re.compile(r"{{[A-Z0-9_]+}}")
CAPTION_NUMBER_RE = re.compile(r'asset-caption__chip">(?:\s*)(그림|표)\s+([1-9][0-9]*)')
MERGE_MARKER_RE = re.compile(r"(?m)^(?:<<<<<<<(?: .*)?|=======|>>>>>>>(?: .*)?)$")
STANDALONE_CSS_PLUS_RE = re.compile(r"(?m)^\s*\+\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, help="Study id from studies.toml")
    parser.add_argument("--session", required=True, help="Lowercase ASCII session slug")
    parser.add_argument("--title", required=True, help="Presentation title")
    parser.add_argument("--subtitle", help="Cover subtitle; defaults to the chapter list")
    parser.add_argument("--date", required=True, help="Presentation date in YYYY-MM-DD")
    parser.add_argument(
        "--presenter", action="append", required=True, help="Presenter; repeat as needed"
    )
    parser.add_argument(
        "--chapter", action="append", required=True, help="Chapter; repeat as needed"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--report-template", type=Path, default=DEFAULT_REPORT_TEMPLATE
    )
    return parser.parse_args()


def load_registry(path: Path) -> list[dict]:
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"registry not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid registry TOML: {exc}") from exc

    if data.get("schema_version") != 1 or not isinstance(data.get("studies"), list):
        raise ValueError(f"unsupported study registry: {path}")
    return data["studies"]


def render(source: str, replacements: dict[str, str], label: str) -> str:
    result = source
    for token, value in replacements.items():
        result = result.replace("{{" + token + "}}", value)
    unresolved = sorted(set(TOKEN_RE.findall(result)))
    if unresolved:
        raise ValueError(f"unresolved template token(s) in {label}: {unresolved}")
    return result


def toml_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def validate_args(args: argparse.Namespace) -> None:
    if not SLUG_RE.fullmatch(args.study):
        raise ValueError(f"invalid study id: {args.study!r}")
    if not SLUG_RE.fullmatch(args.session):
        raise ValueError(f"invalid session slug: {args.session!r}")
    try:
        date.fromisoformat(args.date)
    except ValueError as exc:
        raise ValueError(f"date must use YYYY-MM-DD: {args.date!r}") from exc
    if not args.title.strip():
        raise ValueError("title must not be empty")
    if any(not value.strip() for value in args.presenter):
        raise ValueError("presenter must not be empty")
    if any(not value.strip() for value in args.chapter):
        raise ValueError("chapter must not be empty")


def validate_template_sources(template: Path, report_template: Path) -> None:
    for root in (template, report_template):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".css",
                ".html",
                ".js",
                ".svg",
                ".tmpl",
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if MERGE_MARKER_RE.search(text):
                raise ValueError(f"unresolved merge marker in template: {path}")
            if path.suffix.lower() == ".css" and STANDALONE_CSS_PLUS_RE.search(text):
                raise ValueError(f"standalone '+' diff artifact in template CSS: {path}")

    report_html = (report_template / "index.html").read_text(encoding="utf-8")
    numbers: dict[str, list[int]] = {"그림": [], "표": []}
    for kind, number in CAPTION_NUMBER_RE.findall(report_html):
        numbers[kind].append(int(number))
    for kind, found in numbers.items():
        expected = list(range(1, len(found) + 1))
        if found != expected:
            raise ValueError(
                f"template {kind} caption numbers must be consecutive 1..N: "
                f"found {found}, expected {expected}"
            )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        studies = load_registry(args.registry.resolve())
        matches = [study for study in studies if study.get("id") == args.study]
        if len(matches) != 1:
            raise ValueError(f"study id must match exactly one registry entry: {args.study}")
        study = matches[0]
        slug = study.get("slug")
        if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
            raise ValueError(f"invalid study slug in registry: {slug!r}")
        expected_path = f"docs/studies/{slug}"
        if study.get("presentation_path") != expected_path:
            raise ValueError(f"presentation_path must remain {expected_path}")

        template = args.template.resolve()
        html_template = template / "index.html"
        metadata_template = template / "presentation.toml.tmpl"
        assets_template = template / "assets"
        report_template = args.report_template.resolve()
        report_html_template = report_template / "index.html"
        report_assets_template = report_template / "assets"
        for required in (
            html_template,
            metadata_template,
            assets_template,
            report_html_template,
            report_assets_template,
        ):
            if not required.exists():
                raise ValueError(f"template component not found: {required}")
        validate_template_sources(template, report_template)

        presenters = [value.strip() for value in args.presenter]
        chapters = [value.strip() for value in args.chapter]
        subtitle = args.subtitle.strip() if args.subtitle else " · ".join(chapters)
        html_replacements = {
            "DECK_TITLE": html.escape(args.title.strip()),
            "SUBTITLE": html.escape(subtitle),
            "STUDY_TITLE": html.escape(str(study.get("title_ko") or study.get("title"))),
            "PRESENTERS": html.escape(", ".join(presenters)),
            "CHAPTERS": html.escape(" · ".join(chapters)),
            "DATE": html.escape(args.date),
        }
        metadata_replacements = {
            "STUDY_ID_TOML": toml_value(args.study),
            "SESSION_ID_TOML": toml_value(args.session),
            "DECK_TITLE_TOML": toml_value(args.title.strip()),
            "DATE_TOML": toml_value(args.date),
            "PRESENTERS_TOML": toml_value(presenters),
            "CHAPTERS_TOML": toml_value(chapters),
        }
        deck_html = render(
            html_template.read_text(encoding="utf-8"), html_replacements, html_template.name
        )
        report_html = render(
            report_html_template.read_text(encoding="utf-8"),
            html_replacements,
            f"{report_template.name}/{report_html_template.name}",
        )
        metadata = render(
            metadata_template.read_text(encoding="utf-8"),
            metadata_replacements,
            metadata_template.name,
        )

        target = args.site.resolve() / "studies" / slug / "presentations" / args.session
        if target.exists():
            raise ValueError(f"presentation directory already exists: {target}")
        target.mkdir(parents=True)
        (target / "index.html").write_text(deck_html, encoding="utf-8")
        (target / "report.html").write_text(report_html, encoding="utf-8")
        (target / "presentation.toml").write_text(metadata, encoding="utf-8")
        shutil.copytree(assets_template, target / "assets")
        shutil.copytree(report_assets_template, target / "assets", dirs_exist_ok=True)
        (target / "assets" / "figs").mkdir(exist_ok=True)

        try:
            display = target.relative_to(REPO_ROOT)
        except ValueError:
            display = target
        print(f"created session report and presentation: {display}")
        print(
            "next: follow agent-support/templates/STUDY_SESSION_BLUEPRINT.md; "
            "finish and validate report.html first, then derive index.html from it"
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
