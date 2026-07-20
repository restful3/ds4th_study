#!/usr/bin/env python3
"""Build deterministic GitHub Pages indexes from study and presentation metadata."""

from __future__ import annotations

import argparse
import html
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "agent-support" / "studies.toml"
DEFAULT_SITE = REPO_ROOT / "docs"
SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
REQUIRED_STUDY_FIELDS = {
    "id",
    "slug",
    "title",
    "title_ko",
    "status",
    "start_date",
    "end_date",
    "materials_path",
    "archive_path",
    "presentation_path",
}
REQUIRED_DECK_FIELDS = {
    "study_id",
    "session_id",
    "title",
    "date",
    "presenters",
    "chapters",
    "artifacts",
}
SUPPORTED_ARTIFACTS = {"report", "slides"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed index files differ from generated output.",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"metadata file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc


def validate_slug(value: object, label: str) -> str:
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase ASCII slug: {value!r}")
    return value


def load_studies(registry: Path) -> list[dict]:
    data = load_toml(registry)
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {registry}")

    studies = data.get("studies")
    if not isinstance(studies, list) or not studies:
        raise ValueError(f"{registry} must contain at least one [[studies]] entry")

    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for study in studies:
        missing = REQUIRED_STUDY_FIELDS - set(study)
        if missing:
            raise ValueError(f"study entry is missing fields: {sorted(missing)}")

        study_id = validate_slug(study["id"], "study id")
        slug = validate_slug(study["slug"], "study slug")
        if study_id in seen_ids:
            raise ValueError(f"duplicate study id: {study_id}")
        if slug in seen_slugs:
            raise ValueError(f"duplicate study slug: {slug}")
        if study["status"] not in {"active", "archived"}:
            raise ValueError(f"invalid study status for {study_id}: {study['status']}")

        expected_path = f"docs/studies/{slug}"
        if study["presentation_path"] != expected_path:
            raise ValueError(
                f"presentation_path for {study_id} must stay at {expected_path}"
            )
        seen_ids.add(study_id)
        seen_slugs.add(slug)

    return studies


def load_presentations(site: Path, studies: list[dict]) -> dict[str, list[dict]]:
    by_study = {study["id"]: [] for study in studies}
    seen_sessions: set[tuple[str, str]] = set()

    pattern = "studies/*/presentations/*/presentation.toml"
    for metadata_path in sorted(site.glob(pattern)):
        deck = load_toml(metadata_path)
        missing = REQUIRED_DECK_FIELDS - set(deck)
        if missing:
            raise ValueError(f"{metadata_path} is missing fields: {sorted(missing)}")

        study_id = deck["study_id"]
        if study_id not in by_study:
            raise ValueError(f"unknown study_id in {metadata_path}: {study_id}")

        session_id = validate_slug(deck["session_id"], "session id")
        if metadata_path.parent.name != session_id:
            raise ValueError(
                f"session_id {session_id} must match directory {metadata_path.parent.name}"
            )

        study = next(item for item in studies if item["id"] == study_id)
        actual_slug = metadata_path.parents[2].name
        if actual_slug != study["slug"]:
            raise ValueError(
                f"{metadata_path} is under {actual_slug}, expected {study['slug']}"
            )
        if not isinstance(deck["presenters"], list) or not deck["presenters"]:
            raise ValueError(f"presenters must be a non-empty list in {metadata_path}")
        if not isinstance(deck["chapters"], list) or not deck["chapters"]:
            raise ValueError(f"chapters must be a non-empty list in {metadata_path}")
        artifacts = deck["artifacts"]
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(artifacts) != len(set(artifacts))
            or not set(artifacts).issubset(SUPPORTED_ARTIFACTS)
        ):
            raise ValueError(
                f"artifacts must be a unique non-empty subset of "
                f"{sorted(SUPPORTED_ARTIFACTS)} in {metadata_path}"
            )
        artifact_files = {"report": "report.html", "slides": "index.html"}
        for artifact in artifacts:
            if not (metadata_path.parent / artifact_files[artifact]).is_file():
                raise ValueError(
                    f"{artifact} artifact is missing in {metadata_path.parent}: "
                    f"{artifact_files[artifact]}"
                )

        key = (study_id, session_id)
        if key in seen_sessions:
            raise ValueError(f"duplicate presentation session: {study_id}/{session_id}")
        seen_sessions.add(key)
        deck["_path"] = metadata_path.parent
        by_study[study_id].append(deck)

    for decks in by_study.values():
        decks.sort(key=lambda item: (str(item["date"]), item["session_id"]))
    return by_study


def page_shell(title: str, stylesheet: str, body: str) -> str:
    favicon = stylesheet.rsplit("/", 1)[0] + "/favicon.svg"
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="icon" href="{favicon}" type="image/svg+xml">
  <link rel="stylesheet" href="{stylesheet}">
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def study_card(study: dict) -> str:
    status = study["status"]
    status_label = "진행 중" if status == "active" else "완료"
    date_range = f"{study['start_date']} – {study['end_date']}"
    return f"""  <a class="card" href="studies/{study['slug']}/">
    <span class="badge {status}">{status_label}</span>
    <h3>{html.escape(study['title_ko'])}</h3>
    <p>{html.escape(study['title'])}</p>
    <p class="meta">{html.escape(date_range)}</p>
  </a>"""


def render_root(studies: list[dict]) -> str:
    sections: list[str] = [
        "  <h1>ds4th study</h1>",
        '  <p class="lead">진행 중인 스터디와 공개 리포트·발표자료를 한곳에서 봅니다. '
        "학습자료가 source에서 archive로 이동해도 공개 URL은 유지됩니다.</p>",
    ]
    for status, heading in (("active", "진행 중인 스터디"), ("archived", "완료된 스터디")):
        matching = [study for study in studies if study["status"] == status]
        sections.append(f"  <h2>{heading}</h2>")
        if matching:
            cards = "\n".join(study_card(study) for study in matching)
            sections.append(f'  <div class="grid">\n{cards}\n  </div>')
        else:
            sections.append('  <p class="empty">등록된 스터디가 없습니다.</p>')
    sections.append("  <footer>GitHub Pages로 제공되는 ds4th 스터디 발표 아카이브</footer>")
    return page_shell("ds4th study", "assets/site.css", "\n".join(sections))


def render_study(study: dict, decks: list[dict]) -> str:
    status = study["status"]
    status_label = "진행 중" if status == "active" else "완료"
    repo_path = quote(study["materials_path"], safe="/")
    material_url = f"https://github.com/restful3/ds4th_study/tree/main/{repo_path}"
    body = [
        '  <a class="back" href="../../">← 전체 스터디</a>',
        f'  <span class="badge {status}">{status_label}</span>',
        f"  <h1>{html.escape(study['title_ko'])}</h1>",
        f'  <p class="lead">{html.escape(study["title"])}</p>',
        f'  <p class="meta">{html.escape(study["start_date"])} – '
        f'{html.escape(study["end_date"])} · '
        f'<a href="{html.escape(material_url)}">학습자료</a></p>',
        "  <h2>회차 자료</h2>",
    ]
    if not decks:
        body.append('  <p class="empty">아직 등록된 발표자료가 없습니다.</p>')
    else:
        cards = []
        for deck in decks:
            presenters = ", ".join(str(item) for item in deck["presenters"])
            chapters = " · ".join(str(item) for item in deck["chapters"])
            links = []
            if "report" in deck["artifacts"]:
                links.append(
                    f'<a class="artifact-link" href="presentations/{deck["session_id"]}/report.html">상세 리포트</a>'
                )
            if "slides" in deck["artifacts"]:
                links.append(
                    f'<a class="artifact-link artifact-link--primary" href="presentations/{deck["session_id"]}/">발표자료</a>'
                )
            cards.append(
                '  <article class="card session-card">\n'
                f'    <h3>{html.escape(str(deck["title"]))}</h3>\n'
                f'    <p>{html.escape(chapters)}</p>\n'
                f'    <p class="meta">{html.escape(str(deck["date"]))} · '
                f'{html.escape(presenters)}</p>\n'
                f'    <div class="artifact-links">{"".join(links)}</div>\n'
                "  </article>"
            )
        body.append(f'  <div class="grid">\n{"\n".join(cards)}\n  </div>')
    body.append("  <footer>공개 리포트·발표자료 경로는 스터디 아카이브 이후에도 유지됩니다.</footer>")
    return page_shell(study["title_ko"], "../../assets/site.css", "\n".join(body))


def write_or_check(path: Path, content: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    if check:
        raise ValueError(f"generated index is stale or missing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = path
    print(f"updated {display_path}")
    return True


def main() -> int:
    args = parse_args()
    registry = args.registry.resolve()
    site = args.site.resolve()
    try:
        studies = load_studies(registry)
        decks = load_presentations(site, studies)
        changed = write_or_check(site / "index.html", render_root(studies), args.check)
        for study in studies:
            target = site / "studies" / study["slug"] / "index.html"
            changed |= write_or_check(
                target, render_study(study, decks[study["id"]]), args.check
            )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print("generated indexes are up to date")
    elif not changed:
        print("generated indexes already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
