#!/usr/bin/env python3
"""Validate the ds4th GitHub Pages tree without third-party dependencies."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "agent-support" / "studies.toml"
DEFAULT_SITE = REPO_ROOT / "docs"
SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_DECK_BYTES = 30 * 1024 * 1024


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[tuple[str, str, str]] = []
        self.images_without_alt = 0
        self.has_title = False
        self.has_viewport = False
        self.html_lang = ""
        self.forms = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        if tag == "html":
            self.html_lang = values.get("lang", "")
        elif tag == "title":
            self.has_title = True
        elif tag == "meta" and values.get("name", "").lower() == "viewport":
            self.has_viewport = True
        elif tag == "img" and "alt" not in values:
            self.images_without_alt += 1
        elif tag == "form":
            self.forms += 1

        for attribute in ("href", "src", "poster", "action"):
            if attribute in values and values[attribute]:
                self.references.append((tag, attribute, values[attribute]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--site", type=Path, default=DEFAULT_SITE)
    parser.add_argument(
        "--check-materials",
        action="store_true",
        help="Also require every registry materials_path to exist locally.",
    )
    return parser.parse_args()


def load_toml(path: Path, errors: list[str]) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError:
        errors.append(f"missing TOML file: {path}")
    except tomllib.TOMLDecodeError as exc:
        errors.append(f"invalid TOML in {path}: {exc}")
    return {}


def within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_registry(
    registry: Path, site: Path, check_materials: bool, errors: list[str]
) -> dict[str, dict]:
    data = load_toml(registry, errors)
    if data.get("schema_version") != 1:
        errors.append(f"unsupported registry schema_version: {registry}")

    studies = data.get("studies", [])
    if not studies:
        errors.append("study registry has no entries")
        return {}

    result: dict[str, dict] = {}
    seen_slugs: set[str] = set()
    for study in studies:
        study_id = study.get("id")
        slug = study.get("slug")
        if not isinstance(study_id, str) or not SLUG_RE.fullmatch(study_id):
            errors.append(f"invalid study id: {study_id!r}")
            continue
        if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
            errors.append(f"invalid study slug for {study_id}: {slug!r}")
            continue
        if study_id in result:
            errors.append(f"duplicate study id: {study_id}")
        if slug in seen_slugs:
            errors.append(f"duplicate study slug: {slug}")
        if study.get("status") not in {"active", "archived"}:
            errors.append(f"invalid status for {study_id}: {study.get('status')!r}")

        expected_public = f"docs/studies/{slug}"
        if study.get("presentation_path") != expected_public:
            errors.append(
                f"presentation_path for {study_id} must remain {expected_public}"
            )

        materials = study.get("materials_path", "")
        archive_path = study.get("archive_path", "")
        if not isinstance(materials, str) or Path(materials).is_absolute() or ".." in Path(materials).parts:
            errors.append(f"invalid materials_path for {study_id}: {materials!r}")
        if not isinstance(archive_path, str) or not archive_path.startswith("archive/"):
            errors.append(f"invalid archive_path for {study_id}: {archive_path!r}")
        if study.get("status") == "active" and not str(materials).startswith("source/"):
            errors.append(f"active study {study_id} must use a source/ materials_path")
        if study.get("status") == "archived" and materials != archive_path:
            errors.append(
                f"archived study {study_id} materials_path must equal archive_path"
            )
        if check_materials and materials and not (REPO_ROOT / materials).is_dir():
            errors.append(f"materials_path does not exist for {study_id}: {materials}")

        public_dir = site / "studies" / slug
        if not (public_dir / "index.html").is_file():
            errors.append(f"missing generated study index: {public_dir / 'index.html'}")
        result[study_id] = study
        seen_slugs.add(slug)
    return result


def validate_metadata(site: Path, studies: dict[str, dict], errors: list[str]) -> None:
    seen: set[tuple[str, str]] = set()
    for metadata_path in sorted(
        site.glob("studies/*/presentations/*/presentation.toml")
    ):
        metadata = load_toml(metadata_path, errors)
        study_id = metadata.get("study_id")
        session_id = metadata.get("session_id")
        if study_id not in studies:
            errors.append(f"unknown study_id in {metadata_path}: {study_id!r}")
            continue
        if not isinstance(session_id, str) or not SLUG_RE.fullmatch(session_id):
            errors.append(f"invalid session_id in {metadata_path}: {session_id!r}")
            continue
        if metadata_path.parent.name != session_id:
            errors.append(
                f"session_id does not match directory in {metadata_path}: {session_id}"
            )
        if metadata_path.parents[2].name != studies[study_id]["slug"]:
            errors.append(f"presentation is under the wrong study: {metadata_path}")
        for field in ("title", "date"):
            if not metadata.get(field):
                errors.append(f"missing {field} in {metadata_path}")
        for field in ("presenters", "chapters"):
            if not isinstance(metadata.get(field), list) or not metadata[field]:
                errors.append(f"{field} must be a non-empty list in {metadata_path}")
        deck_html = metadata_path.parent / "index.html"
        if not deck_html.is_file():
            errors.append(f"presentation is missing index.html: {metadata_path.parent}")

        template_id = metadata.get("template")
        if template_id == "study-deck-v1" and deck_html.is_file():
            deck_text = deck_html.read_text(encoding="utf-8", errors="replace")
            if 'data-deck-template="study-deck-v1"' not in deck_text:
                errors.append(f"study-deck-v1 marker is missing in {deck_html}")
            for asset in ("assets/deck.css", "assets/deck.js"):
                if not (metadata_path.parent / asset).is_file():
                    errors.append(
                        f"study-deck-v1 asset is missing in {metadata_path.parent}: {asset}"
                    )

        key = (study_id, session_id)
        if key in seen:
            errors.append(f"duplicate presentation id: {study_id}/{session_id}")
        seen.add(key)

        deck_size = sum(
            path.stat().st_size for path in metadata_path.parent.rglob("*") if path.is_file()
        )
        if deck_size > MAX_DECK_BYTES:
            errors.append(
                f"presentation exceeds {MAX_DECK_BYTES // 1024 // 1024} MiB: "
                f"{metadata_path.parent} ({deck_size} bytes)"
            )


def validate_reference(
    html_path: Path,
    site: Path,
    tag: str,
    attribute: str,
    value: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    stripped = value.strip()
    if not stripped or stripped.startswith("#"):
        return
    if stripped.startswith("//"):
        errors.append(f"protocol-relative URL in {html_path}: {value}")
        return

    parsed = urlparse(stripped)
    scheme = parsed.scheme.lower()
    if scheme in {"mailto", "tel"}:
        return
    if scheme == "data":
        if len(stripped) > 1_000_000:
            errors.append(f"embedded data URL is too large in {html_path}")
        return
    if scheme:
        if scheme != "https":
            errors.append(f"non-HTTPS external URL in {html_path}: {value}")
        elif tag in {"script", "iframe"}:
            warnings.append(f"review external {tag} in {html_path}: {value}")
        return
    if stripped.startswith("/"):
        errors.append(f"root-relative Pages URL in {html_path}: {value}")
        return

    local_part = unquote(parsed.path)
    if not local_part:
        return
    target = (html_path.parent / local_part).resolve()
    site_root = site.resolve()
    if not within(target, site_root):
        errors.append(f"reference escapes docs/ in {html_path}: {value}")
        return
    if target.is_dir():
        target = target / "index.html"
    if not target.exists():
        errors.append(f"broken local reference in {html_path}: {value}")


def validate_html(site: Path, errors: list[str], warnings: list[str]) -> None:
    for html_path in sorted(site.rglob("*.html")):
        text = html_path.read_text(encoding="utf-8", errors="replace")
        parser = PageParser()
        parser.feed(text)
        if not parser.has_title:
            errors.append(f"missing <title> in {html_path}")
        if not parser.has_viewport:
            errors.append(f"missing viewport meta tag in {html_path}")
        if parser.html_lang not in {"ko", "en"}:
            errors.append(f"missing or unsupported html lang in {html_path}")
        if parser.images_without_alt:
            errors.append(
                f"{parser.images_without_alt} image(s) without alt text in {html_path}"
            )
        if parser.forms:
            errors.append(f"forms are not allowed on the public study site: {html_path}")
        for tag, attribute, value in parser.references:
            validate_reference(
                html_path, site, tag, attribute, value, errors, warnings
            )


def validate_files(site: Path, errors: list[str]) -> None:
    if not site.is_dir():
        errors.append(f"site directory does not exist: {site}")
        return
    for required in (site / ".nojekyll", site / "index.html", site / "assets" / "site.css"):
        if not required.exists():
            errors.append(f"missing required site file: {required}")
    for path in site.rglob("*"):
        if path.is_symlink() and not within(path.resolve(), site.resolve()):
            errors.append(f"site symlink escapes docs/: {path}")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            errors.append(
                f"file exceeds {MAX_FILE_BYTES // 1024 // 1024} MiB: {path}"
            )


def main() -> int:
    args = parse_args()
    registry = args.registry.resolve()
    site = args.site.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    validate_files(site, errors)
    studies = validate_registry(
        registry, site, args.check_materials, errors
    )
    validate_metadata(site, studies, errors)
    validate_html(site, errors, warnings)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"site validation failed with {len(errors)} error(s)", file=sys.stderr)
        return 1

    html_count = sum(1 for _ in site.rglob("*.html"))
    print(f"site validation passed ({html_count} HTML file(s), {len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
