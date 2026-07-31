#!/usr/bin/env python3
"""Validate the ds4th GitHub Pages tree without third-party dependencies."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "agent-support" / "studies.toml"
DEFAULT_SITE = REPO_ROOT / "docs"
SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SESSION_BYTES = 30 * 1024 * 1024
SUPPORTED_ARTIFACTS = {"report", "slides"}
CAPTION_NUMBER_RE = re.compile(r"^(그림|표)\s+([1-9][0-9]*)$")
CAPTION_REFERENCE_GROUP_RE = re.compile(
    r"(그림|표)\s*"
    r"([1-9][0-9]*(?!\.[0-9])"
    r"(?:\s*(?:[-–·,]\s*[1-9][0-9]*)*)?)"
)
EXTERNAL_CAPTION_REFERENCE_RE = re.compile(
    r"(?:교재|책|원문)\s+(?:그림|표)\s*"
    r"[1-9][0-9]*(?:\.[0-9]+)?"
    r"(?:\s*(?:[-–·,]\s*[1-9][0-9]*(?:\.[0-9]+)?)*)?"
)
MERGE_MARKER_RE = re.compile(r"(?m)^(?:<<<<<<<(?: .*)?|=======|>>>>>>>(?: .*)?)$")
STANDALONE_CSS_PLUS_RE = re.compile(r"(?m)^\s*\+\s*$")
FONT_FAMILY_RE = re.compile(r"font-family\s*:\s*([^;}]+)", re.IGNORECASE)
FONT_ATTRIBUTE_RE = re.compile(r"""font-family\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
CSS_RULE_RE = re.compile(r"([^{}]*)\{([^{}]*)\}")
CJK_WRAP_DECLARATIONS = (
    ("word-break: keep-all", re.compile(r"word-break\s*:\s*keep-all", re.IGNORECASE)),
    (
        "overflow-wrap: break-word",
        re.compile(r"overflow-wrap\s*:\s*break-word", re.IGNORECASE),
    ),
)

# Registered 2026-07-27. Remove one entry only after that published session's
# SVGs have been updated, rendered at final size, and visually rechecked.
FONT_STACK_DRIFT_ALLOWLIST = {
    "kg-llm-in-action-2026/2026-07-25-ch01",
    "kg-llm-in-action-2026/2026-08-01-ch03",
}


@dataclass
class FigureTrace:
    element_id: str
    required: bool
    image_sources: set[str] = field(default_factory=set)
    has_accessible_visual: bool = False
    caption_parts: list[str] = field(default_factory=list)
    note_parts: list[str] = field(default_factory=list)

    @property
    def caption(self) -> str:
        return " ".join(" ".join(self.caption_parts).split())

    @property
    def note(self) -> str:
        return " ".join(" ".join(self.note_parts).split())


@dataclass
class ReportReferenceTrace:
    target: str
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.text_parts).split())


@dataclass
class SlideTrace:
    label: str
    refs: list[str]
    image_sources: set[str] = field(default_factory=set)
    report_links: set[str] = field(default_factory=set)
    report_references: list[ReportReferenceTrace] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.text_parts).split())


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


class ReportDeckTraceParser(HTMLParser):
    """Collect report assets and the deck evidence that derives from them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.report_sections: set[str] = set()
        self.required_figures: set[str] = set()
        self.required_figures_without_id = 0
        self.figures: dict[str, FigureTrace] = {}
        self.caption_numbers: dict[str, list[int]] = {"그림": [], "표": []}
        self.caption_labels_by_id: dict[str, tuple[str, int]] = {}
        self.invalid_caption_chips: list[str] = []
        self.internal_report_references: list[ReportReferenceTrace] = []
        self.unlinked_report_text_parts: list[str] = []
        self.slides: list[SlideTrace] = []
        self.deck_report_source = ""
        self._current_figure: FigureTrace | None = None
        self._current_slide: SlideTrace | None = None
        self._section_depth = 0
        self._slide_section_depth = 0
        self._in_figure_caption = False
        self._in_asset_note = False
        self._caption_chip_parts: list[str] | None = None
        self._caption_owner_id = ""
        self._current_table_id = ""
        self._current_report_reference: ReportReferenceTrace | None = None
        self._report_text_section_depth = 0
        self._in_table_caption = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        classes = set(values.get("class", "").split())
        element_id = values.get("id", "").strip()
        if element_id:
            if element_id in self.ids:
                self.duplicate_ids.add(element_id)
            self.ids.add(element_id)

        if tag == "main" and values.get("data-report-source"):
            self.deck_report_source = values["data-report-source"].strip()
        if tag == "section":
            self._section_depth += 1
            if "slide" in classes:
                self._current_slide = SlideTrace(
                    label=values.get("aria-label", "").strip(),
                    refs=values.get("data-report-refs", "").split(),
                )
                self._slide_section_depth = self._section_depth
                self.slides.append(self._current_slide)
            if "report-section" in classes and element_id:
                self.report_sections.add(element_id)
                self._report_text_section_depth = self._section_depth

        if tag == "figure" and "report-figure" in classes:
            required = values.get("data-deck-use") == "required"
            if required and not element_id:
                self.required_figures_without_id += 1
            self._current_figure = FigureTrace(element_id, required)
            if element_id:
                self.figures[element_id] = self._current_figure
                if required:
                    self.required_figures.add(element_id)
        elif tag == "figcaption" and self._current_figure is not None:
            self._in_figure_caption = True
        elif tag == "small" and "asset-note" in classes and self._current_figure is not None:
            self._in_asset_note = True
        elif tag == "table":
            self._current_table_id = element_id
        elif tag == "caption":
            self._in_table_caption = True

        if tag == "span" and "asset-caption__chip" in classes:
            self._caption_chip_parts = []
            self._caption_owner_id = (
                self._current_figure.element_id
                if self._current_figure is not None
                else self._current_table_id
            )

        if tag == "img":
            source = values.get("src", "").strip()
            if self._current_figure is not None:
                if source:
                    self._current_figure.image_sources.add(source)
                if values.get("alt", "").strip():
                    self._current_figure.has_accessible_visual = True
            if self._current_slide is not None and source:
                self._current_slide.image_sources.add(source)
        elif (
            tag == "svg"
            and self._current_figure is not None
            and values.get("role") == "img"
            and values.get("aria-label", "").strip()
        ):
            self._current_figure.has_accessible_visual = True

        if tag == "a" and self._current_slide is not None:
            parsed = urlparse(values.get("href", "").strip())
            if parsed.path == "report.html" and parsed.fragment:
                target = unquote(parsed.fragment)
                self._current_slide.report_links.add(target)
                if "report-ref" in classes:
                    self._current_report_reference = ReportReferenceTrace(target=target)
                    self._current_slide.report_references.append(
                        self._current_report_reference
                    )
        elif tag == "a" and "asset-ref" in classes:
            parsed = urlparse(values.get("href", "").strip())
            if not parsed.path and parsed.fragment:
                self._current_report_reference = ReportReferenceTrace(
                    target=unquote(parsed.fragment)
                )
                self.internal_report_references.append(
                    self._current_report_reference
                )

    def handle_data(self, data: str) -> None:
        if self._caption_chip_parts is not None:
            self._caption_chip_parts.append(data)
        if self._in_figure_caption and self._current_figure is not None:
            self._current_figure.caption_parts.append(data)
        if self._in_asset_note and self._current_figure is not None:
            self._current_figure.note_parts.append(data)
        if self._current_report_reference is not None:
            self._current_report_reference.text_parts.append(data)
        if self._current_slide is not None:
            self._current_slide.text_parts.append(data)
        if (
            self._report_text_section_depth
            and not self._in_figure_caption
            and not self._in_table_caption
            and not self._in_asset_note
            and self._caption_chip_parts is None
            and self._current_report_reference is None
        ):
            self.unlinked_report_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._caption_chip_parts is not None:
            chip = " ".join(" ".join(self._caption_chip_parts).split())
            match = CAPTION_NUMBER_RE.fullmatch(chip)
            if match:
                label = (match.group(1), int(match.group(2)))
                self.caption_numbers[label[0]].append(label[1])
                if self._caption_owner_id:
                    self.caption_labels_by_id[self._caption_owner_id] = label
            elif chip:
                self.invalid_caption_chips.append(chip)
            self._caption_chip_parts = None
            self._caption_owner_id = ""
        elif tag == "figcaption":
            self._in_figure_caption = False
        elif tag == "small":
            self._in_asset_note = False
        elif tag == "caption":
            self._in_table_caption = False
        elif tag == "figure":
            self._current_figure = None
            self._in_figure_caption = False
            self._in_asset_note = False
        elif tag == "table":
            self._current_table_id = ""
        elif tag == "a":
            self._current_report_reference = None
        elif tag == "section":
            if self._section_depth == self._report_text_section_depth:
                self._report_text_section_depth = 0
            if (
                self._current_slide is not None
                and self._section_depth == self._slide_section_depth
            ):
                self._current_slide = None
                self._slide_section_depth = 0
            self._section_depth = max(0, self._section_depth - 1)


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


def normalize_font_stack(value: str) -> tuple[str, ...]:
    return tuple(
        part.strip().strip("\"'").lower()
        for part in value.replace("\n", " ").split(",")
        if part.strip()
    )


def report_font_stack(css_text: str) -> tuple[str, ...]:
    variable = re.search(r"--font-sans-ko\s*:\s*([^;]+);", css_text, re.IGNORECASE)
    if variable:
        return normalize_font_stack(variable.group(1))
    body = re.search(
        r"\bbody\s*\{[^}]*font-family\s*:\s*([^;]+);",
        css_text,
        re.IGNORECASE | re.DOTALL,
    )
    return normalize_font_stack(body.group(1)) if body else ()


def svg_primary_font_stack(svg_text: str) -> tuple[str, ...]:
    for pattern in (
        r"\.t\s*\{[^}]*font-family\s*:\s*([^;}]+)",
        r"\.text\s*\{[^}]*font-family\s*:\s*([^;}]+)",
        r"\btext\s*\{[^}]*font-family\s*:\s*([^;}]+)",
    ):
        match = re.search(pattern, svg_text, re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_font_stack(match.group(1))
    for declaration in FONT_FAMILY_RE.finditer(svg_text):
        stack = normalize_font_stack(declaration.group(1))
        # A code-label class can legitimately use a mono face before the
        # diagram's default text class is declared. It is not the SVG's
        # primary Korean font contract, so continue to the next declaration.
        if "monospace" not in stack:
            return stack
    attribute = FONT_ATTRIBUTE_RE.search(svg_text)
    return normalize_font_stack(attribute.group(1)) if attribute else ()


def validate_caption_sequence(
    trace: ReportDeckTraceParser, report_html: Path, errors: list[str]
) -> None:
    for chip in trace.invalid_caption_chips:
        errors.append(f"invalid report caption number in {report_html}: {chip!r}")
    for kind, numbers in trace.caption_numbers.items():
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            errors.append(
                f"report {kind} caption numbers must be consecutive 1..N in "
                f"{report_html}: found {numbers}, expected {expected}"
            )


def extract_caption_references(text: str) -> set[tuple[str, int]]:
    """Expand visible report labels such as '그림 1–3' and '표 2·4'.

    Decimal source labels such as '교재 표 3.3' are deliberately ignored because
    they identify the book's assets, not this report's caption sequence.
    """
    references: set[tuple[str, int]] = set()
    report_text = EXTERNAL_CAPTION_REFERENCE_RE.sub("", text)
    for match in CAPTION_REFERENCE_GROUP_RE.finditer(report_text):
        kind, specification = match.groups()
        for part in re.split(r"\s*[·,]\s*", specification):
            bounds = re.split(r"\s*[-–]\s*", part)
            if len(bounds) == 1:
                references.add((kind, int(bounds[0])))
                continue
            if len(bounds) == 2:
                start, end = (int(value) for value in bounds)
                step = 1 if end >= start else -1
                references.update(
                    (kind, number) for number in range(start, end + step, step)
                )
    return references


def validate_internal_report_references(
    trace: ReportDeckTraceParser, report_html: Path, errors: list[str]
) -> None:
    for report_reference in trace.internal_report_references:
        expected = trace.caption_labels_by_id.get(report_reference.target)
        if expected is None:
            errors.append(
                f"internal report asset reference targets an id without a caption "
                f"label in {report_html}: #{report_reference.target}"
            )
            continue
        visible = extract_caption_references(report_reference.text)
        if visible != {expected}:
            expected_text = f"{expected[0]} {expected[1]}"
            found_text = (
                ", ".join(f"{kind} {number}" for kind, number in sorted(visible))
                if visible
                else "<missing>"
            )
            errors.append(
                f"internal report caption reference is stale in {report_html}: "
                f"#{report_reference.target} expects {expected_text}, found "
                f"{found_text} in {report_reference.text!r}"
            )
    unlinked = extract_caption_references(" ".join(trace.unlinked_report_text_parts))
    if unlinked:
        found_text = ", ".join(
            f"{kind} {number}" for kind, number in sorted(unlinked)
        )
        errors.append(
            f"internal report caption reference must use an asset-ref anchor in "
            f"{report_html}: {found_text}"
        )


def validate_font_stack_contract(
    session_dir: Path,
    session_key: str,
    errors: list[str],
) -> None:
    css_path = session_dir / "assets" / "report.css"
    figures_dir = session_dir / "assets" / "figs"
    if not css_path.is_file() or not figures_dir.is_dir():
        return
    expected = report_font_stack(css_path.read_text(encoding="utf-8", errors="replace"))
    if not expected:
        errors.append(f"report CSS has no readable body font stack: {css_path}")
        return

    drift: list[str] = []
    for svg_path in sorted(figures_dir.glob("*.svg")):
        svg_text = svg_path.read_text(encoding="utf-8", errors="replace")
        if "<text" not in svg_text:
            continue
        actual = svg_primary_font_stack(svg_text)
        if not actual:
            drift.append(f"{svg_path.name}=<missing>")
        elif actual != expected:
            drift.append(f"{svg_path.name}={','.join(actual)}")

    if drift and session_key not in FONT_STACK_DRIFT_ALLOWLIST:
        errors.append(
            f"SVG/report.css font stack mismatch in {session_dir}; expected "
            f"{','.join(expected)}; drift: {drift}"
        )


def body_rule_blocks(css_text: str) -> list[str]:
    """Declaration blocks of every rule whose selector is exactly `body`.

    The rule regex cannot nest, so a `body` rule wrapped in `@media ... { }` is
    reached the same way a top-level one is.
    """
    stripped = CSS_COMMENT_RE.sub(" ", css_text)
    return [
        match.group(2)
        for match in CSS_RULE_RE.finditer(stripped)
        if " ".join(match.group(1).split()).lower() == "body"
    ]


def validate_cjk_wrap_contract(session_dir: Path, errors: list[str]) -> None:
    """Both CJK wrap declarations must sit on `body` so they inherit everywhere.

    Declaring them only on table cells or the cover title leaves section
    headings and paragraphs on the browser default, which breaks Korean
    mid-어절. Declaring `keep-all` without `overflow-wrap: break-word` is the
    banned solo form: it then lets long Latin tokens and URLs overflow.
    """
    for name in ("report.css", "deck.css"):
        css_path = session_dir / "assets" / name
        if not css_path.is_file():
            continue
        blocks = body_rule_blocks(css_path.read_text(encoding="utf-8", errors="replace"))
        missing = [
            declaration
            for declaration, pattern in CJK_WRAP_DECLARATIONS
            if not any(pattern.search(block) for block in blocks)
        ]
        if missing:
            errors.append(
                f"the `body` rule must declare {' and '.join(missing)} in {css_path}"
            )


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
        artifacts = metadata.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(artifacts) != len(set(artifacts))
            or not set(artifacts).issubset(SUPPORTED_ARTIFACTS)
        ):
            errors.append(
                f"artifacts must be a unique non-empty subset of "
                f"{sorted(SUPPORTED_ARTIFACTS)} in {metadata_path}"
            )
            artifacts = []

        deck_html = metadata_path.parent / "index.html"
        report_html = metadata_path.parent / "report.html"
        if "slides" in artifacts and not deck_html.is_file():
            errors.append(f"slides artifact is missing index.html: {metadata_path.parent}")
        if "report" in artifacts and not report_html.is_file():
            errors.append(f"report artifact is missing report.html: {metadata_path.parent}")

        template_id = metadata.get("template")
        if "slides" in artifacts and template_id == "study-deck-v1" and deck_html.is_file():
            deck_text = deck_html.read_text(encoding="utf-8", errors="replace")
            if 'data-deck-template="study-deck-v1"' not in deck_text:
                errors.append(f"study-deck-v1 marker is missing in {deck_html}")
            for asset in ("assets/deck.css", "assets/deck.js"):
                if not (metadata_path.parent / asset).is_file():
                    errors.append(
                        f"study-deck-v1 asset is missing in {metadata_path.parent}: {asset}"
                    )

        report_template_id = metadata.get("report_template")
        if (
            "report" in artifacts
            and report_template_id == "study-report-v1"
            and report_html.is_file()
        ):
            report_text = report_html.read_text(encoding="utf-8", errors="replace")
            if 'data-report-template="study-report-v1"' not in report_text:
                errors.append(f"study-report-v1 marker is missing in {report_html}")
            for asset in ("assets/report.css", "assets/report.js"):
                if not (metadata_path.parent / asset).is_file():
                    errors.append(
                        f"study-report-v1 asset is missing in {metadata_path.parent}: {asset}"
                    )

        validate_cjk_wrap_contract(metadata_path.parent, errors)

        if (
            {"report", "slides"}.issubset(artifacts)
            and template_id == "study-deck-v1"
            and report_template_id == "study-report-v1"
            and deck_html.is_file()
            and report_html.is_file()
        ):
            expected_workflow = "raw-report-deck-v1"
            if metadata.get("workflow") != expected_workflow:
                errors.append(
                    f"paired study artifacts must use workflow={expected_workflow!r} "
                    f"in {metadata_path}"
                )
            if metadata.get("report_source") != "report.html":
                errors.append(
                    f"paired study artifacts must use report_source='report.html' "
                    f"in {metadata_path}"
                )

            report_trace = ReportDeckTraceParser()
            report_trace.feed(report_html.read_text(encoding="utf-8", errors="replace"))
            deck_trace = ReportDeckTraceParser()
            deck_trace.feed(deck_html.read_text(encoding="utf-8", errors="replace"))

            for duplicate in sorted(report_trace.duplicate_ids):
                errors.append(f"duplicate report id in {report_html}: {duplicate}")
            for duplicate in sorted(deck_trace.duplicate_ids):
                errors.append(f"duplicate deck id in {deck_html}: {duplicate}")
            validate_caption_sequence(report_trace, report_html, errors)
            validate_internal_report_references(report_trace, report_html, errors)
            if report_trace.required_figures_without_id:
                errors.append(
                    f"{report_trace.required_figures_without_id} required report figure(s) "
                    f"lack a stable id in {report_html}"
                )
            for figure_id, figure in sorted(report_trace.figures.items()):
                if not figure.caption:
                    errors.append(
                        f"report figure lacks a non-empty figcaption in "
                        f"{report_html}: {figure_id}"
                    )
                if not figure.note:
                    errors.append(
                        f"report figure lacks a non-empty asset note in "
                        f"{report_html}: {figure_id}"
                    )
                if not figure.has_accessible_visual:
                    errors.append(
                        f"report figure lacks non-empty image alt text or an accessible "
                        f"inline SVG in {report_html}: {figure_id}"
                    )
            if deck_trace.deck_report_source != "report.html":
                errors.append(
                    f"deck must declare data-report-source='report.html' on its main "
                    f"element: {deck_html}"
                )
            if not deck_trace.slides:
                errors.append(f"deck has no traceable slides: {deck_html}")

            labels = [slide.label for slide in deck_trace.slides]
            for index, label in enumerate(labels, start=1):
                if not label:
                    errors.append(
                        f"slide {index} lacks a non-empty aria-label in {deck_html}"
                    )
            duplicates = sorted(
                label for label in set(labels) if label and labels.count(label) > 1
            )
            if duplicates:
                errors.append(
                    f"deck slide aria-label values must be unique in {deck_html}: "
                    f"{duplicates}"
                )

            referenced: set[str] = set()
            deck_image_sources: set[str] = set()
            for slide in deck_trace.slides:
                label = slide.label or "unnamed slide"
                if not slide.refs:
                    errors.append(
                        f"slide lacks data-report-refs in {deck_html}: {label}"
                    )
                    continue
                referenced.update(slide.refs)
                deck_image_sources.update(slide.image_sources)
                unknown = sorted(set(slide.refs) - report_trace.ids)
                if unknown:
                    errors.append(
                        f"slide references unknown report id(s) in {deck_html} "
                        f"({label}): {unknown}"
                    )
                untraced_links = sorted(slide.report_links - set(slide.refs))
                if untraced_links:
                    errors.append(
                        f"slide report.html anchor is absent from data-report-refs in "
                        f"{deck_html} ({label}): {untraced_links}"
                    )
                for report_reference in slide.report_references:
                    expected = report_trace.caption_labels_by_id.get(
                        report_reference.target
                    )
                    if expected is None:
                        continue
                    visible = extract_caption_references(report_reference.text)
                    if visible and visible != {expected}:
                        expected_text = f"{expected[0]} {expected[1]}"
                        found_text = ", ".join(
                            f"{kind} {number}" for kind, number in sorted(visible)
                        )
                        errors.append(
                            f"deck report reference caption label is stale in "
                            f"{deck_html} ({label}): report.html#"
                            f"{report_reference.target} expects {expected_text}, "
                            f"found {found_text} in {report_reference.text!r}"
                        )
                visible_slide_labels = extract_caption_references(slide.text)
                expected_slide_labels = {
                    report_trace.caption_labels_by_id[report_id]
                    for report_id in slide.refs
                    if report_id in report_trace.caption_labels_by_id
                }
                stale_slide_labels = visible_slide_labels - expected_slide_labels
                if stale_slide_labels:
                    found_text = ", ".join(
                        f"{kind} {number}"
                        for kind, number in sorted(stale_slide_labels)
                    )
                    expected_text = (
                        ", ".join(
                            f"{kind} {number}"
                            for kind, number in sorted(expected_slide_labels)
                        )
                        if expected_slide_labels
                        else "<none>"
                    )
                    errors.append(
                        f"deck slide contains caption label(s) not backed by "
                        f"data-report-refs in {deck_html} ({label}): "
                        f"{found_text}; expected one of {expected_text}"
                    )

            missing_sections = sorted(report_trace.report_sections - referenced)
            if missing_sections:
                errors.append(
                    f"deck does not cover report section id(s) in {deck_html}: "
                    f"{missing_sections}"
                )
            missing_figures = sorted(report_trace.required_figures - referenced)
            if missing_figures:
                errors.append(
                    f"deck does not reuse required report figure id(s) in {deck_html}: "
                    f"{missing_figures}"
                )
            for figure_id in sorted(report_trace.required_figures):
                figure = report_trace.figures[figure_id]
                if not figure.image_sources:
                    errors.append(
                        f"required report figure must use a reusable img src in "
                        f"{report_html}: {figure_id}"
                    )
                    continue
                if not (figure.image_sources & deck_image_sources):
                    errors.append(
                        f"deck does not visually reuse required report figure src in "
                        f"{deck_html}: {figure_id} -> {sorted(figure.image_sources)}"
                    )

            validate_font_stack_contract(
                metadata_path.parent,
                f"{study_id}/{session_id}",
                errors,
            )

        key = (study_id, session_id)
        if key in seen:
            errors.append(f"duplicate presentation id: {study_id}/{session_id}")
        seen.add(key)

        session_size = sum(
            path.stat().st_size for path in metadata_path.parent.rglob("*") if path.is_file()
        )
        if session_size > MAX_SESSION_BYTES:
            errors.append(
                f"session artifacts exceed {MAX_SESSION_BYTES // 1024 // 1024} MiB: "
                f"{metadata_path.parent} ({session_size} bytes)"
            )


def validate_reference(
    html_path: Path,
    site: Path,
    tag: str,
    attribute: str,
    value: str,
    errors: list[str],
    warnings: list[str],
    check_local_targets: bool = True,
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
    if check_local_targets and not target.exists():
        errors.append(f"broken local reference in {html_path}: {value}")


def validate_html(site: Path, errors: list[str], warnings: list[str]) -> None:
    # docs/archive/ 아래는 legacy-pages.toml에서 preserve로 결정된 과거 스냅샷이다.
    # 저작 시점 기준이 다르므로 품질 규칙(title/viewport/lang/alt/링크 무결성)은
    # 면제하고, 안전 규칙(form, protocol-relative, 비HTTPS, docs 이탈)은 유지한다.
    archive_root = site / "archive"
    for html_path in sorted(site.rglob("*.html")):
        legacy = archive_root in html_path.parents
        text = html_path.read_text(encoding="utf-8", errors="replace")
        parser = PageParser()
        parser.feed(text)
        if not legacy:
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
                html_path, site, tag, attribute, value, errors, warnings,
                check_local_targets=not legacy,
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
        if not path.is_file() or path.suffix.lower() not in {".css", ".html", ".js", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if MERGE_MARKER_RE.search(text):
            errors.append(f"unresolved merge marker in public asset: {path}")
        if path.suffix.lower() == ".css" and STANDALONE_CSS_PLUS_RE.search(text):
            errors.append(f"standalone '+' diff artifact in public CSS: {path}")


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
