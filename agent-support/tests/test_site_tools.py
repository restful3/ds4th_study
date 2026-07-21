from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_INDEX = REPO_ROOT / "agent-support" / "scripts" / "build-index.py"
VALIDATE_SITE = REPO_ROOT / "agent-support" / "scripts" / "validate-site.py"
NEW_PRESENTATION = REPO_ROOT / "agent-support" / "scripts" / "new-presentation.py"


ACTIVE_REGISTRY = '''schema_version = 1

[[studies]]
id = "sample-2026"
slug = "sample-study"
title = "Sample Study"
title_ko = "샘플 스터디"
status = "active"
start_date = "2026-01-01"
end_date = "2026-02-01"
materials_path = "source/Sample Study"
archive_path = "archive/Sample Study"
presentation_path = "docs/studies/sample-study"
'''


ARCHIVED_REGISTRY = ACTIVE_REGISTRY.replace(
    'status = "active"', 'status = "archived"'
).replace(
    'materials_path = "source/Sample Study"',
    'materials_path = "archive/Sample Study"',
)


DECK_METADATA = '''study_id = "sample-2026"
session_id = "2026-01-01-ch01"
title = "첫 번째 발표"
date = "2026-01-01"
presenters = ["발표자"]
chapters = ["Chapter 1"]
template = "study-deck-v1"
report_template = "study-report-v1"
workflow = "raw-report-deck-v1"
report_source = "report.html"
artifacts = ["report", "slides"]
'''


DECK_HTML = '''<!doctype html>
<html lang="ko" data-deck-template="study-deck-v1">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>첫 번째 발표</title>
</head>
<body><main data-report-source="report.html"><section class="slide" aria-label="표지" data-report-refs="summary"><h1>첫 번째 발표</h1></section></main></body>
</html>
'''


REPORT_HTML = '''<!doctype html>
<html lang="ko" data-report-template="study-report-v1">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>첫 번째 발표 · 상세 리포트</title>
</head>
<body><main><section class="report-section" id="summary"><h1>첫 번째 발표 · 상세 리포트</h1></section></main></body>
</html>
'''


class SiteToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.registry = self.root / "studies.toml"
        self.site = self.root / "docs"
        (self.site / "assets").mkdir(parents=True)
        (self.site / ".nojekyll").touch()
        (self.site / "assets" / "site.css").write_text(
            "body { font-family: sans-serif; }\n", encoding="utf-8"
        )
        (self.site / "assets" / "favicon.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>\n',
            encoding="utf-8",
        )
        self.registry.write_text(ACTIVE_REGISTRY, encoding="utf-8")

        self.deck = (
            self.site
            / "studies"
            / "sample-study"
            / "presentations"
            / "2026-01-01-ch01"
        )
        self.deck.mkdir(parents=True)
        (self.deck / "presentation.toml").write_text(
            DECK_METADATA, encoding="utf-8"
        )
        (self.deck / "index.html").write_text(DECK_HTML, encoding="utf-8")
        (self.deck / "report.html").write_text(REPORT_HTML, encoding="utf-8")
        (self.deck / "assets").mkdir()
        for name in ("deck.css", "deck.js", "report.css", "report.js"):
            (self.deck / "assets" / name).write_text("/* test */\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_tool(self, script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def build(self) -> subprocess.CompletedProcess[str]:
        return self.run_tool(
            BUILD_INDEX,
            "--registry",
            str(self.registry),
            "--site",
            str(self.site),
        )

    def validate(self) -> subprocess.CompletedProcess[str]:
        return self.run_tool(
            VALIDATE_SITE,
            "--registry",
            str(self.registry),
            "--site",
            str(self.site),
        )

    def test_public_url_survives_archive_transition(self) -> None:
        active_result = self.build()
        self.assertEqual(active_result.returncode, 0, active_result.stderr)
        study_page = self.site / "studies" / "sample-study" / "index.html"
        active_page = study_page.read_text(encoding="utf-8")
        self.assertIn("진행 중", active_page)
        self.assertIn("presentations/2026-01-01-ch01/", active_page)
        self.assertIn("presentations/2026-01-01-ch01/report.html", active_page)

        self.registry.write_text(ARCHIVED_REGISTRY, encoding="utf-8")
        archived_result = self.build()
        self.assertEqual(archived_result.returncode, 0, archived_result.stderr)
        archived_page = study_page.read_text(encoding="utf-8")
        self.assertIn("완료", archived_page)
        self.assertIn("presentations/2026-01-01-ch01/", archived_page)
        self.assertIn("presentations/2026-01-01-ch01/report.html", archived_page)
        self.assertEqual(study_page, self.site / "studies" / "sample-study" / "index.html")

        validation = self.validate()
        self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_validator_rejects_broken_local_asset(self) -> None:
        build_result = self.build()
        self.assertEqual(build_result.returncode, 0, build_result.stderr)
        broken = DECK_HTML.replace(
            "<h1>첫 번째 발표</h1>",
            '<h1>첫 번째 발표</h1><img src="missing.png" alt="누락 이미지">',
        )
        (self.deck / "index.html").write_text(broken, encoding="utf-8")

        validation = self.validate()
        self.assertNotEqual(validation.returncode, 0)
        self.assertIn("broken local reference", validation.stderr)

    def test_validator_rejects_missing_report_artifact(self) -> None:
        build_result = self.build()
        self.assertEqual(build_result.returncode, 0, build_result.stderr)
        (self.deck / "report.html").unlink()

        validation = self.validate()
        self.assertNotEqual(validation.returncode, 0)
        self.assertIn("report artifact is missing", validation.stderr)

    def test_validator_rejects_untraceable_deck_slide(self) -> None:
        build_result = self.build()
        self.assertEqual(build_result.returncode, 0, build_result.stderr)
        untraceable = DECK_HTML.replace(' data-report-refs="summary"', "")
        (self.deck / "index.html").write_text(untraceable, encoding="utf-8")

        validation = self.validate()
        self.assertNotEqual(validation.returncode, 0)
        self.assertIn("slide lacks data-report-refs", validation.stderr)

    def test_validator_rejects_unknown_report_reference(self) -> None:
        build_result = self.build()
        self.assertEqual(build_result.returncode, 0, build_result.stderr)
        unknown = DECK_HTML.replace(
            'data-report-refs="summary"', 'data-report-refs="summary missing-anchor"'
        )
        (self.deck / "index.html").write_text(unknown, encoding="utf-8")

        validation = self.validate()
        self.assertNotEqual(validation.returncode, 0)
        self.assertIn("slide references unknown report id", validation.stderr)

    def test_new_presentation_uses_canonical_template_without_overwrite(self) -> None:
        session = "2026-01-15-ch02-ch03"
        result = self.run_tool(
            NEW_PRESENTATION,
            "--study",
            "sample-2026",
            "--session",
            session,
            "--title",
            "두 번째 발표",
            "--subtitle",
            "개념에서 적용까지",
            "--date",
            "2026-01-15",
            "--presenter",
            "발표자 A",
            "--presenter",
            "발표자 B",
            "--chapter",
            "Chapter 2",
            "--chapter",
            "Chapter 3",
            "--registry",
            str(self.registry),
            "--site",
            str(self.site),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        target = self.site / "studies" / "sample-study" / "presentations" / session
        generated_html = (target / "index.html").read_text(encoding="utf-8")
        with (target / "presentation.toml").open("rb") as stream:
            metadata = tomllib.load(stream)
        self.assertIn('data-deck-template="study-deck-v1"', generated_html)
        self.assertIn("두 번째 발표", generated_html)
        self.assertNotIn("ConnectBrick", generated_html)
        self.assertNotIn("{{", generated_html)
        self.assertEqual(generated_html.count('<section class="slide'), 22)
        self.assertEqual(generated_html.count('slide slide--section'), 4)
        self.assertGreaterEqual(generated_html.count('aria-label="'), 22)
        self.assertIn('data-report-source="report.html"', generated_html)
        self.assertEqual(generated_html.count('data-report-refs="'), 22)
        self.assertIn('class="deck-settings"', generated_html)
        self.assertIn('id="settingsToggle"', generated_html)
        self.assertIn('href="../../">Index</a>', generated_html)
        self.assertIn('href="./" aria-current="page">Slides</a>', generated_html)
        self.assertIn('href="report.html">Report</a>', generated_html)
        self.assertEqual(metadata["template"], "study-deck-v1")
        self.assertEqual(metadata["report_template"], "study-report-v1")
        self.assertEqual(metadata["workflow"], "raw-report-deck-v1")
        self.assertEqual(metadata["report_source"], "report.html")
        self.assertEqual(metadata["artifacts"], ["report", "slides"])
        self.assertEqual(metadata["presenters"], ["발표자 A", "발표자 B"])
        self.assertTrue((target / "assets" / "deck.css").is_file())
        self.assertTrue((target / "assets" / "deck.js").is_file())
        self.assertIn(
            "setSettings",
            (target / "assets" / "deck.js").read_text(encoding="utf-8"),
        )
        generated_report = (target / "report.html").read_text(encoding="utf-8")
        self.assertIn('data-report-template="study-report-v1"', generated_report)
        self.assertIn("두 번째 발표", generated_report)
        self.assertNotIn("{{", generated_report)
        self.assertEqual(generated_report.count('<section class="report-section'), 7)
        self.assertGreaterEqual(generated_report.count('class="report-figure"'), 3)
        self.assertGreaterEqual(generated_report.count('data-deck-use="required"'), 3)
        self.assertIn('id="fig-concept"', generated_report)
        self.assertIn('id="table-decision"', generated_report)
        self.assertTrue((target / "assets" / "report.css").is_file())
        self.assertTrue((target / "assets" / "report.js").is_file())
        self.assertTrue((target / "assets" / "figs").is_dir())
        self.assertIn(
            "setupImageLightbox",
            (target / "assets" / "report.js").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "Extended study presentation components",
            (target / "assets" / "deck.css").read_text(encoding="utf-8"),
        )

        build_result = self.build()
        self.assertEqual(build_result.returncode, 0, build_result.stderr)
        validation = self.validate()
        self.assertEqual(validation.returncode, 0, validation.stderr)

        second_result = self.run_tool(
            NEW_PRESENTATION,
            "--study",
            "sample-2026",
            "--session",
            session,
            "--title",
            "덮어쓰기 시도",
            "--date",
            "2026-01-15",
            "--presenter",
            "발표자 A",
            "--chapter",
            "Chapter 2",
            "--registry",
            str(self.registry),
            "--site",
            str(self.site),
        )
        self.assertNotEqual(second_result.returncode, 0)
        self.assertIn("already exists", second_result.stderr)


if __name__ == "__main__":
    unittest.main()
