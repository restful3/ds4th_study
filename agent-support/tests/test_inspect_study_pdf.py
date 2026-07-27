from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "agent-support" / "scripts" / "inspect-study-pdf.py"


def load_module():
    spec = importlib.util.spec_from_file_location("inspect_study_pdf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PdfInfoParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pdf = load_module()

    def test_parses_positive_page_count(self) -> None:
        self.assertEqual(self.pdf.parse_page_count("Title: Demo\nPages:          38\n"), 38)

    def test_rejects_missing_page_count(self) -> None:
        with self.assertRaises(ValueError):
            self.pdf.parse_page_count("Title: Demo\n")

    def test_parses_standard_and_page_specific_sizes(self) -> None:
        self.assertEqual(
            self.pdf.parse_page_size("Page size: 595.28 x 841.89 pts (A4)"),
            (595.28, 841.89),
        )
        self.assertEqual(
            self.pdf.parse_page_size("Page    3 size: 595.28 x 841.89 pts (A4)"),
            (595.28, 841.89),
        )

    def test_browser_shots_path_contract(self) -> None:
        self.assertTrue(
            self.pdf.is_under_browser_shots(
                Path("/workspace/tmp/browser-shots/report.pdf")
            )
        )
        self.assertFalse(self.pdf.is_under_browser_shots(Path("/workspace/docs/report.pdf")))


if __name__ == "__main__":
    unittest.main()
