from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "agent-support" / "scripts" / "render-shot-manifest.py"


FAKE_RUNNER = """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
value = {args[index][2:]: args[index + 1] for index in range(0, len(args), 2)}
with Path(os.environ["FAKE_RUNNER_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(value, ensure_ascii=False) + "\\n")
if "fail" in value["url"]:
    raise SystemExit(7)
output = Path(value["output"])
output.write_bytes(
    b"%PDF-test" if value["format"] == "pdf" else b"\\x89PNG\\r\\n\\x1a\\n"
)
"""


class ShotManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output_root = self.root / "tmp" / "browser-shots"
        self.output_root.mkdir(parents=True)
        self.runner = self.root / "fake-runner.py"
        self.runner.write_text(FAKE_RUNNER, encoding="utf-8")
        self.runner.chmod(0o700)
        self.log = self.root / "runner.log"

    def run_driver(self, captures: list[dict[str, object]]) -> subprocess.CompletedProcess[str]:
        manifest = self.root / "manifest.json"
        manifest.write_text(
            json.dumps({"version": 1, "captures": captures}),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["FAKE_RUNNER_LOG"] = str(self.log)
        return subprocess.run(
            [
                sys.executable,
                str(DRIVER),
                "--manifest",
                str(manifest),
                "--runner",
                str(self.runner),
                "--output-root",
                str(self.output_root),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_runs_png_and_pdf_serially(self) -> None:
        png = self.output_root / "deck.png"
        pdf = self.output_root / "report.pdf"

        result = self.run_driver(
            [
                {"url": "http://localhost:8000/deck", "output": str(png)},
                {
                    "url": "http://localhost:8000/report",
                    "output": str(pdf),
                    "format": "pdf",
                },
            ]
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([row["format"] for row in rows], ["png", "pdf"])
        self.assertEqual([row["output"] for row in rows], [str(png), str(pdf)])
        self.assertTrue(png.is_file())
        self.assertTrue(pdf.is_file())

    def test_stops_after_the_first_failed_capture(self) -> None:
        result = self.run_driver(
            [
                {
                    "url": "http://localhost:8000/first",
                    "output": str(self.output_root / "first.png"),
                },
                {
                    "url": "http://localhost:8000/fail",
                    "output": str(self.output_root / "failed.png"),
                },
                {
                    "url": "http://localhost:8000/third",
                    "output": str(self.output_root / "third.png"),
                },
            ]
        )

        self.assertEqual(result.returncode, 7)
        rows = self.log.read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertFalse((self.output_root / "third.png").exists())

    def test_rejects_output_outside_browser_shots(self) -> None:
        result = self.run_driver(
            [
                {
                    "url": "http://localhost:8000/deck",
                    "output": str(self.root / "outside.png"),
                }
            ]
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must stay under", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
