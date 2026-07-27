"""저장소 source 와 owner-only 설치본이 어긋난 것을 무엇이 알려주는가.

guarded browser runner 는 설치본만 실행하도록 강제한다(저장소 사본을 실행하면
거부한다). 그래서 저장소를 수정해도 자동 반영되지 않고, 반대로 설치본이 조용히
낡아도 아무도 모른다. 최초 배포에서 동일성은 사람이 손으로 sha256 을 비교한 것이
전부였고, 설치 절차도 커밋돼 있지 않았다 — 저장소 mode(755/664)와 설치 요구
(700/600)가 달라 재설치 때마다 chmod 를 기억해야 했다.

그래서 설치를 스크립트로 고정하고, 설치 시 기록한 manifest 로 (1) 저장소↔설치본
staleness 와 (2) 설치 후 변조를 한 명령으로 판정한다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "agent-support" / "scripts"
INSTALLER_NAME = "install-safe-browser-shot.sh"
WRAPPER_NAME = "safe-browser-shot.sh"
GUARD_NAME = "safe_browser_guard.py"
RUNNER_DIR_NAME = "ds4th-safe-browser-shot"
MANIFEST_NAME = "manifest.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InstallerTestCase(unittest.TestCase):
    """설치를 격리된 source/prefix 쌍에서 재현한다."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)

        self.source = root / "source"
        self.source.mkdir()
        for name in (INSTALLER_NAME, WRAPPER_NAME, GUARD_NAME):
            origin = SCRIPTS / name
            self.assertTrue(origin.exists(), msg=f"missing source file: {origin}")
            shutil.copy2(origin, self.source / name)
        (self.source / INSTALLER_NAME).chmod(0o700)

        self.prefix = root / "prefix"
        self.prefix.mkdir()
        self.runner_dir = self.prefix / RUNNER_DIR_NAME

    def installer(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.source / INSTALLER_NAME), "--prefix", str(self.prefix), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def install(self) -> subprocess.CompletedProcess[str]:
        result = self.installer()
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result


class InstallTests(InstallerTestCase):
    """설치는 owner-only mode 와 manifest 를 함께 남긴다."""

    def test_the_runner_directory_is_owner_only(self) -> None:
        self.install()

        self.assertEqual(oct(self.runner_dir.stat().st_mode & 0o777), oct(0o700))

    def test_the_wrapper_is_owner_executable_and_the_guard_is_not(self) -> None:
        self.install()

        wrapper = self.runner_dir / WRAPPER_NAME
        guard = self.runner_dir / GUARD_NAME
        self.assertEqual(oct(wrapper.stat().st_mode & 0o777), oct(0o700))
        self.assertEqual(oct(guard.stat().st_mode & 0o777), oct(0o600))

    def test_installed_bytes_match_the_source(self) -> None:
        self.install()

        for name in (WRAPPER_NAME, GUARD_NAME):
            with self.subTest(name=name):
                self.assertEqual(sha256(self.runner_dir / name), sha256(self.source / name))

    def test_installed_runner_and_guard_include_pdf_mode(self) -> None:
        self.install()

        wrapper = (self.runner_dir / WRAPPER_NAME).read_text()
        guard = (self.runner_dir / GUARD_NAME).read_text()
        self.assertIn("--format png|pdf", wrapper)
        self.assertIn("--print-to-pdf=", guard)

    def test_the_manifest_records_both_files(self) -> None:
        self.install()

        manifest = (self.runner_dir / MANIFEST_NAME).read_text()
        for name in (WRAPPER_NAME, GUARD_NAME):
            with self.subTest(name=name):
                self.assertIn(name, manifest)
                self.assertIn(sha256(self.source / name), manifest)

    def test_the_manifest_is_owner_readable_only(self) -> None:
        self.install()

        manifest = self.runner_dir / MANIFEST_NAME
        self.assertEqual(oct(manifest.stat().st_mode & 0o777), oct(0o600))

    def test_reinstalling_over_an_existing_install_keeps_a_backup(self) -> None:
        self.install()
        (self.source / GUARD_NAME).write_text("# changed source\n")

        self.install()

        backups = sorted(self.runner_dir.glob("rollback-*"))
        self.assertTrue(backups, msg="no rollback backup directory was created")
        self.assertTrue((backups[-1] / GUARD_NAME).exists())


class VerifyTests(InstallerTestCase):
    """--verify 는 staleness 와 변조를 모두 non-zero 로 알린다."""

    def test_verify_passes_immediately_after_install(self) -> None:
        self.install()

        result = self.installer("--verify")

        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_verify_fails_when_nothing_is_installed(self) -> None:
        result = self.installer("--verify")

        self.assertNotEqual(result.returncode, 0)

    def test_verify_detects_a_modified_installed_file(self) -> None:
        self.install()
        installed_guard = self.runner_dir / GUARD_NAME
        installed_guard.chmod(0o600)
        installed_guard.write_text("# tampered after install\n")

        result = self.installer("--verify")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(GUARD_NAME, result.stdout + result.stderr)

    def test_verify_detects_a_stale_install_after_the_source_changes(self) -> None:
        self.install()
        (self.source / WRAPPER_NAME).write_text("#!/bin/bash\n# newer source\n")

        result = self.installer("--verify")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(WRAPPER_NAME, result.stdout + result.stderr)

    def test_verify_fails_when_the_manifest_is_missing(self) -> None:
        self.install()
        (self.runner_dir / MANIFEST_NAME).unlink()

        result = self.installer("--verify")

        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
