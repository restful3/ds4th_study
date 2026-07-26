"""guarded browser 실행 중 containment 를 얼마나 싸게 확인하는가.

최초 guard 는 폴링마다(0.25초) 전 프로세스의 `/proc/<pid>/environ` 을 읽고, 이어서
추적 중인 pid 마다 `/proc/<pid>/cgroup` 을 따로 읽었다. 프로세스가 1,400개대인
호스트에서는 초당 수천 건의 read 가 되고, 하필 메모리 압력에 가장 민감한 구간에서
돈다 — 보호장치가 스스로 부하를 만든다.

cgroup v2 는 서비스 cgroup 의 `cgroup.procs` **한 파일** 에 멤버십 전체를 담는다.
그래서 "안에 누가 있나" 는 1회 read 로 끝나고, per-pid cgroup read 는 필요 없다.
다만 `cgroup.procs` 는 이탈자를 알려주지 못한다(이탈하면 목록에서 사라진다).
이탈 탐지는 "우리 run 의 pid 인데 목록에 없다" 로 하고, 한 번도 자손으로 관측되지
못한 채 reparent 된 프로세스를 잡기 위한 marker 스캔은 백스톱으로 낮은 주기만
남긴다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "agent-support" / "scripts" / "safe_browser_guard.py"


def load_guard():
    spec = importlib.util.spec_from_file_location("safe_browser_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ContainedPidsTests(unittest.TestCase):
    """서비스 cgroup 안에 있는 pid 를 한 번의 read 로 얻는다."""

    def setUp(self) -> None:
        self.guard = load_guard()
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_membership_comes_from_the_cgroup_procs_file(self) -> None:
        (self.root / "cgroup.procs").write_text("101\n102\n103\n")

        self.assertEqual(self.guard.contained_pids(self.root), {101, 102, 103})

    def test_nested_cgroup_members_are_also_contained(self) -> None:
        (self.root / "cgroup.procs").write_text("101\n")
        nested = self.root / "child.scope"
        nested.mkdir()
        (nested / "cgroup.procs").write_text("202\n203\n")

        self.assertEqual(self.guard.contained_pids(self.root), {101, 202, 203})

    def test_blank_and_malformed_lines_are_ignored(self) -> None:
        (self.root / "cgroup.procs").write_text("101\n\n  \nnot-a-pid\n102\n")

        self.assertEqual(self.guard.contained_pids(self.root), {101, 102})

    def test_an_empty_cgroup_reports_no_members(self) -> None:
        (self.root / "cgroup.procs").write_text("")

        self.assertEqual(self.guard.contained_pids(self.root), set())

    def test_an_unreadable_cgroup_root_fails_closed(self) -> None:
        with io.StringIO() as noise, contextlib.redirect_stderr(noise):
            with self.assertRaises(SystemExit):
                self.guard.contained_pids(self.root / "absent")
            self.assertIn("cannot read", noise.getvalue())


class EscapeDetectionTests(unittest.TestCase):
    """이탈은 '우리 pid 인데 cgroup 멤버십에 없다' 로 판정한다."""

    def setUp(self) -> None:
        self.guard = load_guard()

    def test_a_live_run_pid_outside_the_cgroup_is_reported(self) -> None:
        escaped = self.guard.escaped_pids(live={101, 102, 999}, contained={101, 102})

        self.assertEqual(escaped, {999})

    def test_nothing_is_reported_when_every_live_pid_is_contained(self) -> None:
        escaped = self.guard.escaped_pids(live={101, 102}, contained={101, 102, 103})

        self.assertEqual(escaped, set())

    def test_no_live_pids_means_no_escape(self) -> None:
        self.assertEqual(self.guard.escaped_pids(live=set(), contained={101}), set())


class MarkerScanCadenceTests(unittest.TestCase):
    """전 프로세스 environ 스캔은 백스톱이므로 매 폴링마다 돌지 않는다."""

    def setUp(self) -> None:
        self.guard = load_guard()

    def test_the_first_poll_always_scans_the_marker(self) -> None:
        self.assertTrue(self.guard.should_scan_marker(0))

    def test_intermediate_polls_skip_the_marker_scan(self) -> None:
        skipped = [i for i in range(1, self.guard.MARKER_SCAN_EVERY)
                   if not self.guard.should_scan_marker(i)]

        self.assertEqual(skipped, list(range(1, self.guard.MARKER_SCAN_EVERY)))

    def test_the_marker_scan_repeats_on_the_configured_interval(self) -> None:
        interval = self.guard.MARKER_SCAN_EVERY

        self.assertTrue(self.guard.should_scan_marker(interval))
        self.assertTrue(self.guard.should_scan_marker(interval * 3))

    def test_the_backstop_interval_stays_within_a_few_seconds(self) -> None:
        """폴링 0.25초 기준으로 백스톱이 몇 초 안에 한 번은 돌아야 한다."""
        seconds = self.guard.MARKER_SCAN_EVERY * self.guard.POLL_SECONDS

        self.assertLessEqual(seconds, 5.0)
        self.assertGreater(seconds, self.guard.POLL_SECONDS)


if __name__ == "__main__":
    unittest.main()
