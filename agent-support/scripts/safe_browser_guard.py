#!/usr/bin/env python3
"""Run one browser tree inside a verified cgroup and kill any escaped descendants."""

from __future__ import annotations

import argparse
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path


UNIT_NAME = "ds4th-safe-browser-shot.service"
EXPECTED_CONTROLLERS = {
    "memory.high": "3221225472",
    "memory.max": "4294967296",
    "memory.swap.max": "0",
    "pids.max": "1024",
    "cpu.max": "300000 100000",
}
RUN_MARKER_NAME = "DS4TH_SAFE_BROWSER_RUN_ID"
POLL_SECONDS = 0.25
# Containment comes from one cgroup.procs read per poll. The all-process environ
# scan only catches a process that was reparented before we ever saw it as a
# descendant, so it runs as a low-frequency backstop instead of every poll.
MARKER_SCAN_EVERY = 8
TERM_GRACE_SECONDS = 0.50
POST_EXIT_GRACE_SECONDS = 0.50


def fail(message: str, status: int = 70) -> "NoReturn":
    print(f"safe_browser_guard.py: {message}", file=sys.stderr, flush=True)
    raise SystemExit(status)


def current_cgroup() -> str:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, _, path = line.partition("::")
            if hierarchy == "0":
                return path
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""
    return ""


def verify_own_cgroup() -> Path:
    if not os.environ.get("INVOCATION_ID"):
        fail("systemd INVOCATION_ID is absent")

    expected = current_cgroup()
    if not expected.endswith(f"/app.slice/{UNIT_NAME}"):
        fail(f"unexpected service cgroup: {expected or '<unreadable>'}")

    root = Path("/sys/fs/cgroup") / expected.lstrip("/")
    for filename, wanted in EXPECTED_CONTROLLERS.items():
        try:
            actual = (root / filename).read_text().strip()
        except OSError as exc:
            fail(f"cannot read {filename}: {exc}")
        if actual != wanted:
            fail(f"{filename}={actual!r}, expected {wanted!r}")
    return root


def contained_pids(cgroup_root: Path) -> set[int]:
    """Read the whole service cgroup subtree membership from its cgroup.procs files."""
    cgroup_root = Path(cgroup_root)
    root_procs = cgroup_root / "cgroup.procs"
    try:
        texts = [root_procs.read_text()]
    except OSError as exc:
        fail(f"cannot read {root_procs}: {exc}")

    for nested in sorted(cgroup_root.rglob("cgroup.procs")):
        if nested == root_procs:
            continue
        try:
            texts.append(nested.read_text())
        except OSError:
            continue

    pids: set[int] = set()
    for text in texts:
        for token in text.split():
            if token.isdigit():
                pids.add(int(token))
    return pids


def escaped_pids(live: set[int], contained: set[int]) -> set[int]:
    """Run processes that are alive but no longer inside the service cgroup."""
    return set(live) - set(contained)


def should_scan_marker(poll_index: int) -> bool:
    return poll_index % MARKER_SCAN_EVERY == 0


def parse_proc_stat(text: str) -> tuple[int, int] | None:
    """Return ``(ppid, starttime)`` for a live task; zombies are already dead.

    A zombie retains a ``/proc/<pid>`` entry until its parent reaps it, but the
    kernel has already removed it from ``cgroup.procs``. Treating that short
    window as a cgroup escape makes successful one-shot Chrome runs fail after
    they have written their artifact.
    """
    fields = text[text.rfind(")") + 2 :].split()
    if len(fields) < 20 or fields[0] == "Z":
        return None
    return int(fields[1]), int(fields[19])


def proc_snapshot() -> dict[int, tuple[int, int]]:
    """Return live pid -> (ppid, starttime) from one /proc pass."""
    result: dict[int, tuple[int, int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            text = Path(entry.path, "stat").read_text()
            task = parse_proc_stat(text)
            if task is not None:
                result[pid] = task
        except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def marker_pids(marker: bytes) -> set[int]:
    """Find processes that inherited this run's unique environment marker."""
    tagged: set[int] = set()
    needle = RUN_MARKER_NAME.encode() + b"=" + marker
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            values = Path(entry.path, "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in values:
            tagged.add(int(entry.name))
    return tagged


def expand_descendants(
    snapshot: dict[int, tuple[int, int]],
    known: set[tuple[int, int]],
    roots: set[int],
) -> set[tuple[int, int]]:
    """Track descendants and retain already-seen PIDs without PID-reuse confusion."""
    live_known = {(pid, start) for pid, start in known if snapshot.get(pid, (0, -1))[1] == start}
    tracked_pids = {pid for pid, _ in live_known} | {pid for pid in roots if pid in snapshot}

    changed = True
    while changed:
        changed = False
        for pid, (ppid, start) in snapshot.items():
            if pid not in tracked_pids and ppid in tracked_pids:
                tracked_pids.add(pid)
                live_known.add((pid, start))
                changed = True
    for pid in roots:
        if pid in snapshot:
            live_known.add((pid, snapshot[pid][1]))
    return live_known


def collect_run_processes(
    marker: bytes,
    known: set[tuple[int, int]],
    root_key: tuple[int, int],
    scan_marker: bool = True,
) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    snapshot = proc_snapshot()
    tagged = marker_pids(marker) if scan_marker else set()
    root_pid, root_start = root_key
    roots = set(tagged)
    if snapshot.get(root_pid, (0, -1))[1] == root_start:
        roots.add(root_pid)
    known = expand_descendants(snapshot, known, roots)
    for pid in tagged:
        if pid in snapshot:
            known.add((pid, snapshot[pid][1]))
    return snapshot, known


def assert_contained(
    cgroup_root: Path,
    snapshot: dict[int, tuple[int, int]],
    known: set[tuple[int, int]],
) -> None:
    live = live_pids(snapshot, known)
    if not live:
        return

    escaped = escaped_pids(live, contained_pids(cgroup_root))
    if not escaped:
        return

    # A process that exited between the two reads left the membership list
    # without escaping it, so confirm the candidates are still alive.
    recheck = proc_snapshot()
    surviving = {
        pid
        for pid, start in known
        if pid in escaped and recheck.get(pid, (0, -1))[1] == start
    }
    if surviving:
        fail(
            "run processes escaped the service cgroup: "
            + ",".join(map(str, sorted(surviving)))
        )


def live_pids(
    snapshot: dict[int, tuple[int, int]], known: set[tuple[int, int]]
) -> set[int]:
    return {
        pid
        for pid, start in known
        if snapshot.get(pid, (0, -1))[1] == start
    }


def signal_pids(pids: set[int], sig: signal.Signals) -> None:
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, sig)
        except (PermissionError, ProcessLookupError):
            continue


def cleanup_processes(
    proc: subprocess.Popen[bytes],
    marker: bytes,
    known: set[tuple[int, int]],
    root_key: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[int]]:
    proc.poll()
    snapshot, known = collect_run_processes(marker, known, root_key)
    signal_pids(live_pids(snapshot, known), signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        proc.poll()
        snapshot, known = collect_run_processes(
            marker, known, root_key, scan_marker=False
        )
        if not live_pids(snapshot, known):
            return known, set()
        time.sleep(0.05)

    signal_pids(live_pids(snapshot, known), signal.SIGKILL)
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        proc.poll()
        snapshot, known = collect_run_processes(
            marker, known, root_key, scan_marker=False
        )
        remaining = live_pids(snapshot, known)
        if not remaining:
            return known, set()
        time.sleep(0.05)

    proc.poll()
    snapshot, known = collect_run_processes(marker, known, root_key)
    remaining = live_pids(snapshot, known)
    if remaining:
        print(
            "safe_browser_guard.py: unable to kill run processes: "
            + ",".join(map(str, sorted(remaining))),
            file=sys.stderr,
            flush=True,
        )
    return known, remaining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--browser", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument("--viewport", required=True)
    parser.add_argument("--virtual-time-budget", required=True)
    parser.add_argument("--url", required=True)
    return parser.parse_args()


def build_browser_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.browser,
        "--headless=new",
        "--disable-gpu",
        "--disable-crash-reporter",
        "--disable-breakpad",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-extensions",
        "--disable-sync",
        "--metrics-recording-only",
        "--mute-audio",
        "--run-all-compositor-stages-before-draw",
        f"--user-data-dir={args.profile}",
        f"--virtual-time-budget={args.virtual_time_budget}",
        f"--window-size={args.viewport}",
    ]
    if args.format == "pdf":
        command.extend(
            [
                "--no-pdf-header-footer",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={args.output}",
            ]
        )
    else:
        command.append(f"--screenshot={args.output}")
    command.append(args.url)
    return command


def main() -> int:
    args = parse_args()
    cgroup_root = verify_own_cgroup()
    marker = secrets.token_hex(16).encode()
    env = os.environ.copy()
    env[RUN_MARKER_NAME] = marker.decode()
    # Chrome can ask the user manager to relocate helpers into independent
    # scopes over the session bus. Make that bus deliberately unreachable so
    # every browser process remains a descendant of this transient service.
    env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/dev/null"
    env.pop("DBUS_STARTER_ADDRESS", None)
    env.pop("DBUS_STARTER_BUS_TYPE", None)

    command = build_browser_command(args)

    proc = subprocess.Popen(command, env=env, start_new_session=True)
    root_key = (proc.pid, -1)
    known: set[tuple[int, int]] = set()
    pin_deadline = time.monotonic() + 0.50
    while time.monotonic() < pin_deadline:
        snapshot = proc_snapshot()
        if proc.pid in snapshot:
            root_key = (proc.pid, snapshot[proc.pid][1])
            known.add(root_key)
            break
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    if root_key[1] < 0 and proc.poll() is None:
        proc.terminate()
        fail("could not pin the browser root PID and start time")

    stop_signal = 0

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        stop_signal = signum

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)

    try:
        poll_index = 0
        while proc.poll() is None and not stop_signal:
            snapshot, known = collect_run_processes(
                marker, known, root_key, scan_marker=should_scan_marker(poll_index)
            )
            assert_contained(cgroup_root, snapshot, known)
            poll_index += 1
            time.sleep(POLL_SECONDS)

        if stop_signal:
            known, survivors = cleanup_processes(proc, marker, known, root_key)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            if survivors:
                return 70
            return 128 + stop_signal

        status = proc.wait()
        deadline = time.monotonic() + POST_EXIT_GRACE_SECONDS
        while time.monotonic() < deadline:
            snapshot, known = collect_run_processes(
                marker, known, root_key, scan_marker=False
            )
            assert_contained(cgroup_root, snapshot, known)
            if not live_pids(snapshot, known):
                return status
            time.sleep(0.05)

        snapshot, known = collect_run_processes(marker, known, root_key)
        survivors = live_pids(snapshot, known)
        if survivors:
            print(
                "safe_browser_guard.py: descendants survived browser exit: "
                + ",".join(map(str, sorted(survivors))),
                file=sys.stderr,
                flush=True,
            )
            cleanup_processes(proc, marker, known, root_key)
            return 70
        return status
    except BaseException:
        cleanup_processes(proc, marker, known, root_key)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
