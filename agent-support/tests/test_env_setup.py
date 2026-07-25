"""교재 실행 환경이 올바르게 구성되었는지 검증.

setup_env.py 가 만들어야 하는 상태를 그대로 확인한다. 새 환경에서
`python setup_env.py` 를 돌린 뒤 이 테스트가 통과하면 재현에 성공한 것이다.

실행:
    .venv/bin/python tests/test_env_setup.py
"""
import configparser
import os
import subprocess
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parent.parent

# 업스트림이 테스트한 파이썬 버전 (README: 3.8/3.9/3.10)
SUPPORTED_MINORS = {8, 9, 10}


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "✓" if condition else "✗"
    print(f"  {mark} {label}" + (f" — {detail}" if detail else ""))
    return condition


def venv_python() -> Path:
    return BOOK_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    results = []

    print("[1] 부트스트랩 스크립트")
    results.append(check(
        "setup_env.py 존재", (BOOK_ROOT / "setup_env.py").exists()
    ))

    print("[2] 가상환경")
    py = venv_python()
    results.append(check(".venv 인터프리터 존재", py.exists(), str(py.relative_to(BOOK_ROOT))))
    if py.exists():
        version = subprocess.run(
            [str(py), "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        minor = int(version.split(".")[1])
        results.append(check(
            f"파이썬 버전이 업스트림 지원 범위", minor in SUPPORTED_MINORS,
            f"{version} (지원: 3.8/3.9/3.10)",
        ))
        results.append(check(
            "pip 사용 가능",
            subprocess.run([str(py), "-m", "pip", "--version"],
                           capture_output=True).returncode == 0,
        ))

    print("[3] 공용 자원")
    results.append(check("util/ 패키지", (BOOK_ROOT / "util" / "graphdb_base.py").exists()))
    results.append(check("config.ini", (BOOK_ROOT / "config.ini").exists()))
    results.append(check("data/bbc 코퍼스", (BOOK_ROOT / "data" / "bbc").is_dir()))
    results.append(check("dataset/ + 자체 .gitignore",
                         (BOOK_ROOT / "dataset" / ".gitignore").exists()))

    if (BOOK_ROOT / "config.ini").exists():
        parser = configparser.ConfigParser()
        parser.read(BOOK_ROOT / "config.ini")
        results.append(check("config.ini 에 [neo4j] 섹션", parser.has_section("neo4j")))

    print("[4] 깊은 cwd 에서의 임포트 (PYTHONPATH 없이)")
    if py.exists():
        deep = next(BOOK_ROOT.glob("chapter_*/src/ch*"), None)
        results.append(check("챕터 src 폴더 존재", deep is not None))
        if deep is not None:
            probe = (
                "import os, util, util.graphdb_base as g, kgbook;"
                "cfg = os.path.normpath(os.path.join(os.path.dirname(g.__file__),"
                "'..','config.ini'));"
                "assert os.path.exists(cfg), cfg;"
                f"assert str(kgbook.BOOK_ROOT) == {str(BOOK_ROOT)!r}, kgbook.BOOK_ROOT;"
                "assert kgbook.CONFIG.exists(), kgbook.CONFIG;"
                "print('ok')"
            )
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            proc = subprocess.run([str(py), "-c", probe], cwd=str(deep),
                                  capture_output=True, text=True, env=env)
            results.append(check(
                "util + kgbook 임포트 및 config.ini 해석",
                proc.returncode == 0 and proc.stdout.strip() == "ok",
                (proc.stderr.strip().splitlines() or [""])[-1],
            ))

    print("[5] 챕터 태스크 러너")
    if py.exists():
        deep = next(BOOK_ROOT.glob("chapter_*/src/ch*/tasks.py"), None)
        results.append(check("tasks.py 존재", deep is not None))
        if deep is not None:
            proc = subprocess.run([str(py), "tasks.py", "--list"],
                                  cwd=str(deep.parent), capture_output=True, text=True)
            results.append(check("tasks.py --list 동작", proc.returncode == 0))

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} 통과")
    if failed:
        print(f"실패 {failed}건 — `python setup_env.py` 를 실행하라.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
