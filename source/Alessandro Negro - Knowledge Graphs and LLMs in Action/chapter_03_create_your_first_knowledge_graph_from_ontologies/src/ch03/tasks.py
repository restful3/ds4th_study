"""책 3장 (업스트림 ch03) 실행 태스크. 업스트림 Makefile의 Python 대체.

    python tasks.py --list
    python tasks.py init
    python tasks.py import

사전 조건: Neo4j 실행 + Neosemantics(n10s) 플러그인 설치. Readme.md 참고.
"""
from pathlib import Path

from studykit import PipInstall, RunScript, main
from studykit.actions import run as _run

HERE = Path(__file__).resolve().parent

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
    "import": [
        RunScript("importer/import_hpo.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
