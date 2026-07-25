"""책 9장 (업스트림 ch11, Graph ML 입문) 실행 태스크.

    python tasks.py --list
    python tasks.py init

업스트림 Makefile에는 init 만 있다. 예제 스크립트는 직접 실행한다:

    python analysis/simple_classification_example.py
    python analysis/simple_clustering_example.py
"""
from pathlib import Path

from kgbook import PipInstall, main
from kgbook.actions import run as _run

HERE = Path(__file__).resolve().parent

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("init")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
