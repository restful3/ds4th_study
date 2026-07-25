"""책 10장 (업스트림 ch12, 그래프 피처 엔지니어링) 실행 태스크.

    python tasks.py --list
    python tasks.py init

업스트림 Makefile에는 init 만 있다. 본문 리스팅은 listings/ 에 책 번호 -2 로
들어 있다 (책 10.4 = listings/'12.4 Computing triangle metrics...py').
Python 리스팅은 직접 실행하고, Cypher 리스팅은 studykit.cypher 로 실행한다:

    from studykit import cypher
    cypher.listings()             # 목록
    print(cypher.read("12.15"))   # 원문
    cypher.run("12.15")           # Neo4j 실행
"""
from pathlib import Path

from studykit import PipInstall, main
from studykit.actions import run as _run

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
