"""책 11장 (업스트림 ch13, 표현학습 · GNN) 실행 태스크.

    python tasks.py --list
    python tasks.py init
    python tasks.py node2vec      # 책 11.1: Karate Club 네트워크 node2vec 임베딩
    python tasks.py gnn           # GNN 통합 예제
"""
from pathlib import Path

from kgbook import PipInstall, RunScript, main
from kgbook.actions import run as _run

HERE = Path(__file__).resolve().parent

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
    "node2vec": [
        RunScript(
            "listings/13.1 Applying node2vec embeddings to Karate Club’s Network.py"
        ),
    ],
    "gnn": [
        RunScript("listings/GNN_all_in_one.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("node2vec")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
