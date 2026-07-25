"""책 4장 전반부 (업스트림 ch04, biomedical/PPI 예제) 실행 태스크.

    python tasks.py --list
    python tasks.py import
    python tasks.py analysis

주의 — 업스트림 Makefile의 끊긴 참조를 그대로 옮겼다:
  * init 의 requirements.lock 이 업스트림 ch04 에 없다. 4장은 meap-only/meap-ch05 의
    requirements.lock 을 쓰거나 필요한 패키지를 직접 설치해야 한다.
  * analysis 의 analysis/pharma_analysis.py 가 업스트림에 없다
    (실제로 있는 파일은 multiomic_analysis.py, louvain_cluster_analysis.py).
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
        RunScript("importer/import_seed.py"),
    ],
    "analysis": [
        RunScript("analysis/louvain_cluster_analysis.py"),
        RunScript("analysis/pharma_analysis.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
