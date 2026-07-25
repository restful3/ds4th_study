"""책 15장 (업스트림 ch17, LangGraph QA 에이전트) 실행 태스크.

    python tasks.py --list
    python tasks.py init
    python tasks.py import        # chicago.ila Neo4j 백업 시드 적재
    python tasks.py app           # Streamlit 앱 실행

필요한 환경변수 (업스트림 Makefile 주석 기준):
    NEO4J_URI=neo4j://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password
    NEO4J_DATABASE=chicago.ila
    OPENAI_API_KEY=...
    # 또는 Azure: AZURE_OPENAI_API_KEY / _ENDPOINT / _DEPLOYMENT / _API_VERSION

이 값들이 없으면 교재 루트 config.ini 의 [neo4j] 설정이 쓰인다.
"""
from pathlib import Path

from studykit import PipInstall, RunScript, Streamlit, main
from studykit.actions import run as _run

HERE = Path(__file__).resolve().parent

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
    "import": [
        RunScript("importer/import_seed.py"),
    ],
    "app": [
        Streamlit("app.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
