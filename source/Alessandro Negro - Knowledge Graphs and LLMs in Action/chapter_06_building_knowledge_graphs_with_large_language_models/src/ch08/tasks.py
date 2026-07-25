"""책 6장 (업스트림 ch08, LLM으로 KG 구축) 실행 태스크.

    python tasks.py --list
    python tasks.py init
    python tasks.py import

import 는 OpenAI 호환 API를 호출한다. 업스트림 Makefile이 하드코딩했던
자리표시자 대신 환경변수를 쓴다:

    export OPENAI_KEY=sk-...
    export OPENAI_BASE_URL=https://api.openai.com/v1   # 기본값
    export OPENAI_MODEL=gpt-4o-mini                     # 기본값

교재 루트 data/cache_llm/ 에 응답 캐시가 동봉되어 있어, 캐시가 적용되는
범위에서는 API 호출 없이도 재현된다.
"""
from pathlib import Path

from studykit import PipInstall, RunScript, main
from studykit.actions import run as _run

HERE = Path(__file__).resolve().parent

OPENAI_ENV = {
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_MODEL": "gpt-4o-mini",
}

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
    "import": [
        RunScript("importer/ingest_and_process.py", env=OPENAI_ENV),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
