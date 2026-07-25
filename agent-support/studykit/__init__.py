"""ds4th study 교재 학습자료 툴킷.

저장소에 한 벌만 두고 모든 교재가 공유한다. 교재별 설정은 각 교재 폴더의
study.toml 에 있고, 교재별 가상환경은 그 폴더의 .venv 다.

챕터 tasks.py 와 해설 노트북에서 쓰는 진입점:

    from studykit import PipInstall, RunScript, main    # 실행 태스크 선언
    from studykit import cypher                         # 책 리스팅 읽기·실행
    from studykit import config                         # 교재 경로

절차와 규칙은 agent-support/procedures/study-materials.md 와
agent-support/templates/study-materials/DESIGN.md 를 따른다.
"""
from studykit import config
from studykit.actions import (
    Download,
    Gunzip,
    Mkdir,
    Move,
    PipInstall,
    RunModule,
    RunScript,
    Streamlit,
    Unzip,
    main,
    run,
    study,
)
from studykit.config import REPO_ROOT, Study, StudyConfigError

__all__ = [
    "Download",
    "Gunzip",
    "Mkdir",
    "Move",
    "PipInstall",
    "REPO_ROOT",
    "RunModule",
    "RunScript",
    "Streamlit",
    "Study",
    "StudyConfigError",
    "Unzip",
    "config",
    "main",
    "run",
    "study",
]
