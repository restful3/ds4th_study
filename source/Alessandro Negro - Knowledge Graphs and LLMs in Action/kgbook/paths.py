"""교재 루트 기준 공용 경로.

kgbook 패키지가 <교재 루트>/kgbook/ 에 있으므로 부모의 부모가 교재 루트다.
.venv/lib/python3.10/site-packages/_kgbook_root.pth 가 교재 루트를 sys.path에
넣어주므로, 어느 디렉터리에서 실행해도 import kgbook / import util 이 동작한다.
"""
import os
import re
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parent.parent

#: make import 으로 내려받는 대용량 데이터셋 (git 추적 제외)
DATASET = BOOK_ROOT / "dataset"

#: 저장소에 동봉된 데이터 (bbc 코퍼스, LLM 응답 캐시)
DATA = BOOK_ROOT / "data"

#: Neo4j 접속정보. util.graphdb_base 가 util/../config.ini 로 읽는 파일과 동일하다.
CONFIG = BOOK_ROOT / "config.ini"

VENV = BOOK_ROOT / ".venv"


def venv_python() -> Path:
    """교재 전용 .venv의 python. 이미 그 안에서 돌고 있으면 현재 인터프리터."""
    candidate = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def venv_bin(name: str) -> Path:
    """.venv 안의 실행파일 경로 (streamlit 등)."""
    bindir = VENV / ("Scripts" if os.name == "nt" else "bin")
    candidate = bindir / (f"{name}.exe" if os.name == "nt" else name)
    return candidate if candidate.exists() else Path(name)


def normalize_path(raw: str) -> str:
    """'../../dataset/x/' -> 'dataset/x'. 교재 루트 기준 상대경로로 통일."""
    return re.sub(r"^(\.\./)+", "", raw).rstrip("/")


def resolve(raw: str) -> Path:
    """교재 루트 기준 상대경로를 절대경로로."""
    return BOOK_ROOT / normalize_path(raw)
