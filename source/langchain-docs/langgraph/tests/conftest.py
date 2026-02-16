"""
pytest 설정 및 공통 fixture

이 파일은 모든 테스트에서 사용할 수 있는 공통 설정과 fixture를 제공합니다.
"""

import os
import sys
import pytest
from typing import Generator

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_state() -> dict:
    """기본 State fixture"""
    return {
        "message": "테스트 메시지",
        "count": 0
    }


@pytest.fixture
def sample_messages() -> list:
    """메시지 리스트 fixture"""
    from langchain_core.messages import HumanMessage, AIMessage

    return [
        HumanMessage(content="안녕하세요"),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
    ]


@pytest.fixture
def memory_checkpointer():
    """MemorySaver fixture"""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


@pytest.fixture
def thread_config() -> dict:
    """Thread 설정 fixture"""
    return {"configurable": {"thread_id": "test_thread_1"}}


@pytest.fixture(autouse=True)
def reset_env():
    """각 테스트 전후로 환경 초기화"""
    # 테스트 전 설정
    original_env = os.environ.copy()

    yield

    # 테스트 후 원래 환경으로 복원
    os.environ.clear()
    os.environ.update(original_env)


def pytest_configure(config):
    """pytest 설정"""
    # 커스텀 마커 등록
    config.addinivalue_line(
        "markers", "slow: 느린 테스트 (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: 통합 테스트 (외부 API 필요)"
    )
    config.addinivalue_line(
        "markers", "llm: LLM API 호출 테스트"
    )
