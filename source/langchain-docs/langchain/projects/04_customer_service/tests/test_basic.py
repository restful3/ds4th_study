"""
Customer Service Agent 기본 테스트
"""

import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_main():
    """main 모듈 임포트 테스트"""
    import main
    assert hasattr(main, "main"), "main() 함수가 존재해야 합니다"


def test_import_config():
    """config 모듈 임포트 테스트"""
    import config
    assert config is not None


def test_agents_package():
    """agents 패키지 임포트 테스트"""
    from agents import BaseAgent
    assert BaseAgent is not None, "BaseAgent 클래스가 존재해야 합니다"


def test_router_agent():
    """router 에이전트 임포트 테스트"""
    from agents.router import RouterAgent
    assert RouterAgent is not None


def test_middleware_directory():
    """middleware 디렉토리 구조 확인"""
    middleware_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "middleware"
    )
    assert os.path.isdir(middleware_dir), "middleware/ 디렉토리가 존재해야 합니다"


if __name__ == "__main__":
    test_import_main()
    test_import_config()
    test_agents_package()
    test_router_agent()
    test_middleware_directory()
    print("모든 기본 테스트 통과!")
