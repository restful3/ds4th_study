"""
Research Agent 기본 테스트
"""

import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_main():
    """main 모듈 임포트 테스트"""
    import main
    assert hasattr(main, "main"), "main() 함수가 존재해야 합니다"


def test_import_multi_agent_system():
    """multi_agent_system 모듈 임포트 테스트"""
    import multi_agent_system
    assert multi_agent_system is not None


def test_agents_directory():
    """agents 디렉토리 구조 확인"""
    agents_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agents"
    )
    assert os.path.isdir(agents_dir), "agents/ 디렉토리가 존재해야 합니다"


if __name__ == "__main__":
    test_import_main()
    test_import_multi_agent_system()
    test_agents_directory()
    print("모든 기본 테스트 통과!")
