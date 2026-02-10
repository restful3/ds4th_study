"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: Deployment - 실습 과제 1 해답
================================================================================

과제: 테스트 스위트 구축
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. Unit 테스트: 개별 컴포넌트
2. Integration 테스트: Agent 통합
3. End-to-End 테스트: 전체 플로우

학습 목표:
- pytest 활용
- Mock 객체 사용
- 테스트 자동화

================================================================================
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ============================================================================
# 테스트 대상: 간단한 Agent 시스템
# ============================================================================

@tool
def calculator(expression: str) -> float:
    """수식을 계산합니다."""
    try:
        return eval(expression)
    except:
        return "계산 오류"

@tool
def get_weather(city: str) -> str:
    """날씨 정보를 조회합니다."""
    # 실제로는 API 호출
    weather_data = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C"
    }
    return weather_data.get(city, "정보 없음")

def create_test_agent():
    """테스트용 Agent 생성"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, get_weather]
    return create_react_agent(model, tools)

# ============================================================================
# Unit 테스트
# ============================================================================

class TestTools:
    """도구 Unit 테스트"""
    
    def test_calculator_addition(self):
        """계산기: 덧셈"""
        result = calculator.invoke({"expression": "2 + 2"})
        assert result == 4
    
    def test_calculator_complex(self):
        """계산기: 복잡한 수식"""
        result = calculator.invoke({"expression": "(10 + 5) * 2"})
        assert result == 30
    
    def test_calculator_error(self):
        """계산기: 에러 처리"""
        result = calculator.invoke({"expression": "invalid"})
        assert result == "계산 오류"
    
    def test_get_weather_seoul(self):
        """날씨: 서울"""
        result = get_weather.invoke({"city": "서울"})
        assert "맑음" in result
        assert "15°C" in result
    
    def test_get_weather_unknown_city(self):
        """날씨: 알 수 없는 도시"""
        result = get_weather.invoke({"city": "화성"})
        assert result == "정보 없음"

# ============================================================================
# Integration 테스트
# ============================================================================

class TestAgentIntegration:
    """Agent 통합 테스트"""
    
    @pytest.fixture
    def agent(self):
        """Agent fixture"""
        return create_test_agent()
    
    def test_simple_calculation(self, agent):
        """간단한 계산 요청"""
        result = agent.invoke({
            "messages": [HumanMessage(content="5 곱하기 3은?")]
        })
        
        # 응답 확인
        assert result["messages"]
        # 계산 결과가 포함되어 있어야 함
        response = str(result["messages"][-1].content)
        assert "15" in response
    
    @patch('langchain_openai.ChatOpenAI')
    def test_agent_with_mock_llm(self, mock_llm):
        """Mock LLM으로 Agent 테스트"""
        # Mock 설정
        mock_response = AIMessage(content="계산 결과는 15입니다")
        mock_llm.return_value.invoke.return_value = mock_response
        
        # Agent 생성 (Mock LLM 사용)
        agent = create_react_agent(mock_llm.return_value, [calculator])
        
        # 실행
        result = agent.invoke({
            "messages": [HumanMessage(content="5 * 3")]
        })
        
        # Mock 호출 확인
        assert mock_llm.return_value.invoke.called

# ============================================================================
# End-to-End 테스트
# ============================================================================

class TestEndToEnd:
    """E2E 테스트"""
    
    @pytest.mark.e2e
    def test_full_conversation_flow(self):
        """전체 대화 플로우"""
        agent = create_test_agent()
        
        # 대화 시퀀스
        conversations = [
            ("10 더하기 5는?", "15"),
            ("서울 날씨는?", "맑음"),
        ]
        
        for question, expected_keyword in conversations:
            result = agent.invoke({
                "messages": [HumanMessage(content=question)]
            })
            
            response = str(result["messages"][-1].content)
            assert expected_keyword in response

# ============================================================================
# Performance 테스트
# ============================================================================

class TestPerformance:
    """성능 테스트"""
    
    @pytest.mark.performance
    def test_response_time(self):
        """응답 시간 테스트"""
        import time
        
        agent = create_test_agent()
        
        start = time.time()
        result = agent.invoke({
            "messages": [HumanMessage(content="2 + 2")]
        })
        elapsed = time.time() - start
        
        # 5초 이내 응답
        assert elapsed < 5.0, f"Too slow: {elapsed}s"
    
    @pytest.mark.performance
    def test_concurrent_requests(self):
        """동시 요청 처리"""
        import concurrent.futures
        
        agent = create_test_agent()
        
        def make_request(question):
            return agent.invoke({"messages": [HumanMessage(content=question)]})
        
        questions = [f"{i} + {i}" for i in range(5)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, q) for q in questions]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 5

# ============================================================================
# 테스트 실행 및 리포트
# ============================================================================

def run_tests():
    """테스트 실행"""
    print("=" * 70)
    print("🧪 테스트 스위트 실행")
    print("=" * 70)
    
    # pytest 옵션
    pytest_args = [
        __file__,
        "-v",  # verbose
        "-s",  # print output
        "--tb=short",  # short traceback
        "-m", "not performance",  # skip performance tests
    ]
    
    # pytest 실행
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ 모든 테스트 통과!")
    else:
        print("❌ 일부 테스트 실패")
    print("=" * 70)
    
    return exit_code

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🧪 Part 10: 테스트 스위트 구축 - 실습 과제 1 해답")
    print("=" * 70)
    
    print("""
테스트 구조:

1. Unit 테스트 (TestTools)
   - 개별 도구 테스트
   - 독립적 실행
   - 빠른 피드백

2. Integration 테스트 (TestAgentIntegration)
   - Agent 통합 테스트
   - Mock 활용
   - 컴포넌트 간 상호작용

3. E2E 테스트 (TestEndToEnd)
   - 전체 플로우 테스트
   - 실제 사용 시나리오
   - 프로덕션 환경 시뮬레이션

4. Performance 테스트 (TestPerformance)
   - 응답 시간
   - 동시성
   - 부하 테스트

실행 방법:
  pytest exercise_01.py -v
  pytest exercise_01.py -m performance  # 성능 테스트만
  pytest exercise_01.py --cov  # 커버리지 포함
    """)
    
    print("\n💡 학습 포인트:")
    print("  1. pytest 프레임워크 활용")
    print("  2. Mock 객체로 외부 의존성 제거")
    print("  3. 다양한 테스트 레벨")
    print("  4. 자동화 테스트 파이프라인")
    
    print("\n" + "=" * 70)
    print("테스트를 실행하려면 'pytest exercise_01.py -v' 명령을 사용하세요")
    print("=" * 70)

if __name__ == "__main__":
    main()
