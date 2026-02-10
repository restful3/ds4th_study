"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: Fundamentals - 실습 과제 1 해답
================================================================================

과제: 계산기 도구 만들기
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. add, subtract, multiply, divide 4개 도구 구현
2. 각 도구는 두 개의 숫자를 받아 연산 수행
3. divide는 0으로 나누기 방지

학습 목표:
- @tool 데코레이터 사용법
- 타입 힌트와 docstring 작성
- 에러 핸들링

================================================================================
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def add(a: float, b: float) -> float:
    """두 숫자를 더합니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        a + b의 결과
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """첫 번째 숫자에서 두 번째 숫자를 뺍니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        a - b의 결과
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """두 숫자를 곱합니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        a * b의 결과
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """첫 번째 숫자를 두 번째 숫자로 나눕니다.

    Args:
        a: 분자 (나눠지는 수)
        b: 분모 (나누는 수)

    Returns:
        a / b의 결과

    Raises:
        ValueError: b가 0일 때
    """
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다")
    return a / b


# ============================================================================
# 테스트 코드
# ============================================================================

def test_calculator_tools():
    """도구 단위 테스트"""
    print("=" * 70)
    print("🧪 계산기 도구 테스트")
    print("=" * 70)

    # Add 테스트
    result = add.invoke({"a": 10, "b": 5})
    print(f"\n✅ add(10, 5) = {result}")
    assert result == 15, "Add 테스트 실패"

    # Subtract 테스트
    result = subtract.invoke({"a": 10, "b": 5})
    print(f"✅ subtract(10, 5) = {result}")
    assert result == 5, "Subtract 테스트 실패"

    # Multiply 테스트
    result = multiply.invoke({"a": 10, "b": 5})
    print(f"✅ multiply(10, 5) = {result}")
    assert result == 50, "Multiply 테스트 실패"

    # Divide 테스트
    result = divide.invoke({"a": 10, "b": 5})
    print(f"✅ divide(10, 5) = {result}")
    assert result == 2.0, "Divide 테스트 실패"

    # 0으로 나누기 테스트
    try:
        divide.invoke({"a": 10, "b": 0})
        print("❌ 0으로 나누기 에러 처리 실패")
    except ValueError as e:
        print(f"✅ divide(10, 0) → ValueError: {e}")

    print("\n" + "=" * 70)
    print("✅ 모든 테스트 통과!")
    print("=" * 70)


def test_with_llm():
    """LLM과 통합 테스트"""
    print("\n" + "=" * 70)
    print("🤖 LLM 통합 테스트")
    print("=" * 70)

    # 모델 생성
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 도구 바인딩
    tools = [add, subtract, multiply, divide]
    model_with_tools = model.bind_tools(tools)

    # 테스트 질문
    test_cases = [
        "123 더하기 456은 얼마인가요?",
        "1000에서 234를 빼면?",
        "12 곱하기 8은?",
        "144를 12로 나누면?",
    ]

    for question in test_cases:
        print(f"\n👤 질문: {question}")
        response = model_with_tools.invoke([{"role": "user", "content": question}])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 도구 호출: {tool_call['name']}({tool_call['args']})")

            # 도구 실행
            tool_map = {t.name: t for t in tools}
            tool = tool_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            print(f"📊 결과: {result}")
        else:
            print(f"🤖 응답: {response.content}")

    print("\n" + "=" * 70)
    print("✅ LLM 통합 테스트 완료!")
    print("=" * 70)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📐 Part 2: 계산기 도구 - 실습 과제 1 해답")
    print("=" * 70)

    # 1. 도구 단위 테스트
    test_calculator_tools()

    # 2. LLM 통합 테스트
    try:
        test_with_llm()
    except Exception as e:
        print(f"\n⚠️  LLM 통합 테스트 스킵: {e}")
        print("(API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("💡 추가 학습 포인트:")
    print("  1. 더 많은 수학 함수 추가해보세요 (power, sqrt, abs)")
    print("  2. 복잡한 수식 파싱 (예: \"(10 + 5) * 2\")")
    print("  3. 계산 히스토리 저장 기능")
    print("  4. 단위 변환 기능 (km <-> miles)")
    print("=" * 70)


if __name__ == "__main__":
    main()
