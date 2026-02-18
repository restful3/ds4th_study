"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: First Agent - 실습 과제 1 해답
================================================================================

과제: 계산기 Agent 만들기
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. add(a, b), subtract(a, b), multiply(a, b), divide(a, b) 도구 구현
2. "5 더하기 3을 한 다음, 그 결과에 2를 곱해줘" 같은 복잡한 계산 처리
3. System Prompt로 Agent를 "친절한 수학 선생님" 페르소나로 설정
4. 0으로 나누기 같은 에러를 적절히 처리

학습 목표:
- create_agent() 사용
- @tool 데코레이터로 도구 정의
- Agent의 순차 도구 호출 (ReAct 패턴)
- 에러 핸들링

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


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
        a + b
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """첫 번째 숫자에서 두 번째 숫자를 뺍니다.

    Args:
        a: 첫 번째 숫자 (피감수)
        b: 두 번째 숫자 (감수)

    Returns:
        a - b
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """두 숫자를 곱합니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        a * b
    """
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """첫 번째 숫자를 두 번째 숫자로 나눕니다.

    Args:
        a: 분자
        b: 분모 (0이 아니어야 함)

    Returns:
        a / b 또는 에러 메시지
    """
    if b == 0:
        return "오류: 0으로 나눌 수 없습니다. 다른 값을 입력해 주세요."
    return str(a / b)


# ============================================================================
# Agent 생성
# ============================================================================

def create_calculator_agent():
    """계산기 Agent 생성"""

    # 모델 설정
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 도구 목록
    tools = [add, subtract, multiply, divide]

    # System Prompt - "친절한 수학 선생님" 페르소나
    system_prompt = """당신은 친절한 수학 선생님입니다.

학생들이 수학 계산을 요청하면 도구를 사용하여 정확히 계산해 주세요.

계산 원칙:
- 항상 도구를 사용하여 정확한 결과를 제공합니다
- 복잡한 수식은 단계별로 나누어 계산합니다
  예: (10 + 5) * 2 -> 먼저 add(10, 5) = 15, 그 다음 multiply(15, 2) = 30
- 각 단계를 학생이 이해할 수 있도록 간략히 설명합니다
- 0으로 나누기 같은 에러가 발생하면 왜 불가능한지 쉽게 설명합니다
- 계산 과정을 단계별로 설명하세요

말투:
- 친근하고 격려하는 톤을 사용합니다"""

    # Agent 생성
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


# ============================================================================
# 테스트 함수
# ============================================================================

def test_calculator_agent():
    """계산기 Agent 테스트"""
    print("=" * 70)
    print("  계산기 Agent 테스트")
    print("=" * 70)

    agent = create_calculator_agent()

    # 테스트 케이스
    test_cases = [
        "123 더하기 456을 계산해줘",
        "1000에서 234를 빼면 얼마야?",
        "12 곱하기 8은?",
        "144를 12로 나누면?",
        "5 더하기 3을 한 다음, 그 결과에 2를 곱해줘",  # 복잡한 수식
        "100을 4로 나누고, 그 결과에 3을 곱해줘",  # 다단계
        "50을 0으로 나눠줘",  # 에러 케이스
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"  테스트 {i}: {question}")
        print("=" * 70)

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            final_message = result["messages"][-1]
            print(f"\n  Agent 응답:\n{final_message.content}\n")

        except Exception as e:
            print(f"\n  오류 발생: {e}\n")

    print("=" * 70)
    print("  모든 테스트 완료!")
    print("=" * 70)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("  Part 3: 계산기 Agent - 실습 과제 1 해답")
    print("=" * 70)

    try:
        test_calculator_agent()
    except Exception as e:
        print(f"\n  테스트 실패: {e}")
        print("(API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    print("\n" + "=" * 70)
    print("  추가 학습 포인트:")
    print("  1. 더 많은 수학 함수 추가: power, sqrt, log")
    print("  2. 수식 파싱 개선: eval 대신 ast 사용")
    print("  3. 계산 히스토리 저장 (메모리)")
    print("  4. 그래프 그리기 도구 추가")
    print("=" * 70)


if __name__ == "__main__":
    main()
