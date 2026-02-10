"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: First Agent - 실습 과제 1 해답
================================================================================

과제: 계산기 Agent 만들기
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. 기본 사칙연산 도구 4개
2. "123 + 456을 계산해줘" 같은 자연어 질문에 응답
3. 복잡한 수식도 처리 (예: "(10 + 5) * 2")

학습 목표:
- create_agent() 사용
- 자연어 → 도구 호출 자동화
- Agent의 추론 능력 활용

================================================================================
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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
def divide(a: float, b: float) -> float:
    """첫 번째 숫자를 두 번째 숫자로 나눕니다.

    Args:
        a: 분자
        b: 분모 (0이 아니어야 함)

    Returns:
        a / b

    Raises:
        ValueError: b가 0일 때
    """
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다")
    return a / b


# ============================================================================
# Agent 생성
# ============================================================================

def create_calculator_agent():
    """계산기 Agent 생성"""

    # 모델 설정
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 도구 목록
    tools = [add, subtract, multiply, divide]

    # System Prompt
    system_prompt = """당신은 수학 계산을 도와주는 전문 계산기 Assistant입니다.

사용자가 계산을 요청하면:
1. 수식을 분석합니다
2. 필요한 도구를 순서대로 호출합니다
3. 최종 결과를 명확하게 설명합니다

복잡한 수식은 단계별로 계산하세요.
예: (10 + 5) * 2
  → 먼저 10 + 5 = 15
  → 그 다음 15 * 2 = 30

항상 계산 과정을 간략히 설명해주세요."""

    # Agent 생성
    agent = create_react_agent(model, tools, state_modifier=system_prompt)

    return agent


# ============================================================================
# 테스트 함수
# ============================================================================

def test_calculator_agent():
    """계산기 Agent 테스트"""
    print("=" * 70)
    print("🧮 계산기 Agent 테스트")
    print("=" * 70)

    agent = create_calculator_agent()

    # 테스트 케이스
    test_cases = [
        "123 더하기 456을 계산해줘",
        "1000에서 234를 빼면 얼마야?",
        "12 곱하기 8은?",
        "144를 12로 나누면?",
        "(10 + 5) 곱하기 2를 계산해줘",  # 복잡한 수식
        "100을 4로 나누고, 그 결과에 3을 곱해줘",  # 다단계
        "50을 0으로 나눠줘",  # 에러 케이스
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 테스트 {i}: {question}")
        print('=' * 70)

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})

            # 최종 응답 출력
            final_message = result["messages"][-1]
            print(f"\n🤖 Agent 응답:\n{final_message.content}\n")

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")

    print("=" * 70)
    print("✅ 모든 테스트 완료!")
    print("=" * 70)


def interactive_calculator():
    """대화형 계산기"""
    print("\n" + "=" * 70)
    print("🎮 대화형 계산기 Agent")
    print("=" * 70)
    print("계산 질문을 입력하세요 (종료: 'quit', 'exit', 'q')")
    print("=" * 70)

    agent = create_calculator_agent()

    while True:
        try:
            user_input = input("\n👤 질문: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q', '종료']:
                print("\n👋 계산기를 종료합니다.")
                break

            if not user_input:
                continue

            # Agent 실행
            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

            # 응답 출력
            final_message = result["messages"][-1]
            print(f"\n🤖 Agent: {final_message.content}")

        except KeyboardInterrupt:
            print("\n\n👋 계산기를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🧮 Part 3: 계산기 Agent - 실습 과제 1 해답")
    print("=" * 70)

    try:
        # 자동 테스트
        test_calculator_agent()

        # 대화형 모드 (선택)
        print("\n대화형 모드를 실행하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예']:
            interactive_calculator()

    except Exception as e:
        print(f"\n⚠️  테스트 실패: {e}")
        print("(API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("💡 추가 학습 포인트:")
    print("  1. 더 많은 수학 함수: power, sqrt, log")
    print("  2. 수식 파싱 개선: eval 대신 ast 사용")
    print("  3. 계산 히스토리 저장 (메모리)")
    print("  4. 그래프 그리기 도구 추가")
    print("=" * 70)


if __name__ == "__main__":
    main()
