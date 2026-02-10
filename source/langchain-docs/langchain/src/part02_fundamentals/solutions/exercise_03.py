"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: Fundamentals - 실습 과제 3 해답
================================================================================

과제: Tool Calling 시스템
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. 3개 이상의 도구 정의
2. LLM에게 도구 바인딩
3. Tool Call 감지 및 실행
4. 최종 응답 생성

학습 목표:
- 완전한 Tool Calling 루프 구현
- 도구 실행 자동화
- Agent 기초 이해

================================================================================
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List
import json

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def search_web(query: str) -> str:
    """인터넷에서 정보를 검색합니다.

    Args:
        query: 검색할 키워드

    Returns:
        검색 결과 요약
    """
    # Mock 검색 결과
    mock_results = {
        "langchain": "LangChain은 LLM 애플리케이션 개발 프레임워크입니다. 2022년 출시되어 빠르게 성장했습니다.",
        "python": "Python은 1991년 귀도 반 로섬이 만든 고급 프로그래밍 언어입니다.",
        "ai": "인공지능(AI)은 기계가 인간의 지능을 모방하도록 하는 기술입니다.",
    }

    # 키워드 매칭
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return f"🔍 검색 결과: {value}"

    return f"🔍 '{query}'에 대한 검색 결과를 찾지 못했습니다."


@tool
def calculate(expression: str) -> str:
    """수식을 계산합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 2", "10 * 5")

    Returns:
        계산 결과
    """
    try:
        # 안전한 eval 대신 간단한 파싱
        # 실제로는 ast.literal_eval이나 전용 파서 사용 권장
        result = eval(expression, {"__builtins__": {}}, {})
        return f"📊 계산 결과: {expression} = {result}"
    except Exception as e:
        return f"❌ 계산 오류: {str(e)}"


@tool
def get_current_time() -> str:
    """현재 시간을 조회합니다.

    Returns:
        현재 날짜와 시간
    """
    from datetime import datetime
    now = datetime.now()
    return f"🕐 현재 시간: {now.strftime('%Y년 %m월 %d일 %H:%M:%S')}"


@tool
def translate_text(text: str, target_language: str = "English") -> str:
    """텍스트를 다른 언어로 번역합니다.

    Args:
        text: 번역할 텍스트
        target_language: 목표 언어

    Returns:
        번역 결과
    """
    # Mock 번역 (실제로는 번역 API 사용)
    translations = {
        "안녕하세요": {"English": "Hello", "Japanese": "こんにちは"},
        "감사합니다": {"English": "Thank you", "Japanese": "ありがとう"},
        "좋은 하루": {"English": "Have a nice day", "Japanese": "良い一日を"},
    }

    if text in translations and target_language in translations[text]:
        return f"🌐 번역: {translations[text][target_language]}"
    else:
        return f"🌐 번역: ('{text}'를 {target_language}로 번역)"


# ============================================================================
# Tool Calling 루프 구현
# ============================================================================

class SimpleAgent:
    """간단한 Tool Calling Agent"""

    def __init__(self, model, tools: List):
        self.model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.model_with_tools = model.bind_tools(tools)

    def run(self, user_input: str, max_iterations: int = 5) -> str:
        """Agent 실행

        Args:
            user_input: 사용자 입력
            max_iterations: 최대 반복 횟수

        Returns:
            최종 응답
        """
        messages = [HumanMessage(content=user_input)]

        print(f"\n👤 사용자: {user_input}\n")

        for i in range(max_iterations):
            print(f"🔄 Iteration {i + 1}")
            print("-" * 60)

            # LLM 호출
            response = self.model_with_tools.invoke(messages)
            messages.append(response)

            # Tool Call 확인
            if not response.tool_calls:
                # 최종 응답
                print(f"✅ 최종 응답 생성\n")
                return response.content

            # Tool 실행
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"🔧 도구 호출: {tool_name}")
                print(f"   파라미터: {json.dumps(tool_args, ensure_ascii=False)}")

                # 도구 실행
                tool = self.tool_map[tool_name]
                try:
                    result = tool.invoke(tool_args)
                    print(f"   결과: {result}\n")

                    # Tool Message 추가
                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        )
                    )
                except Exception as e:
                    error_msg = f"도구 실행 오류: {str(e)}"
                    print(f"   ❌ {error_msg}\n")
                    messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call["id"]
                        )
                    )

        return "최대 반복 횟수에 도달했습니다."


# ============================================================================
# 테스트 코드
# ============================================================================

def test_simple_agent():
    """Agent 테스트"""
    print("=" * 70)
    print("🤖 Simple Tool Calling Agent 테스트")
    print("=" * 70)

    # 모델 및 도구 설정
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_web, calculate, get_current_time, translate_text]

    # Agent 생성
    agent = SimpleAgent(model, tools)

    # 테스트 케이스
    test_cases = [
        "LangChain에 대해 검색해줘",
        "25 곱하기 4는 얼마야?",
        "지금 몇 시야?",
        "'안녕하세요'를 영어로 번역해줘",
        "LangChain을 검색하고, 그 정보를 요약해줘",  # 멀티스텝
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 테스트 케이스 {i}")
        print('=' * 70)

        result = agent.run(question)

        print(f"🤖 Agent 최종 응답:")
        print(f"{result}")
        print()

    print("=" * 70)
    print("✅ 모든 테스트 완료!")
    print("=" * 70)


def test_manual_tool_calling():
    """수동 Tool Calling 예시"""
    print("\n" + "=" * 70)
    print("📚 수동 Tool Calling 프로세스")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_web, calculate]
    model_with_tools = model.bind_tools(tools)

    # Step 1: 사용자 입력
    user_input = "Python에 대해 검색하고, 1991 + 33을 계산해줘"
    print(f"\n1️⃣  사용자 입력: {user_input}")

    messages = [HumanMessage(content=user_input)]

    # Step 2: LLM 호출
    print("\n2️⃣  LLM 호출 중...")
    response = model_with_tools.invoke(messages)

    print(f"   Tool Calls 개수: {len(response.tool_calls)}")
    for tc in response.tool_calls:
        print(f"   - {tc['name']}: {tc['args']}")

    # Step 3: Tool 실행
    print("\n3️⃣  도구 실행 중...")
    messages.append(response)

    tool_map = {tool.name: tool for tool in tools}
    for tool_call in response.tool_calls:
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        print(f"   {tool_call['name']}: {result}")

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )

    # Step 4: 최종 응답 생성
    print("\n4️⃣  최종 응답 생성 중...")
    final_response = model_with_tools.invoke(messages)
    print(f"\n🤖 최종 응답:\n{final_response.content}")

    print("\n" + "=" * 70)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🛠️  Part 2: Tool Calling 시스템 - 실습 과제 3 해답")
    print("=" * 70)

    try:
        # 1. 수동 Tool Calling 예시
        test_manual_tool_calling()

        # 2. Simple Agent 테스트
        test_simple_agent()

    except Exception as e:
        print(f"\n⚠️  테스트 실패: {e}")
        print("(API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("💡 추가 학습 포인트:")
    print("  1. 에러 핸들링 개선 (도구 실패 시 재시도)")
    print("  2. 도구 실행 시간 측정 및 로깅")
    print("  3. 병렬 도구 실행 (asyncio)")
    print("  4. Part 3의 create_agent()와 비교")
    print("=" * 70)


if __name__ == "__main__":
    main()
