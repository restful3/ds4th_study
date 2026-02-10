"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 01_middleware_intro.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 30분

📚 학습 목표:
  - 미들웨어의 개념 이해
  - Agent 실행 루프 파악
  - 미들웨어 훅의 종류 학습

📖 공식 문서:
  • Middleware Overview: /official/14-middleware-overview.md
  • API 레퍼런스: https://reference.langchain.com/python/langchain/middleware/

📄 교안 문서:
  • Part 5 개요: /docs/part05_middleware.md
  • 관련 섹션: /docs/part05_middleware.md#1-미들웨어-개념

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 01_middleware_intro.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)

# ============================================================================
# 예제 1: 미들웨어 없는 기본 Agent
# ============================================================================

def example_1_basic_agent():
    """미들웨어 없는 기본 Agent"""
    print("=" * 70)
    print("📌 예제 1: 미들웨어 없는 기본 Agent")
    print("=" * 70)

    @tool
    def get_weather(city: str) -> str:
        """주어진 도시의 날씨를 알려줍니다."""
        return f"{city}의 날씨는 맑고 22도입니다."

    # 미들웨어 없이 Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_weather],
        system_prompt="당신은 친절한 날씨 도우미입니다.",
    )

    print("\n🤖 Agent 실행 (미들웨어 없음)")

    response = agent.invoke({
        "messages": [{"role": "user", "content": "서울 날씨 알려줘"}]
    })

    print(f"\n👤 사용자: 서울 날씨 알려줘")
    print(f"🤖 Agent: {response['messages'][-1].content}")

    print("\n💡 기본 Agent는 단순히 입력 → 모델 → 도구 → 출력으로 동작합니다.")


# ============================================================================
# 예제 2: 간단한 로깅 미들웨어 (데코레이터 방식)
# ============================================================================

def example_2_logging_decorator():
    """데코레이터로 만든 간단한 로깅 미들웨어"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 로깅 미들웨어 (데코레이터 방식)")
    print("=" * 70)

    from langchain.agents.middleware import before_model, after_model, AgentState
    from langgraph.runtime import Runtime
    from typing import Any

    @before_model
    def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """모델 호출 전 로깅"""
        print(f"\n📥 [before_model] 입력 메시지 수: {len(state['messages'])}")
        return None  # 상태 변경 없음

    @after_model
    def log_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """모델 호출 후 로깅"""
        last_msg = state['messages'][-1]
        print(f"📤 [after_model] 모델 응답: {last_msg.content[:50]}...")
        return None

    @tool
    def calculator(expression: str) -> str:
        """수식을 계산합니다."""
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except:
            return "계산 오류"

    # 미들웨어와 함께 Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[calculator],
        middleware=[log_before, log_after],
    )

    print("\n🤖 Agent 실행 (로깅 미들웨어 포함)")

    response = agent.invoke({
        "messages": [{"role": "user", "content": "25 * 4는 얼마야?"}]
    })

    print(f"\n✅ 최종 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 3: 로깅 미들웨어 (클래스 방식)
# ============================================================================

def example_3_logging_class():
    """클래스로 만든 로깅 미들웨어"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 로깅 미들웨어 (클래스 방식)")
    print("=" * 70)

    from langchain.agents.middleware import AgentMiddleware, AgentState
    from langgraph.runtime import Runtime
    from typing import Any

    class LoggingMiddleware(AgentMiddleware):
        """로깅을 위한 미들웨어 (클래스 방식)"""

        def __init__(self, prefix: str = "LOG"):
            super().__init__()
            self.prefix = prefix

        def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"\n[{self.prefix}] 🚀 Agent 시작")
            return None

        def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"[{self.prefix}] 📥 모델 호출 전 (메시지: {len(state['messages'])}개)")
            return None

        def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"[{self.prefix}] 📤 모델 응답 받음")
            return None

        def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
            print(f"[{self.prefix}] ✅ Agent 완료\n")
            return None

    @tool
    def search_wiki(query: str) -> str:
        """위키백과를 검색합니다."""
        return f"'{query}'에 대한 위키 검색 결과: (샘플 데이터)"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_wiki],
        middleware=[LoggingMiddleware(prefix="WIKI")],
    )

    print("\n🤖 Agent 실행")

    response = agent.invoke({
        "messages": [{"role": "user", "content": "파이썬이 뭐야?"}]
    })

    print(f"✅ 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 4: 여러 미들웨어 조합
# ============================================================================

def example_4_multiple_middleware():
    """여러 미들웨어를 조합하여 사용"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 여러 미들웨어 조합")
    print("=" * 70)

    from langchain.agents.middleware import before_model, after_model, AgentState
    from langgraph.runtime import Runtime
    from typing import Any
    import time

    @before_model
    def timestamp_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"\n⏰ [Middleware 1] 시작 시간: {time.strftime('%H:%M:%S')}")
        return None

    @before_model
    def count_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"📊 [Middleware 2] 현재 메시지 수: {len(state['messages'])}")
        return None

    @after_model
    def timestamp_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"⏰ [Middleware 1] 종료 시간: {time.strftime('%H:%M:%S')}")
        return None

    @tool
    def get_time() -> str:
        """현재 시간을 알려줍니다."""
        return f"현재 시간은 {time.strftime('%H:%M:%S')}입니다."

    # 여러 미들웨어 조합
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_time],
        middleware=[
            timestamp_before,
            count_messages,
            timestamp_after,
        ],
    )

    print("\n🤖 Agent 실행")

    response = agent.invoke({
        "messages": [{"role": "user", "content": "지금 몇 시야?"}]
    })

    print(f"\n✅ 응답: {response['messages'][-1].content}")
    print("\n💡 미들웨어는 리스트 순서대로 실행됩니다!")


# ============================================================================
# 예제 5: 미들웨어 실행 순서 확인
# ============================================================================

def example_5_execution_order():
    """미들웨어의 실행 순서 확인"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 미들웨어 실행 순서")
    print("=" * 70)

    from langchain.agents.middleware import (
        before_agent, before_model, after_model, after_agent,
        AgentState
    )
    from langgraph.runtime import Runtime
    from typing import Any

    @before_agent
    def before_agent_a(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("1️⃣ [A] before_agent 실행")
        return None

    @before_model
    def before_model_a(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("2️⃣ [A] before_model 실행")
        return None

    @after_model
    def after_model_a(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("3️⃣ [A] after_model 실행")
        return None

    @after_agent
    def after_agent_a(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("4️⃣ [A] after_agent 실행")
        return None

    @tool
    def simple_tool(text: str) -> str:
        """간단한 도구"""
        return f"처리됨: {text}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[simple_tool],
        middleware=[
            before_agent_a,
            before_model_a,
            after_model_a,
            after_agent_a,
        ],
    )

    print("\n🤖 Agent 실행")
    print("=" * 50)

    response = agent.invoke({
        "messages": [{"role": "user", "content": "테스트"}]
    })

    print("=" * 50)
    print(f"\n✅ 응답: {response['messages'][-1].content[:50]}...")

    print("\n💡 실행 순서:")
    print("  1. before_agent (시작)")
    print("  2. before_model (모델 호출 전)")
    print("  3. [모델 호출 + 도구 실행]")
    print("  4. after_model (모델 응답 후)")
    print("  5. after_agent (완료)")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - 미들웨어 소개")
    print("\n")

    # 예제 실행
    example_1_basic_agent()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_logging_decorator()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_logging_class()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_multiple_middleware()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_execution_order()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 5-1: 미들웨어 소개 완료!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 02_before_after_model.py - before/after 훅 상세")
    print("  2. 03_wrap_model_call.py - wrap_model_call 훅")
    print("  3. 04_wrap_tool_call.py - wrap_tool_call 훅")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 미들웨어 vs 도구:
#    - 도구: Agent가 선택적으로 호출 (필요할 때만)
#    - 미들웨어: 모든 요청에서 자동 실행 (항상)
#
# 2. 미들웨어 활용 사례:
#    - 로깅 및 모니터링
#    - 입출력 검증
#    - 비용 추적
#    - Rate Limiting
#    - 캐싱
#
# 3. 데코레이터 vs 클래스:
#    - 데코레이터: 간단한 단일 훅
#    - 클래스: 복잡한 로직, 여러 훅, 설정 필요
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: 미들웨어가 실행되지 않음
# 해결: middleware 리스트에 올바르게 전달했는지 확인
#
# 문제: 미들웨어에서 에러 발생
# 해결: None을 반환해야 함 (상태 변경 없을 때)
#
# 문제: 미들웨어 순서가 중요한가?
# 해결: 네! 리스트 순서대로 실행됩니다
#
# ============================================================================
