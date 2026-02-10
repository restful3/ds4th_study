"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 02_before_after_model.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 30분

📚 학습 목표:
  - before_model 훅 활용
  - after_model 훅 활용
  - 상태 수정 및 조기 종료 학습

📖 공식 문서:
  • Custom Middleware: /official/16-custom-middleware.md#node-style-hooks

📄 교안 문서:
  • Part 5.3: /docs/part05_middleware.md#31-before_model--after_model-훅

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 02_before_after_model.py

================================================================================
"""

import os
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from typing import Any

load_dotenv()

# ============================================================================
# 예제 1: before_model로 메시지 카운팅
# ============================================================================

def example_1_count_messages():
    """before_model로 메시지 수 추적"""
    print("=" * 70)
    print("📌 예제 1: before_model로 메시지 카운팅")
    print("=" * 70)

    call_count = {"count": 0}

    @before_model
    def count_calls(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        call_count["count"] += 1
        print(f"\n📊 모델 호출 #{call_count['count']}")
        print(f"   메시지 수: {len(state['messages'])}")
        return None

    @tool
    def simple_calc(a: int, b: int) -> int:
        """두 수를 더합니다."""
        return a + b

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[simple_calc],
        middleware=[count_calls],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "10 + 20을 계산해줘"}]
    })

    print(f"\n✅ 총 모델 호출 횟수: {call_count['count']}")
    print(f"📝 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 2: after_model로 응답 로깅
# ============================================================================

def example_2_log_responses():
    """after_model로 모든 모델 응답 로깅"""
    print("\n" + "=" * 70)
    print("📌 예제 2: after_model로 응답 로깅")
    print("=" * 70)

    responses = []

    @after_model
    def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_msg = state['messages'][-1]
        responses.append(last_msg.content)
        print(f"\n📝 응답 #{len(responses)}: {last_msg.content[:60]}...")
        return None

    @tool
    def get_weather(city: str) -> str:
        """날씨를 조회합니다."""
        return f"{city}의 날씨는 맑음, 23도"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_weather],
        middleware=[log_response],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "서울과 부산 날씨 알려줘"}]
    })

    print(f"\n✅ 총 {len(responses)}개의 응답이 기록되었습니다")


# ============================================================================
# 예제 3: before_model로 메시지 제한 (조기 종료)
# ============================================================================

def example_3_message_limit():
    """메시지 제한을 초과하면 조기 종료"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 메시지 제한 (조기 종료)")
    print("=" * 70)

    MAX_MESSAGES = 5

    @before_model(can_jump_to=["end"])
    def limit_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) >= MAX_MESSAGES:
            print(f"\n⚠️ 메시지 제한 초과 ({len(state['messages'])}/{MAX_MESSAGES})")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "대화가 너무 길어졌습니다. 새로운 대화를 시작하세요."
                }],
                "jump_to": "end"
            }
        return None

    @tool
    def echo(text: str) -> str:
        """텍스트를 반환합니다."""
        return text

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[echo],
        middleware=[limit_messages],
    )

    # 여러 메시지 추가
    messages = [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕하세요"},
        {"role": "user", "content": "테스트1"},
        {"role": "assistant", "content": "답변1"},
        {"role": "user", "content": "테스트2"},
    ]

    response = agent.invoke({"messages": messages})

    print(f"\n✅ 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 4: before_model로 시스템 메시지 주입
# ============================================================================

def example_4_inject_system_message():
    """동적으로 시스템 메시지 주입"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 시스템 메시지 주입")
    print("=" * 70)

    @before_model
    def inject_timestamp(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # 시스템 메시지가 없으면 추가
        if not any(m.type == "system" for m in state["messages"]):
            return {
                "messages": [{
                    "role": "system",
                    "content": f"현재 시간은 {current_time}입니다. 답변에 시간 정보를 포함하세요."
                }] + [m.model_dump() for m in state["messages"]]
            }
        return None

    @tool
    def get_info(topic: str) -> str:
        """정보를 조회합니다."""
        return f"{topic}에 대한 정보입니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_info],
        middleware=[inject_timestamp],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "안녕"}]
    })

    print(f"\n✅ 응답: {response['messages'][-1].content}")
    print("💡 시스템 메시지가 자동으로 주입되었습니다!")


# ============================================================================
# 예제 5: after_model로 응답 변환
# ============================================================================

def example_5_transform_response():
    """after_model로 응답을 변환"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 응답 변환")
    print("=" * 70)

    @after_model
    def add_emoji(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_msg = state["messages"][-1]

        # AI 메시지에 이모지 추가
        if last_msg.type == "ai" and not last_msg.tool_calls:
            modified_content = f"🤖 {last_msg.content}"
            print(f"\n✨ 응답에 이모지 추가됨")

            # 메시지 수정
            new_messages = state["messages"][:-1] + [{
                "role": "assistant",
                "content": modified_content
            }]

            return {"messages": new_messages}

        return None

    @tool
    def greet(name: str) -> str:
        """인사합니다."""
        return f"{name}님, 안녕하세요!"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[greet],
        middleware=[add_emoji],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "내 이름은 철수야"}]
    })

    print(f"\n✅ 변환된 응답: {response['messages'][-1].content}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - before/after 훅")
    print("\n")

    example_1_count_messages()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_log_responses()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_message_limit()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_inject_system_message()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_transform_response()

    print("\n" + "=" * 70)
    print("🎉 Part 5-2 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 조기 종료 (Early Exit):
#    - can_jump_to=["end"] 설정
#    - jump_to="end" 반환
#
# 2. 상태 수정:
#    - 딕셔너리 반환으로 상태 업데이트
#    - None 반환 시 상태 유지
#
# 3. 실무 활용:
#    - 대화 길이 제한
#    - 동적 프롬프트 주입
#    - 응답 포맷팅
#
# ============================================================================
