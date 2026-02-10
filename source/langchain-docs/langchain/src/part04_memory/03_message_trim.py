"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory System - Message Trim
================================================================================

파일명: 03_message_trim.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 20분

📚 학습 목표:
  - Context Window 문제 이해
  - before_model 미들웨어로 메시지 Trim 구현
  - 다양한 Trim 전략 학습
  - Delete vs Trim 차이 이해
  - Message 필터링 기법

📖 공식 문서:
  • Short-term Memory: /official/10-short-term-memory.md

📄 교안 문서:
  • Part 4 메모리: /docs/part04_memory.md (Section 3)

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 03_message_trim.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)

# ============================================================================
# 예제 1: Context Window 문제 시연
# ============================================================================

def example_1_context_overflow():
    """Context Window가 가득 찼을 때의 문제"""
    print("=" * 70)
    print("📌 예제 1: Context Window 문제")
    print("=" * 70)
    print("\n💡 긴 대화는 Context Window를 초과할 수 있습니다.\n")

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "overflow-test"}}

    # 많은 메시지 생성 (시뮬레이션)
    print("=" * 50)
    print("📝 긴 대화 시뮬레이션")
    print("=" * 50)

    topics = [
        "파이썬에 대해 알려주세요.",
        "자바스크립트는 어떤가요?",
        "러스트에 대해서도 설명해주세요.",
        "Go 언어는 어떤 특징이 있나요?",
        "코틀린에 대해 알려주세요.",
        "타입스크립트는 왜 인기가 있나요?",
        "스위프트의 장점은 무엇인가요?",
        "C++는 아직도 많이 쓰이나요?",
    ]

    for i, topic in enumerate(topics, 1):
        print(f"\n대화 {i}:")
        print(f"👤 사용자: {topic}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": topic}]},
            config
        )

        # 응답 길이 제한하여 출력
        response = result['messages'][-1].content
        print(f"🤖 AI: {response[:100]}...")

    # 현재 상태 확인
    state = agent.get_state(config)
    messages = state.values["messages"]

    print(f"\n📊 현재 상태:")
    print(f"   - 전체 메시지 수: {len(messages)}")
    print(f"   - 사용자 메시지: {sum(1 for m in messages if m.type == 'human')}")
    print(f"   - AI 메시지: {sum(1 for m in messages if m.type == 'ai')}")

    # 대략적인 토큰 수 추정
    total_chars = sum(len(m.content) for m in messages if hasattr(m, 'content'))
    estimated_tokens = total_chars // 4  # 대략 4자 = 1토큰

    print(f"   - 총 문자 수: {total_chars:,}")
    print(f"   - 예상 토큰 수: ~{estimated_tokens:,}")

    print("\n⚠️  문제점:")
    print("   - 메시지가 계속 쌓이면 Context Window 초과")
    print("   - LLM 성능 저하 (긴 컨텍스트)")
    print("   - API 비용 증가")
    print("   - 응답 속도 저하")

    print("\n💡 해결책:")
    print("   - Message Trim: 오래된 메시지 제거")
    print("   - Message Summarization: 요약 생성")
    print("   - Message Delete: 영구 삭제")


# ============================================================================
# 예제 2: before_model로 메시지 Trim
# ============================================================================

def example_2_trim_messages():
    """before_model 미들웨어로 메시지 제한"""
    print("\n" + "=" * 70)
    print("📌 예제 2: before_model로 메시지 Trim")
    print("=" * 70)
    print("\n💡 LLM 호출 전에 메시지 개수를 제한합니다.\n")

    # Trim 미들웨어 정의
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """최근 N개의 메시지만 유지"""
        messages = state["messages"]

        # 메시지가 5개 이하면 그대로 유지
        if len(messages) <= 5:
            return None

        # 첫 메시지(시스템 메시지) + 최근 4개 메시지
        # 홀수/짝수 조정 (사용자-AI 쌍 유지)
        if len(messages) % 2 == 0:
            recent = messages[-4:]
        else:
            recent = messages[-3:]

        print(f"✂️  Trim: {len(messages)}개 → {len(recent) + 1}개 메시지")

        # REMOVE_ALL_MESSAGES로 전체 삭제 후 재구성
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                messages[0],  # 첫 메시지 유지 (시스템 메시지)
                *recent
            ]
        }

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[trim_messages],  # 미들웨어 추가
        checkpointer=checkpointer,
        system_prompt="당신은 도움이 되는 AI 어시스턴트입니다."
    )

    config: RunnableConfig = {"configurable": {"thread_id": "trim-test"}}

    # 여러 대화 진행
    conversations = [
        "안녕하세요!",
        "제 이름은 김철수입니다.",
        "저는 서울에 살아요.",
        "파이썬을 좋아합니다.",
        "오늘 날씨가 좋네요.",
        "저녁 메뉴 추천해주세요.",
        "제 이름이 뭐였죠?",  # 초기 정보는 잊어버림
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"\n대화 {i}:")
        print(f"👤 사용자: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )

        print(f"🤖 AI: {result['messages'][-1].content}")

        # 현재 메시지 수
        state = agent.get_state(config)
        msg_count = len(state.values["messages"])
        print(f"   📊 현재 메시지 수: {msg_count}")

    print("\n💡 결과:")
    print("   - Trim으로 메시지 수가 제한됨")
    print("   - 오래된 정보는 기억하지 못함")
    print("   - Context Window 초과 방지")


# ============================================================================
# 예제 3: 첫 메시지 + 최근 N개 유지 전략
# ============================================================================

def example_3_keep_first_and_recent():
    """시스템 메시지와 최근 메시지 모두 유지"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 첫 메시지 + 최근 메시지 유지 전략")
    print("=" * 70)
    print("\n💡 중요한 시스템 메시지는 항상 유지합니다.\n")

    @before_model
    def smart_trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """시스템 메시지 + 최근 대화 유지"""
        messages = state["messages"]

        MAX_RECENT = 6  # 최근 6개 유지

        if len(messages) <= MAX_RECENT + 1:
            return None

        # 시스템 메시지들 (처음 N개)
        system_messages = []
        for msg in messages:
            if msg.type == "system":
                system_messages.append(msg)
            else:
                break

        # 최근 메시지들
        recent_messages = messages[-MAX_RECENT:]

        print(f"✂️  Smart Trim:")
        print(f"   - 시스템 메시지: {len(system_messages)}개")
        print(f"   - 최근 메시지: {len(recent_messages)}개")
        print(f"   - 삭제된 메시지: {len(messages) - len(system_messages) - len(recent_messages)}개")

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *system_messages,
                *recent_messages
            ]
        }

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[smart_trim],
        checkpointer=checkpointer,
        system_prompt="당신은 친절한 AI 어시스턴트입니다. 사용자를 존중하며 대화하세요."
    )

    config: RunnableConfig = {"configurable": {"thread_id": "smart-trim-test"}}

    # 긴 대화
    for i in range(1, 11):
        msg = f"메시지 번호 {i}입니다."
        print(f"\n👤 사용자: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )

        print(f"🤖 AI: {result['messages'][-1].content[:50]}...")


# ============================================================================
# 예제 4: after_model로 메시지 영구 삭제
# ============================================================================

def example_4_delete_messages():
    """after_model로 메시지를 영구적으로 삭제"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 메시지 영구 삭제 (Delete)")
    print("=" * 70)
    print("\n💡 after_model은 메시지를 Checkpointer에서도 제거합니다.\n")

    @after_model
    def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
        """오래된 메시지 영구 삭제"""
        messages = state["messages"]

        # 메시지가 8개 초과 시 가장 오래된 2개 삭제
        if len(messages) > 8:
            to_delete = messages[1:3]  # 시스템 메시지는 제외
            print(f"🗑️  영구 삭제: {len(to_delete)}개 메시지")

            return {
                "messages": [RemoveMessage(id=msg.id) for msg in to_delete]
            }

        return None

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[delete_old_messages],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "delete-test"}}

    print("=" * 50)
    print("📝 대화 진행 및 메시지 삭제")
    print("=" * 50)

    for i in range(1, 8):
        msg = f"대화 {i}"
        print(f"\n👤 사용자: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )

        print(f"🤖 AI: {result['messages'][-1].content[:50]}...")

        # 상태 확인
        state = agent.get_state(config)
        msg_count = len(state.values["messages"])
        print(f"   📊 메시지 수: {msg_count}")

    print("\n💡 Trim vs Delete:")
    print("   - Trim: 현재 호출에만 적용, 다음 호출엔 복원")
    print("   - Delete: Checkpointer에서 영구 제거, 복구 불가")


# ============================================================================
# 예제 5: 고급 필터링 전략
# ============================================================================

def example_5_message_filtering():
    """메시지 타입별 필터링"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 고급 메시지 필터링")
    print("=" * 70)
    print("\n💡 특정 조건의 메시지만 선택적으로 유지할 수 있습니다.\n")

    @before_model
    def filter_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Tool 메시지를 제거하고 대화만 유지"""
        messages = state["messages"]

        if len(messages) <= 10:
            return None

        # 시스템 메시지와 대화 메시지만 유지 (Tool 메시지 제외)
        filtered = []
        for msg in messages:
            if msg.type in ["system", "human", "ai"]:
                # Tool call이 없는 AI 메시지만 포함
                if msg.type == "ai":
                    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                        filtered.append(msg)
                else:
                    filtered.append(msg)

        # 너무 많으면 최근 것만
        if len(filtered) > 10:
            filtered = [filtered[0]] + filtered[-9:]

        print(f"🔍 필터링:")
        print(f"   - 원본: {len(messages)}개")
        print(f"   - 필터링 후: {len(filtered)}개")

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *filtered
            ]
        }

    from langchain.tools import tool

    @tool
    def get_info(query: str) -> str:
        """정보를 조회합니다."""
        return f"'{query}'에 대한 정보입니다."

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_info],
        middleware=[filter_messages],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "filter-test"}}

    # Tool을 사용하는 대화
    conversations = [
        "안녕하세요!",
        "날씨 정보를 조회해주세요.",
        "뉴스도 확인해주세요.",
        "오늘 일정을 알려주세요.",
        "감사합니다.",
    ]

    for msg in conversations:
        print(f"\n👤 사용자: {msg}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )
        print(f"🤖 AI: {result['messages'][-1].content[:80]}...")

    print("\n💡 필터링 전략:")
    print("   - Tool 메시지 제거로 토큰 절약")
    print("   - 핵심 대화 내용만 유지")
    print("   - Context Window 효율적 사용")


# ============================================================================
# 보너스: 토큰 기반 Trim
# ============================================================================

def bonus_token_based_trim():
    """토큰 수를 기준으로 메시지 Trim"""
    print("\n" + "=" * 70)
    print("🎁 보너스: 토큰 기반 Trim")
    print("=" * 70)
    print("\n💡 메시지 개수가 아닌 토큰 수를 기준으로 제한합니다.\n")

    def estimate_tokens(messages) -> int:
        """간단한 토큰 수 추정 (실제로는 tiktoken 사용 권장)"""
        total_chars = sum(
            len(m.content) for m in messages
            if hasattr(m, 'content') and m.content
        )
        return total_chars // 4  # 대략 4자 = 1토큰

    @before_model
    def trim_by_tokens(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """토큰 수 기준으로 Trim"""
        messages = state["messages"]
        MAX_TOKENS = 500  # 최대 토큰 수

        current_tokens = estimate_tokens(messages)

        if current_tokens <= MAX_TOKENS:
            return None

        # 뒤에서부터 메시지를 추가하면서 토큰 수 확인
        kept_messages = [messages[0]]  # 시스템 메시지
        current_tokens = estimate_tokens([messages[0]])

        for msg in reversed(messages[1:]):
            msg_tokens = estimate_tokens([msg])
            if current_tokens + msg_tokens <= MAX_TOKENS:
                kept_messages.insert(1, msg)
                current_tokens += msg_tokens
            else:
                break

        print(f"📊 토큰 기반 Trim:")
        print(f"   - 원본 토큰: ~{estimate_tokens(messages)}")
        print(f"   - Trim 후: ~{estimate_tokens(kept_messages)}")
        print(f"   - 유지 메시지: {len(kept_messages)}/{len(messages)}")

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *kept_messages
            ]
        }

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[trim_by_tokens],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "token-trim-test"}}

    # 다양한 길이의 메시지
    long_messages = [
        "짧은 메시지",
        "이것은 조금 더 긴 메시지입니다. " * 5,
        "또 다른 메시지",
        "이것은 매우 긴 메시지입니다. " * 10,
        "마지막 메시지"
    ]

    for i, msg in enumerate(long_messages, 1):
        print(f"\n대화 {i}: (길이: {len(msg)}자)")
        print(f"👤 사용자: {msg[:50]}...")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )

        print(f"🤖 AI: {result['messages'][-1].content[:50]}...")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 4: Memory System - Message Trim")
    print("\n")

    # 예제 1: Context Window 문제
    example_1_context_overflow()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 2: Trim Messages
    example_2_trim_messages()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 3: Smart Trim
    example_3_keep_first_and_recent()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 4: Delete Messages
    example_4_delete_messages()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 5: Filtering
    example_5_message_filtering()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 보너스: Token-based Trim
    print("\n" + "=" * 70)
    choice = input("🎁 보너스 예제를 실행하시겠습니까? (y/n): ").strip().lower()
    if choice == 'y':
        bonus_token_based_trim()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 4-3 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 04_summarization.py - Message Summarization")
    print("  2. 05_custom_state.py - Custom State")
    print("  3. 06_long_term_store.py - Long-term Memory")
    print("\n📚 핵심 개념 복습:")
    print("  • before_model: LLM 호출 전 메시지 처리")
    print("  • after_model: LLM 호출 후 메시지 처리")
    print("  • Trim: 현재 호출에만 적용 (복구 가능)")
    print("  • Delete: 영구 삭제 (복구 불가)")
    print("  • 전략: 개수, 토큰, 필터링 기반")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 실제 토큰 계산:
#    import tiktoken
#    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
#    tokens = encoding.encode(text)
#    token_count = len(tokens)
#
# 2. Trim 전략 선택:
#    - 단순 대화: 최근 N개
#    - 복잡한 작업: 토큰 기반
#    - Tool 많음: 필터링 기반
#
# 3. Context Window 크기:
#    - GPT-4o-mini: 128K tokens
#    - GPT-4o: 128K tokens
#    - Claude 3.5: 200K tokens
#
# 4. 안전 여유분:
#    MAX_TOKENS = CONTEXT_WINDOW * 0.8  # 20% 여유
#
# ============================================================================
