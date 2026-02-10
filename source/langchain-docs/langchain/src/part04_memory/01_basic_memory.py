"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: 메모리 시스템
================================================================================

파일명: 01_basic_memory.py
난이도: ⭐⭐☆☆☆ (초급-중급)
예상 시간: 25분

📚 학습 목표:
  - Checkpointer 개념 이해
  - InMemorySaver로 대화 이력 관리
  - Thread ID를 사용한 세션 관리

📖 공식 문서:
  • Memory: /official/10-short-term-memory.md

🚀 실행 방법:
  python 01_basic_memory.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def save_note(note: str) -> str:
    """중요한 메모를 저장합니다.

    Args:
        note: 저장할 메모 내용
    """
    return f"메모가 저장되었습니다: '{note}'"


@tool
def get_current_time() -> str:
    """현재 시간을 알려줍니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# 예제 1: 메모리 없는 Agent (문제점 확인)
# ============================================================================

def example_1_without_memory():
    """메모리가 없는 Agent의 문제점"""
    print("=" * 70)
    print("📌 예제 1: 메모리 없는 Agent (문제점)")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [save_note]

    # Checkpointer 없이 Agent 생성
    agent = create_agent(model=model, tools=tools)

    # 첫 번째 대화
    print("\n🔹 대화 1:")
    print("👤 사용자: 제 이름은 김철수입니다.")

    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "제 이름은 김철수입니다."}]
    })

    print(f"🤖 Agent: {result1['messages'][-1].content}")

    # 두 번째 대화 (새로운 invoke)
    print("\n🔹 대화 2 (이전 대화와 별개):")
    print("👤 사용자: 제 이름이 뭐라고 했죠?")

    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "제 이름이 뭐라고 했죠?"}]
    })

    print(f"🤖 Agent: {result2['messages'][-1].content}")

    print("\n❌ 문제: Agent가 이전 대화를 기억하지 못합니다!")
    print("💡 해결: Checkpointer를 사용해야 합니다.\n")


# ============================================================================
# 예제 2: InMemorySaver로 메모리 추가
# ============================================================================

def example_2_with_memory():
    """InMemorySaver로 대화 이력 유지"""
    print("=" * 70)
    print("📌 예제 2: InMemorySaver로 메모리 추가")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [save_note]

    # Checkpointer 생성
    checkpointer = InMemorySaver()

    # Checkpointer와 함께 Agent 생성
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
    )

    # Thread ID (세션 ID)
    thread_id = "user-123"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n🧵 Thread ID: {thread_id}")

    # 첫 번째 대화
    print("\n🔹 대화 1:")
    print("👤 사용자: 제 이름은 김철수입니다.")

    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름은 김철수입니다."}]},
        config=config
    )

    print(f"🤖 Agent: {result1['messages'][-1].content}")

    # 두 번째 대화 (같은 thread_id 사용)
    print("\n🔹 대화 2 (같은 세션):")
    print("👤 사용자: 제 이름이 뭐라고 했죠?")

    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름이 뭐라고 했죠?"}]},
        config=config
    )

    print(f"🤖 Agent: {result2['messages'][-1].content}")

    print("\n✅ 성공: Agent가 이전 대화를 기억합니다!\n")


# ============================================================================
# 예제 3: 여러 사용자의 독립적인 세션
# ============================================================================

def example_3_multiple_sessions():
    """Thread ID로 여러 사용자 세션 관리"""
    print("=" * 70)
    print("📌 예제 3: 여러 사용자의 독립적인 세션")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [save_note]
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
    )

    # 사용자 A의 대화
    print("\n👤 사용자 A (thread_id: user-a):")
    config_a = {"configurable": {"thread_id": "user-a"}}

    result_a1 = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름은 Alice입니다."}]},
        config=config_a
    )
    print(f"   🤖 Agent: {result_a1['messages'][-1].content}")

    # 사용자 B의 대화
    print("\n👤 사용자 B (thread_id: user-b):")
    config_b = {"configurable": {"thread_id": "user-b"}}

    result_b1 = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름은 Bob입니다."}]},
        config=config_b
    )
    print(f"   🤖 Agent: {result_b1['messages'][-1].content}")

    # 사용자 A가 다시 질문
    print("\n👤 사용자 A가 다시 질문:")
    result_a2 = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름이 뭐죠?"}]},
        config=config_a
    )
    print(f"   🤖 Agent (to A): {result_a2['messages'][-1].content}")

    # 사용자 B가 다시 질문
    print("\n👤 사용자 B가 다시 질문:")
    result_b2 = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름이 뭐죠?"}]},
        config=config_b
    )
    print(f"   🤖 Agent (to B): {result_b2['messages'][-1].content}")

    print("\n✅ 각 사용자의 세션이 독립적으로 유지됩니다!\n")


# ============================================================================
# 예제 4: 대화 이력 확인
# ============================================================================

def example_4_view_history():
    """저장된 대화 이력 확인"""
    print("=" * 70)
    print("📌 예제 4: 대화 이력 확인")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [save_note, get_current_time]
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
    )

    thread_id = "demo-thread"
    config = {"configurable": {"thread_id": thread_id}}

    # 여러 대화 수행
    conversations = [
        "안녕하세요!",
        "지금 몇 시인가요?",
        "'회의는 3시'라고 메모해주세요.",
        "오늘 날씨가 좋네요.",
    ]

    print(f"\n🧵 Thread ID: {thread_id}\n")

    for i, user_msg in enumerate(conversations, 1):
        print(f"🔹 대화 {i}:")
        print(f"   👤 사용자: {user_msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config
        )

        print(f"   🤖 Agent: {result['messages'][-1].content}\n")

    # 전체 대화 이력 확인
    print("=" * 70)
    print("📜 전체 대화 이력:")
    print("=" * 70)

    # Checkpointer에서 상태 가져오기
    state = checkpointer.get(config)
    if state and "channel_values" in state:
        messages = state["channel_values"].get("messages", [])
        print(f"\n💬 총 {len(messages)}개의 메시지")

        for i, msg in enumerate(messages, 1):
            role = msg.__class__.__name__
            if role == "HumanMessage":
                print(f"\n   {i}. 👤 사용자: {msg.content}")
            elif role == "AIMessage":
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"   {i}. 🔧 Agent가 도구 호출")
                else:
                    print(f"   {i}. 🤖 Agent: {msg.content}")
            elif role == "ToolMessage":
                print(f"   {i}. ✅ 도구 결과: {msg.content}")

    print("\n")


# ============================================================================
# 예제 5: 메모리의 중요성 데모
# ============================================================================

def example_5_memory_importance():
    """메모리가 중요한 실제 시나리오"""
    print("=" * 70)
    print("📌 예제 5: 메모리의 중요성 - 실전 시나리오")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [save_note]
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        system_prompt="당신은 개인 비서입니다. 사용자의 요청을 기억하고 도와주세요."
    )

    thread_id = "personal-assistant"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n🤖 개인 비서 Agent 시작\n")

    # 복잡한 대화 시나리오
    scenario = [
        ("내일 회의가 3시에 있어. 기억해줘.", "회의 일정 저장"),
        ("그리고 김팀장님께 보고서도 준비해야 해.", "보고서 작업 추가"),
        ("내일 뭐 해야 하더라?", "저장된 일정 확인"),
    ]

    for i, (user_msg, description) in enumerate(scenario, 1):
        print(f"🔹 시나리오 {i}: {description}")
        print(f"   👤 사용자: {user_msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config
        )

        print(f"   🤖 Agent: {result['messages'][-1].content}\n")

    print("✅ 메모리 덕분에 Agent가 맥락을 이해하고 유용한 도움을 제공합니다!\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n🎓 Part 4: 메모리 시스템 - InMemorySaver\n")

    example_1_without_memory()
    input("⏎ 계속하려면 Enter...")

    example_2_with_memory()
    input("⏎ 계속하려면 Enter...")

    example_3_multiple_sessions()
    input("⏎ 계속하려면 Enter...")

    example_4_view_history()
    input("⏎ 계속하려면 Enter...")

    example_5_memory_importance()

    print("=" * 70)
    print("🎉 InMemorySaver 학습 완료!")
    print("=" * 70)
    print("\n💡 주요 학습 내용:")
    print("   ✅ Checkpointer의 역할")
    print("   ✅ Thread ID로 세션 관리")
    print("   ✅ 대화 이력 유지 방법")
    print("   ✅ 여러 사용자 세션 분리")
    print("\n📖 다음: 02_postgres_memory.py - 영구 메모리 저장")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 핵심 포인트
# ============================================================================
#
# 1. Checkpointer:
#    - Agent의 상태(대화 이력)를 저장하고 복원
#    - InMemorySaver: 메모리에만 저장 (프로세스 종료 시 삭제)
#    - PostgresSaver: DB에 영구 저장 (Part 4.2)
#
# 2. Thread ID (세션 ID):
#    - 각 사용자/대화를 구분하는 식별자
#    - config = {"configurable": {"thread_id": "user-123"}}
#    - 같은 thread_id = 같은 세션
#
# 3. 언제 메모리가 필요한가?:
#    - 멀티턴 대화 (이전 대화 참조)
#    - 개인화된 경험
#    - 작업 진행 상태 추적
#
# 4. InMemorySaver의 한계:
#    - 프로세스 종료 시 데이터 손실
#    - 프로덕션에는 PostgresSaver 사용 권장
#
# ============================================================================
