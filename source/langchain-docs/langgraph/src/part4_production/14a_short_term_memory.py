"""
[Chapter 14a] 메모리 시스템 - 단기 메모리

📝 설명:
    단기 메모리(Short-term Memory)는 Thread 단위의 대화 기록을 유지합니다.
    Checkpointer를 통해 구현되며, 대화 세션 내에서 컨텍스트를 유지합니다.

🎯 학습 목표:
    - 단기 메모리의 개념과 역할
    - Checkpointer를 통한 대화 기록 유지
    - 세션 관리 패턴

📚 관련 문서:
    - docs/Part4-Production/14-memory.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/memory/

💻 실행 방법:
    python -m src.part4_production.14a_short_term_memory

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# 1. 단기 메모리 개념
# =============================================================================

def explain_short_term_memory():
    """단기 메모리 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 단기 메모리 (Short-term Memory)")
    print("=" * 60)

    print("""
단기 메모리란?
- 단일 대화 세션 내의 컨텍스트 유지
- Thread 단위로 격리
- Checkpointer를 통해 자동 관리

특징:
- 세션 시작 시 초기화
- 대화 내내 지속
- Thread 종료 시 선택적 삭제 가능

사용 사례:
- 챗봇 대화 기록
- 멀티턴 작업 추적
- 컨텍스트 기반 응답

구현 방식:
┌─────────────────────────────────────┐
│  Thread: user-123                   │
│  ├── Checkpoint 1 (첫 대화)         │
│  ├── Checkpoint 2 (두 번째 대화)    │
│  └── Checkpoint 3 (세 번째 대화)    │
└─────────────────────────────────────┘
""")


# =============================================================================
# 2. 기본 대화 메모리
# =============================================================================

def create_chatbot_with_memory():
    """메모리를 가진 챗봇 생성"""

    def chatbot(state: MessagesState) -> MessagesState:
        """간단한 챗봇 응답"""
        messages = state["messages"]
        last_msg = messages[-1].content if messages else ""

        # 대화 기록 활용 시뮬레이션
        context = f"이전 대화 {len(messages) - 1}개를 기억합니다. "

        response = AIMessage(content=f"{context}'{last_msg}'에 대한 응답입니다.")
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_basic_memory_example():
    """기본 대화 메모리 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 대화 메모리")
    print("=" * 60)

    app = create_chatbot_with_memory()
    config = {"configurable": {"thread_id": "user-001"}}

    # 대화 진행
    conversations = [
        "안녕하세요!",
        "제 이름은 홍길동입니다.",
        "제 이름이 뭐라고 했죠?"
    ]

    messages = []
    for msg in conversations:
        messages.append(HumanMessage(content=msg))
        result = app.invoke({"messages": messages}, config=config)
        messages = result["messages"]

        print(f"\n👤 사용자: {msg}")
        print(f"🤖 봇: {result['messages'][-1].content}")


# =============================================================================
# 3. LLM을 사용한 대화 메모리
# =============================================================================

def create_llm_chatbot_with_memory():
    """LLM 기반 챗봇 (메모리 포함)"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

    def chatbot(state: MessagesState) -> MessagesState:
        """LLM 챗봇 응답"""
        # 시스템 메시지 추가
        system_msg = SystemMessage(content="""당신은 친절한 한국어 AI 어시스턴트입니다.
이전 대화 내용을 잘 기억하고, 맥락에 맞게 응답하세요.
사용자가 이전에 말한 내용을 참조할 때 정확하게 답변하세요.""")

        messages = [system_msg] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_llm_memory_example():
    """LLM 대화 메모리 예제"""
    print("\n" + "=" * 60)
    print("예제 2: LLM 기반 대화 메모리")
    print("=" * 60)

    app = create_llm_chatbot_with_memory()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    config = {"configurable": {"thread_id": "user-002"}}

    conversations = [
        "안녕하세요! 저는 김철수입니다.",
        "제가 좋아하는 음식은 피자입니다.",
        "제 이름과 좋아하는 음식이 뭐였죠?"
    ]

    messages = []
    for msg in conversations:
        messages.append(HumanMessage(content=msg))
        result = app.invoke({"messages": messages}, config=config)
        messages = result["messages"]

        print(f"\n👤 사용자: {msg}")
        print(f"🤖 봇: {result['messages'][-1].content[:200]}...")


# =============================================================================
# 4. 세션 관리
# =============================================================================

def run_session_management_example():
    """세션 관리 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 세션 관리 (다중 사용자)")
    print("=" * 60)

    app = create_chatbot_with_memory()

    # 여러 사용자 세션
    users = {
        "user-alice": ["안녕! 나는 앨리스야", "내 이름 기억해?"],
        "user-bob": ["안녕! 나는 밥이야", "내 이름 뭐라고 했지?"],
    }

    for user_id, messages_list in users.items():
        print(f"\n📱 세션: {user_id}")

        config = {"configurable": {"thread_id": user_id}}
        messages = []

        for msg in messages_list:
            messages.append(HumanMessage(content=msg))
            result = app.invoke({"messages": messages}, config=config)
            messages = result["messages"]

            print(f"   👤 {msg}")
            print(f"   🤖 {result['messages'][-1].content}")


# =============================================================================
# 5. 대화 기록 조회
# =============================================================================

def run_history_retrieval_example():
    """대화 기록 조회 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 대화 기록 조회")
    print("=" * 60)

    app = create_chatbot_with_memory()
    config = {"configurable": {"thread_id": "history-demo"}}

    # 대화 진행
    messages = []
    for msg in ["첫 번째 메시지", "두 번째 메시지", "세 번째 메시지"]:
        messages.append(HumanMessage(content=msg))
        result = app.invoke({"messages": messages}, config=config)
        messages = result["messages"]

    # 현재 상태에서 대화 기록 조회
    current_state = app.get_state(config)
    print(f"\n📜 현재 대화 기록 ({len(current_state.values['messages'])}개 메시지):")
    for i, msg in enumerate(current_state.values["messages"]):
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"   {i+1}. {role} {msg.content[:50]}...")


# =============================================================================
# 6. 세션 초기화
# =============================================================================

def run_session_reset_example():
    """세션 초기화 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 세션 초기화")
    print("=" * 60)

    app = create_chatbot_with_memory()

    # 기존 세션
    old_config = {"configurable": {"thread_id": "reset-demo"}}

    messages = [HumanMessage(content="기존 세션의 메시지입니다.")]
    result = app.invoke({"messages": messages}, config=old_config)
    print(f"\n📌 기존 세션: {len(result['messages'])}개 메시지")

    # 새 세션으로 초기화 (새 thread_id 사용)
    new_config = {"configurable": {"thread_id": "reset-demo-new"}}

    messages = [HumanMessage(content="새 세션의 첫 메시지입니다.")]
    result = app.invoke({"messages": messages}, config=new_config)
    print(f"📌 새 세션: {len(result['messages'])}개 메시지")

    print("\n💡 세션 초기화는 새로운 thread_id를 사용하면 됩니다!")


# =============================================================================
# 7. 단기 메모리 패턴 정리
# =============================================================================

def explain_short_term_memory_patterns():
    """단기 메모리 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 단기 메모리 패턴 정리")
    print("=" * 60)

    print("""
단기 메모리 구현 패턴:

1. Thread ID 전략
   - 사용자 ID 기반: "user-{user_id}"
   - 세션 ID 기반: "session-{session_id}"
   - 복합: "user-{user_id}-{timestamp}"

2. 메시지 관리
   - 전체 기록 유지: 모든 메시지 보관
   - 윈도우 방식: 최근 N개만 유지
   - 요약 방식: 오래된 메시지 요약

3. 상태 조회 API
   - get_state(config): 현재 상태
   - get_state_history(config): 전체 히스토리

4. 세션 생명주기
   - 생성: 첫 invoke 시 자동
   - 유지: Checkpointer가 관리
   - 삭제: 수동 또는 TTL 설정

주의사항:
- 메시지가 많아지면 토큰 제한 고려
- 민감한 정보 저장 주의
- 동시 접근 시 충돌 방지
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 14a] 메모리 시스템 - 단기 메모리")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_short_term_memory()

    # 예제 실행
    run_basic_memory_example()
    run_llm_memory_example()
    run_session_management_example()
    run_history_retrieval_example()
    run_session_reset_example()

    # 패턴 정리
    explain_short_term_memory_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 14b_long_term_memory.py (장기 메모리)")
    print("=" * 60)


if __name__ == "__main__":
    main()
