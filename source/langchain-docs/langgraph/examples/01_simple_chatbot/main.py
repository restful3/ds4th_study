"""
간단한 챗봇 (Simple Chatbot)

이 예제는 메모리 기능이 있는 간단한 대화형 챗봇을 구현합니다.
LangGraph의 기본 기능인 State 관리, Checkpointer, MessagesState를 활용합니다.

기능:
- 대화 기록 유지 (메모리)
- 세션별 대화 관리
- 스트리밍 응답

실행 방법:
    python -m examples.01_simple_chatbot.main
"""

import os
from typing import Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# 환경 설정
# =============================================================================

load_dotenv()


# =============================================================================
# 챗봇 그래프 구현
# =============================================================================

def create_chatbot():
    """챗봇 그래프 생성"""

    def chatbot_node(state: MessagesState) -> MessagesState:
        """챗봇 노드 - 응답 생성"""
        messages = state["messages"]

        # LLM이 있으면 사용, 없으면 시뮬레이션
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic

                llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

                # 시스템 메시지 추가
                system_msg = SystemMessage(content="""당신은 친절하고 도움이 되는 AI 어시스턴트입니다.
                사용자의 질문에 명확하고 간결하게 답변해주세요.
                한국어로 대화합니다.""")

                response = llm.invoke([system_msg] + messages)
                return {"messages": [response]}

            except ImportError:
                pass

        # LLM이 없는 경우 시뮬레이션 응답
        last_msg = messages[-1].content if messages else ""
        simulated_response = f"'{last_msg}'에 대한 응답입니다. (시뮬레이션 모드)"
        return {"messages": [AIMessage(content=simulated_response)]}

    # 그래프 구성
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    # 메모리 체크포인터로 컴파일
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 인터랙티브 세션
# =============================================================================

def run_interactive_session():
    """인터랙티브 채팅 세션 실행"""

    print("=" * 60)
    print("🤖 간단한 챗봇")
    print("=" * 60)
    print("\n명령어:")
    print("  /new    - 새 대화 시작")
    print("  /history - 대화 기록 보기")
    print("  /quit   - 종료")
    print("-" * 60)

    chatbot = create_chatbot()
    thread_id = "session_1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            # 명령어 처리
            if user_input.startswith("/"):
                if user_input == "/quit":
                    print("\n👋 안녕히 가세요!")
                    break

                elif user_input == "/new":
                    thread_id = f"session_{hash(str(os.urandom(4)))}"
                    config = {"configurable": {"thread_id": thread_id}}
                    print(f"\n🆕 새 대화를 시작합니다. (세션: {thread_id[:20]}...)")
                    continue

                elif user_input == "/history":
                    state = chatbot.get_state(config)
                    messages = state.values.get("messages", [])
                    print(f"\n📜 대화 기록 ({len(messages)}개 메시지):")
                    for msg in messages:
                        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
                        print(f"   {role} {msg.content[:50]}...")
                    continue

                else:
                    print(f"   알 수 없는 명령어: {user_input}")
                    continue

            # 메시지 전송
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # 응답 출력
            ai_response = result["messages"][-1].content
            print(f"\n🤖 Bot: {ai_response}")

        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break


# =============================================================================
# 스트리밍 예제
# =============================================================================

def run_streaming_example():
    """스트리밍 응답 예제"""

    print("\n" + "=" * 60)
    print("📡 스트리밍 예제")
    print("=" * 60)

    chatbot = create_chatbot()
    config = {"configurable": {"thread_id": "stream_test"}}

    user_input = "LangGraph의 장점을 3가지 알려주세요."
    print(f"\n👤 You: {user_input}")
    print("\n🤖 Bot: ", end="", flush=True)

    # 스트리밍 실행
    for event in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="values"
    ):
        messages = event.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                # 실제 LLM 스트리밍에서는 토큰 단위로 출력
                print(last_msg.content)


# =============================================================================
# 데모 실행
# =============================================================================

def run_demo():
    """데모 실행"""

    print("=" * 60)
    print("🤖 Simple Chatbot Demo")
    print("=" * 60)

    chatbot = create_chatbot()

    # 세션 1: 첫 번째 대화
    print("\n📍 세션 1: 첫 번째 대화")
    config1 = {"configurable": {"thread_id": "demo_session_1"}}

    conversations = [
        "안녕하세요!",
        "제 이름은 철수입니다.",
        "제 이름이 뭐라고 했죠?"  # 메모리 테스트
    ]

    for msg in conversations:
        print(f"\n👤 You: {msg}")
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config=config1
        )
        print(f"🤖 Bot: {result['messages'][-1].content}")

    # 대화 기록 확인
    state = chatbot.get_state(config1)
    print(f"\n📜 세션 1 대화 기록: {len(state.values['messages'])}개 메시지")

    # 세션 2: 별도의 대화
    print("\n\n📍 세션 2: 새로운 대화 (다른 세션)")
    config2 = {"configurable": {"thread_id": "demo_session_2"}}

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="제 이름이 뭐라고 했죠?")]},
        config=config2
    )
    print(f"👤 You: 제 이름이 뭐라고 했죠?")
    print(f"🤖 Bot: {result['messages'][-1].content}")
    print("   (새 세션이므로 이전 대화 기록이 없음)")


# =============================================================================
# 메인
# =============================================================================

def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            run_interactive_session()
        elif sys.argv[1] == "stream":
            run_streaming_example()
        else:
            print(f"알 수 없는 모드: {sys.argv[1]}")
            print("사용법: python main.py [interactive|stream]")
    else:
        # 기본: 데모 실행
        run_demo()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
