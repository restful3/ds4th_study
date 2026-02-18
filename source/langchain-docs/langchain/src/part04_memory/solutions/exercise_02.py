"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory - 실습 과제 2 해답
================================================================================

과제: 자동 요약 시스템
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. 대화가 길어지면 자동으로 요약
2. 요약본을 메모리에 저장하여 컨텍스트 유지
3. 토큰 한도를 고려한 스마트 메모리 관리

학습 목표:
- 메시지 수 기반 요약 트리거
- 요약을 활용한 메모리 최적화
- 컨텍스트 윈도우 관리

================================================================================
"""

from dotenv import load_dotenv
from typing import Sequence, Literal
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# ============================================================================
# State 정의
# ============================================================================

class ChatState(MessagesState):
    """채팅 상태 with 요약"""
    summary: str  # 현재까지의 대화 요약


# ============================================================================
# 자동 요약 챗봇 구축
# ============================================================================

def create_summarizing_chatbot(
    model_name: str = "gpt-4o-mini",
    summary_threshold: int = 6  # 메시지 N개마다 요약
) -> CompiledStateGraph:
    """자동 요약 기능이 있는 챗봇 생성"""

    # LLM 모델
    model = ChatOpenAI(model=model_name, temperature=0.7)

    # 대화 응답 노드
    def chatbot_node(state: ChatState) -> dict:
        """일반 대화 응답"""
        # 요약이 있으면 시스템 메시지로 포함
        messages = list(state["messages"])
        if state.get("summary"):
            summary_msg = SystemMessage(
                content=f"이전 대화 요약:\n{state['summary']}\n\n"
                        f"위 요약을 참고하여 대화를 이어가세요."
            )
            messages = [summary_msg] + messages

        response = model.invoke(messages)
        return {"messages": [response]}

    # 요약 필요 여부 판단
    def should_summarize(state: ChatState) -> Literal["summarize", "continue"]:
        """요약이 필요한지 판단"""
        messages = state["messages"]

        # 시스템 메시지 제외하고 카운트
        user_ai_messages = [
            m for m in messages
            if isinstance(m, (HumanMessage, AIMessage))
        ]

        # threshold 초과 시 요약
        if len(user_ai_messages) > summary_threshold:
            return "summarize"
        return "continue"

    # 요약 생성 노드
    def summarize_node(state: ChatState) -> dict:
        """대화 요약 생성"""
        messages = state["messages"]

        # 요약 대상: HumanMessage와 AIMessage만
        conversation = [
            m for m in messages
            if isinstance(m, (HumanMessage, AIMessage))
        ]

        # 기존 요약이 있으면 포함
        summary_prompt = ""
        if state.get("summary"):
            summary_prompt = f"기존 요약:\n{state['summary']}\n\n"

        summary_prompt += """다음 대화 내용을 간결하게 요약해주세요.
핵심 주제, 중요한 정보, 사용자 선호도 등을 포함하세요.

대화 내용:
"""
        for msg in conversation:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            summary_prompt += f"{role}: {msg.content}\n"

        summary_prompt += "\n요약 (3-5 문장):"

        # 요약 생성
        summary_response = model.invoke([HumanMessage(content=summary_prompt)])
        new_summary = summary_response.content

        print(f"\n📝 대화 요약 생성 완료 ({len(conversation)}개 메시지)")
        print(f"요약: {new_summary[:100]}...")

        # 오래된 메시지 삭제 (최근 2개만 유지)
        messages_to_remove = [
            RemoveMessage(id=m.id)
            for m in conversation[:-2]
        ]

        return {
            "summary": new_summary,
            "messages": messages_to_remove
        }

    # 그래프 구축
    graph_builder = StateGraph(ChatState)

    # 노드 추가
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("summarize", summarize_node)

    # 엣지 추가
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        should_summarize,
        {
            "summarize": "summarize",
            "continue": END
        }
    )
    graph_builder.add_edge("summarize", END)

    # 메모리 추가
    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


# ============================================================================
# 테스트 함수
# ============================================================================

def test_auto_summarization():
    """자동 요약 테스트"""
    print("=" * 70)
    print("📚 자동 요약 챗봇 테스트")
    print("=" * 70)

    # 요약 threshold = 4 (메시지 4개마다 요약)
    chatbot = create_summarizing_chatbot(summary_threshold=4)
    config = {"configurable": {"thread_id": "test_session"}}

    # 긴 대화 시뮬레이션
    conversation_turns = [
        "안녕하세요! 저는 파이썬 개발자입니다.",
        "요즘 LangChain을 공부하고 있어요.",
        "특히 Agent 시스템이 흥미롭네요.",
        "메모리 관리가 중요한 것 같아요.",
        "대화가 길어지면 어떻게 처리하나요?",  # 여기서 요약 발생
        "제가 처음에 뭐라고 했죠?",  # 요약본에서 정보 가져와야 함
        "LangChain 말고 다른 것도 공부 중이에요.",
        "FastAPI로 API를 만들고 있어요.",
        "Docker도 배우고 있고요.",  # 두 번째 요약 발생
        "제 직업이 뭐였죠?",  # 요약본에서 정보 가져와야 함
    ]

    print("\n💬 대화 시작...\n")

    for i, user_msg in enumerate(conversation_turns, 1):
        print("=" * 70)
        print(f"👤 Turn {i}: {user_msg}")

        # 상태 전 확인
        state_before = chatbot.get_state(config)
        msg_count_before = len([
            m for m in state_before.values.get("messages", [])
            if isinstance(m, (HumanMessage, AIMessage))
        ])

        # 메시지 전송
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config
        )

        # 응답 출력
        ai_response = result["messages"][-1].content
        print(f"🤖 {ai_response}")

        # 상태 후 확인
        state_after = chatbot.get_state(config)
        msg_count_after = len([
            m for m in state_after.values.get("messages", [])
            if isinstance(m, (HumanMessage, AIMessage))
        ])
        summary = state_after.values.get("summary", "")

        # 메모리 상태 출력
        if msg_count_after < msg_count_before:
            print(f"\n⚠️  메시지 압축: {msg_count_before} → {msg_count_after}")
            print(f"📝 현재 요약: {summary[:150]}...")
        else:
            print(f"\n📊 현재 메시지 수: {msg_count_after}")
            if summary:
                print(f"📝 현재 요약 존재: {len(summary)} 글자")

        print()

    print("=" * 70)
    print("✅ 자동 요약 테스트 완료!")
    print("=" * 70)

    # 최종 상태 확인
    final_state = chatbot.get_state(config)
    print("\n📊 최종 메모리 상태:")
    print(f"  메시지 수: {len(final_state.values['messages'])}")
    print(f"  요약 길이: {len(final_state.values.get('summary', ''))} 글자")

    if final_state.values.get("summary"):
        print(f"\n📝 최종 요약:\n{final_state.values['summary']}")


def test_memory_efficiency():
    """메모리 효율성 비교 테스트"""
    print("\n" + "=" * 70)
    print("🔬 메모리 효율성 비교")
    print("=" * 70)

    # 일반 챗봇 (요약 없음)
    from langgraph.prebuilt import create_react_agent

    model = ChatOpenAI(model="gpt-4o-mini")
    normal_chatbot = create_react_agent(model, [], checkpointer=InMemorySaver())

    # 요약 챗봇
    summary_chatbot = create_summarizing_chatbot(summary_threshold=4)

    # 테스트 대화
    test_messages = [
        f"메시지 {i}: 이것은 테스트 메시지입니다." for i in range(1, 11)
    ]

    # 일반 챗봇 테스트
    config_normal = {"configurable": {"thread_id": "normal"}}
    for msg in test_messages[:8]:  # 8개만
        normal_chatbot.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config_normal
        )

    state_normal = normal_chatbot.get_state(config_normal)
    normal_msg_count = len(state_normal.values["messages"])

    # 요약 챗봇 테스트
    config_summary = {"configurable": {"thread_id": "summary"}}
    for msg in test_messages[:8]:  # 8개
        summary_chatbot.invoke(
            {"messages": [HumanMessage(content=msg)]},
            config_summary
        )

    state_summary = summary_chatbot.get_state(config_summary)
    summary_msg_count = len([
        m for m in state_summary.values["messages"]
        if isinstance(m, (HumanMessage, AIMessage))
    ])

    # 결과 비교
    print(f"\n📊 결과 비교 (8턴 대화 후):")
    print(f"  일반 챗봇 메시지 수: {normal_msg_count}")
    print(f"  요약 챗봇 메시지 수: {summary_msg_count}")
    print(f"  절감률: {(1 - summary_msg_count / normal_msg_count) * 100:.1f}%")

    if state_summary.values.get("summary"):
        print(f"\n  요약본 존재: ✅")
        print(f"  요약 길이: {len(state_summary.values['summary'])} 글자")

    print("\n💡 요약 시스템의 장점:")
    print("  - 메시지 수 감소로 토큰 비용 절감")
    print("  - 컨텍스트 윈도우 효율적 사용")
    print("  - 핵심 정보는 요약본에 보존")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 자동 요약 챗봇 - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /summary - 현재 요약 보기")
    print("  /messages - 현재 메시지 수")
    print("  /reset - 대화 초기화")
    print("  /quit - 종료")
    print("=" * 70)

    chatbot = create_summarizing_chatbot(summary_threshold=4)
    config = {"configurable": {"thread_id": "interactive"}}

    turn = 0
    while True:
        try:
            turn += 1
            user_input = input(f"\n👤 Turn {turn}: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 종료합니다.")
                break

            elif user_input == "/summary":
                state = chatbot.get_state(config)
                summary = state.values.get("summary", "")
                if summary:
                    print(f"\n📝 현재 요약:\n{summary}")
                else:
                    print("📝 아직 요약이 생성되지 않았습니다.")
                turn -= 1
                continue

            elif user_input == "/messages":
                state = chatbot.get_state(config)
                messages = [
                    m for m in state.values.get("messages", [])
                    if isinstance(m, (HumanMessage, AIMessage))
                ]
                print(f"\n📊 현재 메시지 수: {len(messages)}")
                print("📜 메시지 목록:")
                for i, msg in enumerate(messages, 1):
                    role = "👤" if isinstance(msg, HumanMessage) else "🤖"
                    print(f"  {i}. {role} {msg.content[:50]}...")
                turn -= 1
                continue

            elif user_input == "/reset":
                config = {"configurable": {"thread_id": f"interactive_{turn}"}}
                print("🔄 대화가 초기화되었습니다.")
                turn = 0
                continue

            # 일반 메시지
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            ai_response = result["messages"][-1].content
            print(f"🤖 {ai_response}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📚 Part 4: 자동 요약 시스템 - 실습 과제 2 해답")
    print("=" * 70)

    try:
        # 테스트 1: 자동 요약
        test_auto_summarization()

        # 테스트 2: 메모리 효율성
        test_memory_efficiency()

        # 대화형 모드
        print("\n대화형 모드를 실행하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예']:
            interactive_mode()

    except Exception as e:
        print(f"\n⚠️  오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # 학습 포인트
    print("\n" + "=" * 70)
    print("💡 학습 포인트:")
    print("  1. 메시지 수 기반 자동 요약 트리거")
    print("  2. RemoveMessage로 오래된 메시지 정리")
    print("  3. 요약본을 SystemMessage로 제공")
    print("  4. 메모리 효율성 대폭 향상")
    print("\n💡 추가 학습:")
    print("  1. 토큰 수 기반 요약 (tiktoken 사용)")
    print("  2. 점진적 요약 (요약의 요약)")
    print("  3. 엔티티 추출 및 별도 저장")
    print("  4. 사용자별 요약 전략 커스터마이징")
    print("=" * 70)


if __name__ == "__main__":
    main()
