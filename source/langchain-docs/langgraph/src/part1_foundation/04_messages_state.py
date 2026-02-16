"""
[Chapter 4] State 관리 심화 - MessagesState

📝 설명:
    LangGraph에서 대화형 애플리케이션을 구축할 때 가장 많이 사용하는
    MessagesState에 대해 학습합니다. MessagesState는 add_messages reducer를
    내장하여 메시지 기록을 자동으로 관리합니다.

🎯 학습 목표:
    - MessagesState의 구조와 사용법 이해
    - add_messages reducer의 동작 방식 학습
    - LangChain 메시지 타입 이해 (HumanMessage, AIMessage, SystemMessage)
    - 대화 기록 관리 방법 습득
    - RemoveMessage를 사용한 메시지 삭제

📚 관련 문서:
    - docs/Part1-Foundation/04-state-management.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate

💻 실행 방법:
    python -m src.part1_foundation.04_messages_state

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-core>=0.3.0
"""

import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages, RemoveMessage

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)


# =============================================================================
# 1. 기본 MessagesState 사용
# =============================================================================

def simple_echo(state: MessagesState) -> MessagesState:
    """마지막 메시지를 에코하는 간단한 노드"""
    last_message = state["messages"][-1]
    echo_response = AIMessage(content=f"에코: {last_message.content}")
    return {"messages": [echo_response]}


def run_basic_messages_example():
    """기본 MessagesState 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 MessagesState")
    print("=" * 60)

    # MessagesState를 직접 사용
    # MessagesState는 다음과 같이 정의되어 있음:
    # class MessagesState(TypedDict):
    #     messages: Annotated[List[BaseMessage], add_messages]

    graph = StateGraph(MessagesState)
    graph.add_node("echo", simple_echo)
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    app = graph.compile()

    # 실행
    result = app.invoke({
        "messages": [HumanMessage(content="안녕하세요!")]
    })

    print(f"\n📨 메시지 기록:")
    for i, msg in enumerate(result["messages"], 1):
        role = type(msg).__name__.replace("Message", "")
        print(f"   {i}. [{role}] {msg.content}")


# =============================================================================
# 2. add_messages reducer 동작 이해
# =============================================================================

class CustomMessagesState(TypedDict):
    """직접 정의한 MessagesState"""
    # add_messages reducer를 명시적으로 지정
    messages: Annotated[List[BaseMessage], add_messages]
    # 추가 필드도 정의 가능
    conversation_id: str


def process_message(state: CustomMessagesState) -> CustomMessagesState:
    """메시지 처리 노드"""
    last_msg = state["messages"][-1].content
    response = AIMessage(content=f"'{last_msg}'에 대한 응답입니다.")
    # add_messages reducer 덕분에 기존 메시지에 추가됨
    return {"messages": [response]}


def run_custom_messages_state_example():
    """커스텀 MessagesState 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 커스텀 MessagesState (추가 필드 포함)")
    print("=" * 60)

    graph = StateGraph(CustomMessagesState)
    graph.add_node("process", process_message)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    initial_state = {
        "messages": [
            SystemMessage(content="당신은 친절한 어시스턴트입니다."),
            HumanMessage(content="LangGraph란 무엇인가요?")
        ],
        "conversation_id": "conv_001"
    }

    result = app.invoke(initial_state)

    print(f"\n🆔 대화 ID: {result['conversation_id']}")
    print(f"\n📨 메시지 기록:")
    for i, msg in enumerate(result["messages"], 1):
        role = type(msg).__name__.replace("Message", "")
        print(f"   {i}. [{role}] {msg.content[:50]}...")


# =============================================================================
# 3. 메시지 타입 이해
# =============================================================================

def explain_message_types():
    """메시지 타입 설명"""
    print("\n" + "=" * 60)
    print("예제 3: LangChain 메시지 타입")
    print("=" * 60)

    print("""
📝 메시지 타입:

1. SystemMessage
   - 시스템 프롬프트/지시사항
   - LLM의 행동 방식을 정의
   - 예: "당신은 전문 프로그래머입니다"

2. HumanMessage
   - 사용자 입력
   - 질문, 요청, 명령 등
   - 예: "파이썬으로 피보나치 함수를 작성해주세요"

3. AIMessage
   - AI/LLM의 응답
   - tool_calls 속성을 통해 도구 호출 정보 포함 가능
   - 예: "다음은 피보나치 함수입니다..."

4. ToolMessage
   - 도구 실행 결과
   - tool_call_id로 어떤 도구 호출에 대한 응답인지 연결
   - 예: 계산기 도구의 실행 결과
""")

    # 메시지 타입 데모
    messages = [
        SystemMessage(content="당신은 수학 선생님입니다."),
        HumanMessage(content="1 + 1은?"),
        AIMessage(content="1 + 1 = 2입니다!"),
    ]

    print("📌 메시지 예시:")
    for msg in messages:
        print(f"   {type(msg).__name__}: {msg.content}")


# =============================================================================
# 4. 다중 턴 대화 구현
# =============================================================================

def create_multi_turn_graph():
    """다중 턴 대화 그래프 생성"""

    def respond(state: MessagesState) -> MessagesState:
        """간단한 응답 생성"""
        messages = state["messages"]
        last_human_msg = None

        # 마지막 사용자 메시지 찾기
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break

        # 간단한 응답 로직
        if not last_human_msg:
            response = "무엇을 도와드릴까요?"
        elif "안녕" in last_human_msg:
            response = "안녕하세요! 무엇을 도와드릴까요?"
        elif "이름" in last_human_msg:
            response = "저는 LangGraph 봇입니다!"
        elif "고마워" in last_human_msg or "감사" in last_human_msg:
            response = "도움이 되어 기쁩니다!"
        else:
            response = f"'{last_human_msg}'에 대해 더 자세히 알려주세요."

        return {"messages": [AIMessage(content=response)]}

    graph = StateGraph(MessagesState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_multi_turn_example():
    """다중 턴 대화 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 다중 턴 대화")
    print("=" * 60)

    app = create_multi_turn_graph()

    # 대화 시뮬레이션
    conversation = []

    # 턴 1
    conversation.append(HumanMessage(content="안녕하세요!"))
    result = app.invoke({"messages": conversation})
    conversation = result["messages"]

    # 턴 2
    conversation.append(HumanMessage(content="너의 이름이 뭐야?"))
    result = app.invoke({"messages": conversation})
    conversation = result["messages"]

    # 턴 3
    conversation.append(HumanMessage(content="고마워!"))
    result = app.invoke({"messages": conversation})
    conversation = result["messages"]

    print(f"\n📨 전체 대화 기록:")
    for i, msg in enumerate(conversation, 1):
        role = "👤 사용자" if isinstance(msg, HumanMessage) else "🤖 봇"
        print(f"   {i}. {role}: {msg.content}")


# =============================================================================
# 5. LLM과 연동한 대화
# =============================================================================

def create_llm_chat_graph():
    """LLM을 사용한 대화 그래프 생성"""

    # API 키 확인
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

    def chat(state: MessagesState) -> MessagesState:
        """LLM을 사용하여 응답"""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    return graph.compile()


def run_llm_chat_example():
    """LLM 대화 예제"""
    print("\n" + "=" * 60)
    print("예제 5: LLM 연동 대화")
    print("=" * 60)

    load_dotenv()
    app = create_llm_chat_graph()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        print("   ANTHROPIC_API_KEY를 설정하고 langchain-anthropic을 설치하세요.")
        return

    messages = [
        SystemMessage(content="당신은 친절한 한국어 AI 어시스턴트입니다."),
        HumanMessage(content="LangGraph의 장점을 3가지만 알려주세요.")
    ]

    result = app.invoke({"messages": messages})

    print(f"\n📨 대화 결과:")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n🤖 AI: {msg.content}")


# =============================================================================
# 6. 메시지 삭제 (RemoveMessage)
# =============================================================================

def run_remove_message_example():
    """메시지 삭제 예제"""
    print("\n" + "=" * 60)
    print("예제 6: RemoveMessage로 메시지 삭제")
    print("=" * 60)

    # 메시지에 ID 부여
    msg1 = HumanMessage(content="첫 번째 메시지", id="msg1")
    msg2 = AIMessage(content="첫 번째 응답", id="msg2")
    msg3 = HumanMessage(content="두 번째 메시지", id="msg3")
    msg4 = AIMessage(content="두 번째 응답", id="msg4")

    def remove_first_exchange(state: MessagesState) -> MessagesState:
        """첫 번째 대화 교환을 삭제"""
        return {
            "messages": [
                RemoveMessage(id="msg1"),
                RemoveMessage(id="msg2"),
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("remove", remove_first_exchange)
    graph.add_edge(START, "remove")
    graph.add_edge("remove", END)
    app = graph.compile()

    initial = {"messages": [msg1, msg2, msg3, msg4]}
    result = app.invoke(initial)

    print(f"\n📥 삭제 전 메시지 수: {len(initial['messages'])}")
    print(f"📤 삭제 후 메시지 수: {len(result['messages'])}")

    print(f"\n📨 남은 메시지:")
    for msg in result["messages"]:
        role = type(msg).__name__.replace("Message", "")
        print(f"   [{role}] {msg.content}")


# =============================================================================
# 7. MessagesState 확장 패턴
# =============================================================================

class ExtendedMessagesState(MessagesState):
    """MessagesState를 확장한 State"""
    # MessagesState의 messages 필드를 상속
    # 추가 필드 정의
    user_name: str
    turn_count: int
    is_active: bool


def process_extended(state: ExtendedMessagesState) -> ExtendedMessagesState:
    """확장된 State 처리"""
    name = state.get("user_name", "사용자")
    count = state.get("turn_count", 0) + 1

    response = AIMessage(
        content=f"{name}님, {count}번째 대화입니다!"
    )

    return {
        "messages": [response],
        "turn_count": count
    }


def run_extended_state_example():
    """확장된 MessagesState 예제"""
    print("\n" + "=" * 60)
    print("예제 7: MessagesState 확장")
    print("=" * 60)

    graph = StateGraph(ExtendedMessagesState)
    graph.add_node("process", process_extended)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    # 첫 번째 호출
    result1 = app.invoke({
        "messages": [HumanMessage(content="안녕!")],
        "user_name": "홍길동",
        "turn_count": 0,
        "is_active": True
    })

    # 두 번째 호출 (상태 이어받기)
    result1["messages"].append(HumanMessage(content="두 번째!"))
    result2 = app.invoke(result1)

    print(f"\n📨 대화 기록:")
    for msg in result2["messages"]:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"   {role} {msg.content}")
    print(f"\n   턴 카운트: {result2['turn_count']}")


# =============================================================================
# 8. add_messages reducer 심화
# =============================================================================

def explain_add_messages_behavior():
    """add_messages reducer의 동작 설명"""
    print("\n" + "=" * 60)
    print("📘 add_messages reducer 동작 방식")
    print("=" * 60)

    print("""
add_messages reducer의 특별한 동작:

1. 기본 동작: 메시지 추가
   - 새 메시지를 기존 목록 끝에 추가
   - operator.add와 유사하지만 메시지 전용 최적화

2. 메시지 ID 기반 업데이트
   - 같은 ID의 메시지가 있으면 '대체'
   - ID가 없는 메시지는 항상 '추가'

3. RemoveMessage 처리
   - RemoveMessage(id="xxx")를 반환하면 해당 메시지 삭제
   - 메모리 관리에 유용

4. 사용 예:
   # 추가
   return {"messages": [AIMessage(content="응답")]}

   # 대체 (같은 ID)
   return {"messages": [AIMessage(content="새 응답", id="existing_id")]}

   # 삭제
   return {"messages": [RemoveMessage(id="msg_to_delete")]}

5. 왜 중요한가?
   - 대화 기록 자동 관리
   - 메모리 최적화 (오래된 메시지 삭제)
   - 메시지 수정/업데이트 지원
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 4] State 관리 심화 - MessagesState")
    print("=" * 60)

    # 환경 변수 로드
    load_dotenv()

    # 예제 실행
    run_basic_messages_example()
    run_custom_messages_state_example()
    explain_message_types()
    run_multi_turn_example()
    run_llm_chat_example()
    run_remove_message_example()
    run_extended_state_example()

    # 개념 설명
    explain_add_messages_behavior()

    print("\n" + "=" * 60)
    print("✅ Part 1 완료!")
    print("   다음: Part 2 - 워크플로우 패턴")
    print("=" * 60)


if __name__ == "__main__":
    main()
