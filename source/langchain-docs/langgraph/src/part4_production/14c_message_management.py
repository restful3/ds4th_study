"""
[Chapter 14c] 메시지 관리 (Message Management)

📝 설명:
    긴 대화에서 메시지를 효율적으로 관리하는 방법을 학습합니다.
    토큰 제한을 다루고, 메시지를 요약하거나 잘라내는 전략을 배웁니다.

🎯 학습 목표:
    - trim_messages로 메시지 개수/토큰 제한
    - RemoveMessage로 특정 메시지 삭제
    - 대화 요약을 통한 컨텍스트 압축
    - 슬라이딩 윈도우 패턴

📚 관련 문서:
    - docs/Part4-Production/14-memory.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/

💻 실행 방법:
    python -m src.part4_production.14c_message_management

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
    HumanMessage, AIMessage, SystemMessage, trim_messages, BaseMessage
)


# =============================================================================
# 1. 메시지 관리 개념 설명
# =============================================================================

def explain_message_management():
    """메시지 관리 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 메시지 관리 (Message Management)")
    print("=" * 60)

    print("""
메시지 관리가 필요한 이유:
    - LLM은 토큰 제한이 있음 (예: 128K, 200K)
    - 긴 대화는 컨텍스트 윈도우를 초과할 수 있음
    - 불필요한 메시지는 비용과 품질에 영향

관리 전략:

┌─────────────────┬────────────────────────────────────┐
│     전략        │              설명                  │
├─────────────────┼────────────────────────────────────┤
│ trim_messages   │ 토큰/개수 기준으로 잘라내기        │
│ RemoveMessage   │ 특정 메시지 ID로 삭제              │
│ Summarization   │ 오래된 대화를 요약으로 대체        │
│ Sliding Window  │ 최근 N개만 유지                    │
└─────────────────┴────────────────────────────────────┘

각 전략의 특징:

1. trim_messages
   - 가장 간단한 방법
   - 토큰 수 또는 메시지 개수 기준
   - System 메시지 유지 옵션

2. RemoveMessage
   - 세밀한 제어 가능
   - 메시지 ID 기반 삭제
   - Checkpointer와 연동

3. Summarization
   - 정보 손실 최소화
   - LLM 호출 필요 (비용 발생)
   - 가장 정교한 방법

4. Sliding Window
   - 구현이 간단
   - 최근 대화만 유지
   - 오래된 컨텍스트 손실
""")


# =============================================================================
# 2. trim_messages 사용하기
# =============================================================================

def run_trim_messages_example():
    """trim_messages 예제"""
    print("\n" + "=" * 60)
    print("예제 1: trim_messages로 메시지 잘라내기")
    print("=" * 60)

    # 긴 대화 시뮬레이션
    messages = [
        SystemMessage(content="당신은 친절한 AI 비서입니다."),
        HumanMessage(content="안녕하세요!"),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
        HumanMessage(content="오늘 날씨가 어때요?"),
        AIMessage(content="오늘 서울은 맑고 기온은 15도입니다."),
        HumanMessage(content="내일 비 올까요?"),
        AIMessage(content="내일은 오후에 비가 올 예정입니다."),
        HumanMessage(content="우산 챙겨야겠네요."),
        AIMessage(content="네, 우산을 챙기시는 것이 좋겠습니다."),
        HumanMessage(content="고마워요!"),
    ]

    print(f"\n📝 원본 메시지 수: {len(messages)}")

    # 1. 최근 N개 메시지만 유지 (token_counter 없이)
    # max_tokens 대신 간단히 개수로 제한
    def keep_recent_messages(msgs: List[BaseMessage], n: int) -> List[BaseMessage]:
        """최근 n개 메시지 유지 (System 메시지는 항상 포함)"""
        system_msgs = [m for m in msgs if isinstance(m, SystemMessage)]
        other_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
        return system_msgs + other_msgs[-(n-len(system_msgs)):]

    trimmed = keep_recent_messages(messages, 5)

    print(f"\n🔄 최근 5개 유지 후:")
    for msg in trimmed:
        role = type(msg).__name__.replace("Message", "")
        print(f"   [{role}] {msg.content[:40]}...")

    # 2. 토큰 기반 trim (간단한 예시)
    def estimate_tokens(text: str) -> int:
        """간단한 토큰 추정 (실제로는 tokenizer 사용)"""
        return len(text) // 4  # 대략적인 추정

    def trim_by_tokens(msgs: List[BaseMessage], max_tokens: int) -> List[BaseMessage]:
        """토큰 수 기준으로 잘라내기"""
        total_tokens = 0
        result = []

        # 역순으로 순회하며 토큰 누적
        for msg in reversed(msgs):
            tokens = estimate_tokens(msg.content)
            if total_tokens + tokens <= max_tokens:
                result.insert(0, msg)
                total_tokens += tokens
            elif isinstance(msg, SystemMessage):
                # System 메시지는 항상 포함
                result.insert(0, msg)
                total_tokens += tokens

        return result

    trimmed_by_tokens = trim_by_tokens(messages, 100)

    print(f"\n🔄 100 토큰 제한 후:")
    for msg in trimmed_by_tokens:
        role = type(msg).__name__.replace("Message", "")
        print(f"   [{role}] {msg.content[:40]}...")


# =============================================================================
# 3. RemoveMessage 사용하기
# =============================================================================

class ChatState(TypedDict):
    """채팅 State"""
    messages: Annotated[list, add_messages]


def run_remove_message_example():
    """RemoveMessage 예제"""
    print("\n" + "=" * 60)
    print("예제 2: RemoveMessage로 특정 메시지 삭제")
    print("=" * 60)

    # 메시지에 ID 부여
    messages = [
        HumanMessage(content="첫 번째 메시지", id="msg_1"),
        AIMessage(content="첫 번째 응답", id="msg_2"),
        HumanMessage(content="두 번째 메시지", id="msg_3"),
        AIMessage(content="두 번째 응답", id="msg_4"),
        HumanMessage(content="세 번째 메시지", id="msg_5"),
    ]

    print("\n📝 원본 메시지:")
    for msg in messages:
        print(f"   [{msg.id}] {msg.content}")

    # RemoveMessage를 사용하여 특정 메시지 삭제
    def remove_specific_messages(state: ChatState) -> ChatState:
        """특정 메시지 삭제"""
        # msg_1과 msg_2 삭제
        return {
            "messages": [
                RemoveMessage(id="msg_1"),
                RemoveMessage(id="msg_2"),
            ]
        }

    graph = StateGraph(ChatState)
    graph.add_node("remove", remove_specific_messages)
    graph.add_edge(START, "remove")
    graph.add_edge("remove", END)
    app = graph.compile()

    result = app.invoke({"messages": messages})

    print("\n🗑️  삭제 후 메시지:")
    for msg in result["messages"]:
        if hasattr(msg, 'id') and hasattr(msg, 'content'):
            print(f"   [{msg.id}] {msg.content}")


# =============================================================================
# 4. 대화 요약 (Summarization)
# =============================================================================

class SummarizationState(TypedDict):
    """요약을 포함한 State"""
    messages: Annotated[list, add_messages]
    summary: str


def create_summarization_graph():
    """대화 요약 그래프"""

    def should_summarize(state: SummarizationState) -> str:
        """요약이 필요한지 판단"""
        messages = state.get("messages", [])
        # 메시지가 6개 이상이면 요약
        if len(messages) >= 6:
            return "summarize"
        return "respond"

    def summarize_conversation(state: SummarizationState) -> SummarizationState:
        """대화를 요약 (시뮬레이션)"""
        messages = state.get("messages", [])
        existing_summary = state.get("summary", "")

        # 오래된 메시지들을 요약 (실제로는 LLM 사용)
        old_messages = messages[:-4]  # 최근 4개 제외
        new_summary_parts = []

        for msg in old_messages:
            if isinstance(msg, HumanMessage):
                new_summary_parts.append(f"사용자: {msg.content[:20]}...")
            elif isinstance(msg, AIMessage):
                new_summary_parts.append(f"AI: {msg.content[:20]}...")

        new_summary = existing_summary + "\n[요약] " + " → ".join(new_summary_parts)

        # 오래된 메시지 삭제
        remove_messages = [
            RemoveMessage(id=msg.id) for msg in old_messages
            if hasattr(msg, 'id') and msg.id
        ]

        return {
            "summary": new_summary.strip(),
            "messages": remove_messages
        }

    def respond(state: SummarizationState) -> SummarizationState:
        """응답 생성 (시뮬레이션)"""
        messages = state.get("messages", [])
        summary = state.get("summary", "")

        # 요약이 있으면 컨텍스트로 활용
        context = f"[이전 대화 요약: {summary}]" if summary else ""

        last_msg = messages[-1] if messages else None
        if last_msg and isinstance(last_msg, HumanMessage):
            response = f"{context} '{last_msg.content}'에 대한 응답입니다."
            return {"messages": [AIMessage(content=response, id=f"resp_{len(messages)}")]}

        return {}

    graph = StateGraph(SummarizationState)
    graph.add_node("summarize", summarize_conversation)
    graph.add_node("respond", respond)

    graph.add_conditional_edges(
        START,
        should_summarize,
        {"summarize": "summarize", "respond": "respond"}
    )
    graph.add_edge("summarize", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_summarization_example():
    """요약 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 대화 요약으로 컨텍스트 압축")
    print("=" * 60)

    app = create_summarization_graph()

    # 긴 대화 시뮬레이션
    messages = [
        HumanMessage(content="안녕하세요!", id="m1"),
        AIMessage(content="안녕하세요! 반갑습니다.", id="m2"),
        HumanMessage(content="오늘 날씨 어때요?", id="m3"),
        AIMessage(content="오늘은 맑고 따뜻합니다.", id="m4"),
        HumanMessage(content="산책하기 좋겠네요.", id="m5"),
        AIMessage(content="네, 산책하기 좋은 날씨입니다.", id="m6"),
        HumanMessage(content="추천 산책 코스가 있나요?", id="m7"),  # 7번째 메시지
    ]

    result = app.invoke({
        "messages": messages,
        "summary": ""
    })

    print(f"\n📝 요약:")
    print(f"   {result.get('summary', '없음')}")

    print(f"\n💬 남은 메시지 수: {len(result['messages'])}")
    for msg in result["messages"]:
        if hasattr(msg, 'content'):
            role = type(msg).__name__.replace("Message", "")
            print(f"   [{role}] {msg.content[:50]}...")


# =============================================================================
# 5. 슬라이딩 윈도우 패턴
# =============================================================================

class WindowState(TypedDict):
    """윈도우 State"""
    messages: Annotated[list, add_messages]
    window_size: int


def create_sliding_window_graph():
    """슬라이딩 윈도우 그래프"""

    def trim_to_window(state: WindowState) -> WindowState:
        """윈도우 크기로 제한"""
        messages = state.get("messages", [])
        window_size = state.get("window_size", 10)

        if len(messages) <= window_size:
            return {}

        # 오래된 메시지 삭제
        to_remove = messages[:-window_size]
        remove_messages = [
            RemoveMessage(id=msg.id) for msg in to_remove
            if hasattr(msg, 'id') and msg.id
        ]

        return {"messages": remove_messages}

    def respond(state: WindowState) -> WindowState:
        """응답 생성"""
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None

        if last_msg and isinstance(last_msg, HumanMessage):
            return {
                "messages": [
                    AIMessage(
                        content=f"'{last_msg.content}'에 대한 응답 (윈도우 내 {len(messages)}개 메시지 컨텍스트)",
                        id=f"resp_{len(messages)}"
                    )
                ]
            }
        return {}

    graph = StateGraph(WindowState)
    graph.add_node("trim", trim_to_window)
    graph.add_node("respond", respond)

    graph.add_edge(START, "trim")
    graph.add_edge("trim", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_sliding_window_example():
    """슬라이딩 윈도우 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 슬라이딩 윈도우 패턴")
    print("=" * 60)

    app = create_sliding_window_graph()

    # 많은 메시지 생성
    messages = []
    for i in range(12):
        messages.append(HumanMessage(content=f"메시지 {i+1}", id=f"h{i}"))
        messages.append(AIMessage(content=f"응답 {i+1}", id=f"a{i}"))

    print(f"\n📥 입력 메시지 수: {len(messages)}")

    result = app.invoke({
        "messages": messages,
        "window_size": 6  # 최근 6개만 유지
    })

    print(f"\n📤 윈도우 적용 후 메시지 수: {len(result['messages'])}")
    print("\n💬 남은 메시지:")
    for msg in result["messages"][-6:]:
        if hasattr(msg, 'content'):
            print(f"   [{msg.id}] {msg.content}")


# =============================================================================
# 6. 복합 메시지 관리 전략
# =============================================================================

class AdvancedChatState(TypedDict):
    """고급 채팅 State"""
    messages: Annotated[list, add_messages]
    summary: str
    important_facts: list
    window_size: int


def create_advanced_management_graph():
    """복합 메시지 관리 그래프"""

    def extract_important_facts(state: AdvancedChatState) -> AdvancedChatState:
        """중요한 정보 추출"""
        messages = state.get("messages", [])
        facts = state.get("important_facts", [])

        # 간단한 규칙 기반 추출 (실제로는 LLM)
        for msg in messages:
            content = getattr(msg, 'content', '')
            if any(kw in content for kw in ["기억해", "중요", "꼭"]):
                facts.append({
                    "content": content,
                    "type": "user_request"
                })

        return {"important_facts": facts}

    def manage_messages(state: AdvancedChatState) -> AdvancedChatState:
        """메시지 관리 (윈도우 + 요약)"""
        messages = state.get("messages", [])
        window_size = state.get("window_size", 8)
        summary = state.get("summary", "")

        if len(messages) <= window_size:
            return {}

        # 오래된 메시지 요약 후 삭제
        old_messages = messages[:-window_size]
        new_summary_parts = [summary] if summary else []

        for msg in old_messages:
            if isinstance(msg, HumanMessage):
                new_summary_parts.append(f"[User: {msg.content[:30]}]")

        remove_messages = [
            RemoveMessage(id=msg.id) for msg in old_messages
            if hasattr(msg, 'id') and msg.id
        ]

        return {
            "summary": " ".join(new_summary_parts),
            "messages": remove_messages
        }

    def respond(state: AdvancedChatState) -> AdvancedChatState:
        """응답 생성"""
        messages = state.get("messages", [])
        summary = state.get("summary", "")
        facts = state.get("important_facts", [])

        context_parts = []
        if summary:
            context_parts.append(f"[요약: {summary[:50]}...]")
        if facts:
            context_parts.append(f"[중요 정보: {len(facts)}개]")

        context = " ".join(context_parts)

        last_msg = messages[-1] if messages else None
        if last_msg and isinstance(last_msg, HumanMessage):
            return {
                "messages": [
                    AIMessage(
                        content=f"{context} '{last_msg.content}'에 대한 응답",
                        id=f"resp_{len(messages)}"
                    )
                ]
            }
        return {}

    graph = StateGraph(AdvancedChatState)
    graph.add_node("extract", extract_important_facts)
    graph.add_node("manage", manage_messages)
    graph.add_node("respond", respond)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "manage")
    graph.add_edge("manage", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_advanced_management_example():
    """고급 메시지 관리 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 복합 메시지 관리 전략")
    print("=" * 60)

    app = create_advanced_management_graph()

    messages = [
        HumanMessage(content="안녕하세요!", id="m1"),
        AIMessage(content="안녕하세요!", id="m2"),
        HumanMessage(content="이것 꼭 기억해주세요: 내일 10시 회의", id="m3"),
        AIMessage(content="네, 기억하겠습니다.", id="m4"),
        HumanMessage(content="오늘 날씨가 좋네요", id="m5"),
        AIMessage(content="네, 정말 좋은 날씨입니다.", id="m6"),
        HumanMessage(content="점심 뭐 먹을까요?", id="m7"),
        AIMessage(content="샐러드 어떠세요?", id="m8"),
        HumanMessage(content="좋아요, 샐러드로 할게요", id="m9"),
        AIMessage(content="좋은 선택이에요!", id="m10"),
        HumanMessage(content="내일 회의 시간이 언제였죠?", id="m11"),
    ]

    result = app.invoke({
        "messages": messages,
        "summary": "",
        "important_facts": [],
        "window_size": 6
    })

    print(f"\n📊 결과:")
    print(f"   요약: {result.get('summary', '없음')[:80]}...")
    print(f"   중요 정보: {len(result.get('important_facts', []))}개")
    print(f"   남은 메시지: {len(result['messages'])}개")


# =============================================================================
# 7. 메시지 관리 패턴 정리
# =============================================================================

def explain_management_patterns():
    """메시지 관리 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 메시지 관리 패턴 정리")
    print("=" * 60)

    print("""
메시지 관리 전략 선택 가이드:

1. 단순한 경우: trim_messages 또는 슬라이딩 윈도우
   - 구현 쉬움
   - 추가 LLM 호출 없음
   - 컨텍스트 손실 가능

2. 정보 보존 필요: 요약 (Summarization)
   - 중요 정보 유지
   - LLM 비용 발생
   - 구현 복잡

3. 세밀한 제어: RemoveMessage
   - 특정 메시지 선택적 삭제
   - Checkpointer와 연동
   - ID 관리 필요

복합 전략 구현 팁:

1. 계층적 접근
   - 중요 정보 먼저 추출
   - 나머지 요약 또는 삭제

2. 메타데이터 활용
   - 메시지에 중요도 태그
   - 태그 기반 필터링

3. 적응적 윈도우
   - 대화 복잡도에 따라 윈도우 조정
   - 중요한 대화는 더 많이 유지

4. 비동기 요약
   - 백그라운드에서 요약 수행
   - 응답 지연 최소화
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 14c] 메시지 관리 (Message Management)")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_message_management()

    # 예제 실행
    run_trim_messages_example()
    run_remove_message_example()
    run_summarization_example()
    run_sliding_window_example()
    run_advanced_management_example()

    # 패턴 정리
    explain_management_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 15_streaming.py (스트리밍)")
    print("=" * 60)


if __name__ == "__main__":
    main()
