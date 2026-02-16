"""
연구 어시스턴트 (Research Assistant)

이 예제는 도구(Tool)를 사용하는 ReAct 스타일의 연구 어시스턴트를 구현합니다.
웹 검색, 계산, 메모 작성 등의 도구를 활용하여 사용자의 질문에 답변합니다.

기능:
- 웹 검색 (시뮬레이션)
- 계산기
- 메모 저장/조회
- ReAct 추론 루프

실행 방법:
    python -m examples.02_research_assistant.main
"""

import os
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


# =============================================================================
# 환경 설정
# =============================================================================

load_dotenv()

# 메모 저장소 (간단한 메모리 저장)
NOTES_STORAGE = {}


# =============================================================================
# 도구 정의
# =============================================================================

@tool
def web_search(query: str) -> str:
    """
    웹에서 정보를 검색합니다.

    Args:
        query: 검색할 내용

    Returns:
        검색 결과 요약
    """
    # 실제 구현에서는 Google Search API 등을 사용
    # 여기서는 시뮬레이션
    simulated_results = {
        "langgraph": "LangGraph는 LangChain에서 만든 상태 유지 AI 에이전트 프레임워크입니다. 순환 그래프 구조를 지원합니다.",
        "python": "Python은 1991년 귀도 반 로섬이 만든 프로그래밍 언어입니다. 현재 가장 인기 있는 언어 중 하나입니다.",
        "ai agent": "AI Agent는 LLM을 사용하여 자율적으로 작업을 수행하는 시스템입니다. 도구를 사용하고 추론을 수행합니다.",
    }

    query_lower = query.lower()
    for key, value in simulated_results.items():
        if key in query_lower:
            return f"검색 결과 ({query}):\n{value}"

    return f"검색 결과 ({query}):\n관련 정보를 찾지 못했습니다. 다른 키워드로 검색해보세요."


@tool
def calculator(expression: str) -> str:
    """
    수학 계산을 수행합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")

    Returns:
        계산 결과
    """
    try:
        # 안전한 계산을 위해 제한된 함수만 허용
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"계산 결과: {expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def save_note(title: str, content: str) -> str:
    """
    메모를 저장합니다.

    Args:
        title: 메모 제목
        content: 메모 내용

    Returns:
        저장 결과
    """
    NOTES_STORAGE[title] = {
        "content": content,
        "created_at": datetime.now().isoformat()
    }
    return f"메모 '{title}'가 저장되었습니다."


@tool
def get_notes() -> str:
    """
    저장된 모든 메모를 조회합니다.

    Returns:
        저장된 메모 목록
    """
    if not NOTES_STORAGE:
        return "저장된 메모가 없습니다."

    notes_list = []
    for title, data in NOTES_STORAGE.items():
        notes_list.append(f"- {title}: {data['content'][:50]}...")

    return "저장된 메모:\n" + "\n".join(notes_list)


@tool
def get_current_time() -> str:
    """
    현재 시간을 반환합니다.

    Returns:
        현재 시간
    """
    return f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# =============================================================================
# 연구 어시스턴트 그래프
# =============================================================================

# 도구 목록
TOOLS = [web_search, calculator, save_note, get_notes, get_current_time]


def create_research_assistant():
    """연구 어시스턴트 그래프 생성"""

    def agent_node(state: MessagesState) -> MessagesState:
        """에이전트 노드 - 추론 및 도구 호출 결정"""
        messages = state["messages"]

        # LLM이 있으면 사용
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic

                llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
                llm_with_tools = llm.bind_tools(TOOLS)

                system_msg = SystemMessage(content="""당신은 연구 어시스턴트입니다.
                사용자의 질문에 답하기 위해 필요한 경우 도구를 사용하세요.

                사용 가능한 도구:
                - web_search: 웹에서 정보 검색
                - calculator: 수학 계산
                - save_note: 메모 저장
                - get_notes: 저장된 메모 조회
                - get_current_time: 현재 시간 확인

                도구 사용이 필요 없는 경우 직접 답변하세요.""")

                response = llm_with_tools.invoke([system_msg] + messages)
                return {"messages": [response]}

            except ImportError:
                pass

        # LLM이 없는 경우 시뮬레이션
        last_msg = messages[-1].content if messages else ""

        # 간단한 규칙 기반 시뮬레이션
        if "검색" in last_msg or "찾아" in last_msg:
            return {"messages": [AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "web_search",
                    "args": {"query": last_msg}
                }]
            )]}
        elif "계산" in last_msg or "+" in last_msg or "*" in last_msg:
            # 숫자와 연산자 추출 시도
            import re
            expr = re.findall(r'[\d\+\-\*\/\(\)\.\s]+', last_msg)
            if expr:
                return {"messages": [AIMessage(
                    content="",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "calculator",
                        "args": {"expression": expr[0].strip()}
                    }]
                )]}
        elif "메모" in last_msg and ("저장" in last_msg or "작성" in last_msg):
            return {"messages": [AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "save_note",
                    "args": {"title": "메모", "content": last_msg}
                }]
            )]}
        elif "시간" in last_msg:
            return {"messages": [AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "get_current_time",
                    "args": {}
                }]
            )]}

        # 기본 응답
        return {"messages": [AIMessage(content=f"'{last_msg}'에 대해 답변드립니다. (시뮬레이션 모드)")]}

    def should_continue(state: MessagesState) -> str:
        """도구 호출이 필요한지 확인"""
        messages = state["messages"]
        last_msg = messages[-1]

        # 도구 호출이 있으면 tools 노드로
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
        return "end"

    # 그래프 구성
    graph = StateGraph(MessagesState)

    # 노드 추가
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    # 엣지 추가
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")  # 도구 실행 후 에이전트로 돌아감

    # 컴파일
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 인터랙티브 세션
# =============================================================================

def run_interactive_session():
    """인터랙티브 연구 세션"""

    print("=" * 60)
    print("🔬 연구 어시스턴트")
    print("=" * 60)
    print("\n사용 가능한 기능:")
    print("  - 웹 검색: 'LangGraph에 대해 검색해줘'")
    print("  - 계산기: '123 + 456 계산해줘'")
    print("  - 메모: '이 내용을 메모해줘'")
    print("  - 시간: '현재 시간 알려줘'")
    print("\n명령어: /quit - 종료")
    print("-" * 60)

    assistant = create_research_assistant()
    config = {"configurable": {"thread_id": "research_session_1"}}

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("\n👋 연구 세션을 종료합니다.")
                break

            # 질문 처리
            result = assistant.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )

            # 응답 출력
            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 Assistant: {msg.content}")
                elif isinstance(msg, ToolMessage):
                    print(f"\n🔧 Tool Result: {msg.content}")

        except KeyboardInterrupt:
            print("\n\n👋 연구 세션을 종료합니다.")
            break


# =============================================================================
# 데모 실행
# =============================================================================

def run_demo():
    """데모 실행"""

    print("=" * 60)
    print("🔬 Research Assistant Demo")
    print("=" * 60)

    assistant = create_research_assistant()
    config = {"configurable": {"thread_id": "demo_research"}}

    # 테스트 케이스
    test_cases = [
        "현재 시간을 알려줘",
        "LangGraph에 대해 검색해줘",
        "123 * 456 + 789 계산해줘",
        "오늘 배운 내용: LangGraph는 AI Agent 프레임워크다 - 이걸 메모해줘",
        "저장된 메모를 보여줘",
    ]

    for query in test_cases:
        print(f"\n{'='*50}")
        print(f"👤 You: {query}")

        result = assistant.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        # 최종 AI 응답 출력
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"🤖 Assistant: {msg.content}")
                break
            elif isinstance(msg, ToolMessage):
                print(f"🔧 Tool: {msg.content}")


# =============================================================================
# 메인
# =============================================================================

def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive_session()
    else:
        run_demo()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
