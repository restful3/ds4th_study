"""
[Chapter 9] 도구와 에이전트 - Tool Calling

📝 설명:
    Tool Calling은 LLM이 외부 도구(함수)를 호출하여 정보를 얻거나
    작업을 수행할 수 있게 하는 핵심 기능입니다.
    이것이 AI Agent의 "행동" 능력의 기반입니다.

🎯 학습 목표:
    - @tool 데코레이터를 사용한 도구 정의
    - LLM에 도구 바인딩
    - 도구 호출 감지 및 실행
    - ToolNode 활용

📚 관련 문서:
    - docs/Part3-Agent/09-tools-and-agents.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent

💻 실행 방법:
    python -m src.part3_agent.09_tool_calling

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
    - langchain-core>=0.3.0
"""

import os
from typing import Annotated
from dotenv import load_dotenv
import json

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


# =============================================================================
# 1. 기본 도구 정의 (@tool 데코레이터)
# =============================================================================

@tool
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        두 숫자의 합
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 숫자를 곱합니다.

    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자

    Returns:
        두 숫자의 곱
    """
    return a * b


@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨 정보를 가져옵니다.

    Args:
        city: 날씨를 확인할 도시 이름

    Returns:
        날씨 정보 문자열
    """
    # 실제로는 API를 호출하지만, 여기서는 시뮬레이션
    weather_data = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C",
        "default": "정보 없음"
    }
    return weather_data.get(city, weather_data["default"])


@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색할 내용

    Returns:
        검색 결과 요약
    """
    # 시뮬레이션된 검색 결과
    return f"'{query}'에 대한 검색 결과: 관련 정보를 찾았습니다."


def run_basic_tool_example():
    """기본 도구 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 도구 정의 및 직접 호출")
    print("=" * 60)

    # 도구 직접 호출
    print("\n📌 도구 직접 호출:")
    print(f"   add(3, 5) = {add.invoke({'a': 3, 'b': 5})}")
    print(f"   multiply(4, 7) = {multiply.invoke({'a': 4, 'b': 7})}")
    print(f"   get_weather('서울') = {get_weather.invoke({'city': '서울'})}")

    # 도구 메타데이터 확인
    print("\n📌 도구 메타데이터:")
    print(f"   이름: {add.name}")
    print(f"   설명: {add.description}")
    print(f"   스키마: {add.args_schema.schema()}")


# =============================================================================
# 2. LLM에 도구 바인딩
# =============================================================================

def create_tool_bound_llm():
    """도구가 바인딩된 LLM 생성"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    # 도구 바인딩
    tools = [add, multiply, get_weather, search_web]
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools


def run_tool_binding_example():
    """도구 바인딩 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: LLM에 도구 바인딩")
    print("=" * 60)

    load_dotenv()
    llm = create_tool_bound_llm()

    if llm is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    # LLM에게 도구가 필요한 질문
    messages = [HumanMessage(content="서울의 날씨가 어때요?")]
    response = llm.invoke(messages)

    print(f"\n📝 질문: '서울의 날씨가 어때요?'")
    print(f"\n🤖 LLM 응답:")
    print(f"   content: {response.content[:100] if response.content else '(도구 호출 중)'}")

    if response.tool_calls:
        print(f"\n🔧 도구 호출 요청:")
        for tc in response.tool_calls:
            print(f"   - 도구: {tc['name']}")
            print(f"     인자: {tc['args']}")
            print(f"     ID: {tc['id']}")


# =============================================================================
# 3. 도구 호출 및 결과 처리
# =============================================================================

def run_tool_execution_example():
    """도구 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 도구 호출 감지 및 실행")
    print("=" * 60)

    load_dotenv()
    llm = create_tool_bound_llm()

    if llm is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    # 도구 목록
    tools = [add, multiply, get_weather, search_web]
    tool_map = {tool.name: tool for tool in tools}

    # 질문 → LLM 응답 → 도구 실행 → 결과 전달
    messages = [HumanMessage(content="3과 7을 더한 다음, 그 결과에 4를 곱해주세요.")]

    print(f"\n📝 질문: {messages[0].content}")

    # 반복 처리 (여러 도구 호출 가능)
    for i in range(3):  # 최대 3번 반복
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print(f"\n🤖 최종 응답:\n   {response.content}")
            break

        print(f"\n🔄 라운드 {i + 1}:")
        for tc in response.tool_calls:
            print(f"   🔧 도구 호출: {tc['name']}({tc['args']})")

            # 도구 실행
            tool_func = tool_map[tc["name"]]
            result = tool_func.invoke(tc["args"])

            print(f"   📤 결과: {result}")

            # 결과를 메시지에 추가
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )


# =============================================================================
# 4. ToolNode 사용 (LangGraph 내장)
# =============================================================================

def create_tool_node_graph():
    """ToolNode를 사용하는 그래프 생성"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    # 도구 정의
    tools = [add, multiply, get_weather]

    # LLM with tools
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # ToolNode 생성 - 자동으로 도구 호출 처리
    tool_node = ToolNode(tools)

    def call_llm(state: MessagesState) -> MessagesState:
        """LLM 호출"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        """도구 호출이 있으면 tools로, 없으면 end로"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # 그래프 구성
    graph = StateGraph(MessagesState)

    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "llm")  # 도구 실행 후 다시 LLM으로

    return graph.compile()


def run_tool_node_example():
    """ToolNode 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: ToolNode 사용")
    print("=" * 60)

    app = create_tool_node_graph()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    result = app.invoke({
        "messages": [HumanMessage(content="5와 3을 더하고, 서울 날씨도 알려주세요.")]
    })

    print(f"\n📨 대화 기록:")
    for msg in result["messages"]:
        role = type(msg).__name__.replace("Message", "")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"\n   [{role}] 도구 호출:")
            for tc in msg.tool_calls:
                print(f"      - {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"   [{role}] 결과: {msg.content}")
        else:
            content = msg.content[:200] if msg.content else "(내용 없음)"
            print(f"\n   [{role}] {content}")


# =============================================================================
# 5. 커스텀 도구 정의 (Pydantic 스키마)
# =============================================================================

from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    """검색 입력 스키마"""
    query: str = Field(description="검색할 내용")
    max_results: int = Field(default=5, description="최대 결과 수")
    language: str = Field(default="ko", description="결과 언어")


@tool(args_schema=SearchInput)
def advanced_search(query: str, max_results: int = 5, language: str = "ko") -> str:
    """고급 웹 검색을 수행합니다.

    Args:
        query: 검색할 내용
        max_results: 최대 결과 수
        language: 결과 언어

    Returns:
        검색 결과 요약
    """
    return f"'{query}' 검색 결과 (언어: {language}, 최대: {max_results}개): 관련 정보 발견"


def run_custom_tool_example():
    """커스텀 도구 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 5: Pydantic 스키마를 사용한 커스텀 도구")
    print("=" * 60)

    print("\n📌 도구 스키마:")
    print(f"   {json.dumps(advanced_search.args_schema.schema(), indent=2, ensure_ascii=False)}")

    result = advanced_search.invoke({
        "query": "LangGraph 튜토리얼",
        "max_results": 10,
        "language": "ko"
    })
    print(f"\n📤 실행 결과: {result}")


# =============================================================================
# 6. 도구 에러 핸들링
# =============================================================================

@tool
def divide(a: int, b: int) -> str:
    """두 숫자를 나눕니다.

    Args:
        a: 피제수
        b: 제수

    Returns:
        나눗셈 결과
    """
    try:
        if b == 0:
            return "오류: 0으로 나눌 수 없습니다."
        return str(a / b)
    except Exception as e:
        return f"오류: {str(e)}"


def run_error_handling_example():
    """에러 핸들링 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 6: 도구 에러 핸들링")
    print("=" * 60)

    print("\n📌 정상 실행:")
    result = divide.invoke({"a": 10, "b": 3})
    print(f"   divide(10, 3) = {result}")

    print("\n📌 에러 상황:")
    result = divide.invoke({"a": 10, "b": 0})
    print(f"   divide(10, 0) = {result}")


# =============================================================================
# 7. Tool Calling 개념 정리
# =============================================================================

def explain_tool_calling():
    """Tool Calling 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Tool Calling 개념 정리")
    print("=" * 60)

    print("""
Tool Calling 흐름:

1. 도구 정의
   @tool
   def my_tool(arg: str) -> str:
       '''도구 설명'''
       return result

2. LLM에 바인딩
   llm_with_tools = llm.bind_tools([tool1, tool2])

3. LLM 호출 → 도구 호출 요청 생성
   response = llm_with_tools.invoke(messages)
   # response.tool_calls에 호출 정보 포함

4. 도구 실행
   result = tool.invoke(args)

5. 결과를 ToolMessage로 전달
   ToolMessage(content=result, tool_call_id=tc["id"])

6. LLM이 결과를 바탕으로 응답 생성

핵심 포인트:

- LLM은 도구를 '직접' 실행하지 않음
- LLM은 도구 호출 '요청'만 생성
- 실제 실행은 우리 코드에서 수행
- 결과를 다시 LLM에게 전달

LangGraph에서의 활용:

- ToolNode: 도구 실행을 자동화하는 노드
- 조건부 엣지: 도구 호출 여부에 따라 분기
- Agent 루프: LLM → 도구 → LLM → ... → END
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 9] 도구와 에이전트 - Tool Calling")
    print("=" * 60)

    load_dotenv()

    # 예제 실행
    run_basic_tool_example()
    run_tool_binding_example()
    run_tool_execution_example()
    run_tool_node_example()
    run_custom_tool_example()
    run_error_handling_example()

    # 개념 정리
    explain_tool_calling()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 10_react_agent.py (ReAct Agent)")
    print("=" * 60)


if __name__ == "__main__":
    main()
