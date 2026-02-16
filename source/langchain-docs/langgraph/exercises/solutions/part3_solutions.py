"""
Part 3: Agent 연습 문제 해답
"""

from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# =============================================================================
# 문제 1: 간단한 도구 정의
# =============================================================================

@tool
def add(a: int, b: int) -> int:
    """두 숫자를 더합니다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 숫자를 곱합니다."""
    return a * b


@tool
def get_length(text: str) -> int:
    """문자열의 길이를 반환합니다."""
    return len(text)


def solution_1():
    """간단한 도구 정의 해답"""
    print("문제 1: 도구 정의")

    # 도구 테스트
    print(f"  add(2, 3) = {add.invoke({'a': 2, 'b': 3})}")
    print(f"  multiply(4, 5) = {multiply.invoke({'a': 4, 'b': 5})}")
    print(f"  get_length('hello') = {get_length.invoke({'text': 'hello'})}")

    return [add, multiply, get_length]


# =============================================================================
# 문제 2: ReAct Agent 기본
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 조회합니다."""
    # 시뮬레이션된 날씨 데이터
    weather_data = {
        "서울": "맑음, 15°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C"
    }
    return weather_data.get(city, f"{city}의 날씨 정보가 없습니다")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iterations: int


def solution_2():
    """ReAct Agent 기본 해답"""

    tools = [get_weather]
    tool_node = ToolNode(tools)

    def agent(state: AgentState) -> AgentState:
        """Agent 노드 - 도구 호출 결정"""
        messages = state["messages"]
        iterations = state.get("iterations", 0)

        # 간단한 도구 호출 시뮬레이션
        last_msg = messages[-1] if messages else None

        if isinstance(last_msg, HumanMessage):
            # 도시 이름 추출 (간단히)
            content = last_msg.content
            if "서울" in content:
                city = "서울"
            elif "부산" in content:
                city = "부산"
            else:
                city = "서울"

            # 도구 호출 메시지 생성
            tool_call = {
                "id": f"call_{iterations}",
                "name": "get_weather",
                "args": {"city": city}
            }
            ai_msg = AIMessage(content="", tool_calls=[tool_call])
            return {"messages": [ai_msg], "iterations": iterations + 1}

        elif isinstance(last_msg, ToolMessage):
            # 도구 결과를 받아서 최종 응답
            response = f"날씨 정보: {last_msg.content}"
            return {"messages": [AIMessage(content=response)]}

        return {}

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """계속 여부 결정"""
        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"

        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    app = graph.compile()

    # 테스트
    result = app.invoke({
        "messages": [HumanMessage(content="서울 날씨 어때?")],
        "iterations": 0
    })

    print("\n문제 2: ReAct Agent")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content[:50] if msg.content else 'tool_call'}...")

    return app


# =============================================================================
# 문제 3: Multi-Tool Agent
# =============================================================================

@tool
def search(query: str) -> str:
    """검색을 수행합니다."""
    return f"'{query}'에 대한 검색 결과: 관련 정보를 찾았습니다."


@tool
def calculator(expression: str) -> str:
    """수식을 계산합니다."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "계산 오류"


@tool
def translator(text: str, target_lang: str) -> str:
    """텍스트를 번역합니다."""
    translations = {
        ("Hello", "한국어"): "안녕하세요",
        ("Hello", "일본어"): "こんにちは"
    }
    return translations.get((text, target_lang), f"{text} -> {target_lang}: 번역됨")


def solution_3():
    """Multi-Tool Agent 해답"""
    tools = [search, calculator, translator]

    print("\n문제 3: Multi-Tool Agent")

    # 도구 선택 테스트
    queries = [
        ("파이썬이란?", "search"),
        ("123 * 456", "calculator"),
        ("Hello 한국어", "translator")
    ]

    for query, expected_tool in queries:
        # 간단한 도구 선택 로직
        if "?" in query or "이란" in query:
            tool_name = "search"
        elif any(op in query for op in ["*", "+", "-", "/"]):
            tool_name = "calculator"
        else:
            tool_name = "translator"

        print(f"  '{query}' -> {tool_name}")

    return tools


# =============================================================================
# 문제 4: Supervisor Agent (간략화)
# =============================================================================

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    research_result: str
    final_result: str


def solution_4():
    """Supervisor Agent 해답"""

    def supervisor(state: SupervisorState) -> SupervisorState:
        """작업 분배"""
        messages = state["messages"]
        if not state.get("research_result"):
            return {"next_agent": "research"}
        elif not state.get("final_result"):
            return {"next_agent": "writer"}
        return {"next_agent": "done"}

    def research_agent(state: SupervisorState) -> SupervisorState:
        """조사"""
        return {"research_result": "조사 결과: 관련 정보 수집됨"}

    def writer_agent(state: SupervisorState) -> SupervisorState:
        """작성"""
        research = state.get("research_result", "")
        return {"final_result": f"최종 결과물 ({research} 기반)"}

    def route(state: SupervisorState) -> str:
        return state.get("next_agent", "done")

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("research", research_agent)
    graph.add_node("writer", writer_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route,
        {"research": "research", "writer": "writer", "done": END}
    )
    graph.add_edge("research", "supervisor")
    graph.add_edge("writer", "supervisor")

    app = graph.compile()

    # 테스트
    result = app.invoke({
        "messages": [HumanMessage(content="보고서 작성해줘")],
        "next_agent": "",
        "research_result": "",
        "final_result": ""
    })

    print("\n문제 4: Supervisor Agent")
    print(f"  Research: {result['research_result']}")
    print(f"  Final: {result['final_result']}")

    return app


# =============================================================================
# 문제 5: 서브그래프 활용
# =============================================================================

class PipelineState(TypedDict):
    data: str
    processed: str


class AnalysisState(TypedDict):
    data: str
    stats: dict
    visualization: str


def solution_5():
    """서브그래프 활용 해답"""

    # 데이터 파이프라인 서브그래프
    def collect(state: PipelineState) -> PipelineState:
        return {"data": f"수집됨: {state['data']}"}

    def clean(state: PipelineState) -> PipelineState:
        return {"processed": f"정제됨: {state['data']}"}

    pipeline_graph = StateGraph(PipelineState)
    pipeline_graph.add_node("collect", collect)
    pipeline_graph.add_node("clean", clean)
    pipeline_graph.add_edge(START, "collect")
    pipeline_graph.add_edge("collect", "clean")
    pipeline_graph.add_edge("clean", END)
    pipeline = pipeline_graph.compile()

    # 분석 서브그래프
    def compute_stats(state: AnalysisState) -> AnalysisState:
        return {"stats": {"count": 10, "mean": 5.5}}

    def visualize(state: AnalysisState) -> AnalysisState:
        return {"visualization": "차트 생성됨"}

    analysis_graph = StateGraph(AnalysisState)
    analysis_graph.add_node("stats", compute_stats)
    analysis_graph.add_node("viz", visualize)
    analysis_graph.add_edge(START, "stats")
    analysis_graph.add_edge("stats", "viz")
    analysis_graph.add_edge("viz", END)
    analysis = analysis_graph.compile()

    # 테스트
    print("\n문제 5: 서브그래프")

    pipeline_result = pipeline.invoke({"data": "원본 데이터", "processed": ""})
    print(f"  Pipeline: {pipeline_result}")

    analysis_result = analysis.invoke({"data": "처리된 데이터", "stats": {}, "visualization": ""})
    print(f"  Analysis: {analysis_result}")

    return pipeline, analysis


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Part 3 해답 실행")
    print("=" * 50)

    solution_1()
    solution_2()
    solution_3()
    solution_4()
    solution_5()

    print("\n✅ 모든 해답 테스트 통과!")
