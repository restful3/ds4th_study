"""
Part 3: Agent 테스트

AI Agent 관련 테스트 (Tool Calling, ReAct, Multi-Agent, Subgraph)
"""

import pytest
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


class TestToolCalling:
    """Tool Calling 테스트"""

    def test_tool_definition(self):
        """@tool 데코레이터 테스트"""
        @tool
        def add_numbers(a: int, b: int) -> int:
            """두 숫자를 더합니다."""
            return a + b

        result = add_numbers.invoke({"a": 5, "b": 3})
        assert result == 8

    def test_tool_with_string(self):
        """문자열 도구 테스트"""
        @tool
        def greet(name: str) -> str:
            """인사말을 생성합니다."""
            return f"안녕하세요, {name}님!"

        result = greet.invoke({"name": "철수"})
        assert result == "안녕하세요, 철수님!"

    def test_multiple_tools(self):
        """여러 도구 테스트"""
        @tool
        def multiply(a: int, b: int) -> int:
            """두 숫자를 곱합니다."""
            return a * b

        @tool
        def divide(a: int, b: int) -> float:
            """두 숫자를 나눕니다."""
            return a / b

        tools = [multiply, divide]

        assert multiply.invoke({"a": 4, "b": 5}) == 20
        assert divide.invoke({"a": 10, "b": 2}) == 5.0


class TestReActPattern:
    """ReAct 패턴 테스트"""

    def test_simple_react_loop(self):
        """간단한 ReAct 루프"""
        class AgentState(TypedDict):
            input: str
            thoughts: Annotated[list, operator.add]
            actions: Annotated[list, operator.add]
            observations: Annotated[list, operator.add]
            final_answer: str

        def think(state: AgentState) -> AgentState:
            return {"thoughts": [f"입력 분석: {state['input']}"]}

        def act(state: AgentState) -> AgentState:
            return {"actions": ["search_action"]}

        def observe(state: AgentState) -> AgentState:
            return {"observations": ["검색 결과"]}

        def answer(state: AgentState) -> AgentState:
            return {"final_answer": "최종 답변입니다"}

        def should_continue(state: AgentState) -> str:
            if len(state.get("observations", [])) >= 1:
                return "answer"
            return "act"

        graph = StateGraph(AgentState)
        graph.add_node("think", think)
        graph.add_node("act", act)
        graph.add_node("observe", observe)
        graph.add_node("answer", answer)

        graph.add_edge(START, "think")
        graph.add_conditional_edges(
            "think",
            should_continue,
            {"act": "act", "answer": "answer"}
        )
        graph.add_edge("act", "observe")
        graph.add_edge("observe", "think")
        graph.add_edge("answer", END)

        app = graph.compile()
        result = app.invoke({
            "input": "테스트 질문",
            "thoughts": [],
            "actions": [],
            "observations": [],
            "final_answer": ""
        })

        assert result["final_answer"] == "최종 답변입니다"
        assert len(result["thoughts"]) >= 1


class TestMultiAgent:
    """Multi-Agent 테스트"""

    def test_supervisor_pattern(self):
        """Supervisor 패턴 테스트"""
        class TeamState(TypedDict):
            task: str
            current_agent: str
            results: Annotated[list, operator.add]
            done: bool

        def supervisor(state: TeamState) -> TeamState:
            results = state.get("results", [])
            if len(results) >= 2:
                return {"done": True}
            elif len(results) == 0:
                return {"current_agent": "agent_a"}
            else:
                return {"current_agent": "agent_b"}

        def agent_a(state: TeamState) -> TeamState:
            return {"results": ["Agent A 결과"]}

        def agent_b(state: TeamState) -> TeamState:
            return {"results": ["Agent B 결과"]}

        def route(state: TeamState) -> str:
            if state.get("done"):
                return "end"
            return state.get("current_agent", "agent_a")

        graph = StateGraph(TeamState)
        graph.add_node("supervisor", supervisor)
        graph.add_node("agent_a", agent_a)
        graph.add_node("agent_b", agent_b)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route,
            {"agent_a": "agent_a", "agent_b": "agent_b", "end": END}
        )
        graph.add_edge("agent_a", "supervisor")
        graph.add_edge("agent_b", "supervisor")

        app = graph.compile()
        result = app.invoke({
            "task": "팀 작업",
            "current_agent": "",
            "results": [],
            "done": False
        })

        assert result["done"] == True
        assert len(result["results"]) == 2


class TestSubgraph:
    """Subgraph 테스트"""

    def test_nested_graph(self):
        """중첩 그래프 테스트"""
        class InnerState(TypedDict):
            value: int

        class OuterState(TypedDict):
            value: int
            processed: bool

        # Inner graph
        def inner_process(state: InnerState) -> InnerState:
            return {"value": state["value"] * 2}

        inner_graph = StateGraph(InnerState)
        inner_graph.add_node("process", inner_process)
        inner_graph.add_edge(START, "process")
        inner_graph.add_edge("process", END)
        inner_app = inner_graph.compile()

        # Outer graph using inner
        def outer_pre(state: OuterState) -> OuterState:
            return {"value": state["value"] + 10}

        def call_inner(state: OuterState) -> OuterState:
            inner_result = inner_app.invoke({"value": state["value"]})
            return {"value": inner_result["value"], "processed": True}

        outer_graph = StateGraph(OuterState)
        outer_graph.add_node("pre", outer_pre)
        outer_graph.add_node("inner", call_inner)
        outer_graph.add_edge(START, "pre")
        outer_graph.add_edge("pre", "inner")
        outer_graph.add_edge("inner", END)

        app = outer_graph.compile()
        result = app.invoke({"value": 5, "processed": False})

        # (5 + 10) * 2 = 30
        assert result["value"] == 30
        assert result["processed"] == True
