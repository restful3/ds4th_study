"""
Part 1: Foundation 테스트

기초 개념 (State, Node, Edge, Reducer) 관련 테스트
"""

import pytest
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END


class TestStateBasics:
    """State 기본 기능 테스트"""

    def test_simple_state_graph(self):
        """간단한 StateGraph 생성 및 실행"""
        class SimpleState(TypedDict):
            message: str

        def node(state: SimpleState) -> SimpleState:
            return {"message": f"처리됨: {state['message']}"}

        graph = StateGraph(SimpleState)
        graph.add_node("process", node)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)

        app = graph.compile()
        result = app.invoke({"message": "테스트"})

        assert result["message"] == "처리됨: 테스트"

    def test_state_with_multiple_fields(self):
        """여러 필드를 가진 State"""
        class MultiState(TypedDict):
            name: str
            count: int
            items: list

        def update_node(state: MultiState) -> MultiState:
            return {
                "count": state["count"] + 1,
                "items": state["items"] + ["new_item"]
            }

        graph = StateGraph(MultiState)
        graph.add_node("update", update_node)
        graph.add_edge(START, "update")
        graph.add_edge("update", END)

        app = graph.compile()
        result = app.invoke({
            "name": "test",
            "count": 0,
            "items": []
        })

        assert result["name"] == "test"
        assert result["count"] == 1
        assert result["items"] == ["new_item"]


class TestReducers:
    """Reducer 테스트"""

    def test_add_reducer(self):
        """operator.add reducer 테스트"""
        class ListState(TypedDict):
            items: Annotated[list, operator.add]

        def add_item(state: ListState) -> ListState:
            return {"items": ["item1"]}

        def add_more(state: ListState) -> ListState:
            return {"items": ["item2"]}

        graph = StateGraph(ListState)
        graph.add_node("add1", add_item)
        graph.add_node("add2", add_more)
        graph.add_edge(START, "add1")
        graph.add_edge("add1", "add2")
        graph.add_edge("add2", END)

        app = graph.compile()
        result = app.invoke({"items": []})

        assert result["items"] == ["item1", "item2"]

    def test_custom_reducer(self):
        """커스텀 reducer 테스트"""
        def max_reducer(current: int, update: int) -> int:
            return max(current, update)

        class MaxState(TypedDict):
            max_value: Annotated[int, max_reducer]

        def node1(state: MaxState) -> MaxState:
            return {"max_value": 10}

        def node2(state: MaxState) -> MaxState:
            return {"max_value": 5}

        graph = StateGraph(MaxState)
        graph.add_node("n1", node1)
        graph.add_node("n2", node2)
        graph.add_edge(START, "n1")
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", END)

        app = graph.compile()
        result = app.invoke({"max_value": 0})

        assert result["max_value"] == 10  # max(10, 5) = 10


class TestMultipleNodes:
    """여러 노드 테스트"""

    def test_sequential_nodes(self):
        """순차적 노드 실행"""
        class CountState(TypedDict):
            count: int
            history: Annotated[list, operator.add]

        def increment(state: CountState) -> CountState:
            return {
                "count": state["count"] + 1,
                "history": [f"count: {state['count'] + 1}"]
            }

        graph = StateGraph(CountState)
        graph.add_node("inc1", increment)
        graph.add_node("inc2", increment)
        graph.add_node("inc3", increment)
        graph.add_edge(START, "inc1")
        graph.add_edge("inc1", "inc2")
        graph.add_edge("inc2", "inc3")
        graph.add_edge("inc3", END)

        app = graph.compile()
        result = app.invoke({"count": 0, "history": []})

        assert result["count"] == 3
        assert len(result["history"]) == 3


class TestMessagesState:
    """MessagesState 테스트"""

    def test_messages_state(self):
        """MessagesState 기본 동작"""
        from langgraph.graph import MessagesState
        from langchain_core.messages import HumanMessage, AIMessage

        def respond(state: MessagesState) -> MessagesState:
            return {"messages": [AIMessage(content="응답입니다")]}

        graph = StateGraph(MessagesState)
        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)

        app = graph.compile()
        result = app.invoke({
            "messages": [HumanMessage(content="안녕하세요")]
        })

        assert len(result["messages"]) == 2
        assert result["messages"][0].content == "안녕하세요"
        assert result["messages"][1].content == "응답입니다"
