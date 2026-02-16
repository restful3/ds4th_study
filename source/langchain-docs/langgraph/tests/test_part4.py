"""
Part 4: Production 테스트

프로덕션 기능 테스트 (Checkpointer, Memory, Streaming, HITL, Time Travel)
"""

import pytest
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


class TestCheckpointer:
    """Checkpointer 테스트"""

    def test_memory_saver(self):
        """MemorySaver 기본 동작"""
        class CountState(TypedDict):
            count: int

        def increment(state: CountState) -> CountState:
            return {"count": state["count"] + 1}

        graph = StateGraph(CountState)
        graph.add_node("inc", increment)
        graph.add_edge(START, "inc")
        graph.add_edge("inc", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "test_1"}}

        # 첫 번째 실행
        result1 = app.invoke({"count": 0}, config=config)
        assert result1["count"] == 1

        # 상태 확인
        state = app.get_state(config)
        assert state.values["count"] == 1

    def test_state_persistence(self):
        """상태 지속성 테스트"""
        class ChatState(TypedDict):
            messages: Annotated[list, operator.add]

        def respond(state: ChatState) -> ChatState:
            return {"messages": ["응답"]}

        graph = StateGraph(ChatState)
        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "chat_1"}}

        # 첫 번째 대화
        app.invoke({"messages": ["안녕"]}, config=config)

        # 두 번째 대화
        app.invoke({"messages": ["잘 지내?"]}, config=config)

        # 상태 확인 - 모든 메시지가 누적되어야 함
        state = app.get_state(config)
        assert len(state.values["messages"]) == 4  # 2 입력 + 2 응답

    def test_different_threads(self):
        """다른 스레드 격리 테스트"""
        class ValueState(TypedDict):
            value: str

        def set_value(state: ValueState) -> ValueState:
            return {}

        graph = StateGraph(ValueState)
        graph.add_node("set", set_value)
        graph.add_edge(START, "set")
        graph.add_edge("set", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        # Thread 1
        config1 = {"configurable": {"thread_id": "thread_1"}}
        app.invoke({"value": "A"}, config=config1)

        # Thread 2
        config2 = {"configurable": {"thread_id": "thread_2"}}
        app.invoke({"value": "B"}, config=config2)

        # 각 스레드 상태 확인
        state1 = app.get_state(config1)
        state2 = app.get_state(config2)

        assert state1.values["value"] == "A"
        assert state2.values["value"] == "B"


class TestStateHistory:
    """상태 히스토리 테스트"""

    def test_get_state_history(self):
        """히스토리 조회"""
        class StepState(TypedDict):
            step: int
            history: Annotated[list, operator.add]

        def step1(state: StepState) -> StepState:
            return {"step": 1, "history": ["step1"]}

        def step2(state: StepState) -> StepState:
            return {"step": 2, "history": ["step2"]}

        graph = StateGraph(StepState)
        graph.add_node("s1", step1)
        graph.add_node("s2", step2)
        graph.add_edge(START, "s1")
        graph.add_edge("s1", "s2")
        graph.add_edge("s2", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "history_test"}}
        app.invoke({"step": 0, "history": []}, config=config)

        # 히스토리 조회
        history = list(app.get_state_history(config))

        # 여러 체크포인트가 있어야 함
        assert len(history) > 1


class TestStreaming:
    """스트리밍 테스트"""

    def test_stream_values(self):
        """값 스트리밍"""
        class StreamState(TypedDict):
            count: int

        def inc(state: StreamState) -> StreamState:
            return {"count": state["count"] + 1}

        graph = StateGraph(StreamState)
        graph.add_node("inc1", inc)
        graph.add_node("inc2", inc)
        graph.add_edge(START, "inc1")
        graph.add_edge("inc1", "inc2")
        graph.add_edge("inc2", END)

        app = graph.compile()

        events = list(app.stream(
            {"count": 0},
            stream_mode="values"
        ))

        # 초기 상태 + 2개 노드 = 최소 2개 이벤트
        assert len(events) >= 2

    def test_stream_updates(self):
        """업데이트 스트리밍"""
        class UpdateState(TypedDict):
            value: str

        def update_a(state: UpdateState) -> UpdateState:
            return {"value": "A"}

        def update_b(state: UpdateState) -> UpdateState:
            return {"value": "B"}

        graph = StateGraph(UpdateState)
        graph.add_node("a", update_a)
        graph.add_node("b", update_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)

        app = graph.compile()

        updates = list(app.stream(
            {"value": ""},
            stream_mode="updates"
        ))

        # 각 노드의 업데이트가 있어야 함
        assert len(updates) >= 2


class TestUpdateState:
    """상태 업데이트 테스트"""

    def test_manual_state_update(self):
        """수동 상태 업데이트"""
        class EditState(TypedDict):
            text: str
            edited: bool

        def process(state: EditState) -> EditState:
            return {"edited": True}

        graph = StateGraph(EditState)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "edit_test"}}

        # 실행
        app.invoke({"text": "원본", "edited": False}, config=config)

        # 상태 수동 업데이트
        app.update_state(config, {"text": "수정됨"})

        # 업데이트된 상태 확인
        state = app.get_state(config)
        assert state.values["text"] == "수정됨"
