"""
Part 1: Foundation 연습 문제 해답
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage


# =============================================================================
# 문제 1: 간단한 카운터 그래프
# =============================================================================

class CounterState(TypedDict):
    value: int


def solution_1():
    """카운터 그래프 해답"""

    def increment(state: CounterState) -> CounterState:
        return {"value": state["value"] + 1}

    def decrement(state: CounterState) -> CounterState:
        return {"value": state["value"] - 1}

    def double(state: CounterState) -> CounterState:
        return {"value": state["value"] * 2}

    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_node("double", double)
    graph.add_node("decrement", decrement)

    graph.add_edge(START, "increment")
    graph.add_edge("increment", "double")
    graph.add_edge("double", "decrement")
    graph.add_edge("decrement", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({"value": 5})
    print(f"문제 1 결과: {result}")  # {"value": 11}
    assert result["value"] == 11

    return app


# =============================================================================
# 문제 2: 조건부 인사 그래프
# =============================================================================

class GreetingState(TypedDict):
    hour: int
    greeting: str


def solution_2():
    """조건부 인사 그래프 해답"""

    def route_by_time(state: GreetingState) -> str:
        hour = state["hour"]
        if 6 <= hour <= 11:
            return "morning"
        elif 12 <= hour <= 17:
            return "afternoon"
        elif 18 <= hour <= 21:
            return "evening"
        else:
            return "default"

    def morning_greeting(state: GreetingState) -> GreetingState:
        return {"greeting": "좋은 아침입니다!"}

    def afternoon_greeting(state: GreetingState) -> GreetingState:
        return {"greeting": "좋은 오후입니다!"}

    def evening_greeting(state: GreetingState) -> GreetingState:
        return {"greeting": "좋은 저녁입니다!"}

    def default_greeting(state: GreetingState) -> GreetingState:
        return {"greeting": "안녕하세요!"}

    graph = StateGraph(GreetingState)
    graph.add_node("morning", morning_greeting)
    graph.add_node("afternoon", afternoon_greeting)
    graph.add_node("evening", evening_greeting)
    graph.add_node("default", default_greeting)

    graph.add_conditional_edges(
        START,
        route_by_time,
        {
            "morning": "morning",
            "afternoon": "afternoon",
            "evening": "evening",
            "default": "default"
        }
    )
    graph.add_edge("morning", END)
    graph.add_edge("afternoon", END)
    graph.add_edge("evening", END)
    graph.add_edge("default", END)

    app = graph.compile()

    # 테스트
    test_cases = [
        (8, "좋은 아침입니다!"),
        (14, "좋은 오후입니다!"),
        (19, "좋은 저녁입니다!"),
        (2, "안녕하세요!")
    ]

    for hour, expected in test_cases:
        result = app.invoke({"hour": hour, "greeting": ""})
        print(f"문제 2 - {hour}시: {result['greeting']}")
        assert result["greeting"] == expected

    return app


# =============================================================================
# 문제 3: 메시지 누적 그래프
# =============================================================================

class MessageState(TypedDict):
    messages: Annotated[list, add_messages]


def solution_3():
    """메시지 누적 그래프 해답"""

    def step_1(state: MessageState) -> MessageState:
        return {"messages": [AIMessage(content="Step 1 완료")]}

    def step_2(state: MessageState) -> MessageState:
        return {"messages": [AIMessage(content="Step 2 완료")]}

    def step_3(state: MessageState) -> MessageState:
        return {"messages": [AIMessage(content="Step 3 완료")]}

    graph = StateGraph(MessageState)
    graph.add_node("step1", step_1)
    graph.add_node("step2", step_2)
    graph.add_node("step3", step_3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({"messages": []})
    print(f"문제 3 결과: {len(result['messages'])}개 메시지")
    for msg in result["messages"]:
        print(f"  - {msg.content}")

    assert len(result["messages"]) == 3

    return app


# =============================================================================
# 문제 4: 커스텀 Reducer
# =============================================================================

def unique_merge(current: List[str], new: List[str]) -> List[str]:
    """중복을 제거하며 순서 유지하는 병합"""
    seen = set(current)
    result = list(current)
    for item in new:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class UniqueListState(TypedDict):
    items: Annotated[List[str], unique_merge]


def solution_4():
    """커스텀 Reducer 해답"""

    def add_items_a(state: UniqueListState) -> UniqueListState:
        return {"items": ["a", "b"]}

    def add_items_b(state: UniqueListState) -> UniqueListState:
        return {"items": ["b", "c"]}

    def add_items_c(state: UniqueListState) -> UniqueListState:
        return {"items": ["c", "d"]}

    graph = StateGraph(UniqueListState)
    graph.add_node("add_a", add_items_a)
    graph.add_node("add_b", add_items_b)
    graph.add_node("add_c", add_items_c)

    graph.add_edge(START, "add_a")
    graph.add_edge("add_a", "add_b")
    graph.add_edge("add_b", "add_c")
    graph.add_edge("add_c", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({"items": []})
    print(f"문제 4 결과: {result['items']}")
    assert result["items"] == ["a", "b", "c", "d"]

    return app


# =============================================================================
# 문제 5: 입출력 스키마 분리
# =============================================================================

class InputState(TypedDict):
    raw_data: str


class OutputState(TypedDict):
    word_count: int
    char_count: int


class InternalState(InputState, OutputState):
    pass


def solution_5():
    """입출력 스키마 분리 해답"""

    def analyze_text(state: InternalState) -> InternalState:
        raw_data = state["raw_data"]
        words = raw_data.split()
        return {
            "word_count": len(words),
            "char_count": len(raw_data)
        }

    graph = StateGraph(InternalState, input=InputState, output=OutputState)
    graph.add_node("analyze", analyze_text)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({"raw_data": "hello world"})
    print(f"문제 5 결과: {result}")
    assert result["word_count"] == 2
    assert result["char_count"] == 11

    return app


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Part 1 해답 실행")
    print("=" * 50)

    solution_1()
    print()
    solution_2()
    print()
    solution_3()
    print()
    solution_4()
    print()
    solution_5()

    print("\n✅ 모든 해답 테스트 통과!")
