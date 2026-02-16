"""
[Chapter 3] 첫 번째 그래프 만들기 - Reducer 함수

📝 설명:
    Reducer는 LangGraph에서 State 업데이트 방식을 제어하는 핵심 메커니즘입니다.
    기본적으로 State 필드는 '덮어쓰기'되지만, Reducer를 사용하면
    누적, 병합 등 다양한 업데이트 전략을 적용할 수 있습니다.

🎯 학습 목표:
    - Reducer의 개념과 역할 이해
    - Annotated를 사용한 Reducer 정의
    - 내장 Reducer 활용 (add, operator.add)
    - 커스텀 Reducer 작성
    - add_messages reducer 소개

📚 관련 문서:
    - docs/Part1-Foundation/03-first-graph.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers

💻 실행 방법:
    python -m src.part1_foundation.03_reducers

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

from typing import TypedDict, Annotated, List, Union
import operator

from langgraph.graph import StateGraph, START, END


# =============================================================================
# 1. 기본 동작: 덮어쓰기 (Without Reducer)
# =============================================================================

class StateWithoutReducer(TypedDict):
    """Reducer가 없는 State - 기본 덮어쓰기 동작"""
    items: List[str]
    count: int


def add_item_overwrite(state: StateWithoutReducer) -> StateWithoutReducer:
    """아이템을 추가하지만 덮어쓰기 됨"""
    # 이렇게 하면 기존 items가 완전히 대체됨!
    return {"items": ["새 아이템"]}


def run_without_reducer_example():
    """Reducer 없는 예제 - 문제점 확인"""
    print("\n" + "=" * 60)
    print("예제 1: Reducer 없이 (문제점)")
    print("=" * 60)

    graph = StateGraph(StateWithoutReducer)
    graph.add_node("add", add_item_overwrite)
    graph.add_edge(START, "add")
    graph.add_edge("add", END)
    app = graph.compile()

    initial = {"items": ["기존1", "기존2"], "count": 2}
    result = app.invoke(initial)

    print(f"\n⚠️  문제: List가 덮어쓰기됨!")
    print(f"   입력: {initial['items']}")
    print(f"   출력: {result['items']}")
    print(f"   기존 아이템이 사라졌습니다!")


# =============================================================================
# 2. operator.add를 사용한 List Reducer
# =============================================================================

class StateWithOperatorAdd(TypedDict):
    """operator.add를 사용한 State"""
    # Annotated[타입, Reducer함수]
    # operator.add는 + 연산자를 사용 (List의 경우 연결)
    items: Annotated[List[str], operator.add]
    count: int


def add_item_with_reducer(state: StateWithOperatorAdd) -> StateWithOperatorAdd:
    """아이템을 추가 - Reducer가 누적해줌"""
    return {"items": ["새 아이템"]}


def run_operator_add_example():
    """operator.add Reducer 예제"""
    print("\n" + "=" * 60)
    print("예제 2: operator.add Reducer")
    print("=" * 60)

    graph = StateGraph(StateWithOperatorAdd)
    graph.add_node("add", add_item_with_reducer)
    graph.add_edge(START, "add")
    graph.add_edge("add", END)
    app = graph.compile()

    initial = {"items": ["기존1", "기존2"], "count": 2}
    result = app.invoke(initial)

    print(f"\n✅ 해결: List가 누적됨!")
    print(f"   입력: {initial['items']}")
    print(f"   추가: ['새 아이템']")
    print(f"   출력: {result['items']}")


# =============================================================================
# 3. 숫자 누적을 위한 Reducer
# =============================================================================

class StateWithNumericReducer(TypedDict):
    """숫자 누적을 위한 State"""
    items: Annotated[List[str], operator.add]
    # 숫자도 operator.add로 누적 가능
    total: Annotated[int, operator.add]


def process_with_accumulation(state: StateWithNumericReducer) -> StateWithNumericReducer:
    """아이템을 추가하고 카운트를 증가"""
    return {
        "items": ["처리됨"],
        "total": 1  # 기존 값에 1이 더해짐
    }


def run_numeric_reducer_example():
    """숫자 Reducer 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 숫자 누적 Reducer")
    print("=" * 60)

    graph = StateGraph(StateWithNumericReducer)
    graph.add_node("process", process_with_accumulation)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    initial = {"items": [], "total": 10}
    result = app.invoke(initial)

    print(f"\n✅ 숫자도 누적됨!")
    print(f"   초기 total: {initial['total']}")
    print(f"   추가값: 1")
    print(f"   최종 total: {result['total']}")


# =============================================================================
# 4. 커스텀 Reducer 함수
# =============================================================================

def max_reducer(current: int, new: int) -> int:
    """최대값을 유지하는 Reducer"""
    return max(current, new)


def concat_with_separator(current: str, new: str) -> str:
    """구분자로 연결하는 Reducer"""
    if not current:
        return new
    return f"{current} | {new}"


def unique_list_reducer(current: List[str], new: List[str]) -> List[str]:
    """중복을 제거하는 List Reducer"""
    combined = current + new
    # 순서를 유지하면서 중복 제거
    seen = set()
    result = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class StateWithCustomReducers(TypedDict):
    """커스텀 Reducer를 사용하는 State"""
    high_score: Annotated[int, max_reducer]
    log: Annotated[str, concat_with_separator]
    unique_tags: Annotated[List[str], unique_list_reducer]


def update_scores(state: StateWithCustomReducers) -> StateWithCustomReducers:
    """점수와 로그를 업데이트"""
    return {
        "high_score": 85,  # max(현재값, 85)
        "log": "첫 번째 업데이트",
        "unique_tags": ["python", "langgraph"]
    }


def update_again(state: StateWithCustomReducers) -> StateWithCustomReducers:
    """다시 업데이트"""
    return {
        "high_score": 92,  # max(현재값, 92)
        "log": "두 번째 업데이트",
        "unique_tags": ["langgraph", "ai"]  # "langgraph"는 중복이므로 무시
    }


def run_custom_reducer_example():
    """커스텀 Reducer 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 커스텀 Reducer")
    print("=" * 60)

    graph = StateGraph(StateWithCustomReducers)
    graph.add_node("update1", update_scores)
    graph.add_node("update2", update_again)
    graph.add_edge(START, "update1")
    graph.add_edge("update1", "update2")
    graph.add_edge("update2", END)
    app = graph.compile()

    initial = {"high_score": 70, "log": "", "unique_tags": []}
    result = app.invoke(initial)

    print(f"\n📊 커스텀 Reducer 결과:")
    print(f"\n   max_reducer (최고 점수):")
    print(f"   70 → 85 → 92 = {result['high_score']}")

    print(f"\n   concat_with_separator (로그):")
    print(f"   '{result['log']}'")

    print(f"\n   unique_list_reducer (태그):")
    print(f"   {result['unique_tags']}")


# =============================================================================
# 5. 조건부 Reducer (Union 타입 활용)
# =============================================================================

def conditional_list_reducer(
    current: List[str],
    new: Union[List[str], str]
) -> List[str]:
    """
    조건부 List Reducer

    - new가 리스트면 추가
    - new가 문자열이면 단일 항목으로 추가
    """
    if isinstance(new, str):
        return current + [new]
    return current + new


class StateWithConditionalReducer(TypedDict):
    """조건부 Reducer를 사용하는 State"""
    messages: Annotated[List[str], conditional_list_reducer]


def add_single_message(state: StateWithConditionalReducer) -> StateWithConditionalReducer:
    """단일 메시지 추가"""
    return {"messages": "안녕하세요"}  # 문자열로 전달


def add_multiple_messages(state: StateWithConditionalReducer) -> StateWithConditionalReducer:
    """여러 메시지 추가"""
    return {"messages": ["반갑습니다", "좋은 하루 되세요"]}  # 리스트로 전달


def run_conditional_reducer_example():
    """조건부 Reducer 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 조건부 Reducer")
    print("=" * 60)

    graph = StateGraph(StateWithConditionalReducer)
    graph.add_node("single", add_single_message)
    graph.add_node("multiple", add_multiple_messages)
    graph.add_edge(START, "single")
    graph.add_edge("single", "multiple")
    graph.add_edge("multiple", END)
    app = graph.compile()

    initial = {"messages": ["시작"]}
    result = app.invoke(initial)

    print(f"\n✅ 조건부 Reducer 결과:")
    print(f"   초기: ['시작']")
    print(f"   + '안녕하세요' (문자열)")
    print(f"   + ['반갑습니다', '좋은 하루 되세요'] (리스트)")
    print(f"   결과: {result['messages']}")


# =============================================================================
# 6. Reducer 개념 설명
# =============================================================================

def explain_reducer_concept():
    """Reducer 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Reducer 개념 정리")
    print("=" * 60)

    print("""
Reducer란?
  State 필드가 업데이트될 때 어떻게 처리할지 정의하는 함수입니다.

기본 동작 (Reducer 없음):
  새 값이 기존 값을 완전히 대체합니다.

  현재값: {"items": ["a", "b"]}
  반환값: {"items": ["c"]}
  결과:   {"items": ["c"]}  ← 기존 값 사라짐

Reducer 적용 (operator.add):
  새 값이 기존 값에 누적됩니다.

  현재값: {"items": ["a", "b"]}
  반환값: {"items": ["c"]}
  결과:   {"items": ["a", "b", "c"]}  ← 누적됨

Reducer 함수 시그니처:
  def reducer(current_value, new_value) -> updated_value

내장 Reducer:
  - operator.add: 리스트 연결, 숫자 덧셈
  - add_messages: 메시지 목록 관리 (Chapter 4에서 학습)

커스텀 Reducer 작성:
  원하는 로직으로 직접 작성 가능
  - 최대값 유지
  - 중복 제거
  - 조건부 병합
  - 등등...

사용법:
  from typing import Annotated

  class State(TypedDict):
      field: Annotated[Type, reducer_function]
""")


# =============================================================================
# 7. 병렬 노드에서의 Reducer 동작
# =============================================================================

class ParallelState(TypedDict):
    """병렬 실행을 위한 State"""
    results: Annotated[List[str], operator.add]


def node_a(state: ParallelState) -> ParallelState:
    """노드 A"""
    return {"results": ["A 결과"]}


def node_b(state: ParallelState) -> ParallelState:
    """노드 B"""
    return {"results": ["B 결과"]}


def run_parallel_reducer_example():
    """병렬 노드에서의 Reducer 예제"""
    print("\n" + "=" * 60)
    print("예제 6: 병렬 노드와 Reducer")
    print("=" * 60)

    graph = StateGraph(ParallelState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)

    # 병렬 실행 구성
    # START에서 두 노드로 동시에 연결
    graph.add_edge(START, "node_a")
    graph.add_edge(START, "node_b")

    # 두 노드가 모두 END로 연결
    graph.add_edge("node_a", END)
    graph.add_edge("node_b", END)

    app = graph.compile()

    initial = {"results": ["초기값"]}
    result = app.invoke(initial)

    print(f"\n🔀 병렬 실행 결과:")
    print(f"   초기: {initial['results']}")
    print(f"   node_a 추가: ['A 결과']")
    print(f"   node_b 추가: ['B 결과']")
    print(f"   최종: {result['results']}")
    print(f"\n   ℹ️  두 노드가 병렬로 실행되어 결과가 합쳐짐")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 3] 첫 번째 그래프 만들기 - Reducer 함수")
    print("=" * 60)

    # 예제 실행
    run_without_reducer_example()
    run_operator_add_example()
    run_numeric_reducer_example()
    run_custom_reducer_example()
    run_conditional_reducer_example()
    run_parallel_reducer_example()

    # 개념 설명
    explain_reducer_concept()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 04_messages_state.py (MessagesState)")
    print("=" * 60)


if __name__ == "__main__":
    main()
