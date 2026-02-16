"""
[Chapter 2] 핵심 개념 이해 - State 기초

📝 설명:
    LangGraph의 핵심 개념인 State에 대해 학습합니다.
    State는 그래프의 노드 간에 데이터를 전달하는 핵심 메커니즘입니다.

🎯 학습 목표:
    - State의 역할과 중요성 이해
    - TypedDict를 사용한 State 정의
    - Pydantic을 사용한 State 정의
    - dataclass를 사용한 State 정의
    - State 업데이트 패턴 이해

📚 관련 문서:
    - docs/Part1-Foundation/02-core-concepts.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/low_level/#state

💻 실행 방법:
    python -m src.part1_foundation.02_state_basics

📦 필요한 패키지:
    - langgraph>=0.2.0
    - pydantic>=2.0.0
"""

from typing import TypedDict, Optional, Annotated, List
from dataclasses import dataclass
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END


# =============================================================================
# 1. TypedDict를 사용한 State 정의
# =============================================================================
# TypedDict는 가장 일반적인 State 정의 방법입니다.
# 타입 힌트를 제공하지만 런타임 검증은 하지 않습니다.

class TypedDictState(TypedDict):
    """
    TypedDict를 사용한 State 정의

    장점:
        - 간단하고 직관적
        - 타입 힌트 지원
        - 추가 의존성 없음

    단점:
        - 런타임 타입 검증 없음
        - 기본값 설정 불가
    """
    name: str
    age: int
    email: Optional[str]


def process_typed_dict_state(state: TypedDictState) -> TypedDictState:
    """TypedDict State를 처리하는 노드"""
    return {
        "name": state["name"].upper(),
        "age": state["age"] + 1,
        "email": state.get("email", "없음")
    }


def run_typed_dict_example():
    """TypedDict 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: TypedDict State")
    print("=" * 60)

    graph = StateGraph(TypedDictState)
    graph.add_node("process", process_typed_dict_state)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    # 실행
    result = app.invoke({"name": "hong", "age": 25, "email": None})
    print(f"\n입력: name='hong', age=25, email=None")
    print(f"출력: {result}")


# =============================================================================
# 2. Pydantic을 사용한 State 정의
# =============================================================================
# Pydantic은 런타임 타입 검증과 데이터 변환을 제공합니다.
# 복잡한 검증 로직이 필요한 경우 유용합니다.

class PydanticState(BaseModel):
    """
    Pydantic을 사용한 State 정의

    장점:
        - 런타임 타입 검증
        - 기본값 설정 가능
        - 복잡한 검증 로직 지원
        - 데이터 변환 자동화

    단점:
        - 추가 의존성 (pydantic)
        - 약간의 성능 오버헤드
    """
    name: str = Field(description="사용자 이름")
    age: int = Field(ge=0, le=150, description="나이 (0-150)")
    email: Optional[str] = Field(default=None, description="이메일 주소")
    is_active: bool = Field(default=True, description="활성 상태")


def process_pydantic_state(state: PydanticState) -> dict:
    """Pydantic State를 처리하는 노드"""
    return {
        "name": state.name.title(),
        "age": state.age,
        "email": state.email or "미입력",
        "is_active": state.is_active
    }


def run_pydantic_example():
    """Pydantic 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: Pydantic State")
    print("=" * 60)

    graph = StateGraph(PydanticState)
    graph.add_node("process", process_pydantic_state)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    # 실행
    result = app.invoke({"name": "kim", "age": 30})
    print(f"\n입력: name='kim', age=30 (email, is_active 기본값 사용)")
    print(f"출력: {result}")

    # Pydantic의 검증 기능 데모
    print("\n📌 Pydantic 검증 기능:")
    try:
        invalid_state = PydanticState(name="test", age=200)  # age가 150 초과
    except Exception as e:
        print(f"   검증 실패 (age=200): {type(e).__name__}")


# =============================================================================
# 3. dataclass를 사용한 State 정의
# =============================================================================
# dataclass는 Python 표준 라이브러리로, 간단한 데이터 클래스에 적합합니다.

@dataclass
class DataclassState:
    """
    dataclass를 사용한 State 정의

    장점:
        - Python 표준 라이브러리
        - 기본값 설정 가능
        - __init__, __repr__ 등 자동 생성

    단점:
        - 런타임 타입 검증 없음
        - TypedDict보다 약간 무거움

    Note:
        LangGraph에서 dataclass를 사용할 때는
        as_dict() 메서드나 asdict()를 활용해야 할 수 있습니다.
    """
    name: str
    score: int = 0
    completed: bool = False


def process_dataclass_state(state: dict) -> dict:
    """dataclass State를 처리하는 노드"""
    # dataclass는 dict로 변환되어 전달됨
    return {
        "name": state["name"],
        "score": state["score"] + 10,
        "completed": state["score"] + 10 >= 100
    }


def run_dataclass_example():
    """dataclass 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: dataclass State")
    print("=" * 60)

    # dataclass를 TypedDict처럼 사용
    class DataclassStateDict(TypedDict):
        name: str
        score: int
        completed: bool

    graph = StateGraph(DataclassStateDict)
    graph.add_node("process", process_dataclass_state)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    app = graph.compile()

    # 실행
    result = app.invoke({"name": "player1", "score": 95, "completed": False})
    print(f"\n입력: name='player1', score=95, completed=False")
    print(f"출력: {result}")


# =============================================================================
# 4. 복잡한 State 정의
# =============================================================================
# 실제 애플리케이션에서는 더 복잡한 State가 필요합니다.

class ComplexState(TypedDict):
    """
    복잡한 애플리케이션을 위한 State 정의

    실제 Agent 구현 시 필요한 필드들을 포함합니다.
    """
    # 사용자 입력
    user_input: str

    # 대화 기록 (나중에 messages 타입으로 대체)
    history: List[str]

    # 처리 결과
    result: Optional[str]

    # 메타데이터
    step_count: int
    errors: List[str]
    is_complete: bool


def initialize_state(state: ComplexState) -> ComplexState:
    """State를 초기화하는 노드"""
    return {
        "history": state.get("history", []) + [f"입력: {state['user_input']}"],
        "step_count": state.get("step_count", 0) + 1,
        "errors": state.get("errors", []),
        "is_complete": False
    }


def process_input(state: ComplexState) -> ComplexState:
    """입력을 처리하는 노드"""
    processed = state["user_input"].strip().upper()
    return {
        "result": f"처리됨: {processed}",
        "history": state["history"] + [f"처리: {processed}"],
        "step_count": state["step_count"] + 1
    }


def finalize(state: ComplexState) -> ComplexState:
    """처리를 마무리하는 노드"""
    return {
        "history": state["history"] + [f"완료: {state['result']}"],
        "step_count": state["step_count"] + 1,
        "is_complete": True
    }


def create_complex_graph():
    """복잡한 그래프 생성"""
    graph = StateGraph(ComplexState)

    graph.add_node("initialize", initialize_state)
    graph.add_node("process", process_input)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "process")
    graph.add_edge("process", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_complex_example():
    """복잡한 State 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: 복잡한 State")
    print("=" * 60)

    app = create_complex_graph()

    initial_state = {
        "user_input": "hello world",
        "history": [],
        "result": None,
        "step_count": 0,
        "errors": [],
        "is_complete": False
    }

    result = app.invoke(initial_state)

    print(f"\n📥 초기 입력: '{initial_state['user_input']}'")
    print(f"\n📜 처리 기록:")
    for i, entry in enumerate(result["history"], 1):
        print(f"   {i}. {entry}")
    print(f"\n📤 최종 결과: {result['result']}")
    print(f"   단계 수: {result['step_count']}")
    print(f"   완료 여부: {result['is_complete']}")


# =============================================================================
# 5. State 업데이트 패턴 설명
# =============================================================================

def explain_state_update_patterns():
    """State 업데이트 패턴을 설명합니다."""
    print("\n" + "=" * 60)
    print("📘 State 업데이트 패턴")
    print("=" * 60)

    print("""
LangGraph에서 노드 함수가 State를 반환할 때,
반환된 값은 기존 State와 '병합(merge)' 됩니다.

📌 기본 동작: 값 덮어쓰기 (Overwrite)

   현재 State: {"name": "kim", "age": 25}
   노드 반환값: {"age": 26}
   결과 State: {"name": "kim", "age": 26}

📌 부분 업데이트

   - 노드는 전체 State를 반환할 필요가 없습니다.
   - 변경된 필드만 반환하면 됩니다.
   - 반환하지 않은 필드는 그대로 유지됩니다.

📌 주의사항

   1. List 타입의 기본 동작은 '덮어쓰기'입니다.
      새 리스트가 기존 리스트를 완전히 대체합니다.

   2. List에 항목을 '추가'하려면:
      - 방법 1: state["list"] + [new_item] 반환
      - 방법 2: Reducer 사용 (다음 챕터에서 학습)

   3. None을 반환하면 해당 필드는 업데이트되지 않습니다.
      명시적으로 None을 설정하려면 {"field": None}을 반환하세요.
""")


# =============================================================================
# 6. State 정의 방법 비교
# =============================================================================

def compare_state_definitions():
    """State 정의 방법 비교"""
    print("\n" + "=" * 60)
    print("📊 State 정의 방법 비교")
    print("=" * 60)

    print("""
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   기능       │ TypedDict   │  Pydantic   │ dataclass   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 타입 힌트    │     ✅      │     ✅      │     ✅      │
│ 런타임 검증  │     ❌      │     ✅      │     ❌      │
│ 기본값       │     ❌      │     ✅      │     ✅      │
│ 외부 의존성  │     ❌      │   pydantic  │     ❌      │
│ 성능         │    빠름     │   약간느림  │    빠름     │
│ LangGraph   │   권장✨    │    지원     │    지원     │
└─────────────┴─────────────┴─────────────┴─────────────┘

💡 권장 사용 시나리오:

   - TypedDict: 대부분의 경우 (간단, 빠름, LangGraph 기본)
   - Pydantic: 복잡한 검증이 필요한 경우
   - dataclass: 기존 코드와 호환이 필요한 경우
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 2] 핵심 개념 이해 - State 기초")
    print("=" * 60)

    # 예제 실행
    run_typed_dict_example()
    run_pydantic_example()
    run_dataclass_example()
    run_complex_example()

    # 개념 설명
    explain_state_update_patterns()
    compare_state_definitions()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 03_reducers.py (Reducer 함수)")
    print("=" * 60)


if __name__ == "__main__":
    main()
