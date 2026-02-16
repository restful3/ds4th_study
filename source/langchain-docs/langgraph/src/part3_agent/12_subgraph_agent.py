"""
[Chapter 12] 서브그래프 활용

📝 설명:
    서브그래프는 그래프 내에서 다른 그래프를 노드로 사용하는 기능입니다.
    복잡한 워크플로우를 모듈화하고 재사용 가능한 컴포넌트로 만들 수 있습니다.

🎯 학습 목표:
    - 서브그래프 개념 이해
    - 노드에서 그래프 호출
    - 그래프를 노드로 추가
    - 상태 공유 vs 분리

📚 관련 문서:
    - docs/Part3-Agent/12-subgraphs.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs

💻 실행 방법:
    python -m src.part3_agent.12_subgraph_agent

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, START, END


# =============================================================================
# 1. 기본 서브그래프 개념
# =============================================================================

def explain_subgraph_concept():
    """서브그래프 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 서브그래프 개념")
    print("=" * 60)

    print("""
서브그래프란?
- 그래프 내에서 다른 그래프를 노드로 사용
- 복잡한 로직을 모듈화
- 재사용 가능한 컴포넌트

서브그래프 사용 방법:

1. 노드 내에서 그래프 호출
   def my_node(state):
       subgraph = create_subgraph()
       result = subgraph.invoke(state)
       return result

2. 그래프를 노드로 직접 추가
   subgraph = create_subgraph()
   main_graph.add_node("sub", subgraph)

상태 관리:

- 공유 상태: 부모와 자식이 같은 State 스키마 사용
- 분리 상태: 서브그래프가 자체 State 사용, 변환 필요
""")


# =============================================================================
# 2. 간단한 서브그래프
# =============================================================================

class SimpleState(TypedDict):
    """간단한 State"""
    value: int
    history: Annotated[List[str], operator.add]


def increment(state: SimpleState) -> SimpleState:
    """값 증가"""
    return {
        "value": state["value"] + 1,
        "history": [f"increment: {state['value']} -> {state['value'] + 1}"]
    }


def double(state: SimpleState) -> SimpleState:
    """값 2배"""
    return {
        "value": state["value"] * 2,
        "history": [f"double: {state['value']} -> {state['value'] * 2}"]
    }


def create_math_subgraph():
    """수학 연산 서브그래프 생성"""
    graph = StateGraph(SimpleState)

    graph.add_node("increment", increment)
    graph.add_node("double", double)

    graph.add_edge(START, "increment")
    graph.add_edge("increment", "double")
    graph.add_edge("double", END)

    return graph.compile()


def create_main_graph_with_subgraph():
    """서브그래프를 포함하는 메인 그래프 생성"""

    def prepare(state: SimpleState) -> SimpleState:
        """준비 단계"""
        return {"history": ["prepare: 시작"]}

    def finalize(state: SimpleState) -> SimpleState:
        """마무리 단계"""
        return {"history": [f"finalize: 최종값 = {state['value']}"]}

    # 서브그래프 생성
    math_subgraph = create_math_subgraph()

    # 메인 그래프
    main_graph = StateGraph(SimpleState)

    main_graph.add_node("prepare", prepare)
    main_graph.add_node("math_ops", math_subgraph)  # 서브그래프를 노드로 추가!
    main_graph.add_node("finalize", finalize)

    main_graph.add_edge(START, "prepare")
    main_graph.add_edge("prepare", "math_ops")
    main_graph.add_edge("math_ops", "finalize")
    main_graph.add_edge("finalize", END)

    return main_graph.compile()


def run_basic_subgraph_example():
    """기본 서브그래프 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 서브그래프")
    print("=" * 60)

    # 서브그래프만 실행
    print("\n📊 서브그래프만 실행:")
    subgraph = create_math_subgraph()
    result = subgraph.invoke({"value": 5, "history": []})
    print(f"   입력: 5")
    print(f"   출력: {result['value']}")
    print(f"   기록: {result['history']}")

    # 메인 그래프 (서브그래프 포함) 실행
    print("\n📊 메인 그래프 (서브그래프 포함) 실행:")
    main_graph = create_main_graph_with_subgraph()
    result = main_graph.invoke({"value": 5, "history": []})
    print(f"   입력: 5")
    print(f"   출력: {result['value']}")
    print(f"   전체 기록:")
    for h in result["history"]:
        print(f"      - {h}")


# =============================================================================
# 3. 다른 State를 가진 서브그래프
# =============================================================================

class MainState(TypedDict):
    """메인 그래프 State"""
    input_text: str
    processed_text: str
    final_result: str


class ProcessingState(TypedDict):
    """서브그래프 State (다른 스키마)"""
    text: str
    is_upper: bool
    char_count: int


def create_processing_subgraph():
    """텍스트 처리 서브그래프"""

    def to_upper(state: ProcessingState) -> ProcessingState:
        return {
            "text": state["text"].upper(),
            "is_upper": True
        }

    def count_chars(state: ProcessingState) -> ProcessingState:
        return {"char_count": len(state["text"])}

    graph = StateGraph(ProcessingState)
    graph.add_node("to_upper", to_upper)
    graph.add_node("count_chars", count_chars)
    graph.add_edge(START, "to_upper")
    graph.add_edge("to_upper", "count_chars")
    graph.add_edge("count_chars", END)

    return graph.compile()


def create_main_graph_with_different_state():
    """다른 State를 가진 서브그래프를 사용하는 메인 그래프"""

    processing_subgraph = create_processing_subgraph()

    def preprocess(state: MainState) -> MainState:
        """전처리"""
        return {"input_text": state["input_text"].strip()}

    def call_subgraph(state: MainState) -> MainState:
        """서브그래프 호출 (State 변환 필요)"""
        # MainState -> ProcessingState 변환
        sub_input = {
            "text": state["input_text"],
            "is_upper": False,
            "char_count": 0
        }

        # 서브그래프 실행
        sub_result = processing_subgraph.invoke(sub_input)

        # ProcessingState -> MainState 변환
        return {
            "processed_text": sub_result["text"],
            "final_result": f"처리됨: {sub_result['text']} (길이: {sub_result['char_count']})"
        }

    graph = StateGraph(MainState)
    graph.add_node("preprocess", preprocess)
    graph.add_node("process", call_subgraph)
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "process")
    graph.add_edge("process", END)

    return graph.compile()


def run_different_state_example():
    """다른 State 서브그래프 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: 다른 State를 가진 서브그래프")
    print("=" * 60)

    app = create_main_graph_with_different_state()

    result = app.invoke({
        "input_text": "  hello world  ",
        "processed_text": "",
        "final_result": ""
    })

    print(f"\n📝 입력: '  hello world  '")
    print(f"📤 결과: {result['final_result']}")


# =============================================================================
# 4. 재사용 가능한 서브그래프
# =============================================================================

class ValidationState(TypedDict):
    """검증용 State"""
    data: dict
    errors: Annotated[List[str], operator.add]
    is_valid: bool


def create_validation_subgraph():
    """재사용 가능한 검증 서브그래프"""

    def check_required_fields(state: ValidationState) -> ValidationState:
        """필수 필드 확인"""
        data = state["data"]
        required = ["name", "email"]
        errors = []

        for field in required:
            if field not in data or not data[field]:
                errors.append(f"필수 필드 누락: {field}")

        return {"errors": errors}

    def check_email_format(state: ValidationState) -> ValidationState:
        """이메일 형식 확인"""
        email = state["data"].get("email", "")
        errors = []

        if email and "@" not in email:
            errors.append("잘못된 이메일 형식")

        return {"errors": errors}

    def set_validity(state: ValidationState) -> ValidationState:
        """유효성 설정"""
        is_valid = len(state["errors"]) == 0
        return {"is_valid": is_valid}

    graph = StateGraph(ValidationState)
    graph.add_node("check_required", check_required_fields)
    graph.add_node("check_email", check_email_format)
    graph.add_node("set_validity", set_validity)

    graph.add_edge(START, "check_required")
    graph.add_edge("check_required", "check_email")
    graph.add_edge("check_email", "set_validity")
    graph.add_edge("set_validity", END)

    return graph.compile()


def run_reusable_subgraph_example():
    """재사용 가능한 서브그래프 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: 재사용 가능한 검증 서브그래프")
    print("=" * 60)

    validator = create_validation_subgraph()

    test_cases = [
        {"name": "홍길동", "email": "hong@example.com"},
        {"name": "김철수", "email": "invalid-email"},
        {"name": "", "email": ""},
    ]

    for data in test_cases:
        result = validator.invoke({
            "data": data,
            "errors": [],
            "is_valid": False
        })

        print(f"\n📋 데이터: {data}")
        print(f"   유효: {result['is_valid']}")
        if result["errors"]:
            print(f"   에러: {result['errors']}")


# =============================================================================
# 5. 중첩된 서브그래프
# =============================================================================

class NestedState(TypedDict):
    """중첩 서브그래프용 State"""
    value: int
    operations: Annotated[List[str], operator.add]


def create_inner_subgraph():
    """내부 서브그래프"""

    def add_one(state: NestedState) -> NestedState:
        return {
            "value": state["value"] + 1,
            "operations": ["inner: +1"]
        }

    graph = StateGraph(NestedState)
    graph.add_node("add_one", add_one)
    graph.add_edge(START, "add_one")
    graph.add_edge("add_one", END)

    return graph.compile()


def create_outer_subgraph():
    """외부 서브그래프 (내부 서브그래프 포함)"""

    inner = create_inner_subgraph()

    def multiply_two(state: NestedState) -> NestedState:
        return {
            "value": state["value"] * 2,
            "operations": ["outer: *2"]
        }

    graph = StateGraph(NestedState)
    graph.add_node("inner", inner)  # 내부 서브그래프
    graph.add_node("multiply", multiply_two)

    graph.add_edge(START, "inner")
    graph.add_edge("inner", "multiply")
    graph.add_edge("multiply", END)

    return graph.compile()


def create_root_graph():
    """루트 그래프 (외부 서브그래프 포함)"""

    outer = create_outer_subgraph()

    def initialize(state: NestedState) -> NestedState:
        return {"operations": ["root: 초기화"]}

    def finalize(state: NestedState) -> NestedState:
        return {"operations": [f"root: 최종값 = {state['value']}"]}

    graph = StateGraph(NestedState)
    graph.add_node("init", initialize)
    graph.add_node("outer", outer)  # 외부 서브그래프
    graph.add_node("final", finalize)

    graph.add_edge(START, "init")
    graph.add_edge("init", "outer")
    graph.add_edge("outer", "final")
    graph.add_edge("final", END)

    return graph.compile()


def run_nested_subgraph_example():
    """중첩 서브그래프 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: 중첩된 서브그래프")
    print("=" * 60)

    app = create_root_graph()

    result = app.invoke({
        "value": 5,
        "operations": []
    })

    print(f"\n📊 구조: root > outer > inner")
    print(f"   입력값: 5")
    print(f"   출력값: {result['value']}")
    print(f"   연산 순서:")
    for op in result["operations"]:
        print(f"      - {op}")


# =============================================================================
# 6. 조건부 서브그래프 호출
# =============================================================================

class ConditionalState(TypedDict):
    """조건부 서브그래프용 State"""
    mode: str
    data: str
    result: str


def create_mode_a_subgraph():
    """모드 A 서브그래프"""

    def process_a(state: ConditionalState) -> ConditionalState:
        return {"result": f"[Mode A] 처리: {state['data'].upper()}"}

    graph = StateGraph(ConditionalState)
    graph.add_node("process", process_a)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    return graph.compile()


def create_mode_b_subgraph():
    """모드 B 서브그래프"""

    def process_b(state: ConditionalState) -> ConditionalState:
        return {"result": f"[Mode B] 처리: {state['data'][::-1]}"}

    graph = StateGraph(ConditionalState)
    graph.add_node("process", process_b)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    return graph.compile()


def create_conditional_subgraph_graph():
    """조건부 서브그래프 호출 그래프"""

    mode_a = create_mode_a_subgraph()
    mode_b = create_mode_b_subgraph()

    def route_by_mode(state: ConditionalState) -> str:
        return "mode_a" if state["mode"] == "A" else "mode_b"

    graph = StateGraph(ConditionalState)
    graph.add_node("mode_a", mode_a)
    graph.add_node("mode_b", mode_b)

    graph.add_conditional_edges(
        START,
        route_by_mode,
        {
            "mode_a": "mode_a",
            "mode_b": "mode_b"
        }
    )

    graph.add_edge("mode_a", END)
    graph.add_edge("mode_b", END)

    return graph.compile()


def run_conditional_subgraph_example():
    """조건부 서브그래프 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 5: 조건부 서브그래프 호출")
    print("=" * 60)

    app = create_conditional_subgraph_graph()

    for mode in ["A", "B"]:
        result = app.invoke({
            "mode": mode,
            "data": "hello",
            "result": ""
        })
        print(f"\n📋 모드: {mode}")
        print(f"   결과: {result['result']}")


# =============================================================================
# 7. 서브그래프 패턴 정리
# =============================================================================

def explain_subgraph_patterns():
    """서브그래프 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 서브그래프 패턴 정리")
    print("=" * 60)

    print("""
서브그래프 사용 방법:

1. 그래프를 노드로 추가 (같은 State)
   subgraph = create_subgraph()
   main.add_node("sub", subgraph)

2. 노드 내에서 호출 (다른 State)
   def node(state):
       sub_input = transform(state)
       result = subgraph.invoke(sub_input)
       return reverse_transform(result)

장점:
- 모듈화: 복잡한 로직 분리
- 재사용: 여러 그래프에서 사용
- 테스트: 독립적으로 테스트 가능
- 유지보수: 변경 영향 최소화

주의사항:
- State 스키마 호환성 확인
- 중첩 깊이 관리
- 디버깅 복잡성

사용 시나리오:
- 공통 검증 로직
- 재사용 가능한 처리 파이프라인
- Multi-Agent의 Agent 구현
- 복잡한 워크플로우 분해
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 12] 서브그래프 활용")
    print("=" * 60)

    # 개념 설명
    explain_subgraph_concept()

    # 예제 실행
    run_basic_subgraph_example()
    run_different_state_example()
    run_reusable_subgraph_example()
    run_nested_subgraph_example()
    run_conditional_subgraph_example()

    # 패턴 정리
    explain_subgraph_patterns()

    print("\n" + "=" * 60)
    print("✅ Part 3 완료!")
    print("   다음: Part 4 - 프로덕션 기능")
    print("=" * 60)


if __name__ == "__main__":
    main()
