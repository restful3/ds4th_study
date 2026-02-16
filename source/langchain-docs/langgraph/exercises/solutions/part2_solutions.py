"""
Part 2: Workflows 연습 문제 해답
"""

from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
import operator
import re


# =============================================================================
# 문제 1: 텍스트 처리 파이프라인
# =============================================================================

class TextState(TypedDict):
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    word_count: int


def solution_1():
    """텍스트 처리 파이프라인 해답"""

    def clean(state: TextState) -> TextState:
        text = state["raw_text"]
        cleaned = " ".join(text.lower().split())
        return {"cleaned_text": cleaned}

    def tokenize(state: TextState) -> TextState:
        tokens = state["cleaned_text"].split()
        return {"tokens": tokens}

    def count(state: TextState) -> TextState:
        return {"word_count": len(state["tokens"])}

    graph = StateGraph(TextState)
    graph.add_node("clean", clean)
    graph.add_node("tokenize", tokenize)
    graph.add_node("count", count)

    graph.add_edge(START, "clean")
    graph.add_edge("clean", "tokenize")
    graph.add_edge("tokenize", "count")
    graph.add_edge("count", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({
        "raw_text": "  Hello   WORLD   How  Are  You  ",
        "cleaned_text": "",
        "tokens": [],
        "word_count": 0
    })

    print(f"문제 1 결과:")
    print(f"  cleaned: '{result['cleaned_text']}'")
    print(f"  tokens: {result['tokens']}")
    print(f"  word_count: {result['word_count']}")

    assert result["word_count"] == 5

    return app


# =============================================================================
# 문제 2: 스마트 라우터
# =============================================================================

class RouterState(TypedDict):
    input: str
    input_type: str
    result: str


def solution_2():
    """스마트 라우터 해답"""

    def detect_type(state: RouterState) -> Literal["number", "email", "url", "text"]:
        input_str = state["input"]

        # 숫자만
        if input_str.isdigit():
            return "number"

        # 이메일
        if re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", input_str):
            return "email"

        # URL
        if re.match(r"^https?://", input_str):
            return "url"

        return "text"

    def number_handler(state: RouterState) -> RouterState:
        return {"input_type": "number", "result": f"숫자 처리: {state['input']}"}

    def email_handler(state: RouterState) -> RouterState:
        return {"input_type": "email", "result": f"이메일 처리: {state['input']}"}

    def url_handler(state: RouterState) -> RouterState:
        return {"input_type": "url", "result": f"URL 처리: {state['input']}"}

    def text_handler(state: RouterState) -> RouterState:
        return {"input_type": "text", "result": f"텍스트 처리: {state['input']}"}

    graph = StateGraph(RouterState)
    graph.add_node("number", number_handler)
    graph.add_node("email", email_handler)
    graph.add_node("url", url_handler)
    graph.add_node("text", text_handler)

    graph.add_conditional_edges(START, detect_type)
    graph.add_edge("number", END)
    graph.add_edge("email", END)
    graph.add_edge("url", END)
    graph.add_edge("text", END)

    app = graph.compile()

    # 테스트
    test_cases = [
        ("12345", "number"),
        ("test@example.com", "email"),
        ("https://example.com", "url"),
        ("hello world", "text")
    ]

    for input_val, expected_type in test_cases:
        result = app.invoke({"input": input_val, "input_type": "", "result": ""})
        print(f"문제 2 - '{input_val}': {result['input_type']}")
        assert result["input_type"] == expected_type

    return app


# =============================================================================
# 문제 3: 병렬 데이터 수집기
# =============================================================================

class CollectorState(TypedDict):
    results: Annotated[List[dict], operator.add]
    aggregated: dict


def solution_3():
    """병렬 데이터 수집기 해답"""

    def source_a(state: CollectorState) -> CollectorState:
        return {"results": [{"source": "A", "data": 100}]}

    def source_b(state: CollectorState) -> CollectorState:
        return {"results": [{"source": "B", "data": 200}]}

    def source_c(state: CollectorState) -> CollectorState:
        return {"results": [{"source": "C", "data": 300}]}

    def aggregator(state: CollectorState) -> CollectorState:
        total = sum(r["data"] for r in state["results"])
        sources = [r["source"] for r in state["results"]]
        return {"aggregated": {"total": total, "sources": sources}}

    graph = StateGraph(CollectorState)
    graph.add_node("source_a", source_a)
    graph.add_node("source_b", source_b)
    graph.add_node("source_c", source_c)
    graph.add_node("aggregator", aggregator)

    # Fan-out
    graph.add_edge(START, "source_a")
    graph.add_edge(START, "source_b")
    graph.add_edge(START, "source_c")

    # Fan-in
    graph.add_edge("source_a", "aggregator")
    graph.add_edge("source_b", "aggregator")
    graph.add_edge("source_c", "aggregator")
    graph.add_edge("aggregator", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({"results": [], "aggregated": {}})
    print(f"문제 3 결과: {result['aggregated']}")
    assert result["aggregated"]["total"] == 600

    return app


# =============================================================================
# 문제 4: Orchestrator-Worker
# =============================================================================

class OrchestratorState(TypedDict):
    tasks: List[str]
    results: Annotated[List[str], operator.add]
    current_task: str


def solution_4():
    """Orchestrator-Worker 해답"""
    from langgraph.constants import Send

    def orchestrator(state: OrchestratorState):
        """작업 분배"""
        tasks = state["tasks"]
        return [Send("worker", {"current_task": task}) for task in tasks]

    def worker(state: OrchestratorState) -> OrchestratorState:
        """작업 처리"""
        task = state["current_task"]
        result = f"result_{task}"
        return {"results": [result]}

    graph = StateGraph(OrchestratorState)
    graph.add_node("worker", worker)

    graph.add_conditional_edges(START, orchestrator, ["worker"])
    graph.add_edge("worker", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({
        "tasks": ["task1", "task2", "task3"],
        "results": [],
        "current_task": ""
    })

    print(f"문제 4 결과: {result['results']}")
    assert len(result["results"]) == 3

    return app


# =============================================================================
# 문제 5: Evaluator-Optimizer 루프
# =============================================================================

class EvalState(TypedDict):
    prompt: str
    result: str
    score: int
    attempts: int


def solution_5():
    """Evaluator-Optimizer 해답"""
    import random

    def generator(state: EvalState) -> EvalState:
        """결과 생성"""
        attempt = state.get("attempts", 0) + 1
        # 시도할수록 점수가 높아지도록 시뮬레이션
        result = f"생성된 결과 (시도 {attempt}회)"
        return {"result": result, "attempts": attempt}

    def evaluator(state: EvalState) -> EvalState:
        """평가"""
        # 시도 횟수에 따라 점수 증가 (시뮬레이션)
        base_score = 60 + state["attempts"] * 10
        score = min(base_score + random.randint(-5, 5), 100)
        return {"score": score}

    def should_retry(state: EvalState) -> Literal["pass", "retry", "fail"]:
        """재시도 여부"""
        if state["score"] >= 80:
            return "pass"
        if state["attempts"] >= 3:
            return "fail"
        return "retry"

    def pass_result(state: EvalState) -> EvalState:
        return {"result": f"✅ 통과: {state['result']} (점수: {state['score']})"}

    def fail_result(state: EvalState) -> EvalState:
        return {"result": f"❌ 실패: 최대 시도 횟수 초과 (점수: {state['score']})"}

    graph = StateGraph(EvalState)
    graph.add_node("generator", generator)
    graph.add_node("evaluator", evaluator)
    graph.add_node("pass", pass_result)
    graph.add_node("fail", fail_result)

    graph.add_edge(START, "generator")
    graph.add_edge("generator", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        should_retry,
        {"pass": "pass", "retry": "generator", "fail": "fail"}
    )
    graph.add_edge("pass", END)
    graph.add_edge("fail", END)

    app = graph.compile()

    # 테스트
    result = app.invoke({
        "prompt": "테스트 프롬프트",
        "result": "",
        "score": 0,
        "attempts": 0
    })

    print(f"문제 5 결과:")
    print(f"  시도: {result['attempts']}회")
    print(f"  점수: {result['score']}")
    print(f"  결과: {result['result']}")

    return app


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Part 2 해답 실행")
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
