"""
[Chapter 7] 병렬 실행

📝 설명:
    병렬 실행은 여러 노드를 동시에 실행하여 처리 시간을 단축하는 패턴입니다.
    Fan-out/Fan-in 패턴과 Send API를 사용한 동적 병렬 실행을 학습합니다.

🎯 학습 목표:
    - 병렬 노드 실행 원리 이해
    - Fan-out / Fan-in 패턴 구현
    - Send API를 사용한 동적 워커 생성
    - 병렬 결과 수집 및 집계

📚 관련 문서:
    - docs/Part2-Workflows/07-parallel-execution.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#parallelization

💻 실행 방법:
    python -m src.part2_workflows.07_parallelization

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
import time
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# =============================================================================
# 1. 기본 병렬 실행 (Fan-out/Fan-in)
# =============================================================================

class ParallelState(TypedDict):
    """병렬 처리를 위한 State"""
    input_data: str
    results: Annotated[List[str], operator.add]  # 결과 누적


def task_a(state: ParallelState) -> ParallelState:
    """Task A - 대문자 변환"""
    time.sleep(0.1)  # 시뮬레이션
    result = f"A: {state['input_data'].upper()}"
    return {"results": [result]}


def task_b(state: ParallelState) -> ParallelState:
    """Task B - 길이 계산"""
    time.sleep(0.1)  # 시뮬레이션
    result = f"B: 길이={len(state['input_data'])}"
    return {"results": [result]}


def task_c(state: ParallelState) -> ParallelState:
    """Task C - 단어 수 계산"""
    time.sleep(0.1)  # 시뮬레이션
    word_count = len(state['input_data'].split())
    result = f"C: 단어={word_count}개"
    return {"results": [result]}


def aggregate(state: ParallelState) -> ParallelState:
    """결과 집계"""
    summary = f"[집계] {len(state['results'])}개 작업 완료"
    return {"results": [summary]}


def create_parallel_graph():
    """기본 병렬 그래프 생성"""
    graph = StateGraph(ParallelState)

    # 병렬로 실행될 노드들
    graph.add_node("task_a", task_a)
    graph.add_node("task_b", task_b)
    graph.add_node("task_c", task_c)
    graph.add_node("aggregate", aggregate)

    # Fan-out: START에서 모든 작업으로 분기
    graph.add_edge(START, "task_a")
    graph.add_edge(START, "task_b")
    graph.add_edge(START, "task_c")

    # Fan-in: 모든 작업에서 aggregate로 수렴
    graph.add_edge("task_a", "aggregate")
    graph.add_edge("task_b", "aggregate")
    graph.add_edge("task_c", "aggregate")

    graph.add_edge("aggregate", END)

    return graph.compile()


def run_basic_parallel_example():
    """기본 병렬 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 병렬 실행 (Fan-out/Fan-in)")
    print("=" * 60)

    app = create_parallel_graph()

    start_time = time.time()
    result = app.invoke({
        "input_data": "Hello LangGraph World",
        "results": []
    })
    elapsed = time.time() - start_time

    print(f"\n📊 결과:")
    for r in result["results"]:
        print(f"   {r}")

    print(f"\n⏱️  소요 시간: {elapsed:.2f}초")
    print(f"   (순차 실행이었다면 ~0.3초 이상 소요)")


# =============================================================================
# 2. Send API를 사용한 동적 병렬 실행
# =============================================================================

class DynamicParallelState(TypedDict):
    """동적 병렬 처리를 위한 State"""
    items: List[str]
    current_item: str  # 개별 워커용
    processed: Annotated[List[str], operator.add]


def distribute_work(state: DynamicParallelState) -> List[Send]:
    """
    작업을 동적으로 분배

    Send API를 사용하여 각 아이템에 대해 워커 노드를 생성
    """
    items = state["items"]

    # 각 아이템에 대해 Send 객체 생성
    # Send(node_name, state_update)
    return [
        Send("worker", {"current_item": item})
        for item in items
    ]


def worker(state: DynamicParallelState) -> DynamicParallelState:
    """개별 아이템을 처리하는 워커"""
    item = state["current_item"]
    processed = f"처리됨: {item.upper()}"
    return {"processed": [processed]}


def collect_results(state: DynamicParallelState) -> DynamicParallelState:
    """결과 수집"""
    total = len(state["processed"])
    return {"processed": [f"총 {total}개 아이템 처리 완료"]}


def create_dynamic_parallel_graph():
    """동적 병렬 그래프 생성"""
    graph = StateGraph(DynamicParallelState)

    graph.add_node("distributor", distribute_work)
    graph.add_node("worker", worker)
    graph.add_node("collector", collect_results)

    graph.add_edge(START, "distributor")
    # distributor는 Send를 반환하므로 자동으로 worker로 분기
    graph.add_edge("worker", "collector")
    graph.add_edge("collector", END)

    return graph.compile()


def run_dynamic_parallel_example():
    """동적 병렬 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 2: Send API를 사용한 동적 병렬 실행")
    print("=" * 60)

    app = create_dynamic_parallel_graph()

    result = app.invoke({
        "items": ["apple", "banana", "cherry", "date", "elderberry"],
        "current_item": "",
        "processed": []
    })

    print(f"\n📊 처리 결과:")
    for p in result["processed"]:
        print(f"   {p}")


# =============================================================================
# 3. 조건부 병렬 실행
# =============================================================================

class ConditionalParallelState(TypedDict):
    """조건부 병렬 처리를 위한 State"""
    number: int
    analyses: Annotated[List[str], operator.add]


def analyze_even(state: ConditionalParallelState) -> ConditionalParallelState:
    """짝수 분석"""
    return {"analyses": [f"짝수 분석: {state['number']} / 2 = {state['number'] // 2}"]}


def analyze_odd(state: ConditionalParallelState) -> ConditionalParallelState:
    """홀수 분석"""
    return {"analyses": [f"홀수 분석: {state['number']} * 3 + 1 = {state['number'] * 3 + 1}"]}


def analyze_prime(state: ConditionalParallelState) -> ConditionalParallelState:
    """소수 분석"""
    n = state["number"]
    is_prime = n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))
    return {"analyses": [f"소수 여부: {'예' if is_prime else '아니오'}"]}


def analyze_size(state: ConditionalParallelState) -> ConditionalParallelState:
    """크기 분석"""
    n = state["number"]
    size = "작음" if n < 10 else ("중간" if n < 100 else "큼")
    return {"analyses": [f"크기: {size}"]}


def route_parallel_analyses(state: ConditionalParallelState) -> List[Send]:
    """조건에 따라 병렬 분석 작업 생성"""
    number = state["number"]
    sends = []

    # 항상 실행
    sends.append(Send("analyze_size", state))

    # 조건부 실행
    if number % 2 == 0:
        sends.append(Send("analyze_even", state))
    else:
        sends.append(Send("analyze_odd", state))

    # 양수일 때만 소수 검사
    if number > 0:
        sends.append(Send("analyze_prime", state))

    return sends


def create_conditional_parallel_graph():
    """조건부 병렬 그래프 생성"""
    graph = StateGraph(ConditionalParallelState)

    graph.add_node("router", route_parallel_analyses)
    graph.add_node("analyze_even", analyze_even)
    graph.add_node("analyze_odd", analyze_odd)
    graph.add_node("analyze_prime", analyze_prime)
    graph.add_node("analyze_size", analyze_size)

    graph.add_edge(START, "router")
    graph.add_edge("analyze_even", END)
    graph.add_edge("analyze_odd", END)
    graph.add_edge("analyze_prime", END)
    graph.add_edge("analyze_size", END)

    return graph.compile()


def run_conditional_parallel_example():
    """조건부 병렬 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 조건부 병렬 실행")
    print("=" * 60)

    app = create_conditional_parallel_graph()

    test_numbers = [12, 17, 100]

    for num in test_numbers:
        result = app.invoke({
            "number": num,
            "analyses": []
        })
        print(f"\n📊 숫자: {num}")
        for analysis in result["analyses"]:
            print(f"   {analysis}")


# =============================================================================
# 4. Map-Reduce 패턴
# =============================================================================

class MapReduceState(TypedDict):
    """Map-Reduce를 위한 State"""
    documents: List[str]
    current_doc: str
    summaries: Annotated[List[str], operator.add]
    final_summary: str


def map_documents(state: MapReduceState) -> List[Send]:
    """문서를 개별 요약 작업으로 분배 (Map)"""
    return [
        Send("summarize", {"current_doc": doc})
        for doc in state["documents"]
    ]


def summarize(state: MapReduceState) -> MapReduceState:
    """개별 문서 요약 (Mapper)"""
    doc = state["current_doc"]
    # 간단한 요약: 첫 20자 + ...
    summary = doc[:20] + "..." if len(doc) > 20 else doc
    return {"summaries": [summary]}


def reduce_summaries(state: MapReduceState) -> MapReduceState:
    """요약들을 합쳐서 최종 요약 생성 (Reduce)"""
    all_summaries = state["summaries"]
    final = f"총 {len(all_summaries)}개 문서 요약:\n"
    for i, s in enumerate(all_summaries, 1):
        final += f"  {i}. {s}\n"
    return {"final_summary": final}


def create_map_reduce_graph():
    """Map-Reduce 그래프 생성"""
    graph = StateGraph(MapReduceState)

    graph.add_node("mapper", map_documents)
    graph.add_node("summarize", summarize)
    graph.add_node("reducer", reduce_summaries)

    graph.add_edge(START, "mapper")
    graph.add_edge("summarize", "reducer")
    graph.add_edge("reducer", END)

    return graph.compile()


def run_map_reduce_example():
    """Map-Reduce 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: Map-Reduce 패턴")
    print("=" * 60)

    app = create_map_reduce_graph()

    documents = [
        "LangGraph는 LLM 애플리케이션을 구축하기 위한 프레임워크입니다.",
        "Python은 가장 인기 있는 프로그래밍 언어 중 하나입니다.",
        "인공지능은 현대 기술의 핵심 분야로 빠르게 발전하고 있습니다.",
        "클라우드 컴퓨팅은 IT 인프라의 패러다임을 바꾸었습니다."
    ]

    result = app.invoke({
        "documents": documents,
        "current_doc": "",
        "summaries": [],
        "final_summary": ""
    })

    print(f"\n📊 {result['final_summary']}")


# =============================================================================
# 5. 타임아웃과 에러 처리
# =============================================================================

class RobustParallelState(TypedDict):
    """견고한 병렬 처리를 위한 State"""
    tasks: List[str]
    current_task: str
    results: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]


def distribute_robust_tasks(state: RobustParallelState) -> List[Send]:
    """작업 분배"""
    return [
        Send("process_task", {"current_task": task})
        for task in state["tasks"]
    ]


def process_task(state: RobustParallelState) -> RobustParallelState:
    """작업 처리 (에러 처리 포함)"""
    task = state["current_task"]

    try:
        # 의도적으로 일부 작업 실패
        if "fail" in task.lower():
            raise ValueError(f"'{task}' 처리 중 에러 발생")

        result = f"✅ {task}: 성공"
        return {"results": [result]}

    except Exception as e:
        error = f"❌ {task}: {str(e)}"
        return {"errors": [error]}


def summarize_results(state: RobustParallelState) -> RobustParallelState:
    """결과 요약"""
    success_count = len(state["results"])
    error_count = len(state["errors"])
    summary = f"완료: 성공 {success_count}개, 실패 {error_count}개"
    return {"results": [summary]}


def create_robust_parallel_graph():
    """견고한 병렬 그래프 생성"""
    graph = StateGraph(RobustParallelState)

    graph.add_node("distributor", distribute_robust_tasks)
    graph.add_node("process_task", process_task)
    graph.add_node("summarizer", summarize_results)

    graph.add_edge(START, "distributor")
    graph.add_edge("process_task", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


def run_robust_parallel_example():
    """견고한 병렬 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 에러 처리가 있는 병렬 실행")
    print("=" * 60)

    app = create_robust_parallel_graph()

    result = app.invoke({
        "tasks": ["Task-1", "Task-fail-2", "Task-3", "Task-fail-4", "Task-5"],
        "current_task": "",
        "results": [],
        "errors": []
    })

    print(f"\n📊 성공:")
    for r in result["results"]:
        print(f"   {r}")

    if result["errors"]:
        print(f"\n❌ 에러:")
        for e in result["errors"]:
            print(f"   {e}")


# =============================================================================
# 6. 병렬 실행 패턴 정리
# =============================================================================

def explain_parallel_patterns():
    """병렬 실행 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 병렬 실행 패턴 정리")
    print("=" * 60)

    print("""
1. 정적 Fan-out/Fan-in
   - 고정된 수의 병렬 노드
   - 동일한 입력, 다른 처리
   - add_edge(START, node)로 분기

   START ──┬── task_a ──┐
           ├── task_b ──├── aggregate ── END
           └── task_c ──┘

2. 동적 병렬 (Send API)
   - 런타임에 결정되는 병렬 작업
   - Send(node_name, state_update) 반환
   - 각 Send가 별도의 State로 노드 실행

   def distribute(state):
       return [Send("worker", {...}) for item in items]

3. Map-Reduce
   - 대량 데이터의 병렬 처리
   - Map: 데이터를 개별 작업으로 분배
   - Reduce: 결과를 집계

4. Reducer의 역할
   - 병렬 노드의 결과를 합치는 핵심
   - Annotated[List, operator.add] 사용
   - 각 노드의 결과가 자동으로 누적

팁:
- 독립적인 작업은 병렬로 처리
- I/O 바운드 작업에 특히 효과적
- 에러 처리를 반드시 포함
- State 충돌을 피하기 위해 Reducer 활용
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 7] 병렬 실행")
    print("=" * 60)

    load_dotenv()

    # 예제 실행
    run_basic_parallel_example()
    run_dynamic_parallel_example()
    run_conditional_parallel_example()
    run_map_reduce_example()
    run_robust_parallel_example()

    # 패턴 정리
    explain_parallel_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 08_orchestrator_worker.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
