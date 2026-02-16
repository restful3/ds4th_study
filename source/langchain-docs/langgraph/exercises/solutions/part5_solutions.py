"""
Part 5: Advanced 연습 문제 해답
"""

from typing import TypedDict, Annotated, Optional, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
import time
import operator


# =============================================================================
# 문제 1: Functional API 기본
# =============================================================================

def solution_1():
    """Functional API 기본 해답"""

    @task
    def process(data: str) -> str:
        """데이터 처리"""
        return data.upper()

    @task
    def validate(data: str) -> bool:
        """유효성 검사"""
        return len(data) > 0

    @entrypoint(checkpointer=MemorySaver())
    def workflow(data: str) -> dict:
        """워크플로우"""
        # 조건부 처리
        if validate(data).result():
            processed = process(data).result()
            return {"success": True, "result": processed}
        return {"success": False, "result": None}

    # 테스트
    print("문제 1: Functional API")
    config = {"configurable": {"thread_id": "func_1"}}

    result = workflow.invoke("hello world", config)
    print(f"  결과: {result}")

    return workflow


# =============================================================================
# 문제 2: Durable Execution
# =============================================================================

class DurableState(TypedDict):
    step: int
    data: str
    result: str


def solution_2():
    """Durable Execution 해답"""

    def step_1(state: DurableState) -> DurableState:
        time.sleep(0.1)  # 시뮬레이션
        return {"step": 1, "result": "Step 1 완료"}

    def step_2(state: DurableState) -> DurableState:
        time.sleep(0.1)
        return {"step": 2, "result": "Step 2 완료"}

    def step_3(state: DurableState) -> DurableState:
        time.sleep(0.1)
        return {"step": 3, "result": "Step 3 완료"}

    graph = StateGraph(DurableState)
    graph.add_node("step1", step_1)
    graph.add_node("step2", step_2)
    graph.add_node("step3", step_3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # 테스트
    print("\n문제 2: Durable Execution")
    config = {"configurable": {"thread_id": "durable_1"}}

    result = app.invoke({"step": 0, "data": "test", "result": ""}, config)
    print(f"  정상 실행: step={result['step']}")

    # 복구 테스트 - Step 2 상태에서 재시작
    history = list(app.get_state_history(config))
    for snapshot in history:
        if snapshot.values.get("step") == 2:
            # 새 thread에서 Step 2 상태로 재시작
            resume_config = {"configurable": {"thread_id": "durable_recover"}}
            recovered = app.invoke(
                {"step": 2, "data": "test", "result": "Step 2 완료"},
                resume_config
            )
            print(f"  복구 후 실행: step={recovered['step']}")
            break

    return app


# =============================================================================
# 문제 3: 재시도 로직
# =============================================================================

class RetryState(TypedDict):
    attempt: int
    max_attempts: int
    backoff_seconds: float
    success: bool
    error: Optional[str]


def solution_3():
    """재시도 로직 해답"""

    def attempt_operation(state: RetryState) -> RetryState:
        attempt = state.get("attempt", 0) + 1
        backoff = state.get("backoff_seconds", 1.0)

        # 지수 백오프 대기
        if attempt > 1:
            wait_time = backoff * (2 ** (attempt - 2))
            time.sleep(min(wait_time, 0.1))  # 테스트용으로 짧게

        # 3번째 시도에서 성공 (시뮬레이션)
        if attempt >= 3:
            return {"attempt": attempt, "success": True, "error": None}
        else:
            return {"attempt": attempt, "success": False, "error": f"시도 {attempt} 실패"}

    def should_retry(state: RetryState) -> Literal["retry", "success", "fail"]:
        if state.get("success"):
            return "success"
        if state["attempt"] >= state["max_attempts"]:
            return "fail"
        return "retry"

    def handle_success(state: RetryState) -> RetryState:
        return {"error": None}

    def handle_failure(state: RetryState) -> RetryState:
        return {"error": f"최대 재시도({state['max_attempts']}) 초과"}

    graph = StateGraph(RetryState)
    graph.add_node("attempt", attempt_operation)
    graph.add_node("success", handle_success)
    graph.add_node("fail", handle_failure)

    graph.add_edge(START, "attempt")
    graph.add_conditional_edges(
        "attempt",
        should_retry,
        {"retry": "attempt", "success": "success", "fail": "fail"}
    )
    graph.add_edge("success", END)
    graph.add_edge("fail", END)

    app = graph.compile(checkpointer=MemorySaver())

    # 테스트
    print("\n문제 3: 재시도 로직")
    config = {"configurable": {"thread_id": "retry_1"}}

    result = app.invoke({
        "attempt": 0,
        "max_attempts": 5,
        "backoff_seconds": 1.0,
        "success": False,
        "error": None
    }, config)

    print(f"  시도 횟수: {result['attempt']}")
    print(f"  성공 여부: {result['success']}")

    return app


# =============================================================================
# 문제 4: 멱등성 보장
# =============================================================================

class IdempotentState(TypedDict):
    request_id: str
    data: str
    result: str
    skipped: bool


# 처리된 요청 저장소
PROCESSED: dict = {}


def solution_4():
    """멱등성 보장 해답"""
    global PROCESSED
    PROCESSED.clear()

    def check_and_process(state: IdempotentState) -> IdempotentState:
        request_id = state["request_id"]

        # 이미 처리됨
        if request_id in PROCESSED:
            return {
                "result": PROCESSED[request_id],
                "skipped": True
            }

        # 새로운 처리
        result = f"처리됨: {state['data'].upper()}"
        PROCESSED[request_id] = result

        return {"result": result, "skipped": False}

    graph = StateGraph(IdempotentState)
    graph.add_node("process", check_and_process)

    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    app = graph.compile(checkpointer=MemorySaver())

    # 테스트
    print("\n문제 4: 멱등성 보장")

    request_id = "REQ_001"
    for i in range(3):
        config = {"configurable": {"thread_id": f"idem_{i}"}}
        result = app.invoke({
            "request_id": request_id,
            "data": "test data",
            "result": "",
            "skipped": False
        }, config)
        print(f"  호출 {i+1}: skipped={result['skipped']}, result={result['result'][:20]}...")

    print(f"  실제 처리 횟수: {len(PROCESSED)}")

    return app


# =============================================================================
# 문제 5: 프로덕션 배포 체크리스트
# =============================================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Config:
    """환경 설정"""
    env: str = "development"
    db_url: str = "sqlite:///app.db"
    log_level: str = "INFO"

    def validate(self) -> List[str]:
        errors = []
        if self.env == "production" and "sqlite" in self.db_url:
            errors.append("프로덕션에서 SQLite 사용 불가")
        return errors


@dataclass
class Metrics:
    """메트릭 수집"""
    calls: int = 0
    errors: int = 0
    total_time: float = 0.0
    timings: List[float] = field(default_factory=list)

    def record(self, duration: float, success: bool):
        self.calls += 1
        self.total_time += duration
        self.timings.append(duration)
        if not success:
            self.errors += 1

    def report(self) -> dict:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "avg_time": self.total_time / max(self.calls, 1),
            "error_rate": self.errors / max(self.calls, 1) * 100
        }


class ProductionState(TypedDict):
    input: str
    result: str
    error: Optional[str]


def solution_5():
    """프로덕션 배포 해답"""

    # 설정
    config_obj = Config()

    # 로깅
    logger = logging.getLogger("production_app")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    # 메트릭
    metrics = Metrics()

    def process_with_monitoring(state: ProductionState) -> ProductionState:
        start = time.time()
        success = True

        try:
            logger.info(f"Processing: {state['input'][:20]}...")
            result = state["input"].upper()
            return {"result": result, "error": None}
        except Exception as e:
            success = False
            logger.error(f"Error: {e}")
            return {"result": "", "error": str(e)}
        finally:
            duration = time.time() - start
            metrics.record(duration, success)
            logger.info(f"Completed in {duration:.3f}s")

    def health_check() -> dict:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "config": {"env": config_obj.env},
            "metrics": metrics.report()
        }

    graph = StateGraph(ProductionState)
    graph.add_node("process", process_with_monitoring)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)

    app = graph.compile(checkpointer=MemorySaver())

    # 테스트
    print("\n문제 5: 프로덕션 배포")

    # 여러 번 호출
    for i in range(3):
        config = {"configurable": {"thread_id": f"prod_{i}"}}
        app.invoke({"input": f"test data {i}", "result": "", "error": None}, config)

    # 헬스 체크
    health = health_check()
    print(f"  헬스 체크: {health['status']}")
    print(f"  메트릭: {health['metrics']}")

    return app, health_check


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Part 5 해답 실행")
    print("=" * 50)

    solution_1()
    solution_2()
    solution_3()
    solution_4()
    solution_5()

    print("\n✅ 모든 해답 테스트 완료!")
