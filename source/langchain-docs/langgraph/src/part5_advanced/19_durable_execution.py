"""
[Chapter 19] Durable Execution (내구성 있는 실행)

📝 설명:
    Durable Execution은 장애에도 불구하고 워크플로우가
    안정적으로 완료되도록 보장하는 패턴입니다.
    체크포인트, 재시도, 복구 메커니즘을 활용합니다.

🎯 학습 목표:
    - Durable Execution 개념 이해
    - 체크포인트 기반 복구
    - 재시도 로직 구현
    - 멱등성(Idempotency) 보장

📚 관련 문서:
    - docs/Part5-Advanced/19-durable-execution.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/durable_execution/

💻 실행 방법:
    python -m src.part5_advanced.19_durable_execution

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
import time
import random
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator


# =============================================================================
# 1. Durable Execution 개념 설명
# =============================================================================

def explain_durable_execution():
    """Durable Execution 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Durable Execution (내구성 있는 실행)")
    print("=" * 60)

    print("""
Durable Execution이란?
    시스템 장애, 네트워크 오류, 프로세스 재시작 등의
    상황에서도 워크플로우가 안정적으로 완료되도록 하는 패턴입니다.

핵심 원칙:

┌─────────────────┬────────────────────────────────────┐
│     원칙        │              설명                  │
├─────────────────┼────────────────────────────────────┤
│ 체크포인트      │ 각 단계의 상태를 영구 저장         │
│ 복구 가능성     │ 장애 후 마지막 체크포인트에서 재개 │
│ 멱등성          │ 같은 작업 반복해도 동일 결과       │
│ 재시도          │ 실패한 작업 자동 재시도            │
└─────────────────┴────────────────────────────────────┘

LangGraph의 Durable Execution:

1. 자동 체크포인팅
   - 각 노드 실행 후 상태 저장
   - SqliteSaver, PostgresSaver 등 영구 저장소 지원

2. 복구
   - 같은 thread_id로 재실행 시 자동 복구
   - 마지막 완료된 노드부터 재개

3. 재시도 로직
   - 노드 내에서 재시도 구현
   - 지수 백오프 등 전략 적용

실제 사용 사례:

- 결제 처리: 중간에 실패해도 재시도 보장
- 데이터 파이프라인: 대용량 처리 중 장애 복구
- 외부 API 호출: 네트워크 오류 시 자동 재시도
- 장기 실행 작업: 서버 재시작 후 계속 실행
""")


# =============================================================================
# 2. 기본 체크포인트 복구
# =============================================================================

class ProcessState(TypedDict):
    """처리 State"""
    data: str
    steps_completed: Annotated[List[str], operator.add]
    current_step: int
    result: str
    error: Optional[str]


def create_recoverable_graph():
    """복구 가능한 그래프"""

    def step1(state: ProcessState) -> ProcessState:
        """Step 1: 데이터 검증"""
        time.sleep(0.1)  # 시뮬레이션
        return {
            "steps_completed": ["step1: 데이터 검증 완료"],
            "current_step": 1
        }

    def step2(state: ProcessState) -> ProcessState:
        """Step 2: 데이터 변환"""
        time.sleep(0.1)
        data = state["data"]
        return {
            "steps_completed": ["step2: 데이터 변환 완료"],
            "current_step": 2,
            "result": f"변환됨: {data.upper()}"
        }

    def step3(state: ProcessState) -> ProcessState:
        """Step 3: 결과 저장"""
        time.sleep(0.1)
        return {
            "steps_completed": ["step3: 결과 저장 완료"],
            "current_step": 3
        }

    graph = StateGraph(ProcessState)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_basic_recovery_example():
    """기본 복구 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 체크포인트 기반 복구")
    print("=" * 60)

    app = create_recoverable_graph()
    config = {"configurable": {"thread_id": "durable_1"}}

    # 정상 실행
    print("\n🚀 정상 실행:")
    result = app.invoke({
        "data": "샘플 데이터",
        "steps_completed": [],
        "current_step": 0,
        "result": "",
        "error": None
    }, config=config)

    print(f"   완료된 단계: {len(result['steps_completed'])}")
    for step in result["steps_completed"]:
        print(f"      - {step}")

    # 히스토리 확인
    history = list(app.get_state_history(config))
    print(f"\n📜 체크포인트 수: {len(history)}")

    # 복구 시뮬레이션 - 새로운 thread에서 특정 체크포인트로 시작
    print("\n🔄 복구 시뮬레이션 (Step 1 이후부터):")

    # Step 1 완료 후 체크포인트 찾기
    step1_checkpoint = None
    for snapshot in history:
        if snapshot.values.get("current_step") == 1:
            step1_checkpoint = snapshot
            break

    if step1_checkpoint:
        # 해당 체크포인트에서 재개
        resume_config = {
            "configurable": {
                "thread_id": "durable_1",
                "checkpoint_id": step1_checkpoint.config["configurable"]["checkpoint_id"]
            }
        }

        resumed_result = app.invoke(None, config=resume_config)
        print(f"   복구 후 완료된 단계: {len(resumed_result['steps_completed'])}")


# =============================================================================
# 3. 재시도 로직 구현
# =============================================================================

class RetryState(TypedDict):
    """재시도 State"""
    url: str
    retry_count: int
    max_retries: int
    result: Optional[str]
    error: Optional[str]


def create_retry_graph():
    """재시도 로직이 포함된 그래프"""

    def fetch_with_retry(state: RetryState) -> RetryState:
        """재시도를 포함한 데이터 가져오기"""
        url = state["url"]
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        # 랜덤하게 실패 시뮬레이션 (처음 2번은 실패)
        if retry_count < 2:
            # 실패
            return {
                "retry_count": retry_count + 1,
                "error": f"시도 {retry_count + 1} 실패: 네트워크 오류"
            }
        else:
            # 성공
            return {
                "result": f"성공! {url}에서 데이터 가져옴",
                "error": None
            }

    def should_retry(state: RetryState) -> str:
        """재시도 여부 결정"""
        if state.get("result"):
            return "done"

        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if retry_count < max_retries:
            return "retry"
        return "fail"

    def handle_failure(state: RetryState) -> RetryState:
        """실패 처리"""
        return {
            "error": f"최대 재시도 횟수({state['max_retries']}) 초과. 최종 실패."
        }

    def finalize(state: RetryState) -> RetryState:
        """성공 처리"""
        return {"error": None}

    graph = StateGraph(RetryState)
    graph.add_node("fetch", fetch_with_retry)
    graph.add_node("fail", handle_failure)
    graph.add_node("done", finalize)

    graph.add_edge(START, "fetch")
    graph.add_conditional_edges(
        "fetch",
        should_retry,
        {"retry": "fetch", "done": "done", "fail": "fail"}
    )
    graph.add_edge("done", END)
    graph.add_edge("fail", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_retry_example():
    """재시도 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 재시도 로직 구현")
    print("=" * 60)

    app = create_retry_graph()
    config = {"configurable": {"thread_id": "retry_1"}}

    print("\n🔄 재시도 포함 실행:")
    result = app.invoke({
        "url": "https://api.example.com/data",
        "retry_count": 0,
        "max_retries": 3,
        "result": None,
        "error": None
    }, config=config)

    print(f"   재시도 횟수: {result['retry_count']}")
    print(f"   결과: {result.get('result', 'N/A')}")
    print(f"   에러: {result.get('error', '없음')}")


# =============================================================================
# 4. 지수 백오프 (Exponential Backoff)
# =============================================================================

class BackoffState(TypedDict):
    """백오프 State"""
    operation: str
    attempt: int
    max_attempts: int
    backoff_factor: float
    success: bool
    total_wait_time: float
    result: str


def create_backoff_graph():
    """지수 백오프 그래프"""

    def attempt_operation(state: BackoffState) -> BackoffState:
        """작업 시도"""
        attempt = state.get("attempt", 0) + 1
        backoff_factor = state.get("backoff_factor", 2.0)
        total_wait_time = state.get("total_wait_time", 0)

        # 지수 백오프 대기 시간 계산
        if attempt > 1:
            wait_time = (backoff_factor ** (attempt - 1)) * 0.1  # 시뮬레이션용으로 짧게
            time.sleep(wait_time)
            total_wait_time += wait_time

        # 3번째 시도에서 성공 (시뮬레이션)
        if attempt >= 3:
            return {
                "attempt": attempt,
                "success": True,
                "total_wait_time": total_wait_time,
                "result": f"{state['operation']} 성공 (시도 {attempt}회)"
            }
        else:
            return {
                "attempt": attempt,
                "success": False,
                "total_wait_time": total_wait_time,
                "result": f"시도 {attempt} 실패"
            }

    def should_continue(state: BackoffState) -> str:
        """계속 시도 여부"""
        if state.get("success"):
            return "success"

        if state["attempt"] >= state["max_attempts"]:
            return "failed"

        return "retry"

    def handle_success(state: BackoffState) -> BackoffState:
        """성공 처리"""
        return {}

    def handle_failure(state: BackoffState) -> BackoffState:
        """실패 처리"""
        return {"result": f"최대 시도 횟수 도달. 총 대기 시간: {state['total_wait_time']:.2f}초"}

    graph = StateGraph(BackoffState)
    graph.add_node("attempt", attempt_operation)
    graph.add_node("success", handle_success)
    graph.add_node("failed", handle_failure)

    graph.add_edge(START, "attempt")
    graph.add_conditional_edges(
        "attempt",
        should_continue,
        {"retry": "attempt", "success": "success", "failed": "failed"}
    )
    graph.add_edge("success", END)
    graph.add_edge("failed", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_backoff_example():
    """지수 백오프 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 지수 백오프 (Exponential Backoff)")
    print("=" * 60)

    app = create_backoff_graph()
    config = {"configurable": {"thread_id": "backoff_1"}}

    print("\n⏱️  지수 백오프 실행:")
    result = app.invoke({
        "operation": "API 호출",
        "attempt": 0,
        "max_attempts": 5,
        "backoff_factor": 2.0,
        "success": False,
        "total_wait_time": 0,
        "result": ""
    }, config=config)

    print(f"   총 시도: {result['attempt']}회")
    print(f"   성공 여부: {result['success']}")
    print(f"   총 대기 시간: {result['total_wait_time']:.2f}초")
    print(f"   결과: {result['result']}")


# =============================================================================
# 5. 멱등성 (Idempotency) 보장
# =============================================================================

class IdempotentState(TypedDict):
    """멱등성 State"""
    request_id: str
    data: str
    processed_ids: Annotated[List[str], operator.add]
    result: str


# 처리된 요청 ID를 저장하는 메모리 저장소 (실제로는 DB 사용)
PROCESSED_REQUESTS = set()


def create_idempotent_graph():
    """멱등성이 보장된 그래프"""

    def check_idempotency(state: IdempotentState) -> IdempotentState:
        """멱등성 검사"""
        request_id = state["request_id"]

        if request_id in PROCESSED_REQUESTS:
            # 이미 처리됨
            return {
                "result": f"이미 처리된 요청: {request_id}",
                "processed_ids": [f"(중복) {request_id}"]
            }
        return {}

    def should_process(state: IdempotentState) -> str:
        """처리 여부 결정"""
        if state["request_id"] in PROCESSED_REQUESTS:
            return "skip"
        return "process"

    def process_data(state: IdempotentState) -> IdempotentState:
        """데이터 처리"""
        request_id = state["request_id"]
        data = state["data"]

        # 처리 수행
        result = f"처리됨: {data.upper()}"

        # 처리 완료 기록
        PROCESSED_REQUESTS.add(request_id)

        return {
            "result": result,
            "processed_ids": [request_id]
        }

    def skip_processing(state: IdempotentState) -> IdempotentState:
        """처리 건너뛰기"""
        return {}

    graph = StateGraph(IdempotentState)
    graph.add_node("check", check_idempotency)
    graph.add_node("process", process_data)
    graph.add_node("skip", skip_processing)

    graph.add_edge(START, "check")
    graph.add_conditional_edges(
        "check",
        should_process,
        {"process": "process", "skip": "skip"}
    )
    graph.add_edge("process", END)
    graph.add_edge("skip", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_idempotency_example():
    """멱등성 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 멱등성 (Idempotency) 보장")
    print("=" * 60)

    # 이전 테스트 데이터 초기화
    PROCESSED_REQUESTS.clear()

    app = create_idempotent_graph()

    # 같은 요청 ID로 여러 번 호출
    request_id = "REQ_001"

    print(f"\n🔐 동일 요청 ID ({request_id})로 3번 호출:")

    for i in range(3):
        config = {"configurable": {"thread_id": f"idem_{i}"}}
        result = app.invoke({
            "request_id": request_id,
            "data": "중요한 데이터",
            "processed_ids": [],
            "result": ""
        }, config=config)

        print(f"\n   호출 {i+1}:")
        print(f"      결과: {result['result']}")

    print(f"\n   총 실제 처리 횟수: {len(PROCESSED_REQUESTS)}")


# =============================================================================
# 6. 트랜잭션 패턴
# =============================================================================

class TransactionState(TypedDict):
    """트랜잭션 State"""
    order_id: str
    amount: float
    steps: Annotated[List[str], operator.add]
    committed: bool
    rollback_reason: Optional[str]


def create_transaction_graph():
    """트랜잭션 그래프"""

    def validate_order(state: TransactionState) -> TransactionState:
        """주문 검증"""
        if state["amount"] <= 0:
            return {
                "steps": ["검증 실패: 금액이 0 이하"],
                "committed": False,
                "rollback_reason": "유효하지 않은 금액"
            }
        return {"steps": ["검증 완료"]}

    def reserve_inventory(state: TransactionState) -> TransactionState:
        """재고 예약"""
        if state.get("rollback_reason"):
            return {}

        # 재고 예약 시뮬레이션
        return {"steps": ["재고 예약 완료"]}

    def process_payment(state: TransactionState) -> TransactionState:
        """결제 처리"""
        if state.get("rollback_reason"):
            return {}

        # 결제 시뮬레이션 (큰 금액은 실패)
        if state["amount"] > 1000000:
            return {
                "steps": ["결제 실패"],
                "rollback_reason": "결제 한도 초과"
            }
        return {"steps": ["결제 완료"]}

    def commit_or_rollback(state: TransactionState) -> str:
        """커밋 또는 롤백 결정"""
        if state.get("rollback_reason"):
            return "rollback"
        return "commit"

    def commit_transaction(state: TransactionState) -> TransactionState:
        """트랜잭션 커밋"""
        return {
            "steps": ["트랜잭션 커밋됨"],
            "committed": True
        }

    def rollback_transaction(state: TransactionState) -> TransactionState:
        """트랜잭션 롤백"""
        return {
            "steps": [f"롤백됨: {state.get('rollback_reason', 'Unknown')}"],
            "committed": False
        }

    graph = StateGraph(TransactionState)
    graph.add_node("validate", validate_order)
    graph.add_node("reserve", reserve_inventory)
    graph.add_node("payment", process_payment)
    graph.add_node("commit", commit_transaction)
    graph.add_node("rollback", rollback_transaction)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "reserve")
    graph.add_edge("reserve", "payment")
    graph.add_conditional_edges(
        "payment",
        commit_or_rollback,
        {"commit": "commit", "rollback": "rollback"}
    )
    graph.add_edge("commit", END)
    graph.add_edge("rollback", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_transaction_example():
    """트랜잭션 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 트랜잭션 패턴")
    print("=" * 60)

    app = create_transaction_graph()

    # 성공 케이스
    print("\n💳 성공 케이스 (금액: 50,000원):")
    config1 = {"configurable": {"thread_id": "tx_success"}}
    result1 = app.invoke({
        "order_id": "ORD_001",
        "amount": 50000,
        "steps": [],
        "committed": False,
        "rollback_reason": None
    }, config=config1)

    for step in result1["steps"]:
        print(f"   - {step}")
    print(f"   최종 상태: {'커밋됨' if result1['committed'] else '롤백됨'}")

    # 실패 케이스
    print("\n💳 실패 케이스 (금액: 2,000,000원):")
    config2 = {"configurable": {"thread_id": "tx_fail"}}
    result2 = app.invoke({
        "order_id": "ORD_002",
        "amount": 2000000,
        "steps": [],
        "committed": False,
        "rollback_reason": None
    }, config=config2)

    for step in result2["steps"]:
        print(f"   - {step}")
    print(f"   최종 상태: {'커밋됨' if result2['committed'] else '롤백됨'}")


# =============================================================================
# 7. Durable Execution 패턴 정리
# =============================================================================

def explain_durable_patterns():
    """Durable Execution 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Durable Execution 패턴 정리")
    print("=" * 60)

    print("""
Durable Execution 구현 패턴:

1. 체크포인팅
   - 각 노드 후 자동 저장
   - 영구 저장소 사용 (SqliteSaver, PostgresSaver)
   - thread_id로 복구

2. 재시도 로직
   def node_with_retry(state):
       for attempt in range(max_retries):
           try:
               return do_work()
           except Exception:
               if attempt == max_retries - 1:
                   raise
               time.sleep(backoff_time)

3. 멱등성
   - 요청 ID로 중복 체크
   - 이미 처리된 요청은 건너뛰기
   - 결과 캐싱

4. 트랜잭션
   - 검증 → 예약 → 실행 → 커밋/롤백
   - 실패 시 이전 단계 롤백
   - 보상 트랜잭션 구현

베스트 프랙티스:

1. 영구 저장소 사용
   - 메모리 저장소는 개발용
   - 프로덕션은 DB 기반 저장소

2. 에러 분류
   - 재시도 가능한 에러 (네트워크)
   - 재시도 불가 에러 (비즈니스 로직)

3. 타임아웃 설정
   - 각 단계별 타임아웃
   - 전체 워크플로우 타임아웃

4. 모니터링
   - 실패율 추적
   - 재시도 횟수 메트릭
   - 알림 설정
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 19] Durable Execution (내구성 있는 실행)")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_durable_execution()

    # 예제 실행
    run_basic_recovery_example()
    run_retry_example()
    run_backoff_example()
    run_idempotency_example()
    run_transaction_example()

    # 패턴 정리
    explain_durable_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 20_deployment_ready.py (배포 준비)")
    print("=" * 60)


if __name__ == "__main__":
    main()
