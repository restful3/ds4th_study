"""
Part 5: Advanced 테스트

고급 기능 테스트 (Functional API, Durable Execution, Deployment)
"""

import pytest
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class TestFunctionalPatterns:
    """함수형 패턴 테스트"""

    def test_pipeline_pattern(self):
        """파이프라인 패턴"""
        class PipeState(TypedDict):
            data: str
            steps: Annotated[list, operator.add]

        def step_a(state: PipeState) -> PipeState:
            return {
                "data": state["data"] + "_A",
                "steps": ["A"]
            }

        def step_b(state: PipeState) -> PipeState:
            return {
                "data": state["data"] + "_B",
                "steps": ["B"]
            }

        def step_c(state: PipeState) -> PipeState:
            return {
                "data": state["data"] + "_C",
                "steps": ["C"]
            }

        graph = StateGraph(PipeState)
        graph.add_node("a", step_a)
        graph.add_node("b", step_b)
        graph.add_node("c", step_c)

        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)

        app = graph.compile()
        result = app.invoke({"data": "start", "steps": []})

        assert result["data"] == "start_A_B_C"
        assert result["steps"] == ["A", "B", "C"]


class TestRetryPattern:
    """재시도 패턴 테스트"""

    def test_retry_logic(self):
        """재시도 로직"""
        class RetryState(TypedDict):
            attempt: int
            max_attempts: int
            success: bool
            result: str

        def attempt_operation(state: RetryState) -> RetryState:
            attempt = state["attempt"] + 1
            # 3번째 시도에서 성공
            if attempt >= 3:
                return {
                    "attempt": attempt,
                    "success": True,
                    "result": "성공!"
                }
            return {
                "attempt": attempt,
                "success": False,
                "result": f"실패 (시도 {attempt})"
            }

        def should_retry(state: RetryState) -> str:
            if state["success"]:
                return "done"
            if state["attempt"] >= state["max_attempts"]:
                return "failed"
            return "retry"

        def done_node(state: RetryState) -> RetryState:
            return {}

        def failed_node(state: RetryState) -> RetryState:
            return {"result": "최대 시도 초과"}

        graph = StateGraph(RetryState)
        graph.add_node("attempt", attempt_operation)
        graph.add_node("done", done_node)
        graph.add_node("failed", failed_node)

        graph.add_edge(START, "attempt")
        graph.add_conditional_edges(
            "attempt",
            should_retry,
            {"retry": "attempt", "done": "done", "failed": "failed"}
        )
        graph.add_edge("done", END)
        graph.add_edge("failed", END)

        app = graph.compile()
        result = app.invoke({
            "attempt": 0,
            "max_attempts": 5,
            "success": False,
            "result": ""
        })

        assert result["success"] == True
        assert result["attempt"] == 3


class TestIdempotency:
    """멱등성 테스트"""

    def test_idempotent_operation(self):
        """멱등성 보장"""
        processed_ids = set()

        class IdempotentState(TypedDict):
            request_id: str
            data: str
            processed: bool
            result: str

        def check_and_process(state: IdempotentState) -> IdempotentState:
            request_id = state["request_id"]

            if request_id in processed_ids:
                return {
                    "processed": True,
                    "result": "이미 처리됨 (캐시)"
                }

            processed_ids.add(request_id)
            return {
                "processed": True,
                "result": f"처리됨: {state['data']}"
            }

        graph = StateGraph(IdempotentState)
        graph.add_node("process", check_and_process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)

        app = graph.compile()

        # 첫 번째 처리
        result1 = app.invoke({
            "request_id": "req_001",
            "data": "데이터",
            "processed": False,
            "result": ""
        })
        assert "처리됨: 데이터" in result1["result"]

        # 동일 ID로 재처리
        result2 = app.invoke({
            "request_id": "req_001",
            "data": "데이터",
            "processed": False,
            "result": ""
        })
        assert "이미 처리됨" in result2["result"]


class TestTransactionPattern:
    """트랜잭션 패턴 테스트"""

    def test_commit_rollback(self):
        """커밋/롤백 패턴"""
        class TxState(TypedDict):
            amount: int
            validated: bool
            committed: bool
            error: str

        def validate(state: TxState) -> TxState:
            if state["amount"] <= 0:
                return {
                    "validated": False,
                    "error": "금액은 0보다 커야 합니다"
                }
            return {"validated": True}

        def commit(state: TxState) -> TxState:
            return {"committed": True}

        def rollback(state: TxState) -> TxState:
            return {"committed": False}

        def route(state: TxState) -> str:
            if state.get("validated"):
                return "commit"
            return "rollback"

        graph = StateGraph(TxState)
        graph.add_node("validate", validate)
        graph.add_node("commit", commit)
        graph.add_node("rollback", rollback)

        graph.add_edge(START, "validate")
        graph.add_conditional_edges(
            "validate",
            route,
            {"commit": "commit", "rollback": "rollback"}
        )
        graph.add_edge("commit", END)
        graph.add_edge("rollback", END)

        app = graph.compile()

        # 유효한 금액 - 커밋
        result1 = app.invoke({
            "amount": 100,
            "validated": False,
            "committed": False,
            "error": ""
        })
        assert result1["committed"] == True

        # 유효하지 않은 금액 - 롤백
        result2 = app.invoke({
            "amount": -50,
            "validated": False,
            "committed": False,
            "error": ""
        })
        assert result2["committed"] == False
        assert "0보다 커야" in result2["error"]


class TestRecovery:
    """복구 테스트"""

    def test_checkpoint_recovery(self):
        """체크포인트 복구"""
        class RecoveryState(TypedDict):
            step: int
            data: str

        def step1(state: RecoveryState) -> RecoveryState:
            return {"step": 1, "data": "step1_done"}

        def step2(state: RecoveryState) -> RecoveryState:
            return {"step": 2, "data": "step2_done"}

        graph = StateGraph(RecoveryState)
        graph.add_node("s1", step1)
        graph.add_node("s2", step2)
        graph.add_edge(START, "s1")
        graph.add_edge("s1", "s2")
        graph.add_edge("s2", END)

        checkpointer = MemorySaver()
        app = graph.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "recovery_test"}}

        # 전체 실행
        app.invoke({"step": 0, "data": ""}, config=config)

        # 히스토리에서 step1 완료 시점 찾기
        history = list(app.get_state_history(config))

        step1_checkpoint = None
        for snapshot in history:
            if snapshot.values.get("step") == 1:
                step1_checkpoint = snapshot
                break

        assert step1_checkpoint is not None

        # 해당 시점에서 복구 가능 확인
        checkpoint_id = step1_checkpoint.config["configurable"]["checkpoint_id"]
        assert checkpoint_id is not None
