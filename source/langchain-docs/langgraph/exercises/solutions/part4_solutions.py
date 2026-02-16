"""
Part 4: Production 연습 문제 해답
"""

from typing import TypedDict, Annotated, Optional, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# 문제 1: Checkpointer 기본
# =============================================================================

class CheckpointState(TypedDict):
    step: int
    data: str
    result: str


def solution_1():
    """Checkpointer 기본 해답"""

    def step_1(state: CheckpointState) -> CheckpointState:
        return {"step": 1, "result": "Step 1 완료"}

    def step_2(state: CheckpointState) -> CheckpointState:
        return {"step": 2, "result": "Step 2 완료"}

    def step_3(state: CheckpointState) -> CheckpointState:
        return {"step": 3, "result": "Step 3 완료"}

    graph = StateGraph(CheckpointState)
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
    config = {"configurable": {"thread_id": "solution_1"}}

    # 첫 실행
    result1 = app.invoke({"step": 0, "data": "test", "result": ""}, config)
    print("문제 1: Checkpointer 기본")
    print(f"  첫 실행 결과: step={result1['step']}")

    # 히스토리 확인
    history = list(app.get_state_history(config))
    print(f"  체크포인트 수: {len(history)}")

    # Step 2 시점의 체크포인트 정보 확인
    for snapshot in history:
        if snapshot.values.get("step") == 2:
            print(f"  Step 2 시점 발견: {snapshot.values}")
            # 새 thread에서 같은 상태로 시작
            new_config = {"configurable": {"thread_id": "solution_1_resume"}}
            result2 = app.invoke({"step": 2, "data": "test", "result": "Step 2 완료"}, new_config)
            print(f"  재개 후 결과: step={result2['step']}")
            break

    return app


# =============================================================================
# 문제 2: 메시지 관리
# =============================================================================

class MessageMgmtState(TypedDict):
    messages: Annotated[list, add_messages]


def solution_2():
    """메시지 관리 해답"""

    def manage_messages(state: MessageMgmtState) -> MessageMgmtState:
        messages = state["messages"]

        # 10개 초과하면 오래된 메시지 삭제 (System 제외)
        if len(messages) > 10:
            system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

            # 오래된 메시지 삭제 (최근 8개 유지)
            to_remove = other_msgs[:-8]

            remove_messages = [
                RemoveMessage(id=msg.id)
                for msg in to_remove
                if hasattr(msg, 'id') and msg.id
            ]

            return {"messages": remove_messages}

        return {}

    def respond(state: MessageMgmtState) -> MessageMgmtState:
        return {"messages": [AIMessage(content="응답입니다", id="resp_1")]}

    graph = StateGraph(MessageMgmtState)
    graph.add_node("manage", manage_messages)
    graph.add_node("respond", respond)

    graph.add_edge(START, "manage")
    graph.add_edge("manage", "respond")
    graph.add_edge("respond", END)

    app = graph.compile(checkpointer=MemorySaver())

    # 테스트
    print("\n문제 2: 메시지 관리")

    # 15개 메시지 생성
    messages = [SystemMessage(content="System", id="sys")]
    for i in range(14):
        messages.append(HumanMessage(content=f"Message {i}", id=f"msg_{i}"))

    config = {"configurable": {"thread_id": "solution_2"}}
    result = app.invoke({"messages": messages}, config)

    print(f"  입력 메시지 수: {len(messages)}")
    print(f"  출력 메시지 수: {len(result['messages'])}")

    return app


# =============================================================================
# 문제 3: Human-in-the-Loop
# =============================================================================

class ApprovalState(TypedDict):
    amount: int
    approved: Optional[bool]
    result: str


def solution_3():
    """Human-in-the-Loop 해답"""

    def check_approval(state: ApprovalState) -> ApprovalState:
        amount = state["amount"]

        if amount >= 1000000:
            # 승인 요청
            response = interrupt({
                "message": f"{amount:,}원 결제 승인이 필요합니다",
                "options": ["approved", "rejected"]
            })

            if response == "rejected":
                return {"approved": False, "result": "결제가 거부되었습니다"}

            return {"approved": True}

        # 자동 승인
        return {"approved": True}

    def process_payment(state: ApprovalState) -> ApprovalState:
        if state.get("approved"):
            return {"result": f"{state['amount']:,}원 결제 완료"}
        return {"result": "결제 취소됨"}

    def route(state: ApprovalState) -> Literal["process", END]:
        if state.get("approved") is False:
            return END
        return "process"

    graph = StateGraph(ApprovalState)
    graph.add_node("check", check_approval)
    graph.add_node("process", process_payment)

    graph.add_edge(START, "check")
    graph.add_conditional_edges("check", route)
    graph.add_edge("process", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # 테스트 1: 소액 (자동 승인)
    print("\n문제 3: Human-in-the-Loop")
    config1 = {"configurable": {"thread_id": "solution_3_small"}}
    result1 = app.invoke({"amount": 50000, "approved": None, "result": ""}, config1)
    print(f"  소액(5만원): {result1['result']}")

    # 테스트 2: 대액 (승인 필요)
    config2 = {"configurable": {"thread_id": "solution_3_large"}}
    app.invoke({"amount": 2000000, "approved": None, "result": ""}, config2)

    # 승인
    result2 = app.invoke(Command(resume="approved"), config2)
    print(f"  대액(200만원): {result2['result']}")

    return app


# =============================================================================
# 문제 4: 스트리밍 구현
# =============================================================================

class StreamState(TypedDict):
    value: int
    result: str


def solution_4():
    """스트리밍 구현 해답"""

    def step_1(state: StreamState) -> StreamState:
        return {"value": state["value"] + 1, "result": "Step 1"}

    def step_2(state: StreamState) -> StreamState:
        return {"value": state["value"] * 2, "result": "Step 2"}

    def step_3(state: StreamState) -> StreamState:
        return {"result": f"최종: {state['value']}"}

    graph = StateGraph(StreamState)
    graph.add_node("step1", step_1)
    graph.add_node("step2", step_2)
    graph.add_node("step3", step_3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    app = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "solution_4"}}

    print("\n문제 4: 스트리밍")

    # values 모드
    print("  [values 모드]")
    for chunk in app.stream({"value": 5, "result": ""}, config, stream_mode="values"):
        print(f"    value={chunk.get('value')}")

    # updates 모드
    config2 = {"configurable": {"thread_id": "solution_4_updates"}}
    print("  [updates 모드]")
    for chunk in app.stream({"value": 5, "result": ""}, config2, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"    [{node}] {update}")

    return app


# =============================================================================
# 문제 5: Time Travel
# =============================================================================

class TimeTravelState(TypedDict):
    step: int
    value: int
    history: Annotated[List[str], lambda a, b: a + b]


def solution_5():
    """Time Travel 해답"""

    def make_step(n: int):
        def step_fn(state: TimeTravelState) -> TimeTravelState:
            new_value = state["value"] + n
            return {
                "step": n,
                "value": new_value,
                "history": [f"Step {n}: {state['value']} -> {new_value}"]
            }
        return step_fn

    graph = StateGraph(TimeTravelState)
    for i in range(1, 6):
        graph.add_node(f"step{i}", make_step(i))

    graph.add_edge(START, "step1")
    for i in range(1, 5):
        graph.add_edge(f"step{i}", f"step{i+1}")
    graph.add_edge("step5", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "solution_5"}}

    # 실행
    result = app.invoke({"step": 0, "value": 0, "history": []}, config)

    print("\n문제 5: Time Travel")
    print(f"  최종 값: {result['value']}")
    print(f"  히스토리: {result['history']}")

    # 히스토리 조회
    history = list(app.get_state_history(config))
    print(f"  체크포인트 수: {len(history)}")

    # Step 3 시점에서 Fork (새 thread로 복사)
    for snapshot in history:
        if snapshot.values.get("step") == 3:
            print(f"  Step 3 시점 값: {snapshot.values.get('value')}")

            # Fork - 값을 변경하여 새 thread에서 재실행
            fork_config = {"configurable": {"thread_id": "solution_5_fork"}}

            # Step 3 상태에서 값을 100으로 변경하여 재실행
            fork_input = {
                "step": 3,
                "value": 100,  # 변경된 값
                "history": snapshot.values.get("history", [])
            }
            fork_result = app.invoke(fork_input, fork_config)
            print(f"  Fork 후 최종 값: {fork_result['value']}")
            break

    return app


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Part 4 해답 실행")
    print("=" * 50)

    solution_1()
    solution_2()
    solution_3()
    solution_4()
    solution_5()

    print("\n✅ 모든 해답 테스트 완료!")
