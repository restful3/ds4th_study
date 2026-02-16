"""
[Chapter 16] Human-in-the-Loop (인터럽트)

📝 설명:
    Human-in-the-Loop은 그래프 실행 중 사람의 개입을 허용하는 패턴입니다.
    중요한 결정, 승인, 검증이 필요한 경우에 사용합니다.

🎯 학습 목표:
    - interrupt() 함수 사용법
    - interrupt_before / interrupt_after 설정
    - 사용자 입력 처리
    - 승인 워크플로우 구현

📚 관련 문서:
    - docs/Part4-Production/16-interrupts.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/

💻 실행 방법:
    python -m src.part4_production.16_interrupts

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =============================================================================
# 1. Human-in-the-Loop 개념 설명
# =============================================================================

def explain_human_in_the_loop():
    """Human-in-the-Loop 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Human-in-the-Loop (HITL)")
    print("=" * 60)

    print("""
Human-in-the-Loop이란?
    AI 시스템의 실행 중간에 사람이 개입하여
    검토, 승인, 수정을 할 수 있는 패턴입니다.

사용 사례:

┌─────────────────┬────────────────────────────────────┐
│     사례        │              설명                  │
├─────────────────┼────────────────────────────────────┤
│ 결제 승인       │ 금액이 큰 결제 전 사람의 확인      │
│ 콘텐츠 검토     │ 민감한 콘텐츠 발행 전 검토         │
│ 데이터 확인     │ 중요 데이터 변경 전 검증           │
│ 예외 처리       │ AI가 처리 못하는 상황 위임         │
└─────────────────┴────────────────────────────────────┘

LangGraph의 HITL 방법:

1. interrupt() 함수
   - 그래프 실행 중간에 멈춤
   - 사용자 입력 대기
   - 입력 받은 후 재개

2. interrupt_before / interrupt_after
   - 특정 노드 전/후에 자동 멈춤
   - 컴파일 시 설정

3. Command 객체
   - 재개 시 다음 경로 지정
   - 상태 업데이트 포함

주의사항:
    - Checkpointer 필수 (상태 저장 필요)
    - 비동기 환경에서 더 유용
    - 타임아웃 처리 고려
""")


# =============================================================================
# 2. interrupt() 함수 기본 사용법
# =============================================================================

class ApprovalState(TypedDict):
    """승인 워크플로우 State"""
    request: str
    amount: float
    approved: Optional[bool]
    approver: Optional[str]
    result: str


def create_approval_graph():
    """승인 워크플로우 그래프"""

    def analyze_request(state: ApprovalState) -> ApprovalState:
        """요청 분석"""
        request = state["request"]
        amount = state["amount"]
        return {"result": f"요청 분석 완료: {request} (금액: {amount:,.0f}원)"}

    def request_approval(state: ApprovalState) -> ApprovalState:
        """승인 요청 (interrupt 사용)"""
        amount = state["amount"]

        # 금액이 100만원 이상이면 승인 필요
        if amount >= 1000000:
            # interrupt()로 실행 중단
            approval = interrupt({
                "type": "approval_request",
                "message": f"{amount:,.0f}원 결제 승인이 필요합니다.",
                "options": ["승인", "거절"]
            })

            # 사용자가 재개할 때 여기서 계속됨
            return {
                "approved": approval.get("approved", False),
                "approver": approval.get("approver", "Unknown")
            }
        else:
            # 소액은 자동 승인
            return {
                "approved": True,
                "approver": "Auto-approved"
            }

    def process_result(state: ApprovalState) -> ApprovalState:
        """결과 처리"""
        if state.get("approved"):
            return {
                "result": f"✅ 승인됨 (승인자: {state.get('approver', 'N/A')})"
            }
        else:
            return {
                "result": f"❌ 거절됨 (처리자: {state.get('approver', 'N/A')})"
            }

    graph = StateGraph(ApprovalState)
    graph.add_node("analyze", analyze_request)
    graph.add_node("approval", request_approval)
    graph.add_node("process", process_result)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "approval")
    graph.add_edge("approval", "process")
    graph.add_edge("process", END)

    # Checkpointer 필수!
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_interrupt_example():
    """interrupt() 예제"""
    print("\n" + "=" * 60)
    print("예제 1: interrupt() 기본 사용")
    print("=" * 60)

    app = create_approval_graph()
    config = {"configurable": {"thread_id": "approval_1"}}

    # 1. 소액 요청 (자동 승인)
    print("\n💰 소액 요청 (50만원):")
    result = app.invoke({
        "request": "사무용품 구매",
        "amount": 500000,
        "approved": None,
        "approver": None,
        "result": ""
    }, config={"configurable": {"thread_id": "small_1"}})
    print(f"   결과: {result['result']}")

    # 2. 대액 요청 (승인 필요)
    print("\n💰 대액 요청 (200만원):")
    config = {"configurable": {"thread_id": "large_1"}}

    # 첫 번째 invoke - interrupt에서 멈춤
    result = app.invoke({
        "request": "노트북 구매",
        "amount": 2000000,
        "approved": None,
        "approver": None,
        "result": ""
    }, config=config)

    # interrupt 상태 확인
    state = app.get_state(config)
    print(f"   상태: {state.next}")  # 다음 노드 확인

    if state.next:
        print("   ⏸️  승인 대기 중...")

        # 사용자 승인 시뮬레이션 - Command로 재개
        print("   👤 관리자가 승인함")

        # 승인 정보와 함께 재개
        result = app.invoke(
            Command(
                resume={"approved": True, "approver": "김관리자"}
            ),
            config=config
        )
        print(f"   결과: {result['result']}")


# =============================================================================
# 3. interrupt_before / interrupt_after
# =============================================================================

class TaskState(TypedDict):
    """작업 State"""
    task: str
    validated: bool
    executed: bool
    result: str


def create_interrupt_before_graph():
    """interrupt_before 그래프"""

    def validate_task(state: TaskState) -> TaskState:
        """작업 검증"""
        return {"validated": True, "result": "검증 완료"}

    def execute_task(state: TaskState) -> TaskState:
        """작업 실행"""
        return {"executed": True, "result": f"'{state['task']}' 실행 완료"}

    def finalize(state: TaskState) -> TaskState:
        """작업 마무리"""
        return {"result": f"최종 결과: {state['result']}"}

    graph = StateGraph(TaskState)
    graph.add_node("validate", validate_task)
    graph.add_node("execute", execute_task)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "execute")
    graph.add_edge("execute", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = MemorySaver()

    # execute 노드 전에 자동으로 멈춤
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"]
    )


def run_interrupt_before_example():
    """interrupt_before 예제"""
    print("\n" + "=" * 60)
    print("예제 2: interrupt_before 사용")
    print("=" * 60)

    app = create_interrupt_before_graph()
    config = {"configurable": {"thread_id": "task_1"}}

    # 첫 번째 실행 - execute 전에 멈춤
    result = app.invoke({
        "task": "데이터베이스 마이그레이션",
        "validated": False,
        "executed": False,
        "result": ""
    }, config=config)

    state = app.get_state(config)
    print(f"\n📋 현재 상태:")
    print(f"   검증됨: {result.get('validated')}")
    print(f"   다음 노드: {state.next}")

    if state.next and "execute" in state.next:
        print("\n⏸️  실행 전 확인 대기 중...")
        print("   👤 사용자가 실행 승인함")

        # None을 전달하여 재개 (상태 변경 없이)
        result = app.invoke(None, config=config)
        print(f"\n✅ 최종 결과: {result['result']}")


# =============================================================================
# 4. 다중 인터럽트 처리
# =============================================================================

class MultiStepState(TypedDict):
    """다중 단계 State"""
    data: str
    step1_approved: bool
    step2_approved: bool
    final_result: str


def create_multi_interrupt_graph():
    """다중 인터럽트 그래프"""

    def step1(state: MultiStepState) -> MultiStepState:
        """Step 1: 데이터 준비"""
        data = state["data"]
        approval = interrupt({
            "step": 1,
            "message": f"Step 1 완료: '{data}' 처리됨. 계속할까요?"
        })
        return {"step1_approved": approval.get("continue", False)}

    def step2(state: MultiStepState) -> MultiStepState:
        """Step 2: 데이터 변환"""
        if not state.get("step1_approved"):
            return {"final_result": "Step 1에서 중단됨"}

        approval = interrupt({
            "step": 2,
            "message": "Step 2 완료: 변환됨. 최종 적용할까요?"
        })
        return {"step2_approved": approval.get("continue", False)}

    def finalize(state: MultiStepState) -> MultiStepState:
        """최종 처리"""
        if state.get("step2_approved"):
            return {"final_result": "✅ 모든 단계 완료"}
        elif state.get("step1_approved"):
            return {"final_result": "⚠️ Step 2에서 중단됨"}
        else:
            return {"final_result": "❌ Step 1에서 중단됨"}

    graph = StateGraph(MultiStepState)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "finalize")
    graph.add_edge("finalize", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_multi_interrupt_example():
    """다중 인터럽트 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 다중 인터럽트 처리")
    print("=" * 60)

    app = create_multi_interrupt_graph()
    config = {"configurable": {"thread_id": "multi_1"}}

    # Step 1 시작
    print("\n🚀 다중 단계 프로세스 시작")
    result = app.invoke({
        "data": "중요 데이터",
        "step1_approved": False,
        "step2_approved": False,
        "final_result": ""
    }, config=config)

    # Step 1 승인
    state = app.get_state(config)
    if state.next:
        print(f"\n⏸️  Step 1 완료, 승인 대기...")
        print("   👤 Step 1 승인")
        result = app.invoke(
            Command(resume={"continue": True}),
            config=config
        )

    # Step 2 승인
    state = app.get_state(config)
    if state.next:
        print(f"\n⏸️  Step 2 완료, 승인 대기...")
        print("   👤 Step 2 승인")
        result = app.invoke(
            Command(resume={"continue": True}),
            config=config
        )

    print(f"\n📊 최종 결과: {result['final_result']}")


# =============================================================================
# 5. 조건부 인터럽트
# =============================================================================

class ConditionalState(TypedDict):
    """조건부 인터럽트 State"""
    action: str
    risk_level: str
    approved: bool
    result: str


def create_conditional_interrupt_graph():
    """조건부 인터럽트 그래프"""

    def assess_risk(state: ConditionalState) -> ConditionalState:
        """리스크 평가"""
        action = state["action"]

        # 간단한 리스크 평가 로직
        high_risk_actions = ["delete", "modify", "transfer"]
        medium_risk_actions = ["update", "create"]

        if any(a in action.lower() for a in high_risk_actions):
            return {"risk_level": "high"}
        elif any(a in action.lower() for a in medium_risk_actions):
            return {"risk_level": "medium"}
        else:
            return {"risk_level": "low"}

    def maybe_interrupt(state: ConditionalState) -> ConditionalState:
        """조건에 따라 인터럽트"""
        risk_level = state.get("risk_level", "low")

        if risk_level == "high":
            # 고위험은 반드시 승인 필요
            approval = interrupt({
                "type": "high_risk_approval",
                "message": f"⚠️ 고위험 작업: '{state['action']}' 승인 필요"
            })
            return {"approved": approval.get("approved", False)}

        elif risk_level == "medium":
            # 중위험은 알림만 (자동 진행)
            print(f"   ℹ️ 중위험 알림: '{state['action']}'")
            return {"approved": True}

        else:
            # 저위험은 바로 진행
            return {"approved": True}

    def execute_action(state: ConditionalState) -> ConditionalState:
        """작업 실행"""
        if state.get("approved"):
            return {"result": f"✅ '{state['action']}' 실행 완료"}
        else:
            return {"result": f"❌ '{state['action']}' 거절됨"}

    graph = StateGraph(ConditionalState)
    graph.add_node("assess", assess_risk)
    graph.add_node("interrupt", maybe_interrupt)
    graph.add_node("execute", execute_action)

    graph.add_edge(START, "assess")
    graph.add_edge("assess", "interrupt")
    graph.add_edge("interrupt", "execute")
    graph.add_edge("execute", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_conditional_interrupt_example():
    """조건부 인터럽트 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 조건부 인터럽트")
    print("=" * 60)

    app = create_conditional_interrupt_graph()

    test_cases = [
        ("read data", "조회 작업 (저위험)"),
        ("update settings", "설정 변경 (중위험)"),
        ("delete all records", "전체 삭제 (고위험)"),
    ]

    for i, (action, desc) in enumerate(test_cases):
        config = {"configurable": {"thread_id": f"cond_{i}"}}
        print(f"\n🔹 {desc}:")

        result = app.invoke({
            "action": action,
            "risk_level": "",
            "approved": False,
            "result": ""
        }, config=config)

        state = app.get_state(config)

        if state.next:
            print(f"   ⏸️  승인 대기 중...")
            print(f"   👤 관리자 승인")
            result = app.invoke(
                Command(resume={"approved": True}),
                config=config
            )

        print(f"   결과: {result['result']}")


# =============================================================================
# 6. 타임아웃 및 에러 처리
# =============================================================================

class TimeoutState(TypedDict):
    """타임아웃 State"""
    task: str
    timeout_seconds: int
    response: Optional[str]
    result: str


def run_timeout_handling_example():
    """타임아웃 처리 예제 (시뮬레이션)"""
    print("\n" + "=" * 60)
    print("예제 5: 타임아웃 및 에러 처리 패턴")
    print("=" * 60)

    print("""
타임아웃 처리 패턴:

1. 클라이언트 측 타임아웃
   - 일정 시간 후 자동 재개
   - 기본값으로 진행

2. 백그라운드 모니터링
   - 별도 프로세스에서 체크포인트 모니터링
   - 시간 초과 시 자동 처리

3. 폴링 패턴
   while True:
       state = app.get_state(config)
       if not state.next:  # 완료됨
           break
       if time_elapsed > timeout:
           # 기본 응답으로 재개
           app.invoke(Command(resume=default_response), config)
       time.sleep(poll_interval)

4. 에러 처리
   try:
       result = app.invoke(input, config)
   except InterruptedError:
       # 인터럽트 상태로 종료
       handle_pending_approval(config)
   except Exception as e:
       # 기타 에러
       handle_error(e)
""")


# =============================================================================
# 7. HITL 패턴 정리
# =============================================================================

def explain_hitl_patterns():
    """HITL 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Human-in-the-Loop 패턴 정리")
    print("=" * 60)

    print("""
HITL 구현 패턴:

1. 동기적 승인
   def node(state):
       result = interrupt({"message": "승인 필요"})
       # 승인 후 계속

2. 비동기 승인 (웹 서비스)
   # 첫 번째 요청
   POST /invoke
   -> 202 Accepted (interrupt 상태)

   # 승인 후 재개
   POST /invoke
   body: Command(resume=approval_data)

3. 배치 처리
   - 여러 인터럽트를 모아서 처리
   - 관리자 대시보드에서 일괄 승인

사용 시 고려사항:

1. 사용자 경험
   - 명확한 승인 요청 메시지
   - 필요한 컨텍스트 제공
   - 진행 상황 표시

2. 보안
   - 승인 권한 검증
   - 승인 기록 저장
   - 감사 로그

3. 운영
   - 타임아웃 정책
   - 에스컬레이션 경로
   - 알림 시스템

4. 확장성
   - 다중 승인자 지원
   - 승인 위임
   - 자동 승인 규칙
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 16] Human-in-the-Loop (인터럽트)")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_human_in_the_loop()

    # 예제 실행
    run_interrupt_example()
    run_interrupt_before_example()
    run_multi_interrupt_example()
    run_conditional_interrupt_example()
    run_timeout_handling_example()

    # 패턴 정리
    explain_hitl_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 17_time_travel.py (타임 트래블)")
    print("=" * 60)


if __name__ == "__main__":
    main()
