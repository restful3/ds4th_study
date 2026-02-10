"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent Systems
================================================================================

파일명: 07_custom_workflow.py
난이도: ⭐⭐⭐⭐⭐ (최고급)
예상 시간: 35분

📚 학습 목표:
  - LangGraph 기본 개념 이해
  - StateGraph 생성 및 사용
  - 노드 및 엣지 정의
  - 조건부 라우팅 구현
  - 실전: 복잡한 워크플로우 구현

📖 공식 문서:
  • Custom Workflow: /official/27-custom-workflow.md

📄 교안 문서:
  • Part 7 Workflow: /docs/part07_multi_agent.md (Section 6)

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🚀 실행 방법:
  python 07_custom_workflow.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from datetime import datetime

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# 예제 1: LangGraph 기본 개념
# ============================================================================

def example_1_langgraph_basics():
    """LangGraph의 기본 개념과 구성 요소"""
    print("=" * 70)
    print("📌 예제 1: LangGraph 기본 개념")
    print("=" * 70)

    print("""
💡 LangGraph란?
   - 복잡한 멀티에이전트 워크플로우 구축 도구
   - 상태 기반 그래프 실행 엔진
   - 조건부 라우팅과 사이클 지원

🔧 핵심 구성 요소:

   1. State (상태):
      - 워크플로우 전체에서 공유되는 데이터
      - TypedDict로 정의
      - 각 노드가 읽고 쓸 수 있음

   2. Nodes (노드):
      - 각 단계의 처리 로직
      - Agent 또는 일반 함수
      - State를 입력받고 State를 반환

   3. Edges (엣지):
      - 노드 간 연결
      - 고정 엣지: 항상 다음 노드로
      - 조건부 엣지: 상태에 따라 결정

   4. Graph (그래프):
      - 노드와 엣지의 조합
      - 시작점과 종료점 정의
      - compile()로 실행 가능한 앱 생성
    """)

    print("\n📊 워크플로우 예시:")
    print("-" * 70)
    print("""
    시작
      ↓
   [분류 노드] → 상태 업데이트
      ↓
   {조건부 라우팅}
      ↓
   [처리 노드] → 결과 생성
      ↓
    종료
    """)

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 2: StateGraph 생성
# ============================================================================

def example_2_state_graph():
    """StateGraph를 사용한 간단한 워크플로우"""
    print("\n" + "=" * 70)
    print("📌 예제 2: StateGraph 생성")
    print("=" * 70)

    print("""
💡 StateGraph 생성 단계:
   1. State 타입 정의 (TypedDict)
   2. StateGraph 인스턴스 생성
   3. 노드 추가
   4. 엣지 연결
   5. 컴파일
    """)

    # 간단한 State 정의 (LangGraph 없이 시뮬레이션)
    class SimpleState(TypedDict):
        """워크플로우 상태"""
        input: str
        step: int
        result: str

    # 노드 함수들
    def node1(state: SimpleState) -> SimpleState:
        """첫 번째 노드"""
        print(f"\n[노드1] 입력: {state['input']}")
        state["step"] = 1
        state["result"] = f"노드1 처리: {state['input']}"
        return state

    def node2(state: SimpleState) -> SimpleState:
        """두 번째 노드"""
        print(f"[노드2] 이전 결과: {state['result']}")
        state["step"] = 2
        state["result"] += " → 노드2 처리"
        return state

    def node3(state: SimpleState) -> SimpleState:
        """세 번째 노드"""
        print(f"[노드3] 최종 처리")
        state["step"] = 3
        state["result"] += " → 완료"
        return state

    # 워크플로우 시뮬레이션 (실제 LangGraph 대신)
    print("\n🔄 워크플로우 실행:")
    print("=" * 70)

    user_input = input("입력: ").strip() or "테스트 입력"

    # 초기 상태
    state: SimpleState = {
        "input": user_input,
        "step": 0,
        "result": ""
    }

    # 순차 실행
    print("\n실행 시작...")
    state = node1(state)
    state = node2(state)
    state = node3(state)

    print("\n최종 결과:")
    print(f"  단계: {state['step']}")
    print(f"  결과: {state['result']}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 3: 노드 및 엣지 정의
# ============================================================================

def example_3_nodes_and_edges():
    """노드와 엣지의 다양한 사용법"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 노드 및 엣지 정의")
    print("=" * 70)

    print("""
💡 노드 타입:
   - 처리 노드: 데이터 변환
   - 분기 노드: 경로 결정
   - 통합 노드: 결과 병합

💡 엣지 타입:
   - 고정 엣지: add_edge(A, B)
   - 조건부 엣지: add_conditional_edges(A, router, mapping)
   - 종료 엣지: add_edge(A, END)
    """)

    class WorkflowState(TypedDict):
        """워크플로우 상태"""
        input: str
        category: str
        result: str

    # 분류 노드
    def classifier_node(state: WorkflowState) -> WorkflowState:
        """입력 분류"""
        input_text = state["input"]
        print(f"\n[분류] 입력 분석: {input_text}")

        if "숫자" in input_text or any(c.isdigit() for c in input_text):
            state["category"] = "math"
        elif "날씨" in input_text:
            state["category"] = "weather"
        else:
            state["category"] = "general"

        print(f"[분류] 카테고리: {state['category']}")
        return state

    # 처리 노드들
    def math_node(state: WorkflowState) -> WorkflowState:
        """수학 처리"""
        print(f"[수학 노드] 처리 중...")
        state["result"] = f"수학 처리: {state['input']}"
        return state

    def weather_node(state: WorkflowState) -> WorkflowState:
        """날씨 처리"""
        print(f"[날씨 노드] 처리 중...")
        state["result"] = f"날씨 정보: {state['input']}"
        return state

    def general_node(state: WorkflowState) -> WorkflowState:
        """일반 처리"""
        print(f"[일반 노드] 처리 중...")
        state["result"] = f"일반 응답: {state['input']}"
        return state

    # 라우팅 함수
    def route_by_category(state: WorkflowState) -> str:
        """카테고리별 라우팅"""
        return state["category"]

    # 워크플로우 실행
    print("\n🔄 조건부 라우팅 워크플로우:")
    print("=" * 70)

    test_inputs = [
        "숫자 123 더하기 456",
        "서울 날씨",
        "안녕하세요"
    ]

    for user_input in test_inputs:
        print(f"\n{'='*70}")
        print(f"입력: {user_input}")
        print(f"{'='*70}")

        # 초기 상태
        state: WorkflowState = {
            "input": user_input,
            "category": "",
            "result": ""
        }

        # 실행
        state = classifier_node(state)

        # 조건부 라우팅
        category = route_by_category(state)
        if category == "math":
            state = math_node(state)
        elif category == "weather":
            state = weather_node(state)
        else:
            state = general_node(state)

        print(f"\n최종 결과: {state['result']}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 4: 조건부 라우팅
# ============================================================================

def example_4_conditional_routing():
    """복잡한 조건부 라우팅 구현"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 조건부 라우팅")
    print("=" * 70)

    print("""
💡 조건부 라우팅:
   - 상태 값에 따라 다음 노드 결정
   - 여러 분기 경로 지원
   - 동적 워크플로우 구성
    """)

    class ComplexState(TypedDict):
        """복잡한 상태"""
        input: str
        priority: str
        attempts: int
        resolved: bool
        result: str

    # 우선순위 결정 노드
    def priority_node(state: ComplexState) -> ComplexState:
        """우선순위 결정"""
        input_text = state["input"]
        print(f"\n[우선순위 판단] 입력: {input_text}")

        if "긴급" in input_text or "중요" in input_text:
            state["priority"] = "high"
        elif "일반" in input_text:
            state["priority"] = "medium"
        else:
            state["priority"] = "low"

        print(f"[우선순위] {state['priority']}")
        return state

    # 처리 노드들
    def high_priority_node(state: ComplexState) -> ComplexState:
        """고우선순위 처리"""
        print(f"[긴급 처리] 즉시 처리 중...")
        state["attempts"] += 1
        state["resolved"] = True
        state["result"] = "긴급 처리 완료"
        return state

    def medium_priority_node(state: ComplexState) -> ComplexState:
        """중우선순위 처리"""
        print(f"[일반 처리] 처리 중...")
        state["attempts"] += 1

        if state["attempts"] < 2:
            state["resolved"] = False
            state["result"] = "재시도 필요"
        else:
            state["resolved"] = True
            state["result"] = "일반 처리 완료"

        return state

    def low_priority_node(state: ComplexState) -> ComplexState:
        """저우선순위 처리"""
        print(f"[대기열] 대기 중...")
        state["attempts"] += 1
        state["resolved"] = True
        state["result"] = "대기 후 처리"
        return state

    # 라우팅 함수들
    def route_by_priority(state: ComplexState) -> Literal["high", "medium", "low"]:
        """우선순위별 라우팅"""
        return state["priority"]

    def route_by_resolution(state: ComplexState) -> Literal["retry", "done"]:
        """해결 여부로 라우팅"""
        if state["resolved"]:
            return "done"
        else:
            return "retry"

    # 워크플로우 실행
    print("\n🔄 우선순위 기반 워크플로우:")
    print("=" * 70)

    user_input = input("\n작업 입력 (예: 긴급 작업): ").strip() or "긴급 작업"

    state: ComplexState = {
        "input": user_input,
        "priority": "",
        "attempts": 0,
        "resolved": False,
        "result": ""
    }

    # 우선순위 결정
    state = priority_node(state)

    # 우선순위별 처리
    priority = route_by_priority(state)
    if priority == "high":
        state = high_priority_node(state)
    elif priority == "medium":
        state = medium_priority_node(state)

        # 재시도 로직
        while not state["resolved"] and state["attempts"] < 3:
            print("\n[재시도] 다시 처리 중...")
            state = medium_priority_node(state)

    else:
        state = low_priority_node(state)

    print(f"\n최종 상태:")
    print(f"  시도 횟수: {state['attempts']}")
    print(f"  해결 여부: {'✅' if state['resolved'] else '❌'}")
    print(f"  결과: {state['result']}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 5: 실전 - 복잡한 워크플로우
# ============================================================================

def example_5_complex_workflow():
    """실전: 주문 처리 워크플로우"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 주문 처리 워크플로우")
    print("=" * 70)

    print("""
🎯 실전 시나리오: E-commerce 주문 처리

워크플로우:
   1. 주문 접수
   2. 재고 확인
   3. 결제 처리
   4. 배송 준비
   5. 완료 또는 실패 처리
    """)

    class OrderState(TypedDict):
        """주문 상태"""
        order_id: str
        item: str
        quantity: int
        stock_available: bool
        payment_success: bool
        shipping_ready: bool
        status: str
        timestamp: str

    # 노드 함수들
    def receive_order_node(state: OrderState) -> OrderState:
        """주문 접수"""
        print(f"\n[1. 주문 접수]")
        print(f"  주문번호: {state['order_id']}")
        print(f"  상품: {state['item']}")
        print(f"  수량: {state['quantity']}")

        state["timestamp"] = datetime.now().isoformat()
        state["status"] = "received"
        return state

    def check_stock_node(state: OrderState) -> OrderState:
        """재고 확인"""
        print(f"\n[2. 재고 확인]")

        # 시뮬레이션: 수량 10개 이하는 재고 있음
        state["stock_available"] = state["quantity"] <= 10

        if state["stock_available"]:
            print(f"  ✅ 재고 충분 (요청: {state['quantity']}개)")
            state["status"] = "stock_confirmed"
        else:
            print(f"  ❌ 재고 부족 (요청: {state['quantity']}개)")
            state["status"] = "out_of_stock"

        return state

    def process_payment_node(state: OrderState) -> OrderState:
        """결제 처리"""
        print(f"\n[3. 결제 처리]")

        # 시뮬레이션: 항상 성공
        state["payment_success"] = True
        print(f"  ✅ 결제 완료")
        state["status"] = "paid"

        return state

    def prepare_shipping_node(state: OrderState) -> OrderState:
        """배송 준비"""
        print(f"\n[4. 배송 준비]")

        state["shipping_ready"] = True
        print(f"  ✅ 배송 준비 완료")
        state["status"] = "shipping"

        return state

    def complete_order_node(state: OrderState) -> OrderState:
        """주문 완료"""
        print(f"\n[5. 주문 완료]")
        state["status"] = "completed"
        print(f"  ✅ 주문 처리 완료!")
        return state

    def cancel_order_node(state: OrderState) -> OrderState:
        """주문 취소"""
        print(f"\n[5. 주문 취소]")
        state["status"] = "cancelled"
        print(f"  ❌ 주문 취소됨 (재고 부족)")
        return state

    # 라우팅 함수들
    def route_after_stock_check(state: OrderState) -> Literal["continue", "cancel"]:
        """재고 확인 후 라우팅"""
        if state["stock_available"]:
            return "continue"
        else:
            return "cancel"

    def route_after_payment(state: OrderState) -> Literal["continue", "cancel"]:
        """결제 후 라우팅"""
        if state["payment_success"]:
            return "continue"
        else:
            return "cancel"

    # 워크플로우 실행
    print("\n🛒 주문 처리 시작:")
    print("=" * 70)

    item = input("상품명: ").strip() or "노트북"
    quantity = input("수량: ").strip()
    quantity = int(quantity) if quantity.isdigit() else 5

    # 초기 상태
    state: OrderState = {
        "order_id": f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "item": item,
        "quantity": quantity,
        "stock_available": False,
        "payment_success": False,
        "shipping_ready": False,
        "status": "pending",
        "timestamp": ""
    }

    # 워크플로우 실행
    print(f"\n{'='*70}")

    # 1. 주문 접수
    state = receive_order_node(state)

    # 2. 재고 확인
    state = check_stock_node(state)

    # 조건부 라우팅
    if route_after_stock_check(state) == "cancel":
        state = cancel_order_node(state)
    else:
        # 3. 결제 처리
        state = process_payment_node(state)

        # 조건부 라우팅
        if route_after_payment(state) == "cancel":
            state = cancel_order_node(state)
        else:
            # 4. 배송 준비
            state = prepare_shipping_node(state)

            # 5. 완료
            state = complete_order_node(state)

    # 최종 결과
    print(f"\n{'='*70}")
    print(f"최종 상태: {state['status']}")
    print(f"{'='*70}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("=" * 70)
    print("Part 7: Multi-Agent Systems")
    print("07. Custom Workflow (LangGraph)")
    print("=" * 70)

    while True:
        print("\n")
        print("📚 실행할 예제를 선택하세요:")
        print("-" * 70)
        print("1. LangGraph 기본 개념")
        print("2. StateGraph 생성")
        print("3. 노드 및 엣지 정의")
        print("4. 조건부 라우팅")
        print("5. 실전: 복잡한 워크플로우")
        print("0. 종료")
        print("-" * 70)

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_langgraph_basics()
        elif choice == "2":
            example_2_state_graph()
        elif choice == "3":
            example_3_nodes_and_edges()
        elif choice == "4":
            example_4_conditional_routing()
        elif choice == "5":
            example_5_complex_workflow()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")

    print("\n" + "=" * 70)
    print("📚 학습 완료!")
    print("=" * 70)
    print("""
✅ 배운 내용:
   - LangGraph의 기본 개념과 구성 요소
   - StateGraph 생성 및 사용법
   - 노드와 엣지의 다양한 활용
   - 조건부 라우팅 구현
   - 실전 복잡한 주문 처리 워크플로우

💡 핵심 요약:
   ┌─────────────────────────────────────────────────────────────────┐
   │ LangGraph는 복잡한 멀티에이전트 워크플로우를 구축하는 도구     │
   │                                                                   │
   │ 주요 구성:                                                       │
   │ • State: 워크플로우 전체 공유 데이터                            │
   │ • Nodes: 각 단계의 처리 로직                                    │
   │ • Edges: 노드 간 연결 (고정/조건부)                             │
   │ • Graph: 전체 워크플로우 정의                                   │
   │                                                                   │
   │ 사용 시점:                                                       │
   │ • 복잡한 비즈니스 프로세스                                      │
   │ • 상태 기반 전환 필요                                           │
   │ • 사이클/재시도 로직 구현                                       │
   └─────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()
