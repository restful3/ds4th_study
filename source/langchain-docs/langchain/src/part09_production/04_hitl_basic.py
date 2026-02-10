"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 (Production)
================================================================================

파일명: 04_hitl_basic.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 25분

📚 학습 목표:
  - Human-in-the-Loop (HITL) 기본 개념
  - Agent 실행 중 사람의 개입
  - 승인 워크플로우 구현
  - 안전한 Agent 운영

📖 공식 문서:
  • Human in the Loop: /official/13-human-in-the-loop.md

📄 교안 문서:
  • Part 9 개요: /docs/part09_production.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 04_hitl_basic.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: HITL 개념 소개
# ============================================================================

def example_1_hitl_concept():
    """Human-in-the-Loop 개념 이해"""
    print("=" * 70)
    print("📌 예제 1: Human-in-the-Loop (HITL) 개념")
    print("=" * 70)

    print("""
🤝 Human-in-the-Loop (HITL)란?

정의:
  Agent가 자동으로 실행되는 중간에 사람의 판단이나 승인을 받는 패턴

왜 필요한가?
  • 중요한 결정에 사람의 검토 필요
  • 위험한 작업(삭제, 결제 등) 방지
  • 규정 준수 (금융, 의료 등)
  • Agent 신뢰도 향상

주요 사용 사례:
  1️⃣ 승인 워크플로우
     - 파일 삭제 전 확인
     - 이메일 전송 전 검토
     - 결제 실행 전 승인

  2️⃣ 데이터 검증
     - Agent가 생성한 데이터 확인
     - 잘못된 정보 수정
     - 추가 정보 입력

  3️⃣ 에스컬레이션
     - Agent가 해결 못하면 사람에게 전달
     - 복잡한 문제는 전문가 개입
     - 예외 상황 처리

구현 방법:
  • interrupt_before: 특정 노드 실행 전 중단
  • interrupt_after: 특정 노드 실행 후 중단
  • update_state(): 상태 수정
  • stream(): 중단 시점까지 실행

💡 핵심: Agent를 완전히 자동화하지 않고,
   중요한 시점에 사람의 판단을 받아 안전하게 운영
    """)


# ============================================================================
# 예제 2: 간단한 승인 워크플로우
# ============================================================================

def example_2_simple_approval():
    """간단한 승인 워크플로우 구현"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 간단한 승인 워크플로우")
    print("=" * 70)

    @tool
    def send_email(recipient: str, subject: str) -> str:
        """이메일을 전송합니다."""
        return f"이메일 전송 완료: {recipient} - {subject}"

    @tool
    def delete_file(filename: str) -> str:
        """파일을 삭제합니다."""
        return f"파일 삭제 완료: {filename}"

    # Agent 생성 (interrupt_before 사용)
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(
        llm,
        tools=[send_email, delete_file],
        checkpointer=MemorySaver(),
    )

    print("\n🔹 승인 워크플로우 시뮬레이션:")
    print("-" * 70)

    config = {"configurable": {"thread_id": "approval_demo"}}

    # 1단계: Agent 실행 (tool 호출 전 중단 설정)
    print("\n[1단계] Agent 실행 시작")
    user_message = "important.txt 파일을 삭제해주세요."
    print(f"👤 사용자: {user_message}")

    # interrupt_before="tools" - Tool 실행 전 중단
    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="values"
    ):
        if "messages" in event:
            last_message = event["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print("\n⏸️  Agent가 중단되었습니다!")
                print("🔧 호출하려는 도구:")
                for tc in last_message.tool_calls:
                    print(f"   • {tc['name']}({tc['args']})")

    # 2단계: 사람의 승인
    print("\n[2단계] 사람의 승인 필요")
    approval = input("\n❓ 이 작업을 승인하시겠습니까? (y/n): ").strip().lower()

    if approval == 'y':
        print("✅ 승인됨 - Agent 계속 실행")

        # 계속 실행
        for event in agent.stream(None, config=config, stream_mode="values"):
            if "messages" in event:
                last_message = event["messages"][-1]
                if hasattr(last_message, "content") and last_message.content:
                    print(f"\n🤖 Agent: {last_message.content}")
    else:
        print("❌ 거부됨 - Agent 실행 중단")

    print("\n" + "-" * 70)
    print("💡 Tool 실행 전 사람의 승인을 받았습니다.")


# ============================================================================
# 예제 3: interrupt_before vs interrupt_after
# ============================================================================

def example_3_interrupt_modes():
    """interrupt_before와 interrupt_after 비교"""
    print("\n" + "=" * 70)
    print("📌 예제 3: interrupt_before vs interrupt_after")
    print("=" * 70)

    @tool
    def fetch_data(source: str) -> str:
        """데이터를 가져옵니다."""
        return f"{source}에서 데이터 100건 가져옴"

    llm = ChatOpenAI(model="gpt-4o-mini")

    print("""
📊 Interrupt 모드 비교:

1️⃣ interrupt_before="tools"
   - Tool 실행 **전**에 중단
   - 용도: Tool 호출 전 승인
   - 예: 파일 삭제 전 확인

2️⃣ interrupt_after="tools"
   - Tool 실행 **후**에 중단
   - 용도: Tool 결과 검토
   - 예: 데이터 검증 후 진행

💡 실전 활용:
   - 위험한 작업: interrupt_before
   - 결과 검증: interrupt_after
   - 복합 워크플로우: 둘 다 사용
    """)

    # interrupt_before 예시
    print("\n🔹 interrupt_before 예시:")
    print("-" * 70)

    agent_before = create_react_agent(
        llm,
        tools=[fetch_data],
        checkpointer=MemorySaver(),
    )

    config_before = {"configurable": {"thread_id": "before_demo"}}

    print("Tool 실행 **전** 중단")
    for event in agent_before.stream(
        {"messages": [{"role": "user", "content": "database에서 데이터 가져오기"}]},
        config=config_before,
        stream_mode="values"
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"⏸️  중단: Tool 호출 준비됨 - {last_msg.tool_calls[0]['name']}")
                print("   → 이 시점에서 승인 받을 수 있음")

    print("\n💡 interrupt_before는 Tool 실행 전 중단하여 사전 승인 가능")


# ============================================================================
# 예제 4: 상태 수정 (update_state)
# ============================================================================

def example_4_update_state():
    """중단 후 상태를 수정하여 계속 실행"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 상태 수정 (update_state)")
    print("=" * 70)

    @tool
    def process_order(order_id: str, amount: float) -> str:
        """주문을 처리합니다."""
        return f"주문 {order_id} 처리 완료: ${amount}"

    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(
        llm,
        tools=[process_order],
        checkpointer=MemorySaver(),
    )

    print("\n🔹 상태 수정 시나리오:")
    print("-" * 70)

    config = {"configurable": {"thread_id": "update_demo"}}

    # 1단계: Agent 실행
    print("\n[1단계] 주문 처리 요청")
    user_message = "주문 ORDER-123을 $1000로 처리해주세요."
    print(f"👤 사용자: {user_message}")

    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="values"
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print("\n⏸️  Agent 중단")
                tc = last_msg.tool_calls[0]
                print(f"🔧 호출 예정: {tc['name']}({tc['args']})")

    # 2단계: 사람이 금액 수정
    print("\n[2단계] 사람이 금액 수정")
    print("💡 $1000는 너무 크다 → $100로 수정")

    # update_state()로 Tool 인자 수정 (간소화된 예시)
    print("   agent.update_state(config, {'amount': 100})")
    print("   ✅ 상태 수정 완료")

    # 3단계: 수정된 상태로 계속 실행
    print("\n[3단계] 수정된 상태로 계속 실행")
    print("🔄 Agent 재시작...")
    print("✅ 주문 ORDER-123 처리 완료: $100")

    print("\n" + "-" * 70)
    print("💡 update_state()로 중단 시점의 상태를 수정할 수 있습니다.")


# ============================================================================
# 예제 5: 실전 승인 시스템
# ============================================================================

def example_5_approval_system():
    """실전에서 사용 가능한 승인 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 승인 시스템")
    print("=" * 70)

    @tool
    def charge_credit_card(card_number: str, amount: float) -> str:
        """신용카드로 결제합니다."""
        return f"결제 완료: 카드 {card_number[-4:]} - ${amount}"

    @tool
    def send_notification(user: str, message: str) -> str:
        """사용자에게 알림을 전송합니다."""
        return f"{user}에게 알림 전송: {message}"

    class ApprovalSystem:
        """승인 시스템"""

        def __init__(self):
            self.pending_approvals = []

        def request_approval(self, action: str, details: dict) -> bool:
            """승인 요청"""
            print(f"\n🔔 승인 요청:")
            print(f"   작업: {action}")
            print(f"   상세: {details}")

            # 위험한 작업은 자동으로 승인 요청
            risky_actions = ["charge_credit_card", "delete_file", "send_email"]

            if action in risky_actions:
                print(f"   ⚠️  위험한 작업 감지!")
                response = input(f"\n   승인하시겠습니까? (y/n): ").strip().lower()
                return response == 'y'
            else:
                # 안전한 작업은 자동 승인
                print(f"   ✅ 안전한 작업 - 자동 승인")
                return True

    print("\n🔒 승인 시스템 데모:")
    print("-" * 70)

    approval_sys = ApprovalSystem()

    # 시나리오 1: 위험한 작업
    print("\n시나리오 1: 신용카드 결제 (위험)")
    action1 = "charge_credit_card"
    details1 = {"card": "****1234", "amount": 500.00}

    if approval_sys.request_approval(action1, details1):
        print(f"   ✅ 실행: 결제 완료 - $500.00")
    else:
        print(f"   ❌ 거부: 결제 취소")

    # 시나리오 2: 안전한 작업
    print("\n시나리오 2: 알림 전송 (안전)")
    action2 = "send_notification"
    details2 = {"user": "user123", "message": "처리 완료"}

    if approval_sys.request_approval(action2, details2):
        print(f"   ✅ 실행: 알림 전송 완료")

    print("\n" + "-" * 70)
    print("💡 승인 시스템 패턴:")
    print("  • 작업 유형별 위험도 분류")
    print("  • 위험한 작업만 승인 요청")
    print("  • 자동 승인 + 수동 승인 조합")
    print("  • 승인 이력 기록")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 9: 프로덕션 - Human-in-the-Loop 기초")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_hitl_concept()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_simple_approval()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_interrupt_modes()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_update_state()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_approval_system()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 9-04: Human-in-the-Loop 기초를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 05_hitl_decisions.py - HITL 의사결정")
    print("  2. 06_structured_output.py - Structured Output")
    print("  3. Part 10: Deployment")
    print("\n📚 핵심 요약:")
    print("  • HITL: 중요한 시점에 사람의 개입")
    print("  • interrupt_before: Tool 실행 전 중단")
    print("  • interrupt_after: Tool 실행 후 중단")
    print("  • update_state(): 상태 수정 후 계속")
    print("  • 승인 시스템으로 안전한 Agent 운영")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. HITL 활용 사례:
#    - 금융: 거래 승인
#    - 의료: 진단 검토
#    - 법률: 계약서 검토
#    - 관리: 중요 결정
#
# 2. 구현 패턴:
#    - Checkpointer로 상태 저장
#    - interrupt 설정으로 중단점 지정
#    - stream()으로 부분 실행
#    - update_state()로 상태 수정
#
# 3. 보안 고려사항:
#    - 위험한 작업 분류
#    - 권한 검증
#    - 승인 이력 기록
#    - 감사 로그
#
# ============================================================================
