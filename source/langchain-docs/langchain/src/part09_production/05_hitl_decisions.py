"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 (Production)
================================================================================

파일명: 05_hitl_decisions.py
난이도: ⭐⭐⭐⭐⭐ (전문가)
예상 시간: 25분

📚 학습 목표:
  - HITL을 활용한 복잡한 의사결정
  - 다중 승인 워크플로우
  - 조건부 HITL 패턴
  - 실전 시나리오 구현

📖 공식 문서:
  • Human in the Loop: /official/13-human-in-the-loop.md

📄 교안 문서:
  • Part 9 개요: /docs/part09_production.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 05_hitl_decisions.py

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
from typing import Dict, List
import time

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: 조건부 승인 시스템
# ============================================================================

def example_1_conditional_approval():
    """조건에 따라 승인 여부를 결정"""
    print("=" * 70)
    print("📌 예제 1: 조건부 승인 시스템")
    print("=" * 70)

    @tool
    def transfer_money(from_account: str, to_account: str, amount: float) -> str:
        """계좌 간 송금을 실행합니다."""
        return f"송금 완료: {from_account} → {to_account}, ${amount}"

    class ConditionalApprovalSystem:
        """조건부 승인 시스템"""

        def __init__(self, auto_approve_threshold: float = 100.0):
            self.auto_approve_threshold = auto_approve_threshold
            self.approval_history = []

        def requires_approval(self, action: str, params: Dict) -> bool:
            """승인이 필요한지 판단"""

            # 1. 작업 유형 확인
            risky_actions = ["transfer_money", "delete_database", "send_bulk_email"]
            if action not in risky_actions:
                return False

            # 2. 금액 확인 (송금의 경우)
            if action == "transfer_money":
                amount = params.get("amount", 0)
                if amount > self.auto_approve_threshold:
                    return True  # 큰 금액은 승인 필요
                else:
                    return False  # 소액은 자동 승인

            # 3. 기타 위험한 작업은 항상 승인 필요
            return True

        def request_approval(self, action: str, params: Dict) -> bool:
            """승인 요청"""
            print(f"\n🔔 승인 요청:")
            print(f"   작업: {action}")
            print(f"   파라미터: {params}")

            response = input(f"\n   ❓ 승인하시겠습니까? (y/n): ").strip().lower()
            approved = response == 'y'

            # 이력 기록
            self.approval_history.append({
                "action": action,
                "params": params,
                "approved": approved,
                "timestamp": time.time()
            })

            return approved

    print("\n🎯 조건부 승인 시스템 테스트:")
    print("-" * 70)

    approval_sys = ConditionalApprovalSystem(auto_approve_threshold=100.0)

    # 테스트 케이스 1: 소액 송금 (자동 승인)
    print("\n[테스트 1] 소액 송금: $50")
    action1 = "transfer_money"
    params1 = {"from_account": "A", "to_account": "B", "amount": 50.0}

    if approval_sys.requires_approval(action1, params1):
        if approval_sys.request_approval(action1, params1):
            print("   ✅ 승인됨 - 송금 실행")
        else:
            print("   ❌ 거부됨 - 송금 취소")
    else:
        print("   ✅ 자동 승인 - 송금 실행 (소액)")

    # 테스트 케이스 2: 고액 송금 (승인 필요)
    print("\n[테스트 2] 고액 송금: $5000")
    action2 = "transfer_money"
    params2 = {"from_account": "A", "to_account": "B", "amount": 5000.0}

    if approval_sys.requires_approval(action2, params2):
        if approval_sys.request_approval(action2, params2):
            print("   ✅ 승인됨 - 송금 실행")
        else:
            print("   ❌ 거부됨 - 송금 취소")

    print("\n" + "-" * 70)
    print("💡 조건부 승인으로 효율성과 안전성을 동시에 확보")


# ============================================================================
# 예제 2: 다단계 승인 워크플로우
# ============================================================================

def example_2_multi_stage_approval():
    """여러 단계의 승인이 필요한 워크플로우"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 다단계 승인 워크플로우")
    print("=" * 70)

    class MultiStageApproval:
        """다단계 승인 시스템"""

        def __init__(self):
            self.stages = []

        def add_stage(self, stage_name: str, approver: str, condition=None):
            """승인 단계 추가"""
            self.stages.append({
                "name": stage_name,
                "approver": approver,
                "condition": condition,
                "approved": False
            })

        def execute_approval(self, action: str, params: Dict) -> bool:
            """승인 프로세스 실행"""
            print(f"\n📋 다단계 승인 프로세스 시작:")
            print(f"   작업: {action}")
            print(f"   상세: {params}\n")

            for i, stage in enumerate(self.stages, 1):
                # 조건 확인 (조건이 있는 경우)
                if stage["condition"] and not stage["condition"](params):
                    print(f"[단계 {i}] {stage['name']}: 건너뜀 (조건 불충족)")
                    stage["approved"] = True
                    continue

                # 승인 요청
                print(f"[단계 {i}] {stage['name']}")
                print(f"   승인자: {stage['approver']}")
                response = input(f"   ❓ {stage['approver']}님, 승인하시겠습니까? (y/n): ").strip().lower()

                if response == 'y':
                    stage["approved"] = True
                    print(f"   ✅ 승인됨")
                else:
                    stage["approved"] = False
                    print(f"   ❌ 거부됨 - 프로세스 중단")
                    return False

            print(f"\n🎉 모든 단계 승인 완료!")
            return True

    print("\n🔄 예시: 고액 지출 승인 프로세스")
    print("-" * 70)

    # 승인 시스템 설정
    approval = MultiStageApproval()

    # 단계 1: 팀장 승인
    approval.add_stage(
        "팀장 승인",
        "김팀장",
        condition=lambda params: params.get("amount", 0) > 1000
    )

    # 단계 2: 부서장 승인
    approval.add_stage(
        "부서장 승인",
        "박부장",
        condition=lambda params: params.get("amount", 0) > 5000
    )

    # 단계 3: CFO 승인
    approval.add_stage(
        "CFO 승인",
        "최CFO",
        condition=lambda params: params.get("amount", 0) > 10000
    )

    # 테스트: $15,000 지출
    action = "approve_expense"
    params = {"amount": 15000, "category": "마케팅", "purpose": "광고 캠페인"}

    if approval.execute_approval(action, params):
        print(f"\n✅ 지출 승인 완료: ${params['amount']}")
    else:
        print(f"\n❌ 지출 거부됨")

    print("\n" + "-" * 70)
    print("💡 금액에 따라 필요한 승인 단계가 자동으로 결정됩니다.")


# ============================================================================
# 예제 3: 사용자 피드백 루프
# ============================================================================

def example_3_feedback_loop():
    """Agent가 사용자 피드백을 받아 개선"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 사용자 피드백 루프")
    print("=" * 70)

    @tool
    def generate_report(title: str, sections: List[str]) -> str:
        """보고서를 생성합니다."""
        return f"보고서: {title}\n섹션: {', '.join(sections)}"

    class FeedbackLoop:
        """피드백 루프 시스템"""

        def __init__(self):
            self.iterations = []

        def get_feedback(self, output: str) -> Dict:
            """사용자 피드백 받기"""
            print(f"\n📄 생성된 결과:")
            print("-" * 70)
            print(output)
            print("-" * 70)

            satisfied = input("\n❓ 이 결과에 만족하시나요? (y/n): ").strip().lower()

            if satisfied == 'y':
                return {"satisfied": True, "feedback": None}
            else:
                feedback = input("📝 개선 사항을 입력하세요: ").strip()
                return {"satisfied": False, "feedback": feedback}

        def run_with_feedback(self, initial_input: str, max_iterations: int = 3):
            """피드백을 받아가며 반복 실행"""
            print(f"\n🔄 피드백 루프 시작 (최대 {max_iterations}회)")
            print("-" * 70)

            current_input = initial_input

            for iteration in range(1, max_iterations + 1):
                print(f"\n[반복 {iteration}]")

                # Agent 실행 (시뮬레이션)
                if iteration == 1:
                    output = f"보고서 초안: {current_input}"
                else:
                    output = f"수정된 보고서 (v{iteration}): {current_input} + 피드백 반영"

                # 피드백 받기
                result = self.get_feedback(output)
                self.iterations.append({
                    "iteration": iteration,
                    "output": output,
                    "feedback": result
                })

                if result["satisfied"]:
                    print(f"\n✅ {iteration}번 반복 후 완료!")
                    return output
                else:
                    print(f"\n🔄 피드백 반영 중: {result['feedback']}")
                    current_input += f" (피드백: {result['feedback']})"

            print(f"\n⚠️  최대 반복 횟수 도달")
            return output

    print("\n💬 피드백 루프 데모:")
    print("-" * 70)

    feedback_sys = FeedbackLoop()
    initial_request = "분기별 매출 보고서 작성"

    final_output = feedback_sys.run_with_feedback(initial_request, max_iterations=3)

    print(f"\n📊 피드백 이력:")
    for item in feedback_sys.iterations:
        print(f"  반복 {item['iteration']}: 만족도 {item['feedback']['satisfied']}")

    print(f"\n✅ 최종 결과물: {final_output[:80]}...")

    print("\n" + "-" * 70)
    print("💡 사용자 피드백을 받아 결과를 점진적으로 개선")


# ============================================================================
# 예제 4: 예외 처리 및 에스컬레이션
# ============================================================================

def example_4_escalation():
    """Agent가 처리 못하면 사람에게 에스컬레이션"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 예외 처리 및 에스컬레이션")
    print("=" * 70)

    @tool
    def resolve_issue(issue_type: str, severity: str) -> str:
        """이슈를 해결합니다."""
        if severity == "critical":
            raise Exception("심각한 이슈는 자동 처리 불가")
        return f"{issue_type} 이슈 해결 완료"

    class EscalationSystem:
        """에스컬레이션 시스템"""

        def __init__(self):
            self.escalated_cases = []

        def handle_issue(self, issue: Dict) -> bool:
            """이슈 처리 시도"""
            print(f"\n🔧 이슈 처리 시도:")
            print(f"   유형: {issue['type']}")
            print(f"   심각도: {issue['severity']}")

            # Agent 자동 처리 시도
            if issue['severity'] in ['low', 'medium']:
                print(f"   ✅ Agent가 자동으로 처리함")
                return True
            else:
                # 심각한 이슈는 에스컬레이션
                print(f"   ⚠️  심각한 이슈 감지 - 에스컬레이션 필요")
                return self.escalate(issue)

        def escalate(self, issue: Dict) -> bool:
            """이슈를 사람에게 에스컬레이션"""
            print(f"\n🚨 에스컬레이션:")
            print(f"   이슈: {issue['type']}")
            print(f"   설명: {issue['description']}")
            print(f"   심각도: {issue['severity']}")

            self.escalated_cases.append(issue)

            action = input(f"\n   ❓ 어떻게 처리하시겠습니까? (1:해결, 2:보류, 3:거부): ").strip()

            if action == '1':
                solution = input(f"   📝 해결 방법을 입력하세요: ").strip()
                issue['resolution'] = solution
                issue['status'] = 'resolved'
                print(f"   ✅ 해결: {solution}")
                return True
            elif action == '2':
                issue['status'] = 'pending'
                print(f"   ⏸️  보류됨")
                return False
            else:
                issue['status'] = 'rejected'
                print(f"   ❌ 거부됨")
                return False

    print("\n📋 에스컬레이션 시나리오:")
    print("-" * 70)

    escalation_sys = EscalationSystem()

    # 테스트 케이스
    issues = [
        {"type": "버그", "severity": "low", "description": "UI 버튼 색상 오류"},
        {"type": "장애", "severity": "critical", "description": "결제 시스템 다운"},
        {"type": "요청", "severity": "medium", "description": "기능 추가 요청"},
    ]

    for i, issue in enumerate(issues, 1):
        print(f"\n[이슈 {i}]")
        escalation_sys.handle_issue(issue)

    print("\n" + "-" * 70)
    print(f"📊 에스컬레이션된 케이스: {len(escalation_sys.escalated_cases)}건")


# ============================================================================
# 예제 5: 실전 HITL 패턴
# ============================================================================

def example_5_production_hitl():
    """프로덕션에서 사용 가능한 HITL 패턴"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 HITL 패턴 모음")
    print("=" * 70)

    print("""
🎯 실전 HITL 패턴 모음:

1️⃣ 승인 후 실행 패턴
   - 위험한 작업 전 승인
   - 예: 파일 삭제, 결제, 이메일 전송
   ```python
   if requires_approval(action):
       if not get_approval(action):
           return "작업 취소됨"
   execute(action)
   ```

2️⃣ 검증 후 계속 패턴
   - Agent 출력 검증 후 진행
   - 예: 데이터 생성, 보고서 작성
   ```python
   result = agent.run()
   if not validate(result):
       result = human_correct(result)
   return result
   ```

3️⃣ 에스컬레이션 패턴
   - 처리 못하면 사람에게 전달
   - 예: 복잡한 문제, 예외 상황
   ```python
   try:
       return agent.run()
   except ComplexIssue:
       return escalate_to_human()
   ```

4️⃣ 피드백 루프 패턴
   - 반복적으로 개선
   - 예: 콘텐츠 생성, 디자인
   ```python
   for i in range(max_iterations):
       output = agent.run()
       if user_satisfied(output):
           return output
       feedback = get_feedback()
       agent.update_with_feedback(feedback)
   ```

5️⃣ 조건부 개입 패턴
   - 조건에 따라 개입
   - 예: 임계값 초과 시
   ```python
   result = agent.run()
   if exceeds_threshold(result):
       return human_review(result)
   return result
   ```

💡 선택 가이드:
   - 위험도 높음 → 승인 후 실행
   - 정확도 중요 → 검증 후 계속
   - 복잡도 높음 → 에스컬레이션
   - 품질 중요 → 피드백 루프
   - 효율성 중요 → 조건부 개입
    """)

    print("\n📚 구현 체크리스트:")
    print("-" * 70)
    print("""
✅ HITL 구현 시 고려사항:

□ 중단 시점 설계
  - interrupt_before vs interrupt_after
  - 어느 노드에서 중단할지 결정

□ 상태 관리
  - Checkpointer로 상태 저장
  - 중단 후 재개 가능하도록 설계

□ 사용자 인터페이스
  - 명확한 승인 요청 메시지
  - 필요한 정보 모두 표시
  - 간단한 입력 방법 (y/n)

□ 타임아웃 처리
  - 승인 대기 시간 제한
  - 타임아웃 시 기본 동작 정의

□ 이력 기록
  - 승인/거부 이력 저장
  - 감사 추적 가능하도록

□ 권한 관리
  - 누가 승인할 수 있는지
  - 역할 기반 권한 체계

□ 알림 시스템
  - 승인 요청 알림
  - 이메일/슬랙 등 통합
    """)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 9: 프로덕션 - HITL 의사결정")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_conditional_approval()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_multi_stage_approval()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_feedback_loop()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_escalation()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_production_hitl()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 9-05: HITL 의사결정을 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 06_structured_output.py - Structured Output")
    print("  2. Part 10: Deployment")
    print("  3. 실전 프로젝트 시작")
    print("\n📚 핵심 요약:")
    print("  • 조건부 승인으로 효율성 향상")
    print("  • 다단계 승인으로 리스크 관리")
    print("  • 피드백 루프로 품질 개선")
    print("  • 에스컬레이션으로 복잡한 문제 처리")
    print("  • 상황에 맞는 HITL 패턴 선택")
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
# 1. 고급 승인 시스템:
#    - 역할 기반 접근 제어 (RBAC)
#    - 동적 승인 규칙
#    - 승인 체인
#    - 병렬 승인
#
# 2. 프로덕션 고려사항:
#    - 비동기 승인 (이메일, 슬랙)
#    - 타임아웃 및 기본값
#    - 승인 이력 데이터베이스
#    - 감사 로그
#
# 3. UX 최적화:
#    - 웹 UI 통합
#    - 모바일 알림
#    - 원클릭 승인
#    - 컨텍스트 제공
#
# ============================================================================
