"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent Systems
================================================================================

파일명: 04_handoffs.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - Handoff 패턴의 개념 이해
  - Agent 간 제어 전달 구현
  - Context/State 전달 방법
  - 조건부 Handoff 로직
  - 실전: 고객 서비스 에스컬레이션 시스템

📖 공식 문서:
  • Handoffs: /official/24-handoffs.md

📄 교안 문서:
  • Part 7 Handoffs: /docs/part07_multi_agent.md (Section 3)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 04_handoffs.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict
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
# 예제 1: Handoff의 기본 개념
# ============================================================================

def example_1_handoff_concept():
    """Handoff 패턴의 기본 개념과 Subagent와의 차이"""
    print("=" * 70)
    print("📌 예제 1: Handoff의 기본 개념")
    print("=" * 70)

    print("""
💡 Handoff란?
   - 한 Agent에서 다른 Agent로 제어를 완전히 전달
   - 이전 Agent는 종료되고 새 Agent가 제어권을 가짐
   - 티켓팅 시스템이나 에스컬레이션에 적합

🔄 Subagent vs Handoff:

   Subagent:
   메인 Agent → Subagent 호출 → 결과 반환 → 메인 Agent 계속
   (메인 Agent가 제어 유지)

   Handoff:
   Agent A → Handoff → Agent B
   (Agent A 종료, Agent B가 제어)
    """)

    # 간단한 Handoff 시뮬레이션
    def agent_tier1(user_input: str) -> dict:
        """Tier 1 Agent (기본 지원)"""
        print(f"\n[Tier 1 Agent 실행]")
        print(f"입력: {user_input}")

        # 간단한 문제는 직접 처리
        if "비밀번호" in user_input:
            return {
                "resolved": True,
                "response": "비밀번호 재설정 링크를 이메일로 보내드렸습니다.",
                "handoff": False
            }

        # 복잡한 문제는 Tier 2로 Handoff
        return {
            "resolved": False,
            "response": "더 전문적인 지원이 필요합니다.",
            "handoff": True,
            "context": f"Tier 1에서 처리 불가: {user_input}"
        }

    def agent_tier2(context: str, user_input: str) -> dict:
        """Tier 2 Agent (기술 지원)"""
        print(f"\n[Tier 2 Agent 실행]")
        print(f"전달된 컨텍스트: {context}")
        print(f"입력: {user_input}")

        return {
            "resolved": True,
            "response": "기술 지원팀에서 문제를 해결했습니다.",
            "handoff": False
        }

    # 테스트
    print("\n🧪 테스트 케이스:")
    print("-" * 70)

    test_cases = [
        "비밀번호를 잊어버렸어요",
        "시스템이 계속 오류를 발생시킵니다"
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n케이스 {i}: {test_input}")
        print("-" * 70)

        # Tier 1 실행
        result = agent_tier1(test_input)
        print(f"응답: {result['response']}")

        # Handoff 필요 시 Tier 2 실행
        if result.get("handoff"):
            print("\n🔄 Tier 2로 Handoff...")
            result = agent_tier2(result["context"], test_input)
            print(f"응답: {result['response']}")

        print(f"해결: {'✅' if result['resolved'] else '❌'}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 2: Agent 간 제어 전달
# ============================================================================

def example_2_control_transfer():
    """Agent 간 명시적인 제어 전달"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Agent 간 제어 전달")
    print("=" * 70)

    print("""
💡 제어 전달 패턴:
   - Agent가 handoff 도구를 명시적으로 호출
   - 제어권이 완전히 다른 Agent로 이동
   - 이전 Agent는 더 이상 실행되지 않음
    """)

    # Handoff 도구 생성
    @tool
    def handoff_to_technical(issue: str) -> str:
        """기술 지원 팀으로 이관합니다.

        Args:
            issue: 기술 문제 설명

        Returns:
            기술 팀의 응답
        """
        prompt = f"""
당신은 기술 지원 전문가입니다.
다음 문제를 해결하세요:

{issue}

구체적인 해결 방법을 제시하세요.
"""
        response = llm.invoke(prompt)
        return f"[기술 지원팀]\n{response.content}"

    @tool
    def handoff_to_billing(issue: str) -> str:
        """결제 팀으로 이관합니다.

        Args:
            issue: 결제 문제 설명

        Returns:
            결제 팀의 응답
        """
        prompt = f"""
당신은 결제 담당자입니다.
다음 문제를 해결하세요:

{issue}

명확한 안내를 제공하세요.
"""
        response = llm.invoke(prompt)
        return f"[결제팀]\n{response.content}"

    # 테스트
    print("\n🧪 Handoff 도구 테스트:")
    print("-" * 70)

    test_cases = [
        ("로그인이 안 됩니다", handoff_to_technical),
        ("결제가 실패했습니다", handoff_to_billing)
    ]

    for issue, handoff_tool in test_cases:
        print(f"\n문제: {issue}")
        print(f"Handoff: {handoff_tool.name}")
        print("\n실행 중...")

        result = handoff_tool.invoke({"issue": issue})
        print(f"\n{result}")
        print("-" * 70)

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 3: Context/State 전달
# ============================================================================

def example_3_context_passing():
    """Handoff 시 컨텍스트와 상태 전달"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Context/State 전달")
    print("=" * 70)

    print("""
💡 컨텍스트 전달의 중요성:
   - 이전 Agent의 작업 내역 보존
   - 사용자가 반복 설명하지 않도록
   - 효율적인 문제 해결
    """)

    # 컨텍스트 구조 정의
    class HandoffContext(TypedDict):
        """Handoff 시 전달할 컨텍스트"""
        user_id: str
        issue: str
        attempted_solutions: list[str]
        severity: str
        timestamp: str

    # Tier 1 Agent
    def tier1_agent(user_id: str, issue: str) -> dict:
        """Tier 1: 기본 지원"""
        print(f"\n[Tier 1] 고객 {user_id}의 문제 처리 중...")

        attempted = ["FAQ 검색", "자동 진단"]

        # 간단한 문제 체크
        if "재설정" in issue:
            return {
                "resolved": True,
                "response": "재설정 완료",
                "handoff": False
            }

        # 복잡한 문제는 Handoff
        context: HandoffContext = {
            "user_id": user_id,
            "issue": issue,
            "attempted_solutions": attempted,
            "severity": "medium",
            "timestamp": datetime.now().isoformat()
        }

        return {
            "resolved": False,
            "handoff": True,
            "context": context
        }

    # Tier 2 Agent
    def tier2_agent(context: HandoffContext) -> dict:
        """Tier 2: 전문 지원"""
        print(f"\n[Tier 2] Handoff 받음")
        print(f"고객 ID: {context['user_id']}")
        print(f"문제: {context['issue']}")
        print(f"이전 시도: {', '.join(context['attempted_solutions'])}")
        print(f"심각도: {context['severity']}")

        # 전문 해결
        return {
            "resolved": True,
            "response": "전문가가 문제를 해결했습니다.",
            "handoff": False
        }

    # 테스트
    print("\n🧪 컨텍스트 전달 테스트:")
    print("=" * 70)

    user_id = input("고객 ID (Enter=C12345): ").strip() or "C12345"
    issue = input("문제 설명 (Enter=기본값): ").strip() or "데이터베이스 연결 오류"

    # Tier 1 실행
    result = tier1_agent(user_id, issue)

    if result["handoff"]:
        print("\n🔄 Tier 2로 Handoff...")
        print(f"전달 컨텍스트:\n{result['context']}")

        # Tier 2 실행
        result = tier2_agent(result["context"])

    print(f"\n최종 결과: {result['response']}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 4: 조건부 Handoff
# ============================================================================

def example_4_conditional_handoff():
    """조건에 따라 다른 Agent로 Handoff"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 조건부 Handoff")
    print("=" * 70)

    print("""
💡 조건부 Handoff:
   - 문제 유형에 따라 적절한 Agent로 라우팅
   - 심각도에 따라 에스컬레이션
   - 시도 횟수에 따라 상급자에게 전달
    """)

    # 라우터 Agent
    def router_agent(user_input: str) -> dict:
        """입력을 분석하여 적절한 Agent로 Handoff"""
        print(f"\n[Router] 입력 분석 중...")
        print(f"입력: {user_input}")

        # 키워드 기반 라우팅
        if any(word in user_input for word in ["결제", "환불", "청구"]):
            return {
                "target": "billing",
                "reason": "결제 관련 문의"
            }
        elif any(word in user_input for word in ["오류", "버그", "작동"]):
            return {
                "target": "technical",
                "reason": "기술 문제"
            }
        elif any(word in user_input for word in ["계정", "로그인", "비밀번호"]):
            return {
                "target": "account",
                "reason": "계정 관련"
            }
        else:
            return {
                "target": "general",
                "reason": "일반 문의"
            }

    # 전문 Agent들
    def billing_agent(issue: str) -> str:
        """결제 전문 Agent"""
        return f"[결제팀] {issue}에 대한 결제 문제를 처리했습니다."

    def technical_agent(issue: str) -> str:
        """기술 전문 Agent"""
        return f"[기술팀] {issue}에 대한 기술 문제를 해결했습니다."

    def account_agent(issue: str) -> str:
        """계정 전문 Agent"""
        return f"[계정팀] {issue}에 대한 계정 문제를 처리했습니다."

    def general_agent(issue: str) -> str:
        """일반 Agent"""
        return f"[일반 상담] {issue}에 대해 안내드립니다."

    # Agent 매핑
    agents = {
        "billing": billing_agent,
        "technical": technical_agent,
        "account": account_agent,
        "general": general_agent
    }

    # 테스트
    print("\n🧪 조건부 Handoff 테스트:")
    print("=" * 70)

    test_inputs = [
        "결제가 실패했습니다",
        "앱이 계속 오류를 발생시킵니다",
        "비밀번호를 잊어버렸어요",
        "이용 방법을 알려주세요"
    ]

    for user_input in test_inputs:
        print(f"\n입력: {user_input}")

        # 라우팅 결정
        routing = router_agent(user_input)
        print(f"라우팅: {routing['target']} (이유: {routing['reason']})")

        # 해당 Agent로 Handoff
        target_agent = agents[routing["target"]]
        result = target_agent(user_input)
        print(f"결과: {result}")
        print("-" * 70)

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 5: 실전 - 고객 서비스 에스컬레이션
# ============================================================================

def example_5_customer_service_escalation():
    """실전: 3-Tier 고객 서비스 에스컬레이션 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 고객 서비스 에스컬레이션")
    print("=" * 70)

    print("""
🎯 실전 시나리오: 3-Tier 지원 시스템

   Tier 1 (자동 FAQ)
      ↓ (복잡한 문제)
   Tier 2 (일반 상담원)
      ↓ (해결 불가)
   Tier 3 (전문가/매니저)
    """)

    # 전역 상태
    conversation_history = []

    # Tier 1: FAQ Bot
    def tier1_faq_bot(user_input: str) -> dict:
        """Tier 1: 자동 FAQ 봇"""
        print(f"\n[Tier 1 FAQ Bot]")
        print(f"입력: {user_input}")

        conversation_history.append(f"User: {user_input}")

        # 간단한 FAQ
        faq_responses = {
            "영업시간": "영업시간은 평일 9시-18시입니다.",
            "배송": "배송은 2-3일 소요됩니다.",
            "반품": "구매 후 7일 이내 반품 가능합니다."
        }

        for keyword, response in faq_responses.items():
            if keyword in user_input:
                conversation_history.append(f"Bot: {response}")
                return {
                    "resolved": True,
                    "response": response,
                    "tier": 1
                }

        # FAQ로 해결 안 됨
        conversation_history.append(f"Bot: 상담원 연결이 필요합니다.")
        return {
            "resolved": False,
            "handoff_to": "tier2",
            "reason": "FAQ에 없는 문의",
            "context": {
                "user_input": user_input,
                "history": conversation_history.copy(),
                "attempts": 1
            }
        }

    # Tier 2: 일반 상담원
    def tier2_agent(context: dict) -> dict:
        """Tier 2: 일반 상담원"""
        print(f"\n[Tier 2 상담원]")
        print(f"전달된 문제: {context['user_input']}")
        print(f"시도 횟수: {context['attempts']}")

        prompt = f"""
당신은 고객 서비스 상담원입니다.
다음 고객 문의를 처리하세요:

{context['user_input']}

대화 기록:
{chr(10).join(context['history'])}

전문적이고 친절하게 답변하세요.
"""
        response = llm.invoke(prompt)
        answer = response.content

        conversation_history.append(f"Agent: {answer}")

        # 복잡한 문제는 Tier 3로
        if "복잡" in answer or "전문가" in answer or context["attempts"] >= 2:
            return {
                "resolved": False,
                "handoff_to": "tier3",
                "reason": "전문가 지원 필요",
                "context": {
                    "user_input": context["user_input"],
                    "history": conversation_history.copy(),
                    "tier2_response": answer,
                    "attempts": context["attempts"] + 1
                }
            }

        return {
            "resolved": True,
            "response": answer,
            "tier": 2
        }

    # Tier 3: 전문가/매니저
    def tier3_expert(context: dict) -> dict:
        """Tier 3: 전문가/매니저"""
        print(f"\n[Tier 3 전문가]")
        print(f"에스컬레이션된 문제: {context['user_input']}")
        print(f"총 시도: {context['attempts']}회")

        prompt = f"""
당신은 고객 서비스 매니저입니다.
다음 에스컬레이션된 문제를 해결하세요:

원래 문의: {context['user_input']}

대화 기록:
{chr(10).join(context['history'])}

Tier 2 응답: {context.get('tier2_response', 'N/A')}

모든 권한으로 문제를 해결하세요.
"""
        response = llm.invoke(prompt)
        answer = response.content

        conversation_history.append(f"Manager: {answer}")

        return {
            "resolved": True,
            "response": answer,
            "tier": 3,
            "escalated": True
        }

    # 전체 시스템
    def handle_customer_inquiry(user_input: str):
        """고객 문의 처리"""
        print("\n" + "=" * 70)
        print("고객 문의 처리 시작")
        print("=" * 70)

        # Tier 1 시작
        result = tier1_faq_bot(user_input)

        # Handoff 체인
        while not result.get("resolved"):
            handoff_target = result.get("handoff_to")

            if handoff_target == "tier2":
                print("\n🔄 Tier 2로 Handoff...")
                result = tier2_agent(result["context"])

            elif handoff_target == "tier3":
                print("\n🔄🔄 Tier 3로 에스컬레이션...")
                result = tier3_expert(result["context"])

            else:
                break

        # 최종 결과
        print("\n" + "=" * 70)
        print("처리 완료")
        print("=" * 70)
        print(f"해결 단계: Tier {result.get('tier', 'Unknown')}")
        print(f"에스컬레이션 여부: {'✅' if result.get('escalated') else '❌'}")
        print(f"\n최종 응답:\n{result.get('response', 'N/A')}")

        print("\n대화 기록:")
        for msg in conversation_history:
            print(f"  {msg}")

    # 테스트
    print("\n📞 고객 문의 예시:")
    print("-" * 70)

    choice = input("\n1. 간단한 문의 (FAQ)\n2. 복잡한 문의\n선택: ").strip()

    if choice == "1":
        user_input = "영업시간이 어떻게 되나요?"
    else:
        user_input = input("문의 내용을 입력하세요: ").strip() or "제품이 고장났는데 환불받을 수 있나요?"

    handle_customer_inquiry(user_input)

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
    print("04. Handoffs (제어 전달)")
    print("=" * 70)

    while True:
        print("\n")
        print("📚 실행할 예제를 선택하세요:")
        print("-" * 70)
        print("1. Handoff의 기본 개념")
        print("2. Agent 간 제어 전달")
        print("3. Context/State 전달")
        print("4. 조건부 Handoff")
        print("5. 실전: 고객 서비스 에스컬레이션")
        print("0. 종료")
        print("-" * 70)

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_handoff_concept()
        elif choice == "2":
            example_2_control_transfer()
        elif choice == "3":
            example_3_context_passing()
        elif choice == "4":
            example_4_conditional_handoff()
        elif choice == "5":
            example_5_customer_service_escalation()
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
   - Handoff 패턴의 개념과 Subagent와의 차이
   - Agent 간 명시적인 제어 전달
   - 컨텍스트와 상태를 보존하며 전달
   - 조건에 따른 동적 Handoff
   - 실전 3-Tier 에스컬레이션 시스템

💡 핵심 요약:
   ┌─────────────────────────────────────────────────────────────────┐
   │ Handoff는 Agent 간 제어를 완전히 전달하는 패턴                 │
   │                                                                   │
   │ 주요 특징:                                                       │
   │ • 이전 Agent 종료, 새 Agent가 제어                             │
   │ • 컨텍스트 전달로 연속성 유지                                   │
   │ • 에스컬레이션 및 티켓팅에 최적                                │
   │ • 조건부 라우팅 가능                                            │
   │                                                                   │
   │ 사용 시점:                                                       │
   │ • 고객 서비스 티어 시스템                                       │
   │ • 점진적 복잡도 증가 작업                                       │
   │ • 권한 기반 에스컬레이션                                        │
   └─────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()
