"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent Systems
================================================================================

파일명: 06_router_pattern.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - Router 패턴의 개념 이해
  - LLM 기반 입력 분류
  - 전문가 Agent로 라우팅
  - Fallback 라우팅
  - 실전: 멀티도메인 챗봇

📖 공식 문서:
  • Router: /official/26-router.md

📄 교안 문서:
  • Part 7 Router: /docs/part07_multi_agent.md (Section 5)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 06_router_pattern.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def example_1_router_concept():
    """Router 패턴의 기본 개념"""
    print("=" * 70)
    print("📌 예제 1: Router 패턴 개념")
    print("=" * 70)

    print("""
💡 Router 패턴이란?
   - 사용자 입력을 분석하여 적절한 Agent로 라우팅
   - 여러 전문 Agent 중 하나를 선택
   - 병렬 처리 가능한 독립적 작업에 적합

🔄 작동 방식:
   1. 사용자 입력
   2. Router가 입력 분류
   3. 적절한 전문 Agent로 전달
   4. Agent가 처리 후 결과 반환
    """)

    print("\n📝 예시 시나리오:")
    print("-" * 70)

    scenarios = [
        ("오늘 서울 날씨 알려줘", "Weather Agent"),
        ("2 + 3 * 4 계산해줘", "Math Agent"),
        ("최신 뉴스 보여줘", "News Agent"),
        ("안녕하세요", "General Agent")
    ]

    for user_input, target_agent in scenarios:
        print(f"\n입력: {user_input}")
        print(f"라우팅: → {target_agent}")

    input("\n⏎ Enter를 눌러 계속...")

def example_2_llm_classification():
    """LLM을 사용한 지능형 입력 분류"""
    print("\n" + "=" * 70)
    print("📌 예제 2: LLM 기반 입력 분류")
    print("=" * 70)

    print("""
💡 LLM 기반 분류:
   - Structured Output으로 정확한 분류
   - 신뢰도와 이유 함께 반환
   - 복잡한 의도 파악 가능
    """)

    class RouteCategory(str, Enum):
        WEATHER = "weather"
        MATH = "math"
        NEWS = "news"
        GENERAL = "general"

    class RouteDecision(BaseModel):
        category: RouteCategory = Field(description="입력 카테고리")
        confidence: float = Field(description="분류 신뢰도", ge=0, le=1)
        reasoning: str = Field(description="분류 이유")

    classifier = llm.with_structured_output(RouteDecision)

    test_inputs = [
        "내일 날씨 어때?",
        "15 곱하기 23은?",
        "최신 AI 뉴스",
        "좋은 아침"
    ]

    print("\n🧪 분류 테스트:")
    print("=" * 70)

    for user_input in test_inputs:
        print(f"\n입력: {user_input}")

        prompt = f"""
다음 입력을 분류하세요: "{user_input}"

카테고리:
- weather: 날씨 관련
- math: 수학/계산
- news: 뉴스/최신 정보
- general: 일반 대화
"""
        try:
            result = classifier.invoke(prompt)
            print(f"카테고리: {result.category.value}")
            print(f"신뢰도: {result.confidence:.0%}")
            print(f"이유: {result.reasoning}")
        except Exception as e:
            print(f"분류 실패: {e}")

    input("\n⏎ Enter를 눌러 계속...")

def example_3_expert_routing():
    """전문가 Agent로 라우팅"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 전문가 Agent로 라우팅")
    print("=" * 70)

    print("""
💡 전문가 라우팅:
   - 각 도메인별 전문 Agent 준비
   - Router가 적절한 전문가 선택
   - 높은 정확도와 품질 보장
    """)

    # 전문가 Agent들
    def weather_agent(query: str) -> str:
        """날씨 전문 Agent"""
        prompt = f"날씨 전문가로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[날씨 전문가]\n{response.content}"

    def math_agent(query: str) -> str:
        """수학 전문 Agent"""
        prompt = f"수학 전문가로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[수학 전문가]\n{response.content}"

    def news_agent(query: str) -> str:
        """뉴스 전문 Agent"""
        prompt = f"뉴스 전문가로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[뉴스 전문가]\n{response.content}"

    def general_agent(query: str) -> str:
        """일반 Agent"""
        prompt = f"일반 어시스턴트로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[일반 어시스턴트]\n{response.content}"

    # Router
    def route_to_expert(user_input: str) -> str:
        """입력을 분석하여 전문가로 라우팅"""

        # 키워드 기반 라우팅
        if any(word in user_input for word in ["날씨", "기온", "비", "눈"]):
            print("→ 날씨 전문가로 라우팅")
            return weather_agent(user_input)
        elif any(word in user_input for word in ["계산", "+", "-", "*", "/", "="]):
            print("→ 수학 전문가로 라우팅")
            return math_agent(user_input)
        elif any(word in user_input for word in ["뉴스", "최신", "소식"]):
            print("→ 뉴스 전문가로 라우팅")
            return news_agent(user_input)
        else:
            print("→ 일반 어시스턴트로 라우팅")
            return general_agent(user_input)

    print("\n🧪 라우팅 테스트:")
    print("=" * 70)

    user_input = input("\n입력: ").strip() or "서울 날씨 알려줘"
    print(f"\n처리 중...")
    result = route_to_expert(user_input)
    print(f"\n{result}")

    input("\n⏎ Enter를 눌러 계속...")

def example_4_fallback_routing():
    """Fallback 라우팅 전략"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Fallback 라우팅")
    print("=" * 70)

    print("""
💡 Fallback 전략:
   - 분류 실패 시 기본 Agent 사용
   - 신뢰도 낮을 때 재분류
   - 안전한 폴백 처리
    """)

    class RouteWithConfidence(BaseModel):
        category: str
        confidence: float

    def classify_with_confidence(user_input: str) -> RouteWithConfidence:
        """신뢰도와 함께 분류"""
        classifier = llm.with_structured_output(RouteWithConfidence)
        prompt = f"다음 입력을 분류하세요 (weather/math/news/general): {user_input}"
        return classifier.invoke(prompt)

    def smart_router(user_input: str) -> str:
        """Fallback이 있는 스마트 Router"""

        try:
            # 분류 시도
            result = classify_with_confidence(user_input)
            print(f"분류: {result.category} (신뢰도: {result.confidence:.0%})")

            # 신뢰도 체크
            if result.confidence < 0.6:
                print("⚠️ 신뢰도 낮음 → Fallback to General Agent")
                return "[일반 응답] 무엇을 도와드릴까요?"

            # 카테고리별 처리
            if result.category == "weather":
                return f"[날씨] {user_input}에 대한 날씨 정보"
            elif result.category == "math":
                return f"[계산] {user_input} 계산 결과"
            elif result.category == "news":
                return f"[뉴스] {user_input} 관련 뉴스"
            else:
                return f"[일반] {user_input}에 대한 응답"

        except Exception as e:
            print(f"❌ 분류 실패: {e}")
            print("→ Fallback to General Agent")
            return "[일반 응답] 다시 말씀해 주시겠어요?"

    print("\n🧪 Fallback 테스트:")
    print("=" * 70)

    test_cases = [
        "날씨",
        "애매한 질문",
        "!!!@#$"
    ]

    for test_input in test_cases:
        print(f"\n입력: {test_input}")
        result = smart_router(test_input)
        print(f"결과: {result}")

    input("\n⏎ Enter를 눌러 계속...")

def example_5_multidomain_chatbot():
    """실전: 멀티도메인 챗봇"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 멀티도메인 챗봇")
    print("=" * 70)

    print("""
🎯 실전 시나리오: 기업 통합 챗봇

도메인:
   - HR: 인사/급여
   - IT: 기술 지원
   - Finance: 재무
   - General: 일반 문의
    """)

    class Department(str, Enum):
        HR = "hr"
        IT = "it"
        FINANCE = "finance"
        GENERAL = "general"

    class RoutingDecision(BaseModel):
        department: Department
        confidence: float
        suggested_action: str

    # 부서별 Agent
    def hr_agent(query: str) -> str:
        prompt = f"HR 담당자로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[HR팀]\n{response.content}"

    def it_agent(query: str) -> str:
        prompt = f"IT 지원팀으로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[IT팀]\n{response.content}"

    def finance_agent(query: str) -> str:
        prompt = f"재무팀으로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[재무팀]\n{response.content}"

    def general_agent(query: str) -> str:
        prompt = f"일반 상담원으로서 답변: {query}"
        response = llm.invoke(prompt)
        return f"[일반 상담]\n{response.content}"

    # Multidomain Router
    def route_to_department(user_input: str) -> str:
        classifier = llm.with_structured_output(RoutingDecision)

        prompt = f"""
직원 요청을 부서로 분류하세요: "{user_input}"

부서:
- hr: 급여, 휴가, 복지
- it: 비밀번호, 소프트웨어, 하드웨어
- finance: 예산, 청구서, 경비
- general: 기타
"""

        decision = classifier.invoke(prompt)

        print(f"\n라우팅: {decision.department.value.upper()}팀")
        print(f"신뢰도: {decision.confidence:.0%}")
        print(f"제안: {decision.suggested_action}")

        agents = {
            Department.HR: hr_agent,
            Department.IT: it_agent,
            Department.FINANCE: finance_agent,
            Department.GENERAL: general_agent
        }

        return agents[decision.department](user_input)

    print("\n💼 통합 챗봇 시작")
    print("=" * 70)

    while True:
        user_input = input("\n문의 (종료: quit): ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        result = route_to_department(user_input)
        print(f"\n{result}")

    print("\n챗봇 종료")

    input("\n⏎ Enter를 눌러 계속...")

def main():
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("=" * 70)
    print("Part 7: Multi-Agent Systems")
    print("06. Router Pattern (라우터 패턴)")
    print("=" * 70)

    while True:
        print("\n")
        print("📚 실행할 예제를 선택하세요:")
        print("-" * 70)
        print("1. Router 패턴 개념")
        print("2. LLM 기반 입력 분류")
        print("3. 전문가 Agent로 라우팅")
        print("4. Fallback 라우팅")
        print("5. 실전: 멀티도메인 챗봇")
        print("0. 종료")
        print("-" * 70)

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_router_concept()
        elif choice == "2":
            example_2_llm_classification()
        elif choice == "3":
            example_3_expert_routing()
        elif choice == "4":
            example_4_fallback_routing()
        elif choice == "5":
            example_5_multidomain_chatbot()
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
   - Router 패턴의 개념과 작동 방식
   - LLM 기반 지능형 입력 분류
   - 전문가 Agent로의 정확한 라우팅
   - Fallback 전략으로 안전한 처리
   - 실전 멀티도메인 통합 챗봇

💡 핵심 요약:
   ┌─────────────────────────────────────────────────────────────────┐
   │ Router는 입력을 분석하여 적절한 전문 Agent로 라우팅            │
   │                                                                   │
   │ 주요 특징:                                                       │
   │ • 독립적인 Agent들 병렬 실행                                    │
   │ • Structured Output으로 정확한 분류                             │
   │ • 신뢰도 기반 Fallback 처리                                     │
   │ • 확장 가능한 아키텍처                                          │
   │                                                                   │
   │ 사용 시점:                                                       │
   │ • 여러 도메인을 다루는 챗봇                                     │
   │ • 독립적인 전문가 Agent 조합                                    │
   │ • 병렬 처리 가능한 작업                                         │
   └─────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()
