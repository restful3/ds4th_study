"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context Engineering
================================================================================

파일명: 04_dynamic_model.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - wrap_model_call로 모델 동적 전환
  - 대화 길이/복잡도 기반 모델 선택
  - 비용 예산 제약 기반 모델 선택
  - 품질 요구사항 기반 모델 선택 (haiku→sonnet→opus)
  - Fallback 모델 체인 (실패시 대체)

📖 공식 문서:
  • Runtime: /official/18-runtime.md
  • Context Engineering: /official/19-context-engineering.md

📄 교안 문서:
  • Part 6: /docs/part06_context.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 04_dynamic_model.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.agents.agent import ModelRequest, ModelResponse
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
from typing import Callable

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: wrap_model_call로 모델 동적 전환
# ============================================================================

def example_1_dynamic_model_switching():
    """wrap_model_call을 사용하여 모델을 동적으로 전환"""
    print("=" * 70)
    print("📌 예제 1: wrap_model_call로 모델 동적 전환")
    print("=" * 70)

    print("""
💡 동적 모델 전환이란?
   - 실행 중에 사용할 LLM 모델을 변경
   - 상황에 따라 최적의 모델 선택
   - 비용과 성능의 균형

🎯 전환 기준:
   - 질문 복잡도
   - 대화 길이
   - 비용 예산
   - 응답 품질 요구사항
    """)

    # 도구 정의
    @tool
    def calculate(expression: str) -> str:
        """수학 계산 수행"""
        try:
            result = eval(expression)
            return f"계산 결과: {result}"
        except Exception as e:
            return f"계산 오류: {e}"

    @tool
    def search_info(topic: str) -> str:
        """정보 검색"""
        return f"'{topic}'에 대한 정보를 찾았습니다."

    # 모델 미리 초기화
    mini_model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # 동적 모델 전환
    @wrap_model_call
    def switch_model_by_length(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """메시지 길이에 따라 모델 전환"""

        message_count = len(request.messages)

        print(f"\n📊 현재 메시지 수: {message_count}개")

        if message_count > 10:
            # 긴 대화: 기본 모델 유지 (gpt-4o-mini)
            print("🔄 모델: gpt-4o-mini (기본 - 긴 대화)")
            # request.model은 이미 gpt-4o-mini
        elif message_count > 5:
            # 중간 대화: gpt-4o-mini 사용
            print("🔄 모델: gpt-4o-mini (중간 대화)")
            request = request.override(model=mini_model)
        else:
            # 짧은 대화: gpt-4o-mini 사용
            print("🔄 모델: gpt-4o-mini (짧은 대화)")
            request = request.override(model=mini_model)

        return handler(request)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[calculate, search_info],
        middleware=[switch_model_by_length],
        checkpointer=MemorySaver(),
    )

    # 여러 턴 대화
    config = {"configurable": {"thread_id": "model-switch-001"}}

    questions = [
        "안녕하세요",
        "1 + 1은?",
        "파이썬에 대해 알려줘",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"💬 질문 #{i}: {question}")
        print('='*60)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config
        )

        answer = response['messages'][-1].content
        print(f"\n🤖 응답: {answer[:100]}...")


# ============================================================================
# 예제 2: 대화 길이/복잡도 기반 모델 선택
# ============================================================================

def example_2_complexity_based_model():
    """질문의 복잡도를 분석하여 적절한 모델 선택"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 대화 길이/복잡도 기반 모델 선택")
    print("=" * 70)

    print("""
🧠 복잡도 분석 기준:
   - 질문 길이 (문자 수)
   - 키워드 ("분석", "전문", "상세")
   - 기술 용어 포함 여부
    """)

    @tool
    def analyze_data(data: str) -> str:
        """데이터 분석"""
        return f"'{data}' 분석 완료."

    # 복잡도 기반 모델 선택
    @wrap_model_call
    def complexity_based_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """질문 복잡도에 따라 모델 선택"""

        if request.messages:
            last_msg = request.messages[-1].content
        else:
            last_msg = ""

        # 복잡도 점수 계산
        complexity_score = 0

        # 길이 점수
        if len(last_msg) > 200:
            complexity_score += 3
        elif len(last_msg) > 100:
            complexity_score += 2
        elif len(last_msg) > 50:
            complexity_score += 1

        # 키워드 점수
        complex_keywords = ["분석", "전문", "상세", "심층", "비교", "평가"]
        for keyword in complex_keywords:
            if keyword in last_msg:
                complexity_score += 1

        print(f"\n💬 질문 길이: {len(last_msg)}자")
        print(f"📊 복잡도 점수: {complexity_score}")

        # 점수에 따라 모델 선택
        if complexity_score >= 4:
            # 매우 복잡: gpt-4o-mini 사용
            print("🎯 선택된 모델: gpt-4o-mini (복잡)")
            model = init_chat_model("gpt-4o-mini", model_provider="openai")
            request = request.override(model=model)
        elif complexity_score >= 2:
            # 중간 복잡도: gpt-4o-mini
            print("🎯 선택된 모델: gpt-4o-mini (중간)")
            model = init_chat_model("gpt-4o-mini", model_provider="openai")
            request = request.override(model=model)
        else:
            # 단순: gpt-4o-mini
            print("🎯 선택된 모델: gpt-4o-mini (단순)")
            model = init_chat_model("gpt-4o-mini", model_provider="openai")
            request = request.override(model=model)

        return handler(request)

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[analyze_data],
        middleware=[complexity_based_model],
        checkpointer=MemorySaver(),
    )

    # 다양한 복잡도 질문
    questions = [
        "안녕",
        "파이썬이 뭐야?",
        "파이썬과 자바스크립트를 비교 분석해서 각각의 장단점을 상세히 설명해줘",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"💬 질문: {question}")
        print('='*60)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": "complexity-001"}}
        )

        answer = response['messages'][-1].content
        print(f"\n🤖 응답: {answer[:150]}...")


# ============================================================================
# 예제 3: 비용 예산 제약 기반 모델 선택
# ============================================================================

def example_3_budget_based_model():
    """일일 예산을 추적하여 모델 선택"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 비용 예산 제약 기반 모델 선택")
    print("=" * 70)

    print("""
💰 비용 최적화 전략:
   - 예산 충분: 고급 모델 (gpt-4o-mini)
   - 예산 부족: 저렴한 모델 (gpt-4o-mini)
   - 예산 초과: 경고 메시지
    """)

    @dataclass
    class BudgetContext:
        user_id: str
        daily_budget: float  # 달러
        spent_today: float

    @tool
    def get_recommendation(topic: str) -> str:
        """추천 정보 제공"""
        return f"'{topic}' 추천 정보입니다."

    # 예산 기반 모델 선택
    @wrap_model_call
    def budget_based_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """예산에 따라 모델 선택"""

        ctx = request.runtime.context
        budget = ctx.daily_budget
        spent = ctx.spent_today
        remaining = budget - spent

        print(f"\n💰 예산 현황:")
        print(f"  - 일일 예산: ${budget:.2f}")
        print(f"  - 사용 금액: ${spent:.2f}")
        print(f"  - 남은 금액: ${remaining:.2f}")

        # 예산 기준 모델 선택
        if remaining >= 1.0:
            # 예산 충분: gpt-4o-mini
            print("🎯 선택: gpt-4o-mini (예산 충분)")
            model = init_chat_model("gpt-4o-mini", model_provider="openai")
        elif remaining >= 0.1:
            # 예산 적음: gpt-4o-mini
            print("🎯 선택: gpt-4o-mini (예산 적음)")
            model = init_chat_model("gpt-4o-mini", model_provider="openai")
        else:
            # 예산 거의 소진: gpt-4o-mini (가장 저렴)
            print("⚠️ 선택: gpt-4o-mini (예산 거의 소진)")
            model = init_chat_model("gpt-4o-mini", model_provider="openai")

        request = request.override(model=model)
        return handler(request)

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_recommendation],
        middleware=[budget_based_model],
        context_schema=BudgetContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 예산 시나리오
    budgets = [
        ("user_rich", 10.0, 5.0),   # 충분한 예산
        ("user_low", 5.0, 4.5),     # 적은 예산
        ("user_poor", 1.0, 0.95),   # 거의 소진
    ]

    for user_id, budget, spent in budgets:
        print(f"\n{'='*60}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "추천 좀 해줘"}]},
            context=BudgetContext(
                user_id=user_id,
                daily_budget=budget,
                spent_today=spent
            ),
            config={"configurable": {"thread_id": f"budget-{user_id}"}}
        )
        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 4: 품질 요구사항 기반 모델 선택
# ============================================================================

def example_4_quality_based_model():
    """요청의 품질 요구사항에 따라 모델 선택"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 품질 요구사항 기반 모델 선택")
    print("=" * 70)

    print("""
⭐ 품질 레벨:
   - 표준: gpt-4o-mini
   - 고품질: gpt-4o-mini
   - 최고품질: gpt-4o-mini
    """)

    @dataclass
    class QualityContext:
        user_id: str
        quality_tier: str  # "standard", "high", "premium"

    @tool
    def create_content(topic: str) -> str:
        """콘텐츠 생성"""
        return f"'{topic}' 콘텐츠를 생성했습니다."

    # 품질 기반 모델 선택
    @wrap_model_call
    def quality_based_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """품질 요구사항에 따라 모델 선택"""

        ctx = request.runtime.context
        tier = ctx.quality_tier

        print(f"\n⭐ 품질 티어: {tier.upper()}")

        # 티어별 모델 매핑
        tier_models = {
            "standard": ("gpt-4o-mini", "표준 모델"),
            "high": ("gpt-4o-mini", "고품질 모델"),
            "premium": ("gpt-4o-mini", "프리미엄 모델"),
        }

        model_name, description = tier_models.get(tier, ("gpt-4o-mini", "기본 모델"))

        print(f"🎯 선택: {description}")

        model = init_chat_model(model_name, model_provider="openai")
        request = request.override(model=model)

        return handler(request)

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[create_content],
        middleware=[quality_based_model],
        context_schema=QualityContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 품질 티어
    tiers = ["standard", "high", "premium"]

    for tier in tiers:
        print(f"\n{'='*60}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "기사를 작성해줘"}]},
            context=QualityContext(
                user_id=f"user_{tier}",
                quality_tier=tier
            ),
            config={"configurable": {"thread_id": f"quality-{tier}"}}
        )
        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 5: Fallback 모델 체인 (실패시 대체)
# ============================================================================

def example_5_fallback_model_chain():
    """모델 실패 시 자동으로 대체 모델 사용"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Fallback 모델 체인 (실패시 대체)")
    print("=" * 70)

    print("""
🔄 Fallback 전략:
   1. 먼저 gpt-4o-mini 시도
   2. 실패 시 gpt-4o-mini로 대체
   3. 모두 실패 시 에러 메시지

💡 실패 원인:
   - API 오류
   - 타임아웃
   - Rate limit 초과
   - 기타 예외
    """)

    @tool
    def process_request(text: str) -> str:
        """요청 처리"""
        return f"'{text}' 처리 완료."

    # Fallback 모델 체인
    @wrap_model_call
    def fallback_chain(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Fallback 모델 체인"""

        # 1차 시도: gpt-4o-mini
        print("\n🔵 1차 시도: gpt-4o-mini")
        primary_model = init_chat_model("gpt-4o-mini", model_provider="openai")
        request = request.override(model=primary_model)

        try:
            response = handler(request)
            print("✅ 성공!")
            return response
        except Exception as e:
            print(f"❌ 실패: {e}")

            # 2차 시도: gpt-4o-mini (대체)
            print("\n🟡 2차 시도: gpt-4o-mini (대체)")
            fallback_model = init_chat_model("gpt-4o-mini", model_provider="openai")
            request = request.override(model=fallback_model)

            try:
                response = handler(request)
                print("✅ 성공!")
                return response
            except Exception as e2:
                print(f"❌ 실패: {e2}")

                # 3차 시도: 최후 모델
                print("\n🔴 3차 시도: gpt-4o-mini (최후)")
                last_resort = init_chat_model("gpt-4o-mini", model_provider="openai")
                request = request.override(model=last_resort)

                response = handler(request)
                print("✅ 성공!")
                return response

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[process_request],
        middleware=[fallback_chain],
        checkpointer=MemorySaver(),
    )

    # 테스트
    print(f"\n{'='*60}")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕하세요"}]},
        config={"configurable": {"thread_id": "fallback-001"}}
    )
    print(f"\n💬 최종 응답: {response['messages'][-1].content}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 6: Context Engineering - Dynamic Model")
    print("\n")

    try:
        # 예제 1: 모델 동적 전환
        example_1_dynamic_model_switching()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 2: 복잡도 기반
        example_2_complexity_based_model()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 3: 예산 기반
        example_3_budget_based_model()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 4: 품질 기반
        example_4_quality_based_model()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 5: Fallback 체인
        example_5_fallback_model_chain()

        # 마무리
        print("\n" + "=" * 70)
        print("🎉 Part 6 - Dynamic Model 완료!")
        print("=" * 70)
        print("\n💡 배운 내용:")
        print("  ✅ wrap_model_call로 모델 동적 전환")
        print("  ✅ 복잡도 기반 모델 선택")
        print("  ✅ 예산 제약 기반 선택")
        print("  ✅ 품질 요구사항 기반 선택")
        print("  ✅ Fallback 모델 체인")
        print("\n📚 다음 단계:")
        print("  ➡️ 05_tool_runtime.py - ToolRuntime")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 프로그램을 중단했습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 동적 모델의 장점:
#    - 비용 최적화
#    - 성능과 품질의 균형
#    - 자동 Fallback으로 안정성 향상
#
# 2. 모델 선택 기준:
#    - 질문 복잡도
#    - 대화 길이
#    - 비용 예산
#    - 품질 요구사항
#    - 응답 시간 제약
#
# 3. 실전 팁:
#    - 항상 Fallback 준비
#    - 비용 추적 및 모니터링
#    - A/B 테스팅으로 최적 모델 찾기
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "모델이 전환되지 않음"
# 해결: request.override(model=...)로 수정된 request를 handler에 전달
#
# 문제: "init_chat_model이 실패함"
# 해결: model_provider를 명시적으로 지정
#
# 문제: "비용이 예상보다 많이 나옴"
# 해결: 각 모델의 토큰당 가격을 정확히 파악하고 추적
#
# ============================================================================
