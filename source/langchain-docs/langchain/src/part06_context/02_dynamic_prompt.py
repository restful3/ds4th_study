"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context Engineering
================================================================================

파일명: 02_dynamic_prompt.py
난이도: ⭐⭐⭐ (중급)
예상 시간: 25분

📚 학습 목표:
  - before_model을 사용한 동적 프롬프트 수정
  - 사용자 정보 기반 맞춤형 프롬프트 생성
  - 시간대별 프롬프트 자동 변경
  - 대화 히스토리 분석을 통한 프롬프트 조정
  - A/B 테스팅을 위한 프롬프트 변형

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
  python 02_dynamic_prompt.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import hashlib
import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.tools import tool
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
from typing import Any

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: before_model로 프롬프트 수정
# ============================================================================

def example_1_before_model_basics():
    """before_model 훅으로 프롬프트를 동적으로 수정"""
    print("=" * 70)
    print("📌 예제 1: before_model로 프롬프트 수정")
    print("=" * 70)

    print("""
💡 before_model이란?
   - 모델 호출 직전에 실행되는 훅(Hook)
   - 메시지 목록을 동적으로 수정 가능
   - 시스템 프롬프트를 실시간으로 주입

🎯 사용 사례:
   - 사용자별 맞춤 프롬프트
   - 시간대별 인사말 변경
   - 컨텍스트 정보 주입
    """)

    # 기본 프롬프트 주입 middleware
    @before_model
    def inject_basic_prompt(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """모든 모델 호출에 기본 프롬프트 추가"""

        # 기본 시스템 메시지
        system_msg = """
당신은 친절하고 도움이 되는 AI 어시스턴트입니다.
항상 정중하고 명확하게 답변하세요.
        """.strip()

        print(f"\n✅ 시스템 프롬프트 주입: {system_msg[:50]}...")

        # 메시지 앞에 시스템 메시지 추가
        return {
            "messages": [
                {"role": "system", "content": system_msg}
            ] + state["messages"]
        }

    @tool
    def get_weather(city: str) -> str:
        """도시의 날씨 정보 제공"""
        return f"{city}의 날씨는 맑습니다."

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_weather],
        middleware=[inject_basic_prompt],
        checkpointer=MemorySaver(),
    )

    # 실행
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕하세요"}]},
        config={"configurable": {"thread_id": "prompt-test-001"}}
    )

    print(f"\n💬 응답: {response['messages'][-1].content}")
    print("\n✅ before_model로 프롬프트가 동적으로 주입되었습니다!")


# ============================================================================
# 예제 2: 사용자 이름 기반 맞춤 프롬프트
# ============================================================================

def example_2_user_customized_prompt():
    """사용자 정보를 활용한 개인화된 프롬프트"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 사용자 이름 기반 맞춤 프롬프트")
    print("=" * 70)

    print("""
🎯 개인화 전략:
   - 사용자 이름으로 친근감 증대
   - 사용자 레벨에 따른 응답 스타일 변경
   - 선호도 반영
    """)

    # 사용자 Context 정의
    @dataclass
    class UserContext:
        user_id: str
        user_name: str
        user_level: str  # "bronze", "silver", "gold"
        language: str

    # 사용자별 맞춤 프롬프트
    @before_model
    def personalized_prompt(
        state: AgentState,
        runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        """사용자 정보 기반 프롬프트 생성"""

        ctx = runtime.context

        # 레벨별 응답 스타일
        level_styles = {
            "bronze": "간단하고 핵심만 전달하는",
            "silver": "균형잡힌 상세도의",
            "gold": "매우 상세하고 전문적인"
        }

        style = level_styles.get(ctx.user_level, "기본적인")

        # 개인화된 프롬프트 생성
        prompt = f"""
당신은 {ctx.user_name}님을 위한 개인 AI 어시스턴트입니다.

응답 스타일: {style}
언어: {ctx.language}
사용자 레벨: {ctx.user_level.upper()}

{ctx.user_name}님께 최상의 서비스를 제공하세요.
        """.strip()

        print(f"\n👤 사용자: {ctx.user_name} ({ctx.user_level})")
        print(f"📝 프롬프트 스타일: {style}")

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"]
        }

    @tool
    def search_info(topic: str) -> str:
        """주제에 대한 정보 검색"""
        return f"{topic}에 대한 정보를 찾았습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_info],
        middleware=[personalized_prompt],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # Bronze 사용자
    print("\n🥉 Bronze 사용자:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "인공지능에 대해 설명해줘"}]},
        context=UserContext(
            user_id="user_001",
            user_name="김철수",
            user_level="bronze",
            language="한국어"
        ),
        config={"configurable": {"thread_id": "bronze-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    # Gold 사용자
    print("\n\n🥇 Gold 사용자:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "인공지능에 대해 설명해줘"}]},
        context=UserContext(
            user_id="user_002",
            user_name="박영희",
            user_level="gold",
            language="한국어"
        ),
        config={"configurable": {"thread_id": "gold-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 3: 시간대별 인사말 변경
# ============================================================================

def example_3_time_based_greeting():
    """현재 시간에 따라 인사말과 톤을 자동으로 변경"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 시간대별 인사말 변경")
    print("=" * 70)

    print("""
⏰ 시간대별 최적화:
   - 아침 (06:00-12:00): 활기찬 톤
   - 오후 (12:00-18:00): 전문적인 톤
   - 저녁 (18:00-24:00): 편안한 톤
   - 밤 (00:00-06:00): 간결한 톤
    """)

    @before_model
    def time_based_prompt(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """시간대별 프롬프트 자동 생성"""

        # 현재 시간 정보
        now = datetime.datetime.now()
        hour = now.hour
        day_name = now.strftime("%A")
        date_str = now.strftime("%Y년 %m월 %d일")

        # 시간대별 설정
        if 6 <= hour < 12:
            period = "아침"
            greeting = "좋은 아침입니다!"
            tone = "활기차고 긍정적인"
            emoji = "🌅"
        elif 12 <= hour < 18:
            period = "오후"
            greeting = "좋은 오후입니다!"
            tone = "전문적이고 효율적인"
            emoji = "☀️"
        elif 18 <= hour < 24:
            period = "저녁"
            greeting = "좋은 저녁입니다!"
            tone = "편안하고 친근한"
            emoji = "🌙"
        else:
            period = "밤"
            greeting = "늦은 시간이네요!"
            tone = "간결하고 배려하는"
            emoji = "⭐"

        # 시간 기반 프롬프트
        prompt = f"""
{greeting}

현재 시간: {date_str} {hour}시 ({period})
요일: {day_name}

당신은 {tone} 톤으로 답변하는 AI 어시스턴트입니다.
현재 시간대를 고려하여 적절한 응답을 제공하세요.
        """.strip()

        print(f"\n{emoji} {period} 시간대 감지 ({hour}시)")
        print(f"📝 톤: {tone}")

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"]
        }

    @tool
    def get_schedule(date: str) -> str:
        """일정 조회"""
        return f"{date}의 일정을 확인했습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_schedule],
        middleware=[time_based_prompt],
        checkpointer=MemorySaver(),
    )

    # 현재 시간 기준 응답
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 할 일이 뭐가 있을까?"}]},
        config={"configurable": {"thread_id": "time-test-001"}}
    )

    print(f"\n💬 응답: {response['messages'][-1].content}")
    print("\n✅ 시간대에 맞는 프롬프트가 적용되었습니다!")


# ============================================================================
# 예제 4: 이전 대화 분석 기반 프롬프트 조정
# ============================================================================

def example_4_conversation_aware_prompt():
    """대화 히스토리를 분석하여 프롬프트를 동적으로 조정"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 이전 대화 분석 기반 프롬프트 조정")
    print("=" * 70)

    print("""
🧠 대화 분석 전략:
   - 짧은 대화: 상세하고 친절한 설명
   - 중간 대화: 균형잡힌 응답
   - 긴 대화: 간결하고 핵심적인 답변
   - 반복 질문: 다른 관점의 설명 제공
    """)

    @before_model
    def conversation_aware_prompt(
        state: AgentState,
        runtime: Runtime
    ) -> dict[str, Any] | None:
        """대화 길이와 패턴 분석"""

        messages = state.get("messages", [])
        message_count = len(messages)

        # 최근 대화 분석
        recent_topics = []
        if message_count > 2:
            # 마지막 몇 개 메시지에서 주제 추출 (단순 키워드)
            for msg in messages[-3:]:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                else:
                    content = msg.content
                recent_topics.append(content[:20])

        # 대화 길이별 프롬프트 전략
        if message_count < 5:
            # 초기 대화
            strategy = "상세하고 친절한"
            guidance = "사용자와의 첫 대화이므로 자세하게 설명하고 추가 질문을 유도하세요."
            level = "초기"
        elif message_count < 15:
            # 중간 대화
            strategy = "균형잡힌"
            guidance = "대화가 어느 정도 진행되었으므로 핵심을 유지하되 충분한 정보를 제공하세요."
            level = "중간"
        else:
            # 긴 대화
            strategy = "간결하고 핵심적인"
            guidance = "대화가 많이 진행되었습니다. 요점만 간결하게 전달하세요."
            level = "후반"

        # 프롬프트 생성
        prompt = f"""
당신은 {strategy} 톤으로 답변하는 AI 어시스턴트입니다.

대화 단계: {level} ({message_count}개 메시지)
응답 전략: {guidance}

항상 맥락을 고려하여 적절한 상세도로 답변하세요.
        """.strip()

        print(f"\n📊 대화 분석:")
        print(f"  - 메시지 수: {message_count}개")
        print(f"  - 대화 단계: {level}")
        print(f"  - 전략: {strategy}")

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"]
        }

    @tool
    def ask_question(topic: str) -> str:
        """주제에 대해 질문"""
        return f"{topic}에 대한 답변을 준비했습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[ask_question],
        middleware=[conversation_aware_prompt],
        checkpointer=MemorySaver(),
    )

    # 여러 턴의 대화
    config = {"configurable": {"thread_id": "conversation-001"}}

    questions = [
        "파이썬이 뭐야?",
        "파이썬의 장점은?",
        "그럼 단점은?",
        "다른 언어와 비교하면?",
        "어떻게 배우면 좋을까?",
        "추천 자료 있어?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n\n{'='*50}")
        print(f"💬 질문 #{i}: {question}")
        print('='*50)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config
        )

        answer = response['messages'][-1].content
        print(f"\n🤖 응답: {answer[:200]}...")

        if i < len(questions):
            print("\n⏳ 다음 질문...")


# ============================================================================
# 예제 5: A/B 테스트용 프롬프트 변형
# ============================================================================

def example_5_ab_testing_prompts():
    """사용자 ID를 기반으로 A/B 테스팅 프롬프트 적용"""
    print("\n" + "=" * 70)
    print("📌 예제 5: A/B 테스트용 프롬프트 변형")
    print("=" * 70)

    print("""
🧪 A/B 테스팅 전략:
   - 사용자 ID를 해시하여 일관된 그룹 분할
   - Variant A: 간결한 스타일
   - Variant B: 상세한 스타일
   - 각 변형의 성능 측정 가능
    """)

    @dataclass
    class UserContext:
        user_id: str
        user_name: str

    @before_model
    def ab_test_prompt(
        state: AgentState,
        runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        """A/B 테스팅을 위한 프롬프트 변형"""

        user_id = runtime.context.user_id
        user_name = runtime.context.user_name

        # 사용자 ID를 해시하여 그룹 결정
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant = "A" if hash_value % 2 == 0 else "B"

        # Variant별 프롬프트
        if variant == "A":
            # 간결한 스타일
            prompt = f"""
당신은 {user_name}님을 위한 AI 어시스턴트입니다.

[Variant A - 간결한 스타일]
- 핵심만 전달
- 짧고 명확한 문장
- 불필요한 설명 최소화
- 3-4 문장으로 제한
            """.strip()
            style = "간결"
            emoji = "⚡"
        else:
            # 상세한 스타일
            prompt = f"""
당신은 {user_name}님을 위한 AI 어시스턴트입니다.

[Variant B - 상세한 스타일]
- 충분한 배경 설명 제공
- 예시와 근거 포함
- 추가 정보와 팁 제공
- 이해하기 쉽게 단계별 설명
            """.strip()
            style = "상세"
            emoji = "📚"

        print(f"\n{emoji} A/B 테스트 그룹: Variant {variant} ({style})")
        print(f"👤 사용자: {user_name} (ID: {user_id})")

        # 메트릭 수집 (실제로는 DB나 로그에 기록)
        state["ab_variant"] = variant

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"],
            "ab_variant": variant
        }

    @tool
    def calculate(expression: str) -> str:
        """간단한 계산 수행"""
        try:
            result = eval(expression)
            return f"계산 결과: {result}"
        except:
            return "계산할 수 없습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[calculate],
        middleware=[ab_test_prompt],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # 여러 사용자로 테스트
    test_users = [
        ("user_001", "김철수"),
        ("user_002", "이영희"),
        ("user_003", "박민수"),
        ("user_004", "정지원"),
    ]

    question = "머신러닝이 뭐야?"

    for user_id, user_name in test_users:
        print(f"\n{'='*60}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            context=UserContext(user_id=user_id, user_name=user_name),
            config={"configurable": {"thread_id": f"ab-test-{user_id}"}}
        )

        variant = response.get("ab_variant", "Unknown")
        answer = response['messages'][-1].content

        print(f"\n💬 응답 (길이: {len(answer)}자):")
        print(f"{answer[:150]}...")
        print(f"\n📊 Variant: {variant}")

    print("\n\n✅ A/B 테스팅 완료!")
    print("📈 실제 프로덕션에서는 응답 시간, 사용자 만족도 등을 측정합니다.")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 6: Context Engineering - Dynamic Prompt")
    print("\n")

    try:
        # 예제 1: before_model 기본
        example_1_before_model_basics()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 2: 사용자 맞춤 프롬프트
        example_2_user_customized_prompt()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 3: 시간대별 인사말
        example_3_time_based_greeting()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 4: 대화 분석 기반 조정
        example_4_conversation_aware_prompt()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 5: A/B 테스팅
        example_5_ab_testing_prompts()

        # 마무리
        print("\n" + "=" * 70)
        print("🎉 Part 6 - Dynamic Prompt 완료!")
        print("=" * 70)
        print("\n💡 배운 내용:")
        print("  ✅ before_model로 프롬프트 동적 수정")
        print("  ✅ 사용자 정보 기반 개인화")
        print("  ✅ 시간대별 자동 조정")
        print("  ✅ 대화 히스토리 분석")
        print("  ✅ A/B 테스팅 구현")
        print("\n📚 다음 단계:")
        print("  ➡️ 03_dynamic_tools.py - 동적 도구")
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
# 1. before_model의 특징:
#    - 모델 호출 직전에 실행
#    - 메시지 목록을 수정 가능
#    - State에 저장되어 지속됨 (Persistent)
#
# 2. 동적 프롬프트의 장점:
#    - 개인화된 사용자 경험
#    - 컨텍스트 인식 응답
#    - 시간/상황별 최적화
#
# 3. A/B 테스팅:
#    - 해시 함수로 일관된 그룹 분할
#    - 변형별 성능 측정
#    - 데이터 기반 의사결정
#
# 4. 프롬프트 설계 원칙:
#    - 명확하고 구체적으로
#    - 역할과 제약사항 명시
#    - 예시 제공 시 효과적
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "프롬프트가 적용되지 않음"
# 해결: before_model이 middleware에 등록되었는지 확인
#
# 문제: "Context 정보에 접근할 수 없음"
# 해결: context_schema를 Agent 생성 시 지정했는지 확인
#
# 문제: "시간대가 잘못 표시됨"
# 해결: datetime.now()는 로컬 시간 기준. 타임존 고려 필요
#
# ============================================================================
