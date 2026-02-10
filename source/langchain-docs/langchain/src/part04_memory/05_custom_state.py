"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory System - Custom State
================================================================================

파일명: 05_custom_state.py
난이도: ⭐⭐⭐⭐☆ (중상급)
예상 시간: 25분

📚 학습 목표:
  - AgentState 확장 방법 이해
  - Custom Fields 추가 및 활용
  - state_schema 파라미터 사용
  - Tool에서 Custom State 접근
  - Middleware에서 State 수정

📖 공식 문서:
  • Short-term Memory: /official/10-short-term-memory.md

📄 교안 문서:
  • Part 4 메모리: /docs/part04_memory.md (Section 5)

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 05_custom_state.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, dynamic_prompt
from langchain.agents.middleware import ModelRequest
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)

# ============================================================================
# 예제 1: 기본 AgentState
# ============================================================================

def example_1_default_agent_state():
    """기본 AgentState의 구조"""
    print("=" * 70)
    print("📌 예제 1: 기본 AgentState")
    print("=" * 70)
    print("\n💡 기본 AgentState는 'messages' 필드만 포함합니다.\n")

    print("""
📦 기본 AgentState 구조:

class AgentState(TypedDict):
    messages: list[BaseMessage]

💡 특징:
   - messages: 대화 이력을 저장하는 리스트
   - add_messages reducer 사용 (자동 추가)
   - 다른 정보를 저장하려면 확장 필요

❌ 한계:
   - 사용자 정보 저장 불가
   - 세션 메타데이터 저장 불가
   - 비즈니스 로직 데이터 저장 불가
   - 카운터, 플래그 등 저장 불가

✅ 해결:
   - AgentState를 상속하여 확장
   - state_schema 파라미터로 전달
    """)

    # 기본 Agent 예제
    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "default-test"}}

    print("\n🔹 기본 Agent 테스트:")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕하세요!"}]},
        config
    )

    print(f"👤 사용자: 안녕하세요!")
    print(f"🤖 AI: {result['messages'][-1].content}")

    # State 구조 확인
    print("\n📊 State 구조:")
    print(f"   - 사용 가능한 키: {list(result.keys())}")
    print(f"   - messages 타입: {type(result['messages'])}")
    print(f"   - messages 길이: {len(result['messages'])}")


# ============================================================================
# 예제 2: Custom AgentState 정의
# ============================================================================

def example_2_custom_state():
    """Custom Fields를 가진 AgentState"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Custom AgentState 정의")
    print("=" * 70)
    print("\n💡 AgentState를 확장하여 추가 필드를 정의합니다.\n")

    # Custom State 정의
    class UserAgentState(AgentState):
        """사용자 정보를 포함하는 State"""
        user_id: str
        user_name: Optional[str] = None
        session_start: Optional[str] = None

    print("📦 Custom State 정의:")
    print("""
    class UserAgentState(AgentState):
        user_id: str                    # 필수 필드
        user_name: Optional[str] = None # 선택 필드
        session_start: Optional[str] = None
    """)

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        state_schema=UserAgentState,  # Custom State 지정
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "custom-test"}}

    # Custom State와 함께 호출
    print("\n🔹 Custom State 사용:")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "안녕하세요!"}],
            "user_id": "user-12345",
            "user_name": "김철수",
            "session_start": datetime.now().isoformat(),
        },
        config
    )

    print(f"👤 사용자 (user-12345, 김철수): 안녕하세요!")
    print(f"🤖 AI: {result['messages'][-1].content}")

    # State 확인
    print("\n📊 State 구조:")
    print(f"   - user_id: {result['user_id']}")
    print(f"   - user_name: {result['user_name']}")
    print(f"   - session_start: {result['session_start']}")
    print(f"   - messages: {len(result['messages'])}개")


# ============================================================================
# 예제 3: Tool에서 Custom State 읽기
# ============================================================================

def example_3_state_in_tools():
    """Tool에서 Custom State 접근"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Tool에서 Custom State 읽기")
    print("=" * 70)
    print("\n💡 ToolRuntime을 통해 Tool에서 State를 읽을 수 있습니다.\n")

    # Custom State
    class UserPreferenceState(AgentState):
        user_id: str
        language: str = "en"
        timezone: str = "UTC"
        theme: str = "light"

    # Tool 정의
    @tool
    def get_user_settings(runtime: ToolRuntime[None, UserPreferenceState]) -> str:
        """사용자 설정을 조회합니다."""
        state = runtime.state

        settings = {
            "언어": state["language"],
            "시간대": state["timezone"],
            "테마": state["theme"],
        }

        result = "\n".join([f"   - {k}: {v}" for k, v in settings.items()])
        return f"사용자 설정:\n{result}"

    @tool
    def change_theme(
        new_theme: str,
        runtime: ToolRuntime[None, UserPreferenceState]
    ) -> str:
        """테마를 변경합니다 (light/dark)."""
        # State는 Tool에서 직접 수정 불가
        # Command를 통해 수정해야 함 (고급 주제)
        return f"테마를 '{new_theme}'로 변경했습니다."

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_user_settings, change_theme],
        state_schema=UserPreferenceState,
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "tools-test"}}

    # 설정과 함께 호출
    print("🔹 사용자 설정 조회:")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "내 설정을 보여주세요."}],
            "user_id": "user-001",
            "language": "ko",
            "timezone": "Asia/Seoul",
            "theme": "dark",
        },
        config
    )

    print(f"👤 사용자: 내 설정을 보여주세요.")
    print(f"🤖 AI: {result['messages'][-1].content}")


# ============================================================================
# 예제 4: Middleware에서 State 수정
# ============================================================================

def example_4_state_in_middleware():
    """Middleware에서 State 수정"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Middleware에서 State 수정")
    print("=" * 70)
    print("\n💡 Middleware에서 State를 읽고 수정할 수 있습니다.\n")

    # Custom State with counters
    class CounterState(AgentState):
        user_id: str
        request_count: int = 0
        total_chars: int = 0

    # Request counter middleware
    @before_model
    def count_requests(state: CounterState, runtime: Runtime):
        """요청 수를 카운트"""
        current_count = state.get("request_count", 0)
        new_count = current_count + 1

        # 사용자 입력 길이 계산
        messages = state["messages"]
        if messages and messages[-1].type == "human":
            char_count = len(messages[-1].content)
            total_chars = state.get("total_chars", 0) + char_count
        else:
            total_chars = state.get("total_chars", 0)

        print(f"📊 요청 #{new_count} (총 {total_chars}자)")

        return {
            "request_count": new_count,
            "total_chars": total_chars,
        }

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        state_schema=CounterState,
        middleware=[count_requests],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "counter-test"}}

    # 여러 요청
    messages = [
        "안녕하세요!",
        "날씨가 좋네요.",
        "오늘 할 일을 알려주세요.",
    ]

    for msg in messages:
        print(f"\n👤 사용자: {msg}")

        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": msg}],
                "user_id": "user-001",
            },
            config
        )

        print(f"🤖 AI: {result['messages'][-1].content[:80]}...")

    # 최종 통계
    final_state = agent.get_state(config)
    print(f"\n📊 최종 통계:")
    print(f"   - 총 요청: {final_state.values['request_count']}")
    print(f"   - 총 입력: {final_state.values['total_chars']}자")


# ============================================================================
# 예제 5: 복잡한 Custom State (실전)
# ============================================================================

def example_5_complex_state():
    """실전 수준의 복잡한 Custom State"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 복잡한 Custom State (실전)")
    print("=" * 70)
    print("\n💡 실제 프로덕션 환경에서 사용할 수 있는 State 설계\n")

    # Context 정의
    @dataclass
    class UserContext:
        """사용자 컨텍스트 정보"""
        user_id: str
        session_id: str

    # 복잡한 State 정의
    class ProductionAgentState(AgentState):
        """프로덕션 수준의 Agent State"""

        # 사용자 정보
        user_id: str
        user_name: Optional[str] = None
        user_email: Optional[str] = None

        # 세션 정보
        session_id: str
        session_start: Optional[datetime] = None
        session_end: Optional[datetime] = None

        # 선호도
        language: str = "en"
        timezone: str = "UTC"
        theme: str = "light"

        # 통계
        request_count: int = 0
        error_count: int = 0
        total_tokens: int = 0

        # 메타데이터
        tags: List[str] = []
        metadata: dict = {}

        # 비즈니스 로직
        subscription_tier: str = "free"
        credits_remaining: int = 100

    print("📦 Production State 정의:")
    print("""
    class ProductionAgentState(AgentState):
        # 사용자 정보
        user_id: str
        user_name: Optional[str] = None
        user_email: Optional[str] = None

        # 세션 정보
        session_id: str
        session_start: Optional[datetime] = None

        # 선호도
        language: str = "en"
        timezone: str = "UTC"
        theme: str = "light"

        # 통계
        request_count: int = 0
        error_count: int = 0

        # 메타데이터
        tags: List[str] = []
        metadata: dict = {}

        # 비즈니스 로직
        subscription_tier: str = "free"
        credits_remaining: int = 100
    """)

    # Dynamic System Prompt
    @dynamic_prompt
    def personalized_prompt(request: ModelRequest):
        """사용자별 맞춤 시스템 프롬프트"""
        state = request.runtime.state

        language = state.get("language", "en")
        tier = state.get("subscription_tier", "free")
        credits = state.get("credits_remaining", 0)

        if language == "ko":
            prompt = f"당신은 친절한 AI 어시스턴트입니다."
            if tier == "premium":
                prompt += " 프리미엄 사용자에게 최상의 서비스를 제공하세요."
            prompt += f" (남은 크레딧: {credits})"
        else:
            prompt = f"You are a helpful AI assistant."
            if tier == "premium":
                prompt += " Provide premium service."
            prompt += f" (Credits: {credits})"

        return prompt

    # Tools
    @tool
    def check_subscription(
        runtime: ToolRuntime[UserContext, ProductionAgentState]
    ) -> str:
        """구독 정보를 확인합니다."""
        state = runtime.state

        tier = state["subscription_tier"]
        credits = state["credits_remaining"]

        return f"구독 등급: {tier}, 남은 크레딧: {credits}"

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[check_subscription],
        state_schema=ProductionAgentState,
        middleware=[personalized_prompt],
        checkpointer=checkpointer,
        context_schema=UserContext,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "production-test"}}

    # 복잡한 State로 호출
    print("\n🔹 프리미엄 사용자:")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "제 구독 정보를 알려주세요."}],
            "user_id": "user-premium-001",
            "user_name": "김프리미엄",
            "user_email": "premium@example.com",
            "session_id": "session-001",
            "session_start": datetime.now(),
            "language": "ko",
            "timezone": "Asia/Seoul",
            "theme": "dark",
            "subscription_tier": "premium",
            "credits_remaining": 500,
            "tags": ["vip", "enterprise"],
            "metadata": {"company": "ABC Corp"},
        },
        context=UserContext(
            user_id="user-premium-001",
            session_id="session-001"
        )
    )

    print(f"👤 프리미엄 사용자: 제 구독 정보를 알려주세요.")
    print(f"🤖 AI: {result['messages'][-1].content}")

    print("\n📊 State 활용:")
    print("   ✅ 언어별 맞춤 프롬프트")
    print("   ✅ 구독 등급별 서비스 차별화")
    print("   ✅ 크레딧 추적")
    print("   ✅ 세션 정보 관리")
    print("   ✅ 메타데이터 저장")


# ============================================================================
# 보너스: State 검증
# ============================================================================

def bonus_state_validation():
    """State 유효성 검증"""
    print("\n" + "=" * 70)
    print("🎁 보너스: State 검증")
    print("=" * 70)
    print("\n💡 Pydantic을 활용한 State 검증\n")

    from pydantic import Field, validator

    class ValidatedState(AgentState):
        """검증 로직이 있는 State"""
        user_id: str = Field(..., min_length=1, max_length=100)
        email: Optional[str] = Field(None, regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        age: Optional[int] = Field(None, ge=0, le=150)
        credits: int = Field(default=100, ge=0)

        @validator('user_id')
        def validate_user_id(cls, v):
            if not v.startswith('user-'):
                raise ValueError("user_id must start with 'user-'")
            return v

    print("""
📋 검증 규칙:
   - user_id: 'user-'로 시작, 1-100자
   - email: 유효한 이메일 형식
   - age: 0-150 사이
   - credits: 0 이상

💡 Pydantic이 자동으로 검증하여 잘못된 데이터를 방지합니다.
    """)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 4: Memory System - Custom State")
    print("\n")

    # 예제 1: 기본 State
    example_1_default_agent_state()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 2: Custom State
    example_2_custom_state()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 3: Tool에서 State
    example_3_state_in_tools()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 4: Middleware에서 State
    example_4_state_in_middleware()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 5: 복잡한 State
    example_5_complex_state()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 보너스: 검증
    print("\n" + "=" * 70)
    choice = input("🎁 보너스 예제를 보시겠습니까? (y/n): ").strip().lower()
    if choice == 'y':
        bonus_state_validation()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 4-5 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 06_long_term_store.py - Long-term Memory")
    print("\n📚 핵심 개념 복습:")
    print("  • AgentState 확장: 필요한 필드 추가")
    print("  • state_schema: Custom State 지정")
    print("  • ToolRuntime: Tool에서 State 접근")
    print("  • Middleware: State 읽기/수정")
    print("  • 검증: Pydantic으로 데이터 무결성 보장")
    print("\n💡 설계 팁:")
    print("  • 필요한 필드만 추가 (과도한 State 지양)")
    print("  • Optional 필드로 유연성 확보")
    print("  • 명확한 네이밍 규칙 사용")
    print("  • 검증 로직으로 안정성 향상")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. State 설계 원칙:
#    - 단순성: 필요한 것만 추가
#    - 명확성: 의미 있는 필드명
#    - 타입 안정성: 타입 힌트 활용
#    - 기본값: 합리적인 기본값 설정
#
# 2. State 분류:
#    - 사용자 정보: user_id, name, email
#    - 세션 정보: session_id, start_time
#    - 설정: language, timezone, theme
#    - 통계: request_count, token_count
#    - 비즈니스: subscription, credits
#
# 3. State 크기 관리:
#    - 큰 데이터는 Store에 저장
#    - State는 가벼운 메타데이터만
#    - 필요 시 Store에서 로드
#
# 4. State 마이그레이션:
#    - 버전 필드 추가
#    - 호환성 유지
#    - 점진적 마이그레이션
#
# ============================================================================
