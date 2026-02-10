"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context Engineering
================================================================================

파일명: 01_context_overview.py
난이도: ⭐⭐⭐ (중급)
예상 시간: 25분

📚 학습 목표:
  - Runtime 객체의 개념과 구조 이해
  - Context vs State의 차이점 명확히 구분
  - RunnableConfig를 통한 실행 설정 접근
  - Runtime 정보를 활용한 동적 Agent 구성

📖 공식 문서:
  • Runtime: /official/18-runtime.md
  • Context Engineering: /official/19-context-engineering.md

📄 교안 문서:
  • Part 6 개요: /docs/part06_context.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🚀 실행 방법:
  python 01_context_overview.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, after_model
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
# 예제 1: Runtime 객체 기본 사용
# ============================================================================

def example_1_runtime_basics():
    """Runtime 객체의 기본 구조와 사용법"""
    print("=" * 70)
    print("📌 예제 1: Runtime 객체 기본 사용")
    print("=" * 70)

    print("""
💡 Runtime 객체란?
   - LangGraph가 제공하는 실행 컨텍스트 정보
   - Agent 실행에 필요한 모든 메타데이터 포함
   - Middleware와 Tool에서 접근 가능

📦 Runtime이 포함하는 정보:
   1. context: 정적 설정 (사용자 ID, API 키 등)
   2. store: 장기 메모리 저장소
   3. config: 실행 설정 (thread_id, checkpoint_id 등)
   4. stream: 스트리밍 출력 writer
    """)

    # before_model 훅에서 Runtime 정보 출력
    @before_model
    def inspect_runtime(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("\n🔍 Runtime 객체 검사:")
        print(f"  - Config 타입: {type(runtime.config)}")
        print(f"  - Config 내용: {runtime.config}")

        # Config에서 thread_id 추출
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "없음")
        print(f"  - Thread ID: {thread_id}")

        # Store 확인 (있다면)
        if runtime.store:
            print(f"  - Store 타입: {type(runtime.store)}")
        else:
            print("  - Store: 없음")

        return None

    # 간단한 도구
    @tool
    def get_info(topic: str) -> str:
        """주제에 대한 정보 제공"""
        return f"{topic}에 대한 정보입니다."

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_info],
        middleware=[inspect_runtime],
        checkpointer=MemorySaver(),
    )

    # 실행
    config = {"configurable": {"thread_id": "runtime-test-001"}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Runtime에 대해 설명해줘"}]},
        config=config
    )

    print(f"\n✅ 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 2: Context vs State 비교
# ============================================================================

def example_2_context_vs_state():
    """Context와 State의 차이점 명확히 이해"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Context vs State 비교")
    print("=" * 70)

    print("""
🔑 핵심 차이점:

┌─────────────┬──────────────────┬──────────────────┐
│  특성       │  Context         │  State           │
├─────────────┼──────────────────┼──────────────────┤
│ 변경 가능성  │ 불변 (Immutable) │ 가변 (Mutable)   │
│ 범위        │ 전체 실행         │ 턴마다 변경      │
│ 설정 시점    │ invoke() 호출 시 │ 실행 중 업데이트 │
│ 용도        │ 사용자 ID, 권한  │ 메시지, 파일     │
└─────────────┴──────────────────┴──────────────────┘
    """)

    # Context 스키마 정의
    @dataclass
    class UserContext:
        user_id: str
        user_name: str
        user_tier: str  # "free", "premium"

    # Context와 State 모두 사용하는 middleware
    @before_model
    def compare_context_state(
        state: AgentState,
        runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        print("\n📊 Context vs State 비교:")

        # Context 접근 (불변)
        print(f"\n  🔒 Context (불변):")
        print(f"    - User ID: {runtime.context.user_id}")
        print(f"    - User Name: {runtime.context.user_name}")
        print(f"    - User Tier: {runtime.context.user_tier}")

        # State 접근 (가변)
        print(f"\n  🔄 State (가변):")
        print(f"    - 메시지 수: {len(state.get('messages', []))}")

        # State에 커스텀 필드 추가 가능
        if "visit_count" not in state:
            state["visit_count"] = 0
        state["visit_count"] += 1

        print(f"    - 방문 횟수: {state['visit_count']}")

        # Context 기반으로 프롬프트 변경
        if runtime.context.user_tier == "premium":
            prompt = f"{runtime.context.user_name}님, 프리미엄 회원님을 위한 상세한 답변을 제공하겠습니다."
        else:
            prompt = f"{runtime.context.user_name}님, 간단하게 답변드리겠습니다."

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"],
            "visit_count": state.get("visit_count", 0) + 1
        }

    @tool
    def simple_tool(text: str) -> str:
        """간단한 도구"""
        return f"처리 완료: {text}"

    # Agent 생성 (context_schema 지정)
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[simple_tool],
        middleware=[compare_context_state],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # 프리미엄 사용자로 실행
    print("\n🌟 프리미엄 사용자 테스트:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕하세요"}]},
        context=UserContext(
            user_id="user_001",
            user_name="김철수",
            user_tier="premium"
        ),
        config={"configurable": {"thread_id": "context-test-001"}}
    )
    print(f"\n✅ 응답: {response['messages'][-1].content[:100]}...")

    # 일반 사용자로 실행
    print("\n\n👤 일반 사용자 테스트:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕하세요"}]},
        context=UserContext(
            user_id="user_002",
            user_name="이영희",
            user_tier="free"
        ),
        config={"configurable": {"thread_id": "context-test-002"}}
    )
    print(f"\n✅ 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 3: RunnableConfig 접근
# ============================================================================

def example_3_runnable_config():
    """RunnableConfig를 통한 실행 설정 접근"""
    print("\n" + "=" * 70)
    print("📌 예제 3: RunnableConfig 접근")
    print("=" * 70)

    print("""
🎛️ RunnableConfig란?
   - LangChain의 실행 설정 객체
   - Runtime.config를 통해 접근
   - thread_id, checkpoint_id 등 포함
    """)

    @before_model
    def access_config(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Config 정보 접근 및 출력"""
        config = runtime.config

        print("\n📋 Config 정보:")
        print(f"  - 전체 Config: {config}")

        # Configurable 정보 추출
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "없음")
        checkpoint_id = configurable.get("checkpoint_id", "없음")

        print(f"\n  🔑 주요 설정:")
        print(f"    - Thread ID: {thread_id}")
        print(f"    - Checkpoint ID: {checkpoint_id}")

        # Thread ID 기반으로 다른 프롬프트 제공
        if "vip" in str(thread_id).lower():
            prompt = "VIP 스레드입니다. 최상의 서비스를 제공하겠습니다."
        else:
            prompt = "일반 스레드입니다. 친절하게 답변드리겠습니다."

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"]
        }

    @tool
    def check_thread(runtime_param: Any = None) -> str:
        """현재 스레드 정보 확인"""
        return "스레드 정보를 확인했습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[check_thread],
        middleware=[access_config],
        checkpointer=MemorySaver(),
    )

    # 일반 스레드
    print("\n📌 일반 스레드:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕"}]},
        config={"configurable": {"thread_id": "normal-thread-001"}}
    )
    print(f"✅ 응답: {response['messages'][-1].content[:100]}...")

    # VIP 스레드
    print("\n\n⭐ VIP 스레드:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "안녕"}]},
        config={"configurable": {"thread_id": "vip-thread-001"}}
    )
    print(f"✅ 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 4: Thread ID와 User ID 활용
# ============================================================================

def example_4_thread_and_user():
    """Thread ID와 User ID를 활용한 맞춤형 응답"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Thread ID와 User ID 활용")
    print("=" * 70)

    print("""
🎯 실전 활용:
   - Thread ID: 대화 세션 식별
   - User ID: 사용자 식별 (Context)
   - 두 정보를 조합하여 개인화된 경험 제공
    """)

    @dataclass
    class UserContext:
        user_id: str
        user_name: str
        preferences: dict[str, str]

    @before_model
    def personalized_greeting(
        state: AgentState,
        runtime: Runtime[UserContext]
    ) -> dict[str, Any] | None:
        """Thread와 User 정보 기반 개인화"""

        # User 정보 (Context)
        user_id = runtime.context.user_id
        user_name = runtime.context.user_name
        prefs = runtime.context.preferences

        # Thread 정보 (Config)
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")

        print(f"\n👤 사용자 정보:")
        print(f"  - User ID: {user_id}")
        print(f"  - User Name: {user_name}")
        print(f"  - Thread ID: {thread_id}")
        print(f"  - 선호도: {prefs}")

        # 선호도 기반 프롬프트
        tone = prefs.get("tone", "친절한")
        language = prefs.get("language", "한국어")

        prompt = f"""
당신은 {tone} 톤으로 {language}로 답변하는 도우미입니다.
사용자 이름: {user_name}
사용자 ID: {user_id}
대화 스레드: {thread_id}
        """.strip()

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"]
        }

    @tool
    def get_user_stats(query: str) -> str:
        """사용자 통계 조회"""
        return f"{query}에 대한 통계입니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_user_stats],
        middleware=[personalized_greeting],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # 사용자 1: 전문적인 톤 선호
    print("\n💼 전문적인 톤 선호 사용자:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 날씨가 어때?"}]},
        context=UserContext(
            user_id="user_001",
            user_name="김대리",
            preferences={"tone": "전문적인", "language": "한국어"}
        ),
        config={"configurable": {"thread_id": "work-thread-001"}}
    )
    print(f"✅ 응답: {response['messages'][-1].content[:150]}...")

    # 사용자 2: 친근한 톤 선호
    print("\n\n😊 친근한 톤 선호 사용자:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 날씨가 어때?"}]},
        context=UserContext(
            user_id="user_002",
            user_name="영희",
            preferences={"tone": "친근한", "language": "한국어"}
        ),
        config={"configurable": {"thread_id": "casual-thread-001"}}
    )
    print(f"✅ 응답: {response['messages'][-1].content[:150]}...")


# ============================================================================
# 예제 5: Runtime 정보 종합 활용
# ============================================================================

def example_5_comprehensive_runtime():
    """Runtime의 모든 정보를 활용한 고급 예제"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Runtime 정보 종합 활용")
    print("=" * 70)

    print("""
🚀 고급 활용:
   - Context: 사용자 권한 및 설정
   - Config: 스레드 및 실행 설정
   - State: 대화 상태 및 중간 데이터
   - 모든 정보를 조합하여 똑똑한 Agent 구성
    """)

    @dataclass
    class AdvancedContext:
        user_id: str
        user_role: str  # "admin", "user", "guest"
        org_id: str
        quota: int  # 일일 사용 한도

    @before_model
    def comprehensive_middleware(
        state: AgentState,
        runtime: Runtime[AdvancedContext]
    ) -> dict[str, Any] | None:
        """모든 Runtime 정보 활용"""

        # Context 정보
        user_id = runtime.context.user_id
        user_role = runtime.context.user_role
        org_id = runtime.context.org_id
        quota = runtime.context.quota

        # Config 정보
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")

        # State 정보
        message_count = len(state.get("messages", []))
        usage_count = state.get("usage_count", 0)

        print(f"\n📊 종합 정보:")
        print(f"  📍 Context:")
        print(f"    - User ID: {user_id}")
        print(f"    - Role: {user_role}")
        print(f"    - Org ID: {org_id}")
        print(f"    - Quota: {quota}")
        print(f"  🔧 Config:")
        print(f"    - Thread ID: {thread_id}")
        print(f"  💾 State:")
        print(f"    - 메시지 수: {message_count}")
        print(f"    - 사용 횟수: {usage_count}")

        # 권한 체크
        if usage_count >= quota:
            prompt = f"일일 사용 한도({quota})를 초과했습니다. 내일 다시 이용해주세요."
            return {
                "messages": state["messages"] + [
                    {"role": "assistant", "content": prompt}
                ]
            }

        # 역할별 프롬프트
        role_prompts = {
            "admin": f"관리자({user_id})님, 모든 기능을 사용할 수 있습니다.",
            "user": f"사용자({user_id})님, 기본 기능을 이용할 수 있습니다.",
            "guest": f"게스트({user_id})님, 제한된 기능만 이용 가능합니다."
        }

        prompt = role_prompts.get(user_role, "알 수 없는 역할입니다.")

        return {
            "messages": [
                {"role": "system", "content": prompt}
            ] + state["messages"],
            "usage_count": usage_count + 1
        }

    @after_model
    def log_usage(
        state: AgentState,
        runtime: Runtime[AdvancedContext]
    ) -> dict[str, Any] | None:
        """사용 로그 기록"""
        usage = state.get("usage_count", 0)
        quota = runtime.context.quota

        print(f"\n📈 사용 현황: {usage}/{quota} (남은 횟수: {quota - usage})")

        return None

    @tool
    def admin_tool(command: str) -> str:
        """관리자 전용 도구"""
        return f"관리자 명령 실행: {command}"

    @tool
    def user_tool(query: str) -> str:
        """일반 사용자 도구"""
        return f"일반 쿼리 처리: {query}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[admin_tool, user_tool],
        middleware=[comprehensive_middleware, log_usage],
        context_schema=AdvancedContext,
        checkpointer=MemorySaver(),
    )

    # 관리자 사용
    print("\n👑 관리자 사용:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "시스템 상태 확인"}]},
        context=AdvancedContext(
            user_id="admin_001",
            user_role="admin",
            org_id="org_main",
            quota=100
        ),
        config={"configurable": {"thread_id": "admin-session-001"}}
    )
    print(f"✅ 응답: {response['messages'][-1].content[:100]}...")

    # 일반 사용자 (할당량 제한)
    print("\n\n👤 일반 사용자 (할당량 적음):")
    config = {"configurable": {"thread_id": "user-session-001"}}

    for i in range(3):
        print(f"\n  요청 #{i+1}:")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"질문 {i+1}"}]},
            context=AdvancedContext(
                user_id="user_001",
                user_role="user",
                org_id="org_main",
                quota=2  # 낮은 할당량
            ),
            config=config
        )
        print(f"  ✅ 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 6: Context Engineering - Context Overview")
    print("\n")

    try:
        # 예제 1: Runtime 객체 기본
        example_1_runtime_basics()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 2: Context vs State
        example_2_context_vs_state()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 3: RunnableConfig 접근
        example_3_runnable_config()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 4: Thread ID와 User ID
        example_4_thread_and_user()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 5: 종합 활용
        example_5_comprehensive_runtime()

        # 마무리
        print("\n" + "=" * 70)
        print("🎉 Part 6 - Context Overview 완료!")
        print("=" * 70)
        print("\n💡 배운 내용:")
        print("  ✅ Runtime 객체의 구조와 역할")
        print("  ✅ Context vs State의 핵심 차이")
        print("  ✅ RunnableConfig를 통한 설정 접근")
        print("  ✅ Thread ID와 User ID 활용")
        print("  ✅ Runtime 정보 종합 활용")
        print("\n📚 다음 단계:")
        print("  ➡️ 02_dynamic_prompt.py - 동적 프롬프트")
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
# 1. Runtime의 주요 구성 요소:
#    - context: 정적 설정 (사용자 ID, 권한 등)
#    - store: 장기 메모리 (BaseStore)
#    - config: 실행 설정 (thread_id, checkpoint_id)
#    - stream: 스트리밍 writer
#
# 2. Context vs State:
#    Context는 불변(Immutable), State는 가변(Mutable)
#    Context는 invoke() 호출 시 설정, State는 실행 중 변경
#
# 3. RunnableConfig 활용:
#    thread_id로 대화 세션 관리
#    checkpoint_id로 특정 시점 복원
#    configurable로 커스텀 설정 전달
#
# 4. Context Schema:
#    dataclass로 타입 안전한 Context 정의
#    context_schema 파라미터로 Agent에 전달
#    runtime.context로 접근
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "Context has no attribute 'user_id'"
# 해결: context_schema를 Agent 생성 시 지정했는지 확인
#
# 문제: "State 변경이 다음 턴에 반영되지 않음"
# 해결: Checkpointer를 사용하고 있는지 확인
#
# 문제: "Runtime에서 thread_id를 찾을 수 없음"
# 해결: invoke() 호출 시 config 파라미터로 thread_id 전달
#
# ============================================================================
