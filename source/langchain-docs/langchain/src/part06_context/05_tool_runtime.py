"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context Engineering
================================================================================

파일명: 05_tool_runtime.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - ToolRuntime 파라미터 기본 사용법
  - Tool 내에서 State 읽기 및 수정
  - Tool에서 Store 접근 (장기 메모리)
  - Tool에서 Config 및 Thread ID 활용
  - 실전: 사용자별 설정 저장/로드

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
  python 05_tool_runtime.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
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
# 예제 1: ToolRuntime 파라미터 기본 사용법
# ============================================================================

def example_1_tool_runtime_basics():
    """ToolRuntime 파라미터의 기본 사용법"""
    print("=" * 70)
    print("📌 예제 1: ToolRuntime 파라미터 기본 사용법")
    print("=" * 70)

    print("""
💡 ToolRuntime이란?
   - Tool에서 Runtime 정보에 접근하는 특수 파라미터
   - context, store, config, state 등에 접근 가능
   - 타입 힌트: ToolRuntime[ContextType]

🎯 접근 가능한 정보:
   - runtime.context: 정적 컨텍스트
   - runtime.store: 장기 메모리
   - runtime.config: 실행 설정
   - runtime.state: 현재 상태
    """)

    # Context 정의
    @dataclass
    class UserContext:
        user_id: str
        user_name: str

    # ToolRuntime을 사용하는 도구
    @tool
    def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
        """현재 사용자 정보 조회"""

        # Context 접근
        ctx = runtime.context
        user_id = ctx.user_id
        user_name = ctx.user_name

        # Config 접근
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")

        # State 접근
        message_count = len(runtime.state.get("messages", []))

        info = f"""
👤 사용자 정보:
  - ID: {user_id}
  - 이름: {user_name}
  - 스레드: {thread_id}
  - 메시지 수: {message_count}
        """.strip()

        print(f"\n{info}")

        return info

    @tool
    def simple_calc(a: int, b: int) -> str:
        """간단한 계산 (ToolRuntime 없음)"""
        return f"{a} + {b} = {a + b}"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_user_info, simple_calc],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # 실행
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내 정보를 알려줘"}]},
        context=UserContext(user_id="user_001", user_name="김철수"),
        config={"configurable": {"thread_id": "runtime-001"}}
    )

    print(f"\n💬 응답: {response['messages'][-1].content}")
    print("\n✅ ToolRuntime으로 도구 내에서 런타임 정보에 접근했습니다!")


# ============================================================================
# 예제 2: Tool 내에서 State 읽기 및 수정
# ============================================================================

def example_2_state_access():
    """Tool에서 State 읽기 및 분석"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Tool 내에서 State 읽기 및 수정")
    print("=" * 70)

    print("""
📊 State 접근:
   - runtime.state로 현재 상태 읽기
   - 메시지 히스토리 분석
   - 커스텀 상태 필드 확인
   - ⚠️ Tool에서 State 수정은 불가 (읽기 전용)
    """)

    @tool
    def analyze_conversation(runtime: ToolRuntime) -> str:
        """대화 히스토리 분석"""

        state = runtime.state
        messages = state.get("messages", [])

        # 메시지 통계
        total_count = len(messages)
        user_msgs = [m for m in messages if getattr(m, "type", m.get("role")) == "user"]
        ai_msgs = [m for m in messages if getattr(m, "type", m.get("role")) == "assistant"]

        # 최근 주제 추출 (간단한 방법)
        recent_topics = []
        for msg in messages[-3:]:
            content = getattr(msg, "content", msg.get("content", ""))
            if len(content) > 0:
                recent_topics.append(content[:30])

        analysis = f"""
📊 대화 분석:
  - 전체 메시지: {total_count}개
  - 사용자 메시지: {len(user_msgs)}개
  - AI 응답: {len(ai_msgs)}개
  - 최근 주제: {', '.join(recent_topics) if recent_topics else '없음'}
        """.strip()

        print(f"\n{analysis}")

        return analysis

    @tool
    def check_custom_state(runtime: ToolRuntime) -> str:
        """커스텀 State 필드 확인"""

        state = runtime.state

        # 커스텀 필드 확인
        visit_count = state.get("visit_count", 0)
        last_action = state.get("last_action", "없음")

        info = f"""
📌 커스텀 State:
  - 방문 횟수: {visit_count}
  - 마지막 액션: {last_action}
        """.strip()

        print(f"\n{info}")

        return info

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[analyze_conversation, check_custom_state],
        checkpointer=MemorySaver(),
    )

    # 여러 턴 대화
    config = {"configurable": {"thread_id": "state-test-001"}}

    messages = [
        "안녕하세요",
        "대화를 분석해줘",
        "상태를 확인해줘",
    ]

    for msg in messages:
        print(f"\n{'='*60}")
        print(f"💬 사용자: {msg}")
        print('='*60)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config
        )

        answer = response['messages'][-1].content
        print(f"\n🤖 응답: {answer[:150]}...")


# ============================================================================
# 예제 3: Tool에서 Store 접근 (장기 메모리)
# ============================================================================

def example_3_store_access():
    """Tool에서 Store를 사용한 장기 메모리 접근"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Tool에서 Store 접근 (장기 메모리)")
    print("=" * 70)

    print("""
💾 Store 활용:
   - runtime.store로 장기 메모리 접근
   - 사용자 선호도, 설정 저장
   - 대화 간 정보 유지
   - put/get으로 데이터 관리
    """)

    @dataclass
    class UserContext:
        user_id: str

    # Store에 데이터 저장
    @tool
    def save_preference(
        key: str,
        value: str,
        runtime: ToolRuntime[UserContext]
    ) -> str:
        """사용자 선호도 저장"""

        user_id = runtime.context.user_id
        store = runtime.store

        # 기존 선호도 가져오기
        namespace = ("preferences", user_id)
        existing = store.get(namespace, "data")

        if existing:
            prefs = existing.value
        else:
            prefs = {}

        # 새 선호도 추가
        prefs[key] = value

        # Store에 저장
        store.put(namespace, "data", prefs)

        print(f"\n💾 저장됨: {key} = {value}")
        print(f"📦 전체 선호도: {prefs}")

        return f"'{key}'를 '{value}'(으)로 저장했습니다."

    # Store에서 데이터 조회
    @tool
    def get_preferences(runtime: ToolRuntime[UserContext]) -> str:
        """사용자 선호도 조회"""

        user_id = runtime.context.user_id
        store = runtime.store

        # Store에서 읽기
        namespace = ("preferences", user_id)
        item = store.get(namespace, "data")

        if item:
            prefs = item.value
            pref_list = [f"  - {k}: {v}" for k, v in prefs.items()]
            result = "저장된 선호도:\n" + "\n".join(pref_list)
        else:
            result = "저장된 선호도가 없습니다."

        print(f"\n📖 {result}")

        return result

    # Store 초기화
    store = InMemoryStore()

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[save_preference, get_preferences],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
        store=store,
    )

    # 선호도 저장
    print("\n1️⃣ 선호도 저장:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "테마를 dark로 설정해줘"}]},
        context=UserContext(user_id="user_001"),
        config={"configurable": {"thread_id": "store-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content[:100]}...")

    # 선호도 조회
    print("\n\n2️⃣ 선호도 조회:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내 선호도를 보여줘"}]},
        context=UserContext(user_id="user_001"),
        config={"configurable": {"thread_id": "store-002"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 4: Tool에서 Config 및 Thread ID 활용
# ============================================================================

def example_4_config_and_thread():
    """Tool에서 Config와 Thread ID 활용"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Tool에서 Config 및 Thread ID 활용")
    print("=" * 70)

    print("""
🔧 Config 활용:
   - runtime.config로 실행 설정 접근
   - thread_id로 대화 세션 구분
   - 세션별 데이터 관리
    """)

    @tool
    def save_note(note: str, runtime: ToolRuntime) -> str:
        """Thread별 노트 저장"""

        # Thread ID 가져오기
        thread_id = runtime.config.get("configurable", {}).get("thread_id", "default")
        store = runtime.store

        # Thread별 namespace
        namespace = ("notes", thread_id)

        # 기존 노트 가져오기
        existing = store.get(namespace, "data")
        if existing:
            notes = existing.value
        else:
            notes = []

        # 새 노트 추가
        notes.append(note)

        # 저장
        store.put(namespace, "data", notes)

        print(f"\n💾 노트 저장:")
        print(f"  - Thread: {thread_id}")
        print(f"  - 노트: {note}")
        print(f"  - 총 {len(notes)}개")

        return f"노트를 저장했습니다. (총 {len(notes)}개)"

    @tool
    def list_notes(runtime: ToolRuntime) -> str:
        """현재 Thread의 노트 목록"""

        thread_id = runtime.config.get("configurable", {}).get("thread_id", "default")
        store = runtime.store

        namespace = ("notes", thread_id)
        item = store.get(namespace, "data")

        if item:
            notes = item.value
            note_list = [f"{i+1}. {note}" for i, note in enumerate(notes)]
            result = f"저장된 노트 ({len(notes)}개):\n" + "\n".join(note_list)
        else:
            result = "저장된 노트가 없습니다."

        print(f"\n📝 {result}")

        return result

    store = InMemoryStore()

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[save_note, list_notes],
        checkpointer=MemorySaver(),
        store=store,
    )

    # Thread 1에서 노트 저장
    print("\n📌 Thread 1:")
    config1 = {"configurable": {"thread_id": "thread-001"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "회의 시간 오후 3시라고 노트해줘"}]},
        config=config1
    )
    print(f"💬 응답: {response['messages'][-1].content[:100]}...")

    # Thread 2에서 노트 저장
    print("\n\n📌 Thread 2:")
    config2 = {"configurable": {"thread_id": "thread-002"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내일 발표 준비하기라고 노트해줘"}]},
        config=config2
    )
    print(f"💬 응답: {response['messages'][-1].content[:100]}...")

    # 각 Thread의 노트 조회
    print("\n\n📋 Thread 1 노트 조회:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "노트 목록 보여줘"}]},
        config=config1
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    print("\n\n📋 Thread 2 노트 조회:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "노트 목록 보여줘"}]},
        config=config2
    )
    print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 5: 실전 - 사용자별 설정 저장/로드
# ============================================================================

def example_5_user_settings():
    """실전 예제: 사용자별 설정 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 사용자별 설정 저장/로드")
    print("=" * 70)

    print("""
🎯 사용자 설정 시스템:
   - 언어, 테마, 알림 설정
   - Store를 사용한 영구 저장
   - 사용자별 독립적인 설정
   - 설정 조회, 업데이트, 초기화
    """)

    @dataclass
    class UserContext:
        user_id: str
        user_name: str

    @tool
    def update_settings(
        setting_name: str,
        setting_value: str,
        runtime: ToolRuntime[UserContext]
    ) -> str:
        """설정 업데이트"""

        user_id = runtime.context.user_id
        user_name = runtime.context.user_name
        store = runtime.store

        namespace = ("settings", user_id)

        # 기존 설정 가져오기
        existing = store.get(namespace, "config")
        if existing:
            settings = existing.value
        else:
            # 기본 설정
            settings = {
                "language": "ko",
                "theme": "light",
                "notifications": "enabled"
            }

        # 설정 업데이트
        settings[setting_name] = setting_value

        # 저장
        store.put(namespace, "config", settings)

        print(f"\n⚙️ 설정 업데이트:")
        print(f"  - 사용자: {user_name} ({user_id})")
        print(f"  - 설정: {setting_name} = {setting_value}")

        return f"'{setting_name}'을 '{setting_value}'(으)로 변경했습니다."

    @tool
    def get_settings(runtime: ToolRuntime[UserContext]) -> str:
        """현재 설정 조회"""

        user_id = runtime.context.user_id
        user_name = runtime.context.user_name
        store = runtime.store

        namespace = ("settings", user_id)
        item = store.get(namespace, "config")

        if item:
            settings = item.value
        else:
            settings = {
                "language": "ko",
                "theme": "light",
                "notifications": "enabled"
            }

        settings_str = "\n".join([f"  - {k}: {v}" for k, v in settings.items()])
        result = f"{user_name}님의 설정:\n{settings_str}"

        print(f"\n📋 {result}")

        return result

    @tool
    def reset_settings(runtime: ToolRuntime[UserContext]) -> str:
        """설정 초기화"""

        user_id = runtime.context.user_id
        user_name = runtime.context.user_name
        store = runtime.store

        # 기본 설정으로 초기화
        default_settings = {
            "language": "ko",
            "theme": "light",
            "notifications": "enabled"
        }

        namespace = ("settings", user_id)
        store.put(namespace, "config", default_settings)

        print(f"\n🔄 {user_name}님의 설정을 초기화했습니다.")

        return "설정을 기본값으로 초기화했습니다."

    store = InMemoryStore()

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[update_settings, get_settings, reset_settings],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
        store=store,
    )

    # 사용자 1
    user1 = UserContext(user_id="user_001", user_name="김철수")

    print("\n👤 사용자 1 - 설정 조회:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내 설정을 보여줘"}]},
        context=user1,
        config={"configurable": {"thread_id": "user1-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    print("\n\n👤 사용자 1 - 테마 변경:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "테마를 dark로 변경해줘"}]},
        context=user1,
        config={"configurable": {"thread_id": "user1-002"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    # 사용자 2
    user2 = UserContext(user_id="user_002", user_name="이영희")

    print("\n\n👤 사용자 2 - 설정 조회:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내 설정을 보여줘"}]},
        context=user2,
        config={"configurable": {"thread_id": "user2-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    print("\n\n👤 사용자 1 - 최종 설정 확인:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "설정 확인"}]},
        context=user1,
        config={"configurable": {"thread_id": "user1-003"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 6: Context Engineering - Tool Runtime")
    print("\n")

    try:
        # 예제 1: ToolRuntime 기본
        example_1_tool_runtime_basics()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 2: State 접근
        example_2_state_access()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 3: Store 접근
        example_3_store_access()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 4: Config와 Thread
        example_4_config_and_thread()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 5: 사용자 설정
        example_5_user_settings()

        # 마무리
        print("\n" + "=" * 70)
        print("🎉 Part 6 - Tool Runtime 완료!")
        print("=" * 70)
        print("\n💡 배운 내용:")
        print("  ✅ ToolRuntime 파라미터 사용")
        print("  ✅ Tool에서 State 읽기")
        print("  ✅ Store를 통한 장기 메모리")
        print("  ✅ Config와 Thread ID 활용")
        print("  ✅ 사용자별 설정 시스템")
        print("\n📚 다음 단계:")
        print("  ➡️ 06_context_injection.py - Context Injection")
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
# 1. ToolRuntime의 구조:
#    - context: 정적 컨텍스트 (불변)
#    - store: 장기 메모리 (영구 저장)
#    - config: 실행 설정 (thread_id 등)
#    - state: 현재 상태 (읽기 전용)
#
# 2. Store 활용 패턴:
#    - Namespace로 데이터 구조화
#    - put/get으로 CRUD 작업
#    - 사용자별, 스레드별 데이터 분리
#
# 3. 실전 팁:
#    - Store는 대화 간 정보 유지에 활용
#    - State는 현재 대화의 임시 정보
#    - Config는 실행 환경 설정
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "ToolRuntime 파라미터가 인식되지 않음"
# 해결: 정확히 'runtime: ToolRuntime' 형식으로 선언
#
# 문제: "Store에 데이터가 저장되지 않음"
# 해결: store.put() 호출 확인, namespace 정확히 지정
#
# 문제: "Thread ID를 찾을 수 없음"
# 해결: invoke() 시 config에 thread_id 전달 확인
#
# ============================================================================
