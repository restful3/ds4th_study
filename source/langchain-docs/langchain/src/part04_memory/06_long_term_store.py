"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory System - Long-term Store
================================================================================

파일명: 06_long_term_store.py
난이도: ⭐⭐⭐⭐☆ (중상급)
예상 시간: 30분

📚 학습 목표:
  - Store 개념과 Checkpointer의 차이 이해
  - InMemoryStore 기본 사용법
  - Namespace와 Key 구조 설계
  - Tool에서 Store 접근하기
  - Search와 필터링 활용

📖 공식 문서:
  • Long-term Memory: /official/29-long-term-memory.md

📄 교안 문서:
  • Part 4 메모리: /docs/part04_memory.md (Section 6)

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 06_long_term_store.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)

# ============================================================================
# 예제 1: Store 기본 개념
# ============================================================================

def example_1_store_basics():
    """Store의 기본 개념과 사용법"""
    print("=" * 70)
    print("📌 예제 1: Store 기본 개념")
    print("=" * 70)
    print()

    print("""
🎯 Checkpointer vs Store:

┌─────────────┬────────────────┬──────────────────┐
│ 특성        │ Checkpointer   │ Store            │
├─────────────┼────────────────┼──────────────────┤
│ 범위        │ 단일 Thread    │ 여러 Thread      │
│ 수명        │ 세션 동안      │ 영구적           │
│ 데이터      │ 대화 이력      │ 사용자 정보      │
│ 예시        │ 현재 대화      │ 사용자 프로필    │
│ 용도        │ Short-term     │ Long-term        │
└─────────────┴────────────────┴──────────────────┘

💡 Store 사용 사례:
   - 사용자 프로필 및 선호도
   - 학습 이력 및 진행 상황
   - 장기간 유지되는 설정
   - 여러 대화에서 공유되는 정보

📦 Store 구조:
   store.put(namespace, key, value)
   store.get(namespace, key)

   - namespace: 데이터를 그룹화 (폴더 같은 개념)
   - key: 데이터를 식별하는 고유 키
   - value: 저장할 데이터 (dict)
    """)

    # Store 생성
    store = InMemoryStore()

    print("\n🔹 Store 기본 사용:")

    # 데이터 저장
    print("\n1️⃣  데이터 저장 (put):")
    store.put(
        namespace=("users",),
        key="user-001",
        value={
            "name": "김철수",
            "email": "kim@example.com",
            "language": "ko",
            "created_at": "2024-01-01"
        }
    )
    print("   ✅ 사용자 정보 저장 완료")

    # 데이터 조회
    print("\n2️⃣  데이터 조회 (get):")
    user_info = store.get(namespace=("users",), key="user-001")
    if user_info:
        print(f"   📄 사용자: {user_info.value}")
        print(f"   🆔 네임스페이스: {user_info.namespace}")
        print(f"   🔑 키: {user_info.key}")

    # 데이터 업데이트
    print("\n3️⃣  데이터 업데이트:")
    store.put(
        namespace=("users",),
        key="user-001",
        value={
            "name": "김철수",
            "email": "kim@example.com",
            "language": "ko",
            "created_at": "2024-01-01",
            "last_login": "2024-02-06"  # 새 필드 추가
        }
    )
    print("   ✅ 사용자 정보 업데이트 완료")

    # 여러 사용자 저장
    print("\n4️⃣  여러 데이터 저장:")
    users = [
        ("user-002", {"name": "이영희", "language": "ko"}),
        ("user-003", {"name": "박민수", "language": "ko"}),
        ("user-004", {"name": "John Doe", "language": "en"}),
    ]

    for user_id, user_data in users:
        store.put(namespace=("users",), key=user_id, value=user_data)

    print(f"   ✅ {len(users)}명의 사용자 저장 완료")


# ============================================================================
# 예제 2: Namespace 구조 설계
# ============================================================================

def example_2_namespace_design():
    """효과적인 Namespace 구조 설계"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Namespace 구조 설계")
    print("=" * 70)
    print("\n💡 Namespace는 데이터를 계층적으로 구성합니다.\n")

    store = InMemoryStore()

    print("""
📂 Namespace 설계 패턴:

1️⃣  평면 구조 (Simple):
   ("users",) → user-001
   ("settings",) → theme
   ("products",) → product-123

2️⃣  계층 구조 (Hierarchical):
   ("users", "user-001") → profile
   ("users", "user-001", "preferences") → settings
   ("users", "user-001", "history") → activity

3️⃣  애플리케이션 기반 (Context-based):
   (user_id, "chat") → chat history
   (user_id, "email") → email history
   (user_id, "support") → support tickets

4️⃣  조직 구조 (Organization):
   ("org", org_id, "team", team_id) → team data
   ("org", org_id, "team", team_id, "user", user_id) → user data
    """)

    print("\n🔹 실제 예제:")

    # 1. 사용자 기본 정보
    store.put(
        namespace=("users",),
        key="user-001",
        value={"name": "김철수", "email": "kim@example.com"}
    )
    print("✅ 사용자 기본 정보: ('users',) / 'user-001'")

    # 2. 사용자 선호도
    store.put(
        namespace=("users", "user-001", "preferences"),
        key="ui",
        value={"theme": "dark", "language": "ko"}
    )
    print("✅ 사용자 선호도: ('users', 'user-001', 'preferences') / 'ui'")

    # 3. 사용자 활동 이력
    store.put(
        namespace=("users", "user-001", "history"),
        key="login",
        value={"last_login": "2024-02-06", "login_count": 42}
    )
    print("✅ 활동 이력: ('users', 'user-001', 'history') / 'login'")

    # 4. 앱별 데이터
    store.put(
        namespace=("user-001", "chat"),
        key="summary",
        value={"total_messages": 150, "avg_length": 45}
    )
    print("✅ 채팅 데이터: ('user-001', 'chat') / 'summary'")

    # 5. 조직 데이터
    store.put(
        namespace=("org", "company-abc", "team", "engineering"),
        key="members",
        value={"count": 10, "lead": "user-001"}
    )
    print("✅ 조직 데이터: ('org', 'company-abc', 'team', 'engineering') / 'members'")

    print("\n💡 네이밍 팁:")
    print("   - 일관된 규칙 사용")
    print("   - 명확한 이름 선택")
    print("   - 계층은 3-4단계까지 권장")
    print("   - 약어보다 전체 단어 사용")


# ============================================================================
# 예제 3: Tool에서 Store 사용
# ============================================================================

def example_3_store_in_tools():
    """Tool에서 Store에 접근하고 수정하기"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Tool에서 Store 사용")
    print("=" * 70)
    print("\n💡 ToolRuntime을 통해 Store에 접근합니다.\n")

    # Context 정의
    @dataclass
    class UserContext:
        user_id: str

    # Store 생성 및 초기 데이터
    store = InMemoryStore()

    # 샘플 사용자 데이터
    store.put(
        namespace=("users",),
        key="user-123",
        value={
            "name": "박지민",
            "email": "park@example.com",
            "language": "ko"
        }
    )

    # Tool 정의
    @tool
    def get_user_profile(runtime: ToolRuntime[UserContext]) -> str:
        """사용자 프로필을 조회합니다."""
        store = runtime.store
        user_id = runtime.context.user_id

        user_info = store.get(namespace=("users",), key=user_id)

        if user_info:
            data = user_info.value
            return f"이름: {data.get('name')}, 이메일: {data.get('email')}"
        else:
            return "사용자 정보를 찾을 수 없습니다."

    @tool
    def update_user_name(new_name: str, runtime: ToolRuntime[UserContext]) -> str:
        """사용자 이름을 변경합니다."""
        store = runtime.store
        user_id = runtime.context.user_id

        # 기존 정보 가져오기
        user_info = store.get(namespace=("users",), key=user_id)

        if user_info:
            # 업데이트
            data = user_info.value
            data["name"] = new_name

            store.put(namespace=("users",), key=user_id, value=data)
            return f"이름이 '{new_name}'로 변경되었습니다."
        else:
            return "사용자 정보를 찾을 수 없습니다."

    @tool
    def save_preference(
        key: str,
        value: str,
        runtime: ToolRuntime[UserContext]
    ) -> str:
        """사용자 선호도를 저장합니다."""
        store = runtime.store
        user_id = runtime.context.user_id

        # 선호도 가져오기 또는 생성
        prefs = store.get(
            namespace=("users", user_id, "preferences"),
            key="settings"
        )

        if prefs:
            data = prefs.value
        else:
            data = {}

        data[key] = value

        store.put(
            namespace=("users", user_id, "preferences"),
            key="settings",
            value=data
        )

        return f"선호도 저장 완료: {key} = {value}"

    # Agent 생성
    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_user_profile, update_user_name, save_preference],
        store=store,  # Store 전달
        checkpointer=checkpointer,
        context_schema=UserContext,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "store-tools-test"}}
    context = UserContext(user_id="user-123")

    # 테스트
    print("🔹 Tool을 통한 Store 접근:\n")

    # 1. 프로필 조회
    print("1️⃣  프로필 조회:")
    print("👤 사용자: 내 프로필을 보여줘.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "내 프로필을 보여줘."}]},
        config=config,
        context=context
    )
    print(f"🤖 AI: {result['messages'][-1].content}\n")

    # 2. 이름 변경
    print("2️⃣  이름 변경:")
    print("👤 사용자: 내 이름을 '박예은'으로 바꿔줘.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "내 이름을 '박예은'으로 바꿔줘."}]},
        config=config,
        context=context
    )
    print(f"🤖 AI: {result['messages'][-1].content}\n")

    # 3. 선호도 저장
    print("3️⃣  선호도 저장:")
    print("👤 사용자: 테마를 다크모드로 설정해줘.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "테마를 다크모드로 설정해줘."}]},
        config=config,
        context=context
    )
    print(f"🤖 AI: {result['messages'][-1].content}\n")

    # Store 확인
    print("=" * 50)
    print("📊 Store 내용 확인:")
    print("=" * 50)

    user = store.get(namespace=("users",), key="user-123")
    print(f"\n사용자 정보: {user.value}")

    prefs = store.get(
        namespace=("users", "user-123", "preferences"),
        key="settings"
    )
    if prefs:
        print(f"선호도: {prefs.value}")


# ============================================================================
# 예제 4: Search와 필터링
# ============================================================================

def example_4_search_and_filter():
    """Store에서 데이터 검색 및 필터링"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Search와 필터링")
    print("=" * 70)
    print("\n💡 Namespace 내에서 데이터를 검색할 수 있습니다.\n")

    store = InMemoryStore()

    # 샘플 데이터 생성
    print("📝 샘플 데이터 생성 중...")

    users = [
        ("user-001", {"name": "김철수", "language": "ko", "tier": "free"}),
        ("user-002", {"name": "이영희", "language": "ko", "tier": "premium"}),
        ("user-003", {"name": "박민수", "language": "ko", "tier": "free"}),
        ("user-004", {"name": "John Doe", "language": "en", "tier": "premium"}),
        ("user-005", {"name": "Jane Smith", "language": "en", "tier": "free"}),
    ]

    for user_id, user_data in users:
        store.put(namespace=("users",), key=user_id, value=user_data)

    print(f"✅ {len(users)}명의 사용자 생성 완료\n")

    # 1. 전체 검색
    print("=" * 50)
    print("1️⃣  전체 사용자 검색:")
    print("=" * 50)

    all_users = store.search(namespace=("users",))
    for item in all_users:
        print(f"   - {item.key}: {item.value['name']}")

    # 2. 필터링 (tier=premium)
    print("\n" + "=" * 50)
    print("2️⃣  프리미엄 사용자만 검색:")
    print("=" * 50)

    premium_users = store.search(
        namespace=("users",),
        filter={"tier": "premium"}
    )
    for item in premium_users:
        print(f"   - {item.key}: {item.value['name']} (tier: {item.value['tier']})")

    # 3. 필터링 (language=ko)
    print("\n" + "=" * 50)
    print("3️⃣  한국어 사용자만 검색:")
    print("=" * 50)

    ko_users = store.search(
        namespace=("users",),
        filter={"language": "ko"}
    )
    for item in ko_users:
        print(f"   - {item.key}: {item.value['name']}")

    # 4. 복합 필터링
    print("\n" + "=" * 50)
    print("4️⃣  한국어 + 프리미엄 사용자:")
    print("=" * 50)

    ko_premium = store.search(
        namespace=("users",),
        filter={"language": "ko", "tier": "premium"}
    )
    for item in ko_premium:
        print(f"   - {item.key}: {item.value['name']}")

    # 5. Limit 사용
    print("\n" + "=" * 50)
    print("5️⃣  최대 2명만 조회:")
    print("=" * 50)

    limited = store.search(
        namespace=("users",),
        limit=2
    )
    for item in limited:
        print(f"   - {item.key}: {item.value['name']}")

    print("\n💡 Search 활용:")
    print("   - 관리자 대시보드")
    print("   - 사용자 통계")
    print("   - 타겟팅 마케팅")
    print("   - 데이터 분석")


# ============================================================================
# 예제 5: 실전 패턴 - User Profile System
# ============================================================================

def example_5_user_profile_system():
    """실전 사용자 프로필 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 User Profile System")
    print("=" * 70)
    print("\n💡 Checkpointer + Store를 함께 사용하는 완전한 예제\n")

    @dataclass
    class UserContext:
        user_id: str

    # Store와 Checkpointer 모두 사용
    store = InMemoryStore()
    checkpointer = InMemorySaver()

    # 초기 사용자 프로필 생성
    store.put(
        namespace=("users",),
        key="user-real",
        value={
            "name": "홍길동",
            "email": "hong@example.com",
            "signup_date": "2024-01-15",
            "tier": "free"
        }
    )

    # Tools
    @tool
    def view_profile(runtime: ToolRuntime[UserContext]) -> str:
        """사용자 프로필 전체를 조회합니다."""
        user_id = runtime.context.user_id
        user = runtime.store.get(namespace=("users",), key=user_id)

        if not user:
            return "사용자를 찾을 수 없습니다."

        profile = user.value
        return f"""
사용자 프로필:
- 이름: {profile.get('name')}
- 이메일: {profile.get('email')}
- 가입일: {profile.get('signup_date')}
- 등급: {profile.get('tier')}
"""

    @tool
    def update_profile_field(
        field: str,
        value: str,
        runtime: ToolRuntime[UserContext]
    ) -> str:
        """프로필의 특정 필드를 업데이트합니다."""
        user_id = runtime.context.user_id
        user = runtime.store.get(namespace=("users",), key=user_id)

        if not user:
            return "사용자를 찾을 수 없습니다."

        profile = user.value
        old_value = profile.get(field, "없음")
        profile[field] = value

        runtime.store.put(
            namespace=("users",),
            key=user_id,
            value=profile
        )

        return f"'{field}' 업데이트 완료: {old_value} → {value}"

    @tool
    def get_conversation_stats(runtime: ToolRuntime[UserContext]) -> str:
        """현재 대화 통계를 조회합니다."""
        messages = runtime.state["messages"]

        user_msg_count = sum(1 for m in messages if m.type == "human")
        ai_msg_count = sum(1 for m in messages if m.type == "ai")

        return f"""
대화 통계 (현재 세션):
- 사용자 메시지: {user_msg_count}개
- AI 메시지: {ai_msg_count}개
- 총 메시지: {len(messages)}개
"""

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[view_profile, update_profile_field, get_conversation_stats],
        store=store,           # Long-term (Store)
        checkpointer=checkpointer,  # Short-term (Checkpointer)
        context_schema=UserContext,
        system_prompt="당신은 사용자 프로필 관리를 돕는 어시스턴트입니다."
    )

    config: RunnableConfig = {"configurable": {"thread_id": "profile-session"}}
    context = UserContext(user_id="user-real")

    print("🎬 사용자 프로필 시스템 데모:\n")

    # 시나리오
    interactions = [
        "내 프로필을 보여줘.",
        "이메일을 'newemail@example.com'으로 변경해줘.",
        "현재 대화 통계를 알려줘.",
        "프로필을 다시 확인해줘.",
    ]

    for i, msg in enumerate(interactions, 1):
        print(f"{i}. 👤 사용자: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config,
            context=context
        )

        print(f"   🤖 AI: {result['messages'][-1].content}\n")

    print("=" * 70)
    print("📊 시스템 구조:")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────┐
│ Agent                                       │
│  ├─ Store (Long-term Memory)               │
│  │   └─ 사용자 프로필 (여러 세션에서 공유) │
│  │                                          │
│  └─ Checkpointer (Short-term Memory)       │
│      └─ 대화 이력 (현재 세션만)           │
└─────────────────────────────────────────────┘

✅ Store: 프로필 정보 영구 저장
✅ Checkpointer: 대화 이력 세션 저장
✅ Tools: 두 메모리 모두 접근 가능
    """)


# ============================================================================
# 보너스: Store 고급 패턴
# ============================================================================

def bonus_advanced_patterns():
    """Store 고급 활용 패턴"""
    print("\n" + "=" * 70)
    print("🎁 보너스: Store 고급 패턴")
    print("=" * 70)
    print()

    print("""
🎯 고급 활용 패턴:

1️⃣  캐시 레이어:
   - L1: In-memory dict (초고속)
   - L2: InMemoryStore (빠름)
   - L3: Database Store (영구)

2️⃣  버저닝:
   namespace=("users", user_id, "v2")
   - 데이터 스키마 버전 관리
   - 마이그레이션 지원

3️⃣  Time-to-Live (TTL):
   value = {
       "data": {...},
       "expires_at": "2024-12-31"
   }
   - 주기적으로 만료된 데이터 삭제

4️⃣  복제 및 백업:
   - Store → Database 주기적 동기화
   - 재해 복구 계획

5️⃣  액세스 제어:
   - Namespace별 권한 관리
   - 사용자/역할 기반 접근 제어

6️⃣  압축 및 최적화:
   - 큰 데이터는 압축하여 저장
   - 자주 접근하지 않는 데이터는 아카이브

7️⃣  이벤트 소싱:
   - 변경 이력을 이벤트로 저장
   - 감사 추적 가능

8️⃣  분산 Store:
   - 여러 인스턴스 간 Store 공유
   - Redis 등 외부 저장소 사용

💡 Production 체크리스트:
   □ 데이터 백업 전략
   □ 액세스 로깅
   □ 성능 모니터링
   □ 에러 처리
   □ 데이터 검증
   □ 보안 (암호화, 권한)
   □ 스케일링 계획
   □ 비용 최적화
    """)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 4: Memory System - Long-term Store")
    print("\n")

    # 예제 1: 기본 개념
    example_1_store_basics()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 2: Namespace 설계
    example_2_namespace_design()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 3: Tool에서 Store
    example_3_store_in_tools()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 4: 검색과 필터링
    example_4_search_and_filter()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 5: 실전 시스템
    example_5_user_profile_system()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 보너스: 고급 패턴
    print("\n" + "=" * 70)
    choice = input("🎁 보너스 예제를 보시겠습니까? (y/n): ").strip().lower()
    if choice == 'y':
        bonus_advanced_patterns()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 4 전체 예제를 완료했습니다!")
    print("=" * 70)
    print("\n📚 Part 4 전체 복습:")
    print("\n  1️⃣  Basic Memory (01):")
    print("     • InMemorySaver")
    print("     • Thread 관리")
    print("\n  2️⃣  PostgreSQL (02):")
    print("     • PostgresSaver")
    print("     • Production 메모리")
    print("\n  3️⃣  Message Trim (03):")
    print("     • before_model / after_model")
    print("     • Trim vs Delete")
    print("\n  4️⃣  Summarization (04):")
    print("     • 커스텀 요약")
    print("     • SummarizationMiddleware")
    print("\n  5️⃣  Custom State (05):")
    print("     • AgentState 확장")
    print("     • state_schema")
    print("\n  6️⃣  Long-term Store (06):")
    print("     • InMemoryStore")
    print("     • Namespace 설계")
    print("     • Tool에서 Store 접근")
    print("\n💡 핵심 개념:")
    print("  • Short-term: Checkpointer (Thread 단위)")
    print("  • Long-term: Store (영구 저장)")
    print("  • Trim: 토큰 절약")
    print("  • Summary: 정보 보존")
    print("  • Custom State: 유연한 데이터 관리")
    print("\n🎯 다음 단계:")
    print("  → Part 5: Middleware 심화")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. Store 선택:
#    - InMemoryStore: 개발/테스트
#    - DB Store: Production (PostgreSQL, MongoDB 등)
#    - 하이브리드: 캐시 + DB
#
# 2. Namespace 네이밍:
#    - 일관성 유지
#    - 명확한 의미
#    - 문서화
#
# 3. 데이터 크기:
#    - 작은 데이터: Store에 직접 저장
#    - 큰 데이터: 외부 저장소 사용, Store에는 참조만
#
# 4. 동기화:
#    - Store ↔ 외부 DB 동기화
#    - 실시간 vs 배치 동기화
#
# 5. 모니터링:
#    - Store 크기
#    - 접근 빈도
#    - 성능 메트릭
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "Store가 너무 커짐"
# 해결: 주기적 정리, TTL 구현, 아카이빙
#
# 문제: "데이터가 사라짐"
# 해결: InMemoryStore는 프로세스 종료 시 손실
#       → DB Store 사용
#
# 문제: "성능 저하"
# 해결: 캐싱, 인덱싱, 적절한 Namespace 설계
#
# 문제: "Namespace 충돌"
# 해결: 명확한 네이밍 규칙, 문서화
#
# ============================================================================
