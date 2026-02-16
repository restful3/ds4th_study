"""
[Chapter 14b] 장기 메모리 (Long-Term Memory)

📝 설명:
    장기 메모리는 세션을 넘어서 지속되는 정보를 저장합니다.
    Memory Store를 사용하여 사용자 프로필, 선호도, 학습된 정보를
    영구적으로 저장하고 검색할 수 있습니다.

🎯 학습 목표:
    - Memory Store 개념 이해
    - 사용자 프로필 저장 및 검색
    - Semantic Search를 통한 메모리 검색
    - 네임스페이스를 통한 메모리 조직화

📚 관련 문서:
    - docs/Part4-Production/14-memory.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory

💻 실행 방법:
    python -m src.part4_production.14b_long_term_memory

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
from typing import TypedDict, Optional
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# 1. Memory Store 개념 설명
# =============================================================================

def explain_memory_store():
    """Memory Store 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Memory Store (장기 메모리)")
    print("=" * 60)

    print("""
Memory Store란?
    세션을 넘어서 지속되는 정보를 저장하는 영구 저장소입니다.
    Checkpointer가 대화 히스토리를 저장한다면,
    Memory Store는 학습된 사실, 선호도, 프로필 등을 저장합니다.

단기 메모리 vs 장기 메모리:

┌─────────────────┬────────────────────────────────────┐
│   단기 메모리    │           장기 메모리              │
├─────────────────┼────────────────────────────────────┤
│ Checkpointer    │ Memory Store                       │
│ 대화 히스토리   │ 사용자 프로필, 선호도              │
│ Thread 범위     │ 모든 Thread에서 접근 가능          │
│ 자동 저장       │ 명시적 저장 필요                   │
│ 메시지 형태     │ Key-Value 형태                     │
└─────────────────┴────────────────────────────────────┘

Memory Store 구조:
    - Namespace: 메모리를 조직화하는 계층 구조
    - Key: 각 메모리 항목의 식별자
    - Value: 저장되는 데이터 (dict 형태)

예시:
    namespace = ("users", "user_123", "preferences")
    key = "language"
    value = {"preferred": "ko", "updated_at": "2024-01-01"}
""")


# =============================================================================
# 2. InMemoryStore 기본 사용법
# =============================================================================

def run_basic_store_example():
    """기본 Memory Store 예제"""
    print("\n" + "=" * 60)
    print("예제 1: InMemoryStore 기본 사용법")
    print("=" * 60)

    # Memory Store 생성
    store = InMemoryStore()

    # 데이터 저장 - put(namespace, key, value)
    store.put(
        namespace=("users", "user_001"),
        key="profile",
        value={
            "name": "홍길동",
            "email": "hong@example.com",
            "created_at": datetime.now().isoformat()
        }
    )

    store.put(
        namespace=("users", "user_001"),
        key="preferences",
        value={
            "language": "ko",
            "theme": "dark",
            "notifications": True
        }
    )

    # 데이터 검색 - get(namespace, key)
    profile = store.get(namespace=("users", "user_001"), key="profile")
    preferences = store.get(namespace=("users", "user_001"), key="preferences")

    print("\n📁 저장된 데이터:")
    print(f"   프로필: {profile.value if profile else 'None'}")
    print(f"   선호도: {preferences.value if preferences else 'None'}")

    # 네임스페이스의 모든 항목 검색 - search(namespace)
    all_items = store.search(namespace=("users", "user_001"))

    print(f"\n📋 user_001의 모든 항목:")
    for item in all_items:
        print(f"   - {item.key}: {item.value}")


# =============================================================================
# 3. 그래프에서 Memory Store 사용
# =============================================================================

class UserState(TypedDict):
    """사용자 State"""
    user_id: str
    message: str
    response: str


def create_personalized_graph(store: InMemoryStore):
    """개인화된 응답을 제공하는 그래프"""

    def load_user_context(state: UserState) -> UserState:
        """사용자 컨텍스트 로드"""
        user_id = state["user_id"]

        # Memory Store에서 사용자 정보 로드
        profile = store.get(
            namespace=("users", user_id),
            key="profile"
        )

        preferences = store.get(
            namespace=("users", user_id),
            key="preferences"
        )

        # 컨텍스트 구성
        context = []
        if profile:
            context.append(f"사용자 이름: {profile.value.get('name', '알 수 없음')}")
        if preferences:
            lang = preferences.value.get('language', 'ko')
            context.append(f"선호 언어: {lang}")

        return {"response": f"[컨텍스트: {', '.join(context)}]"}

    def generate_response(state: UserState) -> UserState:
        """응답 생성"""
        context = state.get("response", "")
        message = state["message"]

        # 실제로는 LLM을 사용하지만, 여기서는 시뮬레이션
        response = f"{context}\n입력: '{message}'에 대한 개인화된 응답입니다."
        return {"response": response}

    def save_interaction(state: UserState) -> UserState:
        """상호작용 기록 저장"""
        user_id = state["user_id"]

        # 상호작용 히스토리 업데이트
        history = store.get(
            namespace=("users", user_id),
            key="interaction_history"
        )

        interactions = history.value.get("interactions", []) if history else []
        interactions.append({
            "message": state["message"],
            "response": state["response"],
            "timestamp": datetime.now().isoformat()
        })

        # 최근 10개만 유지
        store.put(
            namespace=("users", user_id),
            key="interaction_history",
            value={"interactions": interactions[-10:]}
        )

        return {}

    # 그래프 구성
    graph = StateGraph(UserState)

    graph.add_node("load_context", load_user_context)
    graph.add_node("generate", generate_response)
    graph.add_node("save", save_interaction)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "generate")
    graph.add_edge("generate", "save")
    graph.add_edge("save", END)

    return graph.compile()


def run_personalized_graph_example():
    """개인화 그래프 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 개인화된 응답 그래프")
    print("=" * 60)

    # Memory Store 생성 및 초기 데이터 설정
    store = InMemoryStore()

    # 사용자 프로필 저장
    store.put(
        namespace=("users", "user_123"),
        key="profile",
        value={"name": "김철수", "tier": "premium"}
    )
    store.put(
        namespace=("users", "user_123"),
        key="preferences",
        value={"language": "ko", "formal": True}
    )

    # 그래프 생성 및 실행
    app = create_personalized_graph(store)

    result = app.invoke({
        "user_id": "user_123",
        "message": "오늘 날씨 어때?",
        "response": ""
    })

    print(f"\n🎯 개인화된 응답:")
    print(f"   {result['response']}")

    # 저장된 상호작용 확인
    history = store.get(
        namespace=("users", "user_123"),
        key="interaction_history"
    )
    if history:
        print(f"\n📜 상호작용 기록:")
        for interaction in history.value.get("interactions", []):
            print(f"   - {interaction['timestamp'][:19]}: {interaction['message'][:30]}...")


# =============================================================================
# 4. 네임스페이스 활용
# =============================================================================

def run_namespace_example():
    """네임스페이스 활용 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 네임스페이스 조직화")
    print("=" * 60)

    store = InMemoryStore()

    # 계층적 네임스페이스 구조
    # /users/user_001/profile
    # /users/user_001/preferences
    # /users/user_001/history
    # /settings/global
    # /settings/features

    # 사용자별 데이터
    for user_id in ["user_001", "user_002"]:
        store.put(
            namespace=("users", user_id, "profile"),
            key="info",
            value={"name": f"User {user_id[-3:]}", "active": True}
        )
        store.put(
            namespace=("users", user_id, "settings"),
            key="notifications",
            value={"email": True, "push": False}
        )

    # 전역 설정
    store.put(
        namespace=("settings", "global"),
        key="app_config",
        value={"version": "1.0.0", "maintenance": False}
    )

    print("\n📂 네임스페이스 구조:")

    # 특정 사용자의 모든 데이터 검색
    user_001_profile = store.search(namespace=("users", "user_001", "profile"))
    print(f"\n   /users/user_001/profile:")
    for item in user_001_profile:
        print(f"      {item.key}: {item.value}")

    user_001_settings = store.search(namespace=("users", "user_001", "settings"))
    print(f"\n   /users/user_001/settings:")
    for item in user_001_settings:
        print(f"      {item.key}: {item.value}")

    # 전역 설정 검색
    global_settings = store.search(namespace=("settings", "global"))
    print(f"\n   /settings/global:")
    for item in global_settings:
        print(f"      {item.key}: {item.value}")


# =============================================================================
# 5. 메모리 업데이트 및 삭제
# =============================================================================

def run_update_delete_example():
    """메모리 업데이트 및 삭제 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 메모리 업데이트 및 삭제")
    print("=" * 60)

    store = InMemoryStore()

    namespace = ("users", "test_user")

    # 초기 데이터 저장
    store.put(
        namespace=namespace,
        key="counter",
        value={"count": 0, "last_updated": datetime.now().isoformat()}
    )

    print("\n📝 초기 상태:")
    item = store.get(namespace=namespace, key="counter")
    print(f"   count: {item.value['count']}")

    # 업데이트 (같은 키로 put하면 덮어쓰기)
    for i in range(3):
        current = store.get(namespace=namespace, key="counter")
        new_count = current.value["count"] + 1
        store.put(
            namespace=namespace,
            key="counter",
            value={"count": new_count, "last_updated": datetime.now().isoformat()}
        )
        print(f"   업데이트 {i+1}: count = {new_count}")

    # 최종 상태
    final = store.get(namespace=namespace, key="counter")
    print(f"\n📊 최종 상태:")
    print(f"   count: {final.value['count']}")

    # 삭제
    store.delete(namespace=namespace, key="counter")
    deleted = store.get(namespace=namespace, key="counter")
    print(f"\n🗑️  삭제 후: {deleted}")


# =============================================================================
# 6. 학습된 사실 저장 패턴
# =============================================================================

class LearningState(TypedDict):
    """학습 State"""
    user_id: str
    messages: list
    learned_facts: list


def create_learning_agent(store: InMemoryStore):
    """사실을 학습하고 기억하는 에이전트"""

    def extract_facts(state: LearningState) -> LearningState:
        """대화에서 사실 추출 (시뮬레이션)"""
        messages = state.get("messages", [])
        facts = []

        # 간단한 규칙 기반 추출 (실제로는 LLM 사용)
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            if "좋아" in content or "싫어" in content:
                facts.append({
                    "type": "preference",
                    "content": content,
                    "confidence": 0.8
                })
            if "이름은" in content or "나는" in content:
                facts.append({
                    "type": "identity",
                    "content": content,
                    "confidence": 0.9
                })

        return {"learned_facts": facts}

    def store_facts(state: LearningState) -> LearningState:
        """추출된 사실을 Memory Store에 저장"""
        user_id = state["user_id"]
        facts = state.get("learned_facts", [])

        for i, fact in enumerate(facts):
            store.put(
                namespace=("users", user_id, "facts"),
                key=f"fact_{datetime.now().timestamp()}_{i}",
                value={
                    **fact,
                    "learned_at": datetime.now().isoformat()
                }
            )

        return {}

    graph = StateGraph(LearningState)
    graph.add_node("extract", extract_facts)
    graph.add_node("store", store_facts)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "store")
    graph.add_edge("store", END)

    return graph.compile()


def run_learning_example():
    """학습 에이전트 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 사실 학습 에이전트")
    print("=" * 60)

    store = InMemoryStore()
    app = create_learning_agent(store)

    # 대화에서 사실 학습
    conversations = [
        {"content": "나는 커피를 좋아해"},
        {"content": "내 이름은 영희야"},
        {"content": "매운 음식은 싫어"},
    ]

    app.invoke({
        "user_id": "user_abc",
        "messages": conversations,
        "learned_facts": []
    })

    # 저장된 사실 확인
    facts = store.search(namespace=("users", "user_abc", "facts"))

    print("\n📚 학습된 사실들:")
    for fact in facts:
        print(f"   - [{fact.value['type']}] {fact.value['content']}")
        print(f"     (신뢰도: {fact.value['confidence']}, 학습: {fact.value['learned_at'][:19]})")


# =============================================================================
# 7. Memory Store 패턴 정리
# =============================================================================

def explain_memory_patterns():
    """Memory Store 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Memory Store 패턴 정리")
    print("=" * 60)

    print("""
Memory Store 사용 패턴:

1. 사용자 프로필 저장
   namespace = ("users", user_id, "profile")
   store.put(namespace, "info", {"name": "...", "email": "..."})

2. 선호도 관리
   namespace = ("users", user_id, "preferences")
   store.put(namespace, "theme", {"dark_mode": True})

3. 학습된 사실 저장
   namespace = ("users", user_id, "facts")
   store.put(namespace, fact_id, {"type": "...", "content": "..."})

4. 세션 간 컨텍스트 공유
   - 모든 Thread에서 동일한 Memory Store 접근
   - 사용자별 네임스페이스로 격리

구현 팁:

1. 네임스페이스 설계
   - 계층적 구조 사용: ("users", user_id, "category")
   - 일관된 명명 규칙 적용

2. 키 관리
   - 의미 있는 키 사용
   - 충돌 방지를 위해 타임스탬프 활용

3. 값 구조화
   - 메타데이터 포함 (created_at, updated_at)
   - 버전 관리 고려

4. 정리 전략
   - 오래된 데이터 삭제
   - 용량 제한 설정
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 14b] 장기 메모리 (Long-Term Memory)")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_memory_store()

    # 예제 실행
    run_basic_store_example()
    run_personalized_graph_example()
    run_namespace_example()
    run_update_delete_example()
    run_learning_example()

    # 패턴 정리
    explain_memory_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 14c_message_management.py (메시지 관리)")
    print("=" * 60)


if __name__ == "__main__":
    main()
