"""
[Chapter 13] 영속성 (Persistence) - Checkpointer

📝 설명:
    Checkpointer는 LangGraph에서 그래프의 상태를 저장하고 복원하는
    핵심 메커니즘입니다. 이를 통해 대화 기록 유지, 장애 복구,
    Time Travel 등의 기능을 구현할 수 있습니다.

🎯 학습 목표:
    - Checkpointer 개념 이해
    - Thread와 Checkpoint의 관계
    - InMemorySaver 사용법
    - SqliteSaver 사용법
    - 상태 조회 및 수정 방법

📚 관련 문서:
    - docs/Part4-Production/13-persistence.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/persistence/

💻 실행 방법:
    python -m src.part4_production.13_checkpointer

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langgraph-checkpoint-sqlite>=1.0.0
"""

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# 1. Checkpointer 기본 개념
# =============================================================================

def explain_checkpointer_concept():
    """Checkpointer 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Checkpointer 기본 개념")
    print("=" * 60)

    print("""
Checkpointer란?
- 그래프 실행 중 상태(State)를 저장하는 메커니즘
- 각 슈퍼스텝(super-step)마다 자동 저장
- Thread 단위로 상태 관리

핵심 용어:

1. Thread (스레드)
   - 독립적인 대화/실행 세션
   - thread_id로 구분
   - 각 thread는 여러 checkpoint를 가짐

2. Checkpoint (체크포인트)
   - 특정 시점의 전체 상태 스냅샷
   - checkpoint_id로 식별
   - 불변(immutable) - 수정 불가

3. Super-step (슈퍼스텝)
   - 그래프 실행의 한 단계
   - 병렬 노드들이 모두 완료되는 시점
   - 각 슈퍼스텝 후 checkpoint 생성

Checkpointer 종류:

- InMemorySaver: 메모리 저장 (개발/테스트용)
- SqliteSaver: SQLite 파일 저장 (로컬)
- PostgresSaver: PostgreSQL (프로덕션)
- RedisSaver: Redis (고성능)
""")


# =============================================================================
# 2. InMemorySaver 기본 사용
# =============================================================================

class CounterState(TypedDict):
    """카운터 State"""
    count: int
    history: Annotated[list, operator.add]


def increment(state: CounterState) -> CounterState:
    """카운트 증가"""
    return {
        "count": state["count"] + 1,
        "history": [f"count: {state['count']} -> {state['count'] + 1}"]
    }


def create_counter_graph_with_memory():
    """MemorySaver를 사용하는 카운터 그래프"""
    graph = StateGraph(CounterState)
    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)

    # MemorySaver 생성 및 적용
    memory = MemorySaver()
    return graph.compile(checkpointer=memory), memory


def run_memory_saver_example():
    """MemorySaver 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: InMemorySaver 기본 사용")
    print("=" * 60)

    app, memory = create_counter_graph_with_memory()

    # Thread 설정 - config에 thread_id 지정
    config = {"configurable": {"thread_id": "thread-1"}}

    # 첫 번째 실행
    result1 = app.invoke({"count": 0, "history": []}, config=config)
    print(f"\n🔄 1차 실행: count = {result1['count']}")

    # 두 번째 실행 (같은 thread)
    # 이전 상태에서 이어서 실행됨!
    result2 = app.invoke({"count": result1["count"], "history": result1["history"]}, config=config)
    print(f"🔄 2차 실행: count = {result2['count']}")

    # 세 번째 실행
    result3 = app.invoke({"count": result2["count"], "history": result2["history"]}, config=config)
    print(f"🔄 3차 실행: count = {result3['count']}")

    print(f"\n📜 전체 기록:")
    for h in result3["history"]:
        print(f"   {h}")


# =============================================================================
# 3. Thread를 사용한 세션 분리
# =============================================================================

def run_multi_thread_example():
    """다중 Thread 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: Thread를 사용한 세션 분리")
    print("=" * 60)

    app, memory = create_counter_graph_with_memory()

    # 두 개의 독립적인 Thread
    thread_a = {"configurable": {"thread_id": "user-alice"}}
    thread_b = {"configurable": {"thread_id": "user-bob"}}

    # Alice의 세션
    result_a1 = app.invoke({"count": 0, "history": []}, config=thread_a)
    result_a2 = app.invoke({"count": result_a1["count"], "history": []}, config=thread_a)

    # Bob의 세션
    result_b1 = app.invoke({"count": 100, "history": []}, config=thread_b)

    print(f"\n👤 Alice (thread: user-alice):")
    print(f"   1차 실행: count = {result_a1['count']}")
    print(f"   2차 실행: count = {result_a2['count']}")

    print(f"\n👤 Bob (thread: user-bob):")
    print(f"   1차 실행: count = {result_b1['count']}")

    print(f"\n📌 두 Thread는 완전히 독립적입니다!")


# =============================================================================
# 4. 상태 조회 (get_state)
# =============================================================================

def run_get_state_example():
    """상태 조회 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: get_state로 상태 조회")
    print("=" * 60)

    app, memory = create_counter_graph_with_memory()
    config = {"configurable": {"thread_id": "state-demo"}}

    # 실행
    app.invoke({"count": 0, "history": []}, config=config)
    app.invoke({"count": 1, "history": ["count: 0 -> 1"]}, config=config)

    # 현재 상태 조회
    current_state = app.get_state(config)

    print(f"\n📊 현재 상태:")
    print(f"   values: {current_state.values}")
    print(f"   next: {current_state.next}")  # 다음 실행할 노드
    print(f"   config: {current_state.config}")

    # 메타데이터
    if hasattr(current_state, 'metadata'):
        print(f"   metadata: {current_state.metadata}")


# =============================================================================
# 5. 상태 히스토리 (get_state_history)
# =============================================================================

def run_state_history_example():
    """상태 히스토리 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: get_state_history로 히스토리 조회")
    print("=" * 60)

    app, memory = create_counter_graph_with_memory()
    config = {"configurable": {"thread_id": "history-demo"}}

    # 여러 번 실행
    for i in range(3):
        app.invoke({"count": i, "history": []}, config=config)

    # 히스토리 조회
    print(f"\n📜 상태 히스토리 (최신순):")
    for i, state in enumerate(app.get_state_history(config)):
        values = state.values
        print(f"   {i+1}. count={values.get('count', 'N/A')}")

        # 최근 5개만 출력
        if i >= 4:
            print(f"   ... (더 많은 기록 있음)")
            break


# =============================================================================
# 6. 상태 수정 (update_state)
# =============================================================================

def run_update_state_example():
    """상태 수정 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 5: update_state로 상태 수정")
    print("=" * 60)

    app, memory = create_counter_graph_with_memory()
    config = {"configurable": {"thread_id": "update-demo"}}

    # 초기 실행
    result = app.invoke({"count": 5, "history": []}, config=config)
    print(f"\n🔄 초기 실행: count = {result['count']}")

    # 상태 수정
    app.update_state(
        config,
        {"count": 100, "history": ["관리자가 값을 수정함"]}
    )

    # 수정된 상태 확인
    updated_state = app.get_state(config)
    print(f"✏️  상태 수정 후: count = {updated_state.values['count']}")

    # 수정된 상태에서 이어서 실행
    result2 = app.invoke({"count": updated_state.values['count'], "history": []}, config=config)
    print(f"🔄 수정 후 실행: count = {result2['count']}")


# =============================================================================
# 7. MessagesState와 Checkpointer
# =============================================================================

def create_chat_graph_with_memory():
    """대화 그래프 (Checkpointer 포함)"""

    def respond(state: MessagesState) -> MessagesState:
        """간단한 응답 생성"""
        last_msg = state["messages"][-1].content
        response = f"'{last_msg}'에 대한 응답입니다."
        return {"messages": [AIMessage(content=response)]}

    graph = StateGraph(MessagesState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_chat_with_memory_example():
    """대화 메모리 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 6: MessagesState와 Checkpointer")
    print("=" * 60)

    app = create_chat_graph_with_memory()
    config = {"configurable": {"thread_id": "chat-session-1"}}

    # 첫 번째 대화
    result1 = app.invoke({
        "messages": [HumanMessage(content="안녕하세요!")]
    }, config=config)

    # 두 번째 대화 (이전 대화에 이어서)
    result2 = app.invoke({
        "messages": result1["messages"] + [HumanMessage(content="오늘 날씨 어때요?")]
    }, config=config)

    # 세 번째 대화
    result3 = app.invoke({
        "messages": result2["messages"] + [HumanMessage(content="감사합니다!")]
    }, config=config)

    print(f"\n💬 전체 대화 기록:")
    for msg in result3["messages"]:
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"   {role} {msg.content}")


# =============================================================================
# 8. SQLite Checkpointer (파일 저장)
# =============================================================================

def run_sqlite_saver_example():
    """SQLite Saver 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 7: SqliteSaver (파일 저장)")
    print("=" * 60)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        # SQLite 데이터베이스 연결 (메모리)
        conn = sqlite3.connect(":memory:", check_same_thread=False)

        # SqliteSaver 생성
        with SqliteSaver(conn) as memory:
            graph = StateGraph(CounterState)
            graph.add_node("increment", increment)
            graph.add_edge(START, "increment")
            graph.add_edge("increment", END)

            app = graph.compile(checkpointer=memory)

            config = {"configurable": {"thread_id": "sqlite-demo"}}

            # 실행
            result = app.invoke({"count": 0, "history": []}, config=config)
            print(f"\n✅ SQLite 저장 성공: count = {result['count']}")

            # 상태 조회
            state = app.get_state(config)
            print(f"📊 저장된 상태: {state.values}")

    except ImportError:
        print("\n⚠️  langgraph-checkpoint-sqlite 패키지가 필요합니다.")
        print("   pip install langgraph-checkpoint-sqlite")


# =============================================================================
# 9. Checkpointer 패턴 정리
# =============================================================================

def explain_checkpointer_patterns():
    """Checkpointer 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Checkpointer 패턴 정리")
    print("=" * 60)

    print("""
기본 사용법:

1. Checkpointer 생성
   memory = MemorySaver()  # 또는 SqliteSaver, PostgresSaver

2. 그래프에 적용
   app = graph.compile(checkpointer=memory)

3. Thread ID와 함께 실행
   config = {"configurable": {"thread_id": "unique-id"}}
   result = app.invoke(state, config=config)

주요 API:

- app.get_state(config): 현재 상태 조회
- app.get_state_history(config): 히스토리 조회
- app.update_state(config, values): 상태 수정

Checkpointer 선택 가이드:

┌────────────────┬──────────────────────────────────┐
│ Checkpointer   │ 사용 시나리오                    │
├────────────────┼──────────────────────────────────┤
│ MemorySaver    │ 개발, 테스트, 단일 프로세스      │
│ SqliteSaver    │ 로컬 개발, 소규모 앱             │
│ PostgresSaver  │ 프로덕션, 분산 환경              │
│ RedisSaver     │ 고성능, 실시간 앱                │
└────────────────┴──────────────────────────────────┘

주의사항:
- Thread ID는 고유해야 함
- Checkpoint는 불변 (수정 불가)
- update_state는 새 checkpoint 생성
- 대용량 상태는 성능 고려 필요
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 13] 영속성 (Persistence) - Checkpointer")
    print("=" * 60)

    # 개념 설명
    explain_checkpointer_concept()

    # 예제 실행
    run_memory_saver_example()
    run_multi_thread_example()
    run_get_state_example()
    run_state_history_example()
    run_update_state_example()
    run_chat_with_memory_example()
    run_sqlite_saver_example()

    # 패턴 정리
    explain_checkpointer_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 14a_short_term_memory.py (단기 메모리)")
    print("=" * 60)


if __name__ == "__main__":
    main()
