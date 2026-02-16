"""
[Chapter 17] 타임 트래블 (Time Travel)

📝 설명:
    타임 트래블은 그래프 실행의 과거 상태로 돌아가거나,
    특정 시점에서 분기하여 다른 경로를 탐색하는 기능입니다.

🎯 학습 목표:
    - Replay (재생) 기능 이해
    - Fork (분기) 기능 이해
    - 체크포인트 기반 시간 이동
    - 디버깅 및 분석에 활용

📚 관련 문서:
    - docs/Part4-Production/17-time-travel.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/time-travel/

💻 실행 방법:
    python -m src.part4_production.17_time_travel

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
from typing import TypedDict, Annotated, List
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
import operator


# =============================================================================
# 1. 타임 트래블 개념 설명
# =============================================================================

def explain_time_travel():
    """타임 트래블 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 타임 트래블 (Time Travel)")
    print("=" * 60)

    print("""
타임 트래블이란?
    그래프 실행의 과거 상태로 돌아가거나,
    특정 시점에서 다른 경로로 분기하는 기능입니다.

두 가지 주요 기능:

┌─────────────────┬────────────────────────────────────┐
│    Replay       │    과거 상태를 그대로 재생         │
├─────────────────┼────────────────────────────────────┤
│    Fork         │    과거 상태에서 새로운 분기       │
└─────────────────┴────────────────────────────────────┘

Replay (재생):
    - 과거의 체크포인트를 지정하여 재실행
    - 동일한 결과를 재현
    - 디버깅에 유용

Fork (분기):
    - 과거의 체크포인트에서 새로운 입력으로 분기
    - "만약 다르게 했다면?" 시나리오 탐색
    - A/B 테스트에 유용

사용 사례:

1. 디버깅
   - 오류가 발생한 시점으로 돌아가기
   - 단계별 실행 재현

2. 분석
   - 다른 선택지의 결과 비교
   - 의사결정 과정 검토

3. 사용자 경험
   - "실행 취소" 기능 구현
   - 대안 제시

필요 조건:
    - Checkpointer 필수
    - 체크포인트 ID 또는 인덱스 사용
""")


# =============================================================================
# 2. 기본 그래프 설정
# =============================================================================

class JourneyState(TypedDict):
    """여정 State"""
    location: str
    history: Annotated[List[str], operator.add]
    items: Annotated[List[str], operator.add]
    score: int


def create_adventure_graph():
    """모험 그래프"""

    def start_journey(state: JourneyState) -> JourneyState:
        """여정 시작"""
        return {
            "location": "마을",
            "history": ["여정 시작: 마을"],
            "score": 0
        }

    def visit_forest(state: JourneyState) -> JourneyState:
        """숲 방문"""
        return {
            "location": "숲",
            "history": ["숲에서 탐험"],
            "items": ["나뭇가지"],
            "score": state["score"] + 10
        }

    def visit_cave(state: JourneyState) -> JourneyState:
        """동굴 방문"""
        return {
            "location": "동굴",
            "history": ["동굴에서 보물 발견"],
            "items": ["보물 상자"],
            "score": state["score"] + 50
        }

    def visit_river(state: JourneyState) -> JourneyState:
        """강 방문"""
        return {
            "location": "강",
            "history": ["강에서 휴식"],
            "items": ["물고기"],
            "score": state["score"] + 20
        }

    def end_journey(state: JourneyState) -> JourneyState:
        """여정 종료"""
        return {
            "history": [f"여정 종료: 최종 점수 {state['score']}점"]
        }

    def route_from_forest(state: JourneyState) -> str:
        """숲에서 다음 경로 결정"""
        # 기본적으로 동굴로
        return "cave"

    graph = StateGraph(JourneyState)

    graph.add_node("start", start_journey)
    graph.add_node("forest", visit_forest)
    graph.add_node("cave", visit_cave)
    graph.add_node("river", visit_river)
    graph.add_node("end", end_journey)

    graph.add_edge(START, "start")
    graph.add_edge("start", "forest")
    graph.add_conditional_edges(
        "forest",
        route_from_forest,
        {"cave": "cave", "river": "river"}
    )
    graph.add_edge("cave", "end")
    graph.add_edge("river", "end")
    graph.add_edge("end", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 3. get_state_history로 과거 상태 조회
# =============================================================================

def run_state_history_example():
    """상태 히스토리 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 상태 히스토리 조회")
    print("=" * 60)

    app = create_adventure_graph()
    config = {"configurable": {"thread_id": "adventure_1"}}

    # 그래프 실행
    result = app.invoke({
        "location": "",
        "history": [],
        "items": [],
        "score": 0
    }, config=config)

    print(f"\n🎮 여정 완료!")
    print(f"   최종 위치: {result['location']}")
    print(f"   획득 아이템: {result['items']}")
    print(f"   최종 점수: {result['score']}")

    # 상태 히스토리 조회
    print(f"\n📜 상태 히스토리:")
    history = list(app.get_state_history(config))

    for i, state_snapshot in enumerate(history):
        state = state_snapshot.values
        checkpoint_id = state_snapshot.config.get("configurable", {}).get("checkpoint_id", "N/A")
        print(f"\n   [{i}] Checkpoint: {checkpoint_id[:20]}...")
        print(f"       위치: {state.get('location', 'N/A')}")
        print(f"       점수: {state.get('score', 0)}")
        print(f"       다음: {state_snapshot.next}")


# =============================================================================
# 4. Replay (재생)
# =============================================================================

def run_replay_example():
    """Replay 예제"""
    print("\n" + "=" * 60)
    print("예제 2: Replay (재생)")
    print("=" * 60)

    app = create_adventure_graph()
    config = {"configurable": {"thread_id": "adventure_2"}}

    # 첫 번째 실행
    print("\n🎮 첫 번째 실행:")
    result = app.invoke({
        "location": "",
        "history": [],
        "items": [],
        "score": 0
    }, config=config)
    print(f"   최종 점수: {result['score']}")

    # 히스토리에서 특정 체크포인트 찾기
    history = list(app.get_state_history(config))

    # 숲(forest) 상태의 체크포인트 찾기
    forest_checkpoint = None
    for state_snapshot in history:
        if state_snapshot.values.get("location") == "숲":
            forest_checkpoint = state_snapshot
            break

    if forest_checkpoint:
        checkpoint_config = forest_checkpoint.config
        checkpoint_id = checkpoint_config.get("configurable", {}).get("checkpoint_id")
        print(f"\n🔄 '숲' 상태로 Replay:")
        print(f"   Checkpoint ID: {checkpoint_id[:30]}...")

        # 해당 체크포인트에서 재개
        replay_config = {
            "configurable": {
                "thread_id": "adventure_2",
                "checkpoint_id": checkpoint_id
            }
        }

        # None을 전달하여 해당 시점부터 재실행
        replayed_result = app.invoke(None, config=replay_config)
        print(f"   Replay 결과 점수: {replayed_result['score']}")


# =============================================================================
# 5. Fork (분기)
# =============================================================================

class ForkableState(TypedDict):
    """분기 가능한 State"""
    path: str
    choices: Annotated[List[str], operator.add]
    result: str


def create_forkable_graph():
    """분기 가능한 그래프"""

    def start(state: ForkableState) -> ForkableState:
        """시작"""
        return {"choices": ["시작"]}

    def choose_path(state: ForkableState) -> ForkableState:
        """경로 선택"""
        path = state.get("path", "A")
        return {"choices": [f"경로 {path} 선택"]}

    def path_a(state: ForkableState) -> ForkableState:
        """경로 A"""
        return {"choices": ["A 경로 진행"], "result": "결과 A: 안전한 도착"}

    def path_b(state: ForkableState) -> ForkableState:
        """경로 B"""
        return {"choices": ["B 경로 진행"], "result": "결과 B: 모험적인 도착"}

    def route_path(state: ForkableState) -> str:
        """경로 라우팅"""
        return state.get("path", "A")

    graph = StateGraph(ForkableState)
    graph.add_node("start", start)
    graph.add_node("choose", choose_path)
    graph.add_node("path_a", path_a)
    graph.add_node("path_b", path_b)

    graph.add_edge(START, "start")
    graph.add_edge("start", "choose")
    graph.add_conditional_edges(
        "choose",
        route_path,
        {"A": "path_a", "B": "path_b"}
    )
    graph.add_edge("path_a", END)
    graph.add_edge("path_b", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_fork_example():
    """Fork 예제"""
    print("\n" + "=" * 60)
    print("예제 3: Fork (분기)")
    print("=" * 60)

    app = create_forkable_graph()
    config = {"configurable": {"thread_id": "fork_1"}}

    # 경로 A로 실행
    print("\n🛤️  경로 A로 실행:")
    result_a = app.invoke({
        "path": "A",
        "choices": [],
        "result": ""
    }, config=config)
    print(f"   선택들: {result_a['choices']}")
    print(f"   결과: {result_a['result']}")

    # 히스토리에서 'choose' 후 상태 찾기
    history = list(app.get_state_history(config))

    choose_checkpoint = None
    for state_snapshot in history:
        # choose 노드 실행 직전 상태 찾기
        if state_snapshot.next and "choose" in state_snapshot.next:
            choose_checkpoint = state_snapshot
            break

    if choose_checkpoint:
        print(f"\n🔀 'choose' 시점에서 Fork하여 경로 B로 분기:")

        checkpoint_config = choose_checkpoint.config
        checkpoint_id = checkpoint_config.get("configurable", {}).get("checkpoint_id")

        # 새로운 thread_id로 Fork
        fork_config = {
            "configurable": {
                "thread_id": "fork_1_branch",  # 새 thread
                "checkpoint_id": checkpoint_id
            }
        }

        # 다른 경로로 상태 업데이트
        app.update_state(fork_config, {"path": "B"})

        # Fork된 상태에서 실행
        result_b = app.invoke(None, config=fork_config)
        print(f"   선택들: {result_b['choices']}")
        print(f"   결과: {result_b['result']}")

    print("\n📊 비교:")
    print(f"   경로 A 결과: {result_a['result']}")
    if choose_checkpoint:
        print(f"   경로 B 결과: {result_b['result']}")


# =============================================================================
# 6. 디버깅을 위한 타임 트래블
# =============================================================================

class DebugState(TypedDict):
    """디버그 State"""
    value: int
    operations: Annotated[List[str], operator.add]
    error: str


def create_debug_graph():
    """디버깅용 그래프"""

    def step1(state: DebugState) -> DebugState:
        """Step 1: 값 증가"""
        new_value = state["value"] + 10
        return {
            "value": new_value,
            "operations": [f"step1: {state['value']} -> {new_value}"]
        }

    def step2(state: DebugState) -> DebugState:
        """Step 2: 값 2배"""
        new_value = state["value"] * 2
        return {
            "value": new_value,
            "operations": [f"step2: {state['value']} -> {new_value}"]
        }

    def step3(state: DebugState) -> DebugState:
        """Step 3: 100 빼기"""
        new_value = state["value"] - 100
        if new_value < 0:
            return {
                "value": new_value,
                "operations": [f"step3: {state['value']} -> {new_value} (음수!)"],
                "error": "결과가 음수입니다"
            }
        return {
            "value": new_value,
            "operations": [f"step3: {state['value']} -> {new_value}"]
        }

    graph = StateGraph(DebugState)
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_debug_example():
    """디버깅 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 디버깅을 위한 타임 트래블")
    print("=" * 60)

    app = create_debug_graph()
    config = {"configurable": {"thread_id": "debug_1"}}

    # 실행 (결과가 음수가 될 수 있음)
    print("\n🔍 실행 (초기값: 5):")
    result = app.invoke({
        "value": 5,
        "operations": [],
        "error": ""
    }, config=config)

    print(f"   최종 값: {result['value']}")
    print(f"   에러: {result.get('error', '없음')}")
    print(f"   연산 기록:")
    for op in result["operations"]:
        print(f"      - {op}")

    # 문제가 발생한 경우 과거 상태 분석
    if result.get("error"):
        print(f"\n🕵️  에러 발생! 과거 상태 분석:")

        history = list(app.get_state_history(config))
        for i, snapshot in enumerate(history):
            state = snapshot.values
            print(f"\n   체크포인트 {i}:")
            print(f"      값: {state.get('value', 'N/A')}")
            print(f"      다음 노드: {snapshot.next}")

        # step2 이후 상태 찾기
        step2_checkpoint = None
        for snapshot in history:
            if snapshot.values.get("value") == 30:  # (5+10)*2 = 30
                step2_checkpoint = snapshot
                break

        if step2_checkpoint:
            print(f"\n💡 step2 이후 상태에서 다른 값으로 시도:")

            fork_config = {
                "configurable": {
                    "thread_id": "debug_1_fix",
                    "checkpoint_id": step2_checkpoint.config["configurable"]["checkpoint_id"]
                }
            }

            # 값을 수정하여 재실행
            app.update_state(fork_config, {"value": 150})  # 더 큰 값으로

            fixed_result = app.invoke(None, config=fork_config)
            print(f"      수정된 값으로 실행: {fixed_result['value']}")
            print(f"      에러: {fixed_result.get('error', '없음')}")


# =============================================================================
# 7. 타임 트래블 패턴 정리
# =============================================================================

def explain_time_travel_patterns():
    """타임 트래블 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 타임 트래블 패턴 정리")
    print("=" * 60)

    print("""
타임 트래블 활용 패턴:

1. 히스토리 조회
   history = app.get_state_history(config)
   for snapshot in history:
       print(snapshot.values, snapshot.next)

2. 특정 시점으로 Replay
   replay_config = {
       "configurable": {
           "thread_id": thread_id,
           "checkpoint_id": checkpoint_id
       }
   }
   result = app.invoke(None, config=replay_config)

3. Fork하여 분기
   # 1. 체크포인트 찾기
   history = app.get_state_history(config)
   target = find_checkpoint(history)

   # 2. 새 thread로 Fork
   fork_config = {
       "configurable": {
           "thread_id": "new_branch",
           "checkpoint_id": target.config["configurable"]["checkpoint_id"]
       }
   }

   # 3. 상태 수정 (선택사항)
   app.update_state(fork_config, new_values)

   # 4. 분기 실행
   result = app.invoke(None, config=fork_config)

활용 시나리오:

1. 디버깅
   - 오류 지점 파악
   - 단계별 상태 확인
   - 수정된 입력으로 재실행

2. A/B 테스트
   - 동일 시점에서 다른 선택
   - 결과 비교 분석

3. 사용자 기능
   - "되돌리기" 구현
   - "다른 옵션 보기" 제공

4. 감사/로깅
   - 의사결정 과정 기록
   - 변경 이력 추적

주의사항:
   - 체크포인트 저장소 용량 관리
   - 민감 정보 포함 여부 확인
   - 동시성 고려
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 17] 타임 트래블 (Time Travel)")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_time_travel()

    # 예제 실행
    run_state_history_example()
    run_replay_example()
    run_fork_example()
    run_debug_example()

    # 패턴 정리
    explain_time_travel_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 18_functional_api.py (Functional API)")
    print("=" * 60)


if __name__ == "__main__":
    main()
