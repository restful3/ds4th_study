"""
[Chapter 15] 스트리밍

📝 설명:
    스트리밍은 그래프 실행 결과를 실시간으로 전달하는 기능입니다.
    사용자에게 즉각적인 피드백을 제공하고, 긴 작업의 진행 상황을 보여줄 수 있습니다.

🎯 학습 목표:
    - Stream Modes 이해 (values, updates, messages)
    - LLM 토큰 스트리밍
    - 진행 상황 모니터링

📚 관련 문서:
    - docs/Part4-Production/15-streaming.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/streaming/

💻 실행 방법:
    python -m src.part4_production.15_streaming

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import operator
import time

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage


# =============================================================================
# 1. 스트리밍 모드 개념
# =============================================================================

def explain_streaming_modes():
    """스트리밍 모드 설명"""
    print("\n" + "=" * 60)
    print("📘 스트리밍 모드 (Stream Modes)")
    print("=" * 60)

    print("""
스트리밍 모드 종류:

1. "values" - 전체 State 스트리밍
   - 각 노드 실행 후 전체 State 반환
   - 가장 단순하지만 데이터량 많음

2. "updates" - 업데이트만 스트리밍
   - 노드가 반환한 변경사항만 반환
   - 효율적이지만 전체 상태 재구성 필요

3. "messages" - 메시지 스트리밍
   - MessagesState 전용
   - LLM 토큰 단위 스트리밍 지원

4. "debug" - 디버그 정보 스트리밍
   - 상세한 실행 정보 포함
   - 디버깅에 유용

사용법:
    for event in app.stream(input, stream_mode="values"):
        print(event)
""")


# =============================================================================
# 2. Values 모드 스트리밍
# =============================================================================

class ProgressState(TypedDict):
    """진행 상황 추적 State"""
    step: int
    status: str
    history: Annotated[list, operator.add]


def step_one(state: ProgressState) -> ProgressState:
    time.sleep(0.3)  # 시뮬레이션
    return {"step": 1, "status": "Step 1 완료", "history": ["Step 1"]}


def step_two(state: ProgressState) -> ProgressState:
    time.sleep(0.3)
    return {"step": 2, "status": "Step 2 완료", "history": ["Step 2"]}


def step_three(state: ProgressState) -> ProgressState:
    time.sleep(0.3)
    return {"step": 3, "status": "Step 3 완료", "history": ["Step 3"]}


def create_progress_graph():
    """진행 상황 그래프"""
    graph = StateGraph(ProgressState)

    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_node("step_three", step_three)

    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "step_three")
    graph.add_edge("step_three", END)

    return graph.compile()


def run_values_streaming_example():
    """Values 모드 스트리밍 예제"""
    print("\n" + "=" * 60)
    print("예제 1: Values 모드 스트리밍")
    print("=" * 60)

    app = create_progress_graph()
    initial = {"step": 0, "status": "시작", "history": []}

    print("\n🔄 실시간 진행 상황 (values 모드):")
    for event in app.stream(initial, stream_mode="values"):
        step = event.get("step", 0)
        status = event.get("status", "")
        print(f"   Step {step}: {status}")


# =============================================================================
# 3. Updates 모드 스트리밍
# =============================================================================

def run_updates_streaming_example():
    """Updates 모드 스트리밍 예제"""
    print("\n" + "=" * 60)
    print("예제 2: Updates 모드 스트리밍")
    print("=" * 60)

    app = create_progress_graph()
    initial = {"step": 0, "status": "시작", "history": []}

    print("\n🔄 실시간 업데이트 (updates 모드):")
    for event in app.stream(initial, stream_mode="updates"):
        for node_name, update in event.items():
            print(f"   [{node_name}] 업데이트: {update}")


# =============================================================================
# 4. 노드별 이벤트 처리
# =============================================================================

def run_node_events_example():
    """노드별 이벤트 처리 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 노드별 이벤트 처리")
    print("=" * 60)

    app = create_progress_graph()
    initial = {"step": 0, "status": "시작", "history": []}

    print("\n🔄 노드별 처리:")
    for event in app.stream(initial, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "step_one":
                print(f"   ⚡ Step 1 시작!")
            elif node_name == "step_two":
                print(f"   ⚡ Step 2 진행 중...")
            elif node_name == "step_three":
                print(f"   ⚡ Step 3 완료!")


# =============================================================================
# 5. 메시지 스트리밍
# =============================================================================

def create_chat_graph():
    """채팅 그래프"""

    def respond(state: MessagesState) -> MessagesState:
        last_msg = state["messages"][-1].content
        response = f"'{last_msg}'에 대한 응답입니다. 이것은 긴 응답의 예시입니다."
        return {"messages": [AIMessage(content=response)]}

    graph = StateGraph(MessagesState)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_messages_streaming_example():
    """메시지 스트리밍 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 메시지 스트리밍")
    print("=" * 60)

    app = create_chat_graph()
    initial = {"messages": [HumanMessage(content="안녕하세요!")]}

    print("\n💬 메시지 스트리밍:")
    for event in app.stream(initial, stream_mode="messages"):
        # messages 모드는 (message, metadata) 튜플 반환
        if isinstance(event, tuple):
            msg, metadata = event
            if hasattr(msg, 'content'):
                print(f"   {type(msg).__name__}: {msg.content}")
        else:
            print(f"   Event: {event}")


# =============================================================================
# 6. 스트리밍 패턴 정리
# =============================================================================

def explain_streaming_patterns():
    """스트리밍 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 스트리밍 패턴 정리")
    print("=" * 60)

    print("""
스트리밍 모드 비교:

┌──────────────┬────────────────────────────────────┐
│ 모드         │ 반환 내용                          │
├──────────────┼────────────────────────────────────┤
│ values       │ 전체 State (각 노드 실행 후)       │
│ updates      │ {node_name: update_dict}           │
│ messages     │ (message, metadata) 튜플           │
│ debug        │ 상세 디버그 정보                   │
└──────────────┴────────────────────────────────────┘

사용 패턴:

# 1. 진행 상황 표시
for event in app.stream(input, stream_mode="values"):
    progress = event.get("progress", 0)
    update_progress_bar(progress)

# 2. 실시간 로깅
for event in app.stream(input, stream_mode="updates"):
    for node, data in event.items():
        log(f"Node {node} completed")

# 3. 에러 처리
try:
    for event in app.stream(input):
        process(event)
except Exception as e:
    handle_error(e)

팁:
- UI 업데이트: values 모드
- 효율성: updates 모드
- 채팅: messages 모드
- 디버깅: debug 모드
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 15] 스트리밍")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_streaming_modes()

    # 예제 실행
    run_values_streaming_example()
    run_updates_streaming_example()
    run_node_events_example()
    run_messages_streaming_example()

    # 패턴 정리
    explain_streaming_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 16_interrupts.py (Human-in-the-Loop)")
    print("=" * 60)


if __name__ == "__main__":
    main()
