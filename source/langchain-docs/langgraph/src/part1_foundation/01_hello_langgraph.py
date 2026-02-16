"""
[Chapter 1] LangGraph 소개 - Hello LangGraph

📝 설명:
    LangGraph의 기본 개념을 이해하고 첫 번째 그래프를 실행합니다.
    간단한 예제부터 LLM을 활용한 예제까지 단계별로 구현합니다.

🎯 학습 목표:
    - LangGraph의 기본 구조 이해 (StateGraph, Node, Edge)
    - 상태(State) 정의 방법 학습
    - 그래프 컴파일 및 실행 방법 습득
    - LLM과 연동하는 방법 이해

📚 관련 문서:
    - docs/Part1-Foundation/01-introduction.md
    - 공식 문서: https://docs.langchain.com/oss/python/langgraph/overview

💻 실행 방법:
    python -m src.part1_foundation.01_hello_langgraph

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
    - python-dotenv>=1.0.0
"""

import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 기본 예제: 가장 간단한 그래프
# ============================================================

class SimpleState(TypedDict):
    """
    가장 간단한 상태 정의.

    Attributes:
        message: 처리할 메시지
    """
    message: str


def greeting_node(state: SimpleState) -> SimpleState:
    """
    인사 메시지를 생성하는 노드.

    Args:
        state: 현재 상태 (message 포함)

    Returns:
        업데이트된 상태 (인사말이 추가된 message)
    """
    original_message = state["message"]
    greeting = f"안녕하세요! 입력하신 메시지: '{original_message}'"
    return {"message": greeting}


def create_simple_graph() -> StateGraph:
    """
    가장 간단한 그래프를 생성합니다.

    Returns:
        컴파일된 그래프
    """
    # StateGraph 초기화 - State 타입을 전달
    graph = StateGraph(SimpleState)

    # 노드 추가 - (노드 이름, 노드 함수)
    graph.add_node("greeting", greeting_node)

    # 엣지 추가 - 노드 간 연결
    graph.add_edge(START, "greeting")  # START -> greeting
    graph.add_edge("greeting", END)     # greeting -> END

    # 그래프 컴파일 - 실행 가능한 형태로 변환
    return graph.compile()


def run_simple_example():
    """기본 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 가장 간단한 LangGraph")
    print("=" * 60)

    # 그래프 생성
    app = create_simple_graph()

    # 그래프 실행
    initial_state = {"message": "LangGraph 시작!"}
    result = app.invoke(initial_state)

    print(f"\n입력 상태: {initial_state}")
    print(f"출력 상태: {result}")


# ============================================================
# 2. 다중 노드 예제: 여러 노드가 연결된 그래프
# ============================================================

class MultiNodeState(TypedDict):
    """
    다중 노드 그래프를 위한 상태.

    Attributes:
        text: 처리할 텍스트
        step_count: 거친 단계 수
    """
    text: str
    step_count: int


def step_one(state: MultiNodeState) -> MultiNodeState:
    """첫 번째 처리 단계: 대문자로 변환"""
    return {
        "text": state["text"].upper(),
        "step_count": state["step_count"] + 1
    }


def step_two(state: MultiNodeState) -> MultiNodeState:
    """두 번째 처리 단계: 느낌표 추가"""
    return {
        "text": state["text"] + "!!!",
        "step_count": state["step_count"] + 1
    }


def step_three(state: MultiNodeState) -> MultiNodeState:
    """세 번째 처리 단계: 완료 메시지 추가"""
    return {
        "text": f"[완료] {state['text']}",
        "step_count": state["step_count"] + 1
    }


def create_multi_node_graph() -> StateGraph:
    """
    다중 노드 그래프를 생성합니다.

    그래프 구조:
        START -> step_one -> step_two -> step_three -> END

    Returns:
        컴파일된 그래프
    """
    graph = StateGraph(MultiNodeState)

    # 노드 추가
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)
    graph.add_node("step_three", step_three)

    # 엣지 추가 - 순차적 연결
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", "step_three")
    graph.add_edge("step_three", END)

    return graph.compile()


def run_multi_node_example():
    """다중 노드 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: 다중 노드 그래프")
    print("=" * 60)

    app = create_multi_node_graph()

    initial_state = {"text": "hello langgraph", "step_count": 0}
    result = app.invoke(initial_state)

    print(f"\n입력: {initial_state}")
    print(f"출력: {result}")
    print(f"\n처리 과정:")
    print(f"  1. 원본: 'hello langgraph'")
    print(f"  2. 대문자: 'HELLO LANGGRAPH'")
    print(f"  3. 느낌표: 'HELLO LANGGRAPH!!!'")
    print(f"  4. 완료: '[완료] HELLO LANGGRAPH!!!'")


# ============================================================
# 3. LLM 연동 예제: Claude와 함께 사용
# ============================================================

class LLMState(TypedDict):
    """
    LLM 연동을 위한 상태.

    Attributes:
        question: 사용자 질문
        answer: LLM 응답
    """
    question: str
    answer: str


def create_llm_graph():
    """
    LLM을 사용하는 그래프를 생성합니다.

    환경 변수 ANTHROPIC_API_KEY가 필요합니다.

    Returns:
        컴파일된 그래프 또는 None (API 키가 없는 경우)
    """
    # API 키 확인
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 추가하세요.")
        return None

    # LangChain의 ChatAnthropic 임포트
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        print("⚠️  langchain-anthropic 패키지가 필요합니다.")
        print("   pip install langchain-anthropic")
        return None

    # LLM 초기화
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    def ask_llm(state: LLMState) -> LLMState:
        """LLM에게 질문하고 답변을 받습니다"""
        response = llm.invoke(state["question"])
        return {"answer": response.content}

    # 그래프 구성
    graph = StateGraph(LLMState)
    graph.add_node("llm", ask_llm)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", END)

    return graph.compile()


def run_llm_example():
    """LLM 연동 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: LLM 연동 그래프")
    print("=" * 60)

    app = create_llm_graph()

    if app is None:
        print("\n⚠️  LLM 그래프를 생성할 수 없습니다.")
        return

    # 질문 실행
    question = "LangGraph를 한 문장으로 설명해주세요."
    result = app.invoke({"question": question, "answer": ""})

    print(f"\n질문: {question}")
    print(f"\n답변: {result['answer']}")


# ============================================================
# 4. 그래프 시각화
# ============================================================

def visualize_graph():
    """그래프 구조를 시각화합니다"""
    print("\n" + "=" * 60)
    print("그래프 시각화")
    print("=" * 60)

    # 다중 노드 그래프 생성
    app = create_multi_node_graph()

    # ASCII 아트로 시각화 (항상 가능)
    print("\n[ASCII 시각화]")
    print(app.get_graph().draw_ascii())

    # Mermaid 다이어그램 생성 (텍스트)
    print("\n[Mermaid 다이어그램]")
    print(app.get_graph().draw_mermaid())

    # PNG 이미지 생성 (graphviz 필요)
    try:
        from IPython.display import Image, display
        img = app.get_graph().draw_mermaid_png()
        print("\n[PNG 이미지 생성 성공]")
        print("Jupyter 환경에서 display(Image(...))로 표시할 수 있습니다.")
    except Exception as e:
        print(f"\n[PNG 생성 불가] {e}")
        print("graphviz 설치 필요: pip install graphviz")


# ============================================================
# 5. 메인 실행 함수
# ============================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 1] LangGraph 소개 - Hello LangGraph")
    print("=" * 60)

    # 환경 변수 로드
    load_dotenv()

    # 예제 1: 가장 간단한 그래프
    run_simple_example()

    # 예제 2: 다중 노드 그래프
    run_multi_node_example()

    # 예제 3: LLM 연동 (API 키가 있는 경우만)
    run_llm_example()

    # 그래프 시각화
    visualize_graph()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
