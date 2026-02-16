"""
[Chapter 5] 워크플로우 패턴 개요 - Prompt Chaining

📝 설명:
    Prompt Chaining은 여러 LLM 호출을 순차적으로 연결하는 가장 기본적인
    워크플로우 패턴입니다. 각 단계의 출력이 다음 단계의 입력이 됩니다.

🎯 학습 목표:
    - Workflow와 Agent의 차이점 이해
    - Prompt Chaining 패턴 구현
    - 순차적 LLM 호출 체인 구성
    - Gate(검증) 단계 추가 방법

📚 관련 문서:
    - docs/Part2-Workflows/05-workflow-patterns.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#prompt-chaining

💻 실행 방법:
    python -m src.part2_workflows.05_prompt_chaining

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
"""

import os
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END


# =============================================================================
# 1. Workflow vs Agent 개념 설명
# =============================================================================

def explain_workflow_vs_agent():
    """Workflow와 Agent의 차이점 설명"""
    print("\n" + "=" * 60)
    print("📘 Workflow vs Agent")
    print("=" * 60)

    print("""
┌─────────────────┬────────────────────────────────────────┐
│    Workflow     │                Agent                   │
├─────────────────┼────────────────────────────────────────┤
│ 정적 경로       │ 동적 경로 (LLM이 다음 단계 결정)       │
│ 예측 가능       │ 유연하지만 예측 어려움                 │
│ 단순한 제어     │ 복잡한 제어 흐름                       │
│ 디버깅 쉬움     │ 디버깅 어려움                          │
│ 제한된 유연성   │ 높은 유연성                            │
└─────────────────┴────────────────────────────────────────┘

Workflow 패턴들:
1. Prompt Chaining - 순차적 LLM 호출
2. Routing - 조건에 따른 분기
3. Parallelization - 병렬 실행
4. Orchestrator-Worker - 작업 분배 및 수집
5. Evaluator-Optimizer - 결과 평가 및 개선

Agent 패턴들:
1. ReAct Agent - 추론-행동 루프
2. Multi-Agent - 여러 Agent 협업
""")


# =============================================================================
# 2. 기본 Prompt Chaining (LLM 없이)
# =============================================================================

class TextProcessingState(TypedDict):
    """텍스트 처리를 위한 State"""
    original_text: str
    step1_result: str
    step2_result: str
    final_result: str


def step1_clean_text(state: TextProcessingState) -> TextProcessingState:
    """Step 1: 텍스트 정리"""
    text = state["original_text"]
    cleaned = text.strip().lower()
    return {"step1_result": cleaned}


def step2_transform_text(state: TextProcessingState) -> TextProcessingState:
    """Step 2: 텍스트 변환"""
    text = state["step1_result"]
    # 단어별로 첫 글자 대문자
    transformed = " ".join(word.capitalize() for word in text.split())
    return {"step2_result": transformed}


def step3_finalize(state: TextProcessingState) -> TextProcessingState:
    """Step 3: 최종 처리"""
    text = state["step2_result"]
    final = f"[처리됨] {text}"
    return {"final_result": final}


def create_basic_chain():
    """기본 Prompt Chaining 그래프 생성"""
    graph = StateGraph(TextProcessingState)

    graph.add_node("clean", step1_clean_text)
    graph.add_node("transform", step2_transform_text)
    graph.add_node("finalize", step3_finalize)

    graph.add_edge(START, "clean")
    graph.add_edge("clean", "transform")
    graph.add_edge("transform", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_basic_chain_example():
    """기본 체이닝 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 Prompt Chaining")
    print("=" * 60)

    app = create_basic_chain()

    initial_state = {
        "original_text": "  HELLO WORLD, THIS IS LANGGRAPH!  ",
        "step1_result": "",
        "step2_result": "",
        "final_result": ""
    }

    result = app.invoke(initial_state)

    print(f"\n📝 처리 과정:")
    print(f"   원본: '{initial_state['original_text']}'")
    print(f"   Step 1 (정리): '{result['step1_result']}'")
    print(f"   Step 2 (변환): '{result['step2_result']}'")
    print(f"   최종: '{result['final_result']}'")


# =============================================================================
# 3. LLM을 사용한 Prompt Chaining
# =============================================================================

class JokeState(TypedDict):
    """농담 생성을 위한 State"""
    topic: str
    initial_joke: str
    critique: str
    improved_joke: str


def create_llm_chain():
    """LLM을 사용한 농담 개선 체인 생성"""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

    def generate_joke(state: JokeState) -> JokeState:
        """Step 1: 초기 농담 생성"""
        messages = [
            SystemMessage(content="당신은 재미있는 농담을 만드는 코미디언입니다."),
            HumanMessage(content=f"'{state['topic']}'에 대한 짧은 농담을 하나 만들어주세요.")
        ]
        response = llm.invoke(messages)
        return {"initial_joke": response.content}

    def critique_joke(state: JokeState) -> JokeState:
        """Step 2: 농담 평가"""
        messages = [
            SystemMessage(content="당신은 코미디 비평가입니다."),
            HumanMessage(content=f"""다음 농담을 평가하고 개선점을 제안해주세요:

농담: {state['initial_joke']}

개선할 점을 2-3가지 제안해주세요.""")
        ]
        response = llm.invoke(messages)
        return {"critique": response.content}

    def improve_joke(state: JokeState) -> JokeState:
        """Step 3: 농담 개선"""
        messages = [
            SystemMessage(content="당신은 창의적인 코미디언입니다."),
            HumanMessage(content=f"""다음 피드백을 바탕으로 농담을 개선해주세요:

원래 농담: {state['initial_joke']}

피드백: {state['critique']}

개선된 농담만 작성해주세요.""")
        ]
        response = llm.invoke(messages)
        return {"improved_joke": response.content}

    graph = StateGraph(JokeState)
    graph.add_node("generate", generate_joke)
    graph.add_node("critique", critique_joke)
    graph.add_node("improve", improve_joke)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "critique")
    graph.add_edge("critique", "improve")
    graph.add_edge("improve", END)

    return graph.compile()


def run_llm_chain_example():
    """LLM 체이닝 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: LLM Prompt Chaining (농담 개선)")
    print("=" * 60)

    load_dotenv()
    app = create_llm_chain()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        print("   ANTHROPIC_API_KEY를 설정하고 langchain-anthropic을 설치하세요.")
        return

    result = app.invoke({
        "topic": "프로그래밍",
        "initial_joke": "",
        "critique": "",
        "improved_joke": ""
    })

    print(f"\n🎯 주제: '{result['topic']}'")
    print(f"\n📝 Step 1 - 초기 농담:")
    print(f"   {result['initial_joke']}")
    print(f"\n🔍 Step 2 - 평가:")
    print(f"   {result['critique'][:200]}...")
    print(f"\n✨ Step 3 - 개선된 농담:")
    print(f"   {result['improved_joke']}")


# =============================================================================
# 4. Gate(검증) 단계가 있는 체이닝
# =============================================================================

class GatedState(TypedDict):
    """검증 단계가 있는 State"""
    input_text: str
    processed: str
    is_valid: bool
    error_message: str
    final_output: str


def process_input(state: GatedState) -> GatedState:
    """입력 처리"""
    processed = state["input_text"].strip().upper()
    return {"processed": processed}


def validate_input(state: GatedState) -> GatedState:
    """입력 검증 (Gate)"""
    processed = state["processed"]

    # 검증 규칙
    if len(processed) < 3:
        return {
            "is_valid": False,
            "error_message": "입력이 너무 짧습니다 (최소 3자)"
        }
    if not processed.replace(" ", "").isalnum():
        return {
            "is_valid": False,
            "error_message": "영숫자만 허용됩니다"
        }

    return {"is_valid": True, "error_message": ""}


def finalize_valid(state: GatedState) -> GatedState:
    """유효한 입력 처리"""
    return {"final_output": f"✅ 성공: {state['processed']}"}


def handle_invalid(state: GatedState) -> GatedState:
    """유효하지 않은 입력 처리"""
    return {"final_output": f"❌ 실패: {state['error_message']}"}


def route_by_validation(state: GatedState) -> str:
    """검증 결과에 따라 라우팅"""
    if state["is_valid"]:
        return "finalize"
    return "handle_error"


def create_gated_chain():
    """검증 단계가 있는 체인 생성"""
    graph = StateGraph(GatedState)

    graph.add_node("process", process_input)
    graph.add_node("validate", validate_input)
    graph.add_node("finalize", finalize_valid)
    graph.add_node("handle_error", handle_invalid)

    graph.add_edge(START, "process")
    graph.add_edge("process", "validate")

    # 조건부 엣지 - 검증 결과에 따라 분기
    graph.add_conditional_edges(
        "validate",
        route_by_validation,
        {
            "finalize": "finalize",
            "handle_error": "handle_error"
        }
    )

    graph.add_edge("finalize", END)
    graph.add_edge("handle_error", END)

    return graph.compile()


def run_gated_chain_example():
    """검증 체이닝 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: Gate(검증) 단계가 있는 체이닝")
    print("=" * 60)

    app = create_gated_chain()

    test_cases = [
        "Hello World",  # 유효
        "AB",           # 너무 짧음
        "Test@123",     # 특수문자 포함
    ]

    for text in test_cases:
        result = app.invoke({
            "input_text": text,
            "processed": "",
            "is_valid": False,
            "error_message": "",
            "final_output": ""
        })
        print(f"\n   입력: '{text}'")
        print(f"   결과: {result['final_output']}")


# =============================================================================
# 5. 스트리밍과 함께 사용
# =============================================================================

class StreamingState(TypedDict):
    """스트리밍을 위한 State"""
    steps: Annotated[list, lambda x, y: x + y]
    current_step: int


def step_a(state: StreamingState) -> StreamingState:
    """Step A"""
    return {"steps": ["A 완료"], "current_step": 1}


def step_b(state: StreamingState) -> StreamingState:
    """Step B"""
    return {"steps": ["B 완료"], "current_step": 2}


def step_c(state: StreamingState) -> StreamingState:
    """Step C"""
    return {"steps": ["C 완료"], "current_step": 3}


def create_streaming_chain():
    """스트리밍 체인 생성"""
    graph = StateGraph(StreamingState)

    graph.add_node("step_a", step_a)
    graph.add_node("step_b", step_b)
    graph.add_node("step_c", step_c)

    graph.add_edge(START, "step_a")
    graph.add_edge("step_a", "step_b")
    graph.add_edge("step_b", "step_c")
    graph.add_edge("step_c", END)

    return graph.compile()


def run_streaming_example():
    """스트리밍 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: 스트리밍으로 단계별 진행 확인")
    print("=" * 60)

    app = create_streaming_chain()

    initial = {"steps": [], "current_step": 0}

    print("\n🔄 진행 상황:")

    # stream() 메서드로 단계별 출력 확인
    for event in app.stream(initial):
        for node_name, state_update in event.items():
            print(f"   [{node_name}] 완료 - Step {state_update.get('current_step', '?')}")


# =============================================================================
# 6. Prompt Chaining 패턴 정리
# =============================================================================

def explain_prompt_chaining_patterns():
    """Prompt Chaining 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Prompt Chaining 패턴 정리")
    print("=" * 60)

    print("""
Prompt Chaining의 특징:
1. 선형적 흐름 - A → B → C → D
2. 각 단계의 출력이 다음 단계의 입력
3. 예측 가능한 실행 순서

사용 시나리오:
- 문서 요약 후 번역
- 코드 생성 후 리뷰
- 데이터 추출 후 분석

일반적인 패턴:

1. 순차적 처리
   START → process_1 → process_2 → ... → END

2. Gate(검증) 포함
   START → process → validate →┬→ success → END
                               └→ failure → END

3. 피드백 루프 (다음 챕터에서 자세히)
   START → generate → evaluate →┬→ END (합격)
                     ↑          └→ improve
                     └────────────────┘

구현 팁:
- State에 중간 결과를 모두 저장하면 디버깅이 쉬움
- stream()을 사용하면 진행 상황 모니터링 가능
- 각 단계를 독립적으로 테스트 가능하게 설계
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 5] 워크플로우 패턴 개요 - Prompt Chaining")
    print("=" * 60)

    # 환경 변수 로드
    load_dotenv()

    # 개념 설명
    explain_workflow_vs_agent()

    # 예제 실행
    run_basic_chain_example()
    run_llm_chain_example()
    run_gated_chain_example()
    run_streaming_example()

    # 패턴 정리
    explain_prompt_chaining_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 06_routing.py (조건부 라우팅)")
    print("=" * 60)


if __name__ == "__main__":
    main()
