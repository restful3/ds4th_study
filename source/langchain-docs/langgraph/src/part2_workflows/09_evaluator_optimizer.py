"""
[Chapter 8 확장] Evaluator-Optimizer 패턴

📝 설명:
    Evaluator-Optimizer 패턴은 결과를 평가하고, 기준을 충족하지 못하면
    개선하는 과정을 반복하는 피드백 루프 패턴입니다.
    품질 향상이 필요한 작업에 적합합니다.

🎯 학습 목표:
    - Evaluator-Optimizer 아키텍처 이해
    - 피드백 루프 구현
    - 종료 조건 설정
    - LLM을 사용한 평가 및 개선

📚 관련 문서:
    - docs/Part2-Workflows/08-orchestrator-worker.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#evaluator-optimizer

💻 실행 방법:
    python -m src.part2_workflows.09_evaluator_optimizer

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
import operator

from langgraph.graph import StateGraph, START, END


# =============================================================================
# 1. 기본 Evaluator-Optimizer 패턴
# =============================================================================

class OptimizationState(TypedDict):
    """최적화를 위한 State"""
    target: int
    current_value: int
    iteration: int
    max_iterations: int
    history: Annotated[list, operator.add]
    is_optimized: bool


def generate(state: OptimizationState) -> OptimizationState:
    """값 생성/조정"""
    current = state["current_value"]
    target = state["target"]

    # 간단한 최적화 로직: 목표에 더 가깝게 조정
    if current < target:
        new_value = min(current + 10, target)
    else:
        new_value = max(current - 10, target)

    return {
        "current_value": new_value,
        "history": [f"Iteration {state['iteration']}: {current} -> {new_value}"]
    }


def evaluate(state: OptimizationState) -> OptimizationState:
    """결과 평가"""
    current = state["current_value"]
    target = state["target"]
    iteration = state["iteration"] + 1

    # 목표 달성 여부 확인
    is_optimized = current == target

    return {
        "iteration": iteration,
        "is_optimized": is_optimized
    }


def should_continue(state: OptimizationState) -> Literal["generate", "end"]:
    """계속 최적화할지 결정"""
    if state["is_optimized"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "generate"


def create_basic_optimizer_graph():
    """기본 Evaluator-Optimizer 그래프 생성"""
    graph = StateGraph(OptimizationState)

    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "generate": "generate",  # 루프백
            "end": END
        }
    )

    return graph.compile()


def run_basic_optimizer_example():
    """기본 Optimizer 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 Evaluator-Optimizer")
    print("=" * 60)

    app = create_basic_optimizer_graph()

    result = app.invoke({
        "target": 100,
        "current_value": 25,
        "iteration": 0,
        "max_iterations": 20,
        "history": [],
        "is_optimized": False
    })

    print(f"\n🎯 목표: {result['target']}")
    print(f"📊 최적화 과정:")
    for h in result["history"]:
        print(f"   {h}")
    print(f"\n✅ 최종 값: {result['current_value']}")
    print(f"   반복 횟수: {result['iteration']}")
    print(f"   최적화 완료: {result['is_optimized']}")


# =============================================================================
# 2. 텍스트 품질 개선 패턴
# =============================================================================

class TextQualityState(TypedDict):
    """텍스트 품질 개선을 위한 State"""
    original_text: str
    current_text: str
    quality_score: int  # 0-100
    feedback: str
    iteration: int
    max_iterations: int
    threshold: int


def improve_text(state: TextQualityState) -> TextQualityState:
    """텍스트 개선"""
    text = state["current_text"]
    feedback = state.get("feedback", "")

    # 간단한 개선 로직
    improvements = []

    # 피드백에 따른 개선
    if "대문자" in feedback or state["iteration"] == 0:
        text = text.capitalize()
        improvements.append("첫 글자 대문자화")

    if "구두점" in feedback or "." not in text:
        if not text.endswith("."):
            text = text + "."
            improvements.append("마침표 추가")

    if "공백" in feedback or "  " in text:
        text = " ".join(text.split())
        improvements.append("공백 정리")

    # 품질 점수 증가
    new_score = min(state["quality_score"] + 20, 100)

    return {
        "current_text": text,
        "quality_score": new_score,
        "feedback": f"적용된 개선: {', '.join(improvements) if improvements else '없음'}"
    }


def evaluate_quality(state: TextQualityState) -> TextQualityState:
    """텍스트 품질 평가"""
    text = state["current_text"]
    iteration = state["iteration"] + 1

    # 품질 평가 기준
    issues = []

    if not text[0].isupper():
        issues.append("대문자로 시작해야 함")
    if not text.endswith("."):
        issues.append("구두점 필요")
    if "  " in text:
        issues.append("공백 정리 필요")
    if len(text) < 10:
        issues.append("텍스트가 너무 짧음")

    feedback = ", ".join(issues) if issues else "품질 기준 충족"

    return {
        "iteration": iteration,
        "feedback": feedback
    }


def should_continue_improving(state: TextQualityState) -> Literal["improve", "end"]:
    """계속 개선할지 결정"""
    if state["quality_score"] >= state["threshold"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if state["feedback"] == "품질 기준 충족":
        return "end"
    return "improve"


def create_text_quality_graph():
    """텍스트 품질 개선 그래프 생성"""
    graph = StateGraph(TextQualityState)

    graph.add_node("improve", improve_text)
    graph.add_node("evaluate", evaluate_quality)

    graph.add_edge(START, "improve")
    graph.add_edge("improve", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_continue_improving,
        {
            "improve": "improve",
            "end": END
        }
    )

    return graph.compile()


def run_text_quality_example():
    """텍스트 품질 개선 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: 텍스트 품질 개선")
    print("=" * 60)

    app = create_text_quality_graph()

    result = app.invoke({
        "original_text": "this is  a test   text without proper formatting",
        "current_text": "this is  a test   text without proper formatting",
        "quality_score": 20,
        "feedback": "",
        "iteration": 0,
        "max_iterations": 5,
        "threshold": 80
    })

    print(f"\n📝 원본: '{result['original_text']}'")
    print(f"✨ 개선: '{result['current_text']}'")
    print(f"📊 품질 점수: {result['quality_score']}/100")
    print(f"   반복 횟수: {result['iteration']}")


# =============================================================================
# 3. LLM 기반 콘텐츠 개선
# =============================================================================

class LLMContentState(TypedDict):
    """LLM 콘텐츠 개선을 위한 State"""
    topic: str
    current_content: str
    evaluation: str
    score: int
    iteration: int
    max_iterations: int


def create_llm_content_graph():
    """LLM 기반 콘텐츠 개선 그래프 생성"""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

    def generate_content(state: LLMContentState) -> LLMContentState:
        """콘텐츠 생성 또는 개선"""
        if not state["current_content"]:
            # 초기 생성
            messages = [
                SystemMessage(content="당신은 기술 블로그 작가입니다."),
                HumanMessage(content=f"'{state['topic']}'에 대한 짧은 소개글(2-3문장)을 작성해주세요.")
            ]
        else:
            # 피드백 기반 개선
            messages = [
                SystemMessage(content="당신은 편집자입니다. 피드백을 바탕으로 글을 개선하세요."),
                HumanMessage(content=f"""원본 글:
{state['current_content']}

피드백:
{state['evaluation']}

피드백을 반영하여 개선된 글만 작성하세요.""")
            ]

        response = llm.invoke(messages)
        return {"current_content": response.content}

    def evaluate_content(state: LLMContentState) -> LLMContentState:
        """콘텐츠 평가"""
        messages = [
            SystemMessage(content="""당신은 콘텐츠 평가 전문가입니다.
다음 기준으로 글을 평가하세요:
1. 명확성 (1-10)
2. 정보성 (1-10)
3. 흥미도 (1-10)

총점(30점 만점)과 개선점을 제시하세요.
형식: "점수: XX/30\n개선점: ..."으로 작성하세요."""),
            HumanMessage(content=state["current_content"])
        ]

        response = llm.invoke(messages)
        evaluation = response.content

        # 점수 추출 (간단한 파싱)
        try:
            score_line = [l for l in evaluation.split("\n") if "점수:" in l][0]
            score = int(score_line.split("/")[0].split(":")[-1].strip())
        except (IndexError, ValueError):
            score = 15  # 기본 점수

        return {
            "evaluation": evaluation,
            "score": score,
            "iteration": state["iteration"] + 1
        }

    def should_continue_improving_content(state: LLMContentState) -> Literal["generate", "end"]:
        """계속 개선할지 결정"""
        if state["score"] >= 25:  # 25/30 이상이면 종료
            return "end"
        if state["iteration"] >= state["max_iterations"]:
            return "end"
        return "generate"

    graph = StateGraph(LLMContentState)

    graph.add_node("generate", generate_content)
    graph.add_node("evaluate", evaluate_content)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_continue_improving_content,
        {
            "generate": "generate",
            "end": END
        }
    )

    return graph.compile()


def run_llm_content_example():
    """LLM 콘텐츠 개선 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: LLM 기반 콘텐츠 개선")
    print("=" * 60)

    load_dotenv()
    app = create_llm_content_graph()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    result = app.invoke({
        "topic": "LangGraph",
        "current_content": "",
        "evaluation": "",
        "score": 0,
        "iteration": 0,
        "max_iterations": 3
    })

    print(f"\n🎯 주제: {result['topic']}")
    print(f"\n📝 최종 콘텐츠:\n{result['current_content']}")
    print(f"\n📊 최종 점수: {result['score']}/30")
    print(f"   반복 횟수: {result['iteration']}")


# =============================================================================
# 4. 다단계 검증 패턴
# =============================================================================

class ValidationState(TypedDict):
    """다단계 검증을 위한 State"""
    data: dict
    validation_results: Annotated[list, operator.add]
    current_stage: int
    all_passed: bool


def validate_format(state: ValidationState) -> ValidationState:
    """형식 검증"""
    data = state["data"]
    results = []

    # 필수 필드 확인
    required = ["name", "email", "age"]
    for field in required:
        if field in data:
            results.append(f"✅ {field}: 존재")
        else:
            results.append(f"❌ {field}: 누락")

    return {"validation_results": results, "current_stage": 1}


def validate_types(state: ValidationState) -> ValidationState:
    """타입 검증"""
    data = state["data"]
    results = []

    if isinstance(data.get("name"), str):
        results.append("✅ name: 문자열")
    else:
        results.append("❌ name: 문자열이어야 함")

    if isinstance(data.get("age"), int):
        results.append("✅ age: 정수")
    else:
        results.append("❌ age: 정수여야 함")

    return {"validation_results": results, "current_stage": 2}


def validate_business(state: ValidationState) -> ValidationState:
    """비즈니스 규칙 검증"""
    data = state["data"]
    results = []

    # 나이 범위 확인
    age = data.get("age", 0)
    if 0 < age < 150:
        results.append("✅ age: 유효한 범위")
    else:
        results.append("❌ age: 0-150 범위여야 함")

    # 이메일 형식 확인 (간단)
    email = data.get("email", "")
    if "@" in email and "." in email:
        results.append("✅ email: 유효한 형식")
    else:
        results.append("❌ email: 유효하지 않은 형식")

    # 전체 결과 확인
    all_passed = all("✅" in r for r in state["validation_results"] + results)

    return {
        "validation_results": results,
        "current_stage": 3,
        "all_passed": all_passed
    }


def create_validation_graph():
    """다단계 검증 그래프 생성"""
    graph = StateGraph(ValidationState)

    graph.add_node("validate_format", validate_format)
    graph.add_node("validate_types", validate_types)
    graph.add_node("validate_business", validate_business)

    graph.add_edge(START, "validate_format")
    graph.add_edge("validate_format", "validate_types")
    graph.add_edge("validate_types", "validate_business")
    graph.add_edge("validate_business", END)

    return graph.compile()


def run_validation_example():
    """다단계 검증 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: 다단계 검증")
    print("=" * 60)

    app = create_validation_graph()

    test_data = {
        "name": "홍길동",
        "email": "hong@example.com",
        "age": 30
    }

    result = app.invoke({
        "data": test_data,
        "validation_results": [],
        "current_stage": 0,
        "all_passed": False
    })

    print(f"\n📋 검증 데이터: {test_data}")
    print(f"\n🔍 검증 결과:")
    for r in result["validation_results"]:
        print(f"   {r}")
    print(f"\n{'✅ 모든 검증 통과!' if result['all_passed'] else '❌ 일부 검증 실패'}")


# =============================================================================
# 5. Evaluator-Optimizer 패턴 정리
# =============================================================================

def explain_evaluator_optimizer_pattern():
    """Evaluator-Optimizer 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Evaluator-Optimizer 패턴 정리")
    print("=" * 60)

    print("""
Evaluator-Optimizer 구조:

    ┌──────────────┐
    │   Generate   │ ◄──────┐
    │  (생성/개선)  │        │
    └──────┬───────┘        │
           │                │
           ▼                │
    ┌──────────────┐        │
    │   Evaluate   │        │
    │    (평가)    │        │
    └──────┬───────┘        │
           │                │
           ▼                │
    ┌──────────────┐        │
    │  Should      │───YES──┘
    │  Continue?   │
    └──────┬───────┘
           │ NO
           ▼
         [END]

핵심 구성요소:

1. Generator (생성기)
   - 초기 결과 생성
   - 피드백 기반 개선
   - 점진적 품질 향상

2. Evaluator (평가기)
   - 결과 품질 평가
   - 피드백 생성
   - 점수/지표 산출

3. Continue Condition (계속 조건)
   - 품질 임계값 도달 여부
   - 최대 반복 횟수
   - 타임아웃

사용 시나리오:
- 콘텐츠 품질 개선
- 코드 최적화
- 데이터 검증
- A/B 테스트 반복

종료 조건 설정 팁:
1. 품질 임계값 설정 (예: 80/100)
2. 최대 반복 횟수 제한 (무한 루프 방지)
3. 타임아웃 설정
4. 개선폭 체크 (더 이상 개선되지 않으면 종료)
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 8 확장] Evaluator-Optimizer 패턴")
    print("=" * 60)

    load_dotenv()

    # 예제 실행
    run_basic_optimizer_example()
    run_text_quality_example()
    run_llm_content_example()
    run_validation_example()

    # 패턴 정리
    explain_evaluator_optimizer_pattern()

    print("\n" + "=" * 60)
    print("✅ Part 2 완료!")
    print("   다음: Part 3 - AI Agent 구현")
    print("=" * 60)


if __name__ == "__main__":
    main()
