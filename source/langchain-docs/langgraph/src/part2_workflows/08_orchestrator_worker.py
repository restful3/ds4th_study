"""
[Chapter 8] Orchestrator-Worker 패턴

📝 설명:
    Orchestrator-Worker 패턴은 중앙의 조율자(Orchestrator)가 작업을 분해하고
    여러 워커(Worker)에게 분배한 후 결과를 수집하는 패턴입니다.
    복잡한 작업을 효율적으로 처리할 수 있습니다.

🎯 학습 목표:
    - Orchestrator-Worker 아키텍처 이해
    - 동적 작업 분배 구현
    - LLM을 사용한 작업 분해
    - 결과 집계 및 합성

📚 관련 문서:
    - docs/Part2-Workflows/08-orchestrator-worker.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#orchestrator-worker

💻 실행 방법:
    python -m src.part2_workflows.08_orchestrator_worker

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
"""

import os
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# =============================================================================
# 1. 기본 Orchestrator-Worker 패턴
# =============================================================================

class OrchestratorState(TypedDict):
    """Orchestrator-Worker State"""
    task: str
    subtasks: List[str]
    current_subtask: str  # Worker용
    results: Annotated[List[str], operator.add]
    final_result: str


def orchestrate(state: OrchestratorState) -> OrchestratorState:
    """
    Orchestrator: 작업을 하위 작업으로 분해

    실제 시나리오에서는 LLM을 사용하여 작업을 분해할 수 있습니다.
    """
    task = state["task"]

    # 간단한 작업 분해 로직
    # 실제로는 LLM이 이 역할을 수행
    subtasks = [
        f"분석: {task}",
        f"검증: {task}",
        f"최적화: {task}"
    ]

    return {"subtasks": subtasks}


def distribute_to_workers(state: OrchestratorState) -> List[Send]:
    """작업을 워커들에게 분배"""
    return [
        Send("worker", {"current_subtask": subtask})
        for subtask in state["subtasks"]
    ]


def worker(state: OrchestratorState) -> OrchestratorState:
    """Worker: 할당된 하위 작업 수행"""
    subtask = state["current_subtask"]

    # 간단한 작업 처리
    result = f"✅ 완료: {subtask}"

    return {"results": [result]}


def synthesize(state: OrchestratorState) -> OrchestratorState:
    """결과 합성"""
    results = state["results"]
    final = f"[최종 결과] {len(results)}개 작업 완료:\n"
    for r in results:
        final += f"  - {r}\n"
    return {"final_result": final}


def create_basic_orchestrator_graph():
    """기본 Orchestrator-Worker 그래프 생성"""
    graph = StateGraph(OrchestratorState)

    graph.add_node("orchestrate", orchestrate)
    graph.add_node("distribute", distribute_to_workers)
    graph.add_node("worker", worker)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "orchestrate")
    graph.add_edge("orchestrate", "distribute")
    # distribute가 Send를 반환하므로 worker로 자동 분기
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_basic_orchestrator_example():
    """기본 Orchestrator 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 Orchestrator-Worker")
    print("=" * 60)

    app = create_basic_orchestrator_graph()

    result = app.invoke({
        "task": "시스템 상태 점검",
        "subtasks": [],
        "current_subtask": "",
        "results": [],
        "final_result": ""
    })

    print(f"\n🎯 원본 작업: {result['task']}")
    print(f"\n📋 분해된 하위 작업:")
    for subtask in result["subtasks"]:
        print(f"   - {subtask}")
    print(f"\n{result['final_result']}")


# =============================================================================
# 2. LLM 기반 작업 분해
# =============================================================================

class LLMOrchestratorState(TypedDict):
    """LLM Orchestrator State"""
    user_request: str
    plan: List[str]
    current_step: str
    step_results: Annotated[List[str], operator.add]
    final_response: str


def create_llm_orchestrator_graph():
    """LLM 기반 Orchestrator 그래프 생성"""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    def plan_tasks(state: LLMOrchestratorState) -> LLMOrchestratorState:
        """LLM을 사용하여 작업 계획 수립"""
        messages = [
            SystemMessage(content="""사용자 요청을 수행하기 위한 단계별 계획을 세우세요.
각 단계를 한 줄씩 나열하세요. 3-5단계가 적절합니다.
단계 번호 없이 작업 내용만 작성하세요."""),
            HumanMessage(content=f"요청: {state['user_request']}")
        ]
        response = llm.invoke(messages)

        # 줄 단위로 분리하여 계획 추출
        plan = [
            step.strip()
            for step in response.content.strip().split("\n")
            if step.strip()
        ]

        return {"plan": plan}

    def distribute_plan(state: LLMOrchestratorState) -> List[Send]:
        """계획을 워커들에게 분배"""
        return [
            Send("execute_step", {"current_step": step})
            for step in state["plan"]
        ]

    def execute_step(state: LLMOrchestratorState) -> LLMOrchestratorState:
        """각 단계 실행"""
        step = state["current_step"]

        messages = [
            SystemMessage(content="당신은 작업 실행 전문가입니다. 주어진 작업을 수행하고 결과를 간단히 보고하세요."),
            HumanMessage(content=f"작업: {step}\n\n결과를 2-3문장으로 보고하세요.")
        ]
        response = llm.invoke(messages)

        return {"step_results": [f"[{step}]\n{response.content}"]}

    def synthesize_response(state: LLMOrchestratorState) -> LLMOrchestratorState:
        """최종 응답 합성"""
        results_text = "\n\n".join(state["step_results"])

        messages = [
            SystemMessage(content="각 단계의 결과를 종합하여 사용자에게 최종 응답을 작성하세요."),
            HumanMessage(content=f"""원본 요청: {state['user_request']}

각 단계 결과:
{results_text}

종합 응답을 작성하세요.""")
        ]
        response = llm.invoke(messages)

        return {"final_response": response.content}

    graph = StateGraph(LLMOrchestratorState)

    graph.add_node("planner", plan_tasks)
    graph.add_node("distributor", distribute_plan)
    graph.add_node("execute_step", execute_step)
    graph.add_node("synthesizer", synthesize_response)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "distributor")
    graph.add_edge("execute_step", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_llm_orchestrator_example():
    """LLM Orchestrator 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: LLM 기반 Orchestrator")
    print("=" * 60)

    load_dotenv()
    app = create_llm_orchestrator_graph()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    result = app.invoke({
        "user_request": "Python으로 간단한 웹 크롤러를 만드는 방법을 설명해주세요",
        "plan": [],
        "current_step": "",
        "step_results": [],
        "final_response": ""
    })

    print(f"\n🎯 사용자 요청: {result['user_request']}")
    print(f"\n📋 실행 계획:")
    for i, step in enumerate(result["plan"], 1):
        print(f"   {i}. {step}")
    print(f"\n📝 최종 응답:\n{result['final_response'][:500]}...")


# =============================================================================
# 3. 전문가 팀 패턴
# =============================================================================

class ExpertTeamState(TypedDict):
    """전문가 팀 State"""
    question: str
    expert_type: str  # 개별 전문가용
    expert_opinions: Annotated[List[str], operator.add]
    consensus: str


def assign_experts(state: ExpertTeamState) -> List[Send]:
    """질문을 여러 전문가에게 할당"""
    experts = ["기술 전문가", "비즈니스 전문가", "보안 전문가"]

    return [
        Send("consult_expert", {"expert_type": expert, "question": state["question"]})
        for expert in experts
    ]


def consult_expert(state: ExpertTeamState) -> ExpertTeamState:
    """전문가 의견 수렴"""
    expert = state["expert_type"]
    question = state["question"]

    # 전문가별 관점 시뮬레이션
    perspectives = {
        "기술 전문가": "기술적 관점에서 구현 가능성과 아키텍처를 고려합니다.",
        "비즈니스 전문가": "비용 효율성과 시장 가치를 분석합니다.",
        "보안 전문가": "보안 위험과 데이터 보호 측면을 검토합니다."
    }

    opinion = f"[{expert}]\n{perspectives.get(expert, '의견 없음')}\n질문 '{question}'에 대해: 긍정적 검토 결과."

    return {"expert_opinions": [opinion]}


def reach_consensus(state: ExpertTeamState) -> ExpertTeamState:
    """전문가 의견 종합"""
    opinions = state["expert_opinions"]
    consensus = f"총 {len(opinions)}명의 전문가 의견 종합:\n"
    consensus += "모든 전문가가 해당 제안에 대해 긍정적인 검토 결과를 제시했습니다."

    return {"consensus": consensus}


def create_expert_team_graph():
    """전문가 팀 그래프 생성"""
    graph = StateGraph(ExpertTeamState)

    graph.add_node("assign", assign_experts)
    graph.add_node("consult_expert", consult_expert)
    graph.add_node("consensus", reach_consensus)

    graph.add_edge(START, "assign")
    graph.add_edge("consult_expert", "consensus")
    graph.add_edge("consensus", END)

    return graph.compile()


def run_expert_team_example():
    """전문가 팀 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: 전문가 팀 패턴")
    print("=" * 60)

    app = create_expert_team_graph()

    result = app.invoke({
        "question": "AI 챗봇을 도입해야 할까요?",
        "expert_type": "",
        "expert_opinions": [],
        "consensus": ""
    })

    print(f"\n❓ 질문: {result['question']}")
    print(f"\n👥 전문가 의견:")
    for opinion in result["expert_opinions"]:
        print(f"\n{opinion}")
    print(f"\n✅ {result['consensus']}")


# =============================================================================
# 4. 재귀적 Orchestrator (복잡한 작업)
# =============================================================================

class RecursiveState(TypedDict):
    """재귀적 처리를 위한 State"""
    task: str
    depth: int
    max_depth: int
    all_results: Annotated[List[str], operator.add]


def check_complexity(state: RecursiveState) -> str:
    """작업 복잡도 확인 및 라우팅"""
    if state["depth"] >= state["max_depth"]:
        return "execute"
    elif len(state["task"]) > 20:  # 간단한 복잡도 기준
        return "decompose"
    else:
        return "execute"


def decompose_task(state: RecursiveState) -> List[Send]:
    """작업을 더 작은 단위로 분해"""
    task = state["task"]
    depth = state["depth"]

    # 간단한 분해 로직
    mid = len(task) // 2
    subtasks = [
        task[:mid].strip(),
        task[mid:].strip()
    ]

    return [
        Send("process", {
            "task": subtask,
            "depth": depth + 1,
            "max_depth": state["max_depth"]
        })
        for subtask in subtasks if subtask
    ]


def execute_task(state: RecursiveState) -> RecursiveState:
    """실제 작업 실행"""
    result = f"[Depth {state['depth']}] 실행: {state['task'][:30]}..."
    return {"all_results": [result]}


def create_recursive_orchestrator_graph():
    """재귀적 Orchestrator 그래프 생성"""
    graph = StateGraph(RecursiveState)

    graph.add_node("check", lambda s: s)  # 라우팅만 담당
    graph.add_node("decompose", decompose_task)
    graph.add_node("execute", execute_task)
    graph.add_node("process", lambda s: s)  # 진입점

    graph.add_edge(START, "process")

    graph.add_conditional_edges(
        "process",
        check_complexity,
        {
            "decompose": "decompose",
            "execute": "execute"
        }
    )

    # decompose는 Send로 process를 재귀 호출
    graph.add_edge("execute", END)

    return graph.compile()


def run_recursive_orchestrator_example():
    """재귀적 Orchestrator 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: 재귀적 작업 분해")
    print("=" * 60)

    app = create_recursive_orchestrator_graph()

    result = app.invoke({
        "task": "대규모 시스템의 성능 분석 및 최적화 보고서 작성",
        "depth": 0,
        "max_depth": 2,
        "all_results": []
    })

    print(f"\n📊 실행 결과:")
    for r in result["all_results"]:
        print(f"   {r}")


# =============================================================================
# 5. Orchestrator-Worker 패턴 정리
# =============================================================================

def explain_orchestrator_pattern():
    """Orchestrator-Worker 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Orchestrator-Worker 패턴 정리")
    print("=" * 60)

    print("""
Orchestrator-Worker 구조:

                    ┌─────────────┐
                    │ Orchestrator│
                    │   (분해)    │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Worker 1 │    │ Worker 2 │    │ Worker N │
    └────┬─────┘    └────┬─────┘    └────┬─────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    ┌──────────────┐
                    │  Synthesizer │
                    │    (합성)    │
                    └──────────────┘

핵심 구성요소:

1. Orchestrator (조율자)
   - 복잡한 작업을 하위 작업으로 분해
   - 작업 계획 수립
   - LLM을 활용한 지능적 분해 가능

2. Workers (작업자)
   - 개별 하위 작업 수행
   - 독립적으로 병렬 실행
   - Send API로 동적 생성

3. Synthesizer (합성기)
   - 워커 결과 수집
   - 최종 결과 합성
   - 일관된 출력 생성

사용 시나리오:
- 복잡한 분석 작업
- 문서 처리 파이프라인
- 다중 전문가 상담
- 대규모 데이터 처리

LLM 활용:
- Orchestrator: 작업 계획 수립
- Worker: 전문화된 처리
- Synthesizer: 결과 종합
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 8] Orchestrator-Worker 패턴")
    print("=" * 60)

    load_dotenv()

    # 예제 실행
    run_basic_orchestrator_example()
    run_llm_orchestrator_example()
    run_expert_team_example()
    run_recursive_orchestrator_example()

    # 패턴 정리
    explain_orchestrator_pattern()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 09_evaluator_optimizer.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
