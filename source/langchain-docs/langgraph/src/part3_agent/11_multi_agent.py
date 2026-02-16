"""
[Chapter 11] Multi-Agent 시스템

📝 설명:
    Multi-Agent 시스템은 여러 전문화된 Agent가 협력하여
    복잡한 작업을 수행하는 아키텍처입니다.
    Supervisor 패턴과 Handoff 패턴을 학습합니다.

🎯 학습 목표:
    - Multi-Agent 아키텍처 이해
    - Agent 간 통신 방법
    - Supervisor 패턴 구현
    - Handoff 패턴 구현

📚 관련 문서:
    - docs/Part3-Agent/11-multi-agent.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/multi_agent/

💻 실행 방법:
    python -m src.part3_agent.11_multi_agent

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
"""

import os
from typing import TypedDict, Annotated, Literal, List
from dotenv import load_dotenv
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command


# =============================================================================
# 1. Multi-Agent 기본 개념
# =============================================================================

def explain_multi_agent_concepts():
    """Multi-Agent 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Multi-Agent 시스템 개념")
    print("=" * 60)

    print("""
Multi-Agent 시스템의 두 가지 주요 패턴:

1. Supervisor 패턴
   - 중앙 조정자(Supervisor)가 작업을 분배
   - 각 Agent는 특정 역할 담당
   - Supervisor가 결과를 종합

   ┌─────────────────────────────────────┐
   │            Supervisor              │
   │         (작업 분배/조정)            │
   └───────────┬───────────┬────────────┘
               │           │
       ┌───────▼───┐   ┌───▼───────┐
       │ Agent A   │   │ Agent B   │
       │ (역할 A)  │   │ (역할 B)  │
       └───────────┘   └───────────┘

2. Handoff 패턴
   - Agent 간 직접 작업 인계
   - 현재 Agent가 다음 Agent 결정
   - 분산된 의사결정

   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ Agent A │────▶│ Agent B │────▶│ Agent C │
   └─────────┘     └─────────┘     └─────────┘
""")


# =============================================================================
# 2. 간단한 Multi-Agent (역할 분담)
# =============================================================================

class MultiAgentState(TypedDict):
    """Multi-Agent State"""
    messages: Annotated[List, operator.add]
    current_agent: str
    task: str
    results: Annotated[List[str], operator.add]


def researcher_agent(state: MultiAgentState) -> MultiAgentState:
    """연구원 Agent - 정보 수집"""
    task = state["task"]

    # 시뮬레이션된 연구 결과
    research = f"[연구원] '{task}'에 대한 조사 결과: 핵심 정보를 수집했습니다."

    return {
        "messages": [AIMessage(content=research, name="researcher")],
        "results": [research]
    }


def writer_agent(state: MultiAgentState) -> MultiAgentState:
    """작가 Agent - 콘텐츠 작성"""
    task = state["task"]
    research = state["results"][-1] if state["results"] else ""

    # 시뮬레이션된 작성 결과
    content = f"[작가] 연구 결과를 바탕으로 '{task}'에 대한 글을 작성했습니다."

    return {
        "messages": [AIMessage(content=content, name="writer")],
        "results": [content]
    }


def reviewer_agent(state: MultiAgentState) -> MultiAgentState:
    """리뷰어 Agent - 검토"""
    task = state["task"]

    # 시뮬레이션된 리뷰 결과
    review = f"[리뷰어] '{task}'에 대한 콘텐츠를 검토했습니다. 품질: 양호"

    return {
        "messages": [AIMessage(content=review, name="reviewer")],
        "results": [review]
    }


def create_sequential_multi_agent():
    """순차적 Multi-Agent 그래프 생성"""
    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reviewer", reviewer_agent)

    # 순차적 실행
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


def run_sequential_multi_agent_example():
    """순차적 Multi-Agent 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 순차적 Multi-Agent")
    print("=" * 60)

    app = create_sequential_multi_agent()

    result = app.invoke({
        "messages": [],
        "current_agent": "",
        "task": "AI Agent의 미래",
        "results": []
    })

    print(f"\n📋 작업: {result['task']}")
    print(f"\n🤖 Agent 실행 결과:")
    for r in result["results"]:
        print(f"   {r}")


# =============================================================================
# 3. Supervisor 패턴
# =============================================================================

class SupervisorState(TypedDict):
    """Supervisor State"""
    messages: Annotated[List, operator.add]
    task: str
    next_agent: str
    agent_outputs: Annotated[List[str], operator.add]
    iteration: int


def supervisor_node(state: SupervisorState) -> SupervisorState:
    """Supervisor 노드 - 작업 분배"""
    iteration = state.get("iteration", 0)
    outputs = state.get("agent_outputs", [])

    # 간단한 분배 로직
    if iteration == 0:
        return {"next_agent": "analyst", "iteration": 1}
    elif iteration == 1:
        return {"next_agent": "coder", "iteration": 2}
    elif iteration == 2:
        return {"next_agent": "tester", "iteration": 3}
    else:
        return {"next_agent": "finish", "iteration": iteration + 1}


def analyst_node(state: SupervisorState) -> SupervisorState:
    """분석가 Agent"""
    output = f"[분석가] 요구사항 분석 완료: {state['task']}"
    return {
        "messages": [AIMessage(content=output, name="analyst")],
        "agent_outputs": [output]
    }


def coder_node(state: SupervisorState) -> SupervisorState:
    """코더 Agent"""
    output = "[코더] 코드 구현 완료: def solution(): pass"
    return {
        "messages": [AIMessage(content=output, name="coder")],
        "agent_outputs": [output]
    }


def tester_node(state: SupervisorState) -> SupervisorState:
    """테스터 Agent"""
    output = "[테스터] 테스트 통과: 3/3 테스트 성공"
    return {
        "messages": [AIMessage(content=output, name="tester")],
        "agent_outputs": [output]
    }


def route_supervisor(state: SupervisorState) -> str:
    """Supervisor 라우팅"""
    next_agent = state.get("next_agent", "finish")
    if next_agent == "finish":
        return "finish"
    return next_agent


def create_supervisor_graph():
    """Supervisor 패턴 그래프 생성"""
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("coder", coder_node)
    graph.add_node("tester", tester_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "analyst": "analyst",
            "coder": "coder",
            "tester": "tester",
            "finish": END
        }
    )

    # 각 Agent 실행 후 Supervisor로 복귀
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("coder", "supervisor")
    graph.add_edge("tester", "supervisor")

    return graph.compile()


def run_supervisor_example():
    """Supervisor 패턴 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: Supervisor 패턴")
    print("=" * 60)

    app = create_supervisor_graph()

    result = app.invoke({
        "messages": [],
        "task": "계산기 앱 개발",
        "next_agent": "",
        "agent_outputs": [],
        "iteration": 0
    })

    print(f"\n📋 작업: {result['task']}")
    print(f"\n🤖 Agent 실행 순서:")
    for output in result["agent_outputs"]:
        print(f"   {output}")


# =============================================================================
# 4. LLM 기반 Supervisor
# =============================================================================

def create_llm_supervisor_graph():
    """LLM 기반 Supervisor 그래프 생성"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    class LLMSupervisorState(TypedDict):
        messages: Annotated[List, operator.add]
        task: str
        next_agent: str
        completed_agents: List[str]

    def llm_supervisor(state: LLMSupervisorState) -> LLMSupervisorState:
        """LLM 기반 Supervisor"""
        completed = state.get("completed_agents", [])
        task = state["task"]

        # 사용 가능한 Agent 목록
        available = ["researcher", "analyst", "writer"]
        remaining = [a for a in available if a not in completed]

        if not remaining:
            return {"next_agent": "FINISH"}

        messages = [
            SystemMessage(content=f"""당신은 팀 관리자입니다.
작업: {task}
완료된 Agent: {completed}
남은 Agent: {remaining}

다음에 실행할 Agent를 선택하세요. 선택지: {remaining}
Agent 이름만 출력하세요."""),
            HumanMessage(content="다음 Agent를 선택하세요.")
        ]

        response = llm.invoke(messages)
        next_agent = response.content.strip().lower()

        # 유효한 Agent인지 확인
        if next_agent not in remaining:
            next_agent = remaining[0] if remaining else "FINISH"

        return {"next_agent": next_agent}

    def llm_researcher(state: LLMSupervisorState) -> LLMSupervisorState:
        """연구원 Agent"""
        messages = [
            SystemMessage(content="당신은 연구원입니다. 주어진 주제를 조사하세요."),
            HumanMessage(content=f"주제: {state['task']}")
        ]
        response = llm.invoke(messages)

        return {
            "messages": [AIMessage(content=f"[연구원] {response.content[:200]}...", name="researcher")],
            "completed_agents": state.get("completed_agents", []) + ["researcher"]
        }

    def llm_analyst(state: LLMSupervisorState) -> LLMSupervisorState:
        """분석가 Agent"""
        return {
            "messages": [AIMessage(content="[분석가] 데이터 분석 완료", name="analyst")],
            "completed_agents": state.get("completed_agents", []) + ["analyst"]
        }

    def llm_writer(state: LLMSupervisorState) -> LLMSupervisorState:
        """작가 Agent"""
        return {
            "messages": [AIMessage(content="[작가] 보고서 작성 완료", name="writer")],
            "completed_agents": state.get("completed_agents", []) + ["writer"]
        }

    def route_llm_supervisor(state: LLMSupervisorState) -> str:
        next_agent = state.get("next_agent", "FINISH")
        if next_agent == "FINISH":
            return "end"
        return next_agent

    graph = StateGraph(LLMSupervisorState)

    graph.add_node("supervisor", llm_supervisor)
    graph.add_node("researcher", llm_researcher)
    graph.add_node("analyst", llm_analyst)
    graph.add_node("writer", llm_writer)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_llm_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "end": END
        }
    )

    graph.add_edge("researcher", "supervisor")
    graph.add_edge("analyst", "supervisor")
    graph.add_edge("writer", "supervisor")

    return graph.compile()


def run_llm_supervisor_example():
    """LLM Supervisor 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: LLM 기반 Supervisor")
    print("=" * 60)

    app = create_llm_supervisor_graph()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    result = app.invoke({
        "messages": [],
        "task": "AI 트렌드 보고서 작성",
        "next_agent": "",
        "completed_agents": []
    })

    print(f"\n📋 작업: {result['task']}")
    print(f"\n🤖 실행된 Agent:")
    for msg in result["messages"]:
        if hasattr(msg, "content"):
            print(f"   {msg.content[:100]}...")


# =============================================================================
# 5. Handoff 패턴 (Command 사용)
# =============================================================================

class HandoffState(TypedDict):
    """Handoff State"""
    messages: Annotated[List, operator.add]
    task: str


def support_agent(state: HandoffState) -> Command[Literal["sales", "tech", "end"]]:
    """지원 Agent - 적절한 Agent로 인계"""
    task = state["task"].lower()

    # 키워드 기반 라우팅
    if "구매" in task or "가격" in task:
        return Command(
            goto="sales",
            update={"messages": [AIMessage(content="[지원] 영업팀으로 연결합니다.", name="support")]}
        )
    elif "기술" in task or "오류" in task:
        return Command(
            goto="tech",
            update={"messages": [AIMessage(content="[지원] 기술팀으로 연결합니다.", name="support")]}
        )
    else:
        return Command(
            goto="end",
            update={"messages": [AIMessage(content="[지원] 문의 처리 완료", name="support")]}
        )


def sales_agent(state: HandoffState) -> HandoffState:
    """영업 Agent"""
    return {
        "messages": [AIMessage(content="[영업] 제품 정보와 가격을 안내해드리겠습니다.", name="sales")]
    }


def tech_agent(state: HandoffState) -> HandoffState:
    """기술 Agent"""
    return {
        "messages": [AIMessage(content="[기술] 기술적인 문제를 해결해드리겠습니다.", name="tech")]
    }


def create_handoff_graph():
    """Handoff 패턴 그래프 생성"""
    graph = StateGraph(HandoffState)

    graph.add_node("support", support_agent)
    graph.add_node("sales", sales_agent)
    graph.add_node("tech", tech_agent)

    graph.add_edge(START, "support")
    # support의 Command가 라우팅 처리
    graph.add_edge("sales", END)
    graph.add_edge("tech", END)

    return graph.compile()


def run_handoff_example():
    """Handoff 패턴 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: Handoff 패턴")
    print("=" * 60)

    app = create_handoff_graph()

    test_cases = [
        "제품 가격이 궁금합니다",
        "기술적인 오류가 발생했어요",
        "일반 문의입니다"
    ]

    for task in test_cases:
        result = app.invoke({
            "messages": [],
            "task": task
        })

        print(f"\n📝 문의: {task}")
        print(f"   처리 결과:")
        for msg in result["messages"]:
            print(f"      {msg.content}")


# =============================================================================
# 6. Multi-Agent 패턴 정리
# =============================================================================

def explain_multi_agent_patterns():
    """Multi-Agent 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Multi-Agent 패턴 정리")
    print("=" * 60)

    print("""
1. Supervisor 패턴
   - 중앙 집중식 제어
   - Supervisor가 모든 결정
   - 일관된 흐름 관리

   구현:
   - supervisor 노드가 next_agent 결정
   - 각 agent 실행 후 supervisor로 복귀
   - 모든 작업 완료 시 END

2. Handoff 패턴
   - 분산된 의사결정
   - Agent가 직접 다음 Agent 결정
   - Command 객체 활용

   구현:
   - Agent가 Command(goto="next") 반환
   - 조건에 따라 동적 라우팅

3. 병렬 Multi-Agent
   - 여러 Agent 동시 실행
   - 결과 집계 필요
   - Reducer 활용

4. 계층적 Multi-Agent
   - Agent 안에 Sub-Agent
   - 서브그래프 활용
   - 복잡한 작업 분해

사용 시나리오:

- 고객 서비스: Handoff (분류 → 전문 상담)
- 콘텐츠 생성: Supervisor (연구 → 작성 → 검토)
- 코드 개발: 병렬 (분석, 구현, 테스트 동시)
- 복잡한 분석: 계층적 (상위 분석 → 세부 분석)
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 11] Multi-Agent 시스템")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_multi_agent_concepts()

    # 예제 실행
    run_sequential_multi_agent_example()
    run_supervisor_example()
    run_llm_supervisor_example()
    run_handoff_example()

    # 패턴 정리
    explain_multi_agent_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 12_subgraph_agent.py (서브그래프)")
    print("=" * 60)


if __name__ == "__main__":
    main()
