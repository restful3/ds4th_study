"""
멀티 에이전트 팀 (Multi-Agent Team)

이 예제는 여러 전문 에이전트가 협업하는 시스템을 구현합니다.
Supervisor 패턴을 사용하여 작업을 분배하고 결과를 수집합니다.

에이전트 구성:
- Supervisor: 작업 분배 및 조율
- Researcher: 정보 수집 및 분석
- Writer: 콘텐츠 작성
- Critic: 품질 검토 및 피드백

실행 방법:
    python -m examples.03_multi_agent_team.main
"""

import os
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator


# =============================================================================
# 환경 설정
# =============================================================================

load_dotenv()


# =============================================================================
# State 정의
# =============================================================================

class TeamState(TypedDict):
    """팀 협업 State"""
    # 원본 요청
    request: str

    # 현재 단계
    current_agent: str

    # 각 에이전트의 작업 결과
    research_result: str
    draft: str
    feedback: str
    final_output: str

    # 작업 히스토리
    history: Annotated[List[str], operator.add]

    # 반복 카운터
    revision_count: int


# =============================================================================
# 에이전트 노드 정의
# =============================================================================

def supervisor_node(state: TeamState) -> TeamState:
    """
    Supervisor 노드 - 작업 분배 및 조율

    다음 단계를 결정합니다:
    - 처음: Researcher에게 작업 할당
    - 리서치 완료: Writer에게 작업 할당
    - 초안 완료: Critic에게 작업 할당
    - 피드백 후: 수정 필요하면 Writer, 아니면 완료
    """
    request = state["request"]
    research = state.get("research_result", "")
    draft = state.get("draft", "")
    feedback = state.get("feedback", "")
    revision_count = state.get("revision_count", 0)

    # 작업 흐름 결정
    if not research:
        # 리서치가 없으면 Researcher에게 할당
        return {
            "current_agent": "researcher",
            "history": ["[Supervisor] Researcher에게 정보 수집 요청"]
        }
    elif not draft:
        # 리서치는 있지만 초안이 없으면 Writer에게 할당
        return {
            "current_agent": "writer",
            "history": ["[Supervisor] Writer에게 초안 작성 요청"]
        }
    elif not feedback:
        # 초안은 있지만 피드백이 없으면 Critic에게 할당
        return {
            "current_agent": "critic",
            "history": ["[Supervisor] Critic에게 검토 요청"]
        }
    else:
        # 피드백이 있으면 수정 여부 결정
        if "수정 필요" in feedback and revision_count < 2:
            return {
                "current_agent": "writer",
                "revision_count": revision_count + 1,
                "feedback": "",  # 피드백 초기화
                "history": [f"[Supervisor] 수정 요청 (revision {revision_count + 1})"]
            }
        else:
            return {
                "current_agent": "done",
                "final_output": draft,
                "history": ["[Supervisor] 작업 완료!"]
            }


def researcher_node(state: TeamState) -> TeamState:
    """
    Researcher 노드 - 정보 수집 및 분석
    """
    request = state["request"]

    # LLM이 있으면 사용
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

            system_msg = SystemMessage(content="""당신은 리서치 전문가입니다.
            주어진 주제에 대해 핵심 정보를 수집하고 분석해주세요.
            결과는 구조화된 형태로 정리해주세요.""")

            response = llm.invoke([
                system_msg,
                HumanMessage(content=f"다음 주제에 대해 리서치해주세요: {request}")
            ])

            return {
                "research_result": response.content,
                "history": ["[Researcher] 리서치 완료"]
            }

        except ImportError:
            pass

    # 시뮬레이션 응답
    simulated_research = f"""
리서치 결과: {request}

1. 핵심 개념
   - 주제의 정의와 배경
   - 관련 기술 및 트렌드

2. 주요 포인트
   - 장점과 특징
   - 사용 사례

3. 참고 자료
   - 공식 문서
   - 관련 논문
"""

    return {
        "research_result": simulated_research,
        "history": ["[Researcher] 리서치 완료 (시뮬레이션)"]
    }


def writer_node(state: TeamState) -> TeamState:
    """
    Writer 노드 - 콘텐츠 작성
    """
    request = state["request"]
    research = state.get("research_result", "")
    feedback = state.get("feedback", "")

    # LLM이 있으면 사용
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.7)

            system_msg = SystemMessage(content="""당신은 콘텐츠 작가입니다.
            리서치 결과를 바탕으로 명확하고 읽기 쉬운 글을 작성해주세요.
            이전 피드백이 있다면 반영해주세요.""")

            prompt = f"주제: {request}\n\n리서치 결과:\n{research}"
            if feedback:
                prompt += f"\n\n이전 피드백:\n{feedback}"

            response = llm.invoke([
                system_msg,
                HumanMessage(content=prompt)
            ])

            return {
                "draft": response.content,
                "history": ["[Writer] 초안 작성 완료"]
            }

        except ImportError:
            pass

    # 시뮬레이션 응답
    revision_note = " (피드백 반영)" if feedback else ""
    simulated_draft = f"""
# {request}{revision_note}

## 서론
{request}에 대해 알아보겠습니다.

## 본론
리서치 결과를 바탕으로 핵심 내용을 정리했습니다.

{research[:200]}...

## 결론
이 글을 통해 {request}에 대한 이해가 깊어졌기를 바랍니다.
"""

    return {
        "draft": simulated_draft,
        "history": [f"[Writer] 초안 작성 완료{revision_note} (시뮬레이션)"]
    }


def critic_node(state: TeamState) -> TeamState:
    """
    Critic 노드 - 품질 검토 및 피드백
    """
    draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)

    # LLM이 있으면 사용
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

            system_msg = SystemMessage(content="""당신은 콘텐츠 품질 검토 전문가입니다.
            초안을 검토하고 구체적인 피드백을 제공해주세요.

            피드백 형식:
            - 좋은 점: ...
            - 개선 필요: ...
            - 결론: "수정 필요" 또는 "승인"
            """)

            response = llm.invoke([
                system_msg,
                HumanMessage(content=f"다음 초안을 검토해주세요:\n\n{draft}")
            ])

            return {
                "feedback": response.content,
                "history": ["[Critic] 검토 완료"]
            }

        except ImportError:
            pass

    # 시뮬레이션 - 첫 번째 검토에서는 수정 요청, 두 번째는 승인
    if revision_count == 0:
        simulated_feedback = """
검토 결과:

좋은 점:
- 구조가 명확함
- 핵심 내용이 포함됨

개선 필요:
- 예시 추가 필요
- 결론 보강 필요

결론: 수정 필요
"""
    else:
        simulated_feedback = """
검토 결과:

좋은 점:
- 구조가 명확함
- 핵심 내용이 잘 정리됨
- 피드백이 잘 반영됨

결론: 승인
"""

    return {
        "feedback": simulated_feedback,
        "history": ["[Critic] 검토 완료 (시뮬레이션)"]
    }


# =============================================================================
# 라우팅 함수
# =============================================================================

def route_to_agent(state: TeamState) -> str:
    """현재 에이전트에 따라 라우팅"""
    current = state.get("current_agent", "")

    if current == "researcher":
        return "researcher"
    elif current == "writer":
        return "writer"
    elif current == "critic":
        return "critic"
    elif current == "done":
        return "done"
    else:
        return "supervisor"


def after_agent(state: TeamState) -> str:
    """에이전트 작업 후 Supervisor로 돌아감"""
    return "supervisor"


# =============================================================================
# 그래프 생성
# =============================================================================

def create_multi_agent_team():
    """멀티 에이전트 팀 그래프 생성"""

    graph = StateGraph(TeamState)

    # 노드 추가
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    # 완료 노드 (아무것도 하지 않음)
    def done_node(state: TeamState) -> TeamState:
        return {}
    graph.add_node("done", done_node)

    # 엣지 추가
    graph.add_edge(START, "supervisor")

    # Supervisor에서 각 에이전트로 조건부 라우팅
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher": "researcher",
            "writer": "writer",
            "critic": "critic",
            "done": "done"
        }
    )

    # 각 에이전트 작업 후 Supervisor로 복귀
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("critic", "supervisor")

    # 완료
    graph.add_edge("done", END)

    # 컴파일
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 데모 실행
# =============================================================================

def run_demo():
    """데모 실행"""

    print("=" * 60)
    print("👥 Multi-Agent Team Demo")
    print("=" * 60)

    team = create_multi_agent_team()
    config = {"configurable": {"thread_id": "team_demo_1"}}

    # 작업 요청
    request = "LangGraph를 사용한 AI Agent 개발에 대한 블로그 포스트"

    print(f"\n📝 작업 요청: {request}")
    print("-" * 60)

    # 스트리밍으로 진행 상황 확인
    print("\n🔄 작업 진행 중...")

    for event in team.stream({
        "request": request,
        "current_agent": "",
        "research_result": "",
        "draft": "",
        "feedback": "",
        "final_output": "",
        "history": [],
        "revision_count": 0
    }, config=config, stream_mode="values"):

        # 히스토리에서 최신 항목 출력
        history = event.get("history", [])
        if history:
            print(f"   {history[-1]}")

    # 최종 결과 확인
    final_state = team.get_state(config)
    state_values = final_state.values

    print("\n" + "=" * 60)
    print("📊 작업 완료!")
    print("=" * 60)

    print(f"\n📋 작업 히스토리:")
    for item in state_values.get("history", []):
        print(f"   {item}")

    print(f"\n📝 최종 결과물:")
    print("-" * 40)
    final_output = state_values.get("final_output", state_values.get("draft", ""))
    print(final_output[:500] + "..." if len(final_output) > 500 else final_output)


# =============================================================================
# 인터랙티브 모드
# =============================================================================

def run_interactive():
    """인터랙티브 모드"""

    print("=" * 60)
    print("👥 Multi-Agent Team - Interactive Mode")
    print("=" * 60)
    print("\n작업을 요청하면 팀이 협업하여 처리합니다.")
    print("예: '파이썬 비동기 프로그래밍에 대한 기술 문서'")
    print("종료: /quit")
    print("-" * 60)

    team = create_multi_agent_team()
    session_count = 0

    while True:
        try:
            request = input("\n📝 작업 요청: ").strip()

            if not request:
                continue

            if request == "/quit":
                print("\n👋 팀 세션을 종료합니다.")
                break

            session_count += 1
            config = {"configurable": {"thread_id": f"interactive_{session_count}"}}

            print("\n🔄 팀이 작업 중...")

            result = team.invoke({
                "request": request,
                "current_agent": "",
                "research_result": "",
                "draft": "",
                "feedback": "",
                "final_output": "",
                "history": [],
                "revision_count": 0
            }, config=config)

            print("\n✅ 작업 완료!")
            print("\n📋 히스토리:")
            for item in result.get("history", []):
                print(f"   {item}")

            print("\n📝 결과물:")
            output = result.get("final_output", result.get("draft", ""))
            print(output[:300] + "..." if len(output) > 300 else output)

        except KeyboardInterrupt:
            print("\n\n👋 팀 세션을 종료합니다.")
            break


# =============================================================================
# 메인
# =============================================================================

def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive()
    else:
        run_demo()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
