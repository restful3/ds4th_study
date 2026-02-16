"""
[Chapter 10] ReAct Agent 구현

📝 설명:
    ReAct(Reasoning + Acting)는 LLM이 추론하고 행동하는 과정을 반복하여
    복잡한 작업을 수행하는 Agent 패턴입니다.
    LangGraph를 사용하여 ReAct Agent를 구현합니다.

🎯 학습 목표:
    - ReAct 패턴 이해
    - Agent 루프 구현
    - should_continue 조건 함수 작성
    - create_react_agent 활용
    - 실전 Agent 예제

📚 관련 문서:
    - docs/Part3-Agent/10-react-agent.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#react-agent

💻 실행 방법:
    python -m src.part3_agent.10_react_agent

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
"""

import os
from typing import Literal
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, create_react_agent


# =============================================================================
# 1. 도구 정의
# =============================================================================

@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수학 표현식 (예: "2 + 3 * 4")

    Returns:
        계산 결과
    """
    try:
        # 안전한 계산을 위해 제한된 eval 사용
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "오류: 허용되지 않은 문자가 포함되어 있습니다."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다.

    Returns:
        현재 시간 문자열
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_knowledge(topic: str) -> str:
    """특정 주제에 대한 지식을 검색합니다.

    Args:
        topic: 검색할 주제

    Returns:
        관련 정보
    """
    # 시뮬레이션된 지식베이스
    knowledge = {
        "python": "Python은 1991년 귀도 반 로섬이 만든 프로그래밍 언어입니다.",
        "langgraph": "LangGraph는 LLM 애플리케이션을 위한 그래프 기반 프레임워크입니다.",
        "agent": "AI Agent는 자율적으로 작업을 수행하는 시스템입니다.",
    }

    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower:
            return value

    return f"'{topic}'에 대한 정보를 찾을 수 없습니다."


# =============================================================================
# 2. 수동 ReAct Agent 구현
# =============================================================================

def create_manual_react_agent():
    """수동으로 ReAct Agent 구현"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    # 도구 목록
    tools = [calculate, get_current_time, search_knowledge]

    # LLM with tools
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # ToolNode
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState) -> MessagesState:
        """모델 호출 노드"""
        # 시스템 메시지 추가
        system_message = SystemMessage(content="""당신은 도움이 되는 AI 어시스턴트입니다.
사용자의 질문에 답하기 위해 필요한 경우 도구를 사용하세요.
도구를 사용할 때는 한 번에 하나씩 사용하고, 결과를 확인한 후 다음 단계를 결정하세요.""")

        messages = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """도구 호출 여부 확인"""
        last_message = state["messages"][-1]

        # AIMessage이고 tool_calls가 있으면 도구 실행
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # 그렇지 않으면 종료
        return "end"

    # 그래프 구성
    graph = StateGraph(MessagesState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_edge("tools", "agent")  # 도구 실행 후 다시 에이전트로

    return graph.compile()


def run_manual_react_example():
    """수동 ReAct Agent 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 수동 구현 ReAct Agent")
    print("=" * 60)

    app = create_manual_react_agent()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    # 테스트 질문들
    questions = [
        "3 + 5 * 2를 계산해주세요.",
        "현재 시간이 몇 시인가요?",
        "Python에 대해 알려주세요."
    ]

    for question in questions:
        print(f"\n📝 질문: {question}")

        result = app.invoke({
            "messages": [HumanMessage(content=question)]
        })

        # 최종 응답 출력
        final_response = result["messages"][-1]
        print(f"🤖 응답: {final_response.content[:200]}...")


# =============================================================================
# 3. create_react_agent 사용 (LangGraph 내장)
# =============================================================================

def create_prebuilt_react_agent():
    """내장 create_react_agent 사용"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    tools = [calculate, get_current_time, search_knowledge]

    # 한 줄로 ReAct Agent 생성!
    agent = create_react_agent(llm, tools)

    return agent


def run_prebuilt_react_example():
    """내장 ReAct Agent 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: create_react_agent (내장 함수)")
    print("=" * 60)

    agent = create_prebuilt_react_agent()

    if agent is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    question = "LangGraph에 대해 설명하고, 2 + 2 * 3을 계산해주세요."
    print(f"\n📝 질문: {question}")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    # 전체 대화 흐름 출력
    print(f"\n🔄 Agent 실행 흐름:")
    for i, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"   {i+1}. [{msg_type}] 도구 호출: {[tc['name'] for tc in msg.tool_calls]}")
        elif msg_type == "ToolMessage":
            print(f"   {i+1}. [{msg_type}] 결과: {msg.content[:50]}...")
        else:
            content = msg.content[:80] if msg.content else "(내용 없음)"
            print(f"   {i+1}. [{msg_type}] {content}...")


# =============================================================================
# 4. 시스템 프롬프트가 있는 Agent
# =============================================================================

def create_agent_with_system_prompt():
    """시스템 프롬프트가 있는 Agent 생성"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    tools = [calculate, get_current_time, search_knowledge]

    system_prompt = """당신은 친절한 한국어 AI 튜터입니다.
사용자의 질문에 교육적인 방식으로 답변하세요.
필요한 경우 도구를 사용하여 정확한 정보를 제공하세요.
항상 한국어로 응답하세요."""

    # 시스템 프롬프트와 함께 Agent 생성
    agent = create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt
    )

    return agent


def run_system_prompt_example():
    """시스템 프롬프트 Agent 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: 시스템 프롬프트가 있는 Agent")
    print("=" * 60)

    agent = create_agent_with_system_prompt()

    if agent is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    question = "제곱근 계산하는 방법을 알려주세요. 16의 제곱근도 구해주세요."
    print(f"\n📝 질문: {question}")

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    final_response = result["messages"][-1]
    print(f"\n🤖 응답:\n{final_response.content}")


# =============================================================================
# 5. 스트리밍 Agent
# =============================================================================

def run_streaming_agent_example():
    """스트리밍 Agent 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: 스트리밍 Agent")
    print("=" * 60)

    agent = create_prebuilt_react_agent()

    if agent is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    question = "현재 시간을 알려주고, 5 * 8을 계산해주세요."
    print(f"\n📝 질문: {question}")
    print(f"\n🔄 실시간 진행 상황:")

    # 스트리밍으로 진행 상황 확인
    for event in agent.stream({"messages": [HumanMessage(content=question)]}):
        for node_name, output in event.items():
            print(f"\n   [{node_name}]")
            if "messages" in output:
                for msg in output["messages"]:
                    msg_type = type(msg).__name__
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"      🔧 도구 호출: {tc['name']}({tc['args']})")
                    elif msg_type == "ToolMessage":
                        print(f"      📤 도구 결과: {msg.content}")
                    elif msg.content:
                        print(f"      💬 {msg.content[:100]}...")


# =============================================================================
# 6. 최대 반복 횟수 제한
# =============================================================================

def create_limited_agent():
    """반복 횟수가 제한된 Agent 생성"""
    load_dotenv()

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    tools = [calculate, search_knowledge]

    # recursion_limit으로 최대 반복 제한
    agent = create_react_agent(llm, tools)

    return agent


def run_limited_agent_example():
    """반복 제한 Agent 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 5: 최대 반복 횟수 제한")
    print("=" * 60)

    agent = create_limited_agent()

    if agent is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    question = "1+1, 2+2, 3+3, 4+4, 5+5를 각각 계산해주세요."
    print(f"\n📝 질문: {question}")

    try:
        # recursion_limit 설정
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": 10}  # 최대 10번 반복
        )
        print(f"\n✅ 완료: {len(result['messages'])}개 메시지")
    except Exception as e:
        print(f"\n⚠️  제한 도달: {e}")


# =============================================================================
# 7. ReAct 패턴 설명
# =============================================================================

def explain_react_pattern():
    """ReAct 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 ReAct 패턴 정리")
    print("=" * 60)

    print("""
ReAct = Reasoning + Acting

    ┌─────────────────────────────────────────────┐
    │              ReAct Agent 루프               │
    │                                             │
    │   ┌──────────┐    tool_calls?    ┌───────┐ │
    │   │   LLM    │────────YES───────▶│ Tools │ │
    │   │ (Agent)  │◀───────────────────│ (Act) │ │
    │   └────┬─────┘                   └───────┘ │
    │        │ NO                                 │
    │        ▼                                    │
    │     [END]                                   │
    │                                             │
    └─────────────────────────────────────────────┘

ReAct의 특징:

1. Reasoning (추론)
   - LLM이 현재 상황을 분석
   - 다음에 취할 행동 결정
   - 필요한 도구 선택

2. Acting (행동)
   - 선택한 도구 실행
   - 결과 획득
   - 상태 업데이트

3. Loop (반복)
   - 목표 달성까지 추론-행동 반복
   - 도구 호출이 없으면 종료
   - 최대 반복 횟수 제한 가능

create_react_agent 사용법:

    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model=llm,                    # LLM 인스턴스
        tools=[tool1, tool2],         # 도구 리스트
        state_modifier=system_prompt  # 시스템 프롬프트 (선택)
    )

    result = agent.invoke({"messages": [HumanMessage(content="...")]})

장점:
- 복잡한 작업을 단계별로 해결
- 외부 도구와 연동 가능
- 추론 과정 추적 가능

주의점:
- 무한 루프 방지 (recursion_limit)
- 도구 에러 처리
- 비용 관리 (API 호출 수)
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 10] ReAct Agent 구현")
    print("=" * 60)

    load_dotenv()

    # 예제 실행
    run_manual_react_example()
    run_prebuilt_react_example()
    run_system_prompt_example()
    run_streaming_agent_example()
    run_limited_agent_example()

    # 개념 정리
    explain_react_pattern()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 11_multi_agent.py (Multi-Agent)")
    print("=" * 60)


if __name__ == "__main__":
    main()
