"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: 첫 번째 Agent
================================================================================

파일명: 01_basic_agent.py
난이도: ⭐⭐☆☆☆ (초급)
예상 시간: 30분

📚 학습 목표:
  - create_agent() API 이해
  - 첫 번째 Agent 만들기
  - Agent의 동작 원리 이해

📖 공식 문서:
  • Agents: /official/06-agents.md
  • Quickstart: /official/03-quickstart.md

🚀 실행 방법:
  python 01_basic_agent.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent, tool
from langchain_openai import ChatOpenAI

load_dotenv()

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 알려줍니다.

    Args:
        city: 도시 이름 (예: 서울, 부산, 뉴욕)
    """
    # 실제로는 날씨 API를 호출하지만, 여기서는 더미 데이터
    weather_info = {
        "서울": "맑음, 기온 22도, 습도 60%",
        "부산": "흐림, 기온 20도, 습도 70%",
        "뉴욕": "비, 기온 15도, 습도 85%",
        "파리": "맑음, 기온 18도, 습도 55%",
        "도쿄": "구름 많음, 기온 19도, 습도 65%",
    }

    weather = weather_info.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")
    return f"{city}의 현재 날씨: {weather}"


@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다.

    Args:
        expression: 계산식 (예: "2 + 2", "10 * 5")
    """
    try:
        # 보안 주의: 실제 프로덕션에서는 eval 사용 금지!
        # 여기서는 교육 목적으로만 사용
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


# ============================================================================
# 예제 1: 가장 간단한 Agent
# ============================================================================

def example_1_simple_agent():
    """가장 기본적인 Agent 생성 및 실행"""
    print("=" * 70)
    print("📌 예제 1: 가장 간단한 Agent")
    print("=" * 70)

    # 1. LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 2. 도구 리스트 정의
    tools = [get_weather, calculate]

    # 3. Agent 생성
    agent = create_agent(
        model=model,
        tools=tools,
    )

    # 4. Agent 실행
    print("\n🤖 Agent 생성 완료!")
    print(f"📋 사용 가능한 도구: {[tool.name for tool in tools]}")

    # 테스트 질문
    question = "서울 날씨는 어때?"

    print(f"\n👤 사용자: {question}")
    print("🤔 Agent가 생각하고 있습니다...\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # 응답 출력
    final_message = result["messages"][-1]
    print(f"🤖 Agent: {final_message.content}\n")


# ============================================================================
# 예제 2: System Prompt가 있는 Agent
# ============================================================================

def example_2_agent_with_system_prompt():
    """System Prompt로 Agent의 성격 지정"""
    print("=" * 70)
    print("📌 예제 2: System Prompt가 있는 Agent")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [get_weather, calculate]

    # System Prompt로 Agent의 역할 정의
    system_prompt = """
당신은 친절하고 유용한 날씨 및 계산 도우미입니다.
사용자의 질문에 정확하게 답변하되, 항상 친근한 톤으로 대화하세요.
답변은 간결하고 명확하게 작성해주세요.
    """

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    print("\n🎭 Agent 역할: 친절한 날씨 및 계산 도우미")

    # 여러 질문 테스트
    questions = [
        "부산 날씨 알려줘",
        "100 곱하기 25는?",
        "뉴욕과 파리 중 어디가 더 따뜻해?",
    ]

    for question in questions:
        print(f"\n👤 사용자: {question}")

        result = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        final_message = result["messages"][-1]
        print(f"🤖 Agent: {final_message.content}")

    print()


# ============================================================================
# 예제 3: Agent의 도구 사용 과정 관찰
# ============================================================================

def example_3_agent_reasoning():
    """Agent가 도구를 사용하는 과정 관찰"""
    print("=" * 70)
    print("📌 예제 3: Agent의 추론 과정 관찰")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="도구를 사용할 때마다 무엇을 하는지 설명하세요."
    )

    question = "서울과 도쿄의 평균 기온은 몇 도인가요?"

    print(f"\n👤 사용자: {question}")
    print("\n🔍 Agent의 추론 과정:\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # 모든 메시지 출력 (추론 과정 확인)
    for msg in result["messages"]:
        role = msg.__class__.__name__
        if role == "HumanMessage":
            print(f"   👤 사용자: {msg.content}")
        elif role == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"   🔧 도구 호출: {tool_call['name']}({tool_call['args']})")
            elif msg.content:
                print(f"   🤖 Agent: {msg.content}")
        elif role == "ToolMessage":
            print(f"   ✅ 도구 결과: {msg.content}")

    print()


# ============================================================================
# 예제 4: 복잡한 질문 처리
# ============================================================================

def example_4_complex_question():
    """여러 도구를 연속으로 사용하는 복잡한 질문"""
    print("=" * 70)
    print("📌 예제 4: 복잡한 질문 처리")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate]

    agent = create_agent(
        model=model,
        tools=tools,
    )

    complex_questions = [
        "서울 날씨를 확인하고, 기온이 20도 이상이면 '더워요', 아니면 '시원해요'라고 알려줘",
        "10 더하기 20을 계산한 다음, 그 결과에 3을 곱해줘",
    ]

    for question in complex_questions:
        print(f"\n👤 사용자: {question}")

        result = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        final_message = result["messages"][-1]
        print(f"🤖 Agent: {final_message.content}")

    print()


# ============================================================================
# 예제 5: Agent vs 일반 LLM 비교
# ============================================================================

def example_5_agent_vs_llm():
    """Agent와 일반 LLM의 차이 비교"""
    print("=" * 70)
    print("📌 예제 5: Agent vs 일반 LLM 비교")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    question = "서울의 현재 날씨는?"

    # 일반 LLM (도구 없음)
    print("\n🔹 일반 LLM (도구 없음):")
    print(f"   👤 사용자: {question}")

    llm_response = model.invoke(question)
    print(f"   🤖 LLM: {llm_response.content}")

    # Agent (도구 있음)
    print("\n🔹 Agent (날씨 도구 있음):")
    print(f"   👤 사용자: {question}")

    tools = [get_weather]
    agent = create_agent(model=model, tools=tools)

    agent_response = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    final_message = agent_response["messages"][-1]
    print(f"   🤖 Agent: {final_message.content}")

    print("\n💡 Agent는 실제 도구를 사용하여 정확한 정보를 제공합니다!\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n🎓 Part 3: 첫 번째 Agent\n")

    example_1_simple_agent()
    input("⏎ 계속하려면 Enter...")

    example_2_agent_with_system_prompt()
    input("⏎ 계속하려면 Enter...")

    example_3_agent_reasoning()
    input("⏎ 계속하려면 Enter...")

    example_4_complex_question()
    input("⏎ 계속하려면 Enter...")

    example_5_agent_vs_llm()

    print("=" * 70)
    print("🎉 첫 번째 Agent 학습 완료!")
    print("=" * 70)
    print("\n💡 주요 학습 내용:")
    print("   ✅ create_agent() API 사용법")
    print("   ✅ Agent에 도구 연결하기")
    print("   ✅ System Prompt로 Agent 성격 지정")
    print("   ✅ Agent의 추론 과정 이해")
    print("\n📖 다음: 02_weather_agent.py - 실전 날씨 Agent")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 핵심 포인트
# ============================================================================
#
# 1. create_agent()의 주요 파라미터:
#    - model: 사용할 LLM
#    - tools: Agent가 사용할 도구 리스트
#    - system_prompt: Agent의 역할 정의 (선택)
#
# 2. Agent의 작동 방식 (ReAct 패턴):
#    Thought (생각) → Action (도구 호출) → Observation (결과 관찰) → 반복
#
# 3. Agent vs 일반 LLM:
#    - LLM: 학습된 지식만 사용
#    - Agent: 도구를 사용하여 실시간 정보 접근 및 작업 수행
#
# 4. 좋은 도구 만들기:
#    - 명확한 docstring
#    - 적절한 파라미터 타입
#    - 에러 핸들링
#
# ============================================================================
