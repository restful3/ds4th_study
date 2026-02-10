"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: 첫 번째 Agent 만들기
================================================================================

파일명: 02_weather_agent.py
난이도: ⭐⭐☆☆☆ (초급)
예상 시간: 20분

📚 학습 목표:
  - 실전 날씨 Agent 구현하기
  - get_weather_for_location과 get_user_location 도구 사용
  - System Prompt로 Agent 성격 정의하기
  - 런타임 컨텍스트를 활용한 개인화 구현

📖 공식 문서:
  • Quickstart: /official/03-quickstart.md
  • Agents: /official/06-agents.md

📄 교안 문서:
  • Part 3 개요: /docs/part03_first_agent.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 02_weather_agent.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime

# ============================================================================
# 환경 설정
# ============================================================================

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: 간단한 날씨 도구
# ============================================================================

def example_1_simple_weather_tool():
    """가장 기본적인 날씨 도구와 Agent"""
    print("=" * 70)
    print("📌 예제 1: 간단한 날씨 도구")
    print("=" * 70)

    @tool
    def get_weather(city: str) -> str:
        """주어진 도시의 현재 날씨를 조회합니다.

        Args:
            city: 날씨를 조회할 도시 이름 (예: 서울, 부산, 뉴욕)
        """
        # 실제로는 날씨 API를 호출하지만, 여기서는 더미 데이터 사용
        weather_data = {
            "서울": "맑음, 22°C, 습도 60%",
            "부산": "흐림, 20°C, 습도 70%",
            "뉴욕": "비, 15°C, 습도 85%",
            "파리": "맑음, 18°C, 습도 55%",
        }
        return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

    # LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Agent 생성
    agent = create_agent(model=model, tools=[get_weather])

    # Agent 실행
    print("\n👤 사용자: 서울 날씨 알려줘")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "서울 날씨 알려줘"}]
    })

    print(f"🤖 Agent: {result['messages'][-1].content}")

    print("\n💡 포인트:")
    print("  - @tool 데코레이터로 도구를 정의했습니다")
    print("  - Agent가 자동으로 get_weather 도구를 호출했습니다")
    print("  - 도구 결과를 바탕으로 자연스러운 답변을 생성했습니다\n")


# ============================================================================
# 예제 2: 두 개의 도구 - 위치 파악과 날씨 조회
# ============================================================================

def example_2_two_tools():
    """get_weather_for_location과 get_user_location 도구"""
    print("=" * 70)
    print("📌 예제 2: 두 개의 도구 - 위치 파악 + 날씨 조회")
    print("=" * 70)

    # 도구 1: 날씨 조회
    @tool
    def get_weather_for_location(city: str) -> str:
        """주어진 도시의 날씨를 조회합니다.

        Args:
            city: 날씨를 조회할 도시 이름
        """
        weather_data = {
            "서울": "맑음, 22°C, 습도 60%",
            "부산": "흐림, 20°C, 습도 70%",
            "뉴욕": "비, 15°C, 습도 85%",
            "플로리다": "맑음, 28°C, 습도 75%",
        }
        return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

    # 컨텍스트 스키마 정의
    @dataclass
    class Context:
        """런타임 컨텍스트 스키마"""
        user_id: str

    # 도구 2: 사용자 위치 조회
    @tool
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        """현재 사용자의 위치를 조회합니다.

        ToolRuntime을 통해 런타임 컨텍스트에 접근합니다.
        """
        user_id = runtime.context.user_id

        # 실제로는 DB나 IP 기반 위치 조회
        location_map = {
            "1": "서울",
            "2": "부산",
            "3": "뉴욕",
        }

        return location_map.get(user_id, "서울")  # 기본값: 서울

    # LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Agent 생성
    agent = create_agent(
        model=model,
        tools=[get_weather_for_location, get_user_location],
        context_schema=Context,
    )

    # Agent 실행 (user_id=1)
    print("\n👤 사용자 1: 밖에 날씨 어때?")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "밖에 날씨 어때?"}]},
        context=Context(user_id="1")
    )

    print(f"🤖 Agent: {result['messages'][-1].content}")

    print("\n💡 포인트:")
    print("  - Agent가 '밖에'라는 표현에서 위치 파악이 필요함을 인지했습니다")
    print("  - get_user_location → get_weather_for_location 순서로 도구를 사용했습니다")
    print("  - ToolRuntime[Context]로 런타임 컨텍스트에 접근했습니다\n")


# ============================================================================
# 예제 3: System Prompt로 Agent 성격 정의
# ============================================================================

def example_3_system_prompt():
    """System Prompt로 재치있는 날씨 Agent 만들기"""
    print("=" * 70)
    print("📌 예제 3: System Prompt로 재치있는 날씨 Agent")
    print("=" * 70)

    # 도구 정의
    @tool
    def get_weather_for_location(city: str) -> str:
        """주어진 도시의 날씨를 조회합니다."""
        weather_data = {
            "서울": "맑음, 22°C, 습도 60%",
            "부산": "흐림, 20°C, 습도 70%",
            "플로리다": "맑음, 28°C, 습도 75%",
        }
        return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

    @dataclass
    class Context:
        user_id: str

    @tool
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        """현재 사용자의 위치를 조회합니다."""
        user_id = runtime.context.user_id
        location_map = {"1": "플로리다", "2": "부산"}
        return location_map.get(user_id, "서울")

    # System Prompt 정의
    SYSTEM_PROMPT = """당신은 전문 날씨 예보관이며, 말장난을 좋아합니다.

사용 가능한 도구:
- get_weather_for_location: 특정 도시의 날씨 조회
- get_user_location: 사용자의 현재 위치 조회

사용자가 날씨를 물어보면 위치를 확인하세요.
사용자가 "여기", "현재", "밖에" 같은 표현을 쓰면 get_user_location 도구를 사용하세요.

답변은 친근하고 재치있게, 날씨 관련 말장난을 섞어서 작성하세요.
예: "화창한 날씨", "구름이 잔뜩 찌푸렸네요", "햇살이 '빛-나'게 웃고 있어요" """

    # LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # 창의성을 위해 temperature 높임

    # Agent 생성 (System Prompt 포함)
    agent = create_agent(
        model=model,
        tools=[get_weather_for_location, get_user_location],
        context_schema=Context,
        system_prompt=SYSTEM_PROMPT,
    )

    # Agent 실행
    print("\n👤 사용자 (플로리다): 밖에 날씨 어때?")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "밖에 날씨 어때?"}]},
        context=Context(user_id="1")  # user_id=1 → 플로리다
    )

    print(f"🤖 Agent: {result['messages'][-1].content}")

    print("\n💡 포인트:")
    print("  - System Prompt로 Agent의 성격을 '재치있는 예보관'으로 정의했습니다")
    print("  - temperature=0.7로 창의적인 말장난을 가능하게 했습니다")
    print("  - 같은 도구, 다른 프롬프트로 완전히 다른 성격의 Agent가 됩니다\n")


# ============================================================================
# 예제 4: Agent 실행 과정 분석
# ============================================================================

def example_4_execution_analysis():
    """Agent의 실행 과정을 단계별로 분석"""
    print("=" * 70)
    print("📌 예제 4: Agent 실행 과정 상세 분석")
    print("=" * 70)

    # 도구 정의
    @tool
    def get_weather_for_location(city: str) -> str:
        """주어진 도시의 날씨를 조회합니다."""
        weather_data = {
            "서울": "맑음, 22°C, 습도 60%",
            "부산": "흐림, 20°C, 습도 70%",
        }
        return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

    @dataclass
    class Context:
        user_id: str

    @tool
    def get_user_location(runtime: ToolRuntime[Context]) -> str:
        """현재 사용자의 위치를 조회합니다."""
        user_id = runtime.context.user_id
        location_map = {"1": "서울", "2": "부산"}
        return location_map.get(user_id, "서울")

    # Agent 생성
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather_for_location, get_user_location],
        context_schema=Context,
        system_prompt="당신은 날씨 정보를 제공하는 친절한 Agent입니다.",
    )

    # Agent 실행
    print("\n👤 사용자: 현재 날씨 알려줘")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "현재 날씨 알려줘"}]},
        context=Context(user_id="1")
    )

    # 실행 과정 분석
    print("\n🔍 실행 과정 분석:\n")
    for i, msg in enumerate(result["messages"], 1):
        role = msg.__class__.__name__

        if role == "HumanMessage":
            print(f"[Step {i}] 👤 사용자 입력")
            print(f"         '{msg.content}'")

        elif role == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\n[Step {i}] 🤔 Agent 추론 + 도구 호출")
                for tc in msg.tool_calls:
                    print(f"         도구: {tc['name']}({tc['args']})")
            else:
                print(f"\n[Step {i}] 💡 최종 답변")
                print(f"         {msg.content}")

        elif role == "ToolMessage":
            print(f"\n[Step {i}] 👀 도구 실행 결과")
            print(f"         결과: {msg.content}")

    print("\n💡 포인트:")
    print("  - Agent의 실행 과정은 messages 리스트로 확인할 수 있습니다")
    print("  - HumanMessage → AIMessage (tool_calls) → ToolMessage → AIMessage (final)")
    print("  - 각 단계를 추적하여 디버깅할 수 있습니다\n")


# ============================================================================
# 예제 5: 동일 Agent로 여러 쿼리 실행
# ============================================================================

def example_5_multiple_queries():
    """같은 Agent를 여러 번 호출하기"""
    print("=" * 70)
    print("📌 예제 5: 동일 Agent로 여러 쿼리 실행")
    print("=" * 70)

    # 도구 정의
    @tool
    def get_weather_for_location(city: str) -> str:
        """주어진 도시의 날씨를 조회합니다."""
        weather_data = {
            "서울": "맑음, 22°C",
            "부산": "흐림, 20°C",
            "제주": "비, 18°C",
            "대전": "맑음, 21°C",
        }
        return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

    # Agent 생성
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather_for_location],
        system_prompt="당신은 간결하고 명확하게 날씨를 알려주는 Agent입니다.",
    )

    # 여러 쿼리 실행
    queries = [
        "서울 날씨는?",
        "부산과 제주 중 어디가 더 따뜻해?",
        "대전 날씨 알려줘",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[쿼리 {i}]")
        print(f"👤 사용자: {query}")

        result = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })

        print(f"🤖 Agent: {result['messages'][-1].content}")

    print("\n💡 포인트:")
    print("  - 같은 Agent를 여러 번 재사용할 수 있습니다")
    print("  - 각 호출은 독립적이며 상태를 공유하지 않습니다")
    print("  - 대화 기록을 유지하려면 checkpointer가 필요합니다 (Part 4에서 학습)\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 3: 첫 번째 Agent 만들기 - 날씨 Agent")
    print("\n")

    # 모든 예제 실행
    example_1_simple_weather_tool()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_two_tools()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_system_prompt()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_execution_analysis()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_multiple_queries()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 날씨 Agent 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 03_react_pattern.py - ReAct 패턴 학습")
    print("  2. 04_custom_prompt.py - System Prompt 커스터마이징")
    print("  3. 05_streaming_agent.py - 실시간 스트리밍 구현")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 런타임 컨텍스트 (ToolRuntime):
#    - 도구가 실행 시점의 정보에 접근할 수 있게 해줍니다
#    - user_id, session_id, 시간 등을 전달 가능
#    - 타입 힌트로 컨텍스트 구조를 명시: ToolRuntime[Context]
#
# 2. System Prompt 작성 팁:
#    - 역할 정의: "당신은 ~입니다"
#    - 도구 설명: 각 도구의 용도 명시
#    - 행동 지침: "~하면 ~하세요"
#    - 응답 스타일: "친근하게", "전문적으로" 등
#
# 3. Agent 디버깅:
#    result["messages"]를 출력하여 모든 중간 단계를 확인하세요
#    각 메시지의 타입(HumanMessage, AIMessage, ToolMessage)을 체크하세요
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: Agent가 도구를 사용하지 않고 임의로 답변함
# 해결:
#   1. 도구의 docstring을 더 명확하게 작성
#   2. System Prompt에 "반드시 도구를 사용하세요" 추가
#   3. temperature를 낮춰서 일관성 향상 (0.0~0.3)
#
# 문제: ToolRuntime 사용 시 "context" 오류
# 해결:
#   1. Agent 생성 시 context_schema=Context 명시
#   2. invoke() 호출 시 context=Context(...) 전달
#
# 문제: 도구가 여러 번 호출됨 (무한 루프)
# 해결:
#   1. 도구가 명확한 결과를 반환하는지 확인
#   2. config={"recursion_limit": 10}으로 최대 반복 제한
#
# ============================================================================
