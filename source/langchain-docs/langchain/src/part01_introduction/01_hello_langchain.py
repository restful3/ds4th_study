"""
================================================================================
LangChain AI Agent 마스터 교안
Part 1: AI Agent의 이해
================================================================================

파일명: 01_hello_langchain.py
난이도: ⭐☆☆☆☆ (입문)
예상 시간: 15분

📚 학습 목표:
  - LangChain의 기본 LLM 호출 이해
  - create_agent()로 첫 번째 Agent 만들기
  - Agent와 단순 LLM 호출의 차이 체험
  - 다양한 LLM 프로바이더 전환 이해
  - System Prompt로 Agent 성격 바꾸기

📖 교안 문서:
  • Part 1 개요: /docs/part01_introduction.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY (또는 다른 LLM 프로바이더 키)

🚀 실행 방법:
  python 01_hello_langchain.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

# .env 파일에서 환경변수 로드 (로컬 .env 우선)
load_dotenv(override=True)

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 src/.env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)


# ============================================================================
# 도구(Tool) 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 알려줍니다."""
    weather_data = {
        "서울": "맑음, 15°C",
        "뉴욕": "흐림, 8°C",
        "샌프란시스코": "화창함, 18°C",
        "도쿄": "비, 12°C",
        "런던": "안개, 10°C",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다. 예: '2 + 3 * 4'"""
    import ast
    import operator

    # eval() 대신 안전한 AST 기반 계산기 사용
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return allowed_operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_safe_eval(node.operand))
        else:
            raise ValueError(f"허용되지 않는 연산: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


# ============================================================================
# 예제 1: 가장 간단한 LLM 호출 (교안 1.1 매칭)
# ============================================================================

def example_1_simple_llm():
    """
    가장 기본적인 LLM 호출.
    Agent가 아닌 단순 텍스트 입력 → 텍스트 출력.
    📖 교안: Part 1 > 1. LangChain이란?
    """
    print("=" * 70)
    print("📌 예제 1: 가장 간단한 LLM 호출")
    print("=" * 70)

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-4.1-nano",  # 최저가 모델
        temperature=0.7,       # 창의성 조절 (0.0 = 결정적, 1.0 = 창의적)
    )

    # 메시지 생성 및 호출
    messages = [
        HumanMessage(content="안녕하세요! LangChain이 무엇인가요? 한 문장으로 답해주세요.")
    ]
    response = llm.invoke(messages)

    print(f"\n👤 사용자: {messages[0].content}")
    print(f"🤖 AI: {response.content}")

    # 한계 시연: 실시간 정보에 답할 수 없음
    messages2 = [
        HumanMessage(content="오늘 서울 날씨는 어때?")
    ]
    response2 = llm.invoke(messages2)

    print(f"\n👤 사용자: {messages2[0].content}")
    print(f"🤖 AI: {response2.content}")
    print("\n⚠️  단순 LLM은 실시간 정보를 알 수 없습니다. Agent가 필요한 이유!")


# ============================================================================
# 예제 2: create_agent()로 첫 번째 Agent 만들기 (교안 1.2 매칭)
# ============================================================================

def example_2_first_agent():
    """
    create_agent()로 도구를 사용하는 Agent 생성.
    📖 교안: Part 1 > 1.2 빠른 예제
    """
    print("\n" + "=" * 70)
    print("📌 예제 2: create_agent()로 첫 번째 Agent 만들기")
    print("=" * 70)

    # Agent 생성 - 도구를 사용할 수 있는 자율적 시스템
    agent = create_agent(
        model="openai:gpt-4.1-nano",
        tools=[get_weather],
        system_prompt="당신은 친절한 도우미입니다. 날씨 질문에 도구를 사용하여 답변하세요.",
    )

    # Agent 실행
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "샌프란시스코 날씨는?"}]}
    )

    # 결과 출력
    print(f"\n👤 사용자: 샌프란시스코 날씨는?")
    final_message = result["messages"][-1]
    print(f"🤖 Agent: {final_message.content}")

    print("\n💡 Agent는 get_weather 도구를 호출하여 실시간 정보를 가져왔습니다!")


# ============================================================================
# 예제 3: Agent vs 단순 LLM 비교 (교안 3.2 매칭)
# ============================================================================

def example_3_agent_vs_llm():
    """
    같은 질문에 대해 단순 LLM과 Agent의 차이를 직접 비교.
    📖 교안: Part 1 > 3. AI Agent란? > 3.2 Agent vs. 단순 LLM 호출
    """
    print("\n" + "=" * 70)
    print("📌 예제 3: Agent vs 단순 LLM 비교")
    print("=" * 70)

    question = "서울 날씨와 뉴욕 날씨를 비교해줘"

    # --- 방법 1: 단순 LLM 호출 ---
    print("\n🔹 [방법 1] 단순 LLM 호출:")
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    response = llm.invoke([HumanMessage(content=question)])
    content = response.content
    display = f"{content[:200]}..." if len(content) > 200 else content
    print(f"   🤖 AI: {display}")

    # --- 방법 2: Agent (도구 사용) ---
    print("\n🔹 [방법 2] Agent (도구 사용):")
    agent = create_agent(
        model="openai:gpt-4.1-nano",
        tools=[get_weather],
        system_prompt="날씨 정보가 필요하면 반드시 get_weather 도구를 사용하세요.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    final_message = result["messages"][-1]
    print(f"   🤖 Agent: {final_message.content}")

    # 비교 설명
    print("""
┌──────────────┬────────────────────┬────────────────────────┐
│     구분     │   단순 LLM 호출    │        Agent           │
├──────────────┼────────────────────┼────────────────────────┤
│   동작       │ 입력→출력 (1회)    │ 입력→추론→도구→재추론  │
│   외부 데이터│ 불가능             │ 가능 (API, DB 등)      │
│   실시간 정보│ 불가능             │ 가능 (도구로 조회)     │
│   복잡한 작업│ 불가능             │ 가능 (다단계 추론)     │
└──────────────┴────────────────────┴────────────────────────┘""")


# ============================================================================
# 예제 4: 다양한 LLM 프로바이더 전환 (교안 2.2 매칭)
# ============================================================================

def example_4_provider_switching():
    """
    LangChain의 핵심 장점: 프로바이더 전환이 코드 한 줄.
    📖 교안: Part 1 > 2.2 LangChain의 두 가지 핵심 목표 > 목표 1
    """
    print("\n" + "=" * 70)
    print("📌 예제 4: 다양한 LLM 프로바이더 전환")
    print("=" * 70)

    question = "AI Agent를 한 문장으로 설명해주세요."

    # OpenAI 사용
    print("\n🔹 [OpenAI] gpt-4.1-nano:")
    model_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
    response = model_openai.invoke([HumanMessage(content=question)])
    print(f"   🤖 {response.content}")

    # 다른 프로바이더 전환 안내
    print("""
💡 LangChain은 프로바이더 전환이 코드 한 줄입니다:

   # OpenAI 사용
   from langchain_openai import ChatOpenAI
   model = ChatOpenAI(model="gpt-4.1-nano")

   # Anthropic으로 교체 (코드 한 줄만 변경)
   from langchain_anthropic import ChatAnthropic
   model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

   # Google로 교체 (코드 한 줄만 변경)
   from langchain_google_genai import ChatGoogleGenerativeAI
   model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

   # create_agent에서도 문자열로 간단히 전환 가능
   agent = create_agent(model="openai:gpt-4.1-nano", ...)
   agent = create_agent(model="anthropic:claude-sonnet-4-5-20250929", ...)
""")


# ============================================================================
# 예제 5: System Prompt로 Agent 성격 바꾸기 (교안 과제 2 매칭)
# ============================================================================

def example_5_system_prompt():
    """
    system_prompt를 변경하여 Agent의 성격을 바꾸는 실험.
    📖 교안: Part 1 > 실습 과제 > 과제 2 (추가 도전)
    """
    print("\n" + "=" * 70)
    print("📌 예제 5: System Prompt로 Agent 성격 바꾸기")
    print("=" * 70)

    question = "서울 날씨 알려줘"

    # 성격 1: 친절한 도우미
    print("\n🔹 [성격 1] 친절한 도우미:")
    agent1 = create_agent(
        model="openai:gpt-4.1-nano",
        tools=[get_weather],
        system_prompt="당신은 친절하고 따뜻한 도우미입니다. 이모티콘을 사용하여 답변하세요.",
    )
    result1 = agent1.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"   🤖 {result1['messages'][-1].content}")

    # 성격 2: 간결한 비서
    print("\n🔹 [성격 2] 간결한 비서:")
    agent2 = create_agent(
        model="openai:gpt-4.1-nano",
        tools=[get_weather],
        system_prompt="당신은 간결한 비서입니다. 핵심만 짧게 답변하세요. 최대 2문장.",
    )
    result2 = agent2.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"   🤖 {result2['messages'][-1].content}")

    # 성격 3: 5살 아이에게 설명하는 선생님
    print("\n🔹 [성격 3] 어린이 선생님:")
    agent3 = create_agent(
        model="openai:gpt-4.1-nano",
        tools=[get_weather],
        system_prompt="당신은 5살 아이에게 설명하듯이 쉽게 설명하는 선생님입니다.",
    )
    result3 = agent3.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"   🤖 {result3['messages'][-1].content}")

    print("\n💡 같은 도구, 같은 질문이지만 system_prompt에 따라 답변 스타일이 달라집니다!")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 1: AI Agent의 이해 - Hello LangChain!")
    print("\n")

    # 예제 1: 단순 LLM 호출 (Agent 없이)
    example_1_simple_llm()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 2: create_agent()로 첫 번째 Agent 만들기
    example_2_first_agent()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 3: Agent vs 단순 LLM 비교
    example_3_agent_vs_llm()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 4: 다양한 LLM 프로바이더 전환
    example_4_provider_switching()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 5: System Prompt로 Agent 성격 바꾸기
    example_5_system_prompt()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 1 예제를 모두 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 02_environment_check.py - 환경 설정 확인")
    print("  2. get_weather 도구의 docstring을 바꿔보세요 (Agent 응답이 달라집니다)")
    print("  3. calculate 도구를 Agent에 추가해보세요")
    print("  4. Part 2: LangChain 기초로 이동")
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
# 1. Temperature 파라미터:
#    - 0.0: 항상 같은 답변 (결정적)
#    - 0.7: 균형잡힌 창의성 (기본값)
#    - 1.0: 매우 창의적 (다양한 답변)
#
# 2. create_agent() 모델 지정 방식:
#    - 문자열: create_agent(model="openai:gpt-4.1-nano", ...)
#    - 객체:  create_agent(model=ChatOpenAI(model="gpt-4.1-nano"), ...)
#
# 3. 도구(Tool) 정의:
#    - @tool 데코레이터 사용
#    - docstring이 도구 설명이 됨 (Agent가 이를 보고 도구 사용 판단)
#    - docstring을 바꾸면 Agent 행동이 달라짐!
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "OPENAI_API_KEY not found"
# 해결: src/.env 파일을 확인하고 API 키를 설정하세요
#
# 문제: "Rate limit exceeded"
# 해결: API 사용량을 확인하거나 더 저렴한 모델(gpt-4.1-nano) 사용
#
# 문제: Agent가 도구를 호출하지 않음
# 해결: system_prompt에 도구 사용을 명시하거나 도구의 docstring을 개선하세요
#
# ============================================================================
