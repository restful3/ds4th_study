"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: LangChain 기초
================================================================================

파일명: 05_tool_calling.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 25분

📚 학습 목표:
  - bind_tools()로 LLM에 도구 연결하기
  - Tool call 요청 검사 및 이해
  - Tool call 실행하기
  - 여러 도구를 한번에 호출하는 케이스
  - Tool call 에러 핸들링 방법

📖 공식 문서:
  • Tool Calling: /official/09-tools.md
  • Agents: /official/11-agents.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 05_tool_calling.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)


# ============================================================================
# 도구 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 조회합니다.

    Args:
        city: 도시 이름 (예: 서울, 부산, 뉴욕)
    """
    weather_data = {
        "서울": "맑음, 22도",
        "부산": "흐림, 20도",
        "뉴욕": "비, 15도",
        "도쿄": "맑음, 18도",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")


@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 2", "10 * 5")
    """
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색어
    """
    # 실제로는 검색 API를 호출
    return f"'{query}'에 대한 검색 결과: LangChain은 LLM 애플리케이션 개발 프레임워크입니다."


# ============================================================================
# 예제 1: bind_tools()로 도구 연결하기
# ============================================================================

def example_1_bind_tools():
    """LLM에 도구를 연결하는 기본 방법"""
    print("=" * 70)
    print("📌 예제 1: bind_tools()로 도구 연결하기")
    print("=" * 70)

    # LLM 초기화
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 도구를 LLM에 연결
    model_with_tools = model.bind_tools([get_weather, calculate])

    print("\n🔧 연결된 도구:")
    print(f"   - {get_weather.name}: {get_weather.description}")
    print(f"   - {calculate.name}: {calculate.description}")

    # LLM 호출 (도구가 필요한 질문)
    response = model_with_tools.invoke("서울의 날씨는 어때?")

    print(f"\n📩 응답 타입: {type(response).__name__}")
    print(f"📩 응답 내용: {response.content}")

    # Tool call 요청 확인
    if response.tool_calls:
        print(f"\n🛠️  도구 호출 요청:")
        for tool_call in response.tool_calls:
            print(f"   도구: {tool_call['name']}")
            print(f"   인자: {tool_call['args']}")
    else:
        print("\n⚠️  도구 호출 요청 없음")

    print("\n💡 LLM이 필요한 도구를 자동으로 선택!\n")


# ============================================================================
# 예제 2: Tool call 요청 상세 검사
# ============================================================================

def example_2_examine_tool_calls():
    """Tool call 요청의 구조 이해하기"""
    print("=" * 70)
    print("📌 예제 2: Tool call 요청 상세 검사")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools([get_weather, calculate, search_web])

    # 다양한 질문으로 테스트
    questions = [
        "서울 날씨 알려줘",
        "25 곱하기 4는 얼마야?",
        "LangChain이 뭐야?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"질문 {i}: {question}")
        print('='*70)

        response = model_with_tools.invoke(question)

        if response.tool_calls:
            print(f"✅ 도구 호출 요청됨:")
            for tool_call in response.tool_calls:
                print(f"\n   🔧 도구명: {tool_call['name']}")
                print(f"   📝 ID: {tool_call['id']}")
                print(f"   📋 인자: {tool_call['args']}")
        else:
            print(f"⚠️  도구 호출 없음 (직접 답변)")
            print(f"   응답: {response.content}")

    print("\n💡 LLM이 질문에 따라 적절한 도구를 자동 선택!\n")


# ============================================================================
# 예제 3: Tool call 실행하기
# ============================================================================

def example_3_execute_tool_calls():
    """Tool call을 실제로 실행하기"""
    print("=" * 70)
    print("📌 예제 3: Tool call 실행하기")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate, search_web]
    model_with_tools = model.bind_tools(tools)

    # 도구 이름으로 매핑
    tools_map = {tool.name: tool for tool in tools}

    # 사용자 질문
    user_question = "서울의 날씨는 어때?"
    print(f"\n👤 사용자: {user_question}")

    # 1단계: LLM이 도구 호출 요청
    messages = [HumanMessage(content=user_question)]
    response = model_with_tools.invoke(messages)

    print(f"\n🤖 LLM 응답:")
    if response.tool_calls:
        print(f"   도구 호출 요청: {response.tool_calls[0]['name']}")

        # 2단계: 도구 실행
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"\n🔧 도구 실행: {tool_name}({tool_args})")

            # 도구 실행
            selected_tool = tools_map[tool_name]
            tool_result = selected_tool.invoke(tool_args)

            print(f"📤 도구 결과: {tool_result}")

            # 3단계: 도구 결과를 LLM에 전달
            messages.append(response)  # LLM의 tool call 요청
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                )
            )

        # 4단계: 최종 답변 생성
        final_response = model_with_tools.invoke(messages)
        print(f"\n🤖 최종 답변: {final_response.content}")

    print("\n💡 LLM 요청 → 도구 실행 → 결과 반환 → 최종 답변!\n")


# ============================================================================
# 예제 4: 여러 도구 동시 호출
# ============================================================================

def example_4_multiple_tool_calls():
    """한 번에 여러 도구 호출하기"""
    print("=" * 70)
    print("📌 예제 4: 여러 도구 동시 호출")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate]
    model_with_tools = model.bind_tools(tools)

    tools_map = {tool.name: tool for tool in tools}

    # 여러 도구가 필요한 복잡한 질문
    user_question = "서울과 부산의 날씨를 알려주고, 두 도시의 평균 온도를 계산해줘"
    print(f"\n👤 사용자: {user_question}")

    messages = [HumanMessage(content=user_question)]
    response = model_with_tools.invoke(messages)

    print(f"\n🤖 LLM이 요청한 도구 개수: {len(response.tool_calls)}")

    if response.tool_calls:
        messages.append(response)

        # 모든 tool call 실행
        for i, tool_call in enumerate(response.tool_calls, 1):
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"\n🔧 도구 {i}: {tool_name}")
            print(f"   인자: {tool_args}")

            selected_tool = tools_map[tool_name]
            tool_result = selected_tool.invoke(tool_args)

            print(f"   결과: {tool_result}")

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                )
            )

        # 최종 답변
        final_response = model_with_tools.invoke(messages)
        print(f"\n🤖 최종 답변:\n   {final_response.content}")

    print("\n💡 복잡한 작업을 여러 도구로 나누어 처리!\n")


# ============================================================================
# 예제 5: Tool call 에러 핸들링
# ============================================================================

@tool
def divide_numbers(a: float, b: float) -> str:
    """두 숫자를 나눕니다.

    Args:
        a: 분자
        b: 분모
    """
    if b == 0:
        raise ValueError("0으로 나눌 수 없습니다")
    result = a / b
    return f"{a} ÷ {b} = {result}"


def example_5_error_handling():
    """Tool call 에러 핸들링"""
    print("=" * 70)
    print("📌 예제 5: Tool call 에러 핸들링")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [divide_numbers, calculate]
    model_with_tools = model.bind_tools(tools)

    tools_map = {tool.name: tool for tool in tools}

    # 에러가 발생할 수 있는 질문
    user_question = "10을 0으로 나누면?"
    print(f"\n👤 사용자: {user_question}")

    messages = [HumanMessage(content=user_question)]
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            print(f"\n🔧 도구 실행: {tool_name}({tool_args})")

            try:
                selected_tool = tools_map[tool_name]
                tool_result = selected_tool.invoke(tool_args)
                print(f"✅ 결과: {tool_result}")

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    )
                )

            except Exception as e:
                error_message = f"오류 발생: {str(e)}"
                print(f"❌ {error_message}")

                # 에러를 ToolMessage로 LLM에 전달
                messages.append(
                    ToolMessage(
                        content=error_message,
                        tool_call_id=tool_call['id'],
                        status="error"
                    )
                )

        # LLM이 에러를 이해하고 답변
        final_response = model_with_tools.invoke(messages)
        print(f"\n🤖 LLM의 에러 처리:\n   {final_response.content}")

    print("\n💡 에러도 ToolMessage로 전달하면 LLM이 처리!\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n🎓 Part 2: LangChain 기초 - Tool Calling\n")

    example_1_bind_tools()
    input("⏎ 계속하려면 Enter...")

    example_2_examine_tool_calls()
    input("⏎ 계속하려면 Enter...")

    example_3_execute_tool_calls()
    input("⏎ 계속하려면 Enter...")

    example_4_multiple_tool_calls()
    input("⏎ 계속하려면 Enter...")

    example_5_error_handling()

    print("=" * 70)
    print("🎉 Tool Calling 학습 완료!")
    print("📖 다음: Part 3 - 첫 번째 Agent 만들기")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. Tool Calling 프로세스:
#    ① 사용자 질문 → LLM
#    ② LLM이 필요한 도구 선택 및 인자 생성
#    ③ 도구 실행
#    ④ 결과를 ToolMessage로 LLM에 전달
#    ⑤ LLM이 최종 답변 생성
#
# 2. bind_tools() vs Agent:
#    - bind_tools(): 수동으로 tool call 실행 필요
#    - Agent: 자동으로 tool call 실행 (Part 3에서 학습)
#
# 3. ToolMessage의 역할:
#    - 도구 실행 결과를 LLM에 전달
#    - tool_call_id로 어떤 요청의 결과인지 연결
#    - 에러도 ToolMessage로 전달 가능
#
# 4. 여러 도구 호출:
#    - LLM이 한번에 여러 도구를 요청할 수 있음
#    - 각 tool call마다 ToolMessage 생성 필요
#    - 순서대로 또는 병렬로 실행 가능
#
# 5. 실전 팁:
#    - 도구 설명을 명확하게 작성 (LLM이 읽음)
#    - 에러 핸들링 필수 (도구가 실패할 수 있음)
#    - Agent를 사용하면 이 과정이 자동화됨
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: LLM이 도구를 호출하지 않고 직접 답변
# 해결: 도구 설명을 더 명확하게 작성, temperature=0으로 설정
#
# 문제: tool_call_id 매칭 오류
# 해결: ToolMessage의 tool_call_id는 반드시 원래 요청의 ID와 일치해야 함
#
# 문제: 여러 도구 호출 시 순서 문제
# 해결: 각 tool call을 순서대로 처리하거나, 병렬 처리 후 모두 전달
#
# 문제: 도구 실행 에러가 발생하면 전체 중단
# 해결: try-except로 에러를 잡아 ToolMessage로 전달하면 LLM이 처리
#
# 문제: 너무 많은 도구를 연결하면 성능 저하
# 해결: 필요한 도구만 선택적으로 연결, 또는 도구를 카테고리로 분류
#
# ============================================================================
