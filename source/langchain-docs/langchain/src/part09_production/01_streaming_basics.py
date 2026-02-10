"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 (Production)
================================================================================

파일명: 01_streaming_basics.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 20분

📚 학습 목표:
  - Streaming의 기본 개념 이해
  - stream() vs invoke() 차이 파악
  - 실시간 응답 처리 구현
  - Streaming의 장점과 활용 사례

📖 공식 문서:
  • Streaming: /official/11-streaming-overview.md

📄 교안 문서:
  • Part 9 개요: /docs/part09_production.md

🔧 필요한 패키지:
  pip install langchain langchain-openai

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 01_streaming_basics.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import time

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    sys.exit(1)

# ============================================================================
# 예제 1: invoke() vs stream() 비교
# ============================================================================

def example_1_invoke_vs_stream():
    """invoke()와 stream()의 차이 체험"""
    print("=" * 70)
    print("📌 예제 1: invoke() vs stream() 비교")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = "LangChain에서 Agent를 만드는 방법을 3단계로 설명해주세요."

    # 1. invoke() 방식 - 전체 응답을 한 번에
    print("\n🔹 invoke() 방식 (일괄 처리):")
    print("⏳ 응답 대기 중...")

    start_time = time.time()
    response = llm.invoke([HumanMessage(content=prompt)])
    elapsed = time.time() - start_time

    print(f"✅ 응답 완료 (소요 시간: {elapsed:.2f}초)")
    print(f"\n{response.content}\n")

    # 2. stream() 방식 - 실시간 스트리밍
    print("\n🔹 stream() 방식 (실시간 스트리밍):")
    print("🌊 실시간 응답:")
    print("-" * 70)

    start_time = time.time()
    full_response = ""

    for chunk in llm.stream([HumanMessage(content=prompt)]):
        content = chunk.content
        full_response += content
        print(content, end="", flush=True)

    elapsed = time.time() - start_time

    print("\n" + "-" * 70)
    print(f"✅ 스트리밍 완료 (소요 시간: {elapsed:.2f}초)")

    print("\n💡 차이점:")
    print("  • invoke(): 전체 응답이 완성될 때까지 대기")
    print("  • stream(): 생성되는 즉시 토큰 단위로 실시간 출력")
    print("  • 사용자 경험: stream()이 더 빠르게 느껴짐")


# ============================================================================
# 예제 2: Agent Streaming 기초
# ============================================================================

def example_2_agent_streaming():
    """Agent에서 Streaming 사용하기"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Agent Streaming 기초")
    print("=" * 70)

    @tool
    def search_documents(query: str) -> str:
        """문서에서 정보를 검색합니다."""
        # 실제로는 벡터 DB 검색 등
        time.sleep(1)  # 검색 시뮬레이션
        return f"'{query}'에 대한 검색 결과: LangChain은 LLM 애플리케이션 개발 프레임워크입니다."

    @tool
    def calculate(expression: str) -> str:
        """수식을 계산합니다."""
        try:
            result = eval(expression)
            return f"{expression} = {result}"
        except:
            return "계산 오류"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_documents, calculate],
    )

    # Streaming으로 실행
    print("\n🌊 Agent 실행 (Streaming):")
    print("-" * 70)

    user_message = "LangChain이 무엇인지 검색하고, 2024 + 1을 계산해주세요."
    print(f"👤 사용자: {user_message}\n")

    for chunk in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        if "messages" in chunk:
            message = chunk["messages"][-1]

            # AI 메시지
            if hasattr(message, "content") and message.content:
                print(f"🤖 {message.content}")

            # Tool 호출
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"🔧 도구 호출: {tool_call['name']}({tool_call['args']})")

    print("-" * 70)
    print("✅ 스트리밍 완료")


# ============================================================================
# 예제 3: 사용자 정의 Streaming 출력
# ============================================================================

def example_3_custom_streaming_output():
    """Streaming 출력을 커스터마이징하기"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 사용자 정의 Streaming 출력")
    print("=" * 70)

    @tool
    def get_weather(city: str) -> str:
        """도시의 날씨를 가져옵니다."""
        time.sleep(0.5)
        return f"{city}의 현재 날씨: 맑음, 기온 22°C"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_weather],
    )

    print("\n🎨 커스터마이징된 Streaming 출력:")
    print("-" * 70)

    user_message = "서울과 부산의 날씨를 알려주세요."
    print(f"👤 사용자: {user_message}\n")

    step_count = 0

    for chunk in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        if "messages" in chunk:
            message = chunk["messages"][-1]

            # 단계 카운트
            step_count += 1

            # AI 생각 중
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"\n[Step {step_count}] 🧠 AI 생각:")
                for tool_call in message.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    print(f"  → {tool_name} 호출 준비: {tool_args}")

            # Tool 실행 결과
            elif hasattr(message, "content") and message.name:
                print(f"\n[Step {step_count}] 🔧 도구 실행 결과:")
                print(f"  → {message.content}")

            # 최종 답변
            elif hasattr(message, "content") and message.content:
                print(f"\n[Step {step_count}] 💬 AI 최종 답변:")
                print(f"  {message.content}")

    print("\n" + "-" * 70)
    print(f"✅ 총 {step_count}개 단계로 완료")


# ============================================================================
# 예제 4: Streaming 진행 상황 표시
# ============================================================================

def example_4_streaming_progress():
    """Streaming 중 진행 상황 표시하기"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Streaming 진행 상황 표시")
    print("=" * 70)

    @tool
    def analyze_data(data_name: str) -> str:
        """데이터를 분석합니다."""
        time.sleep(1.5)
        return f"{data_name} 분석 완료: 평균 85점, 최고 98점, 최저 72점"

    @tool
    def generate_report(title: str) -> str:
        """보고서를 생성합니다."""
        time.sleep(1)
        return f"'{title}' 보고서가 생성되었습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[analyze_data, generate_report],
    )

    print("\n📊 진행 상황 표시:")
    print("-" * 70)

    user_message = "학생 성적 데이터를 분석하고 보고서를 생성해주세요."
    print(f"👤 사용자: {user_message}\n")

    progress_indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    indicator_index = 0

    for chunk in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
        if "messages" in chunk:
            message = chunk["messages"][-1]

            # Tool 호출 시작
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call['name']
                    print(f"\n{progress_indicators[indicator_index % len(progress_indicators)]} {tool_name} 실행 중...", end="", flush=True)
                    indicator_index += 1

            # Tool 결과
            elif hasattr(message, "name"):
                print(f" ✅")

            # 최종 답변
            elif hasattr(message, "content") and message.content:
                print(f"\n\n💬 {message.content}")

    print("\n" + "-" * 70)
    print("✅ 작업 완료")


# ============================================================================
# 예제 5: Streaming Error Handling
# ============================================================================

def example_5_streaming_error_handling():
    """Streaming 중 오류 처리"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Streaming Error Handling")
    print("=" * 70)

    @tool
    def risky_operation(value: int) -> str:
        """오류가 발생할 수 있는 작업"""
        if value < 0:
            raise ValueError("음수는 처리할 수 없습니다!")
        return f"작업 성공: {value} 처리 완료"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[risky_operation],
    )

    print("\n⚠️  오류 처리 테스트:")
    print("-" * 70)

    # 정상 케이스
    print("\n🔹 정상 케이스:")
    user_message = "값 100으로 작업을 실행해주세요."
    print(f"👤 {user_message}")

    try:
        for chunk in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
            if "messages" in chunk:
                message = chunk["messages"][-1]
                if hasattr(message, "content") and message.content:
                    print(f"🤖 {message.content}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    # 오류 케이스
    print("\n🔹 오류 케이스:")
    user_message = "값 -50으로 작업을 실행해주세요."
    print(f"👤 {user_message}")

    try:
        collected_chunks = []
        for chunk in agent.stream({"messages": [{"role": "user", "content": user_message}]}):
            collected_chunks.append(chunk)
            if "messages" in chunk:
                message = chunk["messages"][-1]
                if hasattr(message, "content") and message.content:
                    print(f"🤖 {message.content}")

        print("✅ Streaming 완료 (Agent가 오류를 처리함)")

    except Exception as e:
        print(f"❌ Streaming 중 오류: {e}")
        print("💡 수집된 청크:", len(collected_chunks))

    print("\n" + "-" * 70)
    print("💡 Streaming 오류 처리 팁:")
    print("  • try-except로 예외 처리")
    print("  • 부분 응답 저장 가능")
    print("  • Agent는 Tool 오류를 자동으로 처리")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 9: 프로덕션 - Streaming 기초")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_invoke_vs_stream()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_agent_streaming()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_custom_streaming_output()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_streaming_progress()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_streaming_error_handling()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 9-01: Streaming 기초를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 02_stream_modes.py - Stream Modes 학습")
    print("  2. 03_custom_stream.py - Custom Streaming")
    print("  3. 04_hitl_basic.py - Human-in-the-Loop 기초")
    print("\n📚 핵심 요약:")
    print("  • invoke(): 전체 응답을 한 번에 반환")
    print("  • stream(): 토큰 단위로 실시간 스트리밍")
    print("  • 더 나은 UX를 위해 stream() 사용 권장")
    print("  • Agent도 stream()으로 실시간 처리 가능")
    print("  • 오류 처리는 try-except로 구현")
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
# 1. Streaming의 장점:
#    - 사용자 경험 향상 (즉각적인 피드백)
#    - 긴 응답도 빠르게 느껴짐
#    - 실시간 상호작용 가능
#
# 2. Streaming 활용 사례:
#    - 챗봇 UI (ChatGPT 스타일)
#    - 긴 문서 생성
#    - 실시간 분석 보고서
#    - 진행 상황 표시
#
# 3. 성능 최적화:
#    - 청크 크기 조절
#    - 버퍼링 전략
#    - 네트워크 최적화
#
# ============================================================================
