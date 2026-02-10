"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: 첫 번째 Agent 만들기
================================================================================

파일명: 05_streaming_agent.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 25분

📚 학습 목표:
  - Streaming의 개념과 장점 이해
  - invoke() vs stream() 비교
  - stream_mode 종류 (values, messages, updates)
  - 실시간 UI 시뮬레이션 구현

📖 공식 문서:
  • Streaming: /official/06-agents.md (라인 461-476)
  • LangChain Streaming: /oss/python/langchain/streaming

📄 교안 문서:
  • Part 3 개요: /docs/part03_first_agent.md (섹션 5)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 05_streaming_agent.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

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
# 공통 도구 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 조회합니다.

    Args:
        city: 도시 이름 (예: 서울, 부산)
    """
    time.sleep(1)  # API 호출 시뮬레이션
    weather_data = {
        "서울": "맑음, 22°C, 습도 60%",
        "부산": "흐림, 20°C, 습도 70%",
        "제주": "비, 18°C, 습도 85%",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")


@tool
def get_forecast(city: str, days: int = 3) -> str:
    """며칠간의 날씨 예보를 조회합니다.

    Args:
        city: 도시 이름
        days: 예보 일수 (기본 3일)
    """
    time.sleep(1.5)  # API 호출 시뮬레이션
    forecasts = {
        "서울": "맑음 → 흐림 → 비",
        "부산": "흐림 → 비 → 맑음",
        "제주": "비 → 비 → 흐림",
    }
    forecast = forecasts.get(city, "예보 정보를 찾을 수 없습니다")
    return f"{city}의 {days}일 예보: {forecast}"


# ============================================================================
# 예제 1: invoke() vs stream() 비교
# ============================================================================

def example_1_invoke_vs_stream():
    """invoke()와 stream()의 차이 체감하기"""
    print("=" * 70)
    print("📌 예제 1: invoke() vs stream() 비교")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather, get_forecast],
        system_prompt="당신은 날씨 정보를 제공하는 친절한 Agent입니다.",
    )

    question = {"messages": [{"role": "user", "content": "서울의 현재 날씨와 3일 예보를 알려줘"}]}

    # 방법 1: invoke() - 완료 후 반환
    print("\n🔹 방법 1: invoke() - 완료 후 한 번에 반환")
    print("👤 사용자: 서울의 현재 날씨와 3일 예보를 알려줘")
    print("⏳ Agent가 작업 중... (대기)")

    start_time = time.time()
    result = agent.invoke(question)
    elapsed = time.time() - start_time

    print(f"✅ 완료! (소요 시간: {elapsed:.1f}초)")
    print(f"🤖 Agent: {result['messages'][-1].content}")

    print("\n" + "-" * 70)

    # 방법 2: stream() - 실시간 반환
    print("\n🔹 방법 2: stream() - 실시간으로 중간 과정 표시")
    print("👤 사용자: 서울의 현재 날씨와 3일 예보를 알려줘")
    print("🤖 Agent: (실시간 스트리밍)\n")

    start_time = time.time()
    final_answer = ""

    for chunk in agent.stream(question, stream_mode="values"):
        latest_message = chunk["messages"][-1]

        # 도구 호출 표시
        if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
            for tc in latest_message.tool_calls:
                print(f"   [도구 호출] {tc['name']}({tc['args']}) ...")

        # 도구 결과 표시
        elif latest_message.__class__.__name__ == "ToolMessage":
            print(f"   [도구 결과] {latest_message.content[:50]}...")

        # 최종 답변
        elif hasattr(latest_message, "content") and latest_message.content and not hasattr(latest_message, "tool_calls"):
            final_answer = latest_message.content

    elapsed = time.time() - start_time
    print(f"\n   [최종 답변] {final_answer}")
    print(f"\n✅ 완료! (소요 시간: {elapsed:.1f}초)")

    print("\n💡 핵심 차이:")
    print("  - invoke(): 모든 작업이 끝난 후 결과만 반환")
    print("  - stream(): 작업 진행 중 중간 상태를 실시간으로 반환")
    print("  - stream()은 사용자에게 '대기 중'이 아니라 '진행 중'임을 보여줍니다\n")


# ============================================================================
# 예제 2: stream_mode="values" - 전체 상태 스트리밍
# ============================================================================

def example_2_stream_values_mode():
    """values 모드: 매번 전체 상태 반환"""
    print("=" * 70)
    print("📌 예제 2: stream_mode='values' - 전체 상태")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="당신은 간결한 날씨 Agent입니다.",
    )

    print("\n👤 사용자: 서울 날씨는?")
    print("\n🔄 stream_mode='values' 실행:\n")

    chunk_count = 0
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "서울 날씨는?"}]},
        stream_mode="values"  # 기본값
    ):
        chunk_count += 1
        messages = chunk["messages"]
        latest = messages[-1]

        print(f"[Chunk {chunk_count}]")
        print(f"  전체 메시지 수: {len(messages)}")
        print(f"  최신 메시지 타입: {latest.__class__.__name__}")
        if hasattr(latest, "content") and latest.content:
            print(f"  내용: {latest.content[:50]}...")
        if hasattr(latest, "tool_calls") and latest.tool_calls:
            print(f"  도구 호출: {[tc['name'] for tc in latest.tool_calls]}")
        print()

    print("💡 values 모드의 특징:")
    print("  - 매번 전체 상태(모든 메시지)를 반환합니다")
    print("  - 전체 컨텍스트를 항상 확인할 수 있습니다")
    print("  - 단점: 중복 데이터가 많아 네트워크 부담이 있습니다\n")


# ============================================================================
# 예제 3: stream_mode="messages" - 메시지만 스트리밍
# ============================================================================

def example_3_stream_messages_mode():
    """messages 모드: 메시지만 반환"""
    print("=" * 70)
    print("📌 예제 3: stream_mode='messages' - 메시지 스트림")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="당신은 간결한 날씨 Agent입니다.",
    )

    print("\n👤 사용자: 부산 날씨는?")
    print("🤖 Agent: ", end="", flush=True)

    for msg_tuple in agent.stream(
        {"messages": [{"role": "user", "content": "부산 날씨는?"}]},
        stream_mode="messages"
    ):
        # msg_tuple은 (message, metadata) 형태
        message, metadata = msg_tuple

        # AIMessage의 content만 출력 (실시간 타이핑 효과)
        if hasattr(message, "content") and message.content:
            print(message.content, end="", flush=True)

    print("\n")

    print("\n💡 messages 모드의 특징:")
    print("  - 메시지 객체만 반환합니다")
    print("  - UI에 직접 표시하기 쉽습니다")
    print("  - 타이핑 효과 구현에 적합합니다\n")


# ============================================================================
# 예제 4: stream_mode="updates" - 변경사항만 스트리밍
# ============================================================================

def example_4_stream_updates_mode():
    """updates 모드: 각 단계의 변경사항만 반환"""
    print("=" * 70)
    print("📌 예제 4: stream_mode='updates' - 변경사항만")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="당신은 간결한 날씨 Agent입니다.",
    )

    print("\n👤 사용자: 제주 날씨는?")
    print("\n🔄 stream_mode='updates' 실행:\n")

    update_count = 0
    for update in agent.stream(
        {"messages": [{"role": "user", "content": "제주 날씨는?"}]},
        stream_mode="updates"
    ):
        update_count += 1
        print(f"[Update {update_count}]")
        print(f"  노드: {list(update.keys())}")

        for node_name, node_data in update.items():
            if "messages" in node_data:
                messages = node_data["messages"]
                print(f"  추가된 메시지 수: {len(messages)}")
                for msg in messages:
                    print(f"    - {msg.__class__.__name__}")
        print()

    print("💡 updates 모드의 특징:")
    print("  - 각 단계에서 추가된 내용만 반환합니다")
    print("  - 네트워크 효율적입니다 (중복 없음)")
    print("  - 전체 컨텍스트는 직접 관리해야 합니다\n")


# ============================================================================
# 예제 5: 실시간 UI 시뮬레이션
# ============================================================================

def example_5_realtime_ui_simulation():
    """실제 챗봇처럼 실시간 응답 표시"""
    print("=" * 70)
    print("📌 예제 5: 실시간 UI 시뮬레이션 (챗봇)")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather, get_forecast],
        system_prompt="당신은 친절한 날씨 비서입니다. 답변은 간결하게 제공하세요.",
    )

    print("\n" + "=" * 70)
    print("💬 날씨 챗봇 (실시간 모드)")
    print("=" * 70)

    # 시뮬레이션할 대화
    conversations = [
        "서울 날씨 알려줘",
        "부산과 제주 중 어디가 더 따뜻해?",
    ]

    for query in conversations:
        print(f"\n👤 사용자: {query}")
        print("🤖 Agent: ", end="", flush=True)

        current_status = ""
        final_content = ""

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            latest_message = chunk["messages"][-1]
            msg_type = latest_message.__class__.__name__

            # 상태 표시 업데이트
            if msg_type == "AIMessage" and hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                # 도구 호출 중
                if current_status != "tool_calling":
                    print("\n   [정보 조회 중", end="", flush=True)
                    current_status = "tool_calling"
                else:
                    print(".", end="", flush=True)

            elif msg_type == "ToolMessage":
                # 도구 결과 수신
                if current_status == "tool_calling":
                    print("]", end="", flush=True)
                    current_status = "tool_received"

            elif msg_type == "AIMessage" and hasattr(latest_message, "content") and latest_message.content:
                # 최종 답변
                if current_status in ["tool_calling", "tool_received"]:
                    print("\n   ", end="", flush=True)
                    current_status = "answering"

                # 타이핑 효과 (새로운 내용만 출력)
                new_content = latest_message.content[len(final_content):]
                if new_content:
                    for char in new_content:
                        print(char, end="", flush=True)
                        time.sleep(0.02)  # 타이핑 효과
                    final_content = latest_message.content

        print("\n")

    print("\n" + "=" * 70)

    print("\n💡 실전 UI 구현 포인트:")
    print("  1. 도구 호출 중: 로딩 인디케이터 표시")
    print("  2. 도구 완료: 체크마크 또는 완료 메시지")
    print("  3. 최종 답변: 타이핑 효과로 자연스럽게 표시")
    print("  4. 스트리밍으로 사용자 대기 시간 체감 감소\n")


# ============================================================================
# 보너스: 진행 상황 바 표시
# ============================================================================

def bonus_progress_bar():
    """진행 상황 바와 함께 스트리밍"""
    print("=" * 70)
    print("📌 보너스: 진행 상황 바 표시")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather, get_forecast],
        system_prompt="당신은 날씨 Agent입니다.",
    )

    print("\n👤 사용자: 서울의 날씨와 예보를 알려줘")
    print()

    steps = []
    total_steps = 0

    # 1단계: 스트림 수집 및 단계 파악
    print("🔍 작업 분석 중...", end="", flush=True)
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "서울의 날씨와 예보를 알려줘"}]},
        stream_mode="values"
    ):
        latest = chunk["messages"][-1]
        if hasattr(latest, "tool_calls") and latest.tool_calls:
            for tc in latest.tool_calls:
                steps.append(tc["name"])
                total_steps += 1

    print(f" 완료! (총 {total_steps}개 작업)\n")

    # 2단계: 재실행하며 진행 바 표시
    current_step = 0

    def print_progress(current, total, message=""):
        percent = int((current / total) * 100)
        bar_length = 30
        filled = int((bar_length * current) / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {percent}% - {message}", end="", flush=True)

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "서울의 날씨와 예보를 알려줘"}]},
        stream_mode="values"
    ):
        latest = chunk["messages"][-1]
        msg_type = latest.__class__.__name__

        if msg_type == "AIMessage" and hasattr(latest, "tool_calls") and latest.tool_calls:
            for tc in latest.tool_calls:
                current_step += 1
                print_progress(current_step, total_steps, f"{tc['name']} 실행 중...")
                time.sleep(0.5)

        elif msg_type == "ToolMessage":
            print_progress(current_step, total_steps, "완료")

    print("\n")

    # 최종 답변
    final_result = agent.invoke({"messages": [{"role": "user", "content": "서울의 날씨와 예보를 알려줘"}]})
    print(f"🤖 Agent: {final_result['messages'][-1].content}")

    print("\n💡 진행 바의 효과:")
    print("  - 사용자가 작업 진행도를 시각적으로 확인")
    print("  - 대기 시간이 짧게 느껴짐")
    print("  - 어느 단계에서 시간이 오래 걸리는지 파악 가능\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 3: 첫 번째 Agent 만들기 - Streaming Agent")
    print("\n")

    # 모든 예제 실행
    example_1_invoke_vs_stream()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_stream_values_mode()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_stream_messages_mode()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_stream_updates_mode()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_realtime_ui_simulation()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    bonus_progress_bar()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Streaming Agent 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. Part 4: Memory & Context Management")
    print("  2. 실습 과제: 스트리밍을 활용한 챗봇 만들기")
    print("\n📚 Streaming 활용 팁:")
    print("  • 웹 UI: stream_mode='messages'로 타이핑 효과")
    print("  • 모니터링: stream_mode='values'로 전체 상태 추적")
    print("  • 효율성: stream_mode='updates'로 네트워크 최적화")
    print("  • 사용자 경험: 진행 상황 표시로 대기 시간 체감 감소")
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
# 1. Streaming 모드 선택 가이드:
#    - stream_mode="values": 디버깅, 전체 상태 필요
#    - stream_mode="messages": UI 구현, 타이핑 효과
#    - stream_mode="updates": 네트워크 최적화, 효율성
#
# 2. 타이핑 효과 구현:
#    for msg_tuple in agent.stream(..., stream_mode="messages"):
#        message, metadata = msg_tuple
#        if hasattr(message, "content"):
#            print(message.content, end="", flush=True)
#
# 3. 웹 프레임워크 연동:
#    - Streamlit: st.write_stream()
#    - FastAPI: StreamingResponse()
#    - Gradio: gr.ChatInterface()
#
# 4. 에러 처리:
#    try:
#        for chunk in agent.stream(...):
#            process(chunk)
#    except Exception as e:
#        print(f"스트리밍 오류: {e}")
#        # 폴백 처리
#
# 5. 성능 최적화:
#    - 필요한 정보만 스트리밍
#    - 버퍼링으로 네트워크 효율 향상
#    - 타임아웃 설정
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: 스트리밍이 멈춤 (응답 없음)
# 해결:
#   - timeout 설정 확인
#   - 네트워크 연결 상태 확인
#   - 도구가 무한 루프에 빠지지 않았는지 확인
#
# 문제: 출력이 버퍼링되어 한 번에 나옴
# 해결:
#   - flush=True 옵션 사용
#   - sys.stdout.flush() 명시적 호출
#   - 환경변수 PYTHONUNBUFFERED=1 설정
#
# 문제: stream_mode="messages"에서 도구 호출 정보가 안 보임
# 해결:
#   - stream_mode="values" 사용
#   - 또는 tool_calls 확인 로직 추가
#
# 문제: 웹 UI에서 스트리밍이 작동하지 않음
# 해결:
#   - Server-Sent Events (SSE) 사용
#   - WebSocket 연결 사용
#   - 프레임워크별 스트리밍 API 확인
#
# ============================================================================
