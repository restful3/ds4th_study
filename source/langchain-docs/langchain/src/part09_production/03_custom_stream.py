"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 (Production)
================================================================================

파일명: 03_custom_stream.py
난이도: ⭐⭐⭐⭐⭐ (전문가)
예상 시간: 25분

📚 학습 목표:
  - Custom Streaming 구현 방법
  - Streaming 데이터 가공 및 필터링
  - 실시간 UI 업데이트 패턴
  - 고급 Streaming 기법

📖 공식 문서:
  • Streaming: /official/11-streaming-overview.md

📄 교안 문서:
  • Part 9 개요: /docs/part09_production.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 03_custom_stream.py

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
from langgraph.checkpoint.memory import MemorySaver
from typing import Generator, Dict, Any
import time

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: Custom Streaming Wrapper
# ============================================================================

def example_1_custom_wrapper():
    """Custom Streaming Wrapper 만들기"""
    print("=" * 70)
    print("📌 예제 1: Custom Streaming Wrapper")
    print("=" * 70)

    @tool
    def get_data(source: str) -> str:
        """데이터를 가져옵니다."""
        time.sleep(0.5)
        return f"{source}에서 데이터 100건 로드 완료"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_data],
        checkpointer=MemorySaver(),
    )

    def custom_stream_wrapper(
        agent_stream: Generator,
        add_timestamps: bool = True,
        filter_tool_messages: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Agent 스트림을 커스터마이징하는 래퍼

        Args:
            agent_stream: Agent.stream() 제너레이터
            add_timestamps: 타임스탬프 추가 여부
            filter_tool_messages: Tool 메시지 필터링 여부
        """
        for chunk in agent_stream:
            # 커스텀 데이터 구조 생성
            custom_chunk = {
                "original": chunk,
                "metadata": {}
            }

            # 타임스탬프 추가
            if add_timestamps:
                custom_chunk["metadata"]["timestamp"] = time.time()

            # Tool 메시지 필터링
            if filter_tool_messages and "messages" in chunk:
                messages = chunk["messages"]
                filtered = [
                    msg for msg in messages
                    if not hasattr(msg, "name") or not msg.name
                ]
                if filtered:
                    custom_chunk["filtered_messages"] = filtered
                else:
                    continue  # Tool 메시지만 있으면 스킵

            yield custom_chunk

    print("\n🎨 Custom Wrapper 사용:")
    print("-" * 70)

    user_message = "database에서 데이터를 가져와주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "custom_wrapper"}}

    # Custom wrapper 적용
    base_stream = agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="messages"
    )

    wrapped_stream = custom_stream_wrapper(
        base_stream,
        add_timestamps=True,
        filter_tool_messages=True
    )

    for i, custom_chunk in enumerate(wrapped_stream, 1):
        print(f"\n[Chunk {i}]")

        if "metadata" in custom_chunk and "timestamp" in custom_chunk["metadata"]:
            timestamp = custom_chunk["metadata"]["timestamp"]
            print(f"  🕐 타임스탬프: {timestamp:.3f}")

        message, metadata = custom_chunk["original"]
        if hasattr(message, "content") and message.content:
            print(f"  💬 {message.content[:60]}...")

    print("\n" + "-" * 70)
    print("✅ Custom Wrapper로 스트림 데이터 가공 완료")


# ============================================================================
# 예제 2: 실시간 진행률 표시
# ============================================================================

def example_2_progress_streaming():
    """실시간 진행률을 표시하는 Streaming"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 실시간 진행률 표시")
    print("=" * 70)

    @tool
    def step_1_collect() -> str:
        """Step 1: 데이터 수집"""
        time.sleep(1)
        return "데이터 수집 완료: 1000건"

    @tool
    def step_2_process() -> str:
        """Step 2: 데이터 처리"""
        time.sleep(1.5)
        return "데이터 처리 완료: 검증됨"

    @tool
    def step_3_analyze() -> str:
        """Step 3: 데이터 분석"""
        time.sleep(1)
        return "분석 완료: 평균 85점"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[step_1_collect, step_2_process, step_3_analyze],
        checkpointer=MemorySaver(),
    )

    class ProgressTracker:
        """진행 상황을 추적하는 클래스"""

        def __init__(self, total_steps: int = 3):
            self.total_steps = total_steps
            self.current_step = 0
            self.start_time = time.time()

        def update(self, step_name: str):
            """진행 단계 업데이트"""
            self.current_step += 1
            progress = (self.current_step / self.total_steps) * 100
            elapsed = time.time() - self.start_time

            # 진행률 바 생성
            bar_length = 30
            filled = int(bar_length * self.current_step / self.total_steps)
            bar = "█" * filled + "░" * (bar_length - filled)

            print(f"\n📊 진행률: [{bar}] {progress:.0f}%")
            print(f"   단계: {step_name}")
            print(f"   경과 시간: {elapsed:.1f}초")

    print("\n📊 진행 상황 추적:")
    print("-" * 70)

    user_message = "데이터를 수집, 처리, 분석해주세요."
    print(f"👤 사용자: {user_message}")

    config = {"configurable": {"thread_id": "progress_demo"}}
    tracker = ProgressTracker(total_steps=3)

    for message, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="messages"
    ):
        # Tool 호출 감지
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call["name"]
                tracker.update(tool_name)

        # 최종 답변
        elif hasattr(message, "content") and message.content and "완료" in message.content:
            print(f"\n\n🎉 최종 결과:\n{message.content}")

    print("\n" + "-" * 70)
    print("✅ 진행률 추적 완료")


# ============================================================================
# 예제 3: 스트리밍 데이터 필터링
# ============================================================================

def example_3_stream_filtering():
    """특정 조건으로 스트림 필터링"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 스트리밍 데이터 필터링")
    print("=" * 70)

    @tool
    def check_item(item: str) -> str:
        """아이템을 확인합니다."""
        return f"{item}: OK"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[check_item],
        checkpointer=MemorySaver(),
    )

    def filter_stream(
        stream: Generator,
        include_ai_messages: bool = True,
        include_tool_calls: bool = True,
        include_tool_responses: bool = False
    ) -> Generator:
        """
        스트림을 필터링

        Args:
            include_ai_messages: AI 응답 포함 여부
            include_tool_calls: Tool 호출 포함 여부
            include_tool_responses: Tool 응답 포함 여부
        """
        for message, metadata in stream:
            should_include = False
            message_type = "unknown"

            # AI 메시지
            if hasattr(message, "content") and message.content and not hasattr(message, "name"):
                should_include = include_ai_messages
                message_type = "ai_message"

            # Tool 호출
            elif hasattr(message, "tool_calls") and message.tool_calls:
                should_include = include_tool_calls
                message_type = "tool_call"

            # Tool 응답
            elif hasattr(message, "name") and message.name:
                should_include = include_tool_responses
                message_type = "tool_response"

            if should_include:
                yield {
                    "type": message_type,
                    "message": message,
                    "metadata": metadata
                }

    print("\n🔍 필터링 예시:")
    print("-" * 70)

    user_message = "item1, item2, item3를 확인해주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "filter_demo"}}

    # AI 메시지만 필터링
    print("🔹 AI 메시지만 표시:")
    base_stream = agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="messages"
    )

    filtered = filter_stream(
        base_stream,
        include_ai_messages=True,
        include_tool_calls=False,
        include_tool_responses=False
    )

    for item in filtered:
        if item["type"] == "ai_message":
            content = item["message"].content
            print(f"  🤖 {content[:70]}...")

    print("\n" + "-" * 70)
    print("✅ 필터링된 스트림 처리 완료")


# ============================================================================
# 예제 4: 버퍼링 및 배치 처리
# ============================================================================

def example_4_buffered_streaming():
    """버퍼링 및 배치 처리"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 버퍼링 및 배치 처리")
    print("=" * 70)

    @tool
    def generate_item(index: int) -> str:
        """아이템을 생성합니다."""
        return f"Item-{index}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[generate_item],
        checkpointer=MemorySaver(),
    )

    def buffered_stream(stream: Generator, buffer_size: int = 3):
        """
        스트림을 버퍼링하여 배치로 반환

        Args:
            buffer_size: 버퍼 크기
        """
        buffer = []

        for chunk in stream:
            buffer.append(chunk)

            # 버퍼가 가득 차면 반환
            if len(buffer) >= buffer_size:
                yield {
                    "batch": buffer.copy(),
                    "size": len(buffer)
                }
                buffer.clear()

        # 남은 항목 반환
        if buffer:
            yield {
                "batch": buffer.copy(),
                "size": len(buffer)
            }

    print("\n📦 버퍼링 스트림:")
    print("-" * 70)

    user_message = "5개의 아이템을 생성해주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "buffer_demo"}}

    base_stream = agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="messages"
    )

    buffered = buffered_stream(base_stream, buffer_size=2)

    batch_num = 0
    for batch_data in buffered:
        batch_num += 1
        print(f"\n배치 {batch_num}: {batch_data['size']}개 항목")

        for message, metadata in batch_data["batch"]:
            if hasattr(message, "content") and message.content:
                print(f"  • {message.content[:50]}...")

    print("\n" + "-" * 70)
    print(f"✅ {batch_num}개 배치로 처리 완료")
    print("💡 버퍼링은 네트워크 오버헤드를 줄이는 데 유용합니다.")


# ============================================================================
# 예제 5: 실시간 UI 업데이트 시뮬레이션
# ============================================================================

def example_5_realtime_ui_simulation():
    """실시간 UI 업데이트 시뮬레이션"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실시간 UI 업데이트 시뮬레이션")
    print("=" * 70)

    @tool
    def fetch_news(category: str) -> str:
        """뉴스를 가져옵니다."""
        time.sleep(0.8)
        return f"{category} 뉴스: 최신 기사 10건"

    @tool
    def summarize_news(news: str) -> str:
        """뉴스를 요약합니다."""
        time.sleep(1)
        return f"요약: {news[:30]}..."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[fetch_news, summarize_news],
        checkpointer=MemorySaver(),
    )

    class UISimulator:
        """UI 업데이트 시뮬레이터"""

        def __init__(self):
            self.status = "대기 중"
            self.current_task = ""
            self.messages = []

        def update_status(self, status: str, task: str = ""):
            """상태 업데이트"""
            self.status = status
            self.current_task = task
            self._render()

        def add_message(self, message: str):
            """메시지 추가"""
            self.messages.append(message)
            self._render()

        def _render(self):
            """UI 렌더링 (콘솔 시뮬레이션)"""
            print("\r" + " " * 80, end="")  # 이전 줄 지우기
            print(f"\r상태: {self.status} | 작업: {self.current_task}", end="", flush=True)

    print("\n🖥️  UI 업데이트 시뮬레이션:")
    print("-" * 70)

    user_message = "기술 뉴스를 가져와서 요약해주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "ui_demo"}}
    ui = UISimulator()

    ui.update_status("시작", "Agent 실행 중")

    for message, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="messages"
    ):
        # Tool 호출
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call["name"]
                ui.update_status("실행 중", f"{tool_name} 호출")
                time.sleep(0.3)

        # Tool 응답
        elif hasattr(message, "name"):
            ui.update_status("처리 중", f"{message.name} 완료")

        # AI 답변
        elif hasattr(message, "content") and message.content:
            ui.update_status("완료", "답변 생성")
            print(f"\n\n💬 최종 답변:\n{message.content}")

    print("\n" + "-" * 70)
    print("✅ 실시간 UI 업데이트 시뮬레이션 완료")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 9: 프로덕션 - Custom Streaming")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_custom_wrapper()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_progress_streaming()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_stream_filtering()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_buffered_streaming()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_realtime_ui_simulation()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 9-03: Custom Streaming을 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 04_hitl_basic.py - Human-in-the-Loop 기초")
    print("  2. 05_hitl_decisions.py - HITL 의사결정")
    print("  3. 06_structured_output.py - Structured Output")
    print("\n📚 핵심 요약:")
    print("  • Custom Wrapper로 스트림 데이터 가공")
    print("  • 진행률 표시로 UX 개선")
    print("  • 필터링으로 필요한 데이터만 처리")
    print("  • 버퍼링으로 네트워크 효율 향상")
    print("  • 실시간 UI 업데이트 패턴")
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
# 1. Custom Streaming 활용 사례:
#    - 실시간 채팅 UI
#    - 진행률 표시
#    - 로깅 및 모니터링
#    - 데이터 변환 및 가공
#
# 2. 성능 최적화:
#    - 버퍼 크기 조절
#    - 필터링으로 불필요한 데이터 제거
#    - 비동기 처리
#
# 3. 실전 패턴:
#    - Generator 체이닝
#    - 상태 추적
#    - 오류 처리
#
# ============================================================================
