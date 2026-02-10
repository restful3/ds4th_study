"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 (Production)
================================================================================

파일명: 02_stream_modes.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 20분

📚 학습 목표:
  - Stream Mode의 종류 이해
  - values, updates, messages 모드 활용
  - 각 모드의 차이점과 용도 파악
  - 실전 활용 패턴 학습

📖 공식 문서:
  • Streaming: /official/11-streaming-overview.md

📄 교안 문서:
  • Part 9 개요: /docs/part09_production.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 02_stream_modes.py

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
import time

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: Stream Mode 개요
# ============================================================================

def example_1_stream_modes_overview():
    """Stream Mode의 종류와 차이점"""
    print("=" * 70)
    print("📌 예제 1: Stream Mode 개요")
    print("=" * 70)

    print("""
📊 LangGraph의 3가지 Stream Mode:

1️⃣ "values" 모드 (기본값):
   - 각 단계 후 전체 상태(state) 반환
   - 가장 직관적
   - 메모리 사용량이 높을 수 있음
   - 용도: 전체 상태 추적, 디버깅

2️⃣ "updates" 모드:
   - 각 단계에서 변경된 부분만 반환
   - 효율적 (변경사항만 전송)
   - 델타(delta) 업데이트
   - 용도: 실시간 UI 업데이트, 네트워크 최적화

3️⃣ "messages" 모드:
   - 새로 추가된 메시지만 반환
   - 채팅 UI에 최적화
   - 가장 효율적
   - 용도: 챗봇, 대화형 UI

💡 선택 기준:
   - 디버깅/개발: "values"
   - 프로덕션 챗봇: "messages"
   - 실시간 업데이트: "updates"
    """)


# ============================================================================
# 예제 2: "values" 모드
# ============================================================================

def example_2_values_mode():
    """values 모드: 전체 상태 스트리밍"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 'values' 모드 - 전체 상태 반환")
    print("=" * 70)

    @tool
    def add_numbers(a: int, b: int) -> int:
        """두 숫자를 더합니다."""
        return a + b

    @tool
    def multiply_numbers(a: int, b: int) -> int:
        """두 숫자를 곱합니다."""
        return a * b

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[add_numbers, multiply_numbers],
        checkpointer=MemorySaver(),
    )

    print("\n🔹 'values' 모드로 실행:")
    print("-" * 70)

    user_message = "5와 3을 더하고, 그 결과에 2를 곱해주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "values_demo"}}
    step = 0

    # stream_mode="values" (기본값)
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="values"
    ):
        step += 1
        print(f"\n[Step {step}] 📦 전체 상태:")

        if "messages" in chunk:
            print(f"  총 메시지 수: {len(chunk['messages'])}")
            latest_message = chunk["messages"][-1]

            if hasattr(latest_message, "content"):
                content_preview = str(latest_message.content)[:60]
                print(f"  최신 메시지: {content_preview}...")

            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                print(f"  Tool 호출: {len(latest_message.tool_calls)}개")

    print("\n" + "-" * 70)
    print(f"✅ 총 {step}개 상태 업데이트 수신")
    print("💡 'values' 모드는 매 단계마다 전체 상태를 반환합니다.")


# ============================================================================
# 예제 3: "updates" 모드
# ============================================================================

def example_3_updates_mode():
    """updates 모드: 변경사항만 스트리밍"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 'updates' 모드 - 변경사항만 반환")
    print("=" * 70)

    @tool
    def search_database(query: str) -> str:
        """데이터베이스를 검색합니다."""
        time.sleep(0.5)
        return f"'{query}' 검색 결과: 3건 발견"

    @tool
    def format_results(data: str) -> str:
        """결과를 포맷팅합니다."""
        return f"포맷된 결과:\n{data}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_database, format_results],
        checkpointer=MemorySaver(),
    )

    print("\n🔹 'updates' 모드로 실행:")
    print("-" * 70)

    user_message = "LangChain을 검색하고 결과를 포맷팅해주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "updates_demo"}}
    update_count = 0

    # stream_mode="updates"
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="updates"
    ):
        update_count += 1
        print(f"\n[Update {update_count}] 🔄 변경사항:")

        # updates 모드는 노드별 변경사항을 반환
        if "messages" in chunk:
            new_messages = chunk["messages"]
            print(f"  새 메시지: {len(new_messages)}개 추가")

            for msg in new_messages:
                if hasattr(msg, "content") and msg.content:
                    content_preview = str(msg.content)[:50]
                    print(f"    • {content_preview}...")
                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    • Tool: {tc['name']}")

    print("\n" + "-" * 70)
    print(f"✅ 총 {update_count}개 업데이트 수신")
    print("💡 'updates' 모드는 변경된 부분만 전송하여 효율적입니다.")


# ============================================================================
# 예제 4: "messages" 모드
# ============================================================================

def example_4_messages_mode():
    """messages 모드: 새 메시지만 스트리밍"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 'messages' 모드 - 새 메시지만 반환")
    print("=" * 70)

    @tool
    def get_user_info(user_id: str) -> str:
        """사용자 정보를 가져옵니다."""
        return f"사용자 {user_id}: 김철수, 가입일 2024-01-01"

    @tool
    def get_order_history(user_id: str) -> str:
        """주문 내역을 가져옵니다."""
        return f"{user_id}의 주문: 총 5건, 최근 주문 2024-12-25"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_user_info, get_order_history],
        checkpointer=MemorySaver(),
    )

    print("\n🔹 'messages' 모드로 실행:")
    print("-" * 70)

    user_message = "사용자 USER123의 정보와 주문 내역을 조회해주세요."
    print(f"👤 사용자: {user_message}\n")

    config = {"configurable": {"thread_id": "messages_demo"}}
    message_count = 0

    print("💬 실시간 메시지 스트림:\n")

    # stream_mode="messages"
    for message_tuple in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
        stream_mode="messages"
    ):
        # messages 모드는 (message, metadata) 튜플 반환
        message, metadata = message_tuple
        message_count += 1

        # 메시지 타입에 따라 다르게 표시
        if hasattr(message, "content") and message.content:
            print(f"🤖 AI: {message.content}")

        elif hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"🔧 도구 호출: {tool_call['name']}")

        elif hasattr(message, "name"):  # Tool response
            print(f"📊 도구 결과: {message.content[:50]}...")

    print("\n" + "-" * 70)
    print(f"✅ 총 {message_count}개 메시지 수신")
    print("💡 'messages' 모드는 채팅 UI에 최적화되어 있습니다.")


# ============================================================================
# 예제 5: Stream Mode 비교 및 선택 가이드
# ============================================================================

def example_5_mode_comparison():
    """실전: Stream Mode 선택 가이드"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Stream Mode 비교 및 선택 가이드")
    print("=" * 70)

    @tool
    def simple_tool(text: str) -> str:
        """간단한 텍스트 처리"""
        return f"처리됨: {text}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[simple_tool],
        checkpointer=MemorySaver(),
    )

    user_message = "hello를 처리해주세요."

    # 세 가지 모드로 동일한 요청 실행
    modes = ["values", "updates", "messages"]

    for mode in modes:
        print(f"\n🔹 '{mode}' 모드:")
        print("-" * 70)

        config = {"configurable": {"thread_id": f"compare_{mode}"}}
        chunk_count = 0
        total_size = 0

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            stream_mode=mode
        ):
            chunk_count += 1
            chunk_size = len(str(chunk))
            total_size += chunk_size

            print(f"  Chunk {chunk_count}: {chunk_size} bytes")

        print(f"  ✅ 총 {chunk_count}개 청크, {total_size} bytes")

    print("\n" + "=" * 70)
    print("📊 Stream Mode 선택 가이드:")
    print("=" * 70)
    print("""
🎯 상황별 권장 모드:

1️⃣ 개발/디버깅:
   → "values" 모드
   이유: 전체 상태를 볼 수 있어 문제 파악 용이

2️⃣ 프로덕션 챗봇 UI:
   → "messages" 모드
   이유: 새 메시지만 받아 UI에 표시, 가장 효율적

3️⃣ 복잡한 상태 추적:
   → "updates" 모드
   이유: 변경사항만 받아 상태 병합, 네트워크 효율적

4️⃣ 실시간 대시보드:
   → "updates" 모드
   이유: 델타 업데이트로 화면 갱신

5️⃣ 로깅/모니터링:
   → "values" 모드
   이유: 각 단계의 완전한 스냅샷 저장
    """)

    print("\n💡 성능 최적화 팁:")
    print("  • messages 모드가 가장 경량")
    print("  • values 모드는 메모리 사용량 높음")
    print("  • updates 모드는 균형잡힌 선택")
    print("  • 프로덕션에서는 messages 권장")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 9: 프로덕션 - Stream Modes")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_stream_modes_overview()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_values_mode()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_updates_mode()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_messages_mode()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_mode_comparison()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 9-02: Stream Modes를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 03_custom_stream.py - Custom Streaming")
    print("  2. 04_hitl_basic.py - Human-in-the-Loop 기초")
    print("  3. 05_hitl_decisions.py - HITL 의사결정")
    print("\n📚 핵심 요약:")
    print("  • values: 전체 상태 반환 (디버깅)")
    print("  • updates: 변경사항만 반환 (효율적)")
    print("  • messages: 새 메시지만 반환 (챗봇)")
    print("  • 프로덕션에서는 messages 모드 권장")
    print("  • 상황에 맞는 모드 선택이 중요")
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
# 1. Stream Mode 내부 동작:
#    - values: 각 노드 실행 후 전체 상태 직렬화
#    - updates: 이전 상태와 비교하여 델타 계산
#    - messages: 메시지 리스트에서 새 항목만 추출
#
# 2. 성능 최적화:
#    - 큰 상태의 경우 updates/messages 사용
#    - 작은 상태는 values도 괜찮음
#    - 네트워크 대역폭 고려
#
# 3. 실전 활용:
#    - 채팅: messages 모드 + 실시간 UI 업데이트
#    - 모니터링: values 모드 + 로깅
#    - 대시보드: updates 모드 + 상태 병합
#
# ============================================================================
