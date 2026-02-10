"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 04_wrap_tool_call.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 40분

📚 학습 목표:
  - wrap_tool_call 훅 이해
  - 도구 호출 제어 및 로깅
  - 도구 재시도 구현

📖 공식 문서:
  • Custom Middleware: /official/16-custom-middleware.md#wrap-style-hooks

🚀 실행 방법:
  python 04_wrap_tool_call.py

================================================================================
"""

import os
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, ToolRequest, ToolResponse
from langchain.tools import tool
from typing import Callable

load_dotenv()

# ============================================================================
# 예제 1: 도구 실행 시간 측정
# ============================================================================

def example_1_measure_time():
    """도구 실행 시간 측정"""
    print("=" * 70)
    print("📌 예제 1: 도구 실행 시간 측정")
    print("=" * 70)

    @wrap_tool_call
    def measure_execution(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        print(f"\n⏱️ 도구 시작: {request.tool_name}")
        start_time = time.time()

        result = handler(request)

        duration = time.time() - start_time
        print(f"✅ 도구 완료: {request.tool_name} ({duration:.3f}초)")

        return result

    @tool
    def slow_calculation(x: int) -> int:
        """느린 계산"""
        time.sleep(0.5)  # 시뮬레이션
        return x * 2

    @tool
    def fast_calculation(x: int) -> int:
        """빠른 계산"""
        return x + 10

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[slow_calculation, fast_calculation],
        middleware=[measure_execution],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "5를 2배 하고 10을 더해줘"}]
    })

    print(f"\n✅ 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 2: 도구 재시도
# ============================================================================

def example_2_retry_tool():
    """실패한 도구 호출 재시도"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 도구 재시도")
    print("=" * 70)

    @wrap_tool_call
    def retry_tool(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                print(f"\n🔄 시도 #{attempt + 1}: {request.tool_name}")
                return handler(request)
            except Exception as e:
                print(f"❌ 실패: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)

    @tool
    def unreliable_api(query: str) -> str:
        """불안정한 API (데모용)"""
        import random
        if random.random() < 0.3:  # 30% 성공
            return f"{query}에 대한 결과"
        raise Exception("API 연결 실패")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[unreliable_api],
        middleware=[retry_tool],
    )

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "데이터 조회해줘"}]
        })
        print(f"\n✅ 응답: {response['messages'][-1].content}")
    except Exception as e:
        print(f"\n⛔ 최종 실패: {e}")


# ============================================================================
# 예제 3: 도구 호출 로깅
# ============================================================================

def example_3_log_tools():
    """모든 도구 호출 로깅"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 도구 호출 로깅")
    print("=" * 70)

    tool_logs = []

    @wrap_tool_call
    def log_tool(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        log_entry = {
            "tool": request.tool_name,
            "args": str(request.tool_input),
            "timestamp": time.strftime("%H:%M:%S")
        }

        result = handler(request)

        log_entry["result"] = str(result)[:50]
        tool_logs.append(log_entry)

        print(f"\n📝 로그: {log_entry['tool']} | {log_entry['args']}")

        return result

    @tool
    def add(a: int, b: int) -> int:
        """더하기"""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """곱하기"""
        return a * b

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[add, multiply],
        middleware=[log_tool],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "5 + 3을 하고, 결과에 2를 곱해줘"}]
    })

    print(f"\n📊 총 도구 호출: {len(tool_logs)}개")
    for log in tool_logs:
        print(f"  - {log['tool']}: {log['args']}")


# ============================================================================
# 예제 4: 도구 실행 제한
# ============================================================================

def example_4_limit_tool():
    """도구 실행 횟수 제한"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 도구 실행 제한")
    print("=" * 70)

    call_count = {"count": 0}
    MAX_CALLS = 3

    @wrap_tool_call
    def limit_calls(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        call_count["count"] += 1

        if call_count["count"] > MAX_CALLS:
            print(f"\n⚠️ 도구 호출 제한 초과 ({call_count['count']}/{MAX_CALLS})")
            raise Exception("도구 호출 제한 초과")

        print(f"\n✅ 도구 호출 허용 ({call_count['count']}/{MAX_CALLS})")
        return handler(request)

    @tool
    def search(query: str) -> str:
        """검색"""
        return f"{query} 검색 결과"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search],
        middleware=[limit_calls],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "여러 가지를 검색해줘"}]
    })

    print(f"\n📊 총 호출 횟수: {call_count['count']}")


# ============================================================================
# 예제 5: 도구 결과 변환
# ============================================================================

def example_5_transform_result():
    """도구 결과를 변환"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 도구 결과 변환")
    print("=" * 70)

    @wrap_tool_call
    def uppercase_result(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        result = handler(request)

        # 결과를 대문자로 변환
        if isinstance(result, str):
            transformed = result.upper()
            print(f"\n✏️ 결과 변환: {result} → {transformed}")
            return transformed

        return result

    @tool
    def get_message(msg: str) -> str:
        """메시지 반환"""
        return f"message: {msg}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_message],
        middleware=[uppercase_result],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "hello 메시지 보내줘"}]
    })

    print(f"\n✅ 응답: {response['messages'][-1].content}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - wrap_tool_call")
    print("\n")

    example_1_measure_time()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_retry_tool()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_log_tools()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_limit_tool()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_transform_result()

    print("\n" + "=" * 70)
    print("🎉 Part 5-4 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. wrap_tool_call 활용:
#    - 실행 시간 측정
#    - 재시도 로직
#    - 호출 로깅
#    - 결과 변환
#
# 2. handler 제어:
#    - 여러 번 호출 (재시도)
#    - 호출 전 검증
#    - 호출 후 변환
#
# 3. 실무 패턴:
#    - API Rate Limiting
#    - 비용 추적
#    - 보안 검사
#
# ============================================================================
