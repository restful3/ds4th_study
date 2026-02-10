"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 06_tool_retry.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 40분

📚 학습 목표:
  - Tool Retry 미들웨어 사용
  - 실패한 도구 자동 재시도
  - 지수 백오프 패턴 학습

📖 공식 문서:
  • Built-in Middleware: /official/15-built-in-middleware.md#tool-retry

🚀 실행 방법:
  python 06_tool_retry.py

================================================================================
"""

import os
import time
import random
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.tools import tool

load_dotenv()

# ============================================================================
# 예제 1: 기본 Tool Retry 미들웨어
# ============================================================================

def example_1_basic_retry():
    """기본 도구 재시도"""
    print("=" * 70)
    print("📌 예제 1: 기본 Tool Retry")
    print("=" * 70)

    attempt_count = {"count": 0}

    @tool
    def unreliable_api(query: str) -> str:
        """불안정한 API (30% 성공률)"""
        attempt_count["count"] += 1
        print(f"\n🔄 API 호출 시도 #{attempt_count['count']}")

        if random.random() < 0.3:  # 30% 성공
            print(f"✅ 성공!")
            return f"{query}에 대한 결과"

        print(f"❌ 실패 - API 오류")
        raise Exception("API connection failed")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[unreliable_api],
        middleware=[
            ToolRetryMiddleware(
                max_retries=5,
                backoff_factor=1.0,  # 1초씩 증가
            ),
        ],
    )

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "데이터 조회해줘"}]
        })
        print(f"\n✅ 최종 성공! (총 {attempt_count['count']}번 시도)")
        print(f"📝 응답: {response['messages'][-1].content}")
    except Exception as e:
        print(f"\n⛔ 최종 실패: {e}")


# ============================================================================
# 예제 2: 지수 백오프
# ============================================================================

def example_2_exponential_backoff():
    """지수 백오프를 사용한 재시도"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 지수 백오프")
    print("=" * 70)

    @tool
    def flaky_service(data: str) -> str:
        """불안정한 서비스"""
        if random.random() < 0.4:
            return f"처리됨: {data}"
        raise Exception("Service temporarily unavailable")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[flaky_service],
        middleware=[
            ToolRetryMiddleware(
                max_retries=4,
                backoff_factor=2.0,  # 2^n 지수 백오프
                # 대기 시간: 2s, 4s, 8s, 16s
            ),
        ],
    )

    print("\n🔄 지수 백오프 패턴:")
    print("   1차 실패 → 2초 대기")
    print("   2차 실패 → 4초 대기")
    print("   3차 실패 → 8초 대기")
    print("   4차 실패 → 16초 대기")

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "서비스 호출"}]
        })
        print(f"\n✅ 성공!")
    except:
        print(f"\n⛔ 최대 재시도 초과")


# ============================================================================
# 예제 3: 특정 에러만 재시도
# ============================================================================

def example_3_selective_retry():
    """특정 에러만 재시도 (커스텀 구현)"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 선택적 재시도")
    print("=" * 70)

    from langchain.agents.middleware import wrap_tool_call, ToolRequest, ToolResponse
    from typing import Callable

    @wrap_tool_call
    def selective_retry(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        max_retries = 3
        retryable_errors = ["timeout", "connection", "rate limit"]

        for attempt in range(max_retries):
            try:
                return handler(request)
            except Exception as e:
                error_msg = str(e).lower()

                # 재시도 가능한 에러인지 확인
                should_retry = any(err in error_msg for err in retryable_errors)

                if not should_retry:
                    print(f"\n⛔ 재시도 불가능한 에러: {e}")
                    raise

                if attempt == max_retries - 1:
                    print(f"\n⛔ 최대 재시도 초과")
                    raise

                print(f"\n🔄 재시도 가능한 에러 - 시도 #{attempt + 2}")
                time.sleep(2 ** attempt)

    @tool
    def api_with_errors(action: str) -> str:
        """다양한 에러를 발생시키는 API"""
        error_type = random.choice(["timeout", "invalid_input", "connection"])

        if random.random() < 0.3:
            return f"{action} 완료"

        if error_type == "invalid_input":
            raise Exception("Invalid input - 재시도 불가")
        else:
            raise Exception(f"{error_type} error - 재시도 가능")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[api_with_errors],
        middleware=[selective_retry],
    )

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "작업 실행"}]
        })
        print(f"\n✅ 성공!")
    except Exception as e:
        print(f"\n⛔ 실패: {e}")


# ============================================================================
# 예제 4: 재시도 로깅
# ============================================================================

def example_4_retry_logging():
    """재시도 과정 로깅"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 재시도 로깅")
    print("=" * 70)

    retry_logs = []

    from langchain.agents.middleware import wrap_tool_call, ToolRequest, ToolResponse
    from typing import Callable

    @wrap_tool_call
    def logged_retry(
        request: ToolRequest,
        handler: Callable[[ToolRequest], ToolResponse],
    ) -> ToolResponse:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                log_entry = {
                    "tool": request.tool_name,
                    "attempt": attempt + 1,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "status": "attempting"
                }

                result = handler(request)

                log_entry["status"] = "success"
                retry_logs.append(log_entry)

                return result

            except Exception as e:
                log_entry["status"] = "failed"
                log_entry["error"] = str(e)
                retry_logs.append(log_entry)

                if attempt == max_retries - 1:
                    raise

                time.sleep(1)

    @tool
    def random_fail(x: int) -> int:
        """랜덤 실패"""
        if random.random() < 0.4:
            return x * 2
        raise Exception("Random failure")

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[random_fail],
        middleware=[logged_retry],
    )

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": "10 처리해줘"}]
        })

        print(f"\n📊 재시도 로그:")
        for log in retry_logs:
            print(f"   {log['timestamp']} | 시도 #{log['attempt']} | {log['status']}")

    except:
        print(f"\n📊 재시도 로그 (실패):")
        for log in retry_logs:
            print(f"   {log['timestamp']} | 시도 #{log['attempt']} | {log['status']}")


# ============================================================================
# 예제 5: 프로덕션 재시도 전략
# ============================================================================

def example_5_production_strategy():
    """실전 재시도 전략"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 프로덕션 재시도 전략")
    print("=" * 70)

    @tool
    def external_api(query: str) -> str:
        """외부 API"""
        if random.random() < 0.5:
            return f"{query} 결과"
        raise Exception("API error")

    # 프로덕션 설정
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[external_api],
        middleware=[
            ToolRetryMiddleware(
                max_retries=3,          # 3번까지 재시도
                backoff_factor=2.0,     # 지수 백오프
                # 1차: 2초, 2차: 4초, 3차: 8초
            ),
        ],
    )

    print("\n✅ 프로덕션 재시도 설정:")
    print("   • 최대 재시도: 3번")
    print("   • 백오프: 지수 (2초, 4초, 8초)")
    print("   • 총 최대 시간: ~14초")

    print("\n💡 프로덕션 팁:")
    print("   1. 재시도는 3-5번이 적당")
    print("   2. 지수 백오프로 서버 부하 분산")
    print("   3. 특정 에러만 재시도 (타임아웃, 연결 오류)")
    print("   4. 재시도 로그 남기기")
    print("   5. Circuit Breaker 패턴 고려")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - Tool Retry")
    print("\n")

    example_1_basic_retry()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_exponential_backoff()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_selective_retry()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_retry_logging()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_production_strategy()

    print("\n" + "=" * 70)
    print("🎉 Part 5-6 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 재시도 전략:
#    - 고정 간격: 매번 동일한 시간 대기
#    - 선형 백오프: 1s, 2s, 3s, 4s...
#    - 지수 백오프: 2s, 4s, 8s, 16s... (권장)
#
# 2. 언제 재시도?
#    - ✅ 타임아웃
#    - ✅ 네트워크 오류
#    - ✅ Rate Limit
#    - ❌ 잘못된 입력
#    - ❌ 권한 오류
#
# 3. Circuit Breaker:
#    - 연속 실패 시 일정 시간 차단
#    - 서버 과부하 방지
#    - 빠른 실패 (Fail-fast)
#
# ============================================================================
