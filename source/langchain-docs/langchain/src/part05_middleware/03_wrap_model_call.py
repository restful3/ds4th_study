"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 03_wrap_model_call.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 40분

📚 학습 목표:
  - wrap_model_call 훅 이해
  - 모델 호출 재시도 구현
  - 캐싱 및 변환 패턴 학습

📖 공식 문서:
  • Custom Middleware: /official/16-custom-middleware.md#wrap-style-hooks

📄 교안 문서:
  • Part 5.3.2: /docs/part05_middleware.md#32-wrap_model_call-훅

🚀 실행 방법:
  python 03_wrap_model_call.py

================================================================================
"""

import os
import time
import hashlib
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import Callable

load_dotenv()

# ============================================================================
# 예제 1: 모델 호출 재시도
# ============================================================================

def example_1_retry_model():
    """모델 호출 실패 시 재시도"""
    print("=" * 70)
    print("📌 예제 1: 모델 호출 재시도")
    print("=" * 70)

    attempt_count = {"count": 0}

    @wrap_model_call
    def retry_on_failure(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                attempt_count["count"] += 1
                print(f"\n🔄 시도 #{attempt + 1}")

                response = handler(request)
                print(f"✅ 성공!")
                return response

            except Exception as e:
                print(f"❌ 실패: {str(e)[:50]}...")

                if attempt == max_retries - 1:
                    print(f"⛔ 최대 재시도 횟수 초과")
                    raise

                # 지수 백오프
                wait_time = 2 ** attempt
                print(f"⏳ {wait_time}초 대기...")
                time.sleep(wait_time)

    @tool
    def simple_tool(text: str) -> str:
        """간단한 도구"""
        return f"처리됨: {text}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[simple_tool],
        middleware=[retry_on_failure],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "안녕"}]
    })

    print(f"\n📊 총 시도 횟수: {attempt_count['count']}")
    print(f"✅ 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 2: 모델 호출 로깅
# ============================================================================

def example_2_log_model_calls():
    """모델 호출 전후 로깅"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 모델 호출 로깅")
    print("=" * 70)

    @wrap_model_call
    def log_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        start_time = time.time()

        print(f"\n📥 모델 호출 시작")
        print(f"   메시지 수: {len(request.messages)}")

        response = handler(request)

        duration = time.time() - start_time
        print(f"📤 모델 응답 받음 ({duration:.2f}초)")

        return response

    @tool
    def calculator(expression: str) -> str:
        """계산합니다."""
        try:
            return str(eval(expression))
        except:
            return "오류"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[calculator],
        middleware=[log_call],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "100 * 50은?"}]
    })

    print(f"\n✅ 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 3: 간단한 캐싱
# ============================================================================

def example_3_caching():
    """모델 응답 캐싱"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 모델 응답 캐싱")
    print("=" * 70)

    cache = {}

    def hash_request(request: ModelRequest) -> str:
        """요청 해싱"""
        content = str([m.content for m in request.messages])
        return hashlib.md5(content.encode()).hexdigest()

    @wrap_model_call
    def cache_responses(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        cache_key = hash_request(request)

        if cache_key in cache:
            print(f"\n💾 캐시 히트! (키: {cache_key[:8]}...)")
            return cache[cache_key]

        print(f"\n🔍 캐시 미스 - 모델 호출 (키: {cache_key[:8]}...)")
        response = handler(request)

        cache[cache_key] = response
        return response

    @tool
    def get_data(query: str) -> str:
        """데이터 조회"""
        return f"{query}에 대한 데이터"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_data],
        middleware=[cache_responses],
    )

    # 첫 번째 호출
    print("\n🔹 첫 번째 요청:")
    response1 = agent.invoke({
        "messages": [{"role": "user", "content": "파이썬이 뭐야?"}]
    })

    # 동일한 호출 (캐시됨)
    print("\n🔹 두 번째 요청 (동일):")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "파이썬이 뭐야?"}]
    })

    print(f"\n📊 캐시 크기: {len(cache)}")


# ============================================================================
# 예제 4: 요청 변환
# ============================================================================

def example_4_transform_request():
    """모델 요청을 변환"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 요청 변환")
    print("=" * 70)

    @wrap_model_call
    def add_context(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 모든 사용자 메시지에 컨텍스트 추가
        modified_messages = []

        for msg in request.messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if msg.type == "human":
                    modified_content = f"[중요] {msg.content}"
                    print(f"\n✏️ 메시지 변환: {msg.content} → {modified_content}")

                    # 새 메시지 객체 생성
                    from langchain_core.messages import HumanMessage
                    modified_messages.append(HumanMessage(content=modified_content))
                else:
                    modified_messages.append(msg)
            else:
                modified_messages.append(msg)

        # 변환된 요청으로 모델 호출
        modified_request = ModelRequest(messages=modified_messages)
        return handler(modified_request)

    @tool
    def process(text: str) -> str:
        """처리"""
        return f"처리됨: {text}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[process],
        middleware=[add_context],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "안녕하세요"}]
    })

    print(f"\n✅ 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 5: 조건부 모델 호출 (단락)
# ============================================================================

def example_5_short_circuit():
    """특정 조건에서 모델 호출 건너뛰기"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 조건부 모델 호출")
    print("=" * 70)

    @wrap_model_call
    def skip_simple_questions(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 마지막 메시지 확인
        last_msg = request.messages[-1] if request.messages else None

        if last_msg and hasattr(last_msg, 'content'):
            content = last_msg.content.lower()

            # 간단한 인사는 모델 호출 없이 응답
            if content in ["안녕", "hi", "hello"]:
                print(f"\n⚡ 간단한 인사 감지 - 모델 호출 건너뛰기")

                from langchain_core.messages import AIMessage
                return ModelResponse(message=AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"))

        print(f"\n🔍 일반 질문 - 모델 호출")
        return handler(request)

    @tool
    def help_tool() -> str:
        """도움말"""
        return "도움말 내용"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[help_tool],
        middleware=[skip_simple_questions],
    )

    # 간단한 인사
    print("\n🔹 간단한 인사:")
    response1 = agent.invoke({
        "messages": [{"role": "user", "content": "안녕"}]
    })
    print(f"✅ 응답: {response1['messages'][-1].content}")

    # 복잡한 질문
    print("\n🔹 복잡한 질문:")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "파이썬 설명해줘"}]
    })
    print(f"✅ 응답: {response2['messages'][-1].content[:60]}...")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - wrap_model_call")
    print("\n")

    example_1_retry_model()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_log_model_calls()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_caching()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_transform_request()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_short_circuit()

    print("\n" + "=" * 70)
    print("🎉 Part 5-3 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. wrap_model_call의 강력함:
#    - 재시도 로직 구현
#    - 캐싱으로 비용 절감
#    - 요청/응답 변환
#    - 조건부 모델 호출 (단락)
#
# 2. handler 함수:
#    - 실제 모델 호출을 수행
#    - 여러 번 호출 가능 (재시도)
#    - 호출하지 않을 수도 있음 (단락)
#
# 3. 실무 활용:
#    - API 장애 대응 (재시도)
#    - 비용 최적화 (캐싱)
#    - A/B 테스팅
#
# ============================================================================
