"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 07_guardrails.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 40분

📚 학습 목표:
  - Guardrails 개념 이해
  - 입출력 검증 구현
  - 안전한 Agent 구축

📖 공식 문서:
  • Guardrails: /official/17-guardrails.md
  • PII Detection: /official/15-built-in-middleware.md#pii-detection

🚀 실행 방법:
  python 07_guardrails.py

================================================================================
"""

import os
import re
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.tools import tool
from langgraph.runtime import Runtime
from typing import Any

load_dotenv()

# ============================================================================
# 예제 1: 콘텐츠 필터링 (입력 검증)
# ============================================================================

def example_1_content_filter():
    """부적절한 입력 차단"""
    print("=" * 70)
    print("📌 예제 1: 콘텐츠 필터링")
    print("=" * 70)

    BLOCKED_KEYWORDS = ["해킹", "불법", "폭력", "위험한"]

    @before_model(can_jump_to=["end"])
    def filter_input(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]

        if hasattr(last_message, 'content'):
            content = last_message.content.lower()

            for keyword in BLOCKED_KEYWORDS:
                if keyword in content:
                    print(f"\n⛔ 차단된 키워드 감지: '{keyword}'")
                    return {
                        "messages": [{
                            "role": "assistant",
                            "content": "부적절한 요청입니다. 다른 질문을 해주세요."
                        }],
                        "jump_to": "end"
                    }

        return None

    @tool
    def safe_tool(query: str) -> str:
        """안전한 도구"""
        return f"{query} 처리 완료"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[safe_tool],
        middleware=[filter_input],
    )

    # 안전한 입력
    print("\n✅ 안전한 입력 테스트:")
    response1 = agent.invoke({
        "messages": [{"role": "user", "content": "파이썬 설명해줘"}]
    })
    print(f"응답: {response1['messages'][-1].content[:50]}...")

    # 차단될 입력
    print("\n⛔ 차단될 입력 테스트:")
    response2 = agent.invoke({
        "messages": [{"role": "user", "content": "해킹 방법 알려줘"}]
    })
    print(f"응답: {response2['messages'][-1].content}")


# ============================================================================
# 예제 2: PII (개인정보) 탐지
# ============================================================================

def example_2_pii_detection():
    """개인정보 탐지 및 제거"""
    print("\n" + "=" * 70)
    print("📌 예제 2: PII 탐지")
    print("=" * 70)

    # 간단한 PII 패턴
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'\b\d{3}-\d{4}-\d{4}\b'
    SSN_PATTERN = r'\b\d{6}-\d{7}\b'

    @before_model
    def detect_pii(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]

        if hasattr(last_message, 'content'):
            content = last_message.content

            # PII 탐지
            if re.search(EMAIL_PATTERN, content):
                print(f"\n⚠️ 이메일 주소 감지!")
                content = re.sub(EMAIL_PATTERN, '[EMAIL]', content)

            if re.search(PHONE_PATTERN, content):
                print(f"\n⚠️ 전화번호 감지!")
                content = re.sub(PHONE_PATTERN, '[PHONE]', content)

            if re.search(SSN_PATTERN, content):
                print(f"\n⚠️ 주민번호 감지!")
                content = re.sub(SSN_PATTERN, '[SSN]', content)

            # 변경된 메시지로 교체
            if content != last_message.content:
                messages = state["messages"][:-1] + [{
                    "role": last_message.type,
                    "content": content
                }]
                return {"messages": messages}

        return None

    @tool
    def user_info(data: str) -> str:
        """사용자 정보 처리"""
        return f"정보 처리됨: {data}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[user_info],
        middleware=[detect_pii],
    )

    print("\n📧 PII 포함 입력 테스트:")
    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "제 이메일은 user@example.com이고 전화번호는 010-1234-5678입니다"
        }]
    })

    print(f"\n✅ PII가 제거되어 안전하게 처리되었습니다")


# ============================================================================
# 예제 3: 출력 검증
# ============================================================================

def example_3_output_validation():
    """Agent 출력 검증"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 출력 검증")
    print("=" * 70)

    MAX_LENGTH = 100

    @after_model
    def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]

        if hasattr(last_message, 'content') and last_message.type == "ai":
            content = last_message.content

            # 길이 검증
            if len(content) > MAX_LENGTH:
                print(f"\n✂️ 출력이 너무 깁니다 ({len(content)} > {MAX_LENGTH})")
                truncated = content[:MAX_LENGTH] + "... (생략)"

                messages = state["messages"][:-1] + [{
                    "role": "assistant",
                    "content": truncated
                }]
                return {"messages": messages}

        return None

    @tool
    def long_response(topic: str) -> str:
        """긴 응답 생성"""
        return f"{topic}에 대한 설명입니다. " * 50

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[long_response],
        middleware=[validate_output],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "긴 설명 요청"}]
    })

    print(f"\n✅ 출력 길이 제한 적용됨")
    print(f"📝 응답: {response['messages'][-1].content[:80]}...")


# ============================================================================
# 예제 4: Rate Limiting
# ============================================================================

def example_4_rate_limiting():
    """API 호출 횟수 제한"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Rate Limiting")
    print("=" * 70)

    import time

    class RateLimiter:
        def __init__(self, max_calls: int, window_seconds: int):
            self.max_calls = max_calls
            self.window_seconds = window_seconds
            self.calls = []

        def is_allowed(self) -> bool:
            now = time.time()
            # 윈도우 내의 호출만 유지
            self.calls = [t for t in self.calls if now - t < self.window_seconds]

            if len(self.calls) >= self.max_calls:
                return False

            self.calls.append(now)
            return True

    limiter = RateLimiter(max_calls=3, window_seconds=60)

    @before_model(can_jump_to=["end"])
    def rate_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not limiter.is_allowed():
            print(f"\n⛔ Rate Limit 초과 ({limiter.max_calls}회/{limiter.window_seconds}초)")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "요청이 너무 많습니다. 잠시 후 다시 시도하세요."
                }],
                "jump_to": "end"
            }

        print(f"\n✅ Rate Limit 통과 ({len(limiter.calls)}/{limiter.max_calls})")
        return None

    @tool
    def api_call(data: str) -> str:
        """API 호출"""
        return f"처리: {data}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[api_call],
        middleware=[rate_limit],
    )

    # 여러 번 호출
    for i in range(5):
        print(f"\n🔹 호출 #{i+1}:")
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": f"요청 {i+1}"}]
            })
            print(f"✅ 성공")
        except:
            print(f"⛔ 차단됨")


# ============================================================================
# 예제 5: 종합 Guardrails
# ============================================================================

def example_5_comprehensive_guardrails():
    """종합적인 안전장치"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 종합 Guardrails")
    print("=" * 70)

    # 1. 입력 필터
    @before_model(can_jump_to=["end"])
    def input_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            if any(word in last_msg.content.lower() for word in ["해킹", "불법"]):
                return {
                    "messages": [{"role": "assistant", "content": "차단됨"}],
                    "jump_to": "end"
                }
        return None

    # 2. PII 제거
    @before_model
    def pii_remover(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content'):
            content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', last_msg.content)
            if content != last_msg.content:
                messages = state["messages"][:-1] + [{"role": last_msg.type, "content": content}]
                return {"messages": messages}
        return None

    # 3. 출력 검증
    @after_model
    def output_validator(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, 'content') and len(last_msg.content) > 200:
            messages = state["messages"][:-1] + [{
                "role": "assistant",
                "content": last_msg.content[:200] + "..."
            }]
            return {"messages": messages}
        return None

    @tool
    def process_data(data: str) -> str:
        """데이터 처리"""
        return f"처리됨: {data}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[process_data],
        middleware=[
            input_filter,     # 1. 입력 필터링
            pii_remover,      # 2. PII 제거
            output_validator, # 3. 출력 검증
        ],
    )

    print("\n✅ 종합 Guardrails 적용:")
    print("   1. 입력 필터링 (부적절한 키워드 차단)")
    print("   2. PII 제거 (개인정보 보호)")
    print("   3. 출력 검증 (길이 제한)")

    print("\n💡 프로덕션 Guardrails 권장 사항:")
    print("   • 콘텐츠 필터 (입력/출력)")
    print("   • PII 탐지 및 제거")
    print("   • Rate Limiting")
    print("   • 출력 길이 제한")
    print("   • 민감한 도구 호출 승인")
    print("   • 로깅 및 모니터링")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - Guardrails")
    print("\n")

    example_1_content_filter()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_pii_detection()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_output_validation()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_rate_limiting()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_comprehensive_guardrails()

    print("\n" + "=" * 70)
    print("🎉 Part 5-7 완료! Part 5 전체 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. Guardrails 레이어:
#    - 입력 검증: 요청 시작 전
#    - 처리 제어: 실행 중
#    - 출력 검증: 응답 전
#
# 2. 보안 체크리스트:
#    - ✅ 부적절한 콘텐츠 차단
#    - ✅ PII 보호
#    - ✅ Rate Limiting
#    - ✅ 출력 길이 제한
#    - ✅ 민감한 작업 승인
#
# 3. 프로덕션 패턴:
#    - Defense in Depth (다층 방어)
#    - Fail-safe (안전한 실패)
#    - 로깅 및 알림
#    - 정기 감사
#
# ============================================================================
