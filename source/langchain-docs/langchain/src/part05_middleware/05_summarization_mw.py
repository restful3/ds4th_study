"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: 미들웨어 (Middleware)
================================================================================

파일명: 05_summarization_mw.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 30분

📚 학습 목표:
  - Summarization 미들웨어 사용
  - 토큰 관리 및 대화 요약
  - 긴 대화 처리 전략

📖 공식 문서:
  • Built-in Middleware: /official/15-built-in-middleware.md#summarization

🚀 실행 방법:
  python 05_summarization_mw.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

load_dotenv()

# ============================================================================
# 예제 1: 기본 Summarization 미들웨어
# ============================================================================

def example_1_basic_summarization():
    """기본 요약 미들웨어 사용"""
    print("=" * 70)
    print("📌 예제 1: 기본 Summarization 미들웨어")
    print("=" * 70)

    @tool
    def get_info(topic: str) -> str:
        """정보 조회"""
        return f"{topic}에 대한 상세 정보입니다. " * 20  # 긴 응답

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_info],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("messages", 10),  # 10개 메시지 도달 시
                keep=("messages", 5),       # 최근 5개 유지
            ),
        ],
    )

    print("\n🤖 긴 대화 시뮬레이션")

    # 여러 메시지 추가
    messages = [{"role": "user", "content": "안녕"}]

    for i in range(12):
        messages.append({"role": "user", "content": f"질문 {i+1}"})
        messages.append({"role": "assistant", "content": f"답변 {i+1}"})

    response = agent.invoke({"messages": messages})

    print(f"\n✅ 요약이 적용되어 대화가 관리되었습니다")
    print(f"📊 최종 메시지 수: {len(response['messages'])}")


# ============================================================================
# 예제 2: 토큰 기반 트리거
# ============================================================================

def example_2_token_trigger():
    """토큰 수 기반 요약 트리거"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 토큰 기반 요약")
    print("=" * 70)

    @tool
    def long_response(query: str) -> str:
        """긴 응답 생성"""
        return "이것은 매우 긴 응답입니다. " * 100

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[long_response],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("tokens", 2000),  # 2000 토큰 초과 시
                keep=("messages", 10),
            ),
        ],
    )

    print("\n🤖 토큰 많은 대화 시뮬레이션")

    messages = []
    for i in range(15):
        messages.append({"role": "user", "content": "긴 정보 요청"})
        messages.append({"role": "assistant", "content": "긴 응답" * 50})

    response = agent.invoke({"messages": messages})

    print(f"\n✅ 토큰 기반 요약 완료")


# ============================================================================
# 예제 3: 커스텀 요약 프롬프트
# ============================================================================

def example_3_custom_prompt():
    """커스텀 요약 프롬프트 사용"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 커스텀 요약 프롬프트")
    print("=" * 70)

    CUSTOM_SUMMARY_PROMPT = """
    다음 대화를 3줄 이내로 요약하세요:

    {messages}

    중요한 정보만 포함하고, 불필요한 내용은 제외하세요.
    """

    @tool
    def simple_tool(x: str) -> str:
        return f"처리: {x}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[simple_tool],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("messages", 8),
                keep=("messages", 3),
                summary_prompt=CUSTOM_SUMMARY_PROMPT,
            ),
        ],
    )

    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"요청 {i}"})
        messages.append({"role": "assistant", "content": f"응답 {i}"})

    response = agent.invoke({"messages": messages})

    print(f"\n✅ 커스텀 프롬프트로 요약 완료")


# ============================================================================
# 예제 4: 다중 조건 트리거
# ============================================================================

def example_4_multiple_triggers():
    """여러 조건으로 요약 트리거"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 다중 조건 트리거")
    print("=" * 70)

    @tool
    def info_tool(topic: str) -> str:
        return f"{topic} 정보"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[info_tool],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                # 메시지 15개 OR 토큰 3000개 초과 시
                trigger=[("messages", 15), ("tokens", 3000)],
                keep=("messages", 8),
            ),
        ],
    )

    print("\n🤖 다중 조건 시뮬레이션")
    print("   조건1: 메시지 15개 초과")
    print("   조건2: 토큰 3000개 초과")

    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"질문 {i}"})
        messages.append({"role": "assistant", "content": f"답변 {i}"})

    response = agent.invoke({"messages": messages})

    print(f"\n✅ 다중 조건 중 하나가 트리거됨")


# ============================================================================
# 예제 5: 실전 사용 예시
# ============================================================================

def example_5_production_use():
    """실전 요약 미들웨어 설정"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 요약 설정")
    print("=" * 70)

    @tool
    def search(query: str) -> str:
        """검색 도구"""
        return f"{query}에 대한 검색 결과"

    @tool
    def calculator(expr: str) -> str:
        """계산기"""
        try:
            return str(eval(expr))
        except:
            return "오류"

    # 프로덕션 설정 예시
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search, calculator],
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",  # 저렴한 모델로 요약
                trigger=("tokens", 4000),  # 컨텍스트 윈도우 80% 시점
                keep=("messages", 20),     # 최근 20개 메시지 유지
                trim_tokens_to_summarize=4000,  # 요약 시 최대 토큰
            ),
        ],
    )

    print("\n✅ 프로덕션 요약 설정:")
    print("   • 모델: gpt-4o-mini (비용 효율)")
    print("   • 트리거: 4000 토큰")
    print("   • 유지: 최근 20개 메시지")
    print("   • 요약 최대: 4000 토큰")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 5: 미들웨어 - Summarization")
    print("\n")

    example_1_basic_summarization()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_token_trigger()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_custom_prompt()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_multiple_triggers()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_production_use()

    print("\n" + "=" * 70)
    print("🎉 Part 5-5 완료!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 요약 전략:
#    - 메시지 수 기반: 대화 턴 수 제한
#    - 토큰 기반: 실제 컨텍스트 윈도우 관리
#    - 다중 조건: OR 로직으로 유연하게
#
# 2. 프로덕션 팁:
#    - trigger를 컨텍스트 윈도우의 70-80%로 설정
#    - 저렴한 모델로 요약 (gpt-4o-mini)
#    - keep은 충분히 크게 (중요 정보 보존)
#
# 3. 주의사항:
#    - 요약 시 일부 컨텍스트 손실 가능
#    - 중요한 정보는 keep 범위 내 유지
#    - 요약 모델 비용 고려
#
# ============================================================================
