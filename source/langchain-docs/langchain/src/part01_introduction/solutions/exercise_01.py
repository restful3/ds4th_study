"""
================================================================================
LangChain AI Agent 마스터 교안
Part 1: AI Agent의 이해 - 실습 과제 1 해답
================================================================================

과제: 다양한 LLM 프로바이더로 간단한 챗봇 만들기

요구사항:
1. OpenAI, Anthropic, Google 중 2개 이상의 프로바이더 사용
2. 각 프로바이더의 응답을 비교
3. System Message로 챗봇의 성격 지정
4. 사용자 입력을 받아 응답 출력

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# 환경 설정
load_dotenv()

# ============================================================================
# 솔루션 1: 여러 LLM 프로바이더 사용
# ============================================================================

def create_chatbot(provider: str, personality: str):
    """
    다양한 프로바이더로 챗봇을 생성합니다.

    Args:
        provider: "openai", "anthropic", "google" 중 하나
        personality: 챗봇의 성격 (예: "친절한", "전문적인", "유머러스한")

    Returns:
        설정된 LLM 객체
    """
    system_prompts = {
        "친절한": "당신은 매우 친절하고 공감 능력이 뛰어난 챗봇입니다. 😊",
        "전문적인": "당신은 전문적이고 정확한 정보를 제공하는 챗봇입니다.",
        "유머러스한": "당신은 유머 감각이 뛰어나고 재미있는 챗봇입니다. 😄",
        "간결한": "당신은 간결하고 명확하게 답변하는 챗봇입니다.",
    }

    system_message = system_prompts.get(personality, system_prompts["친절한"])

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        ), system_message

    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            return None
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
        ), system_message

    elif provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
        ), system_message

    else:
        raise ValueError(f"지원하지 않는 프로바이더: {provider}")


def compare_responses(question: str, personalities=["친절한", "전문적인"]):
    """
    여러 프로바이더와 성격으로 응답을 비교합니다.

    Args:
        question: 질문 내용
        personalities: 테스트할 성격 목록
    """
    providers = ["openai", "anthropic", "google"]
    available_providers = []

    # 사용 가능한 프로바이더 확인
    for provider in providers:
        result = create_chatbot(provider, "친절한")
        if result is not None:
            available_providers.append(provider)

    if len(available_providers) < 2:
        print("⚠️ 최소 2개의 LLM 프로바이더 API 키가 필요합니다.")
        print("설정된 프로바이더:", available_providers)
        return

    print("=" * 80)
    print(f"🤔 질문: {question}")
    print("=" * 80)
    print()

    # 각 프로바이더와 성격 조합으로 테스트
    for personality in personalities:
        print(f"\n{'━' * 80}")
        print(f"🎭 성격: {personality}")
        print('━' * 80)

        for provider in available_providers[:2]:  # 처음 2개만 사용
            llm, system_message = create_chatbot(provider, personality)

            if llm is None:
                continue

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=question)
            ]

            print(f"\n🤖 {provider.upper()}:")
            try:
                response = llm.invoke(messages)
                print(response.content)
            except Exception as e:
                print(f"❌ 오류: {e}")

        print()


# ============================================================================
# 솔루션 2: 대화형 챗봇
# ============================================================================

def interactive_chatbot(provider="openai", personality="친절한"):
    """
    대화형 챗봇을 실행합니다.

    Args:
        provider: LLM 프로바이더
        personality: 챗봇 성격
    """
    result = create_chatbot(provider, personality)

    if result is None:
        print(f"❌ {provider.upper()} API 키가 설정되지 않았습니다.")
        return

    llm, system_message = result

    print("=" * 80)
    print(f"🤖 {personality} 챗봇 ({provider.upper()})")
    print("=" * 80)
    print()
    print("💬 대화를 시작하세요! (종료하려면 'quit' 입력)")
    print()

    conversation_history = [SystemMessage(content=system_message)]

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() in ['quit', 'exit', '종료']:
            print("\n👋 대화를 종료합니다!")
            break

        if not user_input:
            continue

        # 사용자 메시지 추가
        conversation_history.append(HumanMessage(content=user_input))

        try:
            # LLM 호출
            response = llm.invoke(conversation_history)

            # 응답 출력
            print(f"\n🤖 Bot: {response.content}\n")

            # 대화 이력에 추가
            conversation_history.append(response)

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""

    print("=" * 80)
    print("✅ Part 1 실습 과제 1 - 해답")
    print("=" * 80)
    print()

    # 예제 1: 응답 비교
    print("📊 예제 1: 여러 프로바이더 응답 비교\n")

    compare_responses(
        question="LangChain이 무엇인가요? 3문장으로 설명해주세요.",
        personalities=["친절한", "전문적인"]
    )

    # 예제 2: 대화형 챗봇
    print("\n" + "=" * 80)
    print("💬 예제 2: 대화형 챗봇")
    print("=" * 80)
    print()

    # 사용 가능한 프로바이더 확인
    available = []
    for p in ["openai", "anthropic", "google"]:
        if create_chatbot(p, "친절한") is not None:
            available.append(p)

    if available:
        interactive_chatbot(
            provider=available[0],
            personality="친절한"
        )
    else:
        print("❌ 사용 가능한 LLM 프로바이더가 없습니다.")
        print("📝 .env 파일에 API 키를 설정해주세요:")
        print("   OPENAI_API_KEY=your-key")
        print("   ANTHROPIC_API_KEY=your-key")
        print("   GOOGLE_API_KEY=your-key")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 학습 포인트
# ============================================================================
#
# 1. 여러 LLM 프로바이더 사용:
#    - OpenAI (GPT-4o-mini)
#    - Anthropic (Claude 3.5 Sonnet)
#    - Google (Gemini 1.5 Pro)
#
# 2. System Message:
#    - 챗봇의 성격과 행동 정의
#    - 응답 스타일 제어
#
# 3. 대화 이력 관리:
#    - conversation_history 리스트로 관리
#    - 이전 대화 컨텍스트 유지
#
# 4. 에러 처리:
#    - API 키 확인
#    - 프로바이더 사용 가능 여부 확인
#    - 예외 처리
#
# ============================================================================
# 🎓 추가 학습
# ============================================================================
#
# - Part 2: Messages 타입 상세 학습
# - Part 3: Agent 생성 및 도구 사용
# - Part 4: 메모리 시스템으로 대화 이력 관리
#
# ============================================================================
