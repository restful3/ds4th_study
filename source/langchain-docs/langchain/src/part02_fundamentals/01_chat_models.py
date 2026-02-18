"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: LangChain 기초
================================================================================

파일명: 01_chat_models.py
난이도: ⭐⭐☆☆☆ (초급)
예상 시간: 20분

📚 학습 목표:
  - Chat Models의 개념 이해
  - 다양한 LLM 프로바이더 사용법
  - init_chat_model() 통합 API 이해

📖 공식 문서:
  • Models: /official/07-models.md

🔧 필요한 패키지:
  pip install langchain-openai langchain-anthropic

🚀 실행 방법:
  python 01_chat_models.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

load_dotenv()

# ============================================================================
# 예제 1: ChatOpenAI 기본 사용
# ============================================================================

def example_1_basic_chat():
    """가장 기본적인 Chat Model 사용"""
    print("=" * 70)
    print("📌 예제 1: ChatOpenAI 기본 사용")
    print("=" * 70)

    # Chat Model 초기화
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )

    # 간단한 호출
    response = model.invoke("LangChain이 무엇인가요? 한 문장으로 설명해주세요.")

    print(f"\n🤖 응답: {response.content}\n")


# ============================================================================
# 예제 2: init_chat_model() - 통합 API
# ============================================================================

def example_2_init_chat_model():
    """환경변수 기반 자동 모델 선택"""
    print("=" * 70)
    print("📌 예제 2: init_chat_model() 통합 API")
    print("=" * 70)

    # 환경변수에 설정된 API 키 기반으로 자동 선택
    model = init_chat_model()

    response = model.invoke("안녕하세요!")

    print(f"\n🤖 모델: {model.model_name}")
    print(f"🤖 응답: {response.content}\n")


# ============================================================================
# 예제 3: Temperature 조절
# ============================================================================

def example_3_temperature():
    """Temperature로 창의성 조절"""
    print("=" * 70)
    print("📌 예제 3: Temperature로 창의성 조절")
    print("=" * 70)

    prompt = "AI에 대한 재미있는 농담 하나 해주세요."

    # Temperature 0.0 (결정적)
    model_deterministic = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    response1 = model_deterministic.invoke(prompt)

    # Temperature 1.0 (창의적)
    model_creative = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
    response2 = model_creative.invoke(prompt)

    print(f"\n🌡️ Temperature 0.0 (결정적):")
    print(f"   {response1.content}")

    print(f"\n🌡️ Temperature 1.0 (창의적):")
    print(f"   {response2.content}\n")


# ============================================================================
# 예제 4: 스트리밍
# ============================================================================

def example_4_streaming():
    """실시간 스트리밍 응답"""
    print("=" * 70)
    print("📌 예제 4: 스트리밍 응답")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)

    print("\n🤖 스트리밍 응답:")
    print("   ", end="", flush=True)

    for chunk in model.stream("파이썬의 장점 3가지를 설명해주세요."):
        print(chunk.content, end="", flush=True)

    print("\n")


# ============================================================================
# 예제 5: 다양한 프로바이더
# ============================================================================

def example_5_multiple_providers():
    """여러 LLM 프로바이더 사용"""
    print("=" * 70)
    print("📌 예제 5: 다양한 LLM 프로바이더")
    print("=" * 70)

    prompt = "안녕하세요!"

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        openai_model = ChatOpenAI(model="gpt-4o-mini")
        response = openai_model.invoke(prompt)
        print(f"\n✅ OpenAI (gpt-4o-mini): {response.content}")

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        anthropic_model = ChatAnthropic(model="claude-haiku-4-5-20251001")
        response = anthropic_model.invoke(prompt)
        print(f"\n✅ Anthropic (claude-haiku-4-5): {response.content}")

    # Google
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        response = google_model.invoke(prompt)
        print(f"\n✅ Google (gemini-2.5-flash-lite): {response.content}")

    print()


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n🎓 Part 2: LangChain 기초 - Chat Models\n")

    example_1_basic_chat()
    input("⏎ 계속하려면 Enter...")

    example_2_init_chat_model()
    input("⏎ 계속하려면 Enter...")

    example_3_temperature()
    input("⏎ 계속하려면 Enter...")

    example_4_streaming()
    input("⏎ 계속하려면 Enter...")

    example_5_multiple_providers()

    print("=" * 70)
    print("🎉 Chat Models 학습 완료!")
    print("📖 다음: 02_messages.py - 메시지 타입")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
