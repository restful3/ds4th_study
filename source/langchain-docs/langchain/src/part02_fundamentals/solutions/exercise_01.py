"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: Fundamentals - 실습 과제 1 해답
================================================================================

과제: 다중 프로바이더 Chat Model
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. OpenAI GPT-4o-mini와 Anthropic Claude 모델 초기화
2. "AI Agent가 무엇이며 어떻게 활용되는지 3문장으로 설명하세요" 질문 실행
3. 두 모델의 응답을 출력하고 차이점 분석

학습 목표:
- init_chat_model()을 사용한 다양한 프로바이더 초기화
- 프로바이더별 응답 스타일 차이 관찰
- Temperature 파라미터의 효과 이해

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# ============================================================================
# 1단계: 다중 프로바이더 모델 초기화
# ============================================================================

def initialize_models():
    """OpenAI와 Anthropic 모델을 초기화합니다."""
    print("=" * 70)
    print("1단계: 모델 초기화")
    print("=" * 70)

    models = {}

    # OpenAI GPT-4o-mini
    if os.getenv("OPENAI_API_KEY"):
        models["GPT-4o-mini"] = init_chat_model(
            "gpt-4o-mini",
            model_provider="openai",
            temperature=0.7,
        )
        print("  OpenAI GPT-4o-mini 초기화 완료")
    else:
        print("  [SKIP] OPENAI_API_KEY가 설정되지 않았습니다")

    # Anthropic Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        models["Claude Haiku"] = init_chat_model(
            "claude-haiku-4-5-20251001",
            model_provider="anthropic",
            temperature=0.7,
        )
        print("  Anthropic Claude Haiku 초기화 완료")
    else:
        print("  [SKIP] ANTHROPIC_API_KEY가 설정되지 않았습니다")

    if not models:
        print("\n  최소 하나의 API 키를 설정해주세요.")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")

    return models


# ============================================================================
# 2단계: 동일한 질문으로 응답 비교
# ============================================================================

def compare_responses(models: dict):
    """동일한 질문을 여러 모델에 실행하고 응답을 비교합니다."""
    print("\n" + "=" * 70)
    print("2단계: 응답 비교")
    print("=" * 70)

    question = "AI Agent가 무엇이며 어떻게 활용되는지 3문장으로 설명하세요"
    print(f"\n질문: {question}\n")

    responses = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")
        response = model.invoke(question)
        print(response.content)

        # 토큰 사용량 확인
        if response.usage_metadata:
            print(f"\n  [토큰] 입력: {response.usage_metadata.get('input_tokens', 'N/A')}, "
                  f"출력: {response.usage_metadata.get('output_tokens', 'N/A')}")

        responses[name] = response.content

    return responses


# ============================================================================
# 3단계: Temperature 변경 실험
# ============================================================================

def temperature_experiment(models: dict):
    """Temperature를 변경하면서 응답 차이를 관찰합니다."""
    print("\n" + "=" * 70)
    print("3단계: Temperature 변경 실험")
    print("=" * 70)

    # 첫 번째 모델로 실험
    model_name = list(models.keys())[0]
    provider = "openai" if "GPT" in model_name else "anthropic"
    model_id = "gpt-4o-mini" if "GPT" in model_name else "claude-haiku-4-5-20251001"

    question = "AI의 미래를 한 문장으로 예측하세요"
    print(f"\n모델: {model_name}")
    print(f"질문: {question}\n")

    for temp in [0.0, 0.7, 1.0]:
        model = init_chat_model(
            model_id,
            model_provider=provider,
            temperature=temp,
        )

        print(f"\nTemperature {temp}:")
        response = model.invoke(question)
        print(f"  {response.content}")

    print("\n" + "-" * 70)
    print("관찰 포인트:")
    print("  - Temperature 0.0: 동일한 질문에 거의 같은 답변")
    print("  - Temperature 0.7: 적당히 다양한 답변 (기본값)")
    print("  - Temperature 1.0: 매번 다른 창의적인 답변")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("Part 2: 다중 프로바이더 Chat Model - 실습 과제 1 해답")
    print("=" * 70)

    # 1. 모델 초기화
    models = initialize_models()
    if not models:
        return

    # 2. 응답 비교
    responses = compare_responses(models)

    # 3. Temperature 실험
    temperature_experiment(models)

    # 분석 포인트
    print("\n" + "=" * 70)
    print("분석 포인트:")
    print("  1. 각 모델의 응답 길이와 스타일 차이를 비교해보세요")
    print("  2. 어떤 모델이 더 구조적으로 답변하나요?")
    print("  3. 한국어 품질은 어떤 모델이 더 자연스러운가요?")
    print("  4. 토큰 사용량(비용) 차이는 어느 정도인가요?")
    print("=" * 70)


if __name__ == "__main__":
    main()
