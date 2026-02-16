"""
LLM 초기화 유틸리티

이 모듈은 다양한 LLM 프로바이더를 초기화하는 공통 함수를 제공합니다.
"""

import os
from typing import Optional


def get_llm(
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0,
    **kwargs
):
    """
    LLM 인스턴스를 초기화합니다.

    Args:
        provider: LLM 프로바이더 ("anthropic", "openai", "google" 등)
        model: 사용할 모델 이름 (None이면 기본 모델 사용)
        temperature: 생성 온도 (0~1)
        **kwargs: 추가 파라미터

    Returns:
        초기화된 LLM 인스턴스

    Raises:
        ValueError: 지원하지 않는 프로바이더인 경우
        Exception: API 키가 설정되지 않은 경우

    Example:
        >>> llm = get_llm("anthropic")
        >>> llm = get_llm("openai", model="gpt-4", temperature=0.7)
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # API 키 확인
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise Exception(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 API 키를 추가하세요."
            )

        # 기본 모델 설정
        if model is None:
            model = "claude-sonnet-4-5-20250929"

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            **kwargs
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            raise Exception(
                "OPENAI_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 API 키를 추가하세요."
            )

        # 기본 모델 설정
        if model is None:
            model = "gpt-4"

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            **kwargs
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # API 키 확인
        if not os.getenv("GOOGLE_API_KEY"):
            raise Exception(
                "GOOGLE_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 API 키를 추가하세요."
            )

        # 기본 모델 설정
        if model is None:
            model = "gemini-pro"

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            **kwargs
        )

    else:
        raise ValueError(
            f"지원하지 않는 프로바이더입니다: {provider}\n"
            f"지원되는 프로바이더: anthropic, openai, google"
        )


def get_default_llm(**kwargs):
    """
    기본 LLM(Anthropic Claude)을 반환합니다.

    Args:
        **kwargs: get_llm()에 전달할 추가 파라미터

    Returns:
        초기화된 Claude LLM 인스턴스

    Example:
        >>> llm = get_default_llm()
        >>> llm = get_default_llm(temperature=0.7)
    """
    return get_llm(provider="anthropic", **kwargs)


if __name__ == "__main__":
    """테스트 코드"""
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("LLM Setup 테스트")
    print("=" * 60)

    try:
        llm = get_default_llm()
        print(f"✅ LLM 초기화 성공: {llm.__class__.__name__}")
        print(f"   모델: {llm.model}")
        print(f"   온도: {llm.temperature}")

        # 간단한 테스트
        response = llm.invoke("안녕하세요!")
        print(f"\n📝 테스트 응답: {response.content}")

    except Exception as e:
        print(f"❌ LLM 초기화 실패: {e}")
