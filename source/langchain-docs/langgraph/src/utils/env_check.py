"""
환경 변수 검증 유틸리티

이 모듈은 필요한 환경 변수가 설정되어 있는지 확인합니다.
"""

import os
from typing import List, Tuple


def check_env_var(var_name: str) -> Tuple[bool, str]:
    """
    환경 변수가 설정되어 있는지 확인합니다.

    Args:
        var_name: 확인할 환경 변수 이름

    Returns:
        (설정 여부, 메시지) 튜플

    Example:
        >>> is_set, message = check_env_var("ANTHROPIC_API_KEY")
        >>> print(message)
    """
    value = os.getenv(var_name)

    if value:
        # 값이 있으면 앞 4자리만 표시
        masked_value = value[:4] + "..." if len(value) > 4 else "***"
        return True, f"✅ {var_name}: {masked_value}"
    else:
        return False, f"❌ {var_name}: 설정되지 않음"


def check_required_env_vars(required_vars: List[str]) -> bool:
    """
    필수 환경 변수들이 모두 설정되어 있는지 확인합니다.

    Args:
        required_vars: 필수 환경 변수 목록

    Returns:
        모두 설정되어 있으면 True

    Example:
        >>> required = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
        >>> if not check_required_env_vars(required):
        ...     print("환경 변수를 설정하세요")
    """
    print("=" * 60)
    print("환경 변수 확인")
    print("=" * 60)

    all_set = True
    for var in required_vars:
        is_set, message = check_env_var(var)
        print(f"  {message}")
        if not is_set:
            all_set = False

    print()

    if not all_set:
        print("⚠️  일부 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일을 확인하거나 다음 명령으로 설정하세요:")
        print()
        for var in required_vars:
            if not os.getenv(var):
                print(f"   export {var}=your-api-key-here")
        print()

    return all_set


def check_anthropic_env() -> bool:
    """
    Anthropic API 키가 설정되어 있는지 확인합니다.

    Returns:
        API 키가 설정되어 있으면 True
    """
    return check_required_env_vars(["ANTHROPIC_API_KEY"])


def check_openai_env() -> bool:
    """
    OpenAI API 키가 설정되어 있는지 확인합니다.

    Returns:
        API 키가 설정되어 있으면 True
    """
    return check_required_env_vars(["OPENAI_API_KEY"])


def check_all_llm_providers() -> dict:
    """
    모든 LLM 프로바이더의 API 키 설정 상태를 확인합니다.

    Returns:
        프로바이더별 설정 상태 딕셔너리
    """
    providers = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY"
    }

    print("=" * 60)
    print("LLM 프로바이더 환경 변수 확인")
    print("=" * 60)

    status = {}
    for provider, var_name in providers.items():
        is_set, message = check_env_var(var_name)
        status[provider] = is_set
        print(f"  {message}")

    print()

    available_providers = [p for p, s in status.items() if s]
    if available_providers:
        print(f"✅ 사용 가능한 프로바이더: {', '.join(available_providers)}")
    else:
        print("⚠️  설정된 프로바이더가 없습니다.")
        print("   최소한 하나의 LLM 프로바이더 API 키를 설정하세요.")

    print()

    return status


def check_langsmith_env() -> bool:
    """
    LangSmith 추적 설정을 확인합니다.

    Returns:
        LangSmith가 활성화되어 있으면 True
    """
    print("=" * 60)
    print("LangSmith 추적 확인 (선택사항)")
    print("=" * 60)

    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if tracing and api_key:
        print("  ✅ LangSmith 추적이 활성화되어 있습니다")
        is_set, message = check_env_var("LANGCHAIN_API_KEY")
        print(f"  {message}")
        result = True
    elif tracing:
        print("  ⚠️  LANGCHAIN_TRACING_V2는 true이지만 API 키가 없습니다")
        result = False
    else:
        print("  ℹ️  LangSmith 추적이 비활성화되어 있습니다")
        result = False

    print()
    return result


if __name__ == "__main__":
    """테스트 코드"""
    from dotenv import load_dotenv

    # .env 파일 로드
    load_dotenv()

    # 모든 프로바이더 확인
    check_all_llm_providers()

    # LangSmith 확인
    check_langsmith_env()

    print("=" * 60)
    print("환경 검증 완료")
    print("=" * 60)
