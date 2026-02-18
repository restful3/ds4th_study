"""
================================================================================
LangChain AI Agent 마스터 교안
Part 1: AI Agent의 이해
================================================================================

파일명: 02_environment_check.py
난이도: ⭐☆☆☆☆ (입문)
예상 시간: 5분

📚 학습 목표:
  - 개발 환경이 올바르게 설정되었는지 확인
  - 필수 패키지 설치 확인
  - API 연결 테스트

📖 공식 문서:
  • Install: /official/02-install.md

📄 교안 문서:
  • Setup Guide: /SETUP_GUIDE.md
  • Troubleshooting: /docs/appendix/troubleshooting.md

🚀 실행 방법:
  python 02_environment_check.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import sys
import os
from dotenv import load_dotenv

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

# ============================================================================
# 체크 함수들
# ============================================================================

def check_python_version():
    """Python 버전 확인"""
    print("🔍 Python 버전 확인...")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 10:
        print(f"   ✅ Python {version_str} (3.10 이상 필요)")
        return True
    else:
        print(f"   ❌ Python {version_str} (3.10 이상 필요)")
        print("   📝 Python을 업그레이드하세요: https://www.python.org/downloads/")
        return False


def check_package(package_name, display_name=None):
    """패키지 설치 확인"""
    if display_name is None:
        display_name = package_name

    try:
        __import__(package_name)
        # 버전 정보 가져오기 시도
        try:
            module = __import__(package_name)
            version = getattr(module, "__version__", "설치됨")
            print(f"   ✅ {display_name}: {version}")
        except:
            print(f"   ✅ {display_name}: 설치됨")
        return True
    except ImportError:
        print(f"   ❌ {display_name}: 설치되지 않음")
        return False


def check_packages():
    """필수 패키지 설치 확인"""
    print("\n🔍 필수 패키지 확인...")

    packages = [
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langgraph", "LangGraph"),
        ("langsmith", "LangSmith"),
        ("dotenv", "python-dotenv"),
    ]

    all_installed = True
    for package, display in packages:
        if not check_package(package, display):
            all_installed = False

    if not all_installed:
        print("\n   📝 패키지 설치:")
        print("      pip install -r requirements.txt")

    return all_installed


def check_llm_providers():
    """LLM 프로바이더 패키지 확인"""
    print("\n🔍 LLM 프로바이더 패키지 확인...")

    providers = [
        ("langchain_openai", "OpenAI"),
        ("langchain_anthropic", "Anthropic"),
        ("langchain_google_genai", "Google"),
    ]

    installed_count = 0
    for package, display in providers:
        if check_package(package, display):
            installed_count += 1

    if installed_count == 0:
        print("\n   ⚠️  최소 하나의 LLM 프로바이더가 필요합니다!")
        print("   📝 설치 예시:")
        print("      pip install langchain-openai")

    return installed_count > 0


def check_env_variables():
    """환경변수 확인"""
    print("\n🔍 환경변수 확인...")

    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
        "LANGSMITH_API_KEY": "LangSmith (선택 사항)",
    }

    found_llm_key = False

    for key, name in api_keys.items():
        value = os.getenv(key)

        if value:
            # 키의 일부만 표시 (보안)
            masked_value = value[:10] + "..." if len(value) > 10 else "***"
            print(f"   ✅ {name}: {masked_value}")

            if key != "LANGSMITH_API_KEY":
                found_llm_key = True
        else:
            if key == "LANGSMITH_API_KEY":
                print(f"   ⚪ {name}: 설정되지 않음")
            else:
                print(f"   ❌ {name}: 설정되지 않음")

    if not found_llm_key:
        print("\n   ⚠️  최소 하나의 LLM API 키가 필요합니다!")
        print("   📝 .env 파일 설정:")
        print("      cp .env.example .env")
        print("      nano .env  # API 키 입력")

    return found_llm_key


def check_api_connection():
    """실제 API 연결 테스트"""
    print("\n🔍 API 연결 테스트...")

    # OpenAI가 설정되어 있으면 테스트
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4.1-nano", timeout=10)
            response = llm.invoke("Hi")

            print(f"   ✅ OpenAI API 연결 성공")
            print(f"   📝 테스트 응답: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"   ❌ OpenAI API 연결 실패: {str(e)[:100]}")
            return False

    # Anthropic이 설정되어 있으면 테스트
    elif os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-haiku-4-5-20251001", timeout=10)
            response = llm.invoke("Hi")

            print(f"   ✅ Anthropic API 연결 성공")
            print(f"   📝 테스트 응답: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"   ❌ Anthropic API 연결 실패: {str(e)[:100]}")
            return False

    # Google이 설정되어 있으면 테스트
    elif os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", timeout=10)
            response = llm.invoke("Hi")

            print(f"   ✅ Google API 연결 성공")
            print(f"   📝 테스트 응답: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"   ❌ Google API 연결 실패: {str(e)[:100]}")
            return False

    else:
        print(f"   ⚠️  API 키가 설정되지 않아 연결 테스트를 건너뜁니다")
        return False


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """환경 확인 메인 함수"""
    print("\n")
    print("=" * 70)
    print("🔍 LangChain 환경 확인")
    print("=" * 70)
    print("\n")

    # 체크리스트
    checks = {
        "Python 버전": check_python_version(),
        "필수 패키지": check_packages(),
        "LLM 프로바이더": check_llm_providers(),
        "환경변수": check_env_variables(),
        "API 연결": check_api_connection(),
    }

    # 결과 요약
    print("\n")
    print("=" * 70)
    print("📊 환경 확인 결과")
    print("=" * 70)
    print("\n")

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    for name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {name}")

    print(f"\n   통과: {passed}/{total}")

    # 최종 판정
    print("\n")
    print("=" * 70)

    if passed == total:
        print("🎉 환경 설정이 완료되었습니다!")
        print("=" * 70)
        print("\n💡 다음 단계:")
        print("   1. 01_hello_langchain.py - 첫 LangChain 프로그램")
        print("   2. Part 2: LangChain 기초 학습")
        print("   3. Part 3: 첫 번째 Agent 만들기")

    elif passed >= 4:
        print("⚠️  거의 완료되었습니다! 몇 가지만 더 확인하세요.")
        print("=" * 70)
        print("\n📝 확인 사항:")
        for name, result in checks.items():
            if not result:
                print(f"   • {name}")
        print("\n📖 도움말: /docs/appendix/troubleshooting.md")

    else:
        print("❌ 환경 설정이 필요합니다.")
        print("=" * 70)
        print("\n📝 설치 가이드: /SETUP_GUIDE.md")
        print("📖 문제 해결: /docs/appendix/troubleshooting.md")

    print("\n" + "=" * 70 + "\n")

    return passed == total


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


# ============================================================================
# 📚 추가 정보
# ============================================================================
#
# 환경 설정이 제대로 안 되면:
#
# 1. Python 버전 확인:
#    python --version
#
# 2. 가상환경 생성:
#    python -m venv .venv
#    source .venv/bin/activate  # Windows: .venv\Scripts\activate
#
# 3. 패키지 설치:
#    pip install -r requirements.txt
#
# 4. .env 파일 설정:
#    cp .env.example .env
#    # .env 파일에 API 키 입력
#
# 5. 재실행:
#    python 02_environment_check.py
#
# ============================================================================
