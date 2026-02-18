"""
================================================================================
LangChain AI Agent 마스터 교안
Part 1: AI Agent의 이해 - 실습 과제 1 해답
================================================================================

과제: 환경 설정 및 확인

난이도: ⭐☆☆ (입문)

요구사항:
1. Python 3.10 이상 설치
2. LangChain 설치
3. API 키 설정 (OpenAI, Anthropic, 또는 Google 중 하나)
4. 02_environment_check.py 실행

학습 목표:
- 개발 환경이 올바르게 설정되었는지 프로그래밍으로 확인
- 필수 패키지와 API 키 검증
- 간단한 LLM 호출로 연결 테스트

================================================================================
"""

import sys
import os
from dotenv import load_dotenv

# 환경 설정
load_dotenv(override=True)


# ============================================================================
# 1. Python 버전 확인
# ============================================================================

def check_python_version():
    """Python 3.10 이상인지 확인합니다."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 10:
        print(f"  ✅ Python {version_str} (3.10 이상)")
        return True
    else:
        print(f"  ❌ Python {version_str} - 3.10 이상이 필요합니다")
        print("     https://www.python.org/downloads/")
        return False


# ============================================================================
# 2. LangChain 설치 확인
# ============================================================================

def check_langchain_installed():
    """LangChain 핵심 패키지가 설치되어 있는지 확인합니다."""
    packages = {
        "langchain": "LangChain",
        "langchain_core": "LangChain Core",
        "langgraph": "LangGraph",
    }

    all_ok = True
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "설치됨")
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: 설치되지 않음")
            all_ok = False

    if not all_ok:
        print("\n  📝 설치 명령어:")
        print("     pip install -U langchain langchain-core langgraph")

    return all_ok


# ============================================================================
# 3. API 키 설정 확인
# ============================================================================

def check_api_keys():
    """최소 하나의 LLM 프로바이더 API 키가 설정되어 있는지 확인합니다."""
    keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
    }

    found = False
    for key, name in keys.items():
        value = os.getenv(key)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"  ✅ {name}: {masked}")
            found = True
        else:
            print(f"  ⚪ {name}: 미설정")

    if not found:
        print("\n  ⚠️  최소 하나의 API 키가 필요합니다!")
        print("  📝 .env 파일에 다음 중 하나를 설정하세요:")
        print("     OPENAI_API_KEY=sk-...")
        print("     ANTHROPIC_API_KEY=sk-ant-...")
        print("     GOOGLE_API_KEY=AI...")

    return found


# ============================================================================
# 4. LLM 연결 테스트
# ============================================================================

def test_llm_connection():
    """실제 LLM API 호출로 연결을 테스트합니다."""
    # OpenAI 테스트
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(model="gpt-4.1-nano", timeout=10)
            response = llm.invoke([HumanMessage(content="Hello!")])
            print(f"  ✅ OpenAI 연결 성공: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"  ❌ OpenAI 연결 실패: {str(e)[:80]}")
            return False

    # Anthropic 테스트
    elif os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage

            llm = ChatAnthropic(model="claude-haiku-4-5-20251001", timeout=10)
            response = llm.invoke([HumanMessage(content="Hello!")])
            print(f"  ✅ Anthropic 연결 성공: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"  ❌ Anthropic 연결 실패: {str(e)[:80]}")
            return False

    # Google 테스트
    elif os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", timeout=10)
            response = llm.invoke([HumanMessage(content="Hello!")])
            print(f"  ✅ Google 연결 성공: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"  ❌ Google 연결 실패: {str(e)[:80]}")
            return False

    else:
        print("  ⚠️  API 키가 없어 연결 테스트를 건너뜁니다")
        return False


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """환경 설정 확인 메인 함수"""
    print("\n" + "=" * 60)
    print("Part 1 실습 과제 1 - 환경 설정 및 확인")
    print("=" * 60)

    results = {}

    # 1. Python 버전
    print("\n[1/4] Python 버전 확인")
    results["Python"] = check_python_version()

    # 2. LangChain 설치
    print("\n[2/4] LangChain 패키지 확인")
    results["LangChain"] = check_langchain_installed()

    # 3. API 키
    print("\n[3/4] API 키 확인")
    results["API 키"] = check_api_keys()

    # 4. LLM 연결
    print("\n[4/4] LLM 연결 테스트")
    results["LLM 연결"] = test_llm_connection()

    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)

    passed = 0
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  통과: {passed}/{len(results)}")

    if passed == len(results):
        print("\n🎉 모든 환경이 정상입니다!")
        print("   다음 단계: 01_hello_langchain.py 실행")
    else:
        print("\n⚠️  일부 항목이 통과하지 못했습니다.")
        print("   📖 도움말: /SETUP_GUIDE.md")

    print("\n" + "=" * 60 + "\n")
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
