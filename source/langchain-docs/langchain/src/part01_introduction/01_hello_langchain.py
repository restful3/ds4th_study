"""
================================================================================
LangChain AI Agent 마스터 교안
Part 1: AI Agent의 이해
================================================================================

파일명: 01_hello_langchain.py
난이도: ⭐☆☆☆☆ (입문)
예상 시간: 10분

📚 학습 목표:
  - LangChain의 기본 개념 이해
  - 첫 번째 LLM 호출 경험
  - Agent와 일반 LLM의 차이 이해

📖 공식 문서:
  • Overview: /official/01-overview.md
  • Quickstart: /official/03-quickstart.md

📄 교안 문서:
  • Part 1 개요: /docs/part01_introduction.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY (또는 다른 LLM 프로바이더 키)

🚀 실행 방법:
  python 01_hello_langchain.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================================
# 환경 설정
# ============================================================================

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 src/.env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: 가장 간단한 LLM 호출
# ============================================================================

def example_1_simple_llm():
    """가장 기본적인 LLM 호출"""
    print("=" * 70)
    print("📌 예제 1: 가장 간단한 LLM 호출")
    print("=" * 70)

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 비용 효율적인 모델
        temperature=0.7,       # 창의성 조절 (0.0 = 결정적, 1.0 = 창의적)
    )

    # 메시지 생성
    messages = [
        HumanMessage(content="안녕하세요! LangChain이 무엇인가요?")
    ]

    # LLM 호출
    response = llm.invoke(messages)

    # 결과 출력
    print(f"\n👤 사용자: 안녕하세요! LangChain이 무엇인가요?")
    print(f"🤖 AI: {response.content}\n")


# ============================================================================
# 예제 2: System Message 사용
# ============================================================================

def example_2_system_message():
    """System Message로 AI의 역할 지정"""
    print("=" * 70)
    print("📌 예제 2: System Message로 AI 역할 지정")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # System Message: AI의 역할과 행동을 정의
    messages = [
        SystemMessage(content="당신은 5살 아이에게 설명하듯이 쉽게 설명하는 선생님입니다."),
        HumanMessage(content="LangChain이 무엇인가요?")
    ]

    response = llm.invoke(messages)

    print(f"\n🎭 역할: 5살 아이에게 설명하는 선생님")
    print(f"👤 사용자: LangChain이 무엇인가요?")
    print(f"🤖 AI: {response.content}\n")


# ============================================================================
# 예제 3: 일반 LLM vs Agent (개념 소개)
# ============================================================================

def example_3_llm_vs_agent_concept():
    """일반 LLM과 Agent의 차이 설명"""
    print("=" * 70)
    print("📌 예제 3: 일반 LLM vs Agent (개념)")
    print("=" * 70)

    print("""
🔹 일반 LLM (Language Model):
   - 텍스트 입력 → 텍스트 출력
   - 단순 질문-답변
   - 실시간 정보 접근 불가
   - 예: "파이썬이란?" → "파이썬은 프로그래밍 언어입니다..."

🔹 Agent (에이전트):
   - 텍스트 입력 → 도구 사용 → 작업 수행 → 텍스트 출력
   - 복잡한 작업 수행
   - 외부 도구 사용 가능 (검색, API 호출 등)
   - 예: "오늘 서울 날씨는?" → [날씨 API 호출] → "서울은 맑고 22도입니다"

📖 다음 예제에서 실제 Agent를 만들어 봅니다!
    """)


# ============================================================================
# 예제 4: LangChain의 주요 구성 요소
# ============================================================================

def example_4_langchain_components():
    """LangChain의 주요 구성 요소 소개"""
    print("=" * 70)
    print("📌 예제 4: LangChain의 주요 구성 요소")
    print("=" * 70)

    print("""
🏗️ LangChain의 주요 구성 요소:

1️⃣ Models (모델)
   - LLM과의 인터페이스
   - OpenAI, Anthropic, Google 등 지원
   - 예: ChatOpenAI, ChatAnthropic

2️⃣ Messages (메시지)
   - 대화의 기본 단위
   - SystemMessage, HumanMessage, AIMessage
   - 예: HumanMessage(content="안녕")

3️⃣ Tools (도구)
   - Agent가 사용할 수 있는 기능
   - 검색, 계산, API 호출 등
   - 예: @tool 데코레이터

4️⃣ Agents (에이전트)
   - 도구를 사용하여 작업 수행
   - 추론 + 행동 반복
   - 예: create_agent()

5️⃣ Memory (메모리)
   - 대화 이력 저장
   - 장기/단기 메모리
   - 예: Checkpointer

📖 이후 파트에서 각 구성 요소를 자세히 배웁니다!
    """)


# ============================================================================
# 예제 5: 간단한 대화 이력
# ============================================================================

def example_5_conversation_history():
    """대화 이력을 유지하는 간단한 예제"""
    print("=" * 70)
    print("📌 예제 5: 대화 이력 유지 (간단한 버전)")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 대화 이력을 리스트로 관리
    conversation = []

    # 첫 번째 대화
    conversation.append(HumanMessage(content="제 이름은 김철수입니다."))
    response1 = llm.invoke(conversation)
    conversation.append(response1)

    print(f"\n👤 사용자: 제 이름은 김철수입니다.")
    print(f"🤖 AI: {response1.content}")

    # 두 번째 대화 (이전 대화 기억)
    conversation.append(HumanMessage(content="제 이름이 뭐라고 했죠?"))
    response2 = llm.invoke(conversation)
    conversation.append(response2)

    print(f"\n👤 사용자: 제 이름이 뭐라고 했죠?")
    print(f"🤖 AI: {response2.content}")

    print("\n💡 대화 이력이 유지되어 이전 정보를 기억합니다!")
    print("📖 Part 4에서 체계적인 메모리 관리를 배웁니다.\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 1: AI Agent의 이해 - Hello LangChain!")
    print("\n")

    # 모든 예제 실행
    example_1_simple_llm()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_system_message()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_llm_vs_agent_concept()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_langchain_components()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_conversation_history()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 1 첫 번째 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 02_environment_check.py - 환경 설정 확인")
    print("  2. Part 2: LangChain 기초 학습")
    print("  3. Part 3: 첫 번째 Agent 만들기")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. Temperature 파라미터:
#    - 0.0: 항상 같은 답변 (결정적)
#    - 0.7: 균형잡힌 창의성 (기본값)
#    - 1.0: 매우 창의적 (다양한 답변)
#
# 2. 다양한 LLM 프로바이더:
#    from langchain_anthropic import ChatAnthropic
#    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
#
# 3. 스트리밍:
#    for chunk in llm.stream(messages):
#        print(chunk.content, end="", flush=True)
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "OPENAI_API_KEY not found"
# 해결: src/.env 파일을 확인하고 API 키를 설정하세요
#
# 문제: "Rate limit exceeded"
# 해결: API 사용량을 확인하거나 더 저렴한 모델(gpt-4o-mini) 사용
#
# 문제: 응답이 너무 느림
# 해결: gpt-4o-mini 모델 사용 또는 스트리밍 모드 활성화
#
# ============================================================================
