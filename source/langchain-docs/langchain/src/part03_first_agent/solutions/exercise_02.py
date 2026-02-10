"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: First Agent - 실습 과제 2 해답
================================================================================

과제: 정보 검색 Agent
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. Wikipedia 검색 도구 (또는 다른 검색 API)
2. 요약 생성 도구
3. "파이썬의 역사를 알려줘" 같은 질문에 응답

학습 목표:
- 외부 API 통합
- 정보 검색 및 처리
- 여러 도구의 조합

================================================================================
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Optional

# ============================================================================
# Mock 지식 베이스
# ============================================================================

KNOWLEDGE_BASE = {
    "파이썬": """
Python은 1991년 네덜란드의 프로그래머 귀도 반 로섬(Guido van Rossum)이 개발한
고급 프로그래밍 언어입니다. 간결하고 읽기 쉬운 문법으로 인해 초보자부터 전문가까지
널리 사용되고 있습니다. 웹 개발, 데이터 분석, 인공지능, 과학 계산 등 다양한
분야에서 활용됩니다. 2020년대 들어 AI 및 머신러닝 분야에서 사실상 표준 언어로
자리잡았습니다.
    """,
    "LangChain": """
LangChain은 2022년 Harrison Chase가 개발한 LLM(대규모 언어 모델) 애플리케이션
개발 프레임워크입니다. LLM을 활용한 애플리케이션을 쉽게 구축할 수 있도록
다양한 구성 요소와 도구를 제공합니다. Agent, Chain, Memory, Tool 등의 개념을
통해 복잡한 AI 시스템을 모듈화하여 개발할 수 있습니다. Python과 JavaScript
버전이 있으며, 오픈소스로 공개되어 빠르게 성장하고 있습니다.
    """,
    "인공지능": """
인공지능(Artificial Intelligence, AI)은 인간의 지능을 기계로 구현하는 기술입니다.
1956년 다트머스 회의에서 처음 용어가 제안되었으며, 이후 수십 년간 발전해왔습니다.
2010년대 딥러닝의 등장으로 큰 발전을 이루었고, 2022년 ChatGPT의 출시로
일반 대중에게도 널리 알려지게 되었습니다. 현재는 자율주행, 의료 진단,
번역, 콘텐츠 생성 등 다양한 분야에서 활용되고 있습니다.
    """,
    "머신러닝": """
머신러닝(Machine Learning)은 데이터를 통해 컴퓨터가 스스로 학습하는 기술입니다.
1950년대 아서 사무엘이 체커 게임을 학습하는 프로그램을 개발하면서 시작되었습니다.
지도 학습, 비지도 학습, 강화 학습 등의 방법론이 있으며, 2010년대 딥러닝의
발전으로 이미지 인식, 음성 인식 등에서 인간 수준의 성능을 달성했습니다.
    """,
}


# ============================================================================
# 도구 정의
# ============================================================================

@tool
def search_knowledge(topic: str) -> str:
    """지식 베이스에서 정보를 검색합니다.

    Wikipedia나 검색 엔진처럼 작동하여 주제에 대한 정보를 찾습니다.

    Args:
        topic: 검색할 주제 또는 키워드

    Returns:
        검색된 정보 또는 "찾을 수 없음" 메시지
    """
    # 키워드 매칭
    for key, value in KNOWLEDGE_BASE.items():
        if key.lower() in topic.lower() or topic.lower() in key.lower():
            return f"📚 '{key}'에 대한 정보:\n{value.strip()}"

    return f"❌ '{topic}'에 대한 정보를 찾을 수 없습니다. 다른 키워드로 검색해보세요."


@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """긴 텍스트를 요약합니다.

    주어진 텍스트에서 핵심 정보를 추출하여 간결하게 요약합니다.

    Args:
        text: 요약할 텍스트
        max_sentences: 요약 문장 개수 (기본값: 3)

    Returns:
        요약된 텍스트
    """
    # 간단한 요약: 첫 N개 문장 추출
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    if len(sentences) <= max_sentences:
        summary = '. '.join(sentences) + '.'
    else:
        summary = '. '.join(sentences[:max_sentences]) + '.'

    return f"📝 요약:\n{summary}"


@tool
def compare_topics(topic1: str, topic2: str) -> str:
    """두 주제를 비교합니다.

    두 주제에 대한 정보를 검색하고 비교하여 공통점과 차이점을 설명합니다.

    Args:
        topic1: 첫 번째 주제
        topic2: 두 번째 주제

    Returns:
        비교 결과
    """
    # 각 주제 정보 검색
    info1 = None
    info2 = None

    for key, value in KNOWLEDGE_BASE.items():
        if key.lower() in topic1.lower():
            info1 = value.strip()
        if key.lower() in topic2.lower():
            info2 = value.strip()

    if not info1 or not info2:
        return f"❌ '{topic1}' 또는 '{topic2}'에 대한 정보를 찾을 수 없습니다."

    return f"""🔍 '{topic1}' vs '{topic2}' 비교:

【{topic1}】
{info1[:200]}...

【{topic2}】
{info2[:200]}...

※ 두 주제 모두 기술 분야와 관련이 있습니다."""


# ============================================================================
# Agent 생성
# ============================================================================

def create_search_agent():
    """정보 검색 Agent 생성"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_knowledge, summarize_text, compare_topics]

    system_prompt = """당신은 정보 검색 및 분석 전문 Assistant입니다.

사용자가 정보를 요청하면:
1. search_knowledge 도구로 관련 정보를 검색합니다
2. 필요하면 summarize_text로 요약합니다
3. 비교가 필요하면 compare_topics를 사용합니다

항상 정확하고 구조화된 답변을 제공하세요.
검색 결과를 바탕으로 명확하게 설명해주세요."""

    agent = create_react_agent(model, tools, state_modifier=system_prompt)

    return agent


# ============================================================================
# 테스트 함수
# ============================================================================

def test_search_agent():
    """검색 Agent 테스트"""
    print("=" * 70)
    print("🔍 정보 검색 Agent 테스트")
    print("=" * 70)

    agent = create_search_agent()

    test_cases = [
        "파이썬의 역사를 알려줘",
        "LangChain이 뭐야?",
        "인공지능에 대해 간단히 요약해줘",
        "파이썬과 LangChain을 비교해줘",
        "머신러닝의 발전 과정은?",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 테스트 {i}: {question}")
        print('=' * 70)

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})

            final_message = result["messages"][-1]
            print(f"\n🤖 Agent 응답:\n{final_message.content}\n")

        except Exception as e:
            print(f"\n❌ 오류: {e}\n")

    print("=" * 70)
    print("✅ 모든 테스트 완료!")
    print("=" * 70)


def interactive_search():
    """대화형 검색"""
    print("\n" + "=" * 70)
    print("🎮 대화형 정보 검색 Agent")
    print("=" * 70)
    print("질문을 입력하세요 (종료: 'quit', 'exit', 'q')")
    print("\n💡 사용 가능한 주제:")
    print("   - 파이썬, LangChain, 인공지능, 머신러닝")
    print("=" * 70)

    agent = create_search_agent()

    while True:
        try:
            user_input = input("\n👤 질문: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q', '종료']:
                print("\n👋 검색을 종료합니다.")
                break

            if not user_input:
                continue

            result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

            final_message = result["messages"][-1]
            print(f"\n🤖 Agent: {final_message.content}")

        except KeyboardInterrupt:
            print("\n\n👋 검색을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🔍 Part 3: 정보 검색 Agent - 실습 과제 2 해답")
    print("=" * 70)

    try:
        # 자동 테스트
        test_search_agent()

        # 대화형 모드 (선택)
        print("\n대화형 모드를 실행하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예']:
            interactive_search()

    except Exception as e:
        print(f"\n⚠️  테스트 실패: {e}")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("💡 추가 학습 포인트:")
    print("  1. 실제 Wikipedia API 연동")
    print("  2. 웹 크롤링으로 최신 정보 수집")
    print("  3. 벡터 DB로 의미 기반 검색")
    print("  4. 다국어 검색 지원")
    print("=" * 70)

    print("\n📚 실제 Wikipedia API 사용 예시:")
    print("""
import wikipedia

@tool
def search_wikipedia(query: str) -> str:
    try:
        result = wikipedia.summary(query, sentences=3)
        return f"Wikipedia: {result}"
    except:
        return "검색 결과 없음"
""")


if __name__ == "__main__":
    main()
