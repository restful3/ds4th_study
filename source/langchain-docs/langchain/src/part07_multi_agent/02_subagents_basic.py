"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent Systems
================================================================================

파일명: 02_subagents_basic.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - Subagent 패턴의 개념과 작동 원리 이해
  - Agent를 도구로 래핑하는 방법 학습
  - 전문화된 Subagent 구현
  - 여러 Subagent를 조합하는 방법
  - 실전 리서치 보조 시스템 구축

📖 공식 문서:
  • Subagents: /official/23-subagents.md

📄 교안 문서:
  • Part 7 Subagents: /docs/part07_multi_agent.md (Section 2)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 02_subagents_basic.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# 예제 1: Subagent의 기본 개념
# ============================================================================

def example_1_subagent_concept():
    """Subagent를 도구로 사용하는 기본 개념"""
    print("=" * 70)
    print("📌 예제 1: Subagent의 기본 개념")
    print("=" * 70)

    print("""
💡 Subagent란?
   - 메인 Agent가 "도구처럼" 사용하는 전문화된 Agent
   - 특정 작업을 독립적으로 처리하고 결과 반환
   - 메인 Agent는 제어를 유지하며 Subagent를 조율

🔄 작동 원리:
   1. 메인 Agent가 작업 필요성 판단
   2. 적절한 Subagent를 도구로 호출
   3. Subagent가 전문 작업 수행
   4. 결과를 메인 Agent에게 반환
   5. 메인 Agent가 다음 작업 진행
    """)

    # 간단한 수학 전문가 Subagent
    @tool
    def math_expert_subagent(problem: str) -> str:
        """수학 문제를 해결하는 전문가 Subagent입니다.

        Args:
            problem: 수학 문제 설명

        Returns:
            문제 해결 과정과 답
        """
        prompt = f"""
당신은 수학 전문가입니다.
다음 문제를 단계별로 해결하세요:

{problem}

해결 과정을 명확하게 설명하고 최종 답을 제시하세요.
"""
        response = llm.invoke(prompt)
        return response.content

    print("\n📝 Subagent 생성 완료:")
    print("-" * 70)
    print("  ✅ math_expert_subagent")
    print("  - 역할: 수학 문제 해결")
    print("  - 입력: 문제 설명")
    print("  - 출력: 해결 과정 + 답")

    # 테스트
    print("\n🧪 Subagent 테스트:")
    print("-" * 70)

    test_problem = input("수학 문제를 입력하세요 (Enter=기본값): ").strip()
    if not test_problem:
        test_problem = "사과 3개가 있고, 5개를 더 샀습니다. 그 중 2개를 먹었습니다. 남은 사과는?"

    print(f"\n문제: {test_problem}")
    print("\n실행 중...")

    result = math_expert_subagent.invoke({"problem": test_problem})
    print(f"\n결과:\n{result}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 2: 전문가 Subagent들
# ============================================================================

def example_2_specialist_subagents():
    """다양한 전문 분야의 Subagent 구현"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 전문가 Subagent들")
    print("=" * 70)

    print("""
💡 전문화의 이점:
   - 각 Subagent는 특정 도메인에 집중
   - 명확한 시스템 프롬프트로 역할 정의
   - 더 높은 정확도와 일관성
    """)

    # 번역 전문가
    @tool
    def translator_subagent(text: str, target_lang: str = "영어") -> str:
        """텍스트를 번역하는 전문가 Subagent입니다.

        Args:
            text: 번역할 텍스트
            target_lang: 목표 언어

        Returns:
            번역된 텍스트
        """
        prompt = f"""
당신은 전문 번역가입니다.
다음 텍스트를 {target_lang}로 정확하게 번역하세요:

{text}

자연스러운 표현을 사용하고 문화적 맥락을 고려하세요.
"""
        response = llm.invoke(prompt)
        return response.content

    # 요약 전문가
    @tool
    def summarizer_subagent(text: str, max_sentences: int = 3) -> str:
        """긴 텍스트를 요약하는 전문가 Subagent입니다.

        Args:
            text: 요약할 텍스트
            max_sentences: 최대 문장 수

        Returns:
            요약된 텍스트
        """
        prompt = f"""
당신은 전문 요약가입니다.
다음 텍스트를 {max_sentences}문장 이하로 요약하세요:

{text}

핵심 정보만 포함하고 간결하게 작성하세요.
"""
        response = llm.invoke(prompt)
        return response.content

    # 코드 리뷰 전문가
    @tool
    def code_reviewer_subagent(code: str) -> str:
        """코드를 리뷰하는 전문가 Subagent입니다.

        Args:
            code: 리뷰할 코드

        Returns:
            코드 리뷰 결과
        """
        prompt = f"""
당신은 Python 전문가입니다.
다음 코드를 리뷰하고 개선점을 제안하세요:

```python
{code}
```

리뷰 항목: 코드 품질, 잠재적 버그, 성능 개선점
"""
        response = llm.invoke(prompt)
        return response.content

    print("\n📝 전문가 Subagent 목록:")
    print("-" * 70)
    print("1. 번역 전문가")
    print("2. 요약 전문가")
    print("3. 코드 리뷰 전문가")

    choice = input("\n테스트할 전문가를 선택하세요 (1-3): ").strip()

    if choice == "1":
        text = input("번역할 텍스트: ").strip()
        if not text:
            text = "안녕하세요. 오늘 날씨가 좋네요."
        print("\n실행 중...")
        result = translator_subagent.invoke({"text": text, "target_lang": "영어"})
        print(f"\n번역 결과:\n{result}")

    elif choice == "2":
        text = input("요약할 텍스트 (Enter=기본값): ").strip()
        if not text:
            text = """
인공지능(AI)은 최근 몇 년간 급격한 발전을 이루었습니다.
특히 자연어 처리 분야에서 GPT-3, GPT-4와 같은 대규모 언어 모델이 등장하면서
인간 수준의 텍스트 생성이 가능해졌습니다.
이러한 기술은 고객 서비스, 콘텐츠 제작, 교육 등 다양한 분야에서 활용되고 있으며,
앞으로 더욱 많은 혁신을 가져올 것으로 예상됩니다.
"""
        print("\n실행 중...")
        result = summarizer_subagent.invoke({"text": text, "max_sentences": 2})
        print(f"\n요약 결과:\n{result}")

    elif choice == "3":
        code = """
def calc(a, b):
    return a + b
result = calc(5, 3)
print(result)
"""
        print(f"\n리뷰할 코드:\n{code}")
        print("\n실행 중...")
        result = code_reviewer_subagent.invoke({"code": code})
        print(f"\n리뷰 결과:\n{result}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 3: Tool로 래핑된 Subagent
# ============================================================================

def example_3_wrapped_subagent():
    """Subagent를 @tool 데코레이터로 래핑"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Tool로 래핑된 Subagent")
    print("=" * 70)

    print("""
💡 Subagent를 도구로 만들기:
   1. @tool 데코레이터 사용
   2. 명확한 docstring 작성
   3. 타입 힌트 제공
   4. 간결한 반환값
    """)

    # 뉴스 검색 Subagent
    @tool
    def news_search_subagent(topic: str, count: int = 3) -> str:
        """특정 주제의 최신 뉴스를 검색하는 Subagent입니다.

        Args:
            topic: 검색할 뉴스 주제
            count: 검색할 뉴스 개수

        Returns:
            뉴스 제목과 요약
        """
        prompt = f"""
당신은 뉴스 검색 전문가입니다.
'{topic}'에 대한 최신 뉴스 {count}개를 요약하세요.

각 뉴스 형식:
1. [제목] - 간단한 요약

실제 뉴스처럼 구체적이고 현실적인 내용으로 작성하세요.
"""
        response = llm.invoke(prompt)
        return response.content

    # 데이터 분석 Subagent
    @tool
    def data_analyst_subagent(data_description: str) -> str:
        """데이터를 분석하고 인사이트를 제공하는 Subagent입니다.

        Args:
            data_description: 분석할 데이터 설명

        Returns:
            분석 결과 및 인사이트
        """
        prompt = f"""
당신은 데이터 분석 전문가입니다.
다음 데이터를 분석하고 주요 인사이트를 제공하세요:

{data_description}

분석 내용: 패턴, 트렌드, 특이사항, 실행 가능한 인사이트
"""
        response = llm.invoke(prompt)
        return response.content

    print("\n📝 래핑된 Subagent 도구:")
    print("-" * 70)
    print(f"1. {news_search_subagent.name}")
    print(f"2. {data_analyst_subagent.name}")

    choice = input("\n테스트할 도구 (1-2): ").strip()

    if choice == "1":
        topic = input("뉴스 주제: ").strip() or "인공지능"
        print(f"\n'{topic}' 관련 뉴스 검색 중...")
        result = news_search_subagent.invoke({"topic": topic, "count": 3})
        print(f"\n결과:\n{result}")

    elif choice == "2":
        data_desc = """
최근 3개월 웹사이트 방문자:
- 1월: 10,000명
- 2월: 15,000명 (+50%)
- 3월: 22,500명 (+50%)
"""
        print(f"\n데이터:\n{data_desc}")
        print("\n분석 중...")
        result = data_analyst_subagent.invoke({"data_description": data_desc})
        print(f"\n분석 결과:\n{result}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 4: 여러 Subagent 조합
# ============================================================================

def example_4_combining_subagents():
    """여러 Subagent를 조합하여 복잡한 작업 수행"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 여러 Subagent 조합")
    print("=" * 70)

    print("""
💡 Subagent 조합 패턴:
   - 메인 Agent가 여러 Subagent를 순차적으로 호출
   - 각 Subagent의 결과를 다음 단계 입력으로 사용
   - 최종 결과를 통합하여 반환
    """)

    # 검색 Subagent
    @tool
    def research_subagent(topic: str) -> str:
        """주제에 대한 정보를 수집하는 리서치 Subagent"""
        prompt = f"""
당신은 리서치 전문가입니다.
'{topic}'에 대한 정보를 조사하여 요약하세요.

포함 내용: 정의, 주요 특징, 현황 및 트렌드
"""
        response = llm.invoke(prompt)
        return response.content

    # 분석 Subagent
    @tool
    def analysis_subagent(research_data: str) -> str:
        """리서치 데이터를 분석하는 Subagent"""
        prompt = f"""
당신은 데이터 분석 전문가입니다.
다음 리서치 데이터를 분석하고 인사이트를 도출하세요:

{research_data}

분석 내용: 핵심 포인트, 장단점, 향후 전망
"""
        response = llm.invoke(prompt)
        return response.content

    # 작성 Subagent
    @tool
    def writing_subagent(analysis_data: str) -> str:
        """분석 결과를 바탕으로 글을 작성하는 Subagent"""
        prompt = f"""
당신은 전문 작가입니다.
다음 분석 데이터를 바탕으로 읽기 쉬운 글을 작성하세요:

{analysis_data}

작성 가이드: 명확하고 간결한 문장, 논리적 구조
"""
        response = llm.invoke(prompt)
        return response.content

    print("\n📝 3단계 콘텐츠 생성 파이프라인:")
    print("-" * 70)
    print("1단계: 리서치")
    print("2단계: 분석")
    print("3단계: 작성")

    topic = input("\n주제를 입력하세요: ").strip() or "양자 컴퓨팅"

    print(f"\n주제: {topic}")
    print("=" * 70)

    # 1단계
    print("\n[1/3] 리서치 중...")
    research_result = research_subagent.invoke({"topic": topic})
    print(f"\n리서치 결과:\n{research_result[:200]}...")

    input("\n⏎ Enter를 눌러 다음 단계로...")

    # 2단계
    print("\n[2/3] 분석 중...")
    analysis_result = analysis_subagent.invoke({"research_data": research_result})
    print(f"\n분석 결과:\n{analysis_result[:200]}...")

    input("\n⏎ Enter를 눌러 다음 단계로...")

    # 3단계
    print("\n[3/3] 글 작성 중...")
    final_result = writing_subagent.invoke({"analysis_data": analysis_result})
    print(f"\n최종 결과:\n{final_result}")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 5: 실전 - 연구 보조 Agent
# ============================================================================

def example_5_research_assistant():
    """리서치와 요약 Subagent를 활용한 연구 보조 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 연구 보조 Agent")
    print("=" * 70)

    print("""
🎯 실전 시나리오: 학술 연구 보조 시스템

구성:
   - Literature Review: 문헌 조사
   - Data Collection: 데이터 수집
   - Summary: 요약 및 정리
   - Citation: 인용 형식 변환
    """)

    # 문헌 조사 Subagent
    @tool
    def literature_review_subagent(research_question: str) -> str:
        """연구 질문에 대한 문헌을 조사"""
        prompt = f"""
당신은 학술 연구 전문가입니다.
다음 연구 질문에 대한 문헌을 조사하세요:

연구 질문: {research_question}

포함 정보: 주요 연구 논문 3-5개, 각 논문의 핵심 내용, 연구 동향
"""
        response = llm.invoke(prompt)
        return response.content

    # 데이터 수집 Subagent
    @tool
    def data_collection_subagent(topic: str) -> str:
        """주제와 관련된 데이터를 수집"""
        prompt = f"""
당신은 데이터 수집 전문가입니다.
'{topic}'에 대한 데이터를 수집하고 정리하세요.

포함: 통계 데이터, 사례 연구, 실증적 증거
"""
        response = llm.invoke(prompt)
        return response.content

    # 요약 Subagent
    @tool
    def summary_subagent(content: str) -> str:
        """내용을 요약"""
        prompt = f"""
당신은 요약 전문가입니다.
다음 내용을 글머리 기호로 요약하세요:

{content}

핵심만 간결하게 정리하세요.
"""
        response = llm.invoke(prompt)
        return response.content

    # 인용 Subagent
    @tool
    def citation_subagent(source: str, style: str = "APA") -> str:
        """출처를 특정 형식으로 인용"""
        prompt = f"""
당신은 학술 인용 전문가입니다.
다음 출처를 {style} 스타일로 인용하세요:

{source}

정확한 {style} 형식을 사용하세요.
"""
        response = llm.invoke(prompt)
        return response.content

    print("\n📚 연구 보조 시스템 시작")
    print("=" * 70)

    research_question = input("연구 질문: ").strip()
    if not research_question:
        research_question = "기계 학습이 의료 진단에 미치는 영향"

    print(f"\n연구 질문: {research_question}")
    print("\n" + "=" * 70)

    # 1. 문헌 조사
    print("\n[1/4] 문헌 조사 중...")
    literature = literature_review_subagent.invoke({"research_question": research_question})
    print(f"\n문헌 조사:\n{literature[:300]}...")

    input("\n⏎ Enter를 눌러 계속...")

    # 2. 데이터 수집
    print("\n[2/4] 데이터 수집 중...")
    data = data_collection_subagent.invoke({"topic": research_question})
    print(f"\n데이터 수집:\n{data[:300]}...")

    input("\n⏎ Enter를 눌러 계속...")

    # 3. 종합 요약
    print("\n[3/4] 종합 요약 중...")
    combined = f"문헌:\n{literature}\n\n데이터:\n{data}"
    summary = summary_subagent.invoke({"content": combined})
    print(f"\n요약:\n{summary}")

    input("\n⏎ Enter를 눌러 계속...")

    # 4. 인용 예시
    print("\n[4/4] 인용 형식 변환 예시...")
    example_source = "Smith, J. (2023). ML in Healthcare. Journal of Medical AI, 15(2), 123-145."
    citation = citation_subagent.invoke({"source": example_source, "style": "APA"})
    print(f"\n인용:\n{citation}")

    print("\n" + "=" * 70)
    print("✅ 연구 보조 완료!")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("=" * 70)
    print("Part 7: Multi-Agent Systems")
    print("02. Subagents (기본)")
    print("=" * 70)

    while True:
        print("\n")
        print("📚 실행할 예제를 선택하세요:")
        print("-" * 70)
        print("1. Subagent의 기본 개념")
        print("2. 전문가 Subagent들")
        print("3. Tool로 래핑된 Subagent")
        print("4. 여러 Subagent 조합")
        print("5. 실전: 연구 보조 Agent")
        print("0. 종료")
        print("-" * 70)

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_subagent_concept()
        elif choice == "2":
            example_2_specialist_subagents()
        elif choice == "3":
            example_3_wrapped_subagent()
        elif choice == "4":
            example_4_combining_subagents()
        elif choice == "5":
            example_5_research_assistant()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")

    print("\n" + "=" * 70)
    print("📚 학습 완료!")
    print("=" * 70)
    print("""
✅ 배운 내용:
   - Subagent 패턴의 개념과 작동 원리
   - @tool 데코레이터로 Subagent를 도구로 래핑
   - 전문화된 Subagent 구현
   - 여러 Subagent를 순차적으로 조합
   - 실전 연구 보조 시스템 구축

💡 핵심 요약:
   ┌─────────────────────────────────────────────────────────────────┐
   │ Subagent는 메인 Agent가 "도구처럼" 사용하는 전문 Agent         │
   │                                                                   │
   │ 주요 특징:                                                       │
   │ • 메인 Agent가 제어 유지                                        │
   │ • 각 Subagent는 특정 작업에 전문화                              │
   │ • @tool 데코레이터로 쉽게 래핑                                  │
   │ • 여러 Subagent를 조합하여 복잡한 작업 수행                     │
   └─────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()
