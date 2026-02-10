"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP
================================================================================

파일명: 01_rag_basics.py
난이도: ⭐⭐⭐ (중급)
예상 시간: 30분

📚 학습 목표:
  - RAG (Retrieval Augmented Generation)의 개념 이해
  - Vector Store의 작동 원리 이해
  - 유사도 검색 구현
  - Top-k retrieval 사용
  - 실전 FAQ 검색 시스템 구축

📖 공식 문서:
  • Retrieval: /official/28-retrieval.md
  • Vector Stores: https://python.langchain.com/docs/concepts/vectorstores/

📄 교안 문서:
  • Part 8: /docs/part08_rag_mcp.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv

🚀 실행 방법:
  python 01_rag_basics.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: RAG가 필요한 이유 (Before & After)
# ============================================================================

def example_1_why_rag():
    """RAG가 없을 때와 있을 때의 차이"""
    print("=" * 70)
    print("📌 예제 1: RAG가 필요한 이유")
    print("=" * 70)

    print("""
💡 RAG (Retrieval Augmented Generation)란?
   - LLM의 생성 능력 + 외부 지식 검색을 결합
   - LLM의 한계 극복:
     1. 유한한 컨텍스트 윈도우
     2. 정적인 지식 (학습 시점 고정)

❌ RAG 없이:
   - 회사 내부 정보 모름
   - 최신 데이터 없음
   - 잘못된 정보 생성 가능

✅ RAG 사용:
   - 실제 문서에서 검색
   - 정확한 정보 제공
   - 출처 추적 가능
    """)

    # 시나리오: 회사 재무 정보 조회
    company_data = [
        "2024년 Q1 매출: $4.2M, 순이익: $0.8M",
        "2024년 Q2 매출: $4.8M, 순이익: $1.1M",
        "2024년 Q3 매출: $5.2M, 순이익: $1.3M",
        "2024년 연간 목표: 매출 $20M, 순이익 $5M",
        "2024년 신규 고객: Q1 125명, Q2 180명, Q3 210명"
    ]

    print("\n📊 회사 데이터:")
    for data in company_data:
        print(f"  • {data}")

    # RAG 없이 (LLM만 사용)
    print("\n" + "=" * 70)
    print("❌ RAG 없이 - LLM만 사용")
    print("=" * 70)

    agent_without_rag = create_agent(
        model="gpt-4o-mini",
        tools=[],  # 도구 없음
    )

    response = agent_without_rag.invoke({
        "messages": [{"role": "user", "content": "우리 회사의 2024년 Q3 매출은 얼마인가요?"}]
    })

    print(f"\n🤖 Agent 답변 (RAG 없음):")
    print(f"{response['messages'][-1].content}")
    print("\n⚠️  결과: LLM은 회사 내부 데이터를 모르므로 정확한 답변 불가")

    # RAG 사용
    print("\n" + "=" * 70)
    print("✅ RAG 사용 - Vector Store로 검색")
    print("=" * 70)

    # 1. Embeddings 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Vector Store 생성
    vectorstore = FAISS.from_texts(
        texts=company_data,
        embedding=embeddings
    )

    # 3. 검색 도구 생성
    @tool
    def search_company_data(query: str) -> str:
        """회사 재무 데이터 검색"""
        docs = vectorstore.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])

    # 4. Agent에 검색 도구 추가
    agent_with_rag = create_agent(
        model="gpt-4o-mini",
        tools=[search_company_data],
        system_prompt="당신은 회사 재무 데이터를 검색하여 정확한 정보를 제공하는 전문가입니다."
    )

    response = agent_with_rag.invoke({
        "messages": [{"role": "user", "content": "우리 회사의 2024년 Q3 매출은 얼마인가요?"}]
    })

    print(f"\n🤖 Agent 답변 (RAG 사용):")
    print(f"{response['messages'][-1].content}")
    print("\n✅ 결과: Vector Store에서 정확한 데이터를 검색하여 정확한 답변 제공")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 2: Vector Store 기본 - 유사도 검색
# ============================================================================

def example_2_similarity_search():
    """Vector Store를 사용한 유사도 검색"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 유사도 검색 (Similarity Search)")
    print("=" * 70)

    print("""
💡 유사도 검색이란?
   - 쿼리와 의미적으로 유사한 문서 찾기
   - 벡터 간 거리(코사인 유사도) 계산
   - 가장 가까운 k개 문서 반환

작동 과정:
   1. 텍스트 → 벡터 변환 (Embedding)
   2. 쿼리 → 벡터 변환
   3. 벡터 간 유사도 계산
   4. 상위 k개 반환
    """)

    # 샘플 문서
    documents = [
        "파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다.",
        "자바는 객체지향 프로그래밍 언어로 많은 기업에서 사용됩니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 기술입니다.",
        "딥러닝은 인공신경망을 사용하는 머신러닝의 한 분야입니다.",
        "자연어처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술입니다.",
        "강아지는 충성스럽고 사람을 좋아하는 애완동물입니다.",
        "고양이는 독립적이고 깨끗한 애완동물입니다.",
        "React는 사용자 인터페이스를 만들기 위한 자바스크립트 라이브러리입니다.",
    ]

    print("\n📚 문서 목록:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    # Vector Store 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(documents, embeddings)

    # 다양한 쿼리로 검색
    queries = [
        "프로그래밍 언어에 대해 알려주세요",
        "AI와 머신러닝이란?",
        "애완동물 추천해주세요"
    ]

    for query in queries:
        print("\n" + "-" * 70)
        print(f"🔍 쿼리: {query}")
        print("-" * 70)

        # 유사도 검색 (k=2)
        results = vectorstore.similarity_search(query, k=2)

        print("\n📄 검색 결과 (상위 2개):")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 3: 유사도 점수와 함께 검색
# ============================================================================

def example_3_search_with_scores():
    """유사도 점수를 함께 반환하는 검색"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 유사도 점수와 함께 검색")
    print("=" * 70)

    print("""
💡 유사도 점수란?
   - 쿼리와 문서 간의 유사도를 숫자로 표현
   - 낮을수록 유사함 (거리 기반)
   - 임계값 설정으로 품질 관리 가능
    """)

    # 기술 문서
    tech_docs = [
        "Docker는 컨테이너 기반 가상화 플랫폼입니다.",
        "Kubernetes는 컨테이너 오케스트레이션 도구입니다.",
        "Git은 분산 버전 관리 시스템입니다.",
        "PostgreSQL은 오픈소스 관계형 데이터베이스입니다.",
        "MongoDB는 NoSQL 문서 데이터베이스입니다.",
        "Redis는 인메모리 키-값 저장소입니다.",
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(tech_docs, embeddings)

    query = "컨테이너 관련 기술"

    print(f"\n🔍 쿼리: {query}\n")

    # 유사도 점수와 함께 검색
    results_with_scores = vectorstore.similarity_search_with_score(query, k=4)

    print("📊 검색 결과 (점수 포함):")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        relevance = "높음" if score < 0.3 else "중간" if score < 0.5 else "낮음"
        print(f"\n  {i}. 점수: {score:.4f} (관련성: {relevance})")
        print(f"     내용: {doc.page_content}")

    # 임계값 적용
    threshold = 0.4
    print(f"\n" + "-" * 70)
    print(f"📌 임계값 {threshold} 이하만 필터링:")
    print("-" * 70)

    filtered_results = [
        (doc, score) for doc, score in results_with_scores
        if score < threshold
    ]

    if filtered_results:
        for i, (doc, score) in enumerate(filtered_results, 1):
            print(f"\n  {i}. 점수: {score:.4f}")
            print(f"     {doc.page_content}")
    else:
        print("  ⚠️  임계값을 통과한 결과 없음")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 4: Top-k Retrieval 비교
# ============================================================================

def example_4_topk_retrieval():
    """다양한 k 값에 따른 검색 결과 비교"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Top-k Retrieval")
    print("=" * 70)

    print("""
💡 Top-k란?
   - 상위 k개의 가장 유사한 문서 반환
   - k 값 선택이 중요:
     • k가 너무 작으면: 충분한 정보 부족
     • k가 너무 크면: 관련 없는 정보 포함, 비용 증가

권장 k 값:
   - 간단한 질문: k=3
   - 복잡한 질문: k=5~10
   - 컨텍스트 윈도우 고려 필요
    """)

    # 영화 리뷰 데이터
    movie_reviews = [
        "어벤져스는 액션과 스토리가 완벽한 슈퍼히어로 영화입니다.",
        "인터스텔라는 감동적인 SF 걸작입니다.",
        "기생충은 사회 계층을 다룬 뛰어난 한국 영화입니다.",
        "타이타닉은 로맨스와 재난을 결합한 감동적인 영화입니다.",
        "조커는 빌런의 심리를 깊이 있게 탐구한 작품입니다.",
        "인셉션은 복잡한 스토리 구조의 SF 스릴러입니다.",
        "라라랜드는 음악과 로맨스가 아름다운 뮤지컬 영화입니다.",
        "매드맥스는 강렬한 액션의 포스트 아포칼립스 영화입니다.",
        "위플래쉬는 재즈 드러머의 열정을 그린 드라마입니다.",
        "그래비티는 우주를 배경으로 한 생존 스릴러입니다.",
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(movie_reviews, embeddings)

    query = "SF 영화 추천해주세요"

    print(f"\n🔍 쿼리: {query}\n")

    # 다양한 k 값으로 검색
    k_values = [1, 3, 5]

    for k in k_values:
        print(f"\n{'=' * 70}")
        print(f"📌 k={k}개 검색 결과:")
        print("=" * 70)

        results = vectorstore.similarity_search(query, k=k)

        for i, doc in enumerate(results, 1):
            print(f"\n  {i}. {doc.page_content}")

        # k 값에 따른 평가
        if k == 1:
            print("\n  💡 k=1: 가장 관련성 높은 단일 결과. 빠르지만 정보 부족 가능.")
        elif k == 3:
            print("\n  ✅ k=3: 균형잡힌 선택. 충분한 정보 + 관련성 유지.")
        elif k == 5:
            print("\n  📚 k=5: 더 많은 컨텍스트. 복잡한 질문에 적합.")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 5: 실전 - FAQ 검색 시스템
# ============================================================================

def example_5_faq_search_system():
    """실전 FAQ 검색 Agent 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 FAQ 검색 시스템")
    print("=" * 70)

    print("""
💡 실전 시나리오:
   - 고객 지원 FAQ 시스템
   - 질문 입력 → 관련 FAQ 검색 → 답변 생성
   - Agent가 검색 결과를 바탕으로 답변

구현 단계:
   1. FAQ 데이터 준비
   2. Vector Store 구축
   3. 검색 도구 생성
   4. Agent 생성
   5. 사용자 질문 처리
    """)

    # FAQ 데이터 (Document 형태로 메타데이터 포함)
    faq_data = [
        Document(
            page_content="Q: 배송은 얼마나 걸리나요?\nA: 일반 배송은 2-3일, 빠른 배송은 1일 소요됩니다.",
            metadata={"category": "배송", "id": 1}
        ),
        Document(
            page_content="Q: 반품은 어떻게 하나요?\nA: 구매 후 7일 이내 미개봉 상품에 한해 반품 가능합니다.",
            metadata={"category": "반품", "id": 2}
        ),
        Document(
            page_content="Q: 결제 수단은 무엇이 있나요?\nA: 신용카드, 계좌이체, 무통장입금, 카카오페이를 지원합니다.",
            metadata={"category": "결제", "id": 3}
        ),
        Document(
            page_content="Q: 회원가입 혜택은 무엇인가요?\nA: 첫 구매 10% 할인, 적립금 5% 지급, 생일 쿠폰 제공됩니다.",
            metadata={"category": "회원", "id": 4}
        ),
        Document(
            page_content="Q: 재고가 없으면 어떻게 되나요?\nA: 재입고 알림 신청 시 입고되면 이메일로 알려드립니다.",
            metadata={"category": "상품", "id": 5}
        ),
        Document(
            page_content="Q: 배송비는 얼마인가요?\nA: 3만원 이상 무료배송, 미만 시 3,000원입니다.",
            metadata={"category": "배송", "id": 6}
        ),
        Document(
            page_content="Q: 교환은 가능한가요?\nA: 사이즈 불만족 시 7일 이내 1회 무료 교환 가능합니다.",
            metadata={"category": "반품", "id": 7}
        ),
        Document(
            page_content="Q: 적립금은 언제 사용 가능한가요?\nA: 구매 확정 후 3일 뒤부터 사용 가능하며, 유효기간은 1년입니다.",
            metadata={"category": "회원", "id": 8}
        ),
    ]

    print("\n📚 FAQ 데이터베이스 구축 중...")

    # Vector Store 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(faq_data, embeddings)

    print("✅ Vector Store 구축 완료!")

    # 검색 도구 생성
    @tool
    def search_faq(question: str) -> str:
        """FAQ 데이터베이스에서 관련 질문 검색"""
        docs = vectorstore.similarity_search(question, k=3)
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"[FAQ {i}]\n{doc.page_content}\n(카테고리: {doc.metadata['category']})")
        return "\n\n".join(results)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_faq],
        system_prompt="""당신은 친절한 고객 지원 상담원입니다.

사용자의 질문에 답변할 때:
1. search_faq 도구로 관련 FAQ를 검색하세요
2. 검색된 FAQ를 바탕으로 친절하게 답변하세요
3. 정확한 정보를 제공하고, 추가 질문이 있는지 물어보세요
"""
    )

    # 테스트 질문들
    test_questions = [
        "배송은 얼마나 걸리나요?",
        "반품하고 싶은데 어떻게 해야 하나요?",
        "회원가입하면 어떤 혜택이 있나요?",
    ]

    print("\n" + "=" * 70)
    print("🤖 FAQ 검색 Agent 테스트")
    print("=" * 70)

    for question in test_questions:
        print(f"\n{'=' * 70}")
        print(f"❓ 사용자 질문: {question}")
        print("=" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 Agent 답변:\n{answer}")

    # 사용자 입력 받기
    print("\n" + "=" * 70)
    print("💬 직접 질문해보세요 (종료: 'quit' 입력)")
    print("=" * 70)

    user_question = input("\n❓ 질문: ").strip()

    if user_question and user_question.lower() != 'quit':
        print("\n🔍 검색 중...")

        response = agent.invoke({
            "messages": [{"role": "user", "content": user_question}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 Agent 답변:\n{answer}")

    print("\n" + "=" * 70)
    print("✅ FAQ 검색 시스템 완료!")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n")
    print("=" * 70)
    print("Part 8: RAG 기초 (01_rag_basics.py)")
    print("=" * 70)

    while True:
        print("\n📚 실행할 예제를 선택하세요:")
        print("  1. RAG가 필요한 이유 (Before & After)")
        print("  2. 유사도 검색 (Similarity Search)")
        print("  3. 유사도 점수와 함께 검색")
        print("  4. Top-k Retrieval 비교")
        print("  5. 실전 FAQ 검색 시스템 ⭐")
        print("  0. 종료")

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_why_rag()
        elif choice == "2":
            example_2_similarity_search()
        elif choice == "3":
            example_3_search_with_scores()
        elif choice == "4":
            example_4_topk_retrieval()
        elif choice == "5":
            example_5_faq_search_system()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
