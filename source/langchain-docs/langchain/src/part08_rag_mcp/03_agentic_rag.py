"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP
================================================================================

파일명: 03_agentic_rag.py
난이도: ⭐⭐⭐⭐ (중상급)
예상 시간: 35분

📚 학습 목표:
  - Retriever를 Tool로 변환
  - Agent에 Retriever tool 통합
  - Query planning (query → subqueries)
  - Self-RAG pattern 구현
  - 실전 지식 기반 Q&A Agent

📖 공식 문서:
  • Retrieval: https://python.langchain.com/docs/concepts/retrieval/
  • Tools: https://python.langchain.com/docs/concepts/tools/

📄 교안 문서:
  • Part 8: /docs/part08_rag_mcp.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv

🚀 실행 방법:
  python 03_agentic_rag.py

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
from langchain.tools.retriever import create_retriever_tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: Retriever를 Tool로 변환
# ============================================================================

def example_1_retriever_as_tool():
    """Retriever를 Tool로 변환하여 Agent에서 사용"""
    print("=" * 70)
    print("📌 예제 1: Retriever를 Tool로 변환")
    print("=" * 70)

    print("""
💡 Retriever Tool이란?
   - Vector Store의 검색 기능을 Tool로 래핑
   - Agent가 필요할 때 검색 호출 가능
   - 명확한 도구 설명이 중요

방법:
   1. Vector Store 생성
   2. as_retriever()로 Retriever 변환
   3. create_retriever_tool()로 Tool 생성
   4. Agent에 Tool 추가
    """)

    # 회사 정책 문서
    policy_docs = [
        Document(
            page_content="연차 휴가는 입사 1년 후 15일이 부여됩니다. 매년 근속 연수에 따라 1일씩 추가됩니다.",
            metadata={"category": "휴가", "doc_id": "POL-001"}
        ),
        Document(
            page_content="재택근무는 주 2회까지 가능하며, 사전에 팀장 승인이 필요합니다.",
            metadata={"category": "근무", "doc_id": "POL-002"}
        ),
        Document(
            page_content="경조사 휴가는 결혼 5일, 출산 3일, 직계 가족 사망 5일입니다.",
            metadata={"category": "휴가", "doc_id": "POL-003"}
        ),
        Document(
            page_content="점심 식대는 1일 만원이 지원되며, 법인 카드로 결제합니다.",
            metadata={"category": "복지", "doc_id": "POL-004"}
        ),
        Document(
            page_content="건강검진은 연 1회 제공되며, 비용은 회사에서 전액 부담합니다.",
            metadata={"category": "복지", "doc_id": "POL-005"}
        ),
    ]

    print(f"\n📚 문서 수: {len(policy_docs)}")

    # Vector Store 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(policy_docs, embeddings)

    print("✅ Vector Store 생성 완료")

    # 1. 기본 Retriever Tool
    print("\n1️⃣ 기본 Retriever Tool 생성")
    print("-" * 70)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="search_company_policy",
        description="회사 정책 및 규정을 검색합니다. 휴가, 근무, 복지 관련 질문에 사용하세요."
    )

    print("✅ Retriever Tool 생성 완료")
    print(f"  • 도구 이름: {retriever_tool.name}")
    print(f"  • 설명: {retriever_tool.description}")

    # 2. Agent 생성
    print("\n2️⃣ Agent 생성 및 테스트")
    print("-" * 70)

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[retriever_tool],
        system_prompt="""당신은 회사 HR 담당자입니다.
        
직원들의 질문에 답변할 때:
1. search_company_policy 도구로 관련 정책 검색
2. 검색된 정보를 바탕으로 친절하게 답변
3. 정책 문서 번호(doc_id)도 함께 안내"""
    )

    # 테스트 질문
    test_questions = [
        "연차는 며칠 받을 수 있나요?",
        "재택근무가 가능한가요?",
        "회사에서 제공하는 복지는 무엇이 있나요?"
    ]

    for question in test_questions:
        print(f"\n❓ 질문: {question}")
        print("-" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        answer = response['messages'][-1].content
        print(f"🤖 답변: {answer}\n")

    print("=" * 70)


# ============================================================================
# 예제 2: Agent에 여러 Retriever 통합
# ============================================================================

def example_2_multiple_retrievers():
    """여러 도메인의 Retriever를 Agent에 통합"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 여러 Retriever 통합")
    print("=" * 70)

    print("""
💡 다중 Retriever 전략:
   - 도메인별로 별도 Vector Store 구축
   - 각각을 독립된 Tool로 제공
   - Agent가 질문에 맞는 Tool 선택

장점:
   • 검색 정확도 향상
   • 명확한 도메인 분리
   • 관련 없는 문서 혼입 방지
    """)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. HR 정책 Vector Store
    hr_docs = [
        "연차는 입사 1년 후 15일 부여됩니다.",
        "육아 휴직은 자녀 만 8세까지 사용 가능합니다.",
        "병가는 연 10일까지 가능하며, 진단서 제출이 필요합니다."
    ]
    hr_vectorstore = FAISS.from_texts(hr_docs, embeddings)
    hr_retriever = hr_vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. 기술 문서 Vector Store
    tech_docs = [
        "Python 개발 환경은 Docker 컨테이너로 제공됩니다.",
        "Git branch는 feature/이슈번호 형식으로 생성하세요.",
        "코드 리뷰는 최소 2명의 승인이 필요합니다."
    ]
    tech_vectorstore = FAISS.from_texts(tech_docs, embeddings)
    tech_retriever = tech_vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. 재무 정책 Vector Store
    finance_docs = [
        "교통비는 월 10만원까지 지원됩니다.",
        "업무 관련 도서는 월 5만원까지 구매 가능합니다.",
        "회식비는 팀당 분기별 30만원 한도입니다."
    ]
    finance_vectorstore = FAISS.from_texts(finance_docs, embeddings)
    finance_retriever = finance_vectorstore.as_retriever(search_kwargs={"k": 2})

    print("✅ 3개 도메인 Vector Store 생성 완료")

    # Tool 생성
    hr_tool = create_retriever_tool(
        hr_retriever,
        "search_hr_policy",
        "인사 정책을 검색합니다 (휴가, 휴직, 병가 등)"
    )

    tech_tool = create_retriever_tool(
        tech_retriever,
        "search_tech_docs",
        "기술 문서를 검색합니다 (개발 환경, Git, 코드 리뷰 등)"
    )

    finance_tool = create_retriever_tool(
        finance_retriever,
        "search_finance_policy",
        "재무 정책을 검색합니다 (교통비, 도서비, 회식비 등)"
    )

    print("\n📦 생성된 Tools:")
    for tool in [hr_tool, tech_tool, finance_tool]:
        print(f"  • {tool.name}: {tool.description}")

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[hr_tool, tech_tool, finance_tool],
        system_prompt="""당신은 회사 통합 정보 안내 Agent입니다.
        
질문의 카테고리에 맞는 검색 도구를 선택하세요:
- 휴가, 인사 관련 → search_hr_policy
- 개발, 기술 관련 → search_tech_docs
- 비용, 재무 관련 → search_finance_policy"""
    )

    # 다양한 도메인 질문
    test_questions = [
        ("연차는 며칠인가요?", "HR"),
        ("Git 브랜치 규칙이 뭐죠?", "Tech"),
        ("도서 구매 예산은 얼마인가요?", "Finance"),
        ("코드 리뷰는 몇 명이 필요한가요?", "Tech"),
    ]

    print("\n" + "=" * 70)
    print("🧪 다중 도메인 질문 테스트")
    print("=" * 70)

    for question, domain in test_questions:
        print(f"\n❓ 질문 ({domain}): {question}")
        print("-" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        answer = response['messages'][-1].content
        print(f"🤖 답변: {answer[:150]}...")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 3: Query Planning - 쿼리 분해
# ============================================================================

def example_3_query_planning():
    """복잡한 질문을 여러 서브 쿼리로 분해"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Query Planning")
    print("=" * 70)

    print("""
💡 Query Planning이란?
   - 복잡한 질문을 여러 단계로 분해
   - 각 단계별로 정보 수집
   - 최종 답변 통합

예시:
   질문: "작년과 올해 매출 비교하면?"
   → 1단계: "작년 매출 검색"
   → 2단계: "올해 매출 검색"
   → 3단계: "두 값 비교"
    """)

    # 재무 데이터
    finance_data = [
        "2023년 Q1 매출: $4.2M, 비용: $3.1M, 순이익: $1.1M",
        "2023년 Q2 매출: $4.8M, 비용: $3.3M, 순이익: $1.5M",
        "2023년 Q3 매출: $5.2M, 비용: $3.5M, 순이익: $1.7M",
        "2023년 Q4 매출: $5.8M, 비용: $3.7M, 순이익: $2.1M",
        "2024년 Q1 매출: $5.5M, 비용: $3.6M, 순이익: $1.9M",
        "2024년 Q2 매출: $6.2M, 비용: $3.9M, 순이익: $2.3M",
        "2024년 Q3 매출: $6.8M, 비용: $4.1M, 순이익: $2.7M",
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(finance_data, embeddings)

    # 검색 도구
    @tool
    def search_finance_data(query: str) -> str:
        """재무 데이터 검색"""
        docs = vectorstore.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])

    # Query Planning 도구
    @tool
    def plan_complex_query(question: str) -> str:
        """복잡한 질문을 여러 단계로 분해
        
        Args:
            question: 복잡한 질문
            
        Returns:
            단계별 검색 계획 (JSON 형식)
        """
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        prompt = f"""다음 질문을 여러 검색 단계로 분해하세요:

질문: {question}

각 단계를 다음 형식으로 작성:
1. [검색 쿼리 1]
2. [검색 쿼리 2]
...

단계별 검색 쿼리만 작성하세요."""

        response = llm.invoke(prompt)
        return response.content

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_finance_data, plan_complex_query],
        system_prompt="""당신은 재무 분석 전문가입니다.

복잡한 질문을 처리하는 방법:
1. plan_complex_query로 질문을 단계별로 분해
2. 각 단계마다 search_finance_data로 데이터 검색
3. 모든 데이터를 종합하여 답변"""
    )

    # 복잡한 질문들
    complex_questions = [
        "2023년과 2024년 Q1 매출을 비교하면?",
        "2024년 상반기 평균 순이익은 얼마인가요?",
    ]

    print("\n" + "=" * 70)
    print("🧪 복잡한 질문 처리 테스트")
    print("=" * 70)

    for question in complex_questions:
        print(f"\n❓ 질문: {question}")
        print("=" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 답변:\n{answer}\n")

    print("=" * 70)


# ============================================================================
# 예제 4: Self-RAG - 검색 결과 자기 평가
# ============================================================================

def example_4_self_rag():
    """검색 결과를 스스로 평가하고 개선하는 Self-RAG"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Self-RAG (자기 검증 RAG)")
    print("=" * 70)

    print("""
💡 Self-RAG란?
   - 검색 결과의 품질을 Agent가 스스로 평가
   - 부족하면 쿼리 개선 후 재검색
   - 충분할 때까지 반복

프로세스:
   1. 초기 검색
   2. 결과 품질 평가
   3. 부족하면 → 쿼리 개선 → 재검색
   4. 충분하면 → 답변 생성
    """)

    # 기술 문서
    tech_docs = [
        "Docker는 컨테이너 기반 가상화 플랫폼입니다.",
        "Docker Compose는 여러 컨테이너를 정의하고 실행하는 도구입니다.",
        "Kubernetes는 컨테이너 오케스트레이션 플랫폼입니다.",
        "Kubernetes는 자동 스케일링과 로드밸런싱을 제공합니다.",
        "Helm은 Kubernetes 패키지 관리자입니다.",
        "Dockerfile은 컨테이너 이미지를 빌드하기 위한 명령을 정의합니다.",
        "kubectl은 Kubernetes 클러스터를 관리하는 CLI 도구입니다.",
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(tech_docs, embeddings)

    # 도구 정의
    @tool
    def search_tech_docs(query: str) -> str:
        """기술 문서 검색"""
        docs = vectorstore.similarity_search(query, k=3)
        results = "\n".join([f"- {d.page_content}" for d in docs])
        return f"검색 결과:\n{results}"

    @tool
    def evaluate_search_quality(original_question: str, search_results: str) -> str:
        """검색 결과가 질문에 답하기 충분한지 평가
        
        Returns:
            'SUFFICIENT' 또는 'INSUFFICIENT'와 이유
        """
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        prompt = f"""질문: {original_question}

검색 결과:
{search_results}

이 검색 결과가 질문에 답하기 충분한가요?

응답 형식:
판정: SUFFICIENT 또는 INSUFFICIENT
이유: [간단한 설명]"""

        response = llm.invoke(prompt)
        return response.content

    @tool
    def improve_search_query(original_question: str, previous_results: str, feedback: str) -> str:
        """검색 쿼리를 개선
        
        Returns:
            개선된 검색 쿼리
        """
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        prompt = f"""원래 질문: {original_question}

이전 검색 결과:
{previous_results}

피드백:
{feedback}

더 나은 검색 결과를 얻기 위한 새로운 검색 쿼리를 작성하세요.
쿼리만 반환하세요."""

        response = llm.invoke(prompt)
        return response.content

    # Self-RAG Agent
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_tech_docs, evaluate_search_quality, improve_search_query],
        system_prompt="""당신은 기술 문서 검색 전문가입니다.

Self-RAG 프로세스:
1. search_tech_docs로 초기 검색
2. evaluate_search_quality로 결과 평가
3. INSUFFICIENT이면:
   - improve_search_query로 쿼리 개선
   - 개선된 쿼리로 재검색
   - 다시 평가
4. SUFFICIENT이면:
   - 검색 결과를 바탕으로 답변

최대 2회까지만 재검색하세요."""
    )

    # 테스트 질문
    test_questions = [
        "컨테이너 관리 도구는 무엇이 있나요?",
        "Kubernetes의 주요 기능은 무엇인가요?",
    ]

    print("\n" + "=" * 70)
    print("🧪 Self-RAG 테스트")
    print("=" * 70)

    for question in test_questions:
        print(f"\n❓ 질문: {question}")
        print("=" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 최종 답변:\n{answer}\n")

    print("=" * 70)


# ============================================================================
# 예제 5: 실전 - 지식 기반 Q&A Agent
# ============================================================================

def example_5_knowledge_qa_agent():
    """실전 지식 기반 Q&A Agent 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 지식 기반 Q&A Agent")
    print("=" * 70)

    print("""
💡 실전 시나리오:
   - 회사 전체 지식 베이스 검색
   - 다중 도메인 지원
   - 검색 결과 평가 및 개선
   - 출처 추적 및 신뢰도 표시

기능:
   1. 여러 지식 베이스 통합
   2. 자동 쿼리 개선
   3. 답변 신뢰도 평가
   4. 출처 문서 표시
    """)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 지식 베이스 구축
    knowledge_base = [
        Document(
            page_content="LangChain은 LLM 애플리케이션을 위한 프레임워크입니다. Agent, Chain, Tool 등의 개념을 제공합니다.",
            metadata={"source": "tech_docs", "topic": "langchain", "reliability": "high"}
        ),
        Document(
            page_content="RAG는 Retrieval Augmented Generation의 약자로, 외부 지식을 검색하여 LLM 답변을 개선하는 기법입니다.",
            metadata={"source": "tech_docs", "topic": "rag", "reliability": "high"}
        ),
        Document(
            page_content="Vector Store는 임베딩된 문서를 저장하고 유사도 검색을 수행하는 데이터베이스입니다. FAISS, Chroma, Pinecone 등이 있습니다.",
            metadata={"source": "tech_docs", "topic": "vector_store", "reliability": "high"}
        ),
        Document(
            page_content="Agent는 LLM이 도구를 사용하여 작업을 수행할 수 있게 하는 패턴입니다. ReAct 패턴을 주로 사용합니다.",
            metadata={"source": "tech_docs", "topic": "agent", "reliability": "high"}
        ),
        Document(
            page_content="Embedding은 텍스트를 고차원 벡터로 변환하는 과정입니다. OpenAI의 text-embedding-3-small 모델을 권장합니다.",
            metadata={"source": "tech_docs", "topic": "embedding", "reliability": "medium"}
        ),
        Document(
            page_content="회사는 연 15일의 연차를 제공하며, 근속 연수에 따라 추가 부여됩니다.",
            metadata={"source": "hr_policy", "topic": "vacation", "reliability": "high"}
        ),
        Document(
            page_content="재택근무는 주 2회까지 가능하며, 팀장 승인이 필요합니다.",
            metadata={"source": "hr_policy", "topic": "remote_work", "reliability": "high"}
        ),
    ]

    vectorstore = FAISS.from_documents(knowledge_base, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(f"✅ 지식 베이스 구축 완료 ({len(knowledge_base)}개 문서)")

    # 고급 검색 도구
    @tool
    def search_knowledge_base(query: str) -> str:
        """회사 지식 베이스 검색 (기술 문서, HR 정책 등)
        
        Returns:
            검색 결과와 메타데이터 (출처, 신뢰도)
        """
        docs = retriever.invoke(query)
        
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            topic = doc.metadata.get("topic", "general")
            reliability = doc.metadata.get("reliability", "medium")
            
            results.append(
                f"[문서 {i}] (출처: {source}, 주제: {topic}, 신뢰도: {reliability})\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(results)

    @tool
    def evaluate_answer_confidence(question: str, search_results: str) -> str:
        """답변 신뢰도 평가
        
        Returns:
            HIGH, MEDIUM, LOW와 이유
        """
        # 간단한 휴리스틱 기반 평가
        result_count = search_results.count("[문서")
        has_high_reliability = "신뢰도: high" in search_results
        
        if result_count >= 2 and has_high_reliability:
            confidence = "HIGH"
            reason = "여러 고신뢰도 문서에서 일관된 정보 발견"
        elif result_count >= 1:
            confidence = "MEDIUM"
            reason = "관련 문서는 있으나 추가 확인 권장"
        else:
            confidence = "LOW"
            reason = "충분한 정보를 찾지 못함"
        
        return f"신뢰도: {confidence}\n이유: {reason}"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_knowledge_base, evaluate_answer_confidence],
        system_prompt="""당신은 회사의 통합 지식 검색 Agent입니다.

질문에 답변할 때:
1. search_knowledge_base로 관련 정보 검색
2. 검색 결과를 분석하여 답변 작성
3. evaluate_answer_confidence로 답변 신뢰도 평가
4. 답변과 함께 출처와 신뢰도를 명시

답변 형식:
[답변 내용]

📚 출처: [문서 출처]
🎯 신뢰도: [HIGH/MEDIUM/LOW]"""
    )

    # 다양한 질문 테스트
    test_questions = [
        "RAG가 무엇인가요?",
        "Agent는 어떻게 작동하나요?",
        "연차는 몇 일인가요?",
        "LangChain의 주요 개념은?",
    ]

    print("\n" + "=" * 70)
    print("🧪 지식 기반 Q&A 테스트")
    print("=" * 70)

    for question in test_questions:
        print(f"\n{'=' * 70}")
        print(f"❓ 질문: {question}")
        print("=" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 답변:\n{answer}\n")

    # 사용자 입력
    print("\n" + "=" * 70)
    print("💬 직접 질문해보세요 (종료: 'quit' 입력)")
    print("=" * 70)

    user_question = input("\n❓ 질문: ").strip()

    if user_question and user_question.lower() != 'quit':
        print("\n🔍 검색 중...\n")

        response = agent.invoke({
            "messages": [{"role": "user", "content": user_question}]
        })

        answer = response['messages'][-1].content
        print(f"🤖 답변:\n{answer}\n")

    print("=" * 70)
    print("✅ 지식 기반 Q&A Agent 완료!")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n")
    print("=" * 70)
    print("Part 8: Agentic RAG (03_agentic_rag.py)")
    print("=" * 70)

    while True:
        print("\n📚 실행할 예제를 선택하세요:")
        print("  1. Retriever를 Tool로 변환")
        print("  2. 여러 Retriever 통합")
        print("  3. Query Planning (쿼리 분해)")
        print("  4. Self-RAG (자기 검증)")
        print("  5. 실전 지식 기반 Q&A Agent ⭐")
        print("  0. 종료")

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_retriever_as_tool()
        elif choice == "2":
            example_2_multiple_retrievers()
        elif choice == "3":
            example_3_query_planning()
        elif choice == "4":
            example_4_self_rag()
        elif choice == "5":
            example_5_knowledge_qa_agent()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
