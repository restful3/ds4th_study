"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP
================================================================================

파일명: 02_vector_store.py
난이도: ⭐⭐⭐⭐ (중상급)
예상 시간: 35분

📚 학습 목표:
  - Chroma vector store 사용법 마스터
  - 다양한 Document Loader 활용
  - Text chunking 전략 이해 및 적용
  - Embedding 생성 및 저장
  - 실전 문서 라이브러리 구축

📖 공식 문서:
  • Vector Stores: https://python.langchain.com/docs/concepts/vectorstores/
  • Document Loaders: https://python.langchain.com/docs/concepts/document_loaders/

📄 교안 문서:
  • Part 8: /docs/part08_rag_mcp.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langchain-community chromadb python-dotenv

🚀 실행 방법:
  python 02_vector_store.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: Chroma Vector Store 기본 사용법
# ============================================================================

def example_1_chroma_basics():
    """Chroma vector store 기본 사용법"""
    print("=" * 70)
    print("📌 예제 1: Chroma Vector Store 기본 사용법")
    print("=" * 70)

    print("""
💡 Chroma란?
   - 오픈소스 임베딩 데이터베이스
   - 로컬/서버 모드 지원
   - 메타데이터 필터링 강력
   - 사용이 간단하고 직관적

장점:
   • 로컬 실행 가능 (프라이버시)
   • 메타데이터 기반 필터링
   • 자동 영구 저장
   • 빠른 검색 성능
    """)

    # 샘플 텍스트
    texts = [
        "파이썬은 프로그래밍 언어입니다.",
        "자바스크립트는 웹 개발에 사용됩니다.",
        "SQL은 데이터베이스 쿼리 언어입니다.",
        "Docker는 컨테이너 플랫폼입니다.",
        "Kubernetes는 컨테이너 오케스트레이션 도구입니다.",
    ]

    print("\n📚 저장할 텍스트:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    # Embeddings 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Chroma 생성 (임시 디렉토리)
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n🔧 Chroma 초기화 중...")

        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory=temp_dir,
            collection_name="tech_docs"
        )

        print("✅ Vector Store 생성 완료!")

        # 기본 검색
        query = "프로그래밍"
        print(f"\n🔍 검색 쿼리: '{query}'")

        results = vectorstore.similarity_search(query, k=3)

        print("\n📄 검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")

        # 컬렉션 정보
        print("\n📊 컬렉션 통계:")
        collection = vectorstore._collection
        print(f"  • 총 문서 수: {collection.count()}")
        print(f"  • 컬렉션 이름: {collection.name}")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 2: 문서 로딩 - TextLoader와 CSVLoader
# ============================================================================

def example_2_document_loaders():
    """다양한 Document Loader 사용법"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Document Loaders")
    print("=" * 70)

    print("""
💡 Document Loader란?
   - 다양한 형식의 문서를 로드
   - 자동으로 Document 객체 생성
   - 메타데이터 포함

주요 Loader:
   • TextLoader: 텍스트 파일
   • CSVLoader: CSV 파일
   • PDFLoader: PDF 파일
   • WebBaseLoader: 웹 페이지
    """)

    # 임시 파일 생성 및 로드
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. 텍스트 파일 생성
        txt_path = os.path.join(temp_dir, "sample.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("""LangChain은 LLM 애플리케이션 개발 프레임워크입니다.
Agent는 도구를 사용하여 작업을 수행할 수 있습니다.
RAG는 외부 지식을 검색하여 답변 품질을 향상시킵니다.
Vector Store는 문서를 벡터로 저장하고 검색합니다.""")

        print("\n1️⃣ TextLoader 사용")
        print("-" * 70)

        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(txt_path, encoding='utf-8')
        docs = loader.load()

        print(f"✅ 로드된 문서 수: {len(docs)}")
        print(f"\n📄 문서 내용:\n{docs[0].page_content}")
        print(f"\n🏷️  메타데이터: {docs[0].metadata}")

        # 2. CSV 파일 생성 및 로드
        csv_path = os.path.join(temp_dir, "products.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("""product_id,name,category,price,description
1,노트북,전자제품,1200000,고성능 프로그래밍 노트북
2,마우스,전자제품,35000,무선 마우스
3,키보드,전자제품,89000,기계식 키보드
4,모니터,전자제품,350000,27인치 4K 모니터
5,책상,가구,150000,높이 조절 책상""")

        print("\n2️⃣ CSVLoader 사용")
        print("-" * 70)

        from langchain_community.document_loaders import CSVLoader

        loader = CSVLoader(
            file_path=csv_path,
            encoding='utf-8',
            csv_args={'delimiter': ','}
        )
        docs = loader.load()

        print(f"✅ 로드된 문서 수: {len(docs)}")
        print("\n📄 샘플 문서 (처음 2개):")
        for i, doc in enumerate(docs[:2], 1):
            print(f"\n  문서 {i}:")
            print(f"  내용: {doc.page_content[:100]}...")
            print(f"  메타데이터: {doc.metadata}")

        # 3. Vector Store에 저장
        print("\n3️⃣ Vector Store에 저장")
        print("-" * 70)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name="products"
        )

        print("✅ Vector Store 생성 완료!")

        # 검색 테스트
        query = "프로그래밍에 필요한 제품"
        print(f"\n🔍 검색: '{query}'")

        results = vectorstore.similarity_search(query, k=2)

        print("\n📄 검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"\n  {i}. {doc.page_content[:80]}...")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 3: Text Chunking 전략
# ============================================================================

def example_3_text_chunking():
    """다양한 Text Splitting 전략"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Text Chunking 전략")
    print("=" * 70)

    print("""
💡 Text Chunking이란?
   - 긴 문서를 작은 조각으로 분할
   - 검색 품질과 컨텍스트 관리에 중요
   - 적절한 크기와 겹침이 핵심

전략:
   • RecursiveCharacterTextSplitter: 재귀적 분할 (권장)
   • CharacterTextSplitter: 단순 문자 기반 분할
   • 적절한 chunk_size와 overlap 설정
    """)

    # 긴 문서 샘플
    long_text = """
머신러닝은 인공지능의 한 분야로, 컴퓨터가 명시적으로 프로그래밍되지 않고도 학습할 수 있게 하는 기술입니다.

지도 학습은 레이블이 있는 데이터로 모델을 훈련시키는 방법입니다. 분류와 회귀가 대표적인 예입니다. 분류는 데이터를 범주로 나누고, 회귀는 연속적인 값을 예측합니다.

비지도 학습은 레이블이 없는 데이터에서 패턴을 찾는 방법입니다. 클러스터링과 차원 축소가 주요 기법입니다. 클러스터링은 유사한 데이터를 그룹화하고, 차원 축소는 데이터의 특징을 압축합니다.

강화 학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방법을 학습합니다. 게임 AI와 로봇 제어에 많이 사용됩니다.

딥러닝은 인공신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 기술입니다. 이미지 인식, 자연어 처리, 음성 인식 등에서 뛰어난 성능을 보입니다.
    """.strip()

    print(f"\n📄 원본 문서 길이: {len(long_text)} 문자")

    # 1. RecursiveCharacterTextSplitter
    print("\n1️⃣ RecursiveCharacterTextSplitter")
    print("-" * 70)

    splitter1 = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks1 = splitter1.split_text(long_text)

    print(f"✅ 생성된 청크 수: {len(chunks1)}")
    print(f"설정: chunk_size=200, overlap=50")

    print("\n📦 청크 미리보기:")
    for i, chunk in enumerate(chunks1[:3], 1):
        print(f"\n  청크 {i} (길이: {len(chunk)}):")
        print(f"  {chunk[:100]}...")

    # 2. CharacterTextSplitter
    print("\n2️⃣ CharacterTextSplitter")
    print("-" * 70)

    splitter2 = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="\n\n"
    )

    chunks2 = splitter2.split_text(long_text)

    print(f"✅ 생성된 청크 수: {len(chunks2)}")
    print(f"설정: chunk_size=200, overlap=50, separator='\\n\\n'")

    # 3. 다양한 크기 비교
    print("\n3️⃣ 청크 크기별 비교")
    print("-" * 70)

    sizes = [100, 300, 500]

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=50
        )
        chunks = splitter.split_text(long_text)
        avg_length = sum(len(c) for c in chunks) / len(chunks)

        print(f"\n  chunk_size={size}: {len(chunks)}개 청크, 평균 길이={avg_length:.0f}")

    # 4. Document로 분할
    print("\n4️⃣ Document 객체로 분할")
    print("-" * 70)

    doc = Document(
        page_content=long_text,
        metadata={"source": "ml_guide.txt", "chapter": 1}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents([doc])

    print(f"✅ 분할된 문서 수: {len(split_docs)}")
    print(f"\n📄 첫 번째 분할 문서:")
    print(f"  내용: {split_docs[0].page_content[:100]}...")
    print(f"  메타데이터: {split_docs[0].metadata}")

    print("\n💡 청킹 가이드:")
    print("  • FAQ, 짧은 답변: 200-500")
    print("  • 일반 문서: 1000-1500")
    print("  • 긴 기술 문서: 1500-2000")
    print("  • overlap은 chunk_size의 10-20% 권장")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 4: Embedding 생성 및 저장
# ============================================================================

def example_4_embeddings():
    """Embedding 생성 및 Vector Store 저장"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Embedding 생성 및 저장")
    print("=" * 70)

    print("""
💡 Embedding이란?
   - 텍스트를 고차원 벡터로 변환
   - 의미적 유사도 계산 가능
   - Vector Store의 핵심 구성 요소

모델 선택:
   • text-embedding-3-small: 빠르고 저렴 (권장)
   • text-embedding-3-large: 최고 품질
   • text-embedding-ada-002: 구버전
    """)

    # 문서 준비
    documents = [
        Document(
            page_content="Python은 데이터 과학과 웹 개발에 널리 사용되는 프로그래밍 언어입니다.",
            metadata={"category": "programming", "language": "python"}
        ),
        Document(
            page_content="JavaScript는 프론트엔드와 백엔드 웹 개발을 위한 언어입니다.",
            metadata={"category": "programming", "language": "javascript"}
        ),
        Document(
            page_content="TensorFlow는 구글이 개발한 머신러닝 프레임워크입니다.",
            metadata={"category": "ml", "framework": "tensorflow"}
        ),
        Document(
            page_content="PyTorch는 페이스북이 개발한 딥러닝 프레임워크입니다.",
            metadata={"category": "ml", "framework": "pytorch"}
        ),
        Document(
            page_content="Docker는 애플리케이션을 컨테이너로 패키징하는 플랫폼입니다.",
            metadata={"category": "devops", "tool": "docker"}
        ),
    ]

    print(f"\n📚 문서 수: {len(documents)}")

    # 1. Embeddings 모델 초기화
    print("\n1️⃣ Embeddings 모델 초기화")
    print("-" * 70)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1000  # 배치 크기
    )

    print("✅ OpenAI Embeddings 초기화 완료")
    print("  • 모델: text-embedding-3-small")
    print("  • 차원: 1536")

    # 2. Vector Store 생성 및 저장
    print("\n2️⃣ Vector Store 생성")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        persist_dir = os.path.join(temp_dir, "chroma_db")

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name="tech_kb"
        )

        print(f"✅ Vector Store 생성 완료!")
        print(f"  • 컬렉션: tech_kb")
        print(f"  • 문서 수: {vectorstore._collection.count()}")

        # 3. 검색 테스트
        print("\n3️⃣ 유사도 검색 테스트")
        print("-" * 70)

        queries = [
            "프로그래밍 언어 추천",
            "머신러닝 프레임워크",
            "컨테이너 기술"
        ]

        for query in queries:
            print(f"\n🔍 쿼리: '{query}'")
            results = vectorstore.similarity_search(query, k=2)

            print("  결과:")
            for i, doc in enumerate(results, 1):
                print(f"    {i}. {doc.page_content[:60]}...")
                print(f"       메타데이터: {doc.metadata}")

        # 4. 메타데이터 필터링
        print("\n4️⃣ 메타데이터 필터링 검색")
        print("-" * 70)

        print("\n🔍 카테고리가 'ml'인 문서만 검색")
        results = vectorstore.similarity_search(
            "추천해주세요",
            k=5,
            filter={"category": "ml"}
        )

        print(f"  결과 수: {len(results)}")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content[:50]}...")

        # 5. Vector Store 로드 (재사용)
        print("\n5️⃣ 저장된 Vector Store 로드")
        print("-" * 70)

        # 새로운 인스턴스로 로드
        loaded_vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="tech_kb"
        )

        print("✅ Vector Store 로드 완료!")
        print(f"  • 문서 수: {loaded_vectorstore._collection.count()}")

        # 로드된 store로 검색
        results = loaded_vectorstore.similarity_search("Python", k=1)
        print(f"\n  검색 테스트: {results[0].page_content[:60]}...")

    print("\n" + "=" * 70)


# ============================================================================
# 예제 5: 실전 - 문서 라이브러리 구축 및 검색
# ============================================================================

def example_5_document_library():
    """실전 문서 라이브러리 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 문서 라이브러리 구축")
    print("=" * 70)

    print("""
💡 실전 시나리오:
   - 기술 문서 라이브러리 구축
   - 문서 추가, 검색, 관리
   - 메타데이터 기반 필터링
   - 카테고리별 분류

기능:
   1. 여러 문서 로드 및 청킹
   2. Vector Store 구축
   3. 고급 검색 (필터링, 점수)
   4. 통계 및 관리
    """)

    # 기술 문서 데이터
    tech_articles = [
        {
            "title": "Python 시작하기",
            "content": """Python은 1991년 귀도 반 로섬이 만든 고급 프로그래밍 언어입니다.
간결하고 읽기 쉬운 문법으로 초보자에게 적합합니다. 웹 개발, 데이터 과학,
머신러닝, 자동화 등 다양한 분야에서 사용됩니다. pip를 통한 패키지 관리가
편리하며, 풍부한 라이브러리 생태계를 갖추고 있습니다.""",
            "category": "programming",
            "level": "beginner",
            "tags": ["python", "tutorial"]
        },
        {
            "title": "Docker 컨테이너 가이드",
            "content": """Docker는 애플리케이션을 컨테이너로 패키징하여 어디서든 동일하게
실행할 수 있게 하는 플랫폼입니다. 가상머신보다 가볍고 빠르며, 마이크로서비스
아키텍처에 적합합니다. Dockerfile로 이미지를 정의하고, docker-compose로
여러 컨테이너를 관리할 수 있습니다. DevOps 워크플로우에 필수적인 도구입니다.""",
            "category": "devops",
            "level": "intermediate",
            "tags": ["docker", "container"]
        },
        {
            "title": "React 컴포넌트 설계",
            "content": """React는 Facebook이 개발한 선언적 UI 라이브러리입니다.
컴포넌트 기반 아키텍처로 재사용 가능한 UI를 만들 수 있습니다. Hooks를 통해
함수형 컴포넌트에서 상태 관리가 가능하며, Virtual DOM으로 효율적인 렌더링을
제공합니다. useState, useEffect 등의 Hook을 이해하는 것이 중요합니다.""",
            "category": "frontend",
            "level": "intermediate",
            "tags": ["react", "javascript"]
        },
        {
            "title": "머신러닝 기초",
            "content": """머신러닝은 데이터로부터 패턴을 학습하는 인공지능 기술입니다.
지도학습, 비지도학습, 강화학습으로 분류됩니다. scikit-learn은 전통적인 ML
알고리즘을 제공하며, 분류, 회귀, 클러스터링 등의 작업을 수행할 수 있습니다.
데이터 전처리와 특징 엔지니어링이 성능에 큰 영향을 미칩니다.""",
            "category": "ai",
            "level": "advanced",
            "tags": ["ml", "ai"]
        },
        {
            "title": "SQL 쿼리 최적화",
            "content": """SQL은 관계형 데이터베이스를 조작하는 표준 언어입니다.
쿼리 최적화는 인덱스 사용, JOIN 순서, WHERE 조건 최적화로 이루어집니다.
EXPLAIN을 통해 실행 계획을 분석하고, 느린 쿼리를 개선할 수 있습니다.
정규화와 역정규화의 균형을 맞추는 것이 중요합니다.""",
            "category": "database",
            "level": "advanced",
            "tags": ["sql", "database"]
        },
        {
            "title": "Git 워크플로우",
            "content": """Git은 분산 버전 관리 시스템으로 코드 변경 이력을 추적합니다.
브랜치를 통해 독립적인 작업이 가능하며, merge와 rebase로 코드를 통합합니다.
Git Flow, GitHub Flow 등의 워크플로우 모델이 있으며, 팀 협업에 필수적입니다.
commit 메시지를 명확하게 작성하는 것이 중요합니다.""",
            "category": "devops",
            "level": "beginner",
            "tags": ["git", "version-control"]
        },
    ]

    print(f"\n📚 총 {len(tech_articles)}개 문서 준비 완료")

    # 1. 문서를 Document 객체로 변환 및 청킹
    print("\n1️⃣ 문서 처리 및 청킹")
    print("-" * 70)

    all_documents = []

    for article in tech_articles:
        doc = Document(
            page_content=f"제목: {article['title']}\n\n{article['content']}",
            metadata={
                "title": article["title"],
                "category": article["category"],
                "level": article["level"],
                "tags": ",".join(article["tags"])
            }
        )
        all_documents.append(doc)

    # 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )

    split_docs = text_splitter.split_documents(all_documents)

    print(f"✅ 원본 문서: {len(all_documents)}개")
    print(f"✅ 청킹 후: {len(split_docs)}개")
    print(f"  • chunk_size: 300")
    print(f"  • overlap: 50")

    # 2. Vector Store 구축
    print("\n2️⃣ Vector Store 구축")
    print("-" * 70)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    with tempfile.TemporaryDirectory() as temp_dir:
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=temp_dir,
            collection_name="tech_library"
        )

        print("✅ 문서 라이브러리 구축 완료!")
        print(f"  • 총 청크 수: {vectorstore._collection.count()}")

        # 3. 기본 검색
        print("\n3️⃣ 기본 검색")
        print("-" * 70)

        query = "초보자를 위한 프로그래밍"
        print(f"\n🔍 쿼리: '{query}'")

        results = vectorstore.similarity_search(query, k=3)

        print("\n📄 검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"\n  {i}. 제목: {doc.metadata.get('title', 'Unknown')}")
            print(f"     카테고리: {doc.metadata.get('category', 'Unknown')}")
            print(f"     레벨: {doc.metadata.get('level', 'Unknown')}")
            print(f"     내용: {doc.page_content[:100]}...")

        # 4. 카테고리별 검색
        print("\n4️⃣ 카테고리별 필터링 검색")
        print("-" * 70)

        categories = ["programming", "devops", "ai"]

        for category in categories:
            results = vectorstore.similarity_search(
                "추천",
                k=2,
                filter={"category": category}
            )

            print(f"\n📂 카테고리: {category}")
            print(f"  결과 수: {len(results)}")
            if results:
                print(f"  • {results[0].metadata.get('title', 'Unknown')}")

        # 5. 레벨별 검색
        print("\n5️⃣ 난이도별 검색")
        print("-" * 70)

        levels = ["beginner", "intermediate", "advanced"]

        for level in levels:
            results = vectorstore.similarity_search(
                "학습",
                k=1,
                filter={"level": level}
            )

            print(f"\n🎯 레벨: {level}")
            if results:
                print(f"  • {results[0].metadata.get('title', 'Unknown')}")
                print(f"    {results[0].page_content[:80]}...")

        # 6. 유사도 점수와 함께 검색
        print("\n6️⃣ 유사도 점수 기반 검색")
        print("-" * 70)

        query = "컨테이너와 가상화"
        print(f"\n🔍 쿼리: '{query}'")

        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print("\n📊 점수별 결과:")
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n  {i}. 점수: {score:.4f}")
            print(f"     제목: {doc.metadata.get('title', 'Unknown')}")
            print(f"     내용: {doc.page_content[:80]}...")

        # 7. 통계 정보
        print("\n7️⃣ 라이브러리 통계")
        print("-" * 70)

        print(f"\n📊 전체 통계:")
        print(f"  • 총 문서 수: {len(all_documents)}")
        print(f"  • 총 청크 수: {vectorstore._collection.count()}")

        # 카테고리별 통계
        category_count = {}
        for doc in all_documents:
            cat = doc.metadata["category"]
            category_count[cat] = category_count.get(cat, 0) + 1

        print(f"\n  카테고리별 문서 수:")
        for cat, count in sorted(category_count.items()):
            print(f"    • {cat}: {count}개")

        # 레벨별 통계
        level_count = {}
        for doc in all_documents:
            level = doc.metadata["level"]
            level_count[level] = level_count.get(level, 0) + 1

        print(f"\n  난이도별 문서 수:")
        for level, count in sorted(level_count.items()):
            print(f"    • {level}: {count}개")

        # 8. 사용자 검색
        print("\n8️⃣ 직접 검색해보기")
        print("-" * 70)

        user_query = input("\n🔍 검색어를 입력하세요 (Enter로 건너뛰기): ").strip()

        if user_query:
            print(f"\n검색 중: '{user_query}'...")

            results = vectorstore.similarity_search_with_score(user_query, k=3)

            print(f"\n📄 검색 결과 ({len(results)}개):")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  {i}. {doc.metadata.get('title', 'Unknown')}")
                print(f"     유사도: {score:.4f}")
                print(f"     카테고리: {doc.metadata.get('category', 'Unknown')}")
                print(f"     레벨: {doc.metadata.get('level', 'Unknown')}")
                print(f"     내용: {doc.page_content[:120]}...")
        else:
            print("검색을 건너뜁니다.")

    print("\n" + "=" * 70)
    print("✅ 문서 라이브러리 시스템 완료!")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n")
    print("=" * 70)
    print("Part 8: Vector Store 구축 (02_vector_store.py)")
    print("=" * 70)

    while True:
        print("\n📚 실행할 예제를 선택하세요:")
        print("  1. Chroma Vector Store 기본")
        print("  2. Document Loaders (Text, CSV)")
        print("  3. Text Chunking 전략")
        print("  4. Embedding 생성 및 저장")
        print("  5. 실전 문서 라이브러리 구축 ⭐")
        print("  0. 종료")

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_chroma_basics()
        elif choice == "2":
            example_2_document_loaders()
        elif choice == "3":
            example_3_text_chunking()
        elif choice == "4":
            example_4_embeddings()
        elif choice == "5":
            example_5_document_library()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
