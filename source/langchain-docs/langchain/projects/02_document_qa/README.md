# Project 2: 문서 Q&A 시스템 (Document Q&A System)

> 난이도: 중급-고급
> 예상 소요 시간: 3-4시간
> 관련 파트: Part 4 (RAG), Part 6 (고급 기법)

---

## 프로젝트 개요

RAG (Retrieval-Augmented Generation) 기술을 활용하여 문서 기반 질의응답 시스템을 구축합니다.

### 학습 목표

- RAG 파이프라인 구축
- Vector Store (FAISS) 활용
- 문서 로딩 및 청킹 전략
- 소스 인용 시스템
- 사용자 권한별 문서 접근 제어

---

## 기능 요구사항

### 1. 문서 로딩 및 인덱싱
- `datasets/sample_documents/` 디렉토리에서 문서 자동 로드
- Markdown 파일 파싱
- 텍스트 청킹 (Chunk size: 1000, Overlap: 200)
- FAISS 벡터 스토어 구축

### 2. 질의응답 시스템
- 자연어 질문 처리
- 관련 문서 검색 (Top-K: 3)
- LLM 기반 답변 생성
- 소스 문서 인용

### 3. 사용자 권한 관리
- 사용자별 문서 접근 권한 설정
- 필터링된 검색 결과 반환
- 권한 없는 문서 접근 차단

### 4. 대화형 인터페이스
- CLI 기반 Q&A 인터페이스
- 질문 히스토리 관리
- 명확한 소스 표시

---

## 기술 스택

- **LangChain**: RAG 파이프라인
- **FAISS**: Vector Store
- **OpenAI Embeddings**: 텍스트 임베딩
- **ChatOpenAI**: 답변 생성
- **Python-dotenv**: 환경 변수 관리

---

## 프로젝트 구조

```
02_document_qa/
├── README.md                # 프로젝트 문서
├── main.py                  # 메인 실행 파일
├── rag_pipeline.py          # RAG 구현
├── document_loader.py       # 문서 로딩 유틸
├── access_control.py        # 권한 관리
├── requirements.txt         # 의존성
├── .env.example            # 환경 변수 예시
├── tests/                  # 테스트
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_loader.py
│   └── test_access.py
└── solution/               # 참고 솔루션
    ├── main_solution.py
    └── rag_pipeline_solution.py
```

---

## 시작하기

### 1. 의존성 설치

```bash
cd /Users/restful3/Desktop/langchain/projects/02_document_qa
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
OPENAI_API_KEY=your-openai-api-key
```

### 3. 문서 인덱싱

```bash
python main.py --index
```

### 4. Q&A 시작

```bash
python main.py
```

---

## 사용 예시

```bash
$ python main.py

문서 Q&A 시스템에 오신 것을 환영합니다!
3개의 문서가 로드되었습니다.

사용자 이름을 입력하세요: admin

질문을 입력하세요 (종료: quit): LangChain이란 무엇인가요?

답변:
LangChain은 2022년 Harrison Chase가 개발한 오픈소스 프레임워크로,
대규모 언어 모델(LLM)을 활용한 애플리케이션을 쉽게 구축할 수 있도록
돕습니다. Chains, Agents, Memory, Tools 등의 핵심 개념을 제공하며,
다양한 LLM 프로바이더를 지원합니다.

출처:
- langchain_overview.md (관련도: 0.92)

질문을 입력하세요 (종료: quit): Python의 주요 특징은?

답변:
Python의 주요 특징은 다음과 같습니다:
1. 간결한 문법: 읽기 쉽고 작성하기 쉬운 코드
2. 동적 타이핑: 타입을 자동으로 결정
3. 객체 지향: 클래스와 상속 지원
4. 풍부한 라이브러리: 표준 라이브러리와 PyPI 생태계

출처:
- python_basics.md (관련도: 0.88)

질문을 입력하세요 (종료: quit): quit

감사합니다. 좋은 하루 되세요!
```

---

## 구현 가이드

### Step 1: 문서 로더 구현

```python
# document_loader.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_documents(self) -> List[Document]:
        """문서 디렉토리에서 모든 .md 파일 로드"""
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
```

### Step 2: RAG 파이프라인 구축

```python
# rag_pipeline.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self, documents: List[Document]):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def create_qa_chain(self):
        """Q&A 체인 생성"""
        template = """다음 문맥을 사용하여 질문에 답변하세요.
        답변을 모르면 모른다고 답하고, 추측하지 마세요.

        문맥: {context}

        질문: {question}

        답변:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
```

### Step 3: 접근 제어 구현

```python
# access_control.py
from typing import Dict, List, Set

class AccessControl:
    def __init__(self):
        # 사용자별 접근 가능한 문서 정의
        self.permissions: Dict[str, Set[str]] = {
            "admin": {"langchain_overview.md", "python_basics.md", "ai_ethics.md"},
            "developer": {"langchain_overview.md", "python_basics.md"},
            "guest": {"python_basics.md"},
        }

    def can_access(self, user: str, document: str) -> bool:
        """사용자가 문서에 접근할 수 있는지 확인"""
        if user not in self.permissions:
            return False
        return document in self.permissions[user]

    def filter_documents(self, user: str, documents: List) -> List:
        """사용자 권한에 따라 문서 필터링"""
        return [
            doc for doc in documents
            if self.can_access(user, doc.metadata.get("source", "").split("/")[-1])
        ]
```

---

## 테스트

### 단위 테스트 실행

```bash
pytest tests/ -v
```

### 테스트 시나리오

1. **문서 로딩 테스트**
   - 모든 문서가 올바르게 로드되는지 확인
   - 청킹이 적절하게 이루어지는지 검증

2. **RAG 파이프라인 테스트**
   - 관련 문서 검색 정확도
   - 답변 품질 평가
   - 소스 인용 정확성

3. **접근 제어 테스트**
   - 권한별 문서 필터링
   - 무단 접근 차단
   - 다중 사용자 시나리오

---

## 성능 최적화

### 1. 벡터 스토어 저장/로드

```python
# 인덱스 저장
vectorstore.save_local("faiss_index")

# 인덱스 로드
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

### 2. 캐싱 전략

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### 3. 배치 처리

```python
# 대량 문서 처리 시 배치로 나누어 임베딩
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
```

---

## 고급 기능 (도전 과제)

### 1. 하이브리드 검색

```python
# BM25 + Vector Search 결합
from langchain.retrievers import BM25Retriever, EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(documents)
faiss_retriever = vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)
```

### 2. Re-ranking

```python
# 검색 결과 재순위화
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

### 3. 대화 기록 통합

```python
# 이전 대화 맥락을 고려한 Q&A
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)
```

---

## 문제 해결

### Q: FAISS 설치 오류
```bash
# Apple Silicon Mac의 경우
pip install faiss-cpu

# Intel Mac 또는 Linux
pip install faiss-cpu
```

### Q: 메모리 부족 에러
- 청크 사이즈 줄이기
- 배치 처리 활용
- 문서 수 제한

### Q: 답변 품질이 낮음
- 청킹 전략 조정
- Top-K 값 증가
- 프롬프트 개선
- 더 강력한 모델 사용 (gpt-4)

---

## 평가 기준

### 기능 완성도 (40점)
- [ ] 문서 로딩 및 인덱싱
- [ ] 정확한 질의응답
- [ ] 소스 인용 기능
- [ ] 접근 제어 구현

### 코드 품질 (30점)
- [ ] 모듈화 및 재사용성
- [ ] 에러 핸들링
- [ ] 타입 힌트 사용
- [ ] 코드 주석

### 테스트 (20점)
- [ ] 단위 테스트 작성
- [ ] 통합 테스트
- [ ] 테스트 커버리지 70% 이상

### 문서화 (10점)
- [ ] README 완성도
- [ ] 코드 주석
- [ ] 사용 예시

---

## 참고 자료

- [LangChain RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/)
- [FAISS 문서](https://github.com/facebookresearch/faiss)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- Part 4: RAG 시스템 구축
- Part 6: 고급 RAG 기법

---

## 다음 단계

프로젝트를 완료한 후:
1. Project 3: Research Agent System으로 진행
2. 고급 RAG 기법 학습 (Part 6)
3. 프로덕션 배포 고려사항 학습

**행운을 빕니다!**
