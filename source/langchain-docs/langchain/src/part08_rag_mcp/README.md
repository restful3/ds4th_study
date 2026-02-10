# Part 8: RAG & MCP - 지식 통합과 외부 연결

> 📚 **학습 시간**: 약 4-5시간
> 🎯 **난이도**: ⭐⭐⭐⭐☆ (고급)
> 📖 **공식 문서**: [28-retrieval.md](/official/28-retrieval.md), [20-model-context-protocol.md](/official/20-model-context-protocol.md)
> 📄 **교안 문서**: [part08_rag_mcp.md](/docs/part08_rag_mcp.md)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] RAG (Retrieval Augmented Generation) 개념 이해
- [x] Vector Store로 지식 베이스 구축
- [x] Agentic RAG 패턴 구현
- [x] MCP (Model Context Protocol) 이해 및 사용
- [x] MCP 서버 구현
- [x] Agent와 MCP 통합

---

## 📚 개요

**RAG**는 Agent에게 외부 지식을 제공하고, **MCP**는 외부 시스템과 연결합니다. 이 두 기술로 Agent의 능력을 극대화합니다.

**왜 중요한가?**
- LLM의 지식은 학습 시점까지만 (outdated)
- 내부 문서, 데이터베이스 활용 필요
- 외부 도구 및 서비스 연동

**실무 활용 사례**
- 문서 기반 Q&A 시스템
- 기업 내부 지식 검색
- 외부 API/도구 통합

---

## 📁 예제 파일

### 01_rag_basics.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 40분

RAG의 기본 개념과 간단한 구현을 학습합니다.

**학습 내용**:
- RAG 워크플로우 (Index → Retrieve → Generate)
- 문서 로딩 및 청킹
- 간단한 검색
- LLM과 통합

**실행 방법**:
```bash
python 01_rag_basics.py
```

**주요 개념**:
- **Retrieval**: 관련 문서 검색
- **Augmentation**: 검색 결과를 프롬프트에 추가
- **Generation**: LLM이 답변 생성

---

### 02_vector_store.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

Vector Store를 사용한 의미 기반 검색을 구현합니다.

**학습 내용**:
- Embedding 개념
- Vector Database (Chroma, FAISS 등)
- 의미 기반 유사도 검색
- 지식 베이스 구축 및 관리

**실행 방법**:
```bash
python 02_vector_store.py
```

**주요 개념**:
- **Embedding**: 텍스트 → 벡터 변환
- **Vector Store**: 벡터 저장 및 검색
- **Similarity Search**: 유사도 기반 검색

---

### 03_agentic_rag.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 70분

Agent가 검색 전략을 결정하는 Agentic RAG를 구현합니다.

**학습 내용**:
- 검색을 도구로 제공
- Agent가 검색 필요 여부 판단
- 다중 검색 쿼리
- 검색 결과 평가 및 재검색

**실행 방법**:
```bash
python 03_agentic_rag.py
```

**주요 개념**:
- **기본 RAG**: 항상 검색
- **Agentic RAG**: Agent가 결정
- 더 유연하고 똑똑한 검색

---

### 04_mcp_client.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 45분

MCP (Model Context Protocol) 클라이언트를 사용합니다.

**학습 내용**:
- MCP 개념 및 목적
- MCP 서버 연결
- MCP 도구 사용
- 표준 프로토콜의 이점

**실행 방법**:
```bash
python 04_mcp_client.py
```

**주요 개념**:
- **MCP**: 외부 도구/데이터 연결 표준
- Agent와 외부 시스템 간 통신
- 재사용 가능한 통합

---

### 05_mcp_server.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

자신만의 MCP 서버를 구현합니다.

**학습 내용**:
- MCP 서버 스펙
- 도구 및 리소스 제공
- 보안 및 권한 관리
- 서버 배포

**실행 방법**:
```bash
python 05_mcp_server.py
```

**주요 개념**:
- 자체 도구를 MCP로 노출
- 다른 Agent들이 재사용 가능
- 표준 프로토콜 준수

---

### 06_mcp_agent.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

Agent와 MCP를 완전히 통합합니다.

**학습 내용**:
- Agent에 MCP 도구 연결
- 여러 MCP 서버 통합
- 동적 MCP 서버 발견
- 엔터프라이즈 통합

**실행 방법**:
```bash
python 06_mcp_agent.py
```

**주요 개념**:
- MCP로 확장 가능한 Agent
- 플러그인 아키텍처
- 기업 시스템 통합

---

## 🎓 실습 과제

### 과제 1: 기술 문서 Q&A (⭐⭐⭐)

**목표**: 기술 문서를 읽고 질문에 답하는 RAG 시스템을 만드세요.

**요구사항**:
1. Markdown 문서 3-5개 준비
2. Vector Store에 저장
3. 질문에 대해 관련 문서 검색
4. LLM이 답변 생성 (출처 포함)

**해답**: [solutions/exercise_01.py](/src/part08_rag_mcp/solutions/exercise_01.py)

---

### 과제 2: 스마트 검색 Agent (⭐⭐⭐⭐)

**목표**: 검색 전략을 스스로 결정하는 Agentic RAG를 구현하세요.

**요구사항**:
1. Agent가 검색 필요 여부 판단
2. 필요 시 여러 검색 쿼리 생성
3. 검색 결과 평가
4. 부족하면 추가 검색

**예시**:
```
User: "LangChain의 메모리 시스템과 스트리밍을 비교해줘"
Agent: [검색1: "LangChain memory"], [검색2: "LangChain streaming"]
Agent: [결과 통합 및 비교 답변]
```

**해답**: [solutions/exercise_02.py](/src/part08_rag_mcp/solutions/exercise_02.py)

---

### 과제 3: MCP 기반 통합 시스템 (⭐⭐⭐⭐⭐)

**목표**: MCP 서버를 구현하고 Agent와 통합하세요.

**요구사항**:
1. 자신만의 MCP 서버 구현 (예: 데이터베이스 접근)
2. Agent에 MCP 클라이언트 통합
3. Agent가 MCP 도구 사용하여 작업 수행

**해답**: [solutions/exercise_03.py](/src/part08_rag_mcp/solutions/exercise_03.py)

---

## 💡 실전 팁

### Tip 1: 문서 청킹 전략

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 청크 크기 선택
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 토큰 한도 고려
    chunk_overlap=200,    # 문맥 유지
    separators=["\n\n", "\n", ". ", " "]  # 자연스러운 구분
)

documents = splitter.split_documents(docs)
```

**권장 크기**:
- 작은 청크 (500자): 정확한 검색, 문맥 부족 가능
- 중간 청크 (1000자): 균형 잡힌 선택 ✅
- 큰 청크 (2000자): 넓은 문맥, 검색 정확도 하락

### Tip 2: Embedding 모델 선택

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI (고품질, 유료)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (무료, 로컬)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Tip 3: 검색 결과 재정렬

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 검색 후 LLM으로 관련성 재평가
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: RAG vs Fine-tuning 차이는?</summary>

**A**:
- **RAG**:
  - 장점: 빠른 업데이트, 출처 제공, 비용 저렴
  - 단점: 검색 의존, 추가 latency

- **Fine-tuning**:
  - 장점: 모델에 지식 내재화
  - 단점: 비용 높음, 업데이트 어려움

**추천**: 대부분의 경우 RAG가 더 실용적!
</details>

<details>
<summary>Q2: Vector Store 선택 가이드</summary>

**A**:
- **개발/테스트**: Chroma (로컬, 간단)
- **소규모 프로덕션**: FAISS (빠름, 로컬)
- **대규모 프로덕션**: Pinecone, Weaviate (관리형, 확장 가능)
- **기업**: PostgreSQL with pgvector (기존 인프라 활용)

```python
# Chroma (개발용)
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(docs, embeddings)

# Pinecone (프로덕션)
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(docs, embeddings)
```
</details>

<details>
<summary>Q3: MCP는 언제 사용하나요?</summary>

**A**: 다음 상황에서 유용:
- 표준 프로토콜이 필요할 때
- 여러 Agent가 같은 도구를 사용
- 외부 시스템과 통합 (CRM, ERP 등)
- 도구를 재사용 가능하게 배포

**간단한 경우**는 일반 도구로 충분합니다.
</details>

---

## 🔗 심화 학습

1. **공식 문서 심화**
   - [28-retrieval.md](/official/28-retrieval.md) - RAG 전체 가이드
   - [20-model-context-protocol.md](/official/20-model-context-protocol.md) - MCP 스펙
   - [LangChain RAG Guide](https://python.langchain.com/docs/tutorials/rag/)

2. **관련 논문**
   - [Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401)
   - [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
   - [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)

3. **커뮤니티 리소스**
   - [RAG Best Practices](https://blog.langchain.dev/rag-best-practices/)
   - [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/)
   - [MCP Specification](https://modelcontextprotocol.io/)

4. **다음 단계**
   - [Part 9: Production](/src/part09_production/README.md) - 프로덕션 준비

---

## ✅ 체크리스트

Part 8을 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다 (6개)
- [ ] 실습 과제를 완료했다 (3개)
- [ ] RAG의 3단계를 이해했다 (Index, Retrieve, Generate)
- [ ] Vector Store를 사용할 수 있다
- [ ] Agentic RAG의 장점을 안다
- [ ] MCP의 목적을 이해했다
- [ ] MCP 서버를 구현할 수 있다

---

**이전**: [← Part 7 - Multi-Agent](/src/part07_multi_agent/README.md)
**다음**: [Part 9 - Production으로 이동](/src/part09_production/README.md) →

---

**학습 진도**: ▓▓▓▓▓▓▓▓░░ 80% (Part 8/10 완료)

*마지막 업데이트: 2025-02-06*
