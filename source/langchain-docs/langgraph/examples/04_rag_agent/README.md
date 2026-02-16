# RAG Agent (Retrieval-Augmented Generation Agent)

문서 검색을 통해 정보를 보강하는 RAG Agent 예제입니다.

## 기능

- 문서 검색 (시뮬레이션된 벡터 스토어)
- 검색 결과 기반 답변 생성
- 소스 인용
- Adaptive RAG (검색 필요 여부 판단)

## 아키텍처

```
질문 → [분석] → (검색 필요?) → [검색] → [답변 생성] → 결과
                    ↓ (아니오)
                [답변 생성]
```

## 학습 포인트

1. **Adaptive RAG**: 모든 질문에 검색하지 않고, 필요한 경우에만 검색
2. **조건부 라우팅**: 분석 결과에 따른 동적 경로 결정
3. **컨텍스트 구성**: 검색 결과를 효과적으로 프롬프트에 통합
4. **소스 추적**: 답변의 출처 명시

## 실행 방법

```bash
# 기본 데모 실행
python -m examples.04_rag_agent.main

# 인터랙티브 모드
python -m examples.04_rag_agent.main interactive
```

## 코드 구조

```
04_rag_agent/
├── main.py      # 메인 실행 파일
└── README.md    # 이 파일
```

## 핵심 코드 설명

### 1. State 정의

```python
class RAGState(TypedDict):
    question: str           # 사용자 질문
    retrieved_docs: list    # 검색된 문서
    context: str            # 구성된 컨텍스트
    answer: str             # 생성된 답변
    sources: list           # 인용 출처
    needs_retrieval: bool   # 검색 필요 여부
```

### 2. Adaptive RAG 라우팅

```python
def route_by_retrieval(state: RAGState) -> str:
    if state.get("needs_retrieval"):
        return "retrieve"
    return "generate"

graph.add_conditional_edges(
    "analyze",
    route_by_retrieval,
    {"retrieve": "retrieve", "generate": "generate"}
)
```

### 3. 컨텍스트 구성

```python
def retrieve_documents(state: RAGState) -> RAGState:
    docs = search_documents(question, top_k=3)

    context_parts = []
    for doc in docs:
        context_parts.append(f"[{doc['title']}]\n{doc['content']}")

    context = "\n\n---\n\n".join(context_parts)
    return {"context": context, "sources": sources}
```

## 확장 아이디어

- 실제 벡터 DB 연동 (Chroma, Pinecone, Weaviate)
- 하이브리드 검색 (키워드 + 시맨틱)
- 검색 결과 재순위화 (Reranking)
- 다중 소스 통합 (웹, DB, 파일)
- 답변 품질 자체 평가
