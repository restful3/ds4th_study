# Project 2 참고 솔루션

이 디렉토리에는 Project 2의 참고 솔루션이 포함되어 있습니다.

## 사용 방법

1. 먼저 직접 구현을 시도하세요
2. 막히는 부분이 있을 때만 참고하세요
3. 코드를 복사하지 말고 이해하려 노력하세요

## 주요 구현 포인트

### 1. 문서 로딩
- `DirectoryLoader`를 사용하여 모든 .md 파일 로드
- `RecursiveCharacterTextSplitter`로 청킹
- 메타데이터 강화

### 2. RAG 파이프라인
- FAISS 벡터 스토어 구축
- OpenAI Embeddings 사용
- RetrievalQA 체인 구성

### 3. 접근 제어
- 사용자별 권한 딕셔너리 관리
- 문서 필터링 로직
- 역할 기반 접근 제어 (RBAC)

## 실행 방법

```bash
# 솔루션 실행
python solution/main_solution.py
```

## 개선 아이디어

1. **하이브리드 검색**: BM25 + Vector Search
2. **Re-ranking**: 검색 결과 재순위화
3. **대화 메모리**: 이전 대화 맥락 고려
4. **캐싱**: 자주 묻는 질문 캐싱
5. **모니터링**: 질의응답 성능 추적
