# LangChain 커리큘럼 프로젝트

LangChain을 활용한 실전 AI Agent 시스템 구축 프로젝트입니다.

---

## 📚 프로젝트 개요

총 4개의 점진적 난이도 프로젝트로 구성되어 있으며, 각 프로젝트는 완전히 실행 가능한 코드와 상세한 문서를 포함합니다.

### 프로젝트 목록

| # | 프로젝트 | 난이도 | 소요시간 | 상태 |
|---|---------|-------|---------|------|
| 1 | [Weather Assistant](./01_weather_assistant/) | ⭐⭐⭐ | 2-3시간 | ✅ 완료 |
| 2 | [Document Q&A System](./02_document_qa/) | ⭐⭐⭐⭐ | 3-4시간 | ✅ 완료 |
| 3 | [Research Agent System](./03_research_agent/) | ⭐⭐⭐⭐⭐ | 4-5시간 | ✅ 완료 |
| 4 | [Customer Service Agent](./04_customer_service/) | ⭐⭐⭐⭐⭐⭐ | 5-6시간 | ✅ 완료 |

---

## 🎯 학습 목표

### Project 1: Weather Assistant
- **핵심 개념**: Agent 기초, Tool 통합, API 연동
- **기술 스택**: LangChain Agents, OpenWeatherMap API
- **학습 내용**:
  - 외부 API를 도구로 통합
  - Agent의 기본 작동 원리
  - 자연어 대화 인터페이스

### Project 2: Document Q&A System
- **핵심 개념**: RAG (Retrieval-Augmented Generation), Vector Store
- **기술 스택**: FAISS, OpenAI Embeddings, LangChain RAG
- **학습 내용**:
  - 문서 로딩 및 청킹 전략
  - 벡터 데이터베이스 구축
  - 시맨틱 검색
  - 사용자 권한 관리

### Project 3: Research Agent System
- **핵심 개념**: Multi-Agent 시스템, 웹 검색, 병렬 처리
- **기술 스택**: Tavily/DuckDuckGo Search, Multi-Agent Coordination
- **학습 내용**:
  - 여러 Agent 간 협업
  - 웹 검색 통합
  - 정보 수집 및 분석 파이프라인
  - 구조화된 보고서 생성

### Project 4: Customer Service Agent
- **핵심 개념**: 프로덕션 시스템, HITL, 모니터링
- **기술 스택**: FastAPI, Docker, HITL, Monitoring
- **학습 내용**:
  - 지능형 라우팅 시스템
  - Human-in-the-Loop 구현
  - 시스템 모니터링 및 로깅
  - API 서버 및 배포

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 각 프로젝트 디렉토리에서
cd projects/02_document_qa  # 예시
pip install -r requirements.txt
```

### 2. API 키 설정

모든 프로젝트는 OpenAI API 키가 필요합니다.

```bash
# 각 프로젝트에서 .env 파일 생성
cp .env.example .env

# .env 파일 편집
OPENAI_API_KEY=your-key-here
```

### 3. 프로젝트 실행

```bash
# 대화형 모드
python main.py

# 도움말 확인
python main.py --help
```

---

## 📖 프로젝트 상세 설명

### Project 1: Weather Assistant (날씨 비서)

실시간 날씨 API를 활용하는 대화형 Agent

**주요 기능**:
- 도시별 현재 날씨 조회
- 자연어 질문 처리
- 친근한 한국어 대화

**실행**:
```bash
cd 01_weather_assistant
python main.py
```

**예시**:
```
You: 서울 날씨 어때?
Agent: 서울은 현재 맑음이고 22°C입니다. 산책하기 좋은 날씨네요!
```

---

### Project 2: Document Q&A System (문서 Q&A)

RAG 기술을 활용한 문서 기반 질의응답 시스템

**주요 기능**:
- 자동 문서 인덱싱
- 시맨틱 검색
- 소스 인용
- 사용자 권한별 접근 제어

**실행**:
```bash
cd 02_document_qa
python main.py --reindex  # 첫 실행 시
python main.py
```

**예시**:
```
사용자: admin
질문: LangChain이란?

답변: LangChain은 LLM을 활용한 애플리케이션을 쉽게 구축할 수 있는
프레임워크입니다. Chains, Agents, Memory 등의 핵심 개념을 제공합니다.

출처:
- langchain_overview.md (관련도: 0.92)
```

**고급 기능**:
- 하이브리드 검색 (BM25 + Vector)
- 역할 기반 접근 제어 (RBAC)
- 대화 컨텍스트 유지

---

### Project 3: Research Agent System (연구 에이전트)

4개의 전문 Agent가 협업하여 자동 리서치 보고서를 생성

**Agent 구조**:
```
Planner → Searcher → Analyst → Writer
(계획)    (검색)     (분석)    (작성)
```

**주요 기능**:
- 주제 분석 및 하위 질문 생성
- 병렬 웹 검색
- 데이터 분석 및 인사이트 추출
- 마크다운 보고서 자동 생성

**실행**:
```bash
cd 03_research_agent
python main.py --query "양자 컴퓨팅의 최신 동향"
```

**생성 보고서**:
- 요약
- 주요 발견 사항 (5-7개)
- 상세 분석 (섹션별)
- 결론
- 참고 문헌

**고급 기능**:
- 비동기 병렬 처리
- Agent 간 통신 프로토콜
- 보고서 품질 평가

---

### Project 4: Customer Service Agent (고객 서비스)

프로덕션 수준의 고객 서비스 AI 시스템

**시스템 구성**:
```
Router Agent (라우팅)
  ↓
├─ Support Agent (기술 지원)
├─ Billing Agent (결제)
└─ General Agent (일반)
  ↓
Knowledge Base (RAG)
  ↓
Escalation Agent (HITL)
  ↓
Monitoring (모니터링)
```

**주요 기능**:
1. **지능형 라우팅**
   - 문의 자동 분류
   - 적절한 Agent 배정
   - 신뢰도 계산

2. **RAG 지식 베이스**
   - FAQ, 정책 문서
   - 카테고리별 필터링
   - 실시간 검색

3. **Human-in-the-Loop**
   - 중요 작업 승인
   - 에스컬레이션 처리
   - 승인 기록 관리

4. **모니터링**
   - 응답 시간 추적
   - 만족도 조사
   - 통계 대시보드

**실행**:
```bash
cd 04_customer_service

# CLI 모드
python main.py

# API 모드
python main.py --api --port 8000

# 상세 로그
python main.py --verbose
```

**API 사용**:
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "결제가 안 돼요",
    "session_id": "user123"
})
```

**배포**:
```bash
# Docker
docker build -t customer-service .
docker run -p 8000:8000 customer-service

# Docker Compose
docker-compose up -d
```

---

## 🧪 테스트

각 프로젝트는 테스트 스위트를 포함합니다.

```bash
# 테스트 실행
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov

# 특정 테스트만
pytest tests/test_rag.py -v
```

---

## 📊 프로젝트 비교표

| 특징 | P1 | P2 | P3 | P4 |
|------|----|----|----|----|
| Agent 수 | 1 | 1 | 4 | 5 |
| RAG | ❌ | ✅ | ❌ | ✅ |
| 웹 검색 | ❌ | ❌ | ✅ | ❌ |
| Multi-Agent | ❌ | ❌ | ✅ | ✅ |
| HITL | ❌ | ❌ | ❌ | ✅ |
| 모니터링 | ❌ | ❌ | ❌ | ✅ |
| API 서버 | ❌ | ❌ | ❌ | ✅ |
| 배포 준비 | ❌ | ❌ | ❌ | ✅ |

---

## 🎓 학습 로드맵

### 초보자 경로
1. **Project 1** - Agent 기초 이해
2. **Project 2** - RAG 개념 학습
3. **Project 3** - Multi-Agent 경험
4. **Project 4** - 통합 프로젝트

### 중급자 경로
1. **Project 2** - RAG 심화
2. **Project 3** - 고급 Agent 패턴
3. **Project 4** - 프로덕션 기술

### 고급자 경로
- 각 프로젝트의 도전 과제
- 프로젝트 통합 및 확장
- 성능 최적화
- 실제 서비스 배포

---

## 💡 학습 팁

### 효과적인 학습 방법

1. **순차적 진행**
   - 난이도순으로 프로젝트 완수
   - 각 프로젝트의 README를 꼼꼼히 읽기
   - 코드를 직접 작성하며 이해

2. **실험과 수정**
   - 프롬프트 변경하며 결과 관찰
   - 파라미터 조정 (temperature, top_k 등)
   - 새로운 기능 추가

3. **문제 해결**
   - 에러 메시지를 주의 깊게 읽기
   - verbose 모드로 디버깅
   - 테스트 코드 작성

4. **커뮤니티 활용**
   - GitHub Issues에 질문
   - 다른 학습자와 코드 리뷰
   - 자신의 개선사항 공유

---

## 🛠 문제 해결

### 일반적인 문제

**Q: API 키 오류**
```bash
❌ OPENAI_API_KEY not found
A: .env 파일에 올바른 API 키를 설정하세요
```

**Q: 패키지 설치 오류**
```bash
❌ ModuleNotFoundError
A: pip install -r requirements.txt 재실행
```

**Q: FAISS 설치 실패**
```bash
❌ Failed to install faiss-cpu
A: Apple Silicon: pip install faiss-cpu
   Intel/AMD: pip install faiss-cpu
```

**Q: 느린 응답 속도**
```bash
A: - GPT-4o-mini 사용 (GPT-4 대신)
   - 캐싱 활성화
   - 청크 크기 조정
```

---

## 📦 의존성

### 공통 의존성
- Python 3.10+
- langchain >= 0.3.18
- langchain-openai >= 0.2.14
- python-dotenv >= 1.0.1

### 프로젝트별 추가 의존성

**Project 2**:
- faiss-cpu
- unstructured

**Project 3**:
- tavily-python (선택)
- duckduckgo-search

**Project 4**:
- fastapi
- uvicorn
- structlog

---

## 🌟 다음 단계

### 프로젝트 완료 후

1. **포트폴리오 구축**
   - GitHub에 업로드
   - README 작성
   - 데모 영상 제작

2. **실전 적용**
   - 실제 문제에 적용
   - 성능 최적화
   - 사용자 피드백 수집

3. **고급 학습**
   - LangGraph 마이그레이션
   - 커스텀 Agent 개발
   - 프로덕션 배포

4. **기여하기**
   - 버그 리포트
   - 개선 제안
   - 새로운 프로젝트 추가

---

## 📚 추가 리소스

### 공식 문서
- [LangChain 문서](https://python.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)

### 커뮤니티
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

### 관련 도구
- [LangSmith](https://smith.langchain.com/) - Tracing & Monitoring
- [Tavily](https://tavily.com/) - AI Search API
- [Pinecone](https://www.pinecone.io/) - Vector Database

---

## 📝 라이선스

이 프로젝트는 교육 목적으로 제공됩니다.

---

## 🤝 기여

버그 리포트, 기능 제안, Pull Request 환영합니다!

---

## ✨ 완료 체크리스트

- [ ] Project 1: Weather Assistant 완료
- [ ] Project 2: Document Q&A 완료
- [ ] Project 3: Research Agent 완료
- [ ] Project 4: Customer Service 완료
- [ ] 모든 테스트 통과
- [ ] 자신만의 기능 추가
- [ ] 포트폴리오에 추가

---

**행운을 빕니다! Happy Coding!** 🚀

*자세한 내용은 각 프로젝트의 README.md를 참조하세요.*
