# LangChain 커리큘럼 프로젝트 요약

이 문서는 완성된 4개 프로젝트의 개요를 제공합니다.

## 📁 프로젝트 구조

```
langchain/projects/
├── 01_weather_assistant/      ✅ (기존 완료)
├── 02_document_qa/            ✅ (새로 생성)
├── 03_research_agent/         ✅ (새로 생성)
└── 04_customer_service/       ✅ (새로 생성)
```

---

## Project 1: Weather Assistant (날씨 비서)

**상태**: 기존 완료
**난이도**: 중급
**주요 학습**: Agent 기초, Tool 통합, API 연동

### 특징
- OpenWeatherMap API 통합
- 자연어 날씨 조회
- 한국어 친화적 대화

---

## Project 2: Document Q&A System (문서 Q&A 시스템)

**상태**: ✅ 완료
**난이도**: 중급-고급
**주요 학습**: RAG, Vector Store, 접근 제어

### 핵심 파일
```
02_document_qa/
├── README.md                  # 완전한 프로젝트 가이드
├── main.py                    # CLI 인터페이스
├── rag_pipeline.py            # RAG 구현
├── document_loader.py         # 문서 로딩
├── access_control.py          # 권한 관리
├── requirements.txt           # 의존성
├── tests/                     # 테스트 스위트
│   ├── test_rag.py
│   ├── test_loader.py
│   └── test_access.py
└── solution/                  # 참고 솔루션
```

### 주요 기능
1. **문서 로딩 및 인덱싱**
   - Markdown 파일 자동 로드
   - RecursiveCharacterTextSplitter
   - FAISS 벡터 스토어

2. **RAG 파이프라인**
   - OpenAI Embeddings
   - 유사도 검색
   - 소스 인용

3. **접근 제어**
   - 사용자별 권한 관리
   - 역할 기반 접근 제어 (RBAC)
   - 문서 필터링

4. **고급 기능** (도전 과제)
   - 하이브리드 검색 (BM25 + Vector)
   - Re-ranking
   - 대화 기록 통합

### 실행 방법
```bash
cd projects/02_document_qa
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 설정

# 대화형 모드
python main.py

# 단일 쿼리
python main.py --query "LangChain이란?"
```

---

## Project 3: Research Agent System (연구 에이전트 시스템)

**상태**: ✅ 완료
**난이도**: 고급
**주요 학습**: Multi-Agent 시스템, 병렬 처리, 웹 검색

### 핵심 파일
```
03_research_agent/
├── README.md                    # 완전한 프로젝트 가이드
├── main.py                      # CLI 인터페이스
├── multi_agent_system.py        # 시스템 오케스트레이션
├── agents/
│   ├── base.py                 # BaseAgent
│   ├── planner.py              # Planner Agent
│   ├── searcher.py             # Searcher Agent
│   ├── analyst.py              # Analyst Agent
│   └── writer.py               # Writer Agent
├── requirements.txt
└── solution/
```

### 4단계 파이프라인

1. **Planner Agent** (계획 수립)
   - 주제 분석
   - 하위 질문 생성 (3-5개)
   - 검색 키워드 추출

2. **Searcher Agent** (정보 수집)
   - Tavily/DuckDuckGo 검색
   - 병렬 정보 수집
   - 결과 필터링

3. **Analyst Agent** (데이터 분석)
   - 핵심 인사이트 추출
   - 정보 요약
   - 신뢰도 평가

4. **Writer Agent** (보고서 작성)
   - 구조화된 마크다운 보고서
   - 요약, 주요 발견사항, 상세 분석, 결론
   - 참고 문헌 자동 생성

### 실행 방법
```bash
cd projects/03_research_agent
pip install -r requirements.txt
cp .env.example .env
# .env에 OPENAI_API_KEY, TAVILY_API_KEY 설정

# 대화형 모드
python main.py

# 단일 쿼리
python main.py --query "양자 컴퓨팅의 최신 동향"

# 상세 모드
python main.py --query "AI 윤리" --verbose
```

### 고급 기능
- 비동기 병렬 처리 (AsyncResearchAgentSystem)
- Agent 간 메시지 버스
- 상태 관리
- 보고서 자동 저장

---

## Project 4: Customer Service Agent (고객 서비스 에이전트)

**상태**: ✅ 완료
**난이도**: 고급-전문가
**주요 학습**: 프로덕션 시스템, HITL, 모니터링

### 핵심 파일
```
04_customer_service/
├── README.md                      # 완전한 프로젝트 가이드
├── main.py                        # 메인 시스템
├── config.py                      # 설정 관리
├── agents/
│   ├── base.py                   # BaseAgent
│   ├── router.py                 # Router Agent
│   ├── support_agent.py          # Support Agent
│   ├── billing_agent.py          # Billing Agent
│   ├── general_agent.py          # General Agent
│   └── escalation_agent.py       # Escalation Agent
├── knowledge/
│   ├── rag_system.py             # RAG 시스템
│   └── data/
│       ├── faq.md                # FAQ
│       └── policies.md           # 정책
├── middleware/
│   ├── hitl.py                   # Human-in-the-Loop
│   └── monitoring.py             # 모니터링
├── requirements.txt
└── deployment/                    # Docker, K8s 설정
```

### 시스템 아키텍처

```
고객 문의 → Router Agent → [Support/Billing/General] Agent
                               ↓
                         Knowledge Base (RAG)
                               ↓
                         Escalation Agent (HITL)
                               ↓
                         Monitoring & Logging
```

### 핵심 컴포넌트

1. **Router Agent**
   - 키워드 + LLM 기반 분류
   - 3개 카테고리 (support, billing, general)
   - 신뢰도 계산

2. **전문 Agent**
   - Support: 기술 지원
   - Billing: 결제 관련
   - General: 일반 문의
   - 각 Agent는 RAG 통합

3. **RAG 지식 베이스**
   - FAQ, 정책, 가이드
   - 카테고리별 필터링
   - FAISS 벡터 검색

4. **Human-in-the-Loop**
   - 중요 작업 승인
   - 신뢰도 임계값 설정
   - 승인 기록 관리

5. **모니터링**
   - 응답 시간 추적
   - 만족도 조사
   - 에러 로깅
   - 통계 대시보드

### 실행 방법

```bash
cd projects/04_customer_service
pip install -r requirements.txt
cp .env.example .env
# .env에 OPENAI_API_KEY 설정

# CLI 모드
python main.py

# 상세 모드
python main.py --verbose

# API 모드
python main.py --api --port 8000
```

### API 사용 예시
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "결제가 안 돼요",
    "session_id": "user123"
})

print(response.json())
```

### 배포
```bash
# Docker
docker build -t customer-service-agent .
docker run -p 8000:8000 customer-service-agent

# Docker Compose
docker-compose up -d
```

---

## 🎓 학습 경로

### 초급 → 중급
**Project 1** (Weather Assistant)
- Agent 기초
- Tool 통합
- API 연동

### 중급 → 고급
**Project 2** (Document Q&A)
- RAG 파이프라인
- Vector Store
- 접근 제어

### 고급
**Project 3** (Research Agent)
- Multi-Agent 협업
- 웹 검색 통합
- 병렬 처리

### 고급 → 전문가
**Project 4** (Customer Service)
- 프로덕션 시스템
- HITL 구현
- 모니터링 & 배포

---

## 📊 프로젝트 비교

| 프로젝트 | 난이도 | 소요시간 | Agent 수 | 주요 기술 |
|---------|-------|---------|---------|----------|
| Project 1 | ⭐⭐⭐ | 2-3h | 1 | Tools, API |
| Project 2 | ⭐⭐⭐⭐ | 3-4h | 1 | RAG, Vector DB |
| Project 3 | ⭐⭐⭐⭐⭐ | 4-5h | 4 | Multi-Agent, Search |
| Project 4 | ⭐⭐⭐⭐⭐⭐ | 5-6h | 5 | HITL, Monitoring |

---

## 🚀 다음 단계

### 각 프로젝트 완료 후
1. ✅ README 읽기
2. ✅ 코드 실행 및 테스트
3. ✅ 도전 과제 시도
4. ✅ 자신만의 기능 추가

### 전체 완료 후
1. 프로젝트 통합 (예: Weather + Research)
2. 프로덕션 배포
3. 성능 최적화
4. 사용자 피드백 수집

---

## 💡 팁

### 프로젝트 시작 전
- 각 프로젝트의 README를 먼저 읽으세요
- 의존성을 먼저 설치하세요
- .env 파일을 올바르게 설정하세요

### 막힐 때
- solution/ 디렉토리의 참고 솔루션 확인
- tests/ 디렉토리의 테스트 실행
- verbose 모드로 디버깅

### 학습 극대화
- 코드를 직접 작성해보세요
- 각 컴포넌트를 개별적으로 테스트하세요
- 프롬프트를 수정하며 실험하세요

---

## 📚 추가 리소스

- [LangChain 공식 문서](https://python.langchain.com/)
- [OpenAI API 문서](https://platform.openai.com/docs)
- [FAISS 문서](https://github.com/facebookresearch/faiss)
- [FastAPI 문서](https://fastapi.tiangolo.com/)

---

**축하합니다! 4개의 프로덕션 수준 프로젝트가 준비되었습니다!** 🎉
