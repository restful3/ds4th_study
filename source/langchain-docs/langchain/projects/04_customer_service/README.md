# Project 4: 고객 서비스 에이전트 (Customer Service Agent)

> 난이도: 고급-전문가
> 예상 소요 시간: 5-6시간
> 관련 파트: Part 5, 6, 7 (모든 고급 주제)

---

## 프로젝트 개요

프로덕션 수준의 고객 서비스 AI 에이전트 시스템을 구축합니다. Multi-Agent 협업, RAG, HITL(Human-in-the-Loop), 모니터링 등 모든 고급 기능을 통합한 종합 프로젝트입니다.

### 학습 목표

- 프로덕션 수준의 Multi-Agent 시스템 설계
- RAG 기반 지식 베이스 통합
- Human-in-the-Loop (HITL) 구현
- 에러 핸들링 및 폴백 전략
- 모니터링 및 로깅
- 배포 준비

---

## 시스템 아키텍처

```
고객 문의
    ↓
┌─────────────────────────────────┐
│   Router Agent                  │ ← 문의 분류 및 라우팅
│   (라우터)                       │
└─────────────────────────────────┘
    ↓         ↓         ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Support │ │ Billing │ │ General │
│ Agent   │ │ Agent   │ │ Agent   │
└─────────┘ └─────────┘ └─────────┘
    ↓             ↓           ↓
┌─────────────────────────────────┐
│   Knowledge Base (RAG)          │
│   FAQ, 문서, 정책               │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Escalation Agent              │ ← 필요시 사람에게 전달
│   (에스컬레이션)                 │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   Monitoring & Logging          │
│   성능 추적, 품질 보증           │
└─────────────────────────────────┘
```

---

## 핵심 기능

### 1. 지능형 라우팅
- 문의 내용 자동 분류
- 적절한 전문 Agent에 할당
- 우선순위 처리

### 2. 전문 Agent
- **Support Agent**: 기술 지원
- **Billing Agent**: 결제 관련
- **General Agent**: 일반 문의

### 3. RAG 지식 베이스
- FAQ 자동 검색
- 정책 문서 참조
- 이전 대화 기록 활용

### 4. Human-in-the-Loop
- 중요 작업 전 사람 승인
- 불확실한 경우 에스컬레이션
- 실시간 개입 가능

### 5. 모니터링
- 응답 시간 추적
- 품질 메트릭
- 에러 로깅
- 사용자 만족도

---

## 프로젝트 구조

```
04_customer_service/
├── README.md
├── main.py                      # 메인 실행 파일
├── config.py                    # 설정 관리
├── agents/                      # Agent 구현
│   ├── __init__.py
│   ├── base.py                 # BaseAgent
│   ├── router.py               # Router Agent
│   ├── support_agent.py        # Support Agent
│   ├── billing_agent.py        # Billing Agent
│   ├── general_agent.py        # General Agent
│   └── escalation_agent.py     # Escalation Agent
├── knowledge/                   # 지식 베이스
│   ├── __init__.py
│   ├── rag_system.py           # RAG 시스템
│   ├── knowledge_loader.py     # 지식 로더
│   └── data/                   # 지식 데이터
│       ├── faq.md
│       ├── policies.md
│       └── troubleshooting.md
├── middleware/                  # 미들웨어
│   ├── __init__.py
│   ├── hitl.py                 # HITL 구현
│   ├── monitoring.py           # 모니터링
│   └── error_handler.py        # 에러 핸들링
├── utils/                       # 유틸리티
│   ├── __init__.py
│   ├── prompts.py              # 프롬프트 템플릿
│   └── validators.py           # 입력 검증
├── tests/                       # 테스트
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_rag.py
│   ├── test_hitl.py
│   └── test_integration.py
├── deployment/                  # 배포 설정
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/
├── requirements.txt
├── .env.example
└── solution/
    └── README_SOLUTION.md
```

---

## 시작하기

### 1. 의존성 설치

```bash
cd /Users/restful3/Desktop/langchain/projects/04_customer_service
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
cp .env.example .env
# .env 파일 편집
```

### 3. 지식 베이스 인덱싱

```bash
python -m knowledge.rag_system --index
```

### 4. 실행

```bash
# CLI 모드
python main.py

# API 모드
python main.py --api --port 8000

# 데모 모드
python main.py --demo
```

---

## 사용 예시

### CLI 모드

```bash
$ python main.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   고객 서비스 AI Agent 시스템
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

안녕하세요! 무엇을 도와드릴까요?

고객: 결제가 안 돼요

[Router] 문의 분석 중...
[Router] 카테고리: billing (신뢰도: 95%)
[Router] ➜ Billing Agent로 전달

[Billing Agent] 결제 관련 문의 처리 중...
[RAG] 관련 지식 검색 중...
[RAG] 3개 관련 문서 발견

Billing Agent: 결제 문제가 발생하셨군요. 확인해보겠습니다.
어떤 결제 수단을 사용하고 계신가요?

고객: 신용카드요

[Billing Agent] 카드 결제 문제 해결 중...
[RAG] "신용카드 결제 오류" 문서 참조

Billing Agent: 신용카드 결제 실패의 일반적인 원인은:
1. 카드 유효기간 만료
2. 한도 초과
3. 해외 결제 차단

다음 단계를 시도해보시겠어요?
1. 카드 정보 재입력
2. 다른 카드로 시도
3. 은행에 문의

고객: 다른 카드로 해볼게요

Billing Agent: 좋습니다! 다른 카드로 시도해보세요.
여전히 문제가 있으시면 언제든 말씀해주세요.

만족도를 평가해주세요 (1-5): 5

[Monitoring] 세션 종료
- 소요 시간: 45초
- 해결됨: Yes
- 만족도: 5/5

감사합니다!
```

### API 모드

```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "결제가 안 돼요",
    "session_id": "user123"
})

print(response.json())
# {
#   "response": "결제 문제가 발생하셨군요...",
#   "agent": "billing",
#   "confidence": 0.95,
#   "sources": [...]
# }
```

---

## 구현 가이드

### Step 1: Router Agent

```python
# agents/router.py
class RouterAgent(BaseAgent):
    """문의를 적절한 Agent로 라우팅"""

    CATEGORIES = {
        "support": ["기술", "오류", "작동", "설치"],
        "billing": ["결제", "요금", "환불", "구독"],
        "general": ["일반", "정보", "문의"],
    }

    def route(self, message: str) -> Dict:
        """메시지를 분류하고 라우팅"""
        category = self._classify(message)
        confidence = self._calculate_confidence(message, category)

        return {
            "category": category,
            "confidence": confidence,
            "agent": self._get_agent(category),
        }
```

### Step 2: RAG Knowledge Base

```python
# knowledge/rag_system.py
class CustomerServiceRAG:
    """고객 서비스용 RAG 시스템"""

    def __init__(self):
        self.vectorstore = self._build_vectorstore()

    def search(self, query: str, category: str = None) -> List[Document]:
        """지식 베이스 검색"""
        # 카테고리별 필터링
        filter_dict = {"category": category} if category else None

        results = self.vectorstore.similarity_search(
            query,
            k=3,
            filter=filter_dict
        )

        return results
```

### Step 3: HITL Implementation

```python
# middleware/hitl.py
class HumanInTheLoop:
    """Human-in-the-Loop 미들웨어"""

    CRITICAL_ACTIONS = ["refund", "cancel_subscription", "delete_account"]

    def requires_approval(self, action: str) -> bool:
        """승인 필요 여부 확인"""
        return action in self.CRITICAL_ACTIONS

    def request_approval(self, action: str, context: Dict) -> bool:
        """사람의 승인 요청"""
        print(f"\n⚠️  승인 필요: {action}")
        print(f"상황: {context}")
        response = input("승인하시겠습니까? (y/n): ")
        return response.lower() == 'y'
```

### Step 4: Monitoring

```python
# middleware/monitoring.py
class Monitor:
    """시스템 모니터링"""

    def track_request(self, session_id: str):
        """요청 추적 시작"""
        self.sessions[session_id] = {
            "start_time": time.time(),
            "messages": [],
            "agent_switches": 0,
        }

    def log_metric(self, metric: str, value: Any):
        """메트릭 로깅"""
        self.metrics[metric].append({
            "timestamp": time.time(),
            "value": value,
        })
```

---

## 고급 기능

### 1. 컨텍스트 유지

```python
class SessionManager:
    """세션 관리 및 컨텍스트 유지"""

    def __init__(self):
        self.sessions = {}

    def get_context(self, session_id: str) -> List[Message]:
        """세션 컨텍스트 조회"""
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, message: Message):
        """메시지 추가"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)
```

### 2. 폴백 전략

```python
class FallbackHandler:
    """폴백 처리"""

    def handle_failure(self, error: Exception, context: Dict):
        """실패 처리"""
        if isinstance(error, RateLimitError):
            return self._wait_and_retry()
        elif isinstance(error, InvalidRequestError):
            return self._use_simpler_model()
        else:
            return self._escalate_to_human()
```

### 3. A/B 테스팅

```python
class ABTest:
    """A/B 테스팅"""

    def select_variant(self, user_id: str) -> str:
        """사용자에게 변형 할당"""
        hash_value = hash(user_id)
        return "A" if hash_value % 2 == 0 else "B"

    def track_result(self, variant: str, outcome: Dict):
        """결과 추적"""
        self.results[variant].append(outcome)
```

---

## 테스트

### 단위 테스트

```bash
pytest tests/test_agents.py -v
```

### 통합 테스트

```bash
pytest tests/test_integration.py -v
```

### 부하 테스트

```bash
locust -f tests/load_test.py
```

---

## 배포

### Docker

```bash
# 이미지 빌드
docker build -t customer-service-agent .

# 컨테이너 실행
docker run -p 8000:8000 customer-service-agent
```

### Docker Compose

```bash
docker-compose up -d
```

### Kubernetes

```bash
kubectl apply -f deployment/kubernetes/
```

---

## 모니터링 대시보드

### 주요 메트릭

1. **성능 메트릭**
   - 평균 응답 시간
   - 처리량 (requests/min)
   - Agent 전환 횟수

2. **품질 메트릭**
   - 해결율
   - 사용자 만족도
   - 에스컬레이션 비율

3. **비용 메트릭**
   - API 호출 수
   - 토큰 사용량
   - 비용/세션

---

## 평가 기준

### 기능 완성도 (35점)
- [ ] Multi-Agent 라우팅
- [ ] RAG 지식 베이스
- [ ] HITL 구현
- [ ] 에러 핸들링
- [ ] 모니터링

### 코드 품질 (25점)
- [ ] 모듈화 및 재사용성
- [ ] 타입 안전성
- [ ] 테스트 커버리지
- [ ] 문서화

### 성능 (20점)
- [ ] 응답 시간 < 3초
- [ ] 동시 사용자 처리
- [ ] 리소스 효율성

### 프로덕션 준비 (20점)
- [ ] 배포 설정
- [ ] 로깅 및 모니터링
- [ ] 보안 고려사항
- [ ] 확장성

---

## 문제 해결

### Q: Agent가 잘못된 응답을 해요
A: 프롬프트 개선 및 RAG 검색 품질 향상

### Q: 응답이 너무 느려요
A: 캐싱, 병렬 처리, 더 빠른 모델 사용

### Q: 비용이 너무 높아요
A: GPT-4o-mini 활용, 캐싱, 토큰 사용 최적화

### Q: 배포가 안 돼요
A: 환경 변수 확인, 포트 설정, 로그 확인

---

## 참고 자료

- [LangChain Production Guide](https://python.langchain.com/docs/guides/production/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- Part 7: 실전 프로젝트

---

## 다음 단계

프로젝트 완료 후:
1. 실제 서비스에 적용
2. 성능 최적화
3. 추가 기능 개발
4. 사용자 피드백 수집

**축하합니다! 모든 프로젝트를 완료하셨습니다!**
