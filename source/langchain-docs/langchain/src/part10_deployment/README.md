# Part 10: Deployment & Observability - 배포와 관측성

> 📚 **학습 시간**: 약 4-5시간
> 🎯 **난이도**: ⭐⭐⭐⭐☆ (고급)
> 📖 **공식 문서**: [30-langsmith-studio.md](/official/30-langsmith-studio.md), [31-test.md](/official/31-test.md), [33-deployment.md](/official/33-deployment.md), [34-observability.md](/official/34-observability.md)
> 📄 **교안 문서**: [part10_deployment.md](/docs/part10_deployment.md)
> 🎯 **미니 프로젝트**: [Customer Service Agent](/projects/04_customer_service/)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] LangSmith로 Agent 트레이싱 및 디버깅
- [x] 체계적인 Agent 테스트 작성
- [x] Agent 성능 평가 및 벤치마킹
- [x] 프로덕션 배포 전략
- [x] 관측성 및 모니터링 구축

---

## 📚 개요

Agent를 **실제 서비스**로! 배포, 모니터링, 개선의 전체 사이클을 학습합니다.

**왜 중요한가?**
- 개발 환경 ≠ 프로덕션 환경
- 지속적인 모니터링과 개선 필요
- 신뢰성과 품질 보장

**실무 활용 사례**
- 프로덕션 Agent 배포
- 실시간 모니터링 및 알림
- A/B 테스트 및 성능 최적화

---

## 📁 예제 파일

### 01_langsmith_setup.py
**난이도**: ⭐⭐☆☆☆ | **예상 시간**: 30분

LangSmith 설정 및 기본 트레이싱을 학습합니다.

**학습 내용**:
- LangSmith 계정 설정
- API 키 설정
- 기본 트레이싱 활성화
- Studio UI 둘러보기

**실행 방법**:
```bash
export LANGSMITH_API_KEY="your-api-key"
python 01_langsmith_setup.py
```

**주요 개념**:
- LangSmith = Agent 디버깅 플랫폼
- 자동 트레이싱
- 실행 기록 조회

---

### 02_tracing.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 45분

상세한 트레이싱 및 디버깅을 학습합니다.

**학습 내용**:
- 커스텀 트레이스 메타데이터
- 실행 단계별 추적
- 성능 병목 지점 찾기
- 에러 디버깅

**실행 방법**:
```bash
python 02_tracing.py
```

**주요 개념**:
- 각 LLM 호출 추적
- 도구 실행 시간 측정
- 트레이스 공유 및 협업

---

### 03_testing.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

체계적인 Agent 테스트를 작성합니다.

**학습 내용**:
- Unit Tests (도구 개별 테스트)
- Integration Tests (Agent 전체 테스트)
- Regression Tests (회귀 테스트)
- Test Fixtures

**실행 방법**:
```bash
pytest 03_testing.py
```

**주요 개념**:
- LLM 응답 모킹
- 결정론적 테스트
- CI/CD 통합

---

### 04_evaluation.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 70분

Agent 성능을 평가하고 벤치마킹합니다.

**학습 내용**:
- 평가 데이터셋 생성
- 자동 평가 메트릭
- LLM-as-Judge 패턴
- A/B 테스트

**실행 방법**:
```bash
python 04_evaluation.py
```

**주요 개념**:
- Accuracy, Precision, Recall
- Response Quality 평가
- 비용 및 속도 추적

---

### 05_deployment.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

Agent를 프로덕션 환경에 배포합니다.

**학습 내용**:
- LangServe로 API 서버 구축
- Docker 컨테이너화
- 환경 변수 관리
- 배포 옵션 비교

**실행 방법**:
```bash
python 05_deployment.py
# 또는
docker build -t my-agent .
docker run -p 8000:8000 my-agent
```

**주요 개념**:
- REST API 엔드포인트
- Scalability 고려사항
- 클라우드 배포 (AWS, GCP, Azure)

---

### 06_observability.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 70분

프로덕션 관측성 및 모니터링을 구축합니다.

**학습 내용**:
- 로깅 전략
- 메트릭 수집 (Prometheus, CloudWatch)
- 알림 설정
- 대시보드 구축

**실행 방법**:
```bash
python 06_observability.py
```

**주요 개념**:
- SLO/SLA 정의
- 핵심 메트릭: Latency, Error Rate, Cost
- 이상 감지 및 알림

---

## 🎓 실습 과제

### 과제 1: 테스트 스위트 구축 (⭐⭐⭐)

**목표**: 이전 파트의 Agent를 위한 완전한 테스트 스위트를 만드세요.

**요구사항**:
1. 각 도구에 대한 Unit Tests
2. Agent 전체에 대한 Integration Tests
3. Edge Case 테스트 (에러, 빈 입력 등)
4. 테스트 커버리지 80% 이상

**해답**: [solutions/exercise_01.py](/src/part10_deployment/solutions/exercise_01.py)

---

### 과제 2: 평가 시스템 (⭐⭐⭐⭐)

**목표**: Agent 품질을 자동으로 평가하는 시스템을 만드세요.

**요구사항**:
1. 평가 데이터셋 (질문-정답 쌍 20개)
2. 자동 평가 스크립트
3. 정확도, 응답 시간, 비용 측정
4. 결과 리포트 생성

**해답**: [solutions/exercise_02.py](/src/part10_deployment/solutions/exercise_02.py)

---

### 과제 3: 프로덕션 배포 (⭐⭐⭐⭐⭐)

**목표**: Agent를 완전한 프로덕션 시스템으로 배포하세요.

**요구사항**:
1. LangServe API 서버
2. Docker 컨테이너
3. 로깅 및 모니터링
4. Health Check 엔드포인트
5. 환경별 설정 (dev, staging, prod)

**해답**: [solutions/exercise_03.py](/src/part10_deployment/solutions/exercise_03.py)

---

## 💡 실전 팁

### Tip 1: LangSmith 활용

```python
from langsmith import Client
from langchain_core.tracers import LangChainTracer

# LangSmith 클라이언트
client = Client()

# 커스텀 메타데이터
tracer = LangChainTracer(
    project_name="my-agent-prod",
    metadata={
        "user_id": "user123",
        "environment": "production",
        "version": "1.0.0"
    }
)

# Agent 실행 시 트레이서 전달
result = agent.invoke(
    input,
    config={"callbacks": [tracer]}
)
```

### Tip 2: 테스트 전략

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_llm():
    """LLM 응답 모킹"""
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="Mocked response")
    return llm

def test_agent_with_mock(mock_llm):
    """결정론적 테스트"""
    agent = create_agent(model=mock_llm, tools=tools)
    result = agent.invoke({"messages": ["test"]})
    assert "Mocked response" in result["messages"][-1].content
```

### Tip 3: 핵심 메트릭 추적

```python
import time
from functools import wraps

def track_metrics(func):
    """성능 메트릭 추적 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            # 메트릭 기록
            log_metric("agent.latency", duration)
            log_metric("agent.success", 1)

            # 비용 추적
            if hasattr(result, "usage_metadata"):
                tokens = result.usage_metadata.total_tokens
                cost = calculate_cost(tokens)
                log_metric("agent.cost", cost)

            return result
        except Exception as e:
            log_metric("agent.error", 1)
            raise
    return wrapper

@track_metrics
def run_agent(input):
    return agent.invoke(input)
```

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: LangSmith는 필수인가요?</summary>

**A**: 필수는 아니지만 **강력 추천**:
- **개발 단계**: 디버깅에 매우 유용
- **프로덕션**: 실시간 모니터링 및 이슈 추적

**대안**:
- OpenTelemetry + Custom Backend
- 로컬 로깅 + 분석 도구
</details>

<details>
<summary>Q2: Agent 테스트는 어떻게 하나요?</summary>

**A**: 계층별 접근:
1. **도구 테스트**: 각 도구를 독립적으로
2. **Agent 테스트**: LLM 모킹으로 결정론적
3. **통합 테스트**: 실제 LLM으로 E2E
4. **Regression 테스트**: 과거 이슈 재발 방지

```python
# E2E 테스트 예시
def test_weather_agent():
    result = agent.invoke({"messages": ["서울 날씨는?"]})
    assert "서울" in result["messages"][-1].content
    assert any("날씨" in m.content for m in result["messages"])
```
</details>

<details>
<summary>Q3: 프로덕션 배포 시 주의사항은?</summary>

**A**:
1. **API 키 관리**: 환경 변수, Secrets Manager
2. **Rate Limiting**: LLM API 한도 초과 방지
3. **에러 핸들링**: 모든 실패 케이스 대응
4. **로깅**: 충분한 디버깅 정보
5. **모니터링**: 실시간 알림 설정
6. **비용 추적**: 예산 초과 방지
</details>

---

## 🚀 최종 프로젝트

### Project 4: Production Customer Service Agent

지금까지 배운 모든 내용을 종합하여 완전한 고객 서비스 시스템을 구축하세요!

**프로젝트 링크**: [Customer Service Agent](/projects/04_customer_service/)

**주요 기능**:
- 멀티에이전트 라우팅 시스템
- RAG 기반 지식 베이스
- HITL로 중요 작업 승인
- 완전한 테스트 스위트
- 프로덕션 배포 설정

**예상 소요 시간**: 6-8시간
**난이도**: ⭐⭐⭐⭐⭐

---

## 🔗 심화 학습

1. **공식 문서 심화**
   - [30-langsmith-studio.md](/official/30-langsmith-studio.md) - LangSmith
   - [31-test.md](/official/31-test.md) - 테스팅
   - [33-deployment.md](/official/33-deployment.md) - 배포
   - [34-observability.md](/official/34-observability.md) - 관측성

2. **배포 플랫폼**
   - [LangServe](https://python.langchain.com/docs/langserve) - LangChain API 서버
   - [Modal](https://modal.com/) - 서버리스 배포
   - [AWS Lambda](https://aws.amazon.com/lambda/) - 서버리스
   - [Kubernetes](https://kubernetes.io/) - 컨테이너 오케스트레이션

3. **모니터링 도구**
   - [Prometheus](https://prometheus.io/) - 메트릭 수집
   - [Grafana](https://grafana.com/) - 대시보드
   - [Sentry](https://sentry.io/) - 에러 추적

4. **추가 학습**
   - [부록: Troubleshooting](/docs/appendix/troubleshooting.md)
   - [부록: Resources](/docs/appendix/resources.md)

---

## ✅ 최종 체크리스트

🎉 **축하합니다! 전체 교안을 완료했습니다!**

Part 10을 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다 (6개)
- [ ] 실습 과제를 완료했다 (3개)
- [ ] LangSmith로 트레이싱할 수 있다
- [ ] Agent 테스트를 작성할 수 있다
- [ ] 평가 시스템을 구축할 수 있다
- [ ] Agent를 배포할 수 있다
- [ ] 관측성 시스템을 설정할 수 있다

**전체 교안 체크리스트**:
- [ ] Part 1-10 모든 예제 실행 (56개)
- [ ] 모든 실습 과제 완료 (30개)
- [ ] 4개 미니 프로젝트 완료
- [ ] 자신만의 Agent 프로젝트 시작!

---

**이전**: [← Part 9 - Production](/src/part09_production/README.md)

---

## 🎓 다음 단계

교안을 완료한 후:

1. **자신만의 프로젝트**
   - 실제 문제를 해결하는 Agent 구축
   - 오픈소스 기여

2. **커뮤니티 참여**
   - [LangChain Discord](https://discord.gg/langchain)
   - [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

3. **최신 정보 팔로우**
   - [LangChain Blog](https://blog.langchain.dev/)
   - [Twitter/X @LangChainAI](https://twitter.com/LangChainAI)

4. **고급 주제 탐구**
   - Fine-tuning 및 최적화
   - 멀티모달 Agent
   - 강화학습 기반 Agent

---

**학습 진도**: ▓▓▓▓▓▓▓▓▓▓ 100% (Part 10/10 완료) 🎉

**축하합니다! LangChain AI Agent 마스터가 되셨습니다!** 🚀

*마지막 업데이트: 2025-02-06*
