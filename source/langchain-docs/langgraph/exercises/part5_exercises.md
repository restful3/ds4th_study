# Part 5: Advanced 연습 문제

> 고급 주제를 다루는 연습 문제입니다.

---

## 문제 1: Functional API 기본 ⭐⭐

### 설명
Functional API를 사용하여 간단한 워크플로우를 만드세요.

### 요구사항
1. `@entrypoint`로 워크플로우 정의
2. `@task`로 개별 작업 정의
3. Python 제어문으로 조건 분기

### 예시
```python
@task
def process(data: str) -> str:
    return data.upper()

@entrypoint(checkpointer=MemorySaver())
def workflow(data: str) -> str:
    result = process(data).result()
    return result
```

---

## 문제 2: Durable Execution ⭐⭐⭐

### 설명
장애에도 복구 가능한 워크플로우를 구현하세요.

### 요구사항
1. 3단계 처리 파이프라인
2. 각 단계 후 checkpoint 저장
3. 장애 시뮬레이션 및 복구 테스트

### 테스트 시나리오
```python
# 1. 정상 실행
result = app.invoke(input, config)

# 2. 2단계에서 장애 시뮬레이션 (프로세스 재시작)
# 3. 같은 thread_id로 재실행
result = app.invoke(None, config)  # 2단계부터 재개
```

---

## 문제 3: 재시도 로직 ⭐⭐⭐

### 설명
지수 백오프를 적용한 재시도 로직을 구현하세요.

### 요구사항
1. 실패 시 자동 재시도 (최대 5회)
2. 지수 백오프: 1초, 2초, 4초, 8초...
3. 최종 실패 시 에러 상태 기록

### State
```python
class RetryState(TypedDict):
    attempt: int
    max_attempts: int
    backoff_seconds: float
    success: bool
    error: Optional[str]
```

---

## 문제 4: 멱등성 보장 ⭐⭐⭐⭐

### 설명
같은 요청을 여러 번 호출해도 한 번만 처리되는 그래프를 만드세요.

### 요구사항
1. 요청 ID로 중복 체크
2. 이미 처리된 요청은 캐시된 결과 반환
3. 처리 결과 저장

### 힌트
```python
# 처리된 요청 ID 저장
processed_requests = {}

def check_idempotency(state):
    if state["request_id"] in processed_requests:
        return {"result": processed_requests[state["request_id"]], "skipped": True}
    return {}
```

---

## 문제 5: 프로덕션 배포 체크리스트 ⭐⭐⭐⭐

### 설명
프로덕션 배포를 위한 완전한 그래프를 구현하세요.

### 요구사항
1. 환경 설정 관리 (Config 클래스)
2. 구조화된 로깅
3. 메트릭 수집
4. 에러 처리 및 복구
5. 헬스 체크 엔드포인트

### 포함 요소
- `Config` 클래스로 설정 관리
- `logging` 모듈로 구조화된 로깅
- `Metrics` 클래스로 호출 횟수, 응답 시간 수집
- 재시도 로직
- `/health` 엔드포인트 구현

---

## 종합 문제: AI 비서 시스템 ⭐⭐⭐⭐⭐

### 설명
지금까지 배운 모든 개념을 활용하여 AI 비서 시스템을 구축하세요.

### 요구사항
1. **Multi-Agent**: Supervisor + 전문 Agent들
2. **도구 사용**: 검색, 계산, 일정 관리 도구
3. **메모리**: 대화 기록 + 사용자 선호도 저장
4. **Human-in-the-Loop**: 중요 작업 전 승인 요청
5. **Durable Execution**: 장애 복구 지원
6. **스트리밍**: 실시간 응답 출력

### 아키텍처
```
[User Input]
     ↓
[Supervisor Agent]
     ├── [Research Agent] → 검색 도구
     ├── [Assistant Agent] → 일정 관리 도구
     └── [Calculator Agent] → 계산 도구
     ↓
[Human Approval] (필요 시)
     ↓
[Response]
```

---

## 해답

해답은 [solutions/part5_solutions.py](./solutions/part5_solutions.py)를 참조하세요.
