# Part 5: Middleware - Agent 동작 제어하기

> 📚 **학습 시간**: 약 3-4시간
> 🎯 **난이도**: ⭐⭐⭐⭐☆ (고급)
> 📖 **공식 문서**: [14-middleware-overview.md](/official/14-middleware-overview.md), [15-built-in-middleware.md](/official/15-built-in-middleware.md), [16-custom-middleware.md](/official/16-custom-middleware.md), [17-guardrails.md](/official/17-guardrails.md)
> 📄 **교안 문서**: [part05_middleware.md](/docs/part05_middleware.md)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] Middleware의 개념과 Agent 실행 루프 이해
- [x] Built-in Middleware 활용 (Summarization, HITL, Tool Retry)
- [x] Custom Middleware 구현 (before/after/wrap 패턴)
- [x] Guardrails로 안전한 Agent 구축

---

## 📚 개요

**Middleware**는 Agent의 실행 파이프라인에 끼워넣을 수 있는 커스텀 로직입니다. 로깅, 모니터링, 안전 장치 등을 추가할 수 있습니다.

**왜 중요한가?**
- Agent 동작을 세밀하게 제어
- 프로덕션 환경의 필수 기능 (로깅, 모니터링, 안전장치)
- 재사용 가능한 로직 모듈화

**실무 활용 사례**
- 비용 추적 및 최적화
- 콘텐츠 필터링 및 안전 검사
- 자동 재시도 및 폴백
- 성능 모니터링

---

## 📁 예제 파일

### 01_middleware_intro.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 30분

Middleware의 기본 개념과 Agent 실행 루프를 이해합니다.

**학습 내용**:
- Agent 실행 루프 구조
- Middleware가 개입하는 지점
- 간단한 로깅 Middleware
- Middleware 체인

**실행 방법**:
```bash
python 01_middleware_intro.py
```

**주요 개념**:
- Agent Loop: Input → Model → Tool → Model → Output
- Middleware는 각 단계에 끼워넣기 가능
- 여러 Middleware를 조합 가능

---

### 02_before_after_model.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 35분

Model 호출 전후에 실행되는 Middleware를 구현합니다.

**학습 내용**:
- `before_model` hook
- `after_model` hook
- 입력/출력 수정
- 로깅 및 모니터링

**실행 방법**:
```bash
python 02_before_after_model.py
```

**주요 개념**:
- **before_model**: 프롬프트 수정, 입력 검증
- **after_model**: 응답 필터링, 로깅

---

### 03_wrap_model_call.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 45분

Model 호출을 감싸서 완전히 제어하는 Middleware를 구현합니다.

**학습 내용**:
- `wrap_model_call` 패턴
- Try-catch로 에러 처리
- 재시도 로직
- 폴백 응답

**실행 방법**:
```bash
python 03_wrap_model_call.py
```

**주요 개념**:
- 원본 호출을 감싸기
- 에러 핸들링 및 복구
- 성능 측정 (latency, tokens)

---

### 04_wrap_tool_call.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 45분

Tool 호출을 감싸는 Middleware를 구현합니다.

**학습 내용**:
- `wrap_tool_call` 패턴
- 도구 실행 전후 처리
- 도구 에러 핸들링
- 권한 검사

**실행 방법**:
```bash
python 04_wrap_tool_call.py
```

**주요 개념**:
- 도구 호출 감시
- 위험한 도구 차단
- 도구 실행 로그

---

### 05_summarization_mw.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 40분

Built-in Summarization Middleware를 사용합니다.

**학습 내용**:
- 자동 대화 요약
- 토큰 한도 관리
- 요약 전략 설정
- Part 4 메모리와 통합

**실행 방법**:
```bash
python 05_summarization_mw.py
```

**주요 개념**:
- Middleware로 메모리 관리 자동화
- 긴 대화 처리
- 설정 가능한 요약 조건

---

### 06_tool_retry.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 50분

도구 실패 시 자동 재시도하는 Middleware를 구현합니다.

**학습 내용**:
- 재시도 전략 (exponential backoff)
- 재시도 횟수 제한
- 특정 에러만 재시도
- 폴백 응답

**실행 방법**:
```bash
python 06_tool_retry.py
```

**주요 개념**:
- 네트워크 오류 등 일시적 실패 대응
- 지수 백오프 (exponential backoff)
- 최대 재시도 횟수

---

### 07_guardrails.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

안전한 Agent를 위한 Guardrails를 구현합니다.

**학습 내용**:
- 콘텐츠 필터링
- PII (개인정보) 검출
- 유해 콘텐츠 차단
- 안전 점수 평가

**실행 방법**:
```bash
python 07_guardrails.py
```

**주요 개념**:
- **Guardrails**: 안전 장치, 품질 관리
- 입력/출력 모두 검사
- 정책 위반 시 차단

---

## 🎓 실습 과제

### 과제 1: 비용 추적 Middleware (⭐⭐⭐)

**목표**: 모든 LLM 호출의 비용을 추적하는 Middleware를 만드세요.

**요구사항**:
1. 각 호출의 토큰 사용량 기록
2. 비용 계산 (모델별 가격 적용)
3. 누적 비용 출력

**힌트**:
```python
# 모델별 가격 (per 1M tokens)
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}
```

**해답**: [solutions/exercise_01.py](/src/part05_middleware/solutions/exercise_01.py)

---

### 과제 2: 캐싱 Middleware (⭐⭐⭐⭐)

**목표**: 동일한 질문은 캐시에서 응답하는 Middleware를 만드세요.

**요구사항**:
1. 질문을 해시화하여 캐시 키 생성
2. 캐시 히트 시 즉시 응답 (LLM 호출 생략)
3. 캐시 만료 시간 설정 (예: 1시간)

**해답**: [solutions/exercise_02.py](/src/part05_middleware/solutions/exercise_02.py)

---

### 과제 3: 종합 모니터링 시스템 (⭐⭐⭐⭐⭐)

**목표**: 프로덕션급 모니터링 Middleware를 만드세요.

**요구사항**:
1. 모든 호출 로깅 (타임스탬프, 입력, 출력)
2. 성능 메트릭 (latency, tokens, cost)
3. 에러 추적 및 알림
4. JSON 파일 또는 DB에 저장

**해답**: [solutions/exercise_03.py](/src/part05_middleware/solutions/exercise_03.py)

---

## 💡 실전 팁

### Tip 1: Middleware 순서가 중요합니다

```python
# Middleware는 선언 순서대로 실행됨
agent = create_agent(
    model=model,
    tools=tools,
    middlewares=[
        auth_middleware,       # 1. 인증 먼저
        rate_limit_middleware, # 2. 속도 제한
        cost_tracking,         # 3. 비용 추적
        logging_middleware,    # 4. 로깅은 마지막
    ]
)
```

### Tip 2: Decorator vs Class

```python
# 방법 1: Decorator (간단한 경우)
@middleware
def log_middleware(state, next_step):
    print("Before")
    result = next_step(state)
    print("After")
    return result

# 방법 2: Class (상태 관리 필요)
class CostTracker:
    def __init__(self):
        self.total_cost = 0

    def __call__(self, state, next_step):
        # 비용 계산 로직
        pass
```

### Tip 3: 에러 핸들링

```python
def safe_middleware(state, next_step):
    try:
        return next_step(state)
    except RateLimitError:
        # 재시도
        time.sleep(60)
        return next_step(state)
    except Exception as e:
        # 로깅 후 재발생
        logger.error(f"Middleware error: {e}")
        raise
```

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: Middleware는 언제 사용하나요?</summary>

**A**: 다음 상황에서 유용합니다:
- **로깅/모니터링**: 모든 호출 추적
- **비용 관리**: 토큰 사용량 제한
- **안전성**: 콘텐츠 필터링, PII 제거
- **성능**: 캐싱, 재시도 로직
- **디버깅**: 상세한 실행 로그
</details>

<details>
<summary>Q2: Middleware와 Tool의 차이는?</summary>

**A**:
- **Tool**: Agent가 명시적으로 호출 (LLM이 결정)
- **Middleware**: 자동으로 실행 (개발자가 설정)

```python
# Tool: LLM이 필요 시 호출
@tool
def search(query: str):
    return google_search(query)

# Middleware: 항상 실행
def log_all_calls(state, next_step):
    print(f"Calling with: {state}")
    return next_step(state)
```
</details>

<details>
<summary>Q3: 성능에 영향을 주나요?</summary>

**A**: 영향이 있을 수 있습니다:
- **최소 영향**: 간단한 로깅
- **중간 영향**: 캐시 조회, 유효성 검증
- **큰 영향**: 외부 API 호출 (예: 안전 점수 평가)

**최적화 팁**:
- 비동기 로깅
- 캐싱 적극 활용
- 병렬 처리 (가능한 경우)
</details>

---

## 🔗 심화 학습

1. **공식 문서 심화**
   - [14-middleware-overview.md](/official/14-middleware-overview.md) - Middleware 개요
   - [15-built-in-middleware.md](/official/15-built-in-middleware.md) - 내장 Middleware
   - [16-custom-middleware.md](/official/16-custom-middleware.md) - 커스텀 Middleware
   - [17-guardrails.md](/official/17-guardrails.md) - 안전 장치

2. **관련 개념**
   - Aspect-Oriented Programming (AOP)
   - Middleware Pattern (웹 프레임워크)
   - Interceptor Pattern

3. **커뮤니티 리소스**
   - [LangChain Middleware Examples](https://python.langchain.com/docs/how_to/custom_middleware/)
   - [프로덕션 모니터링 패턴](https://blog.langchain.dev/monitoring-patterns/)

4. **다음 단계**
   - [Part 6: Context Engineering](/src/part06_context/README.md) - 동적 컨텍스트

---

## ✅ 체크리스트

Part 5를 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다 (7개)
- [ ] 실습 과제를 완료했다 (3개)
- [ ] Middleware의 역할을 이해했다
- [ ] before/after/wrap 패턴의 차이를 안다
- [ ] Built-in Middleware를 사용할 수 있다
- [ ] Custom Middleware를 작성할 수 있다
- [ ] Guardrails의 중요성을 이해했다

---

**이전**: [← Part 4 - Memory Systems](/src/part04_memory/README.md)
**다음**: [Part 6 - Context Engineering로 이동](/src/part06_context/README.md) →

---

**학습 진도**: ▓▓▓▓▓░░░░░ 50% (Part 5/10 완료)

*마지막 업데이트: 2025-02-06*
