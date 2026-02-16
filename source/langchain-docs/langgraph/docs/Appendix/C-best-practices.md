# Appendix C: 모범 사례

> 📌 LangGraph 애플리케이션 개발의 모범 사례와 설계 원칙을 정리한 가이드입니다.

## 그래프 설계 원칙

### 1. 단일 책임 원칙 (Single Responsibility)

각 노드는 하나의 명확한 책임만 가져야 합니다.

```python
# ❌ 잘못된 예: 너무 많은 책임
def do_everything(state):
    data = fetch_data(state["query"])
    processed = transform_data(data)
    validated = validate_data(processed)
    result = save_data(validated)
    return {"result": result}

# ✅ 올바른 예: 분리된 책임
def fetch_node(state):
    return {"raw_data": fetch_data(state["query"])}

def transform_node(state):
    return {"processed_data": transform_data(state["raw_data"])}

def validate_node(state):
    return {"validated_data": validate_data(state["processed_data"])}

def save_node(state):
    return {"result": save_data(state["validated_data"])}
```

### 2. 명확한 상태 정의

State는 명확하고 타입이 정의되어야 합니다.

```python
# ❌ 잘못된 예: 모호한 상태
class BadState(TypedDict):
    data: dict  # 무엇이 들어있는지 불명확
    flag: bool  # 무슨 의미인지 불명확

# ✅ 올바른 예: 명확한 상태
class GoodState(TypedDict):
    user_query: str
    search_results: List[SearchResult]
    is_processing_complete: bool
    error_message: Optional[str]
```

### 3. 순수 함수 지향

노드 함수는 가능한 순수 함수로 작성합니다.

```python
# ❌ 잘못된 예: 부작용이 있는 함수
global_counter = 0

def impure_node(state):
    global global_counter
    global_counter += 1  # 전역 상태 변경
    return {"count": global_counter}

# ✅ 올바른 예: 순수 함수
def pure_node(state):
    current_count = state.get("count", 0)
    return {"count": current_count + 1}
```

---

## State 설계 패턴

### 1. 최소 필요 원칙

State에는 꼭 필요한 정보만 포함합니다.

```python
# ❌ 잘못된 예: 불필요한 정보 포함
class BloatedState(TypedDict):
    user_input: str
    intermediate_result_1: str
    intermediate_result_2: str
    intermediate_result_3: str
    debug_info: dict
    timestamps: list

# ✅ 올바른 예: 필요한 정보만
class LeanState(TypedDict):
    messages: Annotated[list, add_messages]
    current_step: str
    final_result: Optional[str]
```

### 2. 불변성 고려

State 업데이트 시 불변성을 유지합니다.

```python
# ❌ 잘못된 예: 원본 변경
def bad_update(state):
    state["items"].append(new_item)  # 원본 변경
    return state

# ✅ 올바른 예: 새 객체 반환
def good_update(state):
    return {"items": state["items"] + [new_item]}  # 새 리스트
```

### 3. Reducer 활용

누적되는 데이터는 Reducer를 사용합니다.

```python
from typing import Annotated
from langgraph.graph.message import add_messages
import operator

class WellDesignedState(TypedDict):
    # 메시지는 add_messages로 자동 병합
    messages: Annotated[list, add_messages]

    # 리스트는 operator.add로 연결
    logs: Annotated[list, operator.add]

    # 단일 값은 덮어쓰기 (Reducer 없음)
    current_status: str
```

---

## 에러 처리 패턴

### 1. 명시적 에러 상태

에러를 State에 명시적으로 표현합니다.

```python
class RobustState(TypedDict):
    input: str
    result: Optional[str]
    error: Optional[str]
    error_code: Optional[str]


def safe_node(state):
    try:
        result = process(state["input"])
        return {"result": result, "error": None}
    except ValidationError as e:
        return {"result": None, "error": str(e), "error_code": "VALIDATION"}
    except Exception as e:
        return {"result": None, "error": str(e), "error_code": "UNKNOWN"}


def route_after_process(state):
    if state.get("error"):
        return "error_handler"
    return "next_step"
```

### 2. 재시도 패턴

```python
def create_retry_node(max_retries: int = 3):
    """재시도 노드 생성"""

    def retry_node(state):
        attempt = state.get("attempt", 0)

        try:
            result = risky_operation(state["input"])
            return {"result": result, "attempt": 0}
        except RetryableError as e:
            if attempt < max_retries:
                return {"attempt": attempt + 1, "last_error": str(e)}
            else:
                return {"error": f"최대 재시도 초과: {e}"}

    return retry_node


def route_retry(state):
    if state.get("error"):
        return "error_handler"
    if state.get("attempt", 0) > 0:
        return "retry_node"  # 재시도
    return "next_step"
```

### 3. 폴백 패턴

```python
def primary_node(state):
    """주요 처리"""
    try:
        return {"result": primary_process(state)}
    except Exception:
        return {"use_fallback": True}


def fallback_node(state):
    """대체 처리"""
    return {"result": fallback_process(state)}


def route_fallback(state):
    if state.get("use_fallback"):
        return "fallback"
    return "next"
```

---

## 성능 최적화

### 1. 조기 종료

불필요한 처리를 피하기 위해 조기 종료합니다.

```python
def check_cache(state):
    """캐시 확인"""
    cached = cache.get(state["query"])
    if cached:
        return {"result": cached, "from_cache": True}
    return {"from_cache": False}


def route_cache(state):
    if state.get("from_cache"):
        return END  # 캐시 히트 - 조기 종료
    return "process"
```

### 2. 배치 처리

```python
def batch_processor(state):
    """배치로 처리"""
    items = state["items"]

    # 청크로 나누어 처리
    chunk_size = 50
    results = []

    for i in range(0, len(items), chunk_size):
        chunk = items[i:i+chunk_size]
        chunk_results = process_batch(chunk)
        results.extend(chunk_results)

    return {"results": results}
```

### 3. 비동기 활용

```python
import asyncio

async def parallel_fetch(state):
    """병렬 데이터 조회"""
    queries = state["queries"]

    async def fetch_one(query):
        return await async_api_call(query)

    # 동시에 모든 쿼리 실행
    results = await asyncio.gather(*[
        fetch_one(q) for q in queries
    ])

    return {"results": results}
```

---

## 테스트 가능한 설계

### 1. 의존성 주입

```python
# ❌ 하드코딩된 의존성
def bad_node(state):
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")  # 테스트 어려움
    return {"response": llm.invoke(state["query"])}

# ✅ 의존성 주입
def create_llm_node(llm):
    def llm_node(state):
        return {"response": llm.invoke(state["query"])}
    return llm_node

# 사용
graph.add_node("llm", create_llm_node(llm))

# 테스트
mock_llm = Mock()
test_node = create_llm_node(mock_llm)
```

### 2. 인터페이스 분리

```python
from abc import ABC, abstractmethod

class DataFetcher(ABC):
    @abstractmethod
    def fetch(self, query: str) -> dict:
        pass


class APIDataFetcher(DataFetcher):
    def fetch(self, query: str) -> dict:
        return requests.get(f"/api?q={query}").json()


class MockDataFetcher(DataFetcher):
    def fetch(self, query: str) -> dict:
        return {"mock": True, "query": query}


def create_fetch_node(fetcher: DataFetcher):
    def fetch_node(state):
        return {"data": fetcher.fetch(state["query"])}
    return fetch_node
```

---

## 모니터링 및 관측성

### 1. 구조화된 로깅

```python
import structlog

logger = structlog.get_logger()

def observable_node(state):
    """관측 가능한 노드"""
    logger.info(
        "node_started",
        node="observable_node",
        input_size=len(state.get("input", "")),
        thread_id=state.get("_thread_id")
    )

    try:
        result = process(state)
        logger.info(
            "node_completed",
            node="observable_node",
            success=True
        )
        return result
    except Exception as e:
        logger.error(
            "node_failed",
            node="observable_node",
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

### 2. 메트릭 수집

```python
from prometheus_client import Counter, Histogram

node_executions = Counter(
    'langgraph_node_executions_total',
    'Total node executions',
    ['node_name', 'status']
)

node_duration = Histogram(
    'langgraph_node_duration_seconds',
    'Node execution duration',
    ['node_name']
)

def instrumented_node(state):
    """메트릭이 수집되는 노드"""
    with node_duration.labels(node_name="my_node").time():
        try:
            result = process(state)
            node_executions.labels(
                node_name="my_node",
                status="success"
            ).inc()
            return result
        except Exception:
            node_executions.labels(
                node_name="my_node",
                status="error"
            ).inc()
            raise
```

### 3. 트레이싱

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def traced_node(state):
    """트레이싱이 적용된 노드"""
    with tracer.start_as_current_span("my_node") as span:
        span.set_attribute("input.length", len(state.get("input", "")))

        result = process(state)

        span.set_attribute("result.length", len(result.get("output", "")))
        return result
```

---

## 프로덕션 체크리스트

### 배포 전 필수 확인

```markdown
## 코드 품질
- [ ] 모든 노드 함수에 타입 힌트 적용
- [ ] 에러 핸들링 구현 완료
- [ ] 로깅 구현 완료
- [ ] 단위 테스트 작성 (커버리지 80% 이상)
- [ ] 통합 테스트 작성

## 설정
- [ ] 환경 변수로 설정 관리
- [ ] API 키 시크릿 관리
- [ ] 로그 레벨 설정
- [ ] 타임아웃 설정

## 인프라
- [ ] Checkpointer 설정 (PostgresSaver 등)
- [ ] 헬스 체크 엔드포인트
- [ ] 메트릭 수집 설정
- [ ] 알림 설정

## 보안
- [ ] 입력 검증
- [ ] 출력 필터링
- [ ] Rate Limiting
- [ ] 인증/인가

## 운영
- [ ] 모니터링 대시보드
- [ ] 로그 수집
- [ ] 백업 정책
- [ ] 장애 대응 매뉴얼
```

---

## 안티 패턴

### 피해야 할 것들

1. **God Node**: 모든 것을 처리하는 거대한 노드
2. **Spaghetti Graph**: 복잡하게 얽힌 에지
3. **Global State**: 전역 변수 의존
4. **Hardcoded Config**: 하드코딩된 설정값
5. **Silent Failures**: 조용히 실패하는 에러 처리
6. **Infinite Loops**: 종료 조건 없는 순환
7. **Tight Coupling**: 노드 간 강한 결합

---

## 마무리

이 모범 사례들을 따르면:

- **유지보수성**: 코드 이해와 수정이 쉬움
- **테스트 가능성**: 자동화된 테스트 가능
- **확장성**: 새로운 기능 추가가 용이
- **안정성**: 프로덕션 환경에서 안정적 운영

LangGraph로 멋진 AI 애플리케이션을 만들어 보세요! 🚀
