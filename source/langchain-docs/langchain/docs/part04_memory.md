# Part 4: Memory System (메모리 시스템)

**학습 시간**: 4-5시간
**난이도**: ⭐⭐⭐ (중급)

## 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

1. **Short-term Memory 구현**: Thread 기반 대화 이력 관리
2. **Production Checkpointer 사용**: PostgreSQL을 이용한 상태 영속성
3. **Message Management**: Context window 문제 해결
4. **Summarization 구현**: 긴 대화 요약 전략
5. **Custom State 확장**: AgentState에 사용자 정의 필드 추가
6. **Long-term Memory 구현**: Store를 이용한 장기 메모리 관리

## 개요

AI Agent에게 메모리는 필수적입니다. 메모리가 없다면 Agent는 이전 대화를 기억하지 못하고, 사용자의 선호도를 학습할 수 없으며, 반복적인 질문에 매번 같은 답변을 해야 합니다.

LangChain은 두 가지 유형의 메모리를 제공합니다:

- **Short-term Memory (단기 메모리)**: 단일 대화 세션(Thread) 내의 상호작용을 기억
- **Long-term Memory (장기 메모리)**: 여러 세션에 걸쳐 지속되는 정보 저장

```mermaid
graph TD
    A[User Input] --> B[Agent]
    B --> C{Memory Type}
    C -->|Short-term| D[Checkpointer]
    C -->|Long-term| E[Store]
    D --> F[(Thread State)]
    E --> G[(Persistent Data)]
    F --> B
    G --> B
    B --> H[Response]
```

### Part 4에서 다루는 내용

1. **Section 1**: Short-term Memory 기초
2. **Section 2**: Production Checkpointer (PostgreSQL)
3. **Section 3**: Message Management (Trim/Delete)
4. **Section 4**: Message Summarization
5. **Section 5**: Custom State 확장
6. **Section 6**: Long-term Memory (Store)

## Section 1: Short-term Memory 기초

### 1.1 Memory가 필요한 이유

메모리가 없는 Agent는 각 요청을 독립적으로 처리합니다:

```python
# 메모리 없음
agent.invoke({"messages": "제 이름은 김철수입니다."})
agent.invoke({"messages": "제 이름이 뭐였죠?"})
# ❌ "죄송하지만 이름을 말씀하신 적이 없습니다."
```

메모리가 있는 Agent는 이전 대화를 기억합니다:

```python
# 메모리 있음
config = {"configurable": {"thread_id": "user-1"}}
agent.invoke({"messages": "제 이름은 김철수입니다."}, config)
agent.invoke({"messages": "제 이름이 뭐였죠?"}, config)
# ✅ "김철수님이시죠!"
```

### 1.2 Thread와 Conversation

**Thread**는 단일 대화 세션을 구성하는 여러 상호작용을 그룹화합니다. 이메일의 스레드와 유사한 개념입니다.

```mermaid
graph LR
    A[User A<br/>thread-1] --> B[Agent]
    C[User B<br/>thread-2] --> B
    D[User A<br/>thread-1] --> B
    B --> E[Checkpointer]
    E --> F[(Thread 1:<br/>User A 대화)]
    E --> G[(Thread 2:<br/>User B 대화)]
```

각 Thread는 독립적인 대화 이력을 유지합니다:

```python
# User A의 대화
config_a = {"configurable": {"thread_id": "user-a"}}
agent.invoke({"messages": "제 이름은 Alice입니다."}, config_a)

# User B의 대화 (User A의 정보를 모름)
config_b = {"configurable": {"thread_id": "user-b"}}
agent.invoke({"messages": "제 이름은 Bob입니다."}, config_b)

# User A는 자신의 정보를 기억함
agent.invoke({"messages": "제 이름이 뭐죠?"}, config_a)
# ✅ "Alice님이시죠!"
```

### 1.3 Checkpointer 개념

**Checkpointer**는 Agent의 상태를 저장하고 복원하는 시스템입니다.

```mermaid
graph TD
    A[Agent Execution] --> B{Step Complete}
    B --> C[Save State]
    C --> D[Checkpointer]
    D --> E[(Storage)]
    F[Resume] --> G[Load State]
    G --> D
    D --> A
```

Checkpointer의 역할:

1. **State 저장**: 각 단계(step) 완료 후 상태 저장
2. **State 로드**: Agent 시작 시 이전 상태 복원
3. **Thread 관리**: 여러 Thread의 상태를 독립적으로 관리
4. **시간 여행**: 특정 시점의 상태로 되돌아가기 가능

### 1.4 InMemorySaver 사용

`InMemorySaver`는 메모리에 상태를 저장하는 가장 간단한 Checkpointer입니다.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Checkpointer 생성
checkpointer = InMemorySaver()

# Agent 생성 시 checkpointer 전달
agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    checkpointer=checkpointer,
)

# Thread ID를 지정하여 대화
config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "안녕하세요!"}, config)
agent.invoke({"messages": "방금 뭐라고 했죠?"}, config)
```

**장점**:
- 설정이 간단함
- 빠른 응답 시간
- 개발/테스트에 적합

**단점**:
- 프로세스 종료 시 데이터 손실
- 여러 인스턴스 간 공유 불가
- Production 환경에 부적합

**예제 코드**: `01_basic_memory.py`

## Section 2: Production Checkpointer

### 2.1 PostgresSaver

Production 환경에서는 데이터베이스 기반 Checkpointer를 사용해야 합니다.

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/dbname"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 테이블 자동 생성

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        checkpointer=checkpointer,
    )
```

### 2.2 데이터베이스 설정

PostgreSQL 데이터베이스 설정:

```sql
-- 데이터베이스 생성
CREATE DATABASE langchain_memory;

-- 사용자 생성 및 권한 부여
CREATE USER langchain_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE langchain_memory TO langchain_user;
```

Docker를 사용한 빠른 설정:

```bash
docker run -d \
  --name postgres-langchain \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain_memory \
  -p 5432:5432 \
  postgres:15
```

### 2.3 Thread 관리

데이터베이스 기반 Checkpointer는 Thread를 영구적으로 저장합니다:

```python
# Thread 목록 조회
threads = checkpointer.list_threads()
for thread in threads:
    print(f"Thread ID: {thread.thread_id}")
    print(f"Created: {thread.created_at}")
    print(f"Updated: {thread.updated_at}")

# 특정 Thread 삭제
checkpointer.delete_thread(thread_id="user-123")

# 오래된 Thread 정리
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=30)
old_threads = checkpointer.list_threads(before=cutoff)
for thread in old_threads:
    checkpointer.delete_thread(thread.thread_id)
```

**장점**:
- 영구적 데이터 저장
- 서버 재시작 후에도 대화 유지
- 여러 인스턴스 간 상태 공유
- 백업 및 복구 가능

**단점**:
- 추가 인프라 필요
- 설정이 복잡함
- 네트워크 레이턴시

#### 프로덕션 패턴

프로덕션 환경에서 PostgresSaver를 사용할 때 고려해야 할 패턴들입니다.

**1. Connection Pooling**

PostgreSQL 연결을 효율적으로 관리하기 위해 Connection Pool을 사용합니다.

```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

# Connection Pool 생성
pool = ConnectionPool(
    conninfo="postgresql://user:password@localhost/db",
    min_size=2,      # 최소 연결 수
    max_size=10,     # 최대 연결 수
    timeout=30,      # 연결 대기 시간 (초)
    max_idle=300,    # 유휴 연결 유지 시간 (초)
    max_lifetime=3600  # 연결 최대 수명 (초)
)

# PostgresSaver에 Pool 전달
checkpointer = PostgresSaver(pool)

# Agent 생성
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer
)

# 사용 후 정리
pool.close()
```

**왜 Connection Pool이 필요한가?**
- 연결 생성/해제 오버헤드 감소
- 동시 요청 처리 성능 향상
- 데이터베이스 리소스 효율적 사용

**2. Error Handling & Retry**

네트워크 오류, 데이터베이스 장애 시 재시도 로직을 구현합니다.

```python
import time
from psycopg import OperationalError

def create_checkpointer_with_retry(
    conninfo: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> PostgresSaver:
    """재시도 로직이 있는 Checkpointer 생성"""
    for attempt in range(max_retries):
        try:
            pool = ConnectionPool(conninfo=conninfo)
            checkpointer = PostgresSaver(pool)

            # 연결 테스트
            checkpointer.list_threads(limit=1)

            return checkpointer

        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"연결 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                raise Exception(f"Checkpointer 생성 실패: {e}")

# 사용
checkpointer = create_checkpointer_with_retry(
    "postgresql://user:password@localhost/db"
)
```

**3. Health Check**

Checkpointer 연결 상태를 주기적으로 확인합니다.

```python
def check_checkpointer_health(checkpointer: PostgresSaver) -> bool:
    """Checkpointer 연결 상태 확인"""
    try:
        # 간단한 쿼리로 연결 테스트
        checkpointer.list_threads(limit=1)
        return True
    except Exception as e:
        print(f"Checkpointer health check failed: {e}")
        return False

# 사용 예시 (FastAPI)
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    if check_checkpointer_health(checkpointer):
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy"}, 503
```

**4. Thread Lifecycle Management**

Thread를 효과적으로 관리하기 위한 패턴들입니다.

```python
from datetime import datetime, timedelta
from typing import Optional

class ThreadManager:
    """Thread 생명주기 관리"""

    def __init__(self, checkpointer: PostgresSaver):
        self.checkpointer = checkpointer

    def get_or_create_thread(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> str:
        """사용자별 Thread 조회 또는 생성"""
        thread_id = f"{user_id}:{session_id or 'default'}"

        # Thread 존재 여부 확인
        try:
            threads = self.checkpointer.list_threads(
                filter={"thread_id": thread_id},
                limit=1
            )
            if threads:
                return thread_id
        except:
            pass

        # 새 Thread 생성 (첫 invoke 시 자동 생성됨)
        return thread_id

    def cleanup_old_threads(self, days: int = 30) -> int:
        """오래된 Thread 정리"""
        cutoff = datetime.now() - timedelta(days=days)
        deleted_count = 0

        try:
            old_threads = self.checkpointer.list_threads(before=cutoff)

            for thread in old_threads:
                self.checkpointer.delete_thread(thread.thread_id)
                deleted_count += 1

            return deleted_count

        except Exception as e:
            print(f"Thread cleanup failed: {e}")
            return deleted_count

    def archive_thread(self, thread_id: str) -> bool:
        """Thread를 아카이브 (실제로는 메타데이터 업데이트)"""
        # PostgresSaver는 아카이브 기능이 없으므로
        # 커스텀 메타데이터로 구현
        try:
            # 마지막 checkpoint에 아카이브 플래그 추가
            # (구현 예시)
            return True
        except Exception as e:
            print(f"Archive failed: {e}")
            return False

# 사용
thread_manager = ThreadManager(checkpointer)

# 사용자별 Thread 관리
thread_id = thread_manager.get_or_create_thread(
    user_id="user-123",
    session_id="chat-001"
)

# 주기적 정리 (Cron job 등)
deleted = thread_manager.cleanup_old_threads(days=30)
print(f"Deleted {deleted} old threads")
```

**5. Performance Tuning**

대용량 트래픽 처리를 위한 최적화 패턴입니다.

```python
# 인덱스 추가 (PostgreSQL)
import psycopg

conn = psycopg.connect("postgresql://user:password@localhost/db")

with conn.cursor() as cur:
    # Thread ID 인덱스
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_thread_id
        ON checkpoints (thread_id);
    """)

    # Timestamp 인덱스 (정리 쿼리 최적화)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at
        ON checkpoints (created_at);
    """)

    conn.commit()
```

**6. Graceful Shutdown**

서버 종료 시 Connection Pool을 안전하게 닫습니다.

```python
import signal
import sys

def shutdown_handler(signum, frame):
    """서버 종료 시 정리 작업"""
    print("Shutting down gracefully...")

    # Connection Pool 닫기
    if 'pool' in globals():
        pool.close()
        print("Connection pool closed")

    sys.exit(0)

# Signal handler 등록
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
```

**7. Monitoring & Logging**

Checkpointer 성능을 모니터링합니다.

```python
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def monitor_checkpoint(func):
    """Checkpoint 연산 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            logger.info(f"{func.__name__} completed in {duration:.2f}s")

            # 느린 쿼리 경고
            if duration > 1.0:
                logger.warning(f"Slow checkpoint operation: {func.__name__}")

            return result

        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise

    return wrapper

# Agent invoke 모니터링
@monitor_checkpoint
def invoke_agent(agent, input_data, config):
    return agent.invoke(input_data, config=config)

# 사용
result = invoke_agent(
    agent,
    {"messages": [{"role": "user", "content": "Hello"}]},
    {"configurable": {"thread_id": "thread-123"}}
)
```

**프로덕션 체크리스트**:
- [x] Connection Pool 설정 (min/max size)
- [x] Retry 로직 구현
- [x] Health check 엔드포인트
- [x] Thread cleanup 스케줄러
- [x] 데이터베이스 인덱스 추가
- [x] Graceful shutdown 구현
- [x] 로깅 및 모니터링 설정
- [ ] 백업 전략 수립
- [ ] 장애 복구 테스트

**예제 코드**: `02_postgres_memory.py`

## Section 3: Message Management

### 3.1 Context Window 문제

대부분의 LLM은 제한된 Context Window를 가집니다:

| Model | Context Window | 비고 |
|-------|---------------|------|
| GPT-4o-mini | 128K tokens | ~96,000 단어 |
| GPT-4o | 128K tokens | ~96,000 단어 |
| Claude Sonnet 4 | 200K tokens | ~150,000 단어 |

긴 대화는 다음 문제를 야기합니다:

1. **Context Loss**: Context window 초과 시 오류 발생
2. **Performance Degradation**: 긴 컨텍스트에서 LLM 성능 저하
3. **높은 비용**: 토큰 수에 비례하는 API 비용
4. **느린 응답**: 처리 시간 증가

```mermaid
graph LR
    A[Message 1] --> B[Message 2]
    B --> C[...]
    C --> D[Message N]
    D --> E{Context<br/>Window}
    E -->|Overflow| F[❌ Error]
    E -->|Full| G[⚠️ Slow]
    E -->|OK| H[✅ Fast]
```

### 3.2 Trim Messages

**Trim Messages**는 LLM 호출 전에 메시지를 제거하는 전략입니다.

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime: Runtime):
    """마지막 N개의 메시지만 유지"""
    messages = state["messages"]

    if len(messages) <= 5:
        return None  # 변경 불필요

    # 첫 메시지(시스템 메시지) + 최근 4개 메시지
    first_msg = messages[0]
    recent_messages = messages[-4:]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg,
            *recent_messages
        ]
    }

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```

**Trim 전략**:

1. **Keep Last N**: 최근 N개 메시지만 유지
2. **Keep First + Last N**: 시스템 메시지 + 최근 N개
3. **Token-based**: 토큰 수 기준으로 제거
4. **Time-based**: 시간 기준으로 오래된 메시지 제거

### 3.3 Delete Messages

**Delete Messages**는 메시지를 영구적으로 제거합니다.

```python
from langchain.agents.middleware import after_model

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime):
    """오래된 메시지 영구 삭제"""
    messages = state["messages"]

    if len(messages) > 10:
        # 가장 오래된 2개 메시지 삭제
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

    return None
```

**Trim vs Delete**:

| | Trim | Delete |
|---|------|--------|
| **범위** | 현재 호출만 | 영구적 |
| **저장소** | Checkpointer에는 유지 | Checkpointer에서도 제거 |
| **복구** | 가능 (다음 호출에서) | 불가능 |
| **사용 시점** | @before_model | @after_model |

### 3.4 Message Filtering

특정 조건에 따라 메시지를 필터링할 수 있습니다:

```python
@before_model
def filter_messages(state: AgentState, runtime: Runtime):
    """특정 조건의 메시지만 유지"""
    messages = state["messages"]

    # 시스템 메시지와 최근 대화만 유지
    filtered = [
        msg for msg in messages
        if msg.type == "system" or is_recent(msg)
    ]

    # Tool 메시지 제거 (결과만 남기기)
    filtered = [
        msg for msg in filtered
        if msg.type != "tool"
    ]

    return {"messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *filtered
    ]}
```

**예제 코드**: `03_message_trim.py`

## Section 4: Message Summarization

### 4.1 Summarization 전략

메시지를 제거하는 대신 요약하여 정보 손실을 최소화할 수 있습니다.

```mermaid
graph TD
    A[메시지 1-10] --> B{Summarize}
    B --> C[요약 메시지]
    D[메시지 11-20] --> E[최근 메시지]
    C --> F[LLM]
    E --> F
    F --> G[응답]
```

**Summarization의 장점**:

1. **정보 보존**: 중요한 컨텍스트 유지
2. **Context Window 절약**: 요약본은 훨씬 짧음
3. **성능 유지**: LLM이 필요한 정보에 집중
4. **비용 절감**: 토큰 수 감소

### 4.2 before_model 미들웨어

사용자 정의 Summarization 로직:

```python
from langchain.agents.middleware import before_model
from langchain_core.messages import SystemMessage

@before_model
def summarize_old_messages(state: AgentState, runtime: Runtime):
    """오래된 메시지를 요약"""
    messages = state["messages"]

    if len(messages) < 10:
        return None

    # 오래된 메시지 (처음 6개)
    old_messages = messages[1:7]  # 시스템 메시지 제외
    recent_messages = messages[7:]

    # LLM으로 요약 생성
    summarizer = ChatOpenAI(model="gpt-4o-mini")
    summary_prompt = f"""
    다음 대화를 간단히 요약해주세요:

    {format_messages(old_messages)}

    핵심 정보만 포함하여 2-3문장으로 요약:
    """

    summary = summarizer.invoke([HumanMessage(content=summary_prompt)])

    # 요약 + 최근 메시지로 재구성
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            messages[0],  # 시스템 메시지
            SystemMessage(content=f"[이전 대화 요약]: {summary.content}"),
            *recent_messages
        ]
    }
```

### 4.3 Summary 생성

LangChain은 내장 `SummarizationMiddleware`를 제공합니다:

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",  # 요약에 사용할 모델
            trigger=("tokens", 4000),  # 4000 토큰 초과 시 요약
            keep=("messages", 20),  # 최근 20개 메시지 유지
        )
    ],
    checkpointer=InMemorySaver(),
)
```

**SummarizationMiddleware 옵션**:

| 파라미터 | 설명 | 예시 |
|---------|------|------|
| `model` | 요약에 사용할 LLM | `"gpt-4o-mini"` |
| `trigger` | 요약 트리거 조건 | `("tokens", 4000)` |
| `keep` | 유지할 메시지 수 | `("messages", 20)` |
| `summarize_system` | 시스템 메시지 포함 여부 | `False` |

**Summarization 전략**:

1. **Rolling Summary**: 지속적으로 요약 업데이트
2. **Periodic Summary**: 일정 간격으로 요약
3. **Hierarchical Summary**: 계층적 요약 (요약의 요약)
4. **Selective Summary**: 중요한 메시지만 요약

**예제 코드**: `04_summarization.py`

## Section 5: Custom State

### 5.1 AgentState 확장

기본 `AgentState`는 `messages` 필드만 포함합니다. 추가 정보를 저장하려면 확장해야 합니다.

```python
from langchain.agents import AgentState

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict
    session_start: str
```

### 5.2 Custom Fields 추가

다양한 사용 사례에 맞는 필드 추가:

```python
from typing import Optional, List
from datetime import datetime

class EnhancedAgentState(AgentState):
    # 사용자 정보
    user_id: str
    user_name: Optional[str] = None
    user_email: Optional[str] = None

    # 세션 정보
    session_id: str
    session_start: datetime
    session_end: Optional[datetime] = None

    # 사용자 선호도
    language: str = "ko"
    timezone: str = "Asia/Seoul"
    theme: str = "light"

    # 메타데이터
    request_count: int = 0
    error_count: int = 0
    tags: List[str] = []

    # 비즈니스 로직
    subscription_tier: str = "free"
    credits_remaining: int = 100
```

### 5.3 state_schema 사용

Custom State를 Agent에 적용:

```python
from langchain.agents import create_agent

class UserAgentState(AgentState):
    user_id: str
    preferences: dict = {}

agent = create_agent(
    model="gpt-4o-mini",
    tools=[],
    state_schema=UserAgentState,
    checkpointer=InMemorySaver(),
)

# Custom State로 호출
result = agent.invoke(
    {
        "messages": "안녕하세요!",
        "user_id": "user-123",
        "preferences": {"language": "ko", "theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)

# State 접근
print(result["user_id"])  # user-123
print(result["preferences"])  # {'language': 'ko', 'theme': 'dark'}
```

### Custom State 활용 예시

**1. Tool에서 State 읽기**:

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_settings(runtime: ToolRuntime[None, UserAgentState]) -> str:
    """사용자 설정 조회"""
    preferences = runtime.state["preferences"]
    return f"Language: {preferences.get('language', 'en')}"
```

**2. Middleware에서 State 수정**:

```python
from langchain.agents.middleware import before_model

@before_model
def track_requests(state: UserAgentState, runtime: Runtime):
    """요청 수 카운트"""
    return {"request_count": state.get("request_count", 0) + 1}
```

**3. 동적 System Prompt**:

```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def personalized_prompt(request: ModelRequest):
    preferences = request.runtime.state["preferences"]
    language = preferences.get("language", "en")

    if language == "ko":
        return "당신은 친절한 한국어 AI 어시스턴트입니다."
    else:
        return "You are a helpful AI assistant."
```

**예제 코드**: `05_custom_state.py`

## Section 6: Long-term Memory

### 6.1 Store 개념

**Store**는 여러 Thread에 걸쳐 지속되는 정보를 저장하는 시스템입니다.

```mermaid
graph TD
    A[Thread 1] --> B[Agent]
    C[Thread 2] --> B
    D[Thread 3] --> B
    B --> E[Checkpointer]
    B --> F[Store]
    E --> G[(Thread-level<br/>Short-term)]
    F --> H[(Cross-thread<br/>Long-term)]
```

**Checkpointer vs Store**:

| | Checkpointer | Store |
|---|--------------|-------|
| **범위** | 단일 Thread | 여러 Thread |
| **수명** | 대화 세션 동안 | 영구적 |
| **데이터** | 대화 이력, 상태 | 사용자 정보, 선호도 |
| **예시** | 현재 대화 내용 | 사용자 프로필, 학습 이력 |

### 6.2 InMemoryStore

기본 In-Memory Store 사용:

```python
from langgraph.store.memory import InMemoryStore

# Store 생성
store = InMemoryStore()

# 데이터 저장
store.put(
    namespace=("users",),  # 네임스페이스
    key="user-123",        # 키
    value={                # 값
        "name": "김철수",
        "email": "kim@example.com",
        "language": "ko"
    }
)

# 데이터 조회
user_info = store.get(("users",), "user-123")
print(user_info.value)  # {'name': '김철수', ...}
```

### 6.3 Namespace와 Key

**Namespace**는 데이터를 계층적으로 구성하는 방법입니다.

```python
# 사용자 정보
store.put(("users",), "user-123", {"name": "김철수"})

# 사용자별 선호도
store.put(("users", "user-123", "preferences"), "theme", {"value": "dark"})

# 애플리케이션 설정
store.put(("app", "settings"), "version", {"value": "1.0.0"})

# 대화 메타데이터
store.put(("conversations", "user-123"), "stats", {
    "total_messages": 150,
    "avg_length": 42
})
```

**Namespace 설계 패턴**:

```python
# 패턴 1: 엔티티별 구분
("users", user_id)
("products", product_id)
("orders", order_id)

# 패턴 2: 애플리케이션 컨텍스트
(user_id, "chat")
(user_id, "email")
(user_id, "support")

# 패턴 3: 계층적 구조
("org", org_id, "team", team_id, "user", user_id)
```

### 6.4 Tools에서 Store 접근

Tool에서 Long-term Memory 읽기/쓰기:

```python
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

@tool
def get_user_preferences(runtime: ToolRuntime[Context]) -> str:
    """사용자 선호도 조회"""
    store = runtime.store
    user_id = runtime.context.user_id

    # Store에서 데이터 조회
    prefs = store.get(("users", user_id, "preferences"), "settings")

    if prefs:
        return f"사용자 선호도: {prefs.value}"
    else:
        return "저장된 선호도가 없습니다."

@tool
def save_user_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[Context]
) -> str:
    """사용자 선호도 저장"""
    store = runtime.store
    user_id = runtime.context.user_id

    # Store에 데이터 저장
    store.put(
        namespace=("users", user_id, "preferences"),
        key=preference_key,
        value={"value": preference_value}
    )

    return f"선호도 저장 완료: {preference_key} = {preference_value}"

# Agent 생성
agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_user_preferences, save_user_preference],
    store=store,
    context_schema=Context,
)

# 사용
agent.invoke(
    {"messages": "제 테마 설정을 다크모드로 저장해주세요."},
    context=Context(user_id="user-123")
)
```

### Store 검색 기능

Namespace 내에서 데이터 검색:

```python
# 특정 namespace의 모든 항목
items = store.search(
    namespace=("users",),
    filter={"language": "ko"},
    limit=10
)

for item in items:
    print(f"Key: {item.key}, Value: {item.value}")

# 벡터 검색 (embedding 지원 시)
def embed(texts: list[str]) -> list[list[float]]:
    # 실제 embedding 함수 사용
    return [[1.0, 2.0] * len(texts)]

store_with_embedding = InMemoryStore(
    index={"embed": embed, "dims": 2}
)

# 유사도 기반 검색
similar_items = store_with_embedding.search(
    namespace=("users",),
    query="dark theme preference",
    limit=5
)
```

**예제 코드**: `06_long_term_store.py`

## Memory 시스템 설계 패턴

### 패턴 1: Hybrid Memory

Short-term과 Long-term을 결합:

```python
class HybridMemoryAgent:
    def __init__(self):
        self.checkpointer = PostgresSaver.from_conn_string(DB_URI)
        self.store = InMemoryStore()

        self.agent = create_agent(
            model="gpt-4o-mini",
            tools=[self.load_user_context],
            checkpointer=self.checkpointer,
            store=self.store,
        )

    @tool
    def load_user_context(self, runtime: ToolRuntime):
        """Long-term memory를 Short-term으로 로드"""
        user_id = runtime.context.user_id

        # Long-term에서 사용자 정보 조회
        user_info = self.store.get(("users",), user_id)

        if user_info:
            # Short-term에 추가
            return Command(update={
                "user_name": user_info.value["name"],
                "user_preferences": user_info.value["preferences"]
            })
```

### 패턴 2: Tiered Memory

계층적 메모리 시스템:

```mermaid
graph TD
    A[User Request] --> B[L1: Session Cache]
    B -->|Miss| C[L2: Thread State]
    C -->|Miss| D[L3: Store]
    D --> E[(Database)]
    D --> F[Response]
    C --> F
    B --> F
```

```python
class TieredMemory:
    def __init__(self):
        self.l1_cache = {}  # 세션 캐시
        self.l2_checkpointer = InMemorySaver()  # Thread 상태
        self.l3_store = InMemoryStore()  # 영구 저장소

    def get(self, user_id: str, key: str):
        # L1: 캐시 확인
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2: Thread 상태 확인
        # (구현 생략)

        # L3: Store 확인
        value = self.l3_store.get(("users", user_id), key)

        # 캐시에 저장
        if value:
            self.l1_cache[key] = value.value

        return value
```

### 패턴 3: Event-sourced Memory

이벤트 기반 메모리 관리:

```python
from datetime import datetime

class Event:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data
        self.timestamp = datetime.now()

class EventSourcedMemory:
    def __init__(self, store: InMemoryStore):
        self.store = store

    def record_event(self, user_id: str, event: Event):
        """이벤트 기록"""
        events = self.get_events(user_id)
        events.append({
            "type": event.type,
            "data": event.data,
            "timestamp": event.timestamp.isoformat()
        })

        self.store.put(
            namespace=("users", user_id, "events"),
            key="timeline",
            value={"events": events}
        )

    def get_events(self, user_id: str) -> list:
        """이벤트 히스토리 조회"""
        result = self.store.get(
            namespace=("users", user_id, "events"),
            key="timeline"
        )
        return result.value["events"] if result else []

    def reconstruct_state(self, user_id: str) -> dict:
        """이벤트로부터 현재 상태 재구성"""
        events = self.get_events(user_id)
        state = {}

        for event in events:
            if event["type"] == "preference_updated":
                state["preferences"] = event["data"]
            elif event["type"] == "profile_updated":
                state["profile"] = event["data"]

        return state
```

## 실전 팁

### 1. Checkpointer 선택

**개발 단계**:
- `InMemorySaver`: 빠른 프로토타이핑
- 간단한 테스트

**Production**:
- `PostgresSaver`: 신뢰성 있는 영구 저장
- `SQLiteSaver`: 단일 인스턴스, 파일 기반
- Custom Checkpointer: 특수 요구사항

### 2. Message 관리 전략

**짧은 대화 (<10 메시지)**:
- 관리 불필요
- 모든 메시지 유지

**중간 대화 (10-50 메시지)**:
- Trim messages 사용
- 최근 메시지 + 시스템 메시지

**긴 대화 (50+ 메시지)**:
- Summarization 사용
- 요약 + 최근 메시지

**매우 긴 대화 (100+ 메시지)**:
- Hierarchical summarization
- 주기적 요약 업데이트

### 3. Store 네이밍 규칙

```python
# 좋은 예
("users", user_id)
("users", user_id, "preferences")
("conversations", thread_id, "metadata")

# 나쁜 예
("data",)  # 너무 일반적
("u", user_id)  # 약어 사용
(user_id,)  # 엔티티 타입 누락
```

### 4. 성능 최적화

**Checkpointer**:
```python
# 연결 풀 사용
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DB_URI,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)

checkpointer = PostgresSaver(engine)
```

**Store**:
```python
# 배치 작업
items = [
    (namespace, f"key-{i}", {"data": i})
    for i in range(100)
]

# 일괄 저장 (구현에 따라 다름)
for namespace, key, value in items:
    store.put(namespace, key, value)
```

### 5. 에러 처리

```python
from langgraph.errors import CheckpointError

@before_model
def safe_trim(state: AgentState, runtime: Runtime):
    try:
        # 메시지 처리
        return trim_logic(state)
    except Exception as e:
        # 로깅
        logger.error(f"Trim failed: {e}")
        # 실패 시 원본 반환
        return None
```

### 6. 모니터링

```python
from datetime import datetime

class MonitoredAgent:
    def __init__(self, agent):
        self.agent = agent
        self.metrics = {
            "total_invocations": 0,
            "total_messages": 0,
            "avg_message_length": 0
        }

    def invoke(self, input, config):
        start_time = datetime.now()

        result = self.agent.invoke(input, config)

        # 메트릭 업데이트
        self.metrics["total_invocations"] += 1
        self.metrics["total_messages"] += len(result["messages"])

        duration = (datetime.now() - start_time).total_seconds()
        print(f"Invocation took {duration:.2f}s")

        return result
```

## FAQ

### Q1: InMemorySaver를 Production에서 사용해도 되나요?

**A**: 권장하지 않습니다. InMemorySaver는 다음과 같은 제약이 있습니다:
- 서버 재시작 시 데이터 손실
- 여러 인스턴스 간 상태 공유 불가
- 메모리 제한

Production에서는 PostgresSaver 또는 다른 영구 저장소를 사용하세요.

### Q2: Thread를 언제 삭제해야 하나요?

**A**: 다음과 같은 경우에 고려하세요:
- 사용자가 대화 삭제 요청
- 일정 기간 (예: 30일) 비활성 Thread
- 저장 공간 제약
- 개인정보 보호 정책

```python
# 오래된 Thread 정리
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=30)
old_threads = checkpointer.list_threads(before=cutoff)
for thread in old_threads:
    checkpointer.delete_thread(thread.thread_id)
```

### Q3: Summarization과 Trim 중 어떤 것을 사용해야 하나요?

**A**: 상황에 따라 다릅니다:

**Summarization 사용**:
- 긴 대화에서 컨텍스트가 중요한 경우
- 비용/성능보다 품질이 중요한 경우
- 복잡한 문제 해결 Agent

**Trim 사용**:
- 간단한 질문-답변 Agent
- 비용/성능이 중요한 경우
- 최근 대화만 중요한 경우

**둘 다 사용**:
- Summarization으로 오래된 메시지 처리
- Trim으로 최대 길이 제한

### Q4: Custom State와 Store의 차이는 무엇인가요?

**A**:
- **Custom State**: Thread 내에서 유지되는 정보 (Short-term)
- **Store**: Thread 간에 공유되는 정보 (Long-term)

예시:
```python
# Custom State: 현재 대화의 임시 데이터
class MyAgentState(AgentState):
    current_query: str  # 현재 쿼리
    temp_results: list  # 임시 결과

# Store: 영구적인 사용자 데이터
store.put(("users",), user_id, {
    "name": "김철수",
    "preferences": {"language": "ko"}
})
```

### Q5: Message ID를 어떻게 얻나요?

**A**: 모든 메시지는 자동으로 `id` 속성을 가집니다:

```python
messages = state["messages"]
for msg in messages:
    print(f"ID: {msg.id}, Type: {msg.type}, Content: {msg.content}")

# 특정 메시지 삭제
msg_to_delete = messages[2]
return {"messages": [RemoveMessage(id=msg_to_delete.id)]}
```

### Q6: Store의 데이터는 언제 삭제해야 하나요?

**A**: 비즈니스 로직과 개인정보 보호 정책에 따라:

```python
# 사용자 요청 시
store.delete(namespace=("users",), key=user_id)

# GDPR 준수: 사용자 데이터 완전 삭제
def delete_user_data(user_id: str):
    # 모든 관련 namespace 삭제
    store.delete(("users",), user_id)
    store.delete(("users", user_id, "preferences"), "*")
    store.delete(("conversations", user_id), "*")
```

## 심화 학습

### 1. Custom Checkpointer 구현

자신만의 Checkpointer를 구현할 수 있습니다:

```python
from langgraph.checkpoint import BaseCheckpointSaver

class RedisCheckpointer(BaseCheckpointSaver):
    def __init__(self, redis_client):
        self.redis = redis_client

    def put(self, config, checkpoint, metadata):
        """체크포인트 저장"""
        thread_id = config["configurable"]["thread_id"]
        key = f"checkpoint:{thread_id}"
        self.redis.set(key, pickle.dumps(checkpoint))

    def get(self, config):
        """체크포인트 로드"""
        thread_id = config["configurable"]["thread_id"]
        key = f"checkpoint:{thread_id}"
        data = self.redis.get(key)
        return pickle.loads(data) if data else None
```

### 2. Vector Store 통합

Embedding 기반 검색:

```python
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore

embeddings = OpenAIEmbeddings()

def embed_function(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)

store = InMemoryStore(
    index={"embed": embed_function, "dims": 1536}
)

# 벡터 검색
results = store.search(
    namespace=("documents",),
    query="LangChain memory system",
    limit=5
)
```

### 3. 분산 메모리 시스템

여러 인스턴스 간 메모리 공유:

```python
import redis
from langgraph.checkpoint.postgres import PostgresSaver

class DistributedMemoryAgent:
    def __init__(self, db_uri: str, redis_host: str):
        # PostgreSQL: 영구 저장
        self.checkpointer = PostgresSaver.from_conn_string(db_uri)

        # Redis: 캐시 레이어
        self.cache = redis.Redis(host=redis_host)

    def get_state(self, thread_id: str):
        # 캐시 확인
        cached = self.cache.get(f"state:{thread_id}")
        if cached:
            return pickle.loads(cached)

        # DB에서 로드
        config = {"configurable": {"thread_id": thread_id}}
        state = self.checkpointer.get(config)

        # 캐시에 저장
        if state:
            self.cache.setex(
                f"state:{thread_id}",
                3600,  # 1시간 TTL
                pickle.dumps(state)
            )

        return state
```

### 4. Memory Analytics

메모리 사용 분석:

```python
class MemoryAnalytics:
    def __init__(self, checkpointer, store):
        self.checkpointer = checkpointer
        self.store = store

    def analyze_thread(self, thread_id: str):
        """Thread 분석"""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.checkpointer.get(config)

        messages = state["messages"]

        return {
            "total_messages": len(messages),
            "user_messages": sum(1 for m in messages if m.type == "human"),
            "ai_messages": sum(1 for m in messages if m.type == "ai"),
            "tool_calls": sum(1 for m in messages if m.type == "tool"),
            "total_tokens": estimate_tokens(messages),
            "avg_message_length": sum(len(m.content) for m in messages) / len(messages)
        }

    def get_top_users(self, limit=10):
        """가장 활발한 사용자"""
        # Store에서 사용자별 통계 조회
        users = self.store.search(
            namespace=("users",),
            filter={"active": True},
            limit=limit
        )

        return sorted(
            users,
            key=lambda u: u.value.get("message_count", 0),
            reverse=True
        )
```

### 5. 공식 문서 심화

- [10-short-term-memory.md](../official/10-short-term-memory_ko.md) - Checkpointer 고급 기능
- [29-long-term-memory.md](../official/29-long-term-memory_ko.md) - Store 심화

### 6. 관련 논문

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) - LLM 메모리 관리
- [Generative Agents](https://arxiv.org/abs/2304.03442) - 장기 메모리 활용

---

## 실습 과제

### 과제 1: 세션 기반 챗봇 (⭐⭐)

**목표**: InMemorySaver를 사용하여 여러 사용자의 독립적인 대화를 관리하는 세션 기반 챗봇을 만드세요.

**요구사항**:
1. LangGraph의 StateGraph와 InMemorySaver(checkpointer)를 사용하여 챗봇 구축
2. 여러 사용자(Alice, Bob 등)가 각자의 thread_id로 독립적인 대화 진행
3. 세션별 대화 기록 저장 및 조회 기능
4. 세션 간 격리 확인 (한 사용자의 정보가 다른 사용자에게 노출되지 않음)

**힌트**:
```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 사용자별 세션
config_alice = {"configurable": {"thread_id": "alice_session"}}
config_bob = {"configurable": {"thread_id": "bob_session"}}
```

**해답**: [solutions/exercise_01.py](../src/part04_memory/solutions/exercise_01.py)

---

### 과제 2: 자동 요약 시스템 (⭐⭐⭐)

**목표**: 대화가 길어지면 자동으로 요약하는 LangGraph 기반 챗봇을 구현하세요.

**요구사항**:
1. ChatState에 summary 필드를 추가한 Custom State 정의
2. 메시지 수 기반 요약 트리거 (threshold 초과 시 요약 노드 실행)
3. RemoveMessage로 오래된 메시지를 정리하고 요약본으로 대체
4. 요약 전후의 메시지 수 변화 확인

**해답**: [solutions/exercise_02.py](../src/part04_memory/solutions/exercise_02.py)

---

### 과제 3: 사용자 프로필 시스템 (⭐⭐⭐)

**목표**: InMemoryStore로 사용자별 프로필 정보를 저장하고, 대화에서 자동으로 추출하여 개인화 응답을 제공하는 Agent를 구현하세요.

**요구사항**:
1. InMemoryStore에 사용자 프로필 저장 (이름, 선호도, 관심사)
2. 대화에서 사용자 정보를 자동 추출하여 프로필 업데이트
3. 프로필 기반 개인화 응답 생성
4. Namespace: ("users", user_id) 하위에 구조화된 데이터 관리

**해답**: [solutions/exercise_03.py](../src/part04_memory/solutions/exercise_03.py)

---

## 체크리스트

Part 4를 완료했다면 다음을 할 수 있어야 합니다:

- [ ] InMemorySaver로 기본 메모리 구현
- [ ] PostgresSaver로 Production 메모리 구현
- [ ] Thread 개념 이해 및 활용
- [ ] before_model로 메시지 Trim 구현
- [ ] after_model로 메시지 Delete 구현
- [ ] SummarizationMiddleware 사용
- [ ] Custom State 정의 및 사용
- [ ] state_schema 파라미터 활용
- [ ] InMemoryStore로 장기 메모리 구현
- [ ] Namespace 설계 및 구성
- [ ] Tool에서 Store 접근
- [ ] ToolRuntime 사용
- [ ] Checkpointer와 Store의 차이 이해
- [ ] 메모리 최적화 전략 수립

---

## 다음 단계

Part 4 완료!
[Part 5: Middleware로 이동](./part05_middleware.md)

**학습 진도**: ▓▓▓▓░░░░░░ 40% (Part 4/10 완료)

---

*마지막 업데이트: 2026-02-18*
