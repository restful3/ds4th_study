# Chapter 13: 영속성 (Persistence)

> 📌 **학습 목표**: 이 장을 마치면 Checkpointer를 사용하여 그래프 상태를 저장/복원하고, 다양한 저장소를 활용할 수 있습니다.

## 개요

**영속성(Persistence)**은 그래프 실행 상태를 저장하고 복원하는 기능입니다. 이를 통해 장기 실행 작업, 장애 복구, Human-in-the-Loop 등을 구현할 수 있습니다.

```mermaid
graph TD
    subgraph "Thread 1"
        C1[Checkpoint 1] --> C2[Checkpoint 2]
        C2 --> C3[Checkpoint 3]
    end
    C2 -.->|Fork| C4[Checkpoint 2']
    subgraph "Thread 2"
        C4 --> C5[Checkpoint 3']
    end
```

## 핵심 개념

### Checkpointer란?

| 구성요소 | 설명 |
|---------|------|
| **Thread** | 대화/실행 단위 식별자 |
| **Checkpoint** | 특정 시점의 전체 상태 스냅샷 |
| **Super-step** | 각 실행 단계마다 자동 저장 |

### 저장되는 정보

- **config**: 설정 정보 (thread_id, checkpoint_id 등)
- **metadata**: 메타데이터 (생성 시간, step 번호 등)
- **values**: 실제 State 값
- **next**: 다음 실행할 노드들
- **tasks**: 대기 중인 작업들

## 실습 1: MemorySaver (개발용)

```python
# 📁 src/part4_production/13_checkpointer.py
from typing import Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# 그래프 정의
def echo_node(state: MessagesState) -> MessagesState:
    last_message = state["messages"][-1]
    return {"messages": [f"Echo: {last_message.content}"]}


graph = StateGraph(MessagesState)
graph.add_node("echo", echo_node)
graph.add_edge(START, "echo")
graph.add_edge("echo", END)

# Checkpointer 추가
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Thread ID로 실행
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke({"messages": ["안녕하세요"]}, config=config)

# 같은 thread_id로 이어서 대화
result = app.invoke({"messages": ["다음 메시지"]}, config=config)
```

> 💡 **전체 코드**: [src/part4_production/13_checkpointer.py](../../src/part4_production/13_checkpointer.py)

## 실습 2: 상태 조회 및 히스토리

```python
# 현재 상태 조회
current_state = app.get_state(config)
print(f"현재 상태: {current_state.values}")
print(f"다음 노드: {current_state.next}")

# 전체 히스토리 조회
for state in app.get_state_history(config):
    print(f"Step {state.metadata.get('step', 0)}: {state.values}")
```

### StateSnapshot 구조

```python
# StateSnapshot의 주요 속성
state = app.get_state(config)

state.values      # 현재 State 값
state.next        # 다음 실행할 노드 튜플
state.config      # 현재 설정
state.metadata    # 메타데이터 딕셔너리
state.tasks       # 대기 중인 작업들
state.parent_config  # 이전 체크포인트 설정
```

## 실습 3: 상태 업데이트

Reducer를 적용하여 상태를 업데이트합니다.

```python
from langchain_core.messages import HumanMessage


# 상태 업데이트 (Reducer 적용)
app.update_state(
    config,
    {"messages": [HumanMessage(content="수동으로 추가된 메시지")]},
    as_node="echo"  # 이 노드에서 업데이트된 것처럼 처리
)

# 업데이트 후 상태 확인
updated_state = app.get_state(config)
print(updated_state.values)
```

## 실습 4: SqliteSaver (로컬 저장)

파일 기반의 영구 저장소입니다.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


# 메모리 DB (테스트용)
conn = sqlite3.connect(":memory:", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 파일 DB (영구 저장)
# conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn)

app = graph.compile(checkpointer=checkpointer)
```

### 비동기 버전

```python
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def main():
    async with aiosqlite.connect(":memory:") as conn:
        checkpointer = AsyncSqliteSaver(conn)
        app = graph.compile(checkpointer=checkpointer)

        result = await app.ainvoke(
            {"messages": ["비동기 메시지"]},
            config={"configurable": {"thread_id": "async-thread"}}
        )
```

## 실습 5: PostgresSaver (프로덕션)

프로덕션 환경을 위한 PostgreSQL 저장소입니다.

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool


# 연결 풀 생성
DB_URI = "postgresql://user:password@localhost:5432/langgraph"
pool = ConnectionPool(conninfo=DB_URI)

# Checkpointer 생성 및 테이블 초기화
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # 필요한 테이블 자동 생성

app = graph.compile(checkpointer=checkpointer)
```

### 비동기 버전

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


async def main():
    pool = AsyncConnectionPool(conninfo=DB_URI)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    app = graph.compile(checkpointer=checkpointer)
```

## 실습 6: 특정 체크포인트로 복원

```python
# 히스토리에서 특정 체크포인트 ID 찾기
history = list(app.get_state_history(config))
checkpoint_to_restore = history[2]  # 3번째 체크포인트

# 해당 체크포인트에서 재시작
restore_config = checkpoint_to_restore.config
result = app.invoke({"messages": ["복원 후 메시지"]}, config=restore_config)
```

## 고급 패턴: Fork (분기)

기존 상태에서 분기하여 새로운 실행 경로를 만듭니다.

```python
# Thread 1에서 실행
config1 = {"configurable": {"thread_id": "original"}}
app.invoke({"messages": ["첫 메시지"]}, config=config1)
app.invoke({"messages": ["두번째 메시지"]}, config=config1)

# 특정 시점에서 Fork
history = list(app.get_state_history(config1))
fork_point = history[1].config  # 두번째 체크포인트

# 새 Thread로 Fork
forked_config = {
    "configurable": {
        "thread_id": "forked",
        "checkpoint_id": fork_point["configurable"]["checkpoint_id"]
    }
}

# Fork된 지점에서 다른 방향으로 실행
app.invoke({"messages": ["분기된 메시지"]}, config=forked_config)
```

## 고급 패턴: 암호화된 저장

민감한 데이터를 암호화하여 저장합니다.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import EncryptedSerializer
from cryptography.fernet import Fernet


# 암호화 키 생성 (실제로는 안전하게 관리)
key = Fernet.generate_key()

# 암호화된 Serializer 생성
serde = EncryptedSerializer(
    encryption_key=key,
    encryption_algorithm="fernet"
)

# 암호화된 Checkpointer
checkpointer = SqliteSaver.from_conn_string(
    "checkpoints.db",
    serde=serde
)
```

## 요약

- **MemorySaver**: 개발/테스트용, 메모리에 저장
- **SqliteSaver**: 로컬 파일 저장, 단일 프로세스
- **PostgresSaver**: 프로덕션용, 다중 프로세스/서버 지원
- **get_state()**: 현재 상태 조회
- **get_state_history()**: 전체 히스토리 조회
- **update_state()**: 상태 수동 업데이트

## 다음 단계

다음 장에서는 **메모리 시스템**을 학습합니다. 단기/장기 메모리와 메시지 관리를 다룹니다.

👉 [Chapter 14: 메모리 시스템](./14-memory.md)

---

## 📚 참고 자료

### 공식 문서
- [Persistence (공식 온라인)](https://docs.langchain.com/oss/python/langgraph/persistence) - 영속성 가이드
- [Checkpointers (공식 온라인)](https://docs.langchain.com/oss/python/langgraph/checkpointers) - Checkpointer 레퍼런스

### 실습 코드
- [전체 소스](../../src/part4_production/13_checkpointer.py) - 실행 가능한 전체 코드

### 관련 챕터
- [이전: Chapter 12 - 서브그래프 활용](../Part3-Agent/12-subgraphs.md)
- [다음: Chapter 14 - 메모리 시스템](./14-memory.md)
