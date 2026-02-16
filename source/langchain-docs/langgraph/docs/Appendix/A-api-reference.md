# Appendix A: API 레퍼런스

> 📌 LangGraph의 핵심 API와 클래스를 정리한 레퍼런스 가이드입니다.

## Graph API vs Functional API

### 선택 가이드

| 기준 | Graph API | Functional API |
|-----|-----------|----------------|
| **워크플로우 복잡성** | 복잡한 분기/병합 | 순차적/동적 |
| **시각화 필요성** | 높음 | 낮음 |
| **Python 숙련도** | 중급 | 고급 |
| **유지보수** | 명시적 구조 | 코드 기반 |
| **팀 협업** | 구조 공유 용이 | 코드 리뷰 중심 |

### 추천 시나리오

**Graph API 추천:**
- 에이전트 시스템 (명확한 상태 전이)
- 승인 워크플로우 (시각화 중요)
- 다중 분기 로직
- 팀 협업 프로젝트

**Functional API 추천:**
- 동적 워크플로우
- 빠른 프로토타이핑
- 복잡한 조건/반복 로직
- 개인 프로젝트

---

## Graph API 레퍼런스

### StateGraph

그래프의 기본 클래스입니다.

```python
from langgraph.graph import StateGraph

# 생성
graph = StateGraph(StateType)

# 주요 메서드
graph.add_node(name: str, func: Callable)
graph.add_edge(source: str, target: str)
graph.add_conditional_edges(source: str, condition: Callable, mapping: dict)
graph.compile(checkpointer=None, interrupt_before=None, interrupt_after=None)
```

**파라미터:**
- `StateType`: State를 정의하는 TypedDict
- `checkpointer`: 상태 지속성을 위한 Checkpointer
- `interrupt_before`: 해당 노드 실행 전 중단
- `interrupt_after`: 해당 노드 실행 후 중단

### 특수 노드

```python
from langgraph.graph import START, END

# START: 그래프 시작점
graph.add_edge(START, "first_node")

# END: 그래프 종료점
graph.add_edge("last_node", END)
```

### MessagesState

메시지 기반 State의 편의 클래스입니다.

```python
from langgraph.graph import MessagesState

# 자동으로 messages 필드와 add_messages reducer 제공
class MyState(MessagesState):
    # 추가 필드
    custom_field: str
```

**내장 필드:**
- `messages: Annotated[list, add_messages]`

---

## State 관련 API

### TypedDict State

```python
from typing import TypedDict, Annotated

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # Reducer 사용
    count: int  # 일반 필드 (덮어쓰기)
```

### Reducer 함수

```python
from langgraph.graph.message import add_messages
import operator

# 내장 Reducer
add_messages  # 메시지 리스트 병합
operator.add  # 리스트 연결

# 커스텀 Reducer
def my_reducer(current: list, new: list) -> list:
    return list(set(current + new))  # 중복 제거 병합
```

### RemoveMessage

특정 메시지를 삭제합니다.

```python
from langgraph.graph.message import RemoveMessage

# 메시지 ID로 삭제
return {"messages": [RemoveMessage(id="msg_123")]}
```

---

## Checkpointer API

### MemorySaver

인메모리 Checkpointer (개발용).

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### SqliteSaver

SQLite 기반 Checkpointer.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 파일 기반
checkpointer = SqliteSaver.from_conn_string("state.db")

# 인메모리
checkpointer = SqliteSaver.from_conn_string(":memory:")
```

### PostgresSaver

PostgreSQL 기반 Checkpointer (프로덕션용).

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@host:5432/db"
)
```

---

## Memory Store API

### InMemoryStore

인메모리 장기 저장소.

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 저장
store.put(namespace=("users", "user_1"), key="profile", value={"name": "Kim"})

# 조회
item = store.get(namespace=("users", "user_1"), key="profile")
print(item.value)  # {"name": "Kim"}

# 검색
items = store.search(namespace=("users", "user_1"))

# 삭제
store.delete(namespace=("users", "user_1"), key="profile")
```

---

## Interrupt API

### interrupt_before / interrupt_after

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_node"],
    interrupt_after=["review_node"]
)
```

### interrupt() 함수

노드 내에서 동적 중단.

```python
from langgraph.types import interrupt

def my_node(state):
    response = interrupt({
        "question": "계속하시겠습니까?",
        "options": ["yes", "no"]
    })
    # response는 사용자 응답
    return state
```

### Command

중단된 곳에서 재개.

```python
from langgraph.types import Command

# 응답과 함께 재개
result = app.invoke(
    Command(resume="yes"),
    config=config
)
```

---

## Functional API 레퍼런스

### @entrypoint

워크플로우 진입점을 정의합니다.

```python
from langgraph.func import entrypoint

@entrypoint(checkpointer=MemorySaver())
def my_workflow(input_data: dict) -> dict:
    # 워크플로우 로직
    return result
```

### @task

개별 작업 단위를 정의합니다.

```python
from langgraph.func import task

@task
def process_data(data: str) -> str:
    return f"처리됨: {data}"

# 워크플로우 내에서 호출
result = process_data(data).result()
```

---

## Streaming API

### stream()

동기 스트리밍.

```python
# 기본 (values 모드)
for chunk in app.stream(input, config):
    print(chunk)

# 특정 모드
for chunk in app.stream(input, config, stream_mode="updates"):
    print(chunk)

# 여러 모드
for chunk in app.stream(input, config, stream_mode=["values", "updates"]):
    mode, data = chunk
    print(f"[{mode}] {data}")
```

### astream()

비동기 스트리밍.

```python
async for chunk in app.astream(input, config):
    print(chunk)
```

### 스트리밍 모드

| 모드 | 설명 |
|-----|------|
| `values` | 각 단계의 전체 상태 |
| `updates` | 각 노드의 업데이트만 |
| `messages` | 메시지 관련만 |
| `events` | 모든 내부 이벤트 |

---

## State 관리 API

### get_state()

현재 상태를 조회합니다.

```python
state = app.get_state(config)
print(state.values)  # 현재 상태 값
print(state.next)    # 다음 실행될 노드
```

### get_state_history()

상태 히스토리를 조회합니다.

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values)
    print(snapshot.config)  # checkpoint_id 포함
```

### update_state()

상태를 수정합니다.

```python
app.update_state(
    config,
    {"field": "new_value"},
    as_node="node_name"  # 해당 노드가 수정한 것처럼 처리
)
```

---

## 도구 관련 API

### @tool 데코레이터

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """검색을 수행합니다."""
    return f"'{query}' 검색 결과"
```

### ToolNode

도구 실행 노드.

```python
from langgraph.prebuilt import ToolNode

tools = [search, calculator]
tool_node = ToolNode(tools)

graph.add_node("tools", tool_node)
```

### create_react_agent

ReAct 에이전트 생성.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer
)
```

---

## 타입 정의

### 주요 타입

```python
from typing import TypedDict, Annotated, Literal, Optional, List

# State 타입
class MyState(TypedDict):
    messages: Annotated[List, add_messages]
    status: Literal["pending", "complete"]
    error: Optional[str]

# Config 타입
config = {
    "configurable": {
        "thread_id": str,
        "checkpoint_id": Optional[str]
    }
}
```

---

## 자주 사용되는 패턴

### 조건부 라우팅

```python
def router(state: MyState) -> Literal["path_a", "path_b", END]:
    if state["condition"]:
        return "path_a"
    return "path_b"

graph.add_conditional_edges("decision", router)
```

### 병렬 실행

```python
# Send API 사용
from langgraph.constants import Send

def parallel_router(state):
    return [
        Send("worker", {"task": task})
        for task in state["tasks"]
    ]

graph.add_conditional_edges("distributor", parallel_router)
```

### 서브그래프

```python
# 서브그래프 정의
sub_graph = create_sub_graph()

# 부모 그래프에 추가
parent_graph.add_node("sub", sub_graph)
```

---

## 관련 링크

- [공식 API 문서](https://langchain-ai.github.io/langgraph/reference/)
- [GitHub 저장소](https://github.com/langchain-ai/langgraph)
- [LangChain 문서](https://python.langchain.com/docs/)
