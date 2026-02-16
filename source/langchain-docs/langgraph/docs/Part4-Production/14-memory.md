# Chapter 14: 메모리 시스템

> 📌 **학습 목표**: 이 장을 마치면 단기/장기 메모리를 구현하고, 메시지 관리 패턴을 활용할 수 있습니다.

## 개요

AI Agent가 효과적으로 작동하려면 **메모리**가 필수입니다. 단기 메모리는 현재 대화를, 장기 메모리는 세션 간 정보를 유지합니다.

```mermaid
graph TD
    subgraph "Short-term Memory"
        ST1[Thread 1 대화] --> CP1[Checkpointer]
        ST2[Thread 2 대화] --> CP1
    end

    subgraph "Long-term Memory"
        LT1[사용자 정보] --> STORE[Memory Store]
        LT2[학습된 지식] --> STORE
    end

    CP1 --> AGENT[Agent]
    STORE --> AGENT
```

## 핵심 개념

### 메모리 유형

| 유형 | 범위 | 저장소 | 사용 사례 |
|-----|------|-------|----------|
| **Short-term** | Thread 내 | Checkpointer | 현재 대화 문맥 |
| **Long-term** | Thread 간 | Memory Store | 사용자 정보, 학습된 지식 |

## 실습 1: 단기 메모리 (Checkpointer)

```python
# 📁 src/part4_production/14a_short_term_memory.py
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


def chatbot(state: MessagesState) -> MessagesState:
    """간단한 챗봇 - 메시지 히스토리 활용"""
    messages = state["messages"]

    # 이전 대화 컨텍스트 활용
    context = f"이전 메시지 수: {len(messages)}"
    response = f"[{context}] 응답: {messages[-1].content}에 대한 답변"

    return {"messages": [AIMessage(content=response)]}


# 그래프 생성
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Checkpointer로 단기 메모리 추가
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Thread 별로 대화 유지
config = {"configurable": {"thread_id": "user-123"}}

# 첫 번째 메시지
app.invoke({"messages": [HumanMessage(content="안녕하세요")]}, config=config)

# 두 번째 메시지 - 이전 대화 기억
app.invoke({"messages": [HumanMessage(content="날씨 어때요?")]}, config=config)

# 상태 확인
state = app.get_state(config)
print(f"전체 메시지: {len(state.values['messages'])}개")
```

> 💡 **전체 코드**: [src/part4_production/14a_short_term_memory.py](../../src/part4_production/14a_short_term_memory.py)

## 실습 2: 장기 메모리 (Memory Store)

```python
# 📁 src/part4_production/14b_long_term_memory.py
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END, MessagesState


# Memory Store 생성
memory_store = InMemoryStore()

# 사용자 정보 저장
memory_store.put(
    ("users", "user-123"),  # namespace와 key
    "preferences",
    {"language": "ko", "theme": "dark"}
)

# 그래프에서 Store 사용
def personalized_response(state: MessagesState, store) -> MessagesState:
    """사용자 정보를 활용한 개인화된 응답"""
    # 사용자 정보 조회
    user_prefs = store.get(("users", "user-123"), "preferences")

    if user_prefs:
        language = user_prefs.value.get("language", "en")
        response = f"[{language}] 개인화된 응답입니다."
    else:
        response = "일반 응답입니다."

    return {"messages": [response]}


# 그래프 컴파일 시 store 전달
graph = StateGraph(MessagesState)
graph.add_node("respond", personalized_response)
graph.add_edge(START, "respond")
graph.add_edge("respond", END)

app = graph.compile(store=memory_store)
```

### Memory Store 작업

```python
# 저장
store.put(namespace, key, value)

# 조회
item = store.get(namespace, key)
print(item.value)

# 검색
items = store.search(namespace, filter={"language": "ko"})

# 삭제
store.delete(namespace, key)
```

## 실습 3: 메시지 관리 - trim_messages

토큰 제한을 위해 메시지를 자릅니다.

```python
# 📁 src/part4_production/14c_message_management.py
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage


messages = [
    SystemMessage(content="당신은 AI 어시스턴트입니다."),
    HumanMessage(content="안녕하세요"),
    AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
    HumanMessage(content="날씨가 어때요?"),
    AIMessage(content="오늘은 맑고 따뜻합니다."),
    HumanMessage(content="감사합니다"),
    AIMessage(content="천만에요!"),
]

# 토큰 수 기준으로 자르기
trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",  # 최근 메시지 유지
    token_counter=len,  # 간단한 토큰 카운터
    include_system=True,  # 시스템 메시지 유지
    allow_partial=False,
)

print(f"원본: {len(messages)}개, 트리밍: {len(trimmed)}개")
```

### 그래프에서 trim_messages 사용

```python
from langgraph.graph import MessagesState


def call_model(state: MessagesState) -> MessagesState:
    """메시지를 트리밍하고 모델 호출"""
    # 최근 메시지만 유지
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,
        strategy="last",
        include_system=True,
    )

    # LLM 호출
    response = llm.invoke(trimmed)
    return {"messages": [response]}
```

## 실습 4: 메시지 삭제 - RemoveMessage

특정 메시지를 선택적으로 삭제합니다.

```python
from langchain_core.messages import RemoveMessage


def filter_messages(state: MessagesState) -> MessagesState:
    """특정 조건의 메시지 삭제"""
    remove_list = []

    for msg in state["messages"]:
        # 오래된 도구 메시지 삭제
        if isinstance(msg, ToolMessage) and should_remove(msg):
            remove_list.append(RemoveMessage(id=msg.id))

    return {"messages": remove_list}
```

## 실습 5: 메시지 요약 (Summarization)

오래된 메시지를 요약으로 대체합니다.

```python
from langchain_core.messages import SystemMessage


def summarize_messages(state: MessagesState) -> MessagesState:
    """오래된 메시지를 요약으로 대체"""
    messages = state["messages"]

    if len(messages) <= 10:
        return state  # 충분히 짧으면 그대로

    # 오래된 메시지 추출
    old_messages = messages[:-5]  # 최근 5개 제외
    recent_messages = messages[-5:]

    # 요약 생성
    summary_prompt = f"다음 대화를 한 문장으로 요약하세요:\n{old_messages}"
    summary = llm.invoke(summary_prompt)

    # 새 메시지 리스트 구성
    new_messages = [
        SystemMessage(content=f"이전 대화 요약: {summary.content}"),
        *recent_messages
    ]

    return {"messages": new_messages}


# 그래프에 요약 노드 추가
graph.add_node("summarize", summarize_messages)
graph.add_conditional_edges(
    "check_length",
    lambda s: "summarize" if len(s["messages"]) > 20 else "continue",
)
```

## 고급 패턴: PostgresStore (프로덕션)

```python
from langgraph.store.postgres import PostgresStore


# PostgreSQL 기반 장기 메모리
store = PostgresStore(
    conn_string="postgresql://user:pass@localhost/db"
)
await store.setup()

# 시맨틱 검색을 위한 임베딩 설정
from langchain_openai import OpenAIEmbeddings

store = PostgresStore(
    conn_string="...",
    embeddings=OpenAIEmbeddings()
)

# 시맨틱 검색
results = await store.asearch(
    ("memories",),
    query="사용자가 좋아하는 음식",
    limit=5
)
```

## 고급 패턴: 메모리 네임스페이스

```python
# 계층적 네임스페이스 구조
# ("users", user_id, "preferences")
# ("users", user_id, "history")
# ("knowledge", topic, "facts")

# 사용자별 메모리
store.put(("users", "user-123", "preferences"), "theme", "dark")
store.put(("users", "user-123", "history"), "last_topic", "weather")

# 전역 지식
store.put(("knowledge", "weather", "facts"), "source", {...})

# 네임스페이스로 검색
user_data = store.search(("users", "user-123"))
```

## 요약

- **단기 메모리**: Checkpointer로 Thread 내 대화 유지
- **장기 메모리**: Memory Store로 Thread 간 정보 공유
- **trim_messages**: 토큰 제한 내 메시지 관리
- **RemoveMessage**: 특정 메시지 선택적 삭제
- **Summarization**: 오래된 메시지 요약으로 대체

## 다음 단계

다음 장에서는 **Human-in-the-Loop**을 학습합니다. 중단점과 사용자 승인을 다룹니다.

👉 [Chapter 15: Human-in-the-Loop](./15-human-in-the-loop.md)

---

## 📚 참고 자료

### 공식 문서
- [Memory (공식 온라인)](https://docs.langchain.com/oss/python/langgraph/memory) - 메모리 가이드
- [Message Management (공식 온라인)](https://docs.langchain.com/oss/python/langgraph/memory#message-management) - 메시지 관리

### 실습 코드
- [단기 메모리](../../src/part4_production/14a_short_term_memory.py)
- [장기 메모리](../../src/part4_production/14b_long_term_memory.py)
- [메시지 관리](../../src/part4_production/14c_message_management.py)

### 관련 챕터
- [이전: Chapter 13 - 영속성](./13-persistence.md)
- [다음: Chapter 15 - Human-in-the-Loop](./15-human-in-the-loop.md)
