# 장기 메모리

## 개요

LangChain Agent는 [**LangGraph 영속성**](https://langchain-ai.github.io/langgraph/concepts/persistence/)을 사용하여 장기 메모리를 활성화합니다. 이것은 더 고급 주제이며 LangGraph에 대한 지식이 필요합니다.

## 메모리 저장

LangGraph는 장기 메모리를 [**스토어**](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store)에 JSON 문서로 저장합니다.

각 메모리는 커스텀 `namespace` (폴더와 유사) 및 서로 다른 `key` (파일 이름과 유사) 아래에 구성됩니다. Namespace는 종종 사용자 또는 조직 ID 또는 정보를 구성하기 쉽게 해주는 다른 레이블을 포함합니다.

이 구조는 메모리의 계층적 구성을 활성화합니다. 교차 Namespace 검색은 콘텐츠 필터를 통해 지원됩니다.

```python
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # 실제 임베딩 함수 또는 LangChain 임베딩 객체로 대체합니다
    return [[1.0, 2.0] * len(texts)]

# InMemoryStore는 데이터를 메모리 내 딕셔너리에 저장합니다. 프로덕션 사용에서는 DB 지원 스토어를 사용합니다.
store = InMemoryStore(index={"embed": embed, "dims": 2})

user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)

store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)

# ID로 "memory"를 가져옵니다
item = store.get(namespace, "a-memory")

# 이 Namespace 내에서 "memories"를 검색하고, 콘텐츠 동등성으로 필터링하고, 벡터 유사성으로 정렬합니다
items = store.search(
    namespace,
    filter={"my-key": "my-value"},
    query="language preferences"
)
```

메모리 스토어에 대한 자세한 내용은 [영속성 가이드](https://langchain-ai.github.io/langgraph/concepts/persistence/)를 참조하세요.

## Tool에서 장기 메모리 읽기

```python title="Agent가 사용할 수 있는 사용자 정보 조회 Tool"
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

# InMemoryStore는 데이터를 메모리 내 딕셔너리에 저장합니다. 프로덕션 사용에서는 DB 지원 스토어를 사용합니다.
store = InMemoryStore()

# put 메서드를 사용하여 스토어에 샘플 데이터 작성
store.put(
    ("users",),  # Namespace는 관련 데이터를 함께 그룹화합니다 (사용자 데이터를 위한 사용자 Namespace)
    "user_123",  # Namespace 내의 키 (사용자 ID를 키로)
    {
        "name": "John Smith",
        "language": "English",
    }  # 주어진 사용자를 위해 저장할 데이터
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """사용자 정보를 조회합니다."""
    # 스토어에 접근합니다 - `create_agent`에 제공된 것과 동일합니다
    store = runtime.store
    user_id = runtime.context.user_id

    # 스토어에서 데이터를 검색합니다 - 값과 메타데이터를 포함한 StoreValue 객체를 반환합니다
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_user_info],
    # Agent에 스토어 전달 - Tool을 실행할 때 Agent가 스토어에 접근할 수 있도록 활성화합니다
    store=store,
    context_schema=Context
)

# Agent 실행
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123")
)
```

## Tool에서 장기 메모리 작성

```python title="사용자 정보를 업데이트하는 Tool의 예"
from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

# InMemoryStore는 데이터를 메모리 내 딕셔너리에 저장합니다. 프로덕션 사용에서는 DB 지원 스토어를 사용합니다.
store = InMemoryStore()

@dataclass
class Context:
    user_id: str

# TypedDict는 LLM을 위한 사용자 정보의 구조를 정의합니다
class UserInfo(TypedDict):
    name: str

# Agent가 사용자 정보를 업데이트할 수 있게 하는 Tool (채팅 애플리케이션에 유용합니다)
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """사용자 정보를 저장합니다."""
    # 스토어에 접근합니다 - `create_agent`에 제공된 것과 동일합니다
    store = runtime.store
    user_id = runtime.context.user_id

    # 스토어에 데이터를 저장합니다 (Namespace, 키, 데이터)
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)

# Agent 실행
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # 업데이트되는 정보가 누구의 정보인지 식별하기 위해 컨텍스트로 전달된 user_id
    context=Context(user_id="user_123")
)

# 스토어에 직접 접근하여 값을 가져올 수 있습니다
store.get(("users",), "user_123").value
```
