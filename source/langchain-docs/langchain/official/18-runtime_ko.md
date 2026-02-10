# 런타임

## 개요

LangChain의 `create_agent`는 내부적으로 LangGraph의 런타임에서 실행됩니다.

LangGraph는 다음 정보를 가진 `Runtime` 객체를 노출합니다:

1. **Context**: Agent 호출을 위한 사용자 ID, DB 연결 또는 기타 종속성과 같은 정적 정보
2. **Store**: [장기 메모리](/oss/python/langchain/long-term-memory)에 사용되는 `BaseStore` 인스턴스
3. **Stream writer**: `"custom"` 스트림 모드를 통해 정보를 스트리밍하는 데 사용되는 객체

> [!TIP]
> 런타임 Context는 Tool과 Middleware에 대한 **의존성 주입**을 제공합니다. 값을 하드코딩하거나 전역 상태를 사용하는 대신 Agent를 호출할 때 런타임 종속성(예: 데이터베이스 연결, 사용자 ID 또는 구성)을 주입할 수 있습니다. 이렇게 하면 Tool이 더 테스트 가능하고 재사용 가능하며 유연해집니다.

런타임 정보는 [Tool](/oss/python/langchain/tools) 및 [Middleware](/oss/python/langchain/middleware/overview) 내에서 액세스할 수 있습니다.

## 액세스

`create_agent`로 Agent를 생성할 때 Agent `Runtime`에 저장된 `context`의 구조를 정의하는 `context_schema`를 지정할 수 있습니다.

Agent를 호출할 때 실행을 위한 관련 구성과 함께 `context` 인수를 전달합니다:

```python
from dataclasses import dataclass

from langchain.agents import create_agent

@dataclass
class Context:
    user_name: str

agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)
```

## Tool 내부

Tool 내부의 런타임 정보에 액세스하여 다음을 수행할 수 있습니다:

- Context에 액세스
- 장기 메모리 읽기 또는 쓰기
- **custom 스트림**에 쓰기 (예: Tool 진행 상황/업데이트)

Tool 내부의 `Runtime` 객체에 액세스하려면 `ToolRuntime` 매개변수를 사용합니다.

```python
from dataclasses import dataclass

from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """저장소에서 사용자의 이메일 기본 설정을 가져옵니다."""
    user_id = runtime.context.user_id

    preferences: str = "The user prefers you to write a brief and polite email."
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences
```

## Middleware 내부

Middleware에서 런타임 정보에 액세스하여 동적 프롬프트를 생성하거나, 메시지를 수정하거나, 사용자 Context를 기반으로 Agent 동작을 제어할 수 있습니다.

**node 스타일 후크** 내부의 `Runtime` 객체에 액세스하려면 `Runtime` 매개변수를 사용합니다. **wrap 스타일 후크**의 경우, `Runtime` 객체는 `ModelRequest` 매개변수 내부에서 사용 가능합니다.

```python
from dataclasses import dataclass

from langchain.messages import AnyMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_name: str

# 동적 프롬프트
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

# 모델 호출 전 후크
@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Processing request for user: {runtime.context.user_name}")
    return None

# 모델 호출 후 후크
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Completed request for user: {runtime.context.user_name}")
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    middleware=[dynamic_system_prompt, log_before_model, log_after_model],
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)
```
