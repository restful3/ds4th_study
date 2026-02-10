# 단기 메모리

## 개요

메모리는 이전 상호작용에 대한 정보를 기억하는 시스템입니다. AI Agent의 경우 이전 상호작용을 기억하고, 피드백에서 학습하고, 사용자 선호도에 맞춰 조정할 수 있게 해주므로 메모리는 매우 중요합니다. Agent가 수많은 사용자 상호작용과 함께 더 복잡한 작업을 처리할수록 이 기능은 효율성과 사용자 만족도 모두를 위해 필수적으로 됩니다.

단기 메모리는 애플리케이션이 단일 스레드 또는 대화 내에서 이전 상호작용을 기억할 수 있게 합니다.

> 스레드는 이메일이 단일 대화에서 메시지를 그룹화하는 방식과 유사하게 세션에서 여러 상호작용을 정리합니다.

대화 이력은 가장 일반적인 단기 메모리 형태입니다. 긴 대화는 오늘날의 LLM에 도전이 됩니다. 전체 이력이 LLM의 컨텍스트 윈도우에 맞지 않아 컨텍스트 손실 또는 오류가 발생할 수 있습니다.

모델이 전체 컨텍스트 길이를 지원하더라도 대부분의 LLM은 여전히 긴 컨텍스트에서 성능이 떨어집니다. 이들은 오래되었거나 주제에서 벗어난 콘텐츠에 의해 "산만해"지면서 동시에 더 느린 응답 시간과 높은 비용으로 고통받습니다.

채팅 모델은 메시지를 사용하여 컨텍스트를 수락하며, 여기에는 지침(시스템 메시지)과 입력(인간 메시지)이 포함됩니다. 채팅 애플리케이션에서는 메시지가 인간 입력과 모델 응답 사이에서 번갈아 나타나며, 시간이 지나면서 계속 길어지는 메시지 목록이 생깁니다. 컨텍스트 윈도우가 제한되어 있기 때문에 많은 애플리케이션이 오래된 정보를 제거하거나 "잊도록" 하는 기술을 사용함으로써 이점을 얻을 수 있습니다.

## 사용 방법

Agent에 단기 메모리(스레드 수준 지속성)를 추가하려면 Agent를 생성할 때 checkpointer를 지정해야 합니다.

> LangChain의 Agent는 Agent의 상태의 일부로 단기 메모리를 관리합니다.
> 이들을 그래프의 상태에 저장함으로써 Agent는 서로 다른 스레드 간 분리를 유지하면서 주어진 대화에 대한 전체 컨텍스트에 접근할 수 있습니다.
>
> 상태는 checkpointer를 사용하여 데이터베이스(또는 메모리)에 유지되므로 스레드를 언제든지 재개할 수 있습니다.
>
> 단기 메모리는 Agent가 호출되거나 단계(도구 호출 같은)가 완료될 때 업데이트되며, 상태는 각 단계의 시작에 읽힙니다.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

### 프로덕션 환경에서

프로덕션 환경에서는 데이터베이스로 지원되는 checkpointer를 사용하세요:

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # PostgresSql에서 테이블 자동 생성
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

> SQLite, Postgres, Azure Cosmos DB를 포함한 더 많은 checkpointer 옵션은 지속성 문서의 checkpointer 라이브러리 목록을 참조하세요.

## Agent 메모리 사용자 정의

기본적으로 Agent는 `AgentState`를 사용하여 단기 메모리를 관리하며, 특히 `messages` 키를 통한 대화 이력을 관리합니다.

`AgentState`를 확장하여 추가 필드를 추가할 수 있습니다. 사용자 정의 상태 스키마는 `state_schema` 매개변수를 사용하여 `create_agent`에 전달됩니다.

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

# 사용자 정의 상태는 invoke에서 전달할 수 있습니다
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})
```

## 일반적인 패턴

단기 메모리가 활성화된 상태에서 긴 대화는 LLM의 컨텍스트 윈도우를 초과할 수 있습니다. 일반적인 솔루션은:

| | |
| :--- | :--- |
| **✂️ 메시지 트리밍**<br>첫 번째 또는 마지막 N개 메시지 제거(LLM을 호출하기 전) | **🗑️ 메시지 삭제**<br>LangGraph 상태에서 메시지를 영구적으로 삭제 |
| **📚 메시지 요약**<br>이전 메시지 이력을 요약하고 요약으로 교체 | **⚙️ 사용자 정의 전략**<br>사용자 정의 전략(예: 메시지 필터링 등) |

이를 통해 Agent는 LLM의 컨텍스트 윈도우를 초과하지 않으면서 대화를 추적할 수 있습니다.

### 메시지 트리밍

대부분의 LLM은 최대 지원 컨텍스트 윈도우를 갖습니다(토큰 단위로 계산됨).

메시지를 언제 자를지 결정하는 한 가지 방법은 메시지 이력에서 토큰을 세고 그 제한에 접근할 때마다 자르는 것입니다. LangChain을 사용하는 경우 `trim messages` 유틸리티를 사용하고 목록에서 유지할 토큰 수와 경계 처리 전략(예: 마지막 `max_tokens` 유지)을 지정할 수 있습니다.

Agent에서 메시지 이력을 트리밍하려면 `@before_model` Middleware 데코레이터를 사용하세요:

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """컨텍스트 윈도우에 맞도록 최근 몇 개의 메시지만 유지합니다."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 변경 사항 없음

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
```

### 메시지 삭제

그래프 상태에서 메시지를 삭제하여 메시지 이력을 관리할 수 있습니다.
이는 특정 메시지를 제거하거나 전체 메시지 이력을 지우려고 할 때 유용합니다.
그래프 상태에서 메시지를 삭제하려면 `RemoveMessage`를 사용할 수 있습니다.
`RemoveMessage`가 작동하려면 `add_messages` 리듀서가 있는 상태 키를 사용해야 합니다.
기본 `AgentState`는 이를 제공합니다.

특정 메시지를 제거하려면:

```python
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # 가장 이른 두 개의 메시지 제거
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

모든 메시지를 제거하려면:

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

> 메시지를 삭제할 때 결과 메시지 이력이 유효한지 확인하세요. 사용 중인 LLM 공급자의 제한 사항을 확인하세요. 예를 들어:
> *   일부 공급자는 메시지 이력이 사용자 메시지로 시작되기를 기대합니다
> *   대부분의 공급자는 도구 호출이 있는 조수 메시지 뒤에 해당 도구 결과 메시지가 따라야 합니다.

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """대화를 관리 가능하게 유지하기 위해 오래된 메시지를 제거합니다."""
    messages = state["messages"]
    if len(messages) > 2:
        # 가장 이른 두 개의 메시지 제거
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])
```
```text
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.')]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! Nice to meet you. How can I help you today? I can answer questions, brainstorm ideas, draft text, explain things, or help with code.'), ('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
[('human', "what's my name?"), ('ai', 'Your name is Bob. How can I help you today, Bob?')]
```

### 메시지 요약

위에서 보인 메시지 트리밍 또는 제거의 문제는 메시지 큐를 삭제하면서 정보를 잃을 수 있다는 것입니다. 이 때문에 일부 애플리케이션은 채팅 모델을 사용하여 메시지 이력을 요약하는 보다 정교한 접근 방식으로부터 이점을 얻습니다.

![메시지 요약](images/summary.avif)

Agent에서 메시지 이력을 요약하려면 내장된 `SummarizationMiddleware`를 사용하세요:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob!
"""
```
더 많은 구성 옵션은 `SummarizationMiddleware`를 참조하세요.

## 메모리 접근

Agent의 단기 메모리(상태)를 여러 가지 방법으로 접근하고 수정할 수 있습니다:

### Tool(도구)

#### Tool에서 단기 메모리 읽기
`runtime` 매개변수(`ToolRuntime`로 타입됨)를 사용하여 Tool에서 단기 메모리(상태)에 접근하세요.
`runtime` 매개변수는 도구 서명에서 숨겨져 있으므로(모델이 보지 못함) 도구는 이를 통해 상태에 접근할 수 있습니다.

```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """사용자 정보를 조회합니다."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.
```

#### Tool에서 단기 메모리 쓰기

Agent의 단기 메모리(상태)를 실행 중 수정하려면 Tool에서 직접 상태 업데이트를 반환할 수 있습니다.
이는 중간 결과를 유지하거나 후속 도구 또는 프롬프트에서 정보를 접근 가능하게 하는 데 유용합니다.

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """사용자 정보를 조회하고 업데이트합니다."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # 메시지 이력 업데이트
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """사용자 정보를 찾은 후 사용자에게 인사합니다."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```

### 프롬프트

대화 이력 또는 사용자 정의 상태 필드를 기반으로 동적 프롬프트를 생성하는 Middleware에서 단기 메모리(상태)에 접근하세요.

```python
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class CustomContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """도시의 날씨를 가져옵니다."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()
```

#### 출력
```text
================================ Human Message =================================

What is the weather in SF?
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_WFQlOGn4b2yoJrv7cih342FG)
 Call ID: call_WFQlOGn4b2yoJrv7cih342FG
  Args:
    city: San Francisco
================================= Tool Message =================================
Name: get_weather

The weather in San Francisco is always sunny!
================================== Ai Message ==================================

Hi John Smith, the weather in San Francisco is always sunny!
```

#### Model 이전

모델 호출 전에 메시지를 처리하기 위해 `@before_model` Middleware에서 단기 메모리(상태)에 접근하세요.

```mermaid
graph TD
    __start__ --> before_model
    before_model --> model
    model --> tools
    tools --> before_model
    model -.-> __end__
```

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """컨텍스트 윈도우에 맞도록 최근 몇 개의 메시지만 유지합니다."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 변경 사항 없음

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
```

#### Model 이후

모델 호출 후에 메시지를 처리하기 위해 `@after_model` Middleware에서 단기 메모리(상태)에 접근하세요.

```mermaid
graph TD
    __start__ --> model
    model --> after_model
    after_model -.-> tools
    tools --> model
    after_model -.-> __end__
```

```python
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """민감한 단어가 포함된 메시지를 제거합니다."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)
```
