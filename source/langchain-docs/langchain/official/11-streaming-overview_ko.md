# 개요

Agent 실행에서 실시간 업데이트를 스트리밍합니다.

LangChain은 실시간 업데이트를 제공하는 스트리밍 시스템을 구현합니다.
스트리밍은 LLM을 기반으로 구축된 애플리케이션의 반응성을 향상시키는 데 중요합니다. 완전한 응답이 준비되기 전에 출력을 점진적으로 표시함으로써 스트리밍은 특히 LLM의 지연 시간을 다룰 때 사용자 경험(UX)을 크게 향상시킵니다.

## 개요

LangChain의 스트리밍 시스템을 사용하면 Agent 실행에서 실시간 피드백을 애플리케이션에 표시할 수 있습니다.

LangChain 스트리밍으로 가능한 것:

-  [Agent 진행 상황 스트리밍](#agent-progress) — 각 Agent 단계 후 상태 업데이트 얻기.
-  [LLM 토큰 스트리밍](#llm-tokens) — 언어 모델 토큰을 생성되는 대로 스트리밍.
-  [사용자 정의 업데이트 스트리밍](#custom-updates) — 사용자 정의 신호 내보내기(예: "10/100개 레코드 가져옴").
-  [여러 모드 스트리밍](#stream-multiple-modes) — 업데이트(Agent 진행 상황), 메시지(LLM 토큰 + 메타데이터) 또는 사용자 정의(임의 사용자 데이터) 중에서 선택.

아래의 [일반적인 패턴](#common-patterns) 섹션에서 추가 종단 간 예제를 참조하세요.

## 지원되는 스트리밍 모드

다음 스트리밍 모드 중 하나 이상을 `stream` 또는 `astream` 메서드에 목록으로 전달하세요:

| 모드 | 설명 |
| :--- | :--- |
| `updates` | 각 Agent 단계 후 상태 업데이트를 스트리밍합니다. 같은 단계에서 여러 업데이트가 이루어지면(예: 여러 노드가 실행됨) 그 업데이트들이 별도로 스트리밍됩니다. |
| `messages` | LLM이 호출되는 모든 그래프 노드에서 `(token, metadata)` 튜플을 스트리밍합니다. |
| `custom` | 스트림 라이터를 사용하여 그래프 노드 내부에서 사용자 정의 데이터를 스트리밍합니다. |

## Agent 진행 상황

Agent 진행 상황을 스트리밍하려면 `stream_mode="updates"`와 함께 `stream` 또는 `astream` 메서드를 사용하세요. 이는 모든 Agent 단계 후에 이벤트를 내보냅니다.

예를 들어 한 번 도구를 호출하는 Agent가 있다면 다음 업데이트를 확인할 수 있습니다:

- **LLM 노드**: 도구 호출 요청이 있는 `AIMessage`
- **도구 노드**: 실행 결과가 있는 `ToolMessage`
- **LLM 노드**: 최종 AI 응답

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
```

**출력:**

```text
step: model
content: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_OW2NYNsNSKhRZpjW0wm2Aszd'}]
step: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]
step: model
content: [{'type': 'text', 'text': 'It\'s always sunny in San Francisco!'}]
```

## LLM 토큰

LLM에서 생성되는 토큰을 스트리밍하려면 `stream_mode="messages"`를 사용하세요. 아래에서 Agent가 도구 호출과 최종 응답을 스트리밍하는 출력을 볼 수 있습니다.

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")
```

**출력:**

```text
node: model
content: [{'type': 'tool_call_chunk', 'id': 'call_vbCyBcP8VuneUzyYlSBZZsVa', 'name': 'get_weather', 'args': '', 'index': 0}]
# ... (간략함을 위해 중간 청크는 생략됨) ...
node: model
content: [{'type': 'text', 'text': 'San'}]
node: model
content: [{'type': 'text', 'text': ' Francisco'}]
node: model
content: [{'type': 'text', 'text': '!"\n\n'}]
```

## 사용자 정의 업데이트

도구가 실행될 때 업데이트를 스트리밍하려면 `get_stream_writer`를 사용할 수 있습니다.

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    writer = get_stream_writer()
    # 임의의 데이터를 스트리밍합니다
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)
```

**출력:**

```text
Looking up data for city: San Francisco
Acquired data for city: San Francisco
```

> Tool 내부에 `get_stream_writer`를 추가하면 LangGraph 실행 컨텍스트 외부에서 도구를 호출할 수 없습니다.

## 여러 모드 스트리밍

스트림 모드를 리스트로 전달하여 여러 스트리밍 모드를 지정할 수 있습니다: `stream_mode=["updates", "custom"]`.

스트리밍된 출력은 `(mode, chunk)` 튜플이 되며, 여기서 `mode`는 스트림 모드의 이름이고 `chunk`는 해당 모드에서 스트리밍되는 데이터입니다.

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")
```

**출력:**

```text
stream_mode: updates
content: {'model': {'messages': [...]}}

stream_mode: custom
content: Looking up data for city: San Francisco

stream_mode: custom
content: Acquired data for city: San Francisco

stream_mode: updates
content: {'tools': {'messages': [...]}}

stream_mode: updates
content: {'model': {'messages': [...]}}
```

## 일반적인 패턴

### 도구 호출 스트리밍

다음 두 가지를 모두 스트리밍할 수 있습니다:

1.  부분 JSON - [도구 호출](https://docs.langchain.com/oss/python/langchain/models#tool-calling)이 생성될 때
2.  실행되는 완성되고 파싱된 도구 호출

`stream_mode="messages"`를 지정하면 Agent의 모든 LLM 호출에서 생성된 증분 [메시지 청크](https://docs.langchain.com/oss/python/langchain/messages#message-chunks)를 스트리밍합니다. 파싱된 도구 호출이 있는 완성된 메시지에 접근하려면:

1.  해당 메시지가 [상태](https://docs.langchain.com/oss/python/langgraph/concepts/state)에서 추적되는 경우([`create_agent`](https://docs.langchain.com/oss/python/langchain/agents#create-agent)의 모델 노드처럼) `stream_mode=["messages", "updates"]`를 사용하여 [상태 업데이트](https://docs.langchain.com/oss/python/langgraph/how-tos/stream-updates)를 통해 완성된 메시지에 접근합니다(아래 시연).
2.  해당 메시지가 상태에서 추적되지 않는 경우 [사용자 정의 업데이트](https://docs.langchain.com/oss/python/langchain/streaming/custom-updates) 또는 스트리밍 루프 중에 청크를 집계합니다([다음 섹션](https://docs.langchain.com/oss/python/langchain/streaming/custom-updates)).

> [!INFO]
> Agent에 여러 LLM이 포함된 경우 [sub-agent에서 스트리밍](https://docs.langchain.com/oss/python/langchain/streaming/sub-agents)에 대한 아래 섹션을 참조하세요.

```python
from typing import Any
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

agent = create_agent("openai:gpt-5.2", tools=[get_weather])

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)
    # N.B. 모든 콘텐츠는 token.content_blocks를 통해 사용할 수 있습니다

def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

input_message = {"role": "user", "content": "What is the weather in Boston?"}

for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)

    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                # `source`는 노드 이름을 캡처합니다
                _render_completed_message(update["messages"][-1])
```

**출력:**

```text
[{'name': 'get_weather', 'args': '', 'id': 'call_D3Orjr89KgsLTZ9hTzYv7Hpf', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'city', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '":"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'Boston', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_D3Orjr89KgsLTZ9hTzYv7Hpf', 'type': 'tool_call'}]
Tool response: [{'type': 'text', 'text': "It's always sunny in Boston!"}]
The| weather| in| Boston| is| **|sun|ny|**|.|
```

### 완성된 메시지 접근

> 완성된 메시지가 Agent의 상태에서 추적되면 위에 시연된 대로 `stream_mode=["messages", "updates"]`를 사용하여 스트리밍 중에 완성된 메시지에 접근할 수 있습니다.

경우에 따라 완성된 메시지가 상태 업데이트에 반영되지 않습니다. Agent 내부에 접근할 수 있다면 [사용자 정의 업데이트](#custom-updates)를 사용하여 스트리밍 중에 이러한 메시지에 접근할 수 있습니다. 그 외에는 스트리밍 루프에서 메시지 청크를 집계할 수 있습니다(아래 참조).

아래의 예제를 고려해보세요. 여기서 우리는 스트림 라이터를 간소화된 가드레일 Middleware에 통합합니다. 이 Middleware는 구조화된 "안전/위험" 평가를 생성하기 위해 도구 호출을 시연합니다(이를 위해 구조화된 출력을 사용할 수도 있습니다):

```python
from typing import Any, Literal

from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from pydantic import BaseModel


class ResponseSafety(BaseModel):
    """응답을 안전 또는 위험으로 평가합니다."""
    evaluation: Literal["safe", "unsafe"]


safety_model = init_chat_model("openai:gpt-5.2")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """모델 기반 가드레일: LLM을 사용하여 응답 안전성을 평가합니다."""
    stream_writer = get_stream_writer()
    # 모델 응답을 가져옵니다
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # 다른 모델을 사용하여 안전성을 평가합니다
    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    result = model_with_tools.invoke(
        [
            {
                "role": "system",
                "content": "Evaluate this AI response as generally safe or unsafe."
            },
            {
                "role": "user",
                "content": f"AI response: {last_message.text}"
            }
        ]
    )
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None
```

그러면 이 Middleware를 Agent에 통합하고 사용자 정의 스트림 이벤트를 포함할 수 있습니다:

```python
from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk, AIMessage, AnyMessage


def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""

    return f"It's always sunny in {city}!"


agent = create_agent(
    model="openai:gpt-5.2",
    tools=[get_weather],
    middleware=[safety_guardrail],
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates", "custom"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
    if stream_mode == "custom":
        # 스트림에서 완성된 메시지에 접근합니다
        print(f"Tool calls: {data.tool_calls}")
```

**출력:**

```text
[{'name': 'get_weather', 'args': '', 'id': 'call_je6LWgxYzuZ84mmoDalTYMJC', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'city', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '":"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'Boston', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_je6LWgxYzuZ84mmoDalTYMJC', 'type': 'tool_call'}]
Tool response: [{'type': 'text', 'text': "It's always sunny in Boston!"}]
The| weather| in| **|Boston|**| is| **|sun|ny|**|.|[{'name': 'ResponseSafety', 'args': '', 'id': 'call_O8VJIbOG4Q9nQF0T8ltVi58O', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'evaluation', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '":"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'safe', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'ResponseSafety', 'args': {'evaluation': 'safe'}, 'id': 'call_O8VJIbOG4Q9nQF0T8ltVi58O', 'type': 'tool_call'}]
```

또는 스트림에 사용자 정의 이벤트를 추가할 수 없는 경우 스트리밍 루프 내에서 메시지 청크를 집계할 수 있습니다:

```python
input_message = {"role": "user", "content": "What is the weather in Boston?"}
full_message = None
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
            full_message = token if full_message is None else full_message + token
            if token.chunk_position == "last":
                if full_message.tool_calls:
                    print(f"Tool calls: {full_message.tool_calls}")
                full_message = None
    if stream_mode == "updates":
        for source, update in data.items():
            if source == "tools":
                _render_completed_message(update["messages"][-1])
```

### 인간 in the loop를 사용한 스트리밍

1.  인간 in the loop Middleware와 checkpointer로 Agent를 구성합니다.
2.  "updates" 스트림 모드 중에 생성된 중단을 수집합니다.
3.  명령으로 해당 중단에 응답합니다.

```python
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()

agent = create_agent(
    "openai:gpt-5.2",
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
    ],
    checkpointer=checkpointer,
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)

def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

def _render_interrupt(interrupt: Interrupt) -> None:
    interrupts = interrupt.value
    for request in interrupts["action_requests"]:
        print(request["description"])

input_message = {
    "role": "user",
    "content": (
        "Can you look up the weather in Boston and San Francisco?"
    ),
}
config = {"configurable": {"thread_id": "some_id"}}

interrupts = []
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    config=config,
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)

    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
                _render_interrupt(update[0])
```

**출력:**

```text
[{'name': 'get_weather', 'args': '', 'id': 'call_GOwNaQHeqMixay2qy80padfE', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"ci', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'ty": ', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"Bosto', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'n"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': 'get_weather', 'args': '', 'id': 'call_Ndb4jvWm2uMA0JDQXu37wDH6', 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"ci', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'ty": ', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"San F', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'ranc', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'isco"', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '}', 'id': None, 'index': 1, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_GOwNaQHeqMixay2qy80padfE', 'type': 'tool_call'}, {'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_Ndb4jvWm2uMA0JDQXu37wDH6', 'type': 'tool_call'}]
Tool execution requires approval

Tool: get_weather
Args: {'city': 'Boston'}
Tool execution requires approval

Tool: get_weather
Args: {'city': 'San Francisco'}
```

다음으로 각 중단에 대한 결정을 수집합니다. 중요하게도 결정의 순서는 수집한 작업의 순서와 일치해야 합니다.
설명하기 위해 한 도구 호출을 편집하고 다른 하나를 수락합니다:

```python
def _get_interrupt_decisions(interrupt: Interrupt) -> list[dict]:
    return [
        {
            "type": "edit",
            "edited_action": {
                "name": "get_weather",
                "args": {"city": "Boston, U.K."},
            },
        }
        if "boston" in request["description"].lower()
        else {"type": "approve"}
        for request in interrupt.value["action_requests"]
    ]

decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": _get_interrupt_decisions(interrupt)
    }

decisions
```

**출력:**

```text
{
    'a96c40474e429d661b5b32a8d86f0f3e': {
        'decisions': [
            {
                'type': 'edit',
                 'edited_action': {
                     'name': 'get_weather',
                     'args': {'city': 'Boston, U.K.'}
                 }
            },
            {'type': 'approve'},
        ]
    }
}
```

그러면 같은 스트리밍 루프에 명령을 전달하여 재개할 수 있습니다:

```python
interrupts = []
for stream_mode, data in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
):
    # 스트리밍 루프는 변경 사항 없음
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
            if source == "__interrupt__":
                interrupts.extend(update)
                _render_interrupt(update[0])
```

**출력:**

```text
Tool response: [{'type': 'text', 'text': "It's always sunny in Boston, U.K.!"}]
Tool response: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]
-| **|Boston|**|:| It|'s| always| sunny| in| Boston|,| U|.K|.|
|-| **|San| Francisco|**|:| It|'s| always| sunny| in| San| Francisco|!|
```

### Sub-agent에서 스트리밍

Agent의 어떤 지점에 여러 LLM이 있을 때 생성되는 메시지의 출처를 명확히 해야 하는 경우가 많습니다.

이를 위해 Agent를 만들 때 각 Agent에 `name`을 전달하세요. 이 이름은 `"messages"` 모드에서 스트리밍할 때 `lc_agent_name` 키를 통해 메타데이터에서 사용 가능합니다.

아래에서 [도구 호출 스트리밍](https://docs.langchain.com/oss/python/langchain/streaming/streaming-tool-calls) 예제를 업데이트합니다:

1.  Tool을 내부적으로 Agent를 호출하는 `call_weather_agent` Tool로 교체합니다
2.  각 Agent에 `name`을 추가합니다
3.  스트림을 만들 때 `subgraphs=True`를 지정합니다
4.  스트림 처리는 이전과 동일하지만 `create_agent`의 `name` 매개변수를 사용하여 어느 Agent가 활성 중인지 추적하는 로직을 추가합니다

> [!TIP]
> Agent에 `name`을 설정하면 해당 이름이 Agent에서 생성된 모든 `AIMessage`에도 첨부됩니다.

먼저 Agent를 구성합니다:

```python
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage


def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""

    return f"It's always sunny in {city}!"


weather_model = init_chat_model("openai:gpt-5.2")
weather_agent = create_agent(
    model=weather_model,
    tools=[get_weather],
    name="weather_agent",
)


def call_weather_agent(query: str) -> str:
    """날씨 Agent를 쿼리합니다."""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].text


supervisor_model = init_chat_model("openai:gpt-5.2")
agent = create_agent(
    model=supervisor_model,
    tools=[call_weather_agent],
    name="supervisor",
)
```

다음으로 스트리밍 루프에 로직을 추가하여 어느 Agent가 토큰을 내보내는지 보고합니다:

```python
def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
current_agent = None
for _, stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
):
    if stream_mode == "messages":
        token, metadata = data
        if agent_name := metadata.get("lc_agent_name"):
            if agent_name != current_agent:
                print(f"🤖 {agent_name}: ")
                current_agent = agent_name
        if isinstance(token, AIMessage):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
```

**출력:**

```text
🤖 supervisor:
[{'name': 'call_weather_agent', 'args': '', 'id': 'call_asorzUf0mB6sb7MiKfgojp7I', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'query', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '":"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'Boston', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': ' weather', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': ' right', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': ' now', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': ' and', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': " today's", 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': ' forecast', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'call_weather_agent', 'args': {'query': "Boston weather right now and today's forecast"}, 'id': 'call_asorzUf0mB6sb7MiKfgojp7I', 'type': 'tool_call'}]
🤖 weather_agent:
[{'name': 'get_weather', 'args': '', 'id': 'call_LZ89lT8fW6w8vqck5pZeaDIx', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'city', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '":"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'Boston', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
Tool calls: [{'name': 'get_weather', 'args': {'city': 'Boston'}, 'id': 'call_LZ89lT8fW6w8vqck5pZeaDIx', 'type': 'tool_call'}]
Tool response: [{'type': 'text', 'text': "It's always sunny in Boston!"}]
Boston| weather| right| now|:| **|Sunny|**|.

|Today|'s| forecast| for| Boston|:| **|Sunny| all| day|**|.|Tool response: [{'type': 'text', 'text': 'Boston weather right now: **Sunny**.\n\nToday's forecast for Boston: **Sunny all day**.'}]
🤖 supervisor:
Boston| weather| right| now|:| **|Sunny|**|.

|Today|'s| forecast| for| Boston|:| **|Sunny| all| day|**|.|
```

## 스트리밍 비활성화

일부 애플리케이션에서는 주어진 모델에 대한 개별 토큰의 스트리밍을 비활성화해야 할 수 있습니다. 이는 다음의 경우에 유용합니다:

- [다중 Agent](https://docs.langchain.com/oss/python/langchain/multi-agent) 시스템으로 작업하여 어느 Agent가 출력을 스트리밍할지 제어
- 스트리밍을 지원하는 모델과 지원하지 않는 모델 혼합
- [LangSmith](https://docs.langchain.com/langsmith/home)에 배포하고 특정 모델 출력이 클라이언트로 스트리밍되지 않도록 하고 싶음

모델을 초기화할 때 `streaming=False`를 설정하세요.

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    streaming=False
)
```

> [LangSmith](https://docs.langchain.com/langsmith/home)에 배포할 때 출력을 클라이언트로 스트리밍하지 않으려는 모든 모델에 대해 `streaming=False`를 설정합니다. 이는 배포 전에 그래프 코드에서 구성됩니다.

> 모든 채팅 모델 통합이 `streaming` 매개변수를 지원하는 것은 아닙니다. 모델이 지원하지 않으면 대신 `disable_streaming=True`를 사용하세요. 이 매개변수는 기본 클래스를 통해 모든 채팅 모델에서 사용할 수 있습니다.

자세한 내용은 [LangGraph 스트리밍 가이드](https://docs.langchain.com/oss/python/langgraph/streaming#disable-streaming-for-specific-chat-models)를 참조하세요.

## 관련

- [Frontend 스트리밍](https://docs.langchain.com/oss/python/langchain/streaming/frontend) — useStream을 사용하여 실시간 Agent 상호작용을 위한 React UI 구축
- [채팅 모델로 스트리밍](https://docs.langchain.com/oss/python/langchain/models#stream) — Agent 또는 그래프를 사용하지 않고 채팅 모델에서 직접 토큰 스트리밍
- [인간 in the loop를 사용한 스트리밍](https://docs.langchain.com/oss/python/langchain/human-in-the-loop#streaming-with-hil) — 인간 검토를 위한 중단을 처리하면서 Agent 진행 상황 스트리밍
- [LangGraph 스트리밍](https://docs.langchain.com/oss/python/langgraph/streaming) — 값, 디버그 모드, subgraph 스트리밍을 포함한 고급 스트리밍 옵션
