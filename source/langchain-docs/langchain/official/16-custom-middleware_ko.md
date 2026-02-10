# 사용자 정의 Middleware

Agent 실행 흐름의 특정 지점에서 실행되는 훅을 구현하여 사용자 정의 Middleware를 구축합니다.

---

## 훅

Middleware는 Agent 실행을 가로채기 위한 두 가지 스타일의 훅을 제공합니다:

| | |
|:--|:--|
| **노드 스타일 훅** | **래핑 스타일 훅** |
| 특정 실행 지점에서 순차적으로 실행됩니다. | 각 모델 또는 도구 호출 주변에서 실행됩니다. |

---

## 노드 스타일 훅

특정 실행 지점에서 순차적으로 실행됩니다. 로깅, 검증, 상태 업데이트에 사용하세요.

**사용 가능한 훅:**

- `before_agent` - Agent 시작 전(호출당 한 번)
- `before_model` - 각 모델 호출 전
- `after_model` - 각 모델 응답 후
- `after_agent` - Agent 완료 후(호출당 한 번)

**예제:**

#### 데코레이터

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None
```

#### 클래스

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) == self.max_messages:
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None
```

---

## 래핑 스타일 훅

실행을 가로채고 핸들러가 호출되는 시기를 제어합니다. 재시도, 캐싱, 변환에 사용하세요.

핸들러가 0번(단축), 1번(정상 흐름) 또는 여러 번(재시도 로직) 호출될지 결정합니다.

**사용 가능한 훅:**

- `wrap_model_call` - 각 모델 호출 주변
- `wrap_tool_call` - 각 도구 호출 주변

**예제:**

#### 데코레이터

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```

#### 클래스

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable

class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
```

---

## Middleware 생성

두 가지 방법으로 Middleware를 생성할 수 있습니다:

| | |
|:--|:--|
| **데코레이터 기반 Middleware** | **클래스 기반 Middleware** |
| 단일 훅 Middleware에 대해 빠르고 간단합니다. 데코레이터를 사용하여 개별 함수를 래핑합니다. | 여러 훅 또는 구성이 있는 복잡한 Middleware에 더 강력합니다. |

### 데코레이터 기반 Middleware

단일 훅 Middleware에 대해 빠르고 간단합니다. 데코레이터를 사용하여 개별 함수를 래핑합니다.

**사용 가능한 데코레이터:**

노드 스타일:

- `@before_agent` - Agent 시작 전 실행(호출당 한 번)
- `@before_model` - 각 모델 호출 전 실행
- `@after_model` - 각 모델 응답 후 실행
- `@after_agent` - Agent 완료 후 실행(호출당 한 번)

래핑 스타일:

- `@wrap_model_call` - 사용자 정의 로직으로 각 모델 호출 래핑
- `@wrap_tool_call` - 사용자 정의 로직으로 각 도구 호출 래핑

편의 기능:

- `@dynamic_prompt` - 동적 시스템 프롬프트 생성

**예제:**

```python
from langchain.agents.middleware import (
    before_model,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

agent = create_agent(
    model="gpt-4.1",
    middleware=[log_before_model, retry_model],
    tools=[...],
)
```

> **데코레이터 사용 시기:**
> - 단일 훅이 필요한 경우
> - 복잡한 구성 없음
> - 빠른 프로토타이핑

### 클래스 기반 Middleware

여러 훅 또는 구성이 있는 복잡한 Middleware에 더 강력합니다. 같은 훅에 대해 동기 및 비동기 구현을 모두 정의해야 하거나 단일 Middleware에서 여러 훅을 결합하려고 할 때 클래스를 사용하세요.

**예제:**

```python
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime
from typing import Any, Callable

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

agent = create_agent(
    model="gpt-4.1",
    middleware=[LoggingMiddleware()],
    tools=[...],
)
```

> **클래스 사용 시기:**
> - 같은 훅에 대해 동기 및 비동기 구현을 모두 정의하는 경우
> - 단일 Middleware에서 여러 훅이 필요한 경우
> - 복잡한 구성이 필요한 경우(예: 구성 가능한 임계값, 사용자 정의 모델)
> - 초기화 시간 구성으로 프로젝트 전반에서 재사용

---

## 사용자 정의 상태 스키마

Middleware는 Agent의 상태를 사용자 정의 속성으로 확장할 수 있습니다. 이를 통해 Middleware는:

- **실행 전반에 걸쳐 상태 추적:** 카운터, 플래그 또는 Agent의 실행 수명 주기 동안 지속되는 기타 값을 유지합니다
- **훅 간에 데이터 공유:** `before_model`에서 `after_model`로 또는 다양한 Middleware 인스턴스 간에 정보를 전달합니다
- **교차 관심사 구현:** Agent 로직을 수정하지 않고 비율 제한, 사용량 추적, 사용자 컨텍스트 또는 감사 로깅 같은 기능을 추가합니다
- **조건부 결정:** 누적된 상태를 사용하여 실행을 계속할지, 다른 노드로 이동할지 또는 동적으로 동작을 수정할지 결정합니다

#### 데코레이터

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentState, before_model, after_model
from typing_extensions import NotRequired
from typing import Any
from langgraph.runtime import Runtime

class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]

@before_model(state_schema=CustomState, can_jump_to=["end"])
def check_call_limit(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
    count = state.get("model_call_count", 0)
    if count > 10:
        return {"jump_to": "end"}
    return None

@after_model(state_schema=CustomState)
def increment_counter(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(
    model="gpt-4.1",
    middleware=[check_call_limit, increment_counter],
    tools=[],
)

# 사용자 정의 상태로 호출
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})
```

#### 클래스

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any

class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]

class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            return {"jump_to": "end"}
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(
    model="gpt-4.1",
    middleware=[CallCounterMiddleware()],
    tools=[],
)

# 사용자 정의 상태로 호출
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})
```

---

## 실행 순서

여러 Middleware를 사용할 때 실행 방식을 이해하세요:

```python
agent = create_agent(
    model="gpt-4.1",
    middleware=[middleware1, middleware2, middleware3],
    tools=[...],
)
```

<details>
<summary>실행 흐름</summary>

1. 이전 훅은 순서대로 실행됩니다:
   - `middleware1.before_agent()`
   - `middleware2.before_agent()`
   - `middleware3.before_agent()`

2. Agent 루프 시작
   - `middleware1.before_model()`
   - `middleware2.before_model()`
   - `middleware3.before_model()`

3. 래핑 훅은 함수 호출처럼 중첩됩니다:
   - `middleware1.wrap_model_call()` → `middleware2.wrap_model_call()` → `middleware3.wrap_model_call()` → model

4. 이후 훅은 역순으로 실행됩니다:
   - `middleware3.after_model()`
   - `middleware2.after_model()`
   - `middleware1.after_model()`

5. Agent 루프 종료
   - `middleware3.after_agent()`
   - `middleware2.after_agent()`
   - `middleware1.after_agent()`

</details>

**핵심 규칙:**

- `before_*` 훅: 첫 번째부터 마지막까지
- `after_*` 훅: 마지막부터 첫 번째까지(역순)
- `wrap_*` 훅: 중첩(첫 번째 Middleware가 모든 다른 Middleware를 래핑함)

---

## Agent 점프

Middleware에서 조기에 종료하려면 `jump_to`를 포함하는 딕셔너리를 반환하세요:

**사용 가능한 점프 대상:**

- `'end'`: Agent 실행의 끝으로 이동(또는 첫 번째 `after_agent` 훅)
- `'tools'`: 도구 노드로 이동
- `'model'`: 모델 노드로 이동(또는 첫 번째 `before_model` 훅)

#### 데코레이터

```python
from langchain.agents.middleware import after_model, hook_config, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

@after_model
@hook_config(can_jump_to=["end"])
def check_for_blocked(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return None
```

#### 클래스

```python
from langchain.agents.middleware import AgentMiddleware, hook_config, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

class BlockedContentMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        if "BLOCKED" in last_message.content:
            return {
                "messages": [AIMessage("I cannot respond to that request.")],
                "jump_to": "end"
            }
        return None
```

---

## 모범 사례

- **Middleware를 집중시키세요** - 각각 한 가지를 잘 수행해야 합니다
- **오류를 우아하게 처리하세요** - Middleware 오류가 Agent를 중단시키지 않도록 합니다
- **적절한 훅 타입을 사용하세요:**
  - 노드 스타일은 순차적 로직에(로깅, 검증)
  - 래핑 스타일은 제어 흐름에(재시도, 폴백, 캐싱)
- **모든 사용자 정의 상태 속성을 명확하게 문서화하세요**
- **Middleware를 통합 전에 독립적으로 단위 테스트하세요**
- **실행 순서를 고려하세요** - 중요한 Middleware를 목록의 처음에 배치하세요
- **가능한 경우 내장 Middleware를 사용하세요**

---

## 예제

### 동적 모델 선택

#### 데코레이터

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

complex_model = init_chat_model("gpt-4.1")
simple_model = init_chat_model("gpt-4.1-mini")

@wrap_model_call
def dynamic_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 대화 길이를 기반으로 다른 모델을 사용합니다
    if len(request.messages) > 10:
        model = complex_model
    else:
        model = simple_model
    return handler(request.override(model=model))
```

#### 클래스

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

complex_model = init_chat_model("gpt-4.1")
simple_model = init_chat_model("gpt-4.1-mini")

class DynamicModelMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 대화 길이를 기반으로 다른 모델을 사용합니다
        if len(request.messages) > 10:
            model = complex_model
        else:
            model = simple_model
        return handler(request.override(model=model))
```

### 도구 호출 모니터링

#### 데코레이터

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    print(f"Executing tool: {request.tool_call['name']}")
    print(f"Arguments: {request.tool_call['args']}")

    try:
        result = handler(request)
        print(f"Tool completed successfully")
        return result
    except Exception as e:
        print(f"Tool failed: {e}")
        raise
```

#### 클래스

```python
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

class ToolMonitoringMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"Executing tool: {request.tool_call['name']}")
        print(f"Arguments: {request.tool_call['args']}")

        try:
            result = handler(request)
            print(f"Tool completed successfully")
            return result
        except Exception as e:
            print(f"Tool failed: {e}")
            raise
```

### 도구 동적으로 선택하기

런타임에 관련 도구를 선택하여 성능과 정확성을 향상시킵니다. 이 섹션은 미리 등록된 도구를 필터링하는 것을 다룹니다. 런타임에 발견되는 도구(예: MCP 서버에서)를 등록하려면 [런타임 도구 등록](/oss/python/langchain/runtime#tool-registration)을 참조하세요.

**이점:**

- **더 짧은 프롬프트** - 관련 도구만 노출하여 복잡도 감소
- **더 나은 정확도** - 모델이 더 적은 옵션에서 올바르게 선택
- **권한 제어** - 사용자 접근 권한을 기반으로 동적으로 도구 필터링

#### 데코레이터

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """상태/컨텍스트를 기반으로 관련 도구를 선택하는 Middleware입니다."""
    # 상태/컨텍스트를 기반으로 관련 도구의 작은 부분집합을 선택합니다
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4.1",
    tools=all_tools,  # 모든 사용 가능한 도구는 미리 등록해야 합니다
    middleware=[select_tools],
)
```

#### 클래스

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable

class ToolSelectorMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """상태/컨텍스트를 기반으로 관련 도구를 선택하는 Middleware입니다."""
        # 상태/컨텍스트를 기반으로 관련 도구의 작은 부분집합을 선택합니다
        relevant_tools = select_relevant_tools(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4.1",
    tools=all_tools,  # 모든 사용 가능한 도구는 미리 등록해야 합니다
    middleware=[ToolSelectorMiddleware()],
)
```

### 시스템 메시지 작업

`ModelRequest`의 `system_message` 필드를 사용하여 Middleware에서 시스템 메시지를 수정합니다.

> `system_message` 필드는 Agent가 문자열 `system_prompt`로 만들어졌더라도 `SystemMessage` 객체를 포함합니다.

**예제: 시스템 메시지에 컨텍스트 추가**

#### 데코레이터

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable

@wrap_model_call
def add_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 항상 콘텐츠 블록으로 작업합니다
    new_content = list(request.system_message.content_blocks) + [
        {"type": "text", "text": "Additional context."}
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))
```

#### 클래스

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable

class ContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 항상 콘텐츠 블록으로 작업합니다
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": "Additional context."}
        ]
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))
```

**예제: 캐시 제어 사용(Anthropic)**

Anthropic 모델로 작업할 때 캐시 제어 지시문이 있는 구조화된 콘텐츠 블록을 사용하여 큰 시스템 프롬프트를 캐시할 수 있습니다:

#### 데코레이터

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable

@wrap_model_call
def add_cached_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 항상 콘텐츠 블록으로 작업합니다
    new_content = list(request.system_message.content_blocks) + [
        {
            "type": "text",
            "text": "Here is a large document to analyze:\n\n<document>...</document>",
            # 이 지점까지의 콘텐츠가 캐시됩니다
            "cache_control": {"type": "ephemeral"}
        }
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))
```

#### 클래스

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable

class CachedContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 항상 콘텐츠 블록으로 작업합니다
        new_content = list(request.system_message.content_blocks) + [
            {
                "type": "text",
                "text": "Here is a large document to analyze:\n\n<document>...</document>",
                "cache_control": {"type": "ephemeral"}  # 이 콘텐츠가 캐시됩니다
            }
        ]
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))
```

> **주의:**
> - `ModelRequest.system_message`는 Agent가 `system_prompt="string"`으로 만들어졌더라도 항상 `SystemMessage` 객체입니다
> - `SystemMessage.content_blocks`를 사용하여 원본 콘텐츠가 문자열인지 리스트인지 관계없이 콘텐츠를 블록 리스트로 접근합니다
> - 시스템 메시지를 수정할 때 `content_blocks`를 사용하고 기존 구조를 보존하기 위해 새 블록을 추가합니다
> - `create_agent`의 `system_prompt` 매개변수에 `SystemMessage` 객체를 직접 전달할 수 있습니다(캐시 제어 같은 고급 사용 사례의 경우)

---

## 추가 자료

- [Middleware API 참조](https://reference.langchain.com/python/langchain/middleware/)
- [내장 Middleware](/oss/python/langchain/middleware/built-in)
- [Agent 테스트](/oss/python/langchain/test)
