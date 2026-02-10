# 사람 개입 루프

사람 개입 루프(HITL) [**Middleware**](/oss/python/langchain/middleware/overview)는 Agent Tool 호출에 인간의 감시를 추가할 수 있게 합니다. 모델이 검토가 필요할 수 있는 작업, 예를 들어 파일 작성이나 SQL 실행을 제안할 때 Middleware는 실행을 일시 중지하고 결정을 기다립니다.

이는 설정 가능한 정책에 대해 각 Tool 호출을 확인하여 작동합니다. 개입이 필요하면 Middleware는 실행을 중지하는 [**중단**](https://langchain-ai.github.io/langgraph/concepts/interrupts/)을 발생시킵니다. 그래프 상태는 LangGraph의 [**영속성 계층**](https://langchain-ai.github.io/langgraph/concepts/persistence/)을 사용하여 저장되므로 실행을 안전하게 일시 중지하고 나중에 재개할 수 있습니다.

그 다음 인간의 결정이 어떤 일이 일어날지 결정합니다: 작업을 그대로 승인(`approve`)하거나, 실행 전에 수정(`edit`)하거나, 피드백과 함께 거부(`reject`)할 수 있습니다.

## 중단 결정 유형

[Middleware](/oss/python/langchain/middleware/overview)는 인간이 중단에 대응하는 세 가지 기본 방식을 정의합니다:

| 결정 유형 | 설명 | 사용 사례 예 |
|---------------|-------------|------------------|
| ✅ `approve` | 작업이 그대로 승인되고 변경 없이 실행됩니다. | 작성된 그대로 이메일 초안 보내기 |
| ✏️ `edit` | Tool 호출이 수정 사항과 함께 실행됩니다. | 이메일을 보내기 전에 받는 사람 변경 |
| ❌ `reject` | Tool 호출이 거부되고 설명이 대화에 추가됩니다. | 이메일 초안을 거부하고 재작성 방법 설명 |

각 Tool에 사용 가능한 결정 유형은 `interrupt_on`에서 구성하는 정책에 따라 달라집니다. 동시에 여러 Tool 호출이 일시 중지되면 각 작업에 별도의 결정이 필요합니다. 결정은 중단 요청에서 작업이 나타나는 순서와 같은 순서로 제공되어야 합니다.

> [!팁]
> Tool 인수를 **편집**할 때 변경 사항을 보수적으로 유지합니다. 원본 인수에 대한 상당한 수정은 모델이 접근 방식을 재평가하게 하고 잠재적으로 Tool을 여러 번 실행하거나 예상치 못한 작업을 수행하게 할 수 있습니다.

## 중단 구성

HITL을 사용하려면 Agent를 생성할 때 Middleware를 Agent의 Middleware 리스트에 추가합니다. Tool 작업을 각 작업에 허용되는 결정 유형에 매핑하여 구성합니다. Middleware는 Tool 호출이 매핑의 작업과 일치할 때 실행을 중단합니다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,  # 모든 결정 허용 (approve, edit, reject)
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # 편집 불가능
                # 안전한 작업, 승인 불필요
                "read_data": False,
            },
            # 중단 메시지의 접두사 - Tool 이름 및 인수와 결합되어 전체 메시지 형성
            # 예: "Tool execution pending approval: execute_sql with query='DELETE FROM...'"
            # 개별 Tool은 중단 구성에서 "description"을 지정하여 이를 재정의할 수 있습니다
            description_prefix="Tool execution pending approval",
        ),
    ],
    # Human-in-the-loop는 중단을 처리하기 위해 checkpointing이 필요합니다.
    # 프로덕션에서는 AsyncPostgresSaver와 같은 영구 checkpointer를 사용합니다.
    checkpointer=InMemorySaver(),
)
```

> [!정보]
> 중단 간 그래프 상태를 영속화하려면 **checkpointer**를 구성해야 합니다. 프로덕션에서는 `AsyncPostgresSaver`와 같은 영구 checkpointer를 사용합니다. 테스트 또는 프로토타이핑의 경우 `InMemorySaver`를 사용합니다.

Agent를 호출할 때 실행을 대화 스레드와 연결하는 스레드 ID를 포함하는 `config`를 전달합니다. 자세한 내용은 [LangGraph 중단 문서](https://langchain-ai.github.io/langgraph/concepts/interrupts/)를 참조하세요.

### 구성 옵션

#### `interrupt_on`

| 유형 | 필수 | 설명 |
|------|----------|-------------|
| `dict` | 필수 | Tool 이름을 승인 구성으로 매핑합니다. 값은 `True` (기본 구성으로 중단), `False` (자동 승인), 또는 `InterruptOnConfig` 객체입니다. |

#### `description_prefix`

| 유형 | 기본값 | 설명 |
|------|---------|-------------|
| `string` | `"Tool execution requires approval"` | 작업 요청 설명의 접두사 |

### InterruptOnConfig 옵션

#### `allowed_decisions`

| 유형 | 설명 |
|------|-------------|
| `list[string]` | 허용된 결정의 목록: `'approve'`, `'edit'`, 또는 `'reject'` |

#### `description`

| 유형 | 설명 |
|------|-------------|
| `string` \| `callable` | 정적 문자열 또는 커스텀 설명을 위한 호출 가능 함수 |

## 중단에 대응

Agent를 호출하면 완료되거나 중단이 발생할 때까지 실행됩니다. 중단은 Tool 호출이 `interrupt_on`에서 구성한 정책과 일치할 때 발생합니다.

이 경우 호출 결과는 검토가 필요한 작업이 포함된 `__interrupt__` 필드를 포함합니다. 그 다음 해당 작업을 검토자에게 표시하고 결정이 제공되면 실행을 재개할 수 있습니다.

```python
from langgraph.types import Command

# Human-in-the-loop는 LangGraph의 영속성 계층을 활용합니다.
# 실행을 대화 스레드와 연결하기 위해 스레드 ID를 제공해야 합니다.
# 그래야 대화를 일시 중지하고 재개할 수 있습니다 (인간 검토에 필요).
config = {"configurable": {"thread_id": "some_id"}}

# 중단이 발생할 때까지 그래프를 실행합니다.
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete old records from the database",
            }
        ]
    },
    config=config
)

# 중단은 action_requests와 review_configs가 포함된 전체 HITL 요청을 포함합니다
print(result['__interrupt__'])
# > [
# >     Interrupt(
# >         value={
# >             'action_requests': [
# >                 {
# >                     'name': 'execute_sql',
# >                     'arguments': {'query': 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';'},
# >                     'description': 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
# >                 }
# >             ],
# >             'review_configs': [
# >                 {
# >                     'action_name': 'execute_sql',
# >                     'allowed_decisions': ['approve', 'reject']
# >                 }
# >             ]
# >         }
# >     )
# > ]

# 승인 결정으로 재개합니다
agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # 또는 "reject"
    ),
    config=config  # 일시 중지된 대화를 재개하기 위해 동일한 스레드 ID
)
```

### 결정 유형

#### ✅ approve

Tool 호출을 그대로 승인하고 변경 없이 실행하려면 `approve`를 사용합니다.

```python
agent.invoke(
    Command(
        # 결정은 목록으로 제공되며, 검토 중인 각 작업마다 하나입니다.
        # 결정의 순서는 `__interrupt__` 요청에 나열된 작업의 순서와 일치해야 합니다.
        resume={
            "decisions": [
                {
                    "type": "approve",
                }
            ]
        }
    ),
    config=config  # 일시 중지된 대화를 재개하기 위해 동일한 스레드 ID
)
```

#### ✏️ edit

실행 전에 Tool 호출을 수정하려면 `edit`를 사용합니다. 새 Tool 이름과 인수를 사용하여 편집된 작업을 제공합니다.

```python
agent.invoke(
    Command(
        # 결정은 목록으로 제공되며, 검토 중인 각 작업마다 하나입니다.
        # 결정의 순서는 `__interrupt__` 요청에 나열된 작업의 순서와 일치해야 합니다.
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # Tool 이름과 인수가 있는 편집된 작업
                    "edited_action": {
                        # 호출할 Tool 이름입니다.
                        # 일반적으로 원본 작업과 동일합니다.
                        "name": "new_tool_name",
                        # Tool에 전달할 인수입니다.
                        "args": {"key1": "new_value", "key2": "original_value"},
                    }
                }
            ]
        }
    ),
    config=config  # 일시 중지된 대화를 재개하기 위해 동일한 스레드 ID
)
```

> [!팁]
> Tool 인수를 **편집**할 때 변경 사항을 보수적으로 유지합니다. 원본 인수에 대한 상당한 수정은 모델이 접근 방식을 재평가하게 하고 잠재적으로 Tool을 여러 번 실행하거나 예상치 못한 작업을 수행하게 할 수 있습니다.

#### ❌ reject

Tool 호출을 거부하고 실행 대신 피드백을 제공하려면 `reject`를 사용합니다.

```python
agent.invoke(
    Command(
        # 결정은 목록으로 제공되며, 검토 중인 각 작업마다 하나입니다.
        # 결정의 순서는 `__interrupt__` 요청에 나열된 작업의 순서와 일치해야 합니다.
        resume={
            "decisions": [
                {
                    "type": "reject",
                    # 작업이 거부된 이유에 대한 설명
                    "message": "No, this is wrong because ..., instead do this ...",
                }
            ]
        }
    ),
    config=config  # 일시 중지된 대화를 재개하기 위해 동일한 스레드 ID
)
```

메시지는 대화에 피드백으로 추가되어 Agent가 작업이 거부된 이유와 대신 수행해야 할 작업을 이해할 수 있도록 도와줍니다.

### 다중 결정

여러 작업을 검토 중일 때 중단에서 나타나는 순서와 같은 순서로 각 작업에 대한 결정을 제공합니다:

```python
{
    "decisions": [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {
                "name": "tool_name",
                "args": {"param": "new_value"}
            }
        },
        {
            "type": "reject",
            "message": "This action is not allowed"
        }
    ]
}
```

## Human-in-the-loop를 사용한 스트리밍

`invoke()` 대신 `stream()`을 사용하여 Agent가 실행되고 중단을 처리하는 동안 실시간 업데이트를 받을 수 있습니다. `stream_mode=['updates', 'messages']`를 사용하여 Agent 진행 상황과 LLM 토큰을 모두 스트리밍합니다.

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "some_id"}}

# 중단이 발생할 때까지 Agent 진행 상황 및 LLM 토큰을 스트리밍합니다
for mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
    config=config,
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        # LLM 토큰
        token, metadata = chunk
        if token.content:
            print(token.content, end="", flush=True)
    elif mode == "updates":
        # 중단 확인
        if "__interrupt__" in chunk:
            print(f"\n\nInterrupt: {chunk['__interrupt__']}")

# 인간 결정 후 스트리밍으로 재개합니다
for mode, chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        token, metadata = chunk
        if token.content:
            print(token.content, end="", flush=True)
```

스트림 모드에 대한 자세한 내용은 [스트리밍 가이드](/oss/python/langchain/streaming)를 참조하세요.

## 실행 수명 주기

Middleware는 모델이 응답을 생성한 후이지만 Tool 호출이 실행되기 전에 실행되는 `after_model` 훅을 정의합니다:

1. Agent가 모델을 호출하여 응답을 생성합니다.
2. Middleware가 Tool 호출에 대한 응답을 검사합니다.
3. 호출에 인간 입력이 필요하면 Middleware는 `action_requests`와 `review_configs`가 있는 `HITLRequest`를 구성하고 `interrupt`를 호출합니다.
4. Agent가 인간 결정을 기다립니다.
5. `HITLResponse` 결정에 따라 Middleware는 승인되거나 편집된 호출을 실행하고, 거부된 호출에 대해 `ToolMessage`를 합성하고, 실행을 재개합니다.

## 커스텀 HITL 로직

더 특화된 워크플로의 경우 [중단](https://langchain-ai.github.io/langgraph/concepts/interrupts/) 프리미티브와 [Middleware](/oss/python/langchain/middleware/overview) 추상화를 사용하여 직접 커스텀 HITL 로직을 구성할 수 있습니다. 위의 [실행 수명 주기](#실행-수명-주기)를 검토하여 중단을 Agent의 작동에 통합하는 방법을 이해합니다.
