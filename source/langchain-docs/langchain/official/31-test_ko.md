# 테스트

에이전트 애플리케이션은 LLM이 문제를 해결하기 위해 자신의 다음 단계를 결정할 수 있게 합니다. 이 유연성은 강력하지만 모델의 블랙박스 특성으로 인해 Agent의 한 부분의 조정이 나머지에 어떻게 영향을 미칠지 예측하기 어렵습니다. 프로덕션 준비가 된 Agent를 구축하려면 철저한 테스트가 필수입니다.

Agent를 테스트하는 몇 가지 접근 방식이 있습니다:

- **단위 테스트**는 메모리 내 페이크를 사용하여 Agent의 작고 결정론적인 부분을 격리하여 정확한 동작을 빠르고 결정론적으로 확인할 수 있습니다.
- **통합 테스트**는 실제 네트워크 호출을 사용하여 Agent를 테스트하여 구성 요소가 함께 작동하고, 자격 증명과 스키마가 일치하고, 지연 시간이 허용 가능한지 확인합니다.

에이전트 애플리케이션은 여러 구성 요소를 연결하고 LLM의 비결정론적 특성으로 인한 불안정성을 처리해야 하기 때문에 통합 테스트에 더 많이 의존하는 경향이 있습니다.

## 단위 테스트

### 채팅 모델 모킹

API 호출이 필요하지 않은 로직의 경우 메모리 내 스텁을 사용하여 응답을 모킹할 수 있습니다.

LangChain은 텍스트 응답을 모킹하기 위해 `GenericFakeChatModel`을 제공합니다. 응답의 반복자(AIMessage 또는 문자열)를 받아들이고 호출당 하나를 반환합니다. 일반 및 스트리밍 사용을 모두 지원합니다.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

model = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
    "bar"
]))

model.invoke("hello")
# AIMessage(content='', ..., tool_calls=[{'name': 'foo', 'args': {'bar': 'baz'}, 'id': 'call_1', 'type': 'tool_call'}])
```

다시 모델을 호출하면 반복자의 다음 항목을 반환합니다:

```python
model.invoke("hello, again!")
# AIMessage(content='bar', ...)
```

### InMemorySaver Checkpointer

테스트 중 지속성을 활성화하려면 `InMemorySaver` checkpointer를 사용할 수 있습니다. 이를 통해 여러 턴을 시뮬레이션하여 상태 종속 동작을 테스트할 수 있습니다:

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver()
)

# 첫 번째 호출
agent.invoke(
    {"messages": [HumanMessage(content="I live in Sydney, Australia")]},
    config={"configurable": {"thread_id": "session-1"}}
)

# 두 번째 호출: 첫 번째 메시지가 유지됩니다(시드니 위치). 따라서 모델은 GMT+10 시간을 반환합니다
agent.invoke(
    {"messages": [HumanMessage(content="What's my local time?")]},
    config={"configurable": {"thread_id": "session-1"}}
)
```

## 통합 테스트

Agent의 동작 중 많은 부분이 실제 LLM을 사용할 때만 나타납니다. 예를 들어 Agent가 호출하기로 결정한 도구, 응답 형식 지정 방식, 또는 프롬프트 수정이 전체 실행 궤적에 영향을 미치는지 등입니다. LangChain의 `agentevals` 패키지는 실시간 모델로 Agent 궤적을 테스트하기 위해 특별히 설계된 평가자를 제공합니다.

AgentEvals를 사용하면 **궤적 일치**를 수행하거나 **LLM 심판**을 사용하여 Agent의 궤적(도구 호출을 포함하는 메시지의 정확한 순서)을 쉽게 평가할 수 있습니다:

| 접근 방식 | 설명 |
|----------|-------------|
| **궤적 일치** | 주어진 입력에 대해 참조 궤적을 하드코드하고 단계별 비교를 통해 실행을 검증합니다. 예상 동작을 알고 있는 정의된 워크플로우를 테스트하기에 이상적입니다. 어떤 도구를 호출해야 하는지, 어떤 순서로 호출해야 하는지에 대한 구체적인 기대를 가지고 있을 때 사용합니다. 이 접근 방식은 결정론적이고 빠르며 추가 LLM 호출이 필요하지 않으므로 비용 효과적입니다. |
| **LLM-심판** | LLM을 사용하여 정성적으로 Agent의 실행 궤적을 검증합니다. "심판" LLM은 프롬프트 루브릭(참조 궤적 포함 가능)에 대해 Agent의 결정을 검토합니다. 더 유연하며 효율성 및 적절성과 같은 미묘한 측면을 평가할 수 있지만 LLM 호출이 필요하고 덜 결정론적입니다. 도구 호출이나 순서 지정 요구 사항에 엄격하지 않고 Agent의 궤적의 전반적인 품질과 합리성을 평가하려면 사용하세요. |

### AgentEvals 설치

```bash
pip install agentevals
```

또는 [AgentEvals 저장소](https://github.com/langchain-ai/agentevals)를 직접 클론합니다.

### 궤적 일치 평가자

AgentEvals는 Agent의 궤적을 참조 궤적과 일치시키는 `create_trajectory_match_evaluator` 함수를 제공합니다. 선택할 수 있는 4가지 모드가 있습니다:

| 모드 | 설명 | 사용 사례 |
|------|-------------|----------|
| `strict` | 같은 순서의 메시지 및 도구 호출의 정확한 일치 | 특정 순서 테스트(예: 인증 전 정책 조회) |
| `unordered` | 모든 순서로 허용된 동일한 도구 호출 | 순서가 중요하지 않을 때 정보 검색 검증 |
| `subset` | Agent가 참조의 도구만 호출(추가 없음) | Agent가 예상 범위를 초과하지 않도록 보장 |
| `superset` | Agent가 최소한 참조 도구를 호출(추가 허용) | 필요한 최소 작업이 수행되는지 검증 |

<details>
<summary>엄격한 일치</summary>

`strict` 모드는 메시지 콘텐츠의 차이를 허용하지만 같은 순서의 동일한 메시지와 도구 호출이 있는 궤적을 보장합니다. 이는 정책 조회 후 작업을 인증하도록 하는 등 특정 작업 순서를 시행해야 할 때 유용합니다.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator

@tool
def get_weather(city: str):
    """도시의 날씨 정보를 가져옵니다."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4.1", tools=[get_weather])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
)

def test_weather_tool_called_strict():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in San Francisco?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in San Francisco?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}
        ]),
        ToolMessage(content="It's 75 degrees and sunny in San Francisco.", tool_call_id="call_1"),
        AIMessage(content="The weather in San Francisco is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )

    # {
    #     'key': 'trajectory_strict_match',
    #     'score': True,
    #     'comment': None,
    # }

    assert evaluation["score"] is True
```

</details>

<details>
<summary>정렬되지 않은 일치</summary>

`unordered` 모드는 모든 순서로 동일한 도구 호출을 허용합니다. 특정 정보를 검색해야 하지만 순서는 상관없을 때 유용합니다. 예를 들어 Agent는 도시의 날씨와 이벤트를 모두 확인해야 하지만 순서는 중요하지 않습니다.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator

@tool
def get_weather(city: str):
    """도시의 날씨 정보를 가져옵니다."""
    return f"It's 75 degrees and sunny in {city}."

@tool
def get_events(city: str):
    """도시에서 일어나는 이벤트를 가져옵니다."""
    return f"Concert at the park in {city} tonight."

agent = create_agent("gpt-4.1", tools=[get_weather, get_events])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="unordered",
)

def test_multiple_tools_any_order():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's happening in SF today?")]
    })

    # 참조는 실제 실행과 다른 순서로 호출된 도구를 표시합니다
    reference_trajectory = [
        HumanMessage(content="What's happening in SF today?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_events", "args": {"city": "SF"}},
            {"id": "call_2", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="Concert at the park in SF tonight.", tool_call_id="call_1"),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_2"),
        AIMessage(content="Today in SF: 75 degrees and sunny with a concert at the park tonight."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )

    # {
    #     'key': 'trajectory_unordered_match',
    #     'score': True,
    # }

    assert evaluation["score"] is True
```

</details>

<details>
<summary>부분집합 및 상위집합 일치</summary>

`superset` 및 `subset` 모드는 부분 궤적과 일치합니다. `superset` 모드는 Agent가 참조 궤적의 도구를 최소한 호출했는지 검증하고 추가 도구 호출을 허용합니다. `subset` 모드는 Agent가 참조의 도구 이상으로 호출하지 않았는지 보장합니다.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator

@tool
def get_weather(city: str):
    """도시의 날씨 정보를 가져옵니다."""
    return f"It's 75 degrees and sunny in {city}."

@tool
def get_detailed_forecast(city: str):
    """도시의 상세 날씨 예보를 가져옵니다."""
    return f"Detailed forecast for {city}: sunny all week."

agent = create_agent("gpt-4.1", tools=[get_weather, get_detailed_forecast])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="superset",
)

def test_agent_calls_required_tools_plus_extra():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Boston?")]
    })

    # 참조는 get_weather만 필요하지만 Agent는 추가 도구를 호출할 수 있습니다
    reference_trajectory = [
        HumanMessage(content="What's the weather in Boston?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "Boston"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in Boston.", tool_call_id="call_1"),
        AIMessage(content="The weather in Boston is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )

    # {
    #     'key': 'trajectory_superset_match',
    #     'score': True,
    #     'comment': None,
    # }

    assert evaluation["score"] is True
```

`tool_args_match_mode` 속성 및/또는 `tool_args_match_overrides`를 설정하여 평가자가 실제 궤적과 참조의 도구 호출 간 평등을 어떻게 고려하는지를 사용자 정의할 수도 있습니다. 기본적으로 동일한 도구에 대해 같은 인자를 가진 도구 호출만 동일한 것으로 간주됩니다. 자세한 내용은 [저장소](https://github.com/langchain-ai/agentevals)를 방문하세요.

</details>

### LLM-심판 평가자

`create_trajectory_llm_as_judge` 함수를 사용하여 Agent의 실행 경로를 평가할 수도 있습니다. 궤적 일치 평가자와 달리 참조 궤적이 필요하지 않지만 사용 가능하면 제공할 수 있습니다.

<details>
<summary>참조 궤적 없음</summary>

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

@tool
def get_weather(city: str):
    """도시의 날씨 정보를 가져옵니다."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4.1", tools=[get_weather])

evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def test_trajectory_quality():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Seattle?")]
    })

    evaluation = evaluator(
        outputs=result["messages"],
    )

    # {
    #     'key': 'trajectory_accuracy',
    #     'score': True,
    #     'comment': 'The provided agent trajectory is reasonable...'
    # }

    assert evaluation["score"] is True
```

</details>

<details>
<summary>참조 궤적 포함</summary>

참조 궤적이 있으면 프롬프트에 추가 변수를 추가하고 참조 궤적을 전달할 수 있습니다. 아래에서는 미리 만들어진 `TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE` 프롬프트를 사용하고 `reference_outputs` 변수를 구성합니다:

```python
evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)

evaluation = evaluator(
    outputs=result["messages"],
    reference_outputs=reference_trajectory,
)
```

LLM이 궤적을 평가하는 방식에 대한 더 많은 사용자 정의를 위해 [저장소](https://github.com/langchain-ai/agentevals)를 방문하세요.

</details>

### 비동기 지원

모든 `agentevals` 평가자는 Python asyncio를 지원합니다. 팩토리 함수를 사용하는 평가자의 경우 함수 이름의 `create_` 뒤에 `async`를 추가하여 비동기 버전을 사용할 수 있습니다.

<details>
<summary>비동기 심판 및 평가자 예제</summary>

```python
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from agentevals.trajectory.match import create_async_trajectory_match_evaluator

async_judge = create_async_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

async_evaluator = create_async_trajectory_match_evaluator(
    trajectory_match_mode="strict",
)

async def test_async_evaluation():
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="What's the weather?")]
    })
    evaluation = await async_judge(outputs=result["messages"])
    assert evaluation["score"] is True
```

</details>

## LangSmith 통합

시간 경과에 따라 실험을 추적하려면 평가자 결과를 [LangSmith](https://smith.langchain.com/)에 로그할 수 있습니다. LangSmith는 추적, 평가, 실험 도구를 포함한 프로덕션 등급 LLM 애플리케이션을 구축하기 위한 플랫폼입니다.

먼저 필요한 환경 변수를 설정하여 LangSmith를 설정합니다:

```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"
```

LangSmith는 평가를 실행하기 위한 두 가지 주요 접근 방식을 제공합니다: pytest 통합 및 evaluate 함수.

<details>
<summary>pytest 통합 사용</summary>

```python
import pytest
from langsmith import testing as t
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

@pytest.mark.langsmith
def test_trajectory_accuracy():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in SF?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in SF?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_1"),
        AIMessage(content="The weather in SF is 75 degrees and sunny."),
    ]

    # LangSmith에 입력, 출력 및 참조 출력 로그
    t.log_inputs({})
    t.log_outputs({"messages": result["messages"]})
    t.log_reference_outputs({"messages": reference_trajectory})

    trajectory_evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )
```

pytest로 평가를 실행합니다:

```bash
pytest test_trajectory.py --langsmith-output
```

결과가 자동으로 LangSmith에 로그됩니다.

</details>

<details>
<summary>evaluate 함수 사용</summary>

또는 LangSmith에서 데이터세트를 만들고 `evaluate` 함수를 사용할 수 있습니다:

```python
from langsmith import Client
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

client = Client()

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def run_agent(inputs):
    """궤적 메시지를 반환하는 Agent 함수입니다."""
    return agent.invoke(inputs)["messages"]

experiment_results = client.evaluate(
    run_agent,
    data="your_dataset_name",
    evaluators=[trajectory_evaluator]
)
```

결과가 자동으로 LangSmith에 로그됩니다. Agent 평가에 대해 자세히 알아보려면 [LangSmith 문서](https://docs.smith.langchain.com/)를 참조하세요.

</details>

## HTTP 호출 기록 및 재생

실제 LLM API를 호출하는 통합 테스트는 특히 CI/CD 파이프라인에서 자주 실행할 때 느리고 비쌀 수 있습니다. HTTP 요청 및 응답을 기록한 다음 실제 네트워크 호출을 하지 않고 후속 실행에서 재생하는 라이브러리를 사용하는 것을 권장합니다.

[vcrpy](https://vcrpy.readthedocs.io/)를 사용하여 이를 달성할 수 있습니다. pytest를 사용하는 경우 [pytest-recording](https://pytest-recording.readthedocs.io/) 플러그인은 최소한의 구성으로 이를 활성화하는 간단한 방법을 제공합니다. 요청/응답은 카세트에 기록되어 후속 실행에서 실제 네트워크 호출을 모킹하는 데 사용됩니다.

`conftest.py` 파일을 설정하여 카세트에서 민감한 정보를 필터링합니다:

```python title="conftest.py"
import pytest

@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "XXXX"),
            ("x-api-key", "XXXX"),
            # ... 마스킹하려는 다른 헤더
        ],
        "filter_query_parameters": [
            ("api_key", "XXXX"),
            ("key", "XXXX"),
        ],
    }
```

그런 다음 프로젝트를 구성하여 vcr 마커를 인식합니다:

#### pytest.ini

```ini
[pytest]
markers =
    vcr: record/replay HTTP via VCR
addopts = --record-mode=once
```

#### pyproject.toml

```toml
[tool.pytest.ini_options]
markers = ["vcr: record/replay HTTP via VCR"]
addopts = "--record-mode=once"
```

`--record-mode=once` 옵션은 첫 번째 실행에서 HTTP 상호작용을 기록하고 후속 실행에서 재생합니다.

이제 간단히 vcr 마커로 테스트를 데코레이트합니다:

```python
@pytest.mark.vcr()
def test_agent_trajectory():
    # ...
```

이 테스트를 처음 실행하면 Agent가 실제 네트워크 호출을 하고 pytest는 `tests/cassettes` 디렉토리에 카세트 파일 `test_agent_trajectory.yaml`을 생성합니다. 후속 실행은 해당 카세트를 사용하여 실제 네트워크 호출을 모킹합니다. Agent의 요청이 이전 실행과 변경되지 않는 한 계속됩니다. 변경되면 테스트가 실패하고 카세트를 삭제하고 테스트를 다시 실행하여 새로운 상호작용을 기록해야 합니다.

> [!WARNING]
> 프롬프트를 수정하거나, 새로운 도구를 추가하거나, 예상 궤적을 변경할 때 저장된 카세트가 더 이상 최신이 되고 기존 테스트가 실패합니다. 해당 카세트 파일을 삭제하고 테스트를 다시 실행하여 새로운 상호작용을 기록해야 합니다.
