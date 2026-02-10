# 빠른 시작

이 빠른 시작은 간단한 설정에서 완전히 작동하는 AI Agent까지 단 몇 분 안에 만드는 과정을 안내합니다.

> LangChain Docs MCP server
>
> AI 코딩 어시스턴트나 IDE(예: Claude Code 또는 Cursor)를 사용 중이라면, [LangChain Docs MCP server](/use-these-docs)를 설치하여 최대한 활용하세요. 이는 Agent가 최신 LangChain 문서와 예제에 접근할 수 있도록 보장합니다.

## 요구 사항

이 예제들을 실행하려면 다음을 수행해야 합니다:

*   [LangChain 패키지 설치](/oss/python/langchain/install)
*   [Claude (Anthropic)](https://www.anthropic.com/) 계정을 설정하고 API 키 획득
*   터미널에서 `ANTHROPIC_API_KEY` 환경 변수 설정

이 예제들은 Claude를 사용하지만, 코드의 모델 이름을 변경하고 적절한 API 키를 설정하여 [지원되는 모델](/oss/python/integrations/providers/overview)을 사용할 수 있습니다.

## 기본 Agent 구성

질문에 답하고 Tool을 호출할 수 있는 간단한 Agent를 만드는 것부터 시작합니다. Agent는 Claude Sonnet 4.5를 언어 모델로, 기본 날씨 함수를 Tool로, 간단한 프롬프트를 동작 가이드로 사용합니다.

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

# Agent를 생성합니다
agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Agent를 실행합니다
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

> LangSmith를 사용하여 Agent를 추적하는 방법을 알아보려면 [LangSmith 문서](/langsmith/home)를 참조하세요.

## 실제 Agent 구성

다음으로, 주요 프로덕션 개념을 보여주는 실용적인 날씨 예측 Agent를 구성합니다:
1. **상세한 시스템 프롬프트**로 더 나은 Agent 동작 구현
2. **Tool 생성**으로 외부 데이터 통합
3. **모델 구성**으로 일관된 응답 보장
4. **구조화된 출력**으로 예측 가능한 결과 제공
5. **대화형 메모리**로 채팅 같은 상호작용 가능
6. **Agent 생성 및 실행**으로 완전히 작동하는 Agent 테스트

각 단계를 살펴보겠습니다:

### 1. 시스템 프롬프트 정의

시스템 프롬프트는 Agent의 역할과 동작을 정의합니다. 구체적이고 실행 가능하게 유지하세요:

```python
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""
```

### 2. Tool 생성

Tool은 모델이 정의한 함수를 호출하여 외부 시스템과 상호작용할 수 있게 합니다. Tool은 런타임 컨텍스트에 따라 달라질 수 있으며 Agent 메모리와도 상호작용합니다.

아래 `get_user_location` Tool이 런타임 컨텍스트를 사용하는 방식을 주목하세요:

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """커스텀 런타임 컨텍스트 스키마입니다."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """사용자 ID를 기반으로 사용자 정보를 검색합니다."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

tools = [get_weather_for_location, get_user_location]
```

> Tool은 잘 문서화되어야 합니다: 이름, 설명, 인수 이름이 모델의 프롬프트의 일부가 됩니다. LangChain의 `@tool` 데코레이터는 메타데이터를 추가하고 `ToolRuntime` 파라미터를 사용한 런타임 주입을 가능하게 합니다.

### 3. 모델 구성

사용 사례에 맞는 적절한 파라미터로 언어 모델을 설정합니다:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
```

선택한 모델과 제공자에 따라 초기화 파라미터가 다를 수 있습니다. 자세한 내용은 각 참조 페이지를 참조하세요.

### 4. 응답 형식 정의

필요에 따라 Agent 응답이 특정 스키마와 일치하도록 구조화된 응답 형식을 정의합니다.

```python
from dataclasses import dataclass

# 여기서는 dataclass를 사용하지만, Pydantic 모델도 지원됩니다.
@dataclass
class ResponseFormat:
    """Agent의 응답 스키마입니다."""
    # 재치 있는 응답 (항상 필수)
    punny_response: str
    # 있을 경우 날씨에 대한 흥미로운 정보
    weather_conditions: str | None = None
```

### 5. 메모리 추가

Agent에 메모리를 추가하여 상호작용 간 상태를 유지합니다. 이를 통해 Agent가 이전 대화와 컨텍스트를 기억할 수 있습니다.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

> 프로덕션 환경에서는 메시지 기록을 데이터베이스에 저장하는 영구 checkpointer를 사용하세요. 자세한 내용은 **메모리 추가 및 관리**를 참조하세요.

### 6. Agent 생성 및 실행

이제 모든 구성 요소로 Agent를 조립하고 실행합니다!

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id`는 주어진 대화의 고유 식별자입니다.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# 같은 `thread_id`를 사용하여 대화를 계속할 수 있습니다.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
```

<details>
<summary>전체 예제 코드 보기</summary>

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy


# 시스템 프롬프트 정의
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# 컨텍스트 스키마 정의
@dataclass
class Context:
    """커스텀 런타임 컨텍스트 스키마입니다."""
    user_id: str

# Tool 정의
@tool
def get_weather_for_location(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """사용자 ID를 기반으로 사용자 정보를 검색합니다."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# 모델 구성
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)

# 응답 형식 정의
@dataclass
class ResponseFormat:
    """Agent의 응답 스키마입니다."""
    # 재치 있는 응답 (항상 필수)
    punny_response: str
    # 있을 경우 날씨에 대한 흥미로운 정보
    weather_conditions: str | None = None

# 메모리 설정
checkpointer = InMemorySaver()

# Agent 생성
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Agent 실행
# `thread_id`는 주어진 대화의 고유 식별자입니다.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# 같은 `thread_id`를 사용하여 대화를 계속할 수 있습니다.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
```

</details>

> LangSmith를 사용하여 Agent를 추적하는 방법을 알아보려면 [LangSmith 문서](/langsmith/home)를 참조하세요.

축하합니다! 이제 다음을 수행할 수 있는 AI Agent를 보유했습니다:
* 컨텍스트를 이해하고 대화 기억
* 여러 Tool을 지능적으로 사용
* 일관된 형식의 구조화된 응답 제공
* 컨텍스트를 통한 사용자별 정보 처리
* 상호작용 간 대화 상태 유지
