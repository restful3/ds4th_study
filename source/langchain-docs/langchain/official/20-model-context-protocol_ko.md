# Model Context Protocol (MCP)

[**Model Context Protocol (MCP)**](https://modelcontextprotocol.io/introduction)는 응용 프로그램이 LLM에 Tool과 컨텍스트를 제공하는 방식을 표준화하는 개방형 프로토콜입니다. LangChain Agent는 [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) 라이브러리를 사용하여 MCP 서버에 정의된 Tool을 사용할 수 있습니다.

## 빠른 시작

`langchain-mcp-adapters` 라이브러리를 설치합니다:

#### pip

```bash
pip install langchain-mcp-adapters
```

#### uv

```bash
uv add langchain-mcp-adapters
```

`langchain-mcp-adapters`를 사용하면 Agent가 하나 이상의 MCP 서버 전체에서 정의된 Tool을 사용할 수 있습니다.

> [!INFO]
> `MultiServerMCPClient`는 **기본적으로 상태 비저장**입니다. 각 Tool 호출은 신선한 MCP `ClientSession`을 만들고, Tool을 실행한 다음, 정리합니다. 자세한 내용은 [상태 저장 세션](#stateful-sessions) 섹션을 참조하세요.

```python title="여러 MCP 서버에 접근하기"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",  # 로컬 하위 프로세스 통신
            "command": "python",
            # math_server.py 파일로의 절대 경로
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "http",  # HTTP 기반 원격 서버
            # weather 서버를 포트 8000에서 시작했는지 확인
            "url": "http://localhost:8000/mcp",
        }
    }
)

tools = await client.get_tools()

agent = create_agent(
    "claude-sonnet-4-5-20250929",
    tools
)

math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)

weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
```

## 사용자 정의 서버

사용자 정의 MCP 서버를 만들려면 **FastMCP** 라이브러리를 사용합니다:

#### pip

```bash
pip install fastmcp
```

#### uv

```bash
uv add fastmcp
```

MCP Tool 서버로 Agent를 테스트하려면 다음 예시를 사용합니다:

#### Math 서버(stdio 전송)

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """두 수를 더합니다"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 수를 곱합니다"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

#### Weather 서버(스트리밍 가능한 HTTP 전송)

```python
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """위치에 대한 날씨를 가져옵니다."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

## 전송

MCP는 클라이언트-서버 통신을 위한 다양한 전송 메커니즘을 지원합니다.

### HTTP

`http` 전송(또한 `streamable-http`라고 함)은 클라이언트-서버 통신을 위해 HTTP 요청을 사용합니다. 자세한 내용은 [MCP HTTP 전송 사양](https://modelcontextprotocol.io/docs/spec/basic/transports#http-with-sse)을 참조하세요.

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        }
    }
)
```

### 헤더 전달

HTTP를 통해 MCP 서버에 연결할 때, 연결 구성의 `headers` 필드를 사용하여 사용자 정의 헤더(예: 인증 또는 추적)를 포함할 수 있습니다. 이는 `sse` (MCP 사양에서 deprecated됨) 및 `streamable_http` 전송에서 지원됩니다.

```python title="MultiServerMCPClient로 헤더 전달"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "X-Custom-Header": "custom-value"
            },
        }
    }
)

tools = await client.get_tools()

agent = create_agent("openai:gpt-4.1", tools)

response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
```

### 인증

`langchain-mcp-adapters` 라이브러리는 공식 MCP SDK를 사용하며, `httpx.Auth` 인터페이스를 구현하여 사용자 정의 인증 메커니즘을 제공할 수 있습니다.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "auth": auth,
        }
    }
)
```

<details>
<summary>사용자 정의 인증 구현 예시</summary>

(사용자 정의 인증 구현 세부 정보)

</details>

<details>
<summary>기본 제공 OAuth 흐름</summary>

(기본 제공 OAuth 흐름 세부 정보)

</details>

### stdio

클라이언트가 서버를 하위 프로세스로 시작하고 표준 입력/출력을 통해 통신합니다. 로컬 Tool과 간단한 설정에 최고입니다.

HTTP 전송과 달리, stdio 연결은 본질적으로 상태 저장입니다 - 하위 프로세스는 클라이언트 연결의 수명 동안 유지됩니다. 그러나 명시적 세션 관리 없이 `MultiServerMCPClient`를 사용할 때, 각 Tool 호출은 여전히 새로운 세션을 만듭니다. 지속적인 연결을 관리하려면 [상태 저장 세션](#stateful-sessions)을 참조하세요.

```python
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["/path/to/math_server.py"],
        }
    }
)
```

## 상태 저장 세션

기본적으로, `MultiServerMCPClient`는 상태 비저장입니다 - 각 Tool 호출은 신선한 MCP 세션을 만들고, Tool을 실행한 다음, 정리합니다.

MCP 세션의 라이프사이클을 제어해야 하는 경우(예를 들어, Tool 호출 전체에서 컨텍스트를 유지하는 상태 저장 서버로 작업할 때), `client.session()`을 사용하여 지속적인 `ClientSession`을 만들 수 있습니다.

```python title="상태 저장 Tool 사용을 위해 MCP ClientSession 사용"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

# 세션을 명시적으로 만듭니다
async with client.session("server_name") as session:
    # Tool, 리소스 또는 프롬프트를 로드하도록 세션을 전달
    tools = await load_mcp_tools(session)

    agent = create_agent(
        "anthropic:claude-3-7-sonnet-latest",
        tools
    )
```

## 핵심 기능

### Tool

Tool을 통해 MCP 서버는 LLM이 호출할 수 있는 실행 가능한 함수를 노출할 수 있습니다 - 데이터베이스 쿼리, API 호출, 외부 시스템 상호 작용 같은 작업을 수행합니다.

LangChain은 MCP Tool을 LangChain Tool로 변환하므로, 모든 LangChain Agent 또는 워크플로우에서 직접 사용할 수 있습니다.

#### Tool 로드

`client.get_tools()`를 사용하여 MCP 서버에서 Tool을 검색하고 Agent로 전달합니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

tools = await client.get_tools()

agent = create_agent("claude-sonnet-4-5-20250929", tools)
```

#### 구조화된 콘텐츠

MCP Tool은 인간이 읽을 수 있는 텍스트 응답과 함께 구조화된 콘텐츠를 반환할 수 있습니다. 이는 Tool이 텍스트와 함께 기계가 파싱할 수 있는 데이터(JSON)를 반환해야 하는 경우에 유용합니다.

MCP Tool이 `structuredContent`를 반환할 때, 어댑터는 이를 `MCPToolArtifact`로 래핑하고 Tool의 artifact로 반환합니다. `ToolMessage`의 `artifact` 필드를 사용하여 이에 접근할 수 있습니다.

또한 인터셉터를 사용하여 구조화된 콘텐츠를 자동으로 처리하거나 변환할 수 있습니다.

<details>
<summary>artifact에서 구조화된 콘텐츠 추출</summary>

Agent를 호출한 후 응답의 Tool 메시지에서 구조화된 콘텐츠에 접근할 수 있습니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain.messages import ToolMessage

client = MultiServerMCPClient({...})

tools = await client.get_tools()

agent = create_agent("claude-sonnet-4-5-20250929", tools)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Get data from the server"}]}
)

# Tool 메시지에서 구조화된 콘텐츠 추출
for message in result["messages"]:
    if isinstance(message, ToolMessage) and message.artifact:
        structured_content = message.artifact["structured_content"]
```

</details>

<details>
<summary>인터셉터를 통해 구조화된 콘텐츠 추가</summary>

구조화된 콘텐츠가 대화 기록에 보이기를 원한다면(모델에 표시), 인터셉터를 사용하여 구조화된 콘텐츠를 Tool 결과에 자동으로 추가할 수 있습니다:

```python
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import TextContent

async def append_structured_content(request: MCPToolCallRequest, handler):
    """artifact에서 구조화된 콘텐츠를 Tool 메시지에 추가합니다."""
    result = await handler(request)
    if result.structuredContent:
        result.content += [
            TextContent(type="text", text=json.dumps(result.structuredContent)),
        ]
    return result

client = MultiServerMCPClient({...}, tool_interceptors=[append_structured_content])
```

</details>

#### 멀티모달 Tool 콘텐츠

MCP Tool은 응답에서 멀티모달 콘텐츠(이미지, 텍스트 등)를 반환할 수 있습니다. MCP 서버가 여러 부분의 콘텐츠(예: 텍스트 및 이미지)를 반환할 때, 어댑터는 이를 LangChain의 표준 콘텐츠 블록으로 변환합니다.

`ToolMessage`의 `content_blocks` 속성을 통해 표준화된 표현에 접근할 수 있습니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

tools = await client.get_tools()

agent = create_agent("claude-sonnet-4-5-20250929", tools)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Take a screenshot of the current page"}]}
)

# Tool 메시지에서 멀티모달 콘텐츠에 접근
for message in result["messages"]:
    if message.type == "tool":
        # 제공자 기본 형식의 원본 콘텐츠
        print(f"Raw content: {message.content}")

        # 표준화된 콘텐츠 블록
        #
        for block in message.content_blocks:
            if block["type"] == "text":
                print(f"Text: {block['text']}")
            elif block["type"] == "image":
                print(f"Image URL: {block.get('url')}")
                print(f"Image base64: {block.get('base64', '')[:50]}...")
```

이를 통해 기본 MCP 서버가 콘텐츠를 형식화하는 방식에 관계없이 제공자 불가지론적 방식으로 멀티모달 Tool 응답을 처리할 수 있습니다.

### 리소스

리소스를 통해 MCP 서버는 파일, 데이터베이스 레코드, API 응답 같은 데이터를 노출할 수 있습니다.

LangChain은 MCP 리소스를 `Blob` 객체로 변환하며, 이는 텍스트 및 이진 콘텐츠 처리를 위한 통일된 인터페이스를 제공합니다.

#### 리소스 로드

`client.get_resources()`를 사용하여 MCP 서버에서 리소스를 로드합니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# 서버에서 모든 리소스 로드
blobs = await client.get_resources("server_name")

# 또는 URI로 특정 리소스 로드
blobs = await client.get_resources("server_name", uris=["file:///path/to/file.txt"])

for blob in blobs:
    print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
    print(blob.as_string())  # 텍스트 콘텐츠의 경우
```

더 많은 제어를 위해 세션과 함께 `load_mcp_resources`를 직접 사용할 수도 있습니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources

client = MultiServerMCPClient({...})

async with client.session("server_name") as session:
    # 모든 리소스 로드
    blobs = await load_mcp_resources(session)

    # 또는 URI로 특정 리소스 로드
    blobs = await load_mcp_resources(session, uris=["file:///path/to/file.txt"])
```

### 프롬프트

프롬프트를 통해 MCP 서버는 클라이언트가 검색하고 사용할 수 있는 재사용 가능한 프롬프트 템플릿을 노출할 수 있습니다.

LangChain은 MCP 프롬프트를 메시지로 변환하므로, Chat 기반 워크플로우에 쉽게 통합할 수 있습니다.

#### 프롬프트 로드

`client.get_prompt()`를 사용하여 MCP 서버에서 프롬프트를 로드합니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# 이름으로 프롬프트 로드
messages = await client.get_prompt("server_name", "summarize")

# 인수와 함께 프롬프트 로드
messages = await client.get_prompt(
    "server_name",
    "code_review",
    arguments={"language": "python", "focus": "security"}
)

# 워크플로우에서 메시지 사용
for message in messages:
    print(f"{message.type}: {message.content}")
```

더 많은 제어를 위해 세션과 함께 `load_mcp_prompt`를 직접 사용할 수도 있습니다:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.prompts import load_mcp_prompt

client = MultiServerMCPClient({...})

async with client.session("server_name") as session:
    # 이름으로 프롬프트 로드
    messages = await load_mcp_prompt(session, "summarize")

    # 인수와 함께 프롬프트 로드
    messages = await load_mcp_prompt(
        session,
        "code_review",
        arguments={"language": "python", "focus": "security"}
    )
```

## 고급 기능

### Tool 인터셉터

MCP 서버는 별도의 프로세스로 실행됩니다 - 저장소, 컨텍스트, Agent 상태 같은 LangGraph 런타임 정보에 접근할 수 없습니다. 인터셉터는 MCP Tool 실행 중에 이 런타임 컨텍스트에 접근할 수 있게 하여 이 격차를 메웁니다.

인터셉터는 또한 Tool 호출에 대한 Middleware 같은 제어를 제공합니다: 요청을 수정하고, 재시도를 구현하고, 동적으로 헤더를 추가하거나, 실행을 완전히 우회할 수 있습니다.

| 섹션 | 설명 |
|---------|-------------|
| 런타임 컨텍스트 접근 | 사용자 ID, API 키, 저장소 데이터, Agent 상태 읽기 |
| 상태 업데이트 및 Command | Agent 상태 업데이트 또는 Command로 그래프 흐름 제어 |
| 인터셉터 작성 | 요청 수정, 인터셉터 작성, 오류 처리 패턴 |

### 런타임 컨텍스트 접근

MCP Tool을 LangChain Agent 내에서 사용할 때(via `create_agent`), 인터셉터는 `ToolRuntime` 컨텍스트에 접근할 수 있습니다. 이는 Tool 호출 ID, 상태, 구성, 저장소에 대한 접근을 제공하므로 사용자 데이터에 접근하고, 정보를 유지하고, Agent 동작을 제어하는 강력한 패턴을 가능하게 합니다.

#### 런타임 컨텍스트

호출 시간에 전달되는 사용자 ID, API 키, 권한 같은 사용자별 구성에 접근합니다:

```python title="MCP Tool 호출에 사용자 컨텍스트 주입"
from dataclasses import dataclass
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str
    api_key: str

async def inject_user_context(
    request: MCPToolCallRequest,
    handler,
):
    """MCP Tool 호출에 사용자 자격 증명을 주입합니다."""
    runtime = request.runtime
    user_id = runtime.context.user_id
    api_key = runtime.context.api_key

    # Tool 인수에 사용자 컨텍스트 추가
    modified_request = request.override(
        args={**request.args, "user_id": user_id}
    )

    return await handler(modified_request)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[inject_user_context],
)

tools = await client.get_tools()

agent = create_agent("gpt-4.1", tools, context_schema=Context)

# 사용자 컨텍스트로 호출
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Search my orders"}]},
    context={"user_id": "user_123", "api_key": "sk-..."}
)
```

#### 저장소

사용자 선호사항을 검색하거나 대화 전체에서 데이터를 유지하기 위해 장기 메모리에 접근합니다:

```python title="저장소에서 사용자 선호사항에 접근"
from dataclasses import dataclass
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

async def personalize_search(
    request: MCPToolCallRequest,
    handler,
):
    """저장된 선호사항을 사용하여 MCP Tool 호출을 개인화합니다."""
    runtime = request.runtime
    user_id = runtime.context.user_id
    store = runtime.store

    # 저장소에서 사용자 선호사항 읽기
    prefs = store.get(("preferences",), user_id)

    if prefs and request.name == "search":
        # 사용자의 선호하는 언어 및 결과 제한 적용
        modified_args = {
            **request.args,
            "language": prefs.value.get("language", "en"),
            "limit": prefs.value.get("result_limit", 10),
        }
        request = request.override(args=modified_args)

    return await handler(request)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[personalize_search],
)

tools = await client.get_tools()

agent = create_agent(
    "gpt-4.1",
    tools,
    context_schema=Context,
    store=InMemoryStore()
)
```

#### 상태

현재 세션을 기반으로 결정하기 위해 대화 상태에 접근합니다:

```python title="인증 상태를 기반으로 Tool 필터링"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage

async def require_authentication(
    request: MCPToolCallRequest,
    handler,
):
    """사용자가 인증되지 않은 경우 민감한 MCP Tool을 차단합니다."""
    runtime = request.runtime
    state = runtime.state
    is_authenticated = state.get("authenticated", False)

    sensitive_tools = ["delete_file", "update_settings", "export_data"]

    if request.name in sensitive_tools and not is_authenticated:
        # Tool을 호출하는 대신 오류 반환
        return ToolMessage(
            content="Authentication required. Please log in first.",
            tool_call_id=runtime.tool_call_id,
        )

    return await handler(request)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[require_authentication],
)
```

#### Tool 호출 ID

Tool 호출 ID에 접근하여 적절하게 형식화된 응답을 반환하거나 Tool 실행을 추적합니다:

```python title="Tool 호출 ID와 함께 사용자 정의 응답 반환"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage

async def rate_limit_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """비용이 많이 드는 MCP Tool 호출을 속도 제한합니다."""
    runtime = request.runtime
    tool_call_id = runtime.tool_call_id

    # 속도 제한 확인(간소화된 예시)
    if is_rate_limited(request.name):
        return ToolMessage(
            content="Rate limit exceeded. Please try again later.",
            tool_call_id=tool_call_id,
        )

    result = await handler(request)

    # 성공한 Tool 호출 로그
    log_tool_execution(tool_call_id, request.name, success=True)

    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[rate_limit_interceptor],
)
```

더 많은 컨텍스트 엔지니어링 패턴은 [컨텍스트 엔지니어링](/oss/python/langchain/context-engineering) 및 [Tool](/oss/python/langchain/tools)을 참조하세요.

### 상태 업데이트 및 Command

인터셉터는 `Command` 객체를 반환하여 Agent 상태를 업데이트하거나 그래프 실행 흐름을 제어할 수 있습니다. 이는 작업 진행 상황을 추적하고, Agent 간 전환, 조기 실행 종료 등에 유용합니다.

```python title="작업 완료 표시 및 Agent 전환"
from langchain.agents import AgentState, create_agent
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

async def handle_task_completion(
    request: MCPToolCallRequest,
    handler,
):
    """작업 완료 표시 및 요약 Agent로 전달합니다."""
    result = await handler(request)

    if request.name == "submit_order":
        return Command(
            update={
                "messages": [result] if isinstance(result, ToolMessage) else [],
                "task_status": "completed",
            },
            goto="summary_agent",
        )

    return result
```

`Command`를 `goto="__end__"`와 함께 사용하여 조기에 실행을 종료합니다:

```python title="완료 시 Agent 실행 종료"
async def end_on_success(
    request: MCPToolCallRequest,
    handler,
):
    """작업이 완료로 표시되면 Agent 실행을 종료합니다."""
    result = await handler(request)

    if request.name == "mark_complete":
        return Command(
            update={"messages": [result], "status": "done"},
            goto="__end__",
        )

    return result
```

### 사용자 정의 인터셉터

인터셉터는 Tool 실행을 래핑하는 비동기 함수로, 요청/응답 수정, 재시도 로직, 기타 횡단 관심사를 가능하게 합니다. 이들은 "양파" 패턴을 따르며, 목록의 첫 번째 인터셉터가 가장 바깥쪽 레이어입니다.

#### 기본 패턴

인터셉터는 요청과 handler를 받는 비동기 함수입니다. handler를 호출하기 전에 요청을 수정하거나, 후에 응답을 수정하거나, handler를 완전히 건너뜰 수 있습니다.

```python title="기본 인터셉터 패턴"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def logging_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """실행 전후 Tool 호출을 로깅합니다."""
    print(f"Calling tool: {request.name} with args: {request.args}")

    result = await handler(request)

    print(f"Tool {request.name} returned: {result}")

    return result

client = MultiServerMCPClient(
    {"math": {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}},
    tool_interceptors=[logging_interceptor],
)
```

#### 요청 수정

`request.override()`를 사용하여 수정된 요청을 만듭니다. 이는 불변 패턴을 따르며, 원본 요청은 변경되지 않습니다.

```python title="Tool 인수 수정"
async def double_args_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """실행 전에 모든 숫자 인수를 두 배로 늘립니다."""
    modified_args = {k: v * 2 for k, v in request.args.items()}
    modified_request = request.override(args=modified_args)

    return await handler(modified_request)

# 원본 호출: add(a=2, b=3)은 add(a=4, b=6)이 됩니다
```

#### 런타임에 헤더 수정

인터셉터는 요청 컨텍스트를 기반으로 HTTP 헤더를 동적으로 수정할 수 있습니다:

```python title="동적 헤더 수정"
async def auth_header_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """호출되는 Tool을 기반으로 인증 헤더를 추가합니다."""
    token = get_token_for_tool(request.name)
    modified_request = request.override(
        headers={"Authorization": f"Bearer {token}"}
    )

    return await handler(modified_request)
```

#### 인터셉터 작성

여러 인터셉터는 "양파" 순서로 작성됩니다 - 목록의 첫 번째 인터셉터가 가장 바깥쪽 레이어입니다:

```python title="여러 인터셉터 작성"
async def outer_interceptor(request, handler):
    print("outer: before")
    result = await handler(request)
    print("outer: after")
    return result

async def inner_interceptor(request, handler):
    print("inner: before")
    result = await handler(request)
    print("inner: after")
    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[outer_interceptor, inner_interceptor],
)

# 실행 순서:
# outer: before -> inner: before -> Tool 실행 -> inner: after -> outer: after
```

#### 오류 처리

인터셉터를 사용하여 Tool 실행 오류를 catch하고 재시도 로직을 구현합니다:

```python title="오류 시 재시도"
import asyncio

async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """지수 백오프로 실패한 Tool 호출을 재시도합니다."""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # 지수 백오프
                print(f"Tool {request.name} failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    raise last_error

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[retry_interceptor],
)
```

특정 오류 유형을 catch하고 폴백 값을 반환할 수도 있습니다:

```python title="폴백이 있는 오류 처리"
async def fallback_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Tool 실행이 실패하면 폴백 값을 반환합니다."""
    try:
        return await handler(request)
    except TimeoutError:
        return f"Tool {request.name} timed out. Please try again later."
    except ConnectionError:
        return f"Could not connect to {request.name} service. Using cached data."
```

### 진행 상황 알림

오래된 Tool 실행의 진행 상황 업데이트를 구독합니다:

```python title="진행 상황 콜백"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

async def on_progress(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
):
    """MCP 서버에서 진행 상황 업데이트를 처리합니다."""
    percent = (progress / total * 100) if total else progress
    tool_info = f" ({context.tool_name})" if context.tool_name else ""
    print(f"[{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_progress=on_progress),
)
```

`CallbackContext`는 다음을 제공합니다:

- `server_name`: MCP 서버 이름
- `tool_name`: 실행되는 Tool의 이름(Tool 호출 중 사용 가능)

### 로깅

MCP 프로토콜은 서버의 로깅 알림을 지원합니다. `Callbacks` 클래스를 사용하여 이러한 이벤트를 구독합니다.

```python title="로깅 콜백"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.types import LoggingMessageNotificationParams

async def on_logging_message(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
):
    """MCP 서버에서 로그 메시지를 처리합니다."""
    print(f"[{context.server_name}] {params.level}: {params.data}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_logging_message=on_logging_message),
)
```

### Elicitation

Elicitation을 통해 MCP 서버는 Tool 실행 중 사용자에게 추가 입력을 요청할 수 있습니다. 모든 입력을 미리 필요로 하는 대신, 서버는 필요에 따라 대화식으로 정보를 요청할 수 있습니다.

#### 서버 설정

`ctx.elicit()`를 사용하여 사용자 입력을 스키마로 요청하는 Tool을 정의합니다:

```python title="elicitation이 있는 MCP 서버"
from pydantic import BaseModel
from mcp.server.fastmcp import Context, FastMCP

server = FastMCP("Profile")

class UserDetails(BaseModel):
    email: str
    age: int

@server.tool()
async def create_profile(name: str, ctx: Context) -> str:
    """elicitation을 통해 세부 정보를 요청하여 사용자 프로필을 만듭니다."""
    result = await ctx.elicit(
        message=f"Please provide details for {name}'s profile:",
        schema=UserDetails,
    )

    if result.action == "accept" and result.data:
        return f"Created profile for {name}: email={result.data.email}, age={result.data.age}"

    if result.action == "decline":
        return f"User declined. Created minimal profile for {name}."

    return "Profile creation cancelled."

if __name__ == "__main__":
    server.run(transport="http")
```

#### 클라이언트 설정

`MultiServerMCPClient`에 콜백을 제공하여 elicitation 요청을 처리합니다:

```python title="elicitation 요청 처리"
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

async def on_elicitation(
    mcp_context: RequestContext,
    params: ElicitRequestParams,
    context: CallbackContext,
) -> ElicitResult:
    """MCP 서버에서 elicitation 요청을 처리합니다."""
    # 실제 응용 프로그램에서는 params.message 및 params.requestedSchema를 기반으로
    # 사용자 입력을 프롬프트합니다
    return ElicitResult(
        action="accept",
        content={"email": "user@example.com", "age": 25},
    )

client = MultiServerMCPClient(
    {
        "profile": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        }
    },
    callbacks=Callbacks(on_elicitation=on_elicitation),
)
```

#### 응답 작업

elicitation 콜백은 세 가지 작업 중 하나를 반환할 수 있습니다:

| 작업 | 설명 |
|--------|-------------|
| accept | 사용자가 유효한 입력을 제공했습니다. `content` 필드에 데이터를 포함하세요. |
| decline | 사용자가 요청된 정보를 제공하지 않기로 선택했습니다. |
| cancel | 사용자가 작업을 완전히 취소했습니다. |

```python title="응답 작업 예시"
# 데이터와 함께 수락
ElicitResult(action="accept", content={"email": "user@example.com", "age": 25})

# 거부(사용자가 정보를 제공하지 않으려고 함)
ElicitResult(action="decline")

# 취소(작업 중단)
ElicitResult(action="cancel")
```

## 추가 리소스

- [MCP 문서](https://modelcontextprotocol.io/)
- [MCP 전송 문서](https://modelcontextprotocol.io/docs/spec/basic/transports)
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
