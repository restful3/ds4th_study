# 메시지

메시지는 LangChain에서 모델을 위한 컨텍스트의 기본 단위입니다. 메시지는 모델의 입력 및 출력을 나타내며, LLM과 상호 작용할 때 대화의 상태를 나타내는 데 필요한 콘텐츠와 메타데이터를 모두 포함합니다.

메시지는 다음을 포함하는 객체입니다:

- **역할** - 메시지 유형을 식별합니다(예: system, user)
- **콘텐츠** - 메시지의 실제 콘텐츠를 나타냅니다(텍스트, 이미지, 오디오, 문서 등)
- **메타데이터** - 응답 정보, 메시지 ID, 토큰 사용량 같은 선택 사항 필드

LangChain은 모든 모델 제공자에게 작동하는 표준 메시지 유형을 제공하므로, 호출된 모델에 관계없이 일관된 동작을 보장합니다.

## 기본 사용법

메시지를 사용하는 가장 간단한 방법은 메시지 객체를 생성하고 호출할 때 모델로 전달하는 것입니다.

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Chat 모델과 함께 사용
messages = [system_msg, human_msg]
response = model.invoke(messages)  # AIMessage 반환
```

### 텍스트 프롬프트

텍스트 프롬프트는 문자열입니다. 대화 기록을 유지할 필요가 없는 간단한 생성 작업에 이상적입니다.

```python
response = model.invoke("Write a haiku about spring")
```

텍스트 프롬프트를 사용하세요:

- 단일, 독립 실행형 요청이 있을 때
- 대화 기록이 필요하지 않을 때
- 최소한의 코드 복잡성이 필요할 때

### 메시지 프롬프트

또는 메시지 객체의 목록을 제공하여 메시지 목록을 모델로 전달할 수 있습니다.

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```

메시지 프롬프트를 사용하세요:

- 다중 턴 대화를 관리할 때
- 멀티모달 콘텐츠(이미지, 오디오, 파일)로 작업할 때
- 시스템 지침을 포함할 때

### 딕셔너리 형식

OpenAI Chat 완료 형식으로 메시지를 직접 지정할 수도 있습니다.

```python
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
```

## 메시지 유형

- [System 메시지](#system-message) - 모델이 어떻게 동작할지 알려주고 상호 작용의 컨텍스트 제공
- [Human 메시지](#human-message) - 사용자 입력과 모델과의 상호 작용을 나타냅니다
- [AI 메시지](#ai-message) - 모델이 생성한 응답(텍스트 콘텐츠, Tool 호출 및 메타데이터 포함)
- [Tool 메시지](#tool-message) - [Tool 호출](07-models.md#tool-calling)의 출력을 나타냅니다

### System 메시지

`SystemMessage`는 모델의 동작을 준비하는 초기 지침 집합을 나타냅니다. System 메시지를 사용하여 톤을 설정하고, 모델의 역할을 정의하고, 응답 지침을 확립할 수 있습니다.

#### 기본 지침

```python
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

#### 상세한 페르소나

```python
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

### Human 메시지

`HumanMessage`는 사용자 입력과 상호 작용을 나타냅니다. 텍스트, 이미지, 오디오, 파일 및 기타 모든 멀티모달 콘텐츠를 포함할 수 있습니다.

#### 텍스트 콘텐츠

메시지 객체:

```python
response = model.invoke([
  HumanMessage("What is machine learning?")
])
```

문자열 바로 가기:

```python
response = model.invoke("What is machine learning?")
```

#### 메시지 메타데이터

메타데이터 추가:

```python
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # 선택 사항: 다양한 사용자 식별
    id="msg_123",  # 선택 사항: 추적을 위한 고유 식별자
)
```

> `name` 필드의 동작은 제공자에 따라 다릅니다. 일부는 사용자 식별에 사용하고, 다른 일부는 무시합니다. 확인하려면 모델 제공자의 참조를 참조하세요.

### AI 메시지

`AIMessage`는 모델 호출의 출력을 나타냅니다. 멀티모달 데이터, Tool 호출 및 나중에 접근할 수 있는 제공자별 메타데이터를 포함할 수 있습니다.

```python
response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>
```

`AIMessage` 객체는 모델을 호출할 때 반환되며, 응답의 모든 연관된 메타데이터를 포함합니다.

제공자는 메시지 유형을 다르게 가중치/문맥화하므로, 때때로 수동으로 새 AIMessage 객체를 만들고 모델에서 온 것처럼 메시지 기록에 삽입하는 것이 도움이 됩니다.

```python
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# AI 메시지를 수동으로 생성(예: 대화 기록)
ai_msg = AIMessage("I'd be happy to help you with that question!")

# 대화 기록에 추가
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # 모델에서 온 것처럼 삽입
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)
```

<details>
<summary>속성</summary>

-   `text` `문자열`

    메시지의 텍스트 콘텐츠입니다.

-   `content` `문자열 | dict[]`

    메시지의 원본 콘텐츠입니다.

-   `content_blocks` `ContentBlock[]`

    메시지의 표준화된 [콘텐츠 블록](#standard-content-blocks)입니다.

-   `tool_calls` `dict[] | None`

    모델이 수행한 Tool 호출입니다.

    Tool을 호출하지 않으면 비어있습니다.

-   `id` `문자열`

    메시지의 고유 식별자(LangChain에서 자동 생성되거나 제공자 응답에서 반환됨)

-   `usage_metadata` `dict | None`

    메시지의 사용량 메타데이터로, 사용 가능할 때 토큰 수를 포함할 수 있습니다.

-   `response_metadata` `ResponseMetadata | None`

    메시지의 응답 메타데이터입니다.

</details>

#### Tool 호출

모델이 Tool 호출을 수행하면, 이는 `AIMessage`에 포함됩니다:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    ...

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")
```

추론 또는 인용과 같은 다른 구조화된 데이터도 메시지 콘텐츠에 나타날 수 있습니다.

#### 토큰 사용량

`AIMessage`는 `usage_metadata` 필드에 토큰 수 및 기타 사용량 메타데이터를 보유할 수 있습니다:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

response = model.invoke("Hello!")
response.usage_metadata
```

```python
{'input_tokens': 8,
 'output_tokens': 304,
 'total_tokens': 312,
 'input_token_details': {'audio': 0, 'cache_read': 0},
 'output_token_details': {'audio': 0, 'reasoning': 256}}
```

자세한 내용은 [UsageMetadata](https://reference.langchain.com/python/langchain/messages/#langchain.messages.UsageMetadata)를 참조하세요.

#### 스트리밍 및 청크

스트리밍 중에, 완전한 메시지 객체로 결합할 수 있는 `AIMessageChunk` 객체를 받게 됩니다:

```python
chunks = []
full_message = None
for chunk in model.stream("Hi"):
    chunks.append(chunk)
    print(chunk.text)
    full_message = chunk if full_message is None else full_message + chunk
```

자세히 알아보기:
- [Chat 모델에서 토큰 스트리밍](07-models.md#stream)
- [Agent에서 토큰 및/또는 단계 스트리밍](streaming.md)

### Tool 메시지

Tool 호출을 지원하는 모델의 경우, AI 메시지는 Tool 호출을 포함할 수 있습니다. Tool 메시지는 단일 Tool 실행의 결과를 모델로 다시 전달하는 데 사용됩니다.

Tool은 `ToolMessage` 객체를 직접 생성할 수 있습니다. 아래에 간단한 예시를 보여줍니다. [Tool 가이드](tools.md)에서 더 알아보세요.

```python
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

# 모델이 Tool 호출을 한 후
# (여기서 우리는 간단히 하기 위해 메시지를 수동으로 생성합니다)
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# Tool을 실행하고 결과 메시지 생성
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # 호출 ID와 일치해야 합니다
)

# 대화 계속
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # 모델의 Tool 호출
    tool_message,  # Tool 실행 결과
]
response = model.invoke(messages)  # 모델이 결과 처리
```

<details>
<summary>속성</summary>

-   `content` `문자열` `필수`

    Tool 호출의 문자열화된 출력입니다.

-   `tool_call_id` `문자열` `필수`

    이 메시지가 응답하는 Tool 호출의 ID입니다.

    `AIMessage`의 Tool 호출 ID와 일치해야 합니다.

-   `name` `문자열` `필수`

    호출된 Tool의 이름입니다.

-   `artifact` `dict`

    모델로 전송되지 않지만 프로그래매틱하게 접근 가능한 보조 데이터(예: 원본 결과, 디버깅 정보).

</details>

> artifact 필드는 모델로 전송되지 않지만 프로그래매틱하게 접근할 수 있는 보조 데이터를 저장합니다. 이는 모델의 컨텍스트를 복잡하게 하지 않으면서 원본 결과, 디버깅 정보 또는 다운스트림 처리 데이터를 저장하는 데 유용합니다.

<details>
<summary>예시: 검색 메타데이터에 artifact 사용</summary>

예를 들어, 검색 Tool은 모델의 참조를 위해 문서에서 구절을 검색할 수 있습니다. 메시지 콘텐츠가 모델이 참조할 텍스트를 포함하는 경우, artifact에는 응용 프로그램이 사용할 수 있는 문서 식별자 또는 기타 메타데이터가 포함될 수 있습니다(예: 페이지를 렌더링하기 위해). 아래 예시를 참조하세요:

```python
from langchain.messages import ToolMessage

# 모델로 전송됨
message_content = "It was the best of times, it was the worst of times."

# 다운스트림에서 사용 가능
artifact = {"document_id": "doc_123", "page": 0}

tool_message = ToolMessage(
    content=message_content,
    tool_call_id="call_123",
    name="search_books",
    artifact=artifact,
)
```
LangChain을 사용한 검색 Agent 구축의 엔드투엔드 예시는 RAG 튜토리얼을 참조하세요.

</details>

## 메시지 콘텐츠

메시지의 콘텐츠를 모델로 전송되는 데이터의 페이로드로 생각할 수 있습니다. 메시지는 느슨하게 타입화된 `content` 속성을 가지고 있어서, 문자열과 타입이 지정되지 않은 객체 목록(예: 딕셔너리)을 지원합니다. 이것은 LangChain Chat 모델에서 멀티모달 콘텐츠 및 기타 데이터와 같은 제공자 기본 구조를 직접 지원할 수 있게 합니다.

별도로, LangChain은 텍스트, 추론, 인용, 멀티모달 데이터, 서버 측 Tool 호출 및 기타 메시지 콘텐츠에 대한 전용 콘텐츠 유형을 제공합니다. 아래의 [콘텐츠 블록](#standard-content-blocks)을 참조하세요.

LangChain Chat 모델은 `content` 속성의 메시지 콘텐츠를 수용합니다.

여기에는 다음 중 하나가 포함될 수 있습니다:

1. 문자열
2. 제공자 기본 형식의 콘텐츠 블록 목록
3. LangChain의 표준 콘텐츠 블록 목록

멀티모달 입력을 사용하는 예시는 아래를 참조하세요:

```python
from langchain.messages import HumanMessage

# 문자열 콘텐츠
human_message = HumanMessage("Hello, how are you?")

# 제공자 기본 형식(예: OpenAI)
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# 표준 콘텐츠 블록 목록
human_message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
```

> 메시지를 초기화할 때 `content_blocks`을 지정하면 여전히 메시지 콘텐츠를 채우지만, 이를 수행하기 위한 타입 안전 인터페이스를 제공합니다.

### 표준 콘텐츠 블록

LangChain은 제공자 전체에서 작동하는 메시지 콘텐츠의 표준 표현을 제공합니다.

메시지 객체는 `content` 속성을 표준, 타입 안전 표현으로 지연 파싱하는 `content_blocks` 속성을 구현합니다. 예를 들어, ChatAnthropic 또는 ChatOpenAI에서 생성된 메시지에는 각 제공자의 형식의 생각 또는 추론 블록이 포함되지만, 일관된 `ReasoningContentBlock` 표현으로 지연 파싱될 수 있습니다:

**Anthropic**

```python
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
        {"type": "text", "text": "..."},
    ],
    response_metadata={"model_provider": "anthropic"}
)
message.content_blocks
```

```python
[{'type': 'reasoning',
  'reasoning': '...',
  'extras': {'signature': 'WaUjzkyp...'}},
 {'type': 'text', 'text': '...'}]
```

**OpenAI**

```python
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {
            "type": "reasoning",
            "id": "rs_abc123",
            "summary": [
                {"type": "summary_text", "text": "summary 1"},
                {"type": "summary_text", "text": "summary 2"},
            ],
        },
        {"type": "text", "text": "...", "id": "msg_abc123"},
    ],
    response_metadata={"model_provider": "openai"}
)
message.content_blocks
```
```
[{'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 1'},
 {'type': 'reasoning', 'id': 'rs_abc123', 'reasoning': 'summary 2'},
 {'type': 'text', 'text': '...', 'id': 'msg_abc123'}]
```

[통합 가이드](https://docs.langchain.com/oss/python/integrations/providers/overview)를 참조하여 선택한 추론 제공자로 시작하세요.

#### 표준 콘텐츠 직렬화

LangChain 외부의 응용 프로그램이 표준 콘텐츠 블록 표현에 접근해야 할 경우, 메시지 콘텐츠에 콘텐츠 블록 저장을 선택할 수 있습니다.

이를 수행하려면, `LC_OUTPUT_VERSION` 환경 변수를 `v1`로 설정하거나 모든 Chat 모델을 `output_version="v1"`로 초기화할 수 있습니다:

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano", output_version="v1")
```

### 멀티모달

멀티모달은 텍스트, 오디오, 이미지 및 비디오와 같은 다양한 형태의 데이터로 작업하는 능력을 의미합니다. LangChain은 제공자 전체에서 사용할 수 있는 이러한 데이터에 대한 표준 유형을 포함합니다.

Chat 모델은 입력으로 멀티모달 데이터를 수용하고 출력으로 생성할 수 있습니다. 아래에서 멀티모달 데이터를 포함하는 입력 메시지의 짧은 예시를 보여줍니다.

> 추가 키는 콘텐츠 블록의 상단 레벨이나 `"extras": {"key": value}` 안에 중첩될 수 있습니다.
>
> OpenAI 및 AWS Bedrock Converse는 예를 들어 PDF의 파일 이름이 필요합니다. 선택한 모델에 대한 제공자 페이지를 참조하여 자세한 내용을 확인하세요.

#### 이미지 입력

```python
# URL에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# base64 데이터에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}

# 제공자 관리 파일 ID에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
```

#### PDF 문서 입력

```python
# URL에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {"type": "file", "url": "https://example.com/path/to/document.pdf"},
    ]
}

# base64 데이터에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {
            "type": "file",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "application/pdf",
        },
    ]
}

# 제공자 관리 파일 ID에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this document."},
        {"type": "file", "file_id": "file-abc123"},
    ]
}
```

#### 오디오 입력

```python
# base64 데이터에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {
            "type": "audio",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "audio/wav",
        },
    ]
}

# 제공자 관리 파일 ID에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {"type": "audio", "file_id": "file-abc123"},
    ]
}
```

#### 비디오 입력

```python
# base64 데이터에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this video."},
        {
            "type": "video",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "video/mp4",
        },
    ]
}

# 제공자 관리 파일 ID에서
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this video."},
        {"type": "video", "file_id": "file-abc123"},
    ]
}
```

> 모든 모델이 모든 파일 유형을 지원하지는 않습니다. 모델 제공자의 참조를 확인하여 지원되는 형식 및 크기 제한을 확인하세요.

### 콘텐츠 블록 참조

콘텐츠 블록은 (메시지를 생성할 때 또는 `content_blocks` 속성에 접근할 때) 타입화된 딕셔너리 목록으로 표현됩니다. 목록의 각 항목은 다음 블록 유형 중 하나를 준수해야 합니다:

<details>
<summary>코어</summary>

<details>
<summary>TextContentBlock</summary>

목적: 표준 텍스트 출력

-   `type` `"문자열"` `필수`

    항상 "text"

-   `text` `문자열` `필수`

    텍스트 콘텐츠입니다.

-   `annotations` `객체[]`

    텍스트에 대한 주석 목록입니다.

-   `extras` `객체`

    추가 제공자별 데이터

예시:
```
{
    "type": "text",
    "text": "Hello world",
    "annotations": []
}
```

</details>

<details>
<summary>ReasoningContentBlock</summary>

목적: 모델 추론 단계

-   `type` `"문자열"` `필수`

    항상 "reasoning"

-   `reasoning` `문자열`

    추론 콘텐츠입니다.

-   `extras` `객체`

    추가 제공자별 데이터

예시:
```
{
    "type": "reasoning",
    "reasoning": "The user is asking about...",
    "extras": {"signature": "abc123"},
}
```

</details>

</details>

<details>
<summary>멀티모달</summary>

<details>
<summary>ImageContentBlock</summary>

목적: 이미지 데이터

-   `type` `"문자열"` `필수`

    항상 "image"

-   `url` `문자열`

    이미지 위치를 가리키는 URL입니다.

-   `base64` `문자열`

    Base64 인코딩된 이미지 데이터입니다.

-   `id` `문자열`

    이 콘텐츠 블록의 고유 식별자(제공자 또는 LangChain에서 생성).

-   `mime_type` `문자열`

    이미지 MIME 유형(예: image/jpeg, image/png). Base64 데이터의 경우 필수입니다.

</details>

<details>
<summary>AudioContentBlock</summary>

목적: 오디오 데이터

-   `type` `"문자열"` `필수`

    항상 "audio"

-   `url` `문자열`

    오디오 위치를 가리키는 URL입니다.

-   `base64` `문자열`

    Base64 인코딩된 오디오 데이터입니다.

-   `id` `문자열`

    이 콘텐츠 블록의 고유 식별자(제공자 또는 LangChain에서 생성).

-   `mime_type` `문자열`

    오디오 MIME 유형(예: audio/mpeg, audio/wav). Base64 데이터의 경우 필수입니다.

</details>

<details>
<summary>VideoContentBlock</summary>

목적: 비디오 데이터

-   `type` `"문자열"` `필수`

    항상 "video"

-   `url` `문자열`

    비디오 위치를 가리키는 URL입니다.

-   `base64` `문자열`

    Base64 인코딩된 비디오 데이터입니다.

-   `id` `문자열`

    이 콘텐츠 블록의 고유 식별자(제공자 또는 LangChain에서 생성).

-   `mime_type` `문자열`

    비디오 MIME 유형(예: video/mp4, video/quicktime). Base64 데이터의 경우 필수입니다.

</details>

<details>
<summary>FileContentBlock</summary>

목적: 일반 파일(PDF 등)

-   `type` `"문자열"` `필수`

    항상 "file"

-   `url` `문자열`

    파일 위치를 가리키는 URL입니다.

-   `base64` `문자열`

    Base64 인코딩된 파일 데이터입니다.

-   `id` `문자열`

    이 콘텐츠 블록의 고유 식별자(제공자 또는 LangChain에서 생성).

-   `mime_type` `문자열`

    파일 MIME 유형(예: application/pdf). Base64 데이터의 경우 필수입니다.

</details>

<details>
<summary>PlainTextContentBlock</summary>

목적: 문서 텍스트(.txt, .md)

-   `type` `"문자열"` `필수`

    항상 "text-plain"

-   `text` `문자열`

    텍스트 콘텐츠입니다.

-   `mime_type` `문자열`

    텍스트 MIME 유형(예: text/plain, text/markdown)

</details>

</details>

<details>
<summary>Tool 호출</summary>

<details>
<summary>ToolCall</summary>

목적: Tool 호출

-   `type` `"문자열"` `필수`

    항상 "tool_call"

-   `name` `문자열` `필수`

    호출할 Tool의 이름

-   `args` `객체` `필수`

    Tool 호출에 대한 인수입니다.

-   `id` `문자열` `필수`

    이 Tool 호출의 고유 식별자

예시:
```json
{
    "type": "tool_call",
    "name": "search",
    "args": {"query": "weather"},
    "id": "call_123"
}
```

</details>

<details>
<summary>ToolCallChunk</summary>

목적: Tool 호출 스트리밍 조각

-   `type` `"문자열"` `필수`

    항상 "tool_call_chunk"

-   `name` `문자열`

    호출되는 Tool의 이름

-   `args` `문자열`

    부분 Tool 인수(불완전한 JSON일 수 있음)

-   `id` `문자열`

    Tool 호출 식별자

-   `index` `숫자 | 문자열`

    스트림에서 이 청크의 위치

</details>

<details>
<summary>InvalidToolCall</summary>

목적: 잘못된 호출로, JSON 파싱 오류를 포착합니다.

-   `type` `"문자열"` `필수`

    항상 "invalid_tool_call"

-   `name` `문자열`

    호출되지 못한 Tool의 이름

-   `args` `객체`

    Tool로 전달할 인수

-   `error` `문자열`

    무엇이 잘못되었는지에 대한 설명입니다.

</details>

</details>

<details>
<summary>서버 측 Tool 실행</summary>

<details>
<summary>ServerToolCall</summary>

목적: 서버 측에서 실행되는 Tool 호출입니다.

-   `type` `"문자열"` `필수`

    항상 "server_tool_call"

-   `id` `문자열` `필수`

    Tool 호출과 연결된 식별자입니다.

-   `name` `문자열` `필수`

    호출할 Tool의 이름입니다.

-   `args` `객체` `필수`

    부분 Tool 인수(불완전한 JSON일 수 있음)

</details>

<details>
<summary>ServerToolCallChunk</summary>

목적: Tool 호출 스트리밍 서버 측 조각

-   `type` `"문자열"` `필수`

    항상 "server_tool_call_chunk"

-   `id` `문자열`

    Tool 호출과 연결된 식별자입니다.

-   `name` `문자열`

    호출되는 Tool의 이름

-   `args` `문자열`

    부분 Tool 인수(불완전한 JSON일 수 있음)

-   `index` `숫자 | 문자열`

    스트림에서 이 청크의 위치

</details>

<details>
<summary>ServerToolResult</summary>

목적: 검색 결과

-   `type` `"문자열"` `필수`

    항상 "server_tool_result"

-   `tool_call_id` `문자열` `필수`

    해당 서버 Tool 호출의 식별자입니다.

-   `id` `문자열`

    서버 Tool 결과와 연결된 식별자입니다.

-   `status` `문자열` `필수`

    서버 측 Tool의 실행 상태입니다. "success" 또는 "error"입니다.

-   `output`

    실행된 Tool의 출력입니다.

</details>

</details>

<details>
<summary>제공자별 블록</summary>

<details>
<summary>NonStandardContentBlock</summary>

목적: 제공자별 이스케이프 해치

-   `type` `"문자열"` `필수`

    항상 "non_standard"

-   `value` `객체` `필수`

    제공자별 데이터 구조

사용법: 실험적 또는 제공자별 기능용

</details>

각 모델 제공자의 참조 문서 내에서 추가 제공자별 콘텐츠 유형이 있을 수 있습니다.

</details>

> [API 참조](https://reference.langchain.com/python/langchain/messages)에서 정식 타입 정의를 확인하세요.

> 콘텐츠 블록은 LangChain v1의 메시지의 새로운 속성으로 도입되었으며, 기존 코드와의 역호환성을 유지하면서 제공자 전체에서 콘텐츠 형식을 표준화합니다.
>
> 콘텐츠 블록은 `content` 속성을 대체하지 않으며, 표준화된 형식으로 메시지의 콘텐츠에 접근하는 데 사용할 수 있는 새로운 속성입니다.

## Chat 모델과 함께 사용

Chat 모델은 입력으로 메시지 객체의 시퀀스를 수용하고 출력으로 `AIMessage`를 반환합니다. 상호 작용은 종종 상태가 없으므로, 간단한 대화 루프는 메시지의 증가하는 목록으로 모델을 호출하는 것을 포함합니다.

자세히 알아보려면 아래 가이드를 참조하세요:

- [대화 기록 지속 및 관리](short-term-memory.md)를 위한 기본 제공 기능
- [컨텍스트 윈도우 관리](short-term-memory.md#managing-context-window)를 위한 전략(메시지 정리 및 요약 포함)
