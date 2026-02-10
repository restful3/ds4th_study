# 기본 제공 Middleware

일반적인 Agent 사용 사례를 위한 사전 구축된 Middleware

LangChain은 일반적인 사용 사례를 위한 사전 구축된 Middleware를 제공합니다. 각 Middleware는 프로덕션 준비가 완료되었으며 특정 필요에 맞게 구성할 수 있습니다.

---

## 제공자 불가지론적 Middleware

다음 Middleware는 모든 LLM 제공자와 함께 작동합니다:

| Middleware | 설명 |
|------------|-------------|
| [요약](#summarization) | 토큰 제한에 접근할 때 자동으로 대화 기록을 요약합니다. |
| [Human-in-the-loop](#human-in-the-loop) | Tool 호출의 Human 승인을 위해 실행을 일시 중지합니다. |
| [모델 호출 제한](#model-call-limit) | 과도한 비용을 방지하기 위해 모델 호출 수를 제한합니다. |
| [Tool 호출 제한](#tool-call-limit) | 호출 수를 제한하여 Tool 실행을 제어합니다. |
| [모델 폴백](#model-fallback) | 기본 모델이 실패할 때 자동으로 대체 모델로 전환합니다. |
| [PII 감지](#pii-detection) | 개인식별정보(PII)를 감지하고 처리합니다. |
| [할일 목록](#to-do-list) | Agent에 작업 계획 및 추적 기능을 제공합니다. |
| [LLM Tool 선택기](#llm-tool-selector) | 주 모델을 호출하기 전에 LLM을 사용하여 관련 Tool을 선택합니다. |
| [Tool 재시도](#tool-retry) | 실패한 Tool 호출을 지수 백오프를 사용하여 자동으로 재시도합니다. |
| [모델 재시도](#model-retry) | 실패한 모델 호출을 지수 백오프를 사용하여 자동으로 재시도합니다. |
| [LLM Tool 에뮬레이터](#llm-tool-emulator) | 테스트 목적으로 LLM을 사용하여 Tool 실행을 에뮬레이트합니다. |
| [컨텍스트 편집](#context-editing) | Tool 사용을 정리하거나 지워서 대화 컨텍스트를 관리합니다. |
| [Shell Tool](#shell-tool) | Agent에 명령 실행을 위한 지속적인 shell 세션을 노출합니다. |
| [파일 검색](#file-search) | 파일 시스템 파일을 통해 Glob 및 Grep 검색 Tool을 제공합니다. |

---

## 요약

토큰 제한에 접근할 때 대화 기록을 자동으로 요약하여 최근 메시지를 보존하면서 이전 컨텍스트를 압축합니다. 요약은 다음의 경우에 유용합니다:

- 컨텍스트 윈도우를 초과하는 오래된 대화
- 광범위한 기록이 있는 다중 턴 대화
- 완전한 대화 컨텍스트를 유지하는 것이 중요한 응용 프로그램

**API 참조:** [SummarizationMiddleware](https://reference.langchain.com/python/langchain/middleware/SummarizationMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

> [!TIP]
> `langchain>=1.1`을 사용하는 경우 `trigger` 및 `keep`에 대한 `fraction` 조건은 Chat 모델의 [프로필 데이터](https://reference.langchain.com/python/langchain/chat_models/profile)에 의존합니다. 데이터를 사용할 수 없으면 다른 조건을 사용하거나 수동으로 지정하세요:

```python
from langchain.chat_models import init_chat_model

custom_profile = {
    "max_input_tokens": 100_000,
    # ...
}
model = init_chat_model("gpt-4.1", profile=custom_profile)
```

### model

`문자열 | BaseChatModel` **필수**

요약을 생성하기 위한 모델입니다. 모델 식별자 문자열(예: `'openai:gpt-4.1-mini'`) 또는 `BaseChatModel` 인스턴스입니다. 자세한 내용은 [init_chat_model](https://reference.langchain.com/python/langchain/chat_models/init_chat_model)을 참조하세요.

### trigger

`ContextSize | list[ContextSize] | None`

요약을 트리거하기 위한 조건입니다. 다음 중 하나입니다:

- 단일 `ContextSize` 튜플(지정된 조건을 충족해야 함)
- `ContextSize` 튜플 목록(모든 조건을 충족해야 함 - OR 로직)

조건은 다음 중 하나여야 합니다:

- `fraction` (float): 모델의 컨텍스트 크기의 분수 (0-1)
- `tokens` (int): 절대 토큰 수
- `messages` (int): 메시지 수

최소한 하나의 조건이 지정되어야 합니다. 제공되지 않으면 요약이 자동으로 트리거되지 않습니다.

자세한 내용은 [ContextSize](https://reference.langchain.com/python/langchain/middleware/ContextSize)에 대한 API 참조를 참조하세요.

### keep

`ContextSize` **기본값:** `"('messages', 20)"`

요약 후 보존할 컨텍스트의 양입니다. 정확히 하나를 지정하세요:

- `fraction` (float): 보존할 모델 컨텍스트 크기의 분수 (0-1)
- `tokens` (int): 보존할 절대 토큰 수
- `messages` (int): 보존할 최근 메시지 수

자세한 내용은 [ContextSize](https://reference.langchain.com/python/langchain/middleware/ContextSize)에 대한 API 참조를 참조하세요.

### token_counter

`function`

사용자 정의 토큰 계산 함수입니다. 기본값은 문자 기반 계산입니다.

### summary_prompt

`문자열`

요약을 위한 사용자 정의 프롬프트 템플릿입니다. 지정되지 않으면 기본 제공 템플릿을 사용합니다. 템플릿에는 대화 기록이 삽입될 `{messages}` 자리 표시자가 포함되어야 합니다.

### trim_tokens_to_summarize

`숫자` **기본값:** `"4000"`

요약 생성 시 포함할 최대 토큰 수입니다. 요약 전에 이 제한에 맞게 메시지가 정리됩니다.

### summary_prefix

`문자열` **deprecated**

Deprecated: 전체 프롬프트를 제공하려면 `summary_prompt`를 사용하세요.

### max_tokens_before_summary

`숫자` **deprecated**

Deprecated: 대신 `trigger: ("tokens", value)`를 사용하세요. 요약을 트리거하기 위한 토큰 임계값입니다.

### messages_to_keep

`숫자` **deprecated**

Deprecated: 대신 `keep: ("messages", value)`를 사용하세요. 보존할 최근 메시지입니다.

</details>

<details>
<summary>전체 예시</summary>

요약 Middleware는 메시지 토큰 수를 모니터링하고 임계값에 도달하면 자동으로 이전 메시지를 요약합니다.

**트리거 조건**은 요약 실행 시기를 제어합니다:

- 단일 조건 객체(지정된 것을 충족해야 함)
- 조건 배열(모든 조건을 충족해야 함 - OR 로직)
- 각 조건은 `fraction` (모델의 컨텍스트 크기), `tokens` (절대 수) 또는 `messages` (메시지 수)를 사용할 수 있습니다

**Keep 조건**은 보존할 컨텍스트 양을 제어합니다(정확히 하나 지정):

- `fraction` - 보존할 모델 컨텍스트 크기의 분수
- `tokens` - 보존할 절대 토큰 수
- `messages` - 보존할 최근 메시지 수

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# 단일 조건: 토큰 >= 4000인 경우 트리거
agent = create_agent(
    model="gpt-4.1",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)

# 다중 조건: 토큰 수 >= 3000 또는 메시지 >= 6인 경우 트리거
agent2 = create_agent(
    model="gpt-4.1",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=[
                ("tokens", 3000),
                ("messages", 6),
            ],
            keep=("messages", 20),
        ),
    ],
)

# 분수 제한 사용
agent3 = create_agent(
    model="gpt-4.1",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4.1-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)
```

</details>

---

## Human-in-the-loop

Human 승인, 편집 또는 거부를 위해 Agent 실행을 일시 중지합니다. 실행 전 Tool 호출을 처리합니다. Human-in-the-loop은 다음의 경우에 유용합니다:

- 높은 위험 작업(데이터베이스 쓰기, 재정 거래 등)에 대해 Human 승인이 필요한 경우
- Human 감시가 필수인 규정 준수 워크플로우
- Human 피드백이 Agent를 안내하는 오래된 대화

**API 참조:** [HumanInTheLoopMiddleware](https://reference.langchain.com/python/langchain/middleware/HumanInTheLoopMiddleware)

> [!INFO]
> Human-in-the-loop Middleware는 인터럽트 전반에 상태를 유지하기 위해 checkpointer가 필요합니다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def read_email_tool(email_id: str) -> str:
    """ID로 메일을 읽기 위한 Mock 함수입니다."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """메일을 보내기 위한 Mock 함수입니다."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4.1",
    tools=[your_read_email_tool, your_send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "your_send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "your_read_email_tool": False,
            }
        ),
    ],
)
```

> 완전한 예시, 구성 옵션 및 통합 패턴은 [Human-in-the-loop 문서](/oss/python/langchain/human-in-the-loop)를 참조하세요.

> Human-in-the-loop Middleware 동작을 보여주는 이 비디오 가이드를 시청하세요.

---

## 모델 호출 제한

무한 루프 또는 과도한 비용을 방지하기 위해 모델 호출 수를 제한합니다. 모델 호출 제한은 다음의 경우에 유용합니다:

- 폭주하는 Agent가 너무 많은 API 호출을 하는 것을 방지합니다.
- 프로덕션 배포에서 비용 제어를 시행합니다.
- 특정 호출 예산 내에서 Agent 동작을 테스트합니다.

**API 참조:** [ModelCallLimitMiddleware](https://reference.langchain.com/python/langchain/middleware/ModelCallLimitMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4.1",
    checkpointer=InMemorySaver(),  # 스레드 제한에 필수
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
```

> 모델 호출 제한 Middleware 동작을 보여주는 이 비디오 가이드를 시청하세요.

<details>
<summary>구성 옵션</summary>

### thread_limit

`숫자`

스레드의 모든 실행에서 최대 모델 호출입니다. 기본값은 제한 없음입니다.

### run_limit

`숫자`

단일 호출당 최대 모델 호출입니다. 기본값은 제한 없음입니다.

### exit_behavior

`문자열` **기본값:** `"end"`

제한에 도달했을 때의 동작입니다. 옵션: `'end'` (정상 종료) 또는 `'error'` (예외 발생)

</details>

---

## Tool 호출 제한

모든 Tool에 전역적으로 또는 특정 Tool에 대해 Tool 호출 수를 제한하여 Agent 실행을 제어합니다. Tool 호출 제한은 다음의 경우에 유용합니다:

- 비용이 많이 드는 외부 API로의 과도한 호출을 방지합니다.
- 웹 검색 또는 데이터베이스 쿼리를 제한합니다.
- 특정 Tool 사용에서 속도 제한을 시행합니다.
- 폭주하는 Agent 루프로부터 보호합니다.

**API 참조:** [ToolCallLimitMiddleware](https://reference.langchain.com/python/langchain/middleware/ToolCallLimitMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool],
    middleware=[
        # 전역 제한
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        # Tool별 제한
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        ),
    ],
)
```

> Tool 호출 제한 Middleware 동작을 보여주는 이 비디오 가이드를 시청하세요.

<details>
<summary>구성 옵션</summary>

### tool_name

`문자열`

제한할 특정 Tool의 이름입니다. 제공되지 않으면 제한이 모든 Tool에 전역적으로 적용됩니다.

### thread_limit

`숫자`

스레드(대화)의 모든 실행에서 최대 Tool 호출입니다. 동일한 스레드 ID를 가진 여러 호출에서 유지됩니다. 상태를 유지하기 위해 checkpointer가 필요합니다. `None`은 스레드 제한이 없음을 의미합니다.

### run_limit

`숫자`

단일 호출당 최대 Tool 호출입니다(하나의 사용자 메시지 → 응답 주기). 각 새로운 사용자 메시지로 재설정됩니다. `None`은 실행 제한이 없음을 의미합니다.

> [!INFO]
> 최소한 `thread_limit` 또는 `run_limit` 중 하나가 지정되어야 합니다.

### exit_behavior

`문자열` **기본값:** `"continue"`

제한에 도달했을 때의 동작입니다:

- `'continue'` (기본값) - 초과 Tool 호출을 오류 메시지로 차단하고, 다른 Tool과 모델을 계속할 수 있습니다. 모델은 오류 메시지를 기반으로 종료 시기를 결정합니다.
- `'error'` - `ToolCallLimitExceededError` 예외를 발생시켜 즉시 실행을 중지합니다.
- `'end'` - 초과 Tool 호출에 대해 `ToolMessage`와 AI 메시지를 사용하여 즉시 실행을 중지합니다. 단일 Tool 제한만 작동합니다. 다른 Tool에 보류 중인 호출이 있으면 `NotImplementedError`를 발생시킵니다.

</details>

<details>
<summary>전체 예시</summary>

다음으로 제한을 지정하세요:

- **스레드 제한** - 대화의 모든 실행에서 최대 호출(checkpointer 필요)
- **실행 제한** - 단일 호출당 최대 호출(각 턴마다 재설정)

출력 동작:

- `'continue'` (기본값) - 초과 호출을 오류 메시지로 차단, Agent는 계속됨
- `'error'` - 즉시 예외 발생
- `'end'` - ToolMessage + AI 메시지로 중지(단일 Tool 시나리오만)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
search_limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)
database_limiter = ToolCallLimitMiddleware(tool_name="query_database", thread_limit=10)
strict_limiter = ToolCallLimitMiddleware(tool_name="scrape_webpage", run_limit=2, exit_behavior="error")

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool, scraper_tool],
    middleware=[global_limiter, search_limiter, database_limiter, strict_limiter],
)
```

</details>

---

## 모델 폴백

기본 모델이 실패할 때 자동으로 대체 모델로 전환합니다. 모델 폴백은 다음의 경우에 유용합니다:

- 모델 중단을 처리하는 탄력적 Agent 구축
- 저렴한 모델로 폴백하여 비용 최적화
- OpenAI, Anthropic 등 제공자 전체에서 중복성 확보

**API 참조:** [ModelFallbackMiddleware](https://reference.langchain.com/python/langchain/middleware/ModelFallbackMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4.1-mini",
            "claude-3-5-sonnet-20241022",
        ),
    ],
)
```

> 모델 폴백 Middleware 동작을 보여주는 이 비디오 가이드를 시청하세요.

<details>
<summary>구성 옵션</summary>

### first_model

`문자열 | BaseChatModel` **필수**

기본 모델이 실패할 때 시도할 첫 번째 폴백 모델입니다. 모델 식별자 문자열(예: `'openai:gpt-4.1-mini'`) 또는 `BaseChatModel` 인스턴스입니다.

### *additional_models

`문자열 | BaseChatModel`

이전 모델이 실패할 경우 순서대로 시도할 추가 폴백 모델입니다.

</details>

---

## PII 감지

구성 가능한 전략을 사용하여 대화에서 개인식별정보(PII)를 감지하고 처리합니다. PII 감지는 다음의 경우에 유용합니다:

- 규정 준수 요구 사항이 있는 의료 및 재무 응용 프로그램
- 로그를 정제해야 하는 고객 서비스 Agent
- 민감한 사용자 데이터를 처리하는 모든 응용 프로그램

**API 참조:** [PIIMiddleware](https://reference.langchain.com/python/langchain/middleware/PIIMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
```

### 사용자 정의 PII 유형

`detector` 매개변수를 제공하여 사용자 정의 PII 유형을 만들 수 있습니다. 이를 통해 기본 제공 유형 이상으로 사용 사례에 맞는 패턴을 감지할 수 있습니다.

사용자 정의 감지기를 만드는 세 가지 방법:

1. **Regex 패턴 문자열** - 간단한 패턴 일치
2. **사용자 정의 함수** - 검증을 포함한 복잡한 감지 로직

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
import re

# 방법 1: Regex 패턴 문자열
agent1 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        ),
    ],
)

# 방법 2: 컴파일된 Regex 패턴
agent2 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware(
            "phone_number",
            detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"),
            strategy="mask",
        ),
    ],
)

# 방법 3: 사용자 정의 감지기 함수
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """검증으로 SSN을 감지합니다.
    'text', 'start' 및 'end' 키가 있는 딕셔너리 목록을 반환합니다.
    """
    import re
    matches = []
    pattern = r"\d{3}-\d{2}-\d{4}"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        # 검증: 처음 3자리는 000, 666 또는 900-999이 아니어야 합니다
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text": ssn,
                "start": match.start(),
                "end": match.end(),
            })
    return matches

agent3 = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        PIIMiddleware(
            "ssn",
            detector=detect_ssn,
            strategy="hash",
        ),
    ],
)
```

**사용자 정의 감지기 함수 서명:**

감지기 함수는 문자열을 수용하고 일치하는 항목을 반환해야 합니다:

`text`, `start` 및 `end` 키가 있는 딕셔너리 목록을 반환합니다:

```python
def detector(content: str) -> list[dict[str, str | int]]:
    return [
        {"text": "matched_text", "start": 0, "end": 12},
        # ... 더 많은 일치 항목
    ]
```

사용자 정의 감지기의 경우:

- 간단한 패턴에는 Regex 문자열을 사용하세요
- 플래그가 필요할 때(예: 대소문자를 구분하지 않는 일치) RegExp 객체를 사용하세요
- 패턴 일치 이상의 검증 로직이 필요할 때 사용자 정의 함수를 사용하세요

사용자 정의 함수는 감지 로직을 완전히 제어할 수 있으며 패턴 일치를 넘어 복잡한 검증 규칙을 구현할 수 있습니다.

<details>
<summary>구성 옵션</summary>

### pii_type

`문자열` **필수**

감지할 PII의 유형입니다. 기본 제공 유형(`email`, `credit_card`, `ip`, `mac_address`, `url`) 또는 사용자 정의 유형 이름입니다.

### strategy

`문자열` **기본값:** `"redact"`

감지된 PII를 처리하는 방법입니다. 옵션:

- `'block'` - 감지되면 예외 발생
- `'redact'` - `[REDACTED_{PII_TYPE}]`로 바꾸기
- `'mask'` - 부분적으로 마스킹(예: `****-****-****-1234`)
- `'hash'` - 결정론적 해시로 바꾸기

### detector

`함수 | regex`

사용자 정의 감지기 함수 또는 Regex 패턴입니다. 제공되지 않으면 PII 유형에 대한 기본 제공 감지기를 사용합니다.

### apply_to_input

`boolean` **기본값:** `"True"`

모델 호출 전 사용자 메시지를 확인합니다.

### apply_to_output

`boolean` **기본값:** `"False"`

모델 호출 후 AI 메시지를 확인합니다.

### apply_to_tool_results

`boolean` **기본값:** `"False"`

실행 후 Tool 결과 메시지를 확인합니다.

</details>

---

## 할일 목록

복잡한 다단계 작업을 위해 Agent에 작업 계획 및 추적 기능을 제공합니다. 할일 목록은 다음의 경우에 유용합니다:

- 여러 Tool 전체에 조정이 필요한 복잡한 다단계 작업
- 진행 상황 가시성이 중요한 오래된 작업

> 이 Middleware는 자동으로 Agent에 `write_todos` Tool과 효과적인 작업 계획을 안내하는 시스템 프롬프트를 제공합니다.

**API 참조:** [TodoListMiddleware](https://reference.langchain.com/python/langchain/middleware/TodoListMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
```

> 할일 목록 Middleware 동작을 보여주는 이 비디오 가이드를 시청하세요.

<details>
<summary>구성 옵션</summary>

### system_prompt

`문자열`

할일 사용을 안내하기 위한 사용자 정의 시스템 프롬프트입니다. 지정되지 않으면 기본 제공 프롬프트를 사용합니다.

### tool_description

`문자열`

`write_todos` Tool에 대한 사용자 정의 설명입니다. 지정되지 않으면 기본 제공 설명을 사용합니다.

</details>

---

## LLM Tool 선택기

주 모델을 호출하기 전에 LLM을 사용하여 관련 Tool을 지능적으로 선택합니다. LLM Tool 선택기는 다음의 경우에 유용합니다:

- Tool이 많은 Agent(10개 이상)에서 대부분이 쿼리당 관련되지 않은 경우
- 관련되지 않은 Tool을 필터링하여 토큰 사용 줄이기
- 모델 포커스 및 정확도 개선

이 Middleware는 구조화된 출력을 사용하여 LLM에 현재 쿼리에서 가장 관련된 Tool을 요청합니다. 구조화된 출력 스키마는 사용 가능한 Tool 이름과 설명을 정의합니다. 모델 제공자는 종종 이 구조화된 출력 정보를 배후에서 시스템 프롬프트에 추가합니다.

**API 참조:** [LLMToolSelectorMiddleware](https://reference.langchain.com/python/langchain/middleware/LLMToolSelectorMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[tool1, tool2, tool3, tool4, tool5, ...],
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4.1-mini",
            max_tools=3,
            always_include=["search"],
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

### model

`문자열 | BaseChatModel`

Tool 선택을 위한 모델입니다. 모델 식별자 문자열(예: `'openai:gpt-4.1-mini'`) 또는 `BaseChatModel` 인스턴스입니다. 자세한 내용은 [init_chat_model](https://reference.langchain.com/python/langchain/chat_models/init_chat_model)을 참조하세요.

기본값은 Agent의 주 모델입니다.

### system_prompt

`문자열`

선택 모델에 대한 지침입니다. 지정되지 않으면 기본 제공 프롬프트를 사용합니다.

### max_tools

`숫자`

선택할 최대 Tool 수입니다. 모델이 더 많은 항목을 선택하면 처음 `max_tools`만 사용됩니다. 지정되지 않으면 제한 없음입니다.

### always_include

`list[string]`

선택에 관계없이 항상 포함할 Tool 이름입니다. 이는 `max_tools` 제한에 포함되지 않습니다.

</details>

---

## Tool 재시도

구성 가능한 지수 백오프로 실패한 Tool 호출을 자동으로 재시도합니다. Tool 재시도는 다음의 경우에 유용합니다:

- 외부 API 호출의 일시적 오류 처리
- 네트워크 의존 Tool의 신뢰성 개선
- 일시적 오류를 정상적으로 처리하는 탄력적 Agent 구축

**API 참조:** [ToolRetryMiddleware](https://reference.langchain.com/python/langchain/middleware/ToolRetryMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

### max_retries

`숫자` **기본값:** `"2"`

초기 호출 후 최대 재시도 시도(기본값 3회 총 시도)

### tools

`list[BaseTool | str]`

재시도 로직을 적용할 Tool 또는 Tool 이름의 선택 사항 목록입니다. `None`인 경우 모든 Tool에 적용됩니다.

### retry_on

`tuple[type[Exception], ...] | callable` **기본값:** `"(Exception,)"`

재시도할 예외 유형의 튜플 또는 예외를 받아 재시도해야 하면 `True`를 반환하는 호출 가능입니다.

### on_failure

`문자열 | callable` **기본값:** `"return_message"`

모든 재시도가 소진되었을 때의 동작입니다. 옵션:

- `'return_message'` - 오류 세부 정보가 포함된 `ToolMessage` 반환(LLM이 실패를 처리하도록 허용)
- `'raise'` - 예외 다시 발생(Agent 실행 중지)
- 사용자 정의 호출 가능 - 예외를 받아 `ToolMessage` 콘텐츠를 위한 문자열을 반환하는 함수

### backoff_factor

`숫자` **기본값:** `"2.0"`

지수 백오프의 승수입니다. 각 재시도는 `initial_delay * (backoff_factor ** retry_number)` 초 동안 기다립니다. 일정한 지연은 `0.0`으로 설정하세요.

### initial_delay

`숫자` **기본값:** `"1.0"`

첫 번째 재시도 전 초 단위의 초기 지연입니다.

### max_delay

`숫자` **기본값:** `"60.0"`

재시도 간의 초 단위 최대 지연(지수 백오프 증가를 제한함)

### jitter

`boolean` **기본값:** `"true"`

thundering herd를 피하기 위해 지연에 임의 jitter(±25%)를 추가할지 여부입니다.

</details>

<details>
<summary>전체 예시</summary>

Middleware는 자동으로 지수 백오프로 실패한 Tool 호출을 재시도합니다.

주요 구성:

- `max_retries` - 재시도 시도 횟수(기본값: 2)
- `backoff_factor` - 지수 백오프의 승수(기본값: 2.0)
- `initial_delay` - 초 단위 시작 지연(기본값: 1.0)
- `max_delay` - 지연 증가의 상한(기본값: 60.0)
- `jitter` - 임의 변동 추가(기본값: True)

실패 처리:

- `on_failure='return_message'` - 오류 메시지 반환
- `on_failure='raise'` - 예외 다시 발생
- 사용자 정의 함수 - 오류 메시지를 반환하는 함수

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool, api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
            tools=["api_tool"],
            retry_on=(ConnectionError, TimeoutError),
            on_failure="continue",
        ),
    ],
)
```

</details>

---

## 모델 재시도

구성 가능한 지수 백오프로 실패한 모델 호출을 자동으로 재시도합니다. 모델 재시도는 다음의 경우에 유용합니다:

- 모델 API 호출의 일시적 오류 처리
- 네트워크 의존 모델 요청의 신뢰성 개선
- 일시적 모델 오류를 정상적으로 처리하는 탄력적 Agent 구축

**API 참조:** [ModelRetryMiddleware](https://reference.langchain.com/python/langchain/middleware/ModelRetryMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, database_tool],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

### max_retries

`숫자` **기본값:** `"2"`

초기 호출 후 최대 재시도 시도(기본값 3회 총 시도)

### retry_on

`tuple[type[Exception], ...] | callable` **기본값:** `"(Exception,)"`

재시도할 예외 유형의 튜플 또는 예외를 받아 재시도해야 하면 `True`를 반환하는 호출 가능입니다.

### on_failure

`문자열 | callable` **기본값:** `"continue"`

모든 재시도가 소진되었을 때의 동작입니다. 옵션:

- `'continue'` (기본값) - 오류 세부 정보가 포함된 `AIMessage` 반환(Agent가 실패를 정상적으로 처리하도록 허용)
- `'error'` - 예외 다시 발생(Agent 실행 중지)
- 사용자 정의 호출 가능 - 예외를 받아 `AIMessage` 콘텐츠를 위한 문자열을 반환하는 함수

### backoff_factor

`숫자` **기본값:** `"2.0"`

지수 백오프의 승수입니다. 각 재시도는 `initial_delay * (backoff_factor ** retry_number)` 초 동안 기다립니다. 일정한 지연은 `0.0`으로 설정하세요.

### initial_delay

`숫자` **기본값:** `"1.0"`

첫 번째 재시도 전 초 단위의 초기 지연입니다.

### max_delay

`숫자` **기본값:** `"60.0"`

재시도 간의 초 단위 최대 지연(지수 백오프 증가를 제한함)

### jitter

`boolean` **기본값:** `"true"`

thundering herd를 피하기 위해 지연에 임의 jitter(±25%)를 추가할지 여부입니다.

</details>

<details>
<summary>전체 예시</summary>

Middleware는 자동으로 지수 백오프로 실패한 모델 호출을 재시도합니다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

# 기본 설정을 사용한 기본 사용법(2회 재시도, 지수 백오프)
agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool],
    middleware=[ModelRetryMiddleware()],
)

# 사용자 정의 예외 필터링
class TimeoutError(Exception):
    """타임아웃 오류를 위한 사용자 정의 예외입니다."""
    pass

class ConnectionError(Exception):
    """연결 오류를 위한 사용자 정의 예외입니다."""
    pass

# 특정 예외만 재시도
retry = ModelRetryMiddleware(
    max_retries=4,
    retry_on=(TimeoutError, ConnectionError),
    backoff_factor=1.5,
)

def should_retry(error: Exception) -> bool:
    # 속도 제한 오류에서만 재시도
    if isinstance(error, TimeoutError):
        return True
    # 또는 특정 HTTP 상태 코드 확인
    if hasattr(error, "status_code"):
        return error.status_code in (429, 503)
    return False

retry_with_filter = ModelRetryMiddleware(
    max_retries=3,
    retry_on=should_retry,
)

# 예외를 발생시키는 대신 오류 메시지 반환
retry_continue = ModelRetryMiddleware(
    max_retries=4,
    on_failure="continue",  # 예외를 발생시키는 대신 오류와 함께 AIMessage 반환
)

# 사용자 정의 오류 메시지 형식
def format_error(error: Exception) -> str:
    return f"Model call failed: {error}. Please try again later."

retry_with_formatter = ModelRetryMiddleware(
    max_retries=4,
    on_failure=format_error,
)

# 일정한 백오프(지수 증가 없음)
constant_backoff = ModelRetryMiddleware(
    max_retries=5,
    backoff_factor=0.0,  # 지수 증가 없음
    initial_delay=2.0,   # 항상 2초 대기
)

# 실패 시 예외 발생
strict_retry = ModelRetryMiddleware(
    max_retries=2,
    on_failure="error",  # 메시지 반환 대신 예외 다시 발생
)
```

</details>

---

## LLM Tool 에뮬레이터

테스트 목적으로 실제 Tool 호출을 AI 생성 응답으로 바꾸는 LLM을 사용하여 Tool 실행을 에뮬레이트합니다. LLM Tool 에뮬레이터는 다음의 경우에 유용합니다:

- 실제 Tool을 실행하지 않고 Agent 동작을 테스트합니다.
- 외부 Tool을 사용할 수 없거나 비용이 많이 드는 경우 Agent 개발
- 실제 Tool을 구현하기 전에 Agent 워크플로우 프로토타이핑

**API 참조:** [LLMToolEmulator](https://reference.langchain.com/python/langchain/middleware/LLMToolEmulator)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather, search_database, send_email],
    middleware=[
        LLMToolEmulator(),  # 모든 Tool 에뮬레이트
    ],
)
```

<details>
<summary>구성 옵션</summary>

### tools

`list[str | BaseTool]`

에뮬레이트할 Tool 이름(str) 또는 `BaseTool` 인스턴스 목록입니다. `None` (기본값)인 경우, 모든 Tool이 에뮬레이트됩니다. 빈 목록 `[]`인 경우 Tool이 에뮬레이트되지 않습니다. Tool 이름/인스턴스가 있는 배열인 경우 해당 Tool만 에뮬레이트됩니다.

### model

`문자열 | BaseChatModel`

에뮬레이트된 Tool 응답 생성에 사용할 모델입니다. 모델 식별자 문자열(예: `'anthropic:claude-sonnet-4-5-20250929'`) 또는 `BaseChatModel` 인스턴스입니다. 지정되지 않으면 Agent의 모델로 기본 설정됩니다. 자세한 내용은 [init_chat_model](https://reference.langchain.com/python/langchain/chat_models/init_chat_model)을 참조하세요.

</details>

<details>
<summary>전체 예시</summary>

Middleware는 LLM을 사용하여 실제 Tool을 실행하는 대신 Tool 호출에 대해 그럴듯한 응답을 생성합니다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"

# 모든 Tool 에뮬레이트(기본 동작)
agent = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator()],
)

# 특정 Tool만 에뮬레이트
agent2 = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(tools=["get_weather"])],
)

# 에뮬레이션을 위해 사용자 정의 모델 사용
agent4 = create_agent(
    model="gpt-4.1",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(model="claude-sonnet-4-5-20250929")],
)
```

</details>

---

## 컨텍스트 편집

토큰 제한에 도달했을 때 최근 결과를 보존하면서 이전 Tool 호출 출력을 지워서 대화 컨텍스트를 관리합니다. 이는 Tool 호출이 많은 오래된 대화에서 컨텍스트 윈도우를 관리할 수 있도록 도움이 됩니다. 컨텍스트 편집은 다음의 경우에 유용합니다:

- Tool 호출이 많아서 토큰 제한을 초과하는 오래된 대화
- 더 이상 관련이 없는 오래된 Tool 출력을 제거하여 토큰 비용 절감
- 컨텍스트에서 최근 N개 Tool 결과만 유지

**API 참조:** [ContextEditingMiddleware](https://reference.langchain.com/python/langchain/middleware/ContextEditingMiddleware), [ClearToolUsesEdit](https://reference.langchain.com/python/langchain/middleware/ClearToolUsesEdit)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,
                    keep=3,
                ),
            ],
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

### edits

`list[ContextEdit]` **기본값:** `"[ClearToolUsesEdit()]"`

적용할 `ContextEdit` 전략 목록입니다.

### token_count_method

`문자열` **기본값:** `"approximate"`

토큰 계산 방법입니다. 옵션: `'approximate'` 또는 `'model'`

**ClearToolUsesEdit 옵션:**

### trigger

`숫자` **기본값:** `"100000"`

편집을 트리거하는 토큰 수입니다. 대화가 이 토큰 수를 초과하면 이전 Tool 출력이 지워집니다.

### clear_at_least

`숫자` **기본값:** `"0"`

편집이 실행될 때 회수할 최소 토큰 수입니다. 0으로 설정하면 필요한 만큼 지웁니다.

### keep

`숫자` **기본값:** `"3"`

보존해야 하는 가장 최근의 Tool 결과 수입니다. 이들은 절대 지워지지 않습니다.

### clear_tool_inputs

`boolean` **기본값:** `"False"`

AI 메시지에서 시작 Tool 호출 매개변수를 지울지 여부입니다. `True`인 경우, Tool 호출 인수는 빈 객체로 바뀝니다.

### exclude_tools

`list[string]` **기본값:** `"()"`

지워지지 않아야 할 Tool 이름 목록입니다. 이 Tool은 출력이 절대 지워지지 않습니다.

### placeholder

`문자열` **기본값:** `"[cleared]"`

지워진 Tool 출력에 대해 삽입된 자리 표시자 텍스트입니다. 이는 원본 Tool 메시지 콘텐츠를 바꿉니다.

</details>

<details>
<summary>전체 예시</summary>

Middleware는 토큰 제한에 도달했을 때 컨텍스트 편집 전략을 적용합니다. 가장 일반적인 전략은 `ClearToolUsesEdit`이며, 이는 최근 항목을 보존하면서 이전 Tool 결과를 지웁니다.

작동 방식:

1. 대화의 토큰 수 모니터링
2. 임계값에 도달하면 이전 Tool 출력 지우기
3. 최근 N개 Tool 결과 보존
4. 선택적으로 컨텍스트를 위해 Tool 호출 인수 보존

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, your_calculator_tool, database_tool],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=2000,
                    keep=3,
                    clear_tool_inputs=False,
                    exclude_tools=[],
                    placeholder="[cleared]",
                ),
            ],
        ),
    ],
)
```

</details>

---

## Shell Tool

Agent가 명령 실행을 위해 지속적인 shell 세션을 노출합니다. Shell Tool Middleware는 다음의 경우에 유용합니다:

- 시스템 명령을 실행해야 하는 Agent
- 개발 및 배포 자동화 작업
- 테스트 및 검증 워크플로우
- 파일 시스템 작업 및 스크립트 실행

> [!WARNING]
> **보안 고려 사항:** 배포의 보안 요구 사항과 일치시키기 위해 적절한 실행 정책(`HostExecutionPolicy`, `DockerExecutionPolicy` 또는 `CodexSandboxExecutionPolicy`)을 사용하세요.

> [!INFO]
> **제한 사항:** 지속적인 shell 세션은 현재 인터럽트(Human-in-the-loop)와 함께 작동하지 않습니다. 향후 이에 대한 지원을 추가할 것으로 예상됩니다.

**API 참조:** [ShellToolMiddleware](https://reference.langchain.com/python/langchain/middleware/ShellToolMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
)

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

### workspace_root

`str | Path | None`

shell 세션의 기본 디렉터리입니다. 생략된 경우 Agent가 시작되면 임시 디렉터리가 생성되고 종료되면 제거됩니다.

### startup_commands

`tuple[str, ...] | list[str] | str | None`

세션이 시작된 후 순차적으로 실행되는 선택 사항 명령입니다.

### shutdown_commands

`tuple[str, ...] | list[str] | str | None`

세션이 종료되기 전에 실행되는 선택 사항 명령입니다.

### execution_policy

`BaseExecutionPolicy | None`

타임아웃, 출력 제한 및 리소스 구성을 제어하는 실행 정책입니다. 옵션:

- `HostExecutionPolicy` - 완전한 호스트 접근(기본값); Agent가 이미 컨테이너 또는 VM 내에서 실행되는 신뢰할 수 있는 환경에 최적
- `DockerExecutionPolicy` - 각 Agent 실행마다 별도의 Docker 컨테이너를 시작하여 강화된 격리 제공
- `CodexSandboxExecutionPolicy` - Codex CLI sandbox를 재사용하여 추가 syscall/파일 시스템 제한 제공

### redaction_rules

`tuple[RedactionRule, ...] | list[RedactionRule] | None`

모델로 반환하기 전에 명령 출력을 정제하기 위한 선택 사항 재정의 규칙입니다.

> [!INFO]
> 재정의 규칙은 실행 후에 적용되며 `HostExecutionPolicy`를 사용할 때 비밀 또는 민감 데이터의 반출을 방지하지 않습니다.

### tool_description

`str | None`

등록된 shell Tool 설명에 대한 선택 사항 재정의입니다.

### shell_command

`Sequence[str] | str | None`

지속적인 세션을 시작하는 데 사용되는 선택 사항 shell 실행 파일(문자열) 또는 인수 시퀀스입니다. 기본값은 `/bin/bash`입니다.

### env

`Mapping[str, Any] | None`

shell 세션으로 공급할 선택 사항 환경 변수입니다. 값은 명령 실행 전에 문자열로 강제 변환됩니다.

</details>

<details>
<summary>전체 예시</summary>

Middleware는 Agent가 순차적으로 명령을 실행하는 데 사용할 수 있는 단일 지속적인 shell 세션을 제공합니다.

실행 정책:

- `HostExecutionPolicy` (기본값) - 완전한 호스트 접근 권한이 있는 기본 실행
- `DockerExecutionPolicy` - 격리된 Docker 컨테이너 실행
- `CodexSandboxExecutionPolicy` - Codex CLI를 통한 샌드박스 실행

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
    DockerExecutionPolicy,
    RedactionRule,
)

# 호스트 실행이 있는 기본 shell Tool
agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

# 시작 명령이 있는 Docker 격리
agent_docker = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            startup_commands=["pip install requests", "export PYTHONPATH=/workspace"],
            execution_policy=DockerExecutionPolicy(
                image="python:3.11-slim",
                command_timeout=60.0,
            ),
        ),
    ],
)

# 출력 재정의 포함(실행 후 적용)
agent_redacted = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            redaction_rules=[
                RedactionRule(pii_type="api_key", detector=r"sk-[a-zA-Z0-9]{32}"),
            ],
        ),
    ],
)
```

</details>

---

## 파일 검색

파일 시스템을 통해 Glob 및 Grep 검색 Tool을 제공합니다. 파일 검색 Middleware는 다음의 경우에 유용합니다:

- 코드 탐색 및 분석
- 이름 패턴으로 파일 찾기
- 정규식을 사용한 코드 콘텐츠 검색
- 파일 검색이 필요한 대규모 코드베이스

**API 참조:** [FilesystemFileSearchMiddleware](https://reference.langchain.com/python/langchain/middleware/FilesystemFileSearchMiddleware)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",
            use_ripgrep=True,
        ),
    ],
)
```

<details>
<summary>구성 옵션</summary>

### root_path

`str` **필수**

검색할 루트 디렉터리입니다. 모든 파일 작업은 이 경로를 기준으로 합니다.

### use_ripgrep

`bool` **기본값:** `"True"`

검색에 ripgrep을 사용할지 여부입니다. ripgrep을 사용할 수 없으면 Python Regex로 폴백합니다.

### max_file_size_mb

`int` **기본값:** `"10"`

검색할 최대 파일 크기(MB). 이보다 큰 파일은 건너뜁니다.

</details>

<details>
<summary>전체 예시</summary>

Middleware는 Agent에 두 개의 검색 Tool을 추가합니다:

**Glob Tool** - 빠른 파일 패턴 일치:

- `**/*.py`, `src/**/*.ts` 같은 패턴 지원
- 수정 시간순으로 정렬된 일치 파일 경로 반환

**Grep Tool** - 정규식을 사용한 콘텐츠 검색:

- 전체 정규식 구문 지원
- `include` 매개변수로 파일 패턴별 필터링
- 세 가지 출력 모드: `files_with_matches`, `content`, `count`

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage

agent = create_agent(
    model="gpt-4.1",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",
            use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
)

# Agent는 이제 glob_search 및 grep_search Tool을 사용할 수 있습니다
result = agent.invoke({
    "messages": [HumanMessage("Find all Python files containing 'async def'")]
})

# Agent는 다음을 사용합니다:
# 1. glob_search(pattern="**/*.py") - Python 파일 찾기
# 2. grep_search(pattern="async def", include="*.py") - 비동기 함수 찾기
```

</details>

---

## 제공자별 Middleware

이러한 Middleware는 특정 LLM 제공자에 최적화되어 있습니다. 전체 세부 정보 및 예시는 각 제공자의 문서를 참조하세요.

| 제공자 | 설명 |
|----------|-------------|
| **[Anthropic](/oss/python/langchain/middleware/anthropic)** | Claude 모델을 위한 Prompt 캐싱, Bash Tool, 텍스트 편집기, 메모리 및 파일 검색 Middleware입니다. |
| **[OpenAI](/oss/python/langchain/middleware/openai)** | OpenAI 모델을 위한 콘텐츠 조정 Middleware입니다. |
