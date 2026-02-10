# LangSmith 관찰성

LangChain으로 Agent를 구성하고 실행할 때 Agent가 어떻게 동작하는지 가시성이 필요합니다: 어떤 **Tool**을 호출하고, 어떤 프롬프트를 생성하고, 어떻게 결정을 내리는지. `create_agent`로 구성한 LangChain Agent는 LLM 애플리케이션 동작을 캡처, 디버깅, 평가, 모니터링하는 플랫폼인 [**LangSmith**](https://smith.langchain.com/)를 통한 추적을 자동으로 지원합니다.

*추적*은 초기 사용자 입력에서 최종 응답까지 Tool 호출, 모델 상호작용, 결정 지점을 포함한 Agent 실행의 모든 단계를 기록합니다. 이 실행 데이터는 문제를 디버깅하고 다양한 입력에서 성능을 평가하고 프로덕션에서 사용 패턴을 모니터링하는 데 도움이 됩니다.

이 가이드는 LangChain Agent의 추적을 활성화하고 LangSmith를 사용하여 실행을 분석하는 방법을 보여줍니다.

## 필수 조건

시작하기 전에 다음을 확인합니다:

- **LangSmith 계정**: [smith.langchain.com](https://smith.langchain.com)에서 (무료로) 가입하거나 로그인합니다.
- **LangSmith API 키**: [API 키 생성](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key) 가이드를 따릅니다.

## 추적 활성화

모든 LangChain Agent는 LangSmith 추적을 자동으로 지원합니다. 활성화하려면 다음 환경 변수를 설정합니다:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

## 빠른 시작

LangSmith에 추적을 기록하기 위해 추가 코드가 필요하지 않습니다. 평소처럼 Agent 코드를 실행합니다:

```python
from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """수신인에게 이메일을 보냅니다."""
    # ... 이메일 전송 로직
    return f"Email sent to {to}"

def search_web(query: str):
    """웹에서 정보를 검색합니다."""
    # ... 웹 검색 로직
    return f"Search results for: {query}"

agent = create_agent(
    model="gpt-4.1",
    tools=[send_email, search_web],
    system_prompt="You are a helpful assistant that can send emails and search the web."
)

# Agent를 실행합니다 - 모든 단계가 자동으로 추적됩니다
response = agent.invoke({
    "messages": [{"role": "user", "content": "Search for the latest AI news and email a summary to john@example.com"}]
})
```

기본적으로 추적은 `default`라는 이름의 프로젝트에 기록됩니다. 커스텀 프로젝트 이름을 구성하려면 [프로젝트에 기록](#프로젝트에-기록)을 참조하세요.

## 선택적으로 추적

LangSmith의 `tracing_context` 컨텍스트 관리자를 사용하여 애플리케이션의 특정 호출이나 부분을 선택적으로 추적할 수 있습니다:

```python
import langsmith as ls

# 이 부분은 추적됩니다
with ls.tracing_context(enabled=True):
    agent.invoke({"messages": [{"role": "user", "content": "Send a test email to alice@example.com"}]})

# 이 부분은 추적되지 않습니다 (LANGSMITH_TRACING이 설정되지 않은 경우)
agent.invoke({"messages": [{"role": "user", "content": "Send another email"}]})
```

## 프로젝트에 기록

<details>
<summary>정적으로</summary>

`LANGSMITH_PROJECT` 환경 변수를 설정하여 전체 애플리케이션에 대해 커스텀 프로젝트 이름을 설정할 수 있습니다:

```bash
export LANGSMITH_PROJECT=my-agent-project
```

</details>

<details>
<summary>동적으로</summary>

특정 작업에 대해 프로젝트 이름을 프로그래매틱하게 설정할 수 있습니다:

```python
import langsmith as ls

with ls.tracing_context(project_name="email-agent-test", enabled=True):
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Send a welcome email"}]
    })
```

</details>

## 추적에 메타데이터 추가

추적에 커스텀 메타데이터와 태그를 주석 처리할 수 있습니다:

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Send a welcome email"}]},
    config={
        "tags": ["production", "email-assistant", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        }
    }
)
```

`tracing_context`도 세밀한 제어를 위해 태그 및 메타데이터를 수용합니다:

```python
with ls.tracing_context(
    project_name="email-agent-test",
    enabled=True,
    tags=["production", "email-assistant", "v1.0"],
    metadata={"user_id": "user_123", "session_id": "session_456", "environment": "production"}):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Send a welcome email"}]}
    )
```

이 커스텀 메타데이터와 태그는 LangSmith의 추적에 연결됩니다.

> [!팁]
> 추적을 사용하여 Agent를 디버깅, 평가, 모니터링하는 방법에 대해 자세히 알아보려면 [LangSmith 문서](https://docs.smith.langchain.com/)를 참조하세요.
