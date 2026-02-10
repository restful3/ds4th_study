# LangSmith Studio

LangChain Agent를 로컬에서 구성할 때 Agent 내부에서 어떤 일이 일어나는지 시각화하고, 실시간으로 상호작용하고, 발생하는 문제를 디버깅하는 것이 유용합니다. [**LangSmith Studio**](https://smith.langchain.com/studio)는 로컬 머신에서 LangChain Agent를 개발하고 테스트하기 위한 무료 시각적 인터페이스입니다.

Studio는 로컬에서 실행 중인 Agent에 연결하여 Agent가 취하는 각 단계를 표시합니다: 모델로 전송된 프롬프트, Tool 호출 및 결과, 최종 출력. 다른 입력을 테스트하고 중간 상태를 검사하고 추가 코드나 배포 없이 Agent의 동작을 반복할 수 있습니다.

이 페이지는 로컬 LangChain Agent로 Studio를 설정하는 방법을 설명합니다.

## 필수 조건

시작하기 전에 다음을 확인합니다:

- **LangSmith 계정**: [smith.langchain.com](https://smith.langchain.com)에서 (무료로) 가입하거나 로그인합니다.
- **LangSmith API 키**: [API 키 생성](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key) 가이드를 따릅니다.
- LangSmith에 데이터가 **추적**되지 않길 원하면 애플리케이션의 `.env` 파일에서 `LANGSMITH_TRACING=false`를 설정합니다. 추적이 비활성화되면 데이터가 로컬 서버를 떠나지 않습니다.

## 로컬 Agent 서버 설정

### 1. LangGraph CLI 설치

[LangGraph CLI](https://langchain-ai.github.io/langgraph/reference/cli/)는 Agent를 Studio에 연결하는 로컬 개발 서버 (또한 [**Agent Server**](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/#agent-server)라고도 함)를 제공합니다.

```bash
# Python >= 3.11이 필요합니다.
pip install --upgrade "langgraph-cli[inmem]"
```

### 2. Agent 준비

이미 LangChain Agent가 있는 경우 직접 사용할 수 있습니다. 이 예제는 간단한 이메일 Agent를 사용합니다:

```python title="agent.py"
from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """이메일을 보냅니다"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... 이메일 전송 로직
    return f"Email sent to {to}"

agent = create_agent(
    "gpt-4.1",
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)
```

### 3. 환경 변수

Studio는 로컬 Agent를 연결하기 위해 LangSmith API 키가 필요합니다. 프로젝트 루트에 `.env` 파일을 생성하고 [LangSmith](https://smith.langchain.com)에서 API 키를 추가합니다.

> [!주의]
> `.env` 파일이 Git과 같은 버전 제어에 커밋되지 않도록 합니다.

```text title=".env"
LANGSMITH_API_KEY=lsv2...
```

### 4. LangGraph 구성 파일 생성

LangGraph CLI는 구성 파일을 사용하여 Agent를 찾고 종속성을 관리합니다. 앱 디렉토리에 `langgraph.json` 파일을 생성합니다:

```json title="langgraph.json"
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.py:agent"
  },
  "env": ".env"
}
```

`create_agent` 함수는 자동으로 컴파일된 LangGraph 그래프를 반환하며, 이는 구성 파일의 `graphs` 키가 예상하는 것입니다.

> [!정보]
> 구성 파일의 JSON 객체의 각 키에 대한 자세한 설명은 [LangGraph 구성 파일 참조](https://langchain-ai.github.io/langgraph/reference/cli/#configuration-file)를 참조하세요.

이 시점에서 프로젝트 구조는 다음과 같습니다:

```
my-app/
├── src
│   └── agent.py
├── .env
└── langgraph.json
```

### 5. 종속성 설치

루트 디렉토리에서 프로젝트 종속성을 설치합니다:

#### pip

```bash
pip install langchain langchain-openai
```

#### uv

```bash
uv add langchain langchain-openai
```

### 6. Studio에서 Agent 보기

개발 서버를 시작하여 Agent를 Studio에 연결합니다:

```bash
langgraph dev
```

> [!팁]
> Safari는 Studio에 대한 localhost 연결을 차단합니다. 이를 해결하려면 위 명령을 `--tunnel`과 함께 실행하여 보안 터널을 통해 Studio에 접근합니다.

서버가 실행 중이면 Agent는 `http://127.0.0.1:2024`의 API를 통해, 그리고 `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`의 Studio UI를 통해 접근할 수 있습니다:

![Studio UI의 Agent 뷰](images/studio_create-agent.avif)

Studio가 로컬 Agent에 연결되면 Agent의 동작을 빠르게 반복할 수 있습니다. 테스트 입력을 실행하고 프롬프트, Tool 인수, 반환 값, 토큰/지연 시간 메트릭을 포함한 전체 실행 추적을 검사합니다. 문제가 발생하면 Studio가 주변 상태로 예외를 캡처하여 어떤 일이 일어났는지 이해할 수 있도록 도와줍니다.

개발 서버는 핫 리로딩을 지원합니다. 코드에서 프롬프트 또는 Tool 서명을 변경하면 Studio는 즉시 반영됩니다. 처음부터 시작하지 않고 변경 사항을 테스트하기 위해 어떤 단계에서든 대화 스레드를 다시 실행합니다.

이 워크플로는 간단한 단일 Tool Agent에서 복잡한 다중 노드 그래프로 확장됩니다. Studio 실행 방법에 대한 자세한 내용은 LangSmith 문서의 다음 가이드를 참조하세요:

- [애플리케이션 실행](https://docs.smith.langchain.com/evaluation/tutorials/agents#4-iterate-on-your-agent-in-langsmith)
- [어시스턴트 관리](https://docs.smith.langchain.com/evaluation/tutorials/agents#4-iterate-on-your-agent-in-langsmith)
- [스레드 관리](https://docs.smith.langchain.com/evaluation/tutorials/agents#4-iterate-on-your-agent-in-langsmith)
- [프롬프트 반복](https://docs.smith.langchain.com/evaluation/tutorials/agents#4-iterate-on-your-agent-in-langsmith)
- [LangSmith 추적 디버깅](https://docs.smith.langchain.com/evaluation/tutorials/agents#4-iterate-on-your-agent-in-langsmith)
- [데이터셋에 노드 추가](https://docs.smith.langchain.com/evaluation/tutorials/agents#4-iterate-on-your-agent-in-langsmith)

## 비디오 가이드

> **비디오: 로컬 및 배포된 Agent**
>
> 로컬 및 배포된 Agent에 대한 자세한 내용은 [로컬 Agent 서버 설정](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)과 [배포](https://docs.langchain.com/oss/python/langchain/deploy)를 참조하세요.
