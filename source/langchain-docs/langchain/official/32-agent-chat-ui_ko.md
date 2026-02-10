# Agent Chat UI

[**Agent Chat UI**](https://agentchat.vercel.app/)는 모든 LangChain Agent와 상호작용하기 위한 대화형 인터페이스를 제공하는 Next.js 애플리케이션입니다. 실시간 채팅, Tool 시각화 및 타임트래블 디버깅 및 상태 포킹과 같은 고급 기능을 지원합니다. Agent Chat UI는 `create_agent`를 사용하여 생성된 Agent와 원활하게 작동하며 로컬에서 실행하든 배포된 Context(예: [LangSmith](https://smith.langchain.com/))에서 실행하든 최소한의 설정으로 Agent에 대한 대화형 경험을 제공합니다.

Agent Chat UI는 오픈소스이며 애플리케이션 요구사항에 맞게 조정할 수 있습니다.

[![Agent Chat UI 소개](https://img.youtube.com/vi/lInrwVnZ83o/maxresdefault.jpg)](https://www.youtube.com/watch?v=lInrwVnZ83o)

*▶️ 보기: Agent Chat UI 소개*

> [!TIP]
> Agent Chat UI에서 생성형 UI를 사용할 수 있습니다. 자세한 내용은 [LangGraph로 생성형 사용자 인터페이스 구현하기](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-ui/)를 참조합니다.

## 빠른 시작

시작하는 가장 빠른 방법은 호스팅된 버전을 사용하는 것입니다:

1. **[Agent Chat UI](https://agentchat.vercel.app/) 방문**
2. **배포 URL 또는 로컬 서버 주소를 입력하여 Agent 연결**
3. **채팅 시작** - UI는 자동으로 Tool 호출 및 인터럽트를 감지하고 렌더링합니다

## 로컬 개발

사용자 정의 또는 로컬 개발의 경우 Agent Chat UI를 로컬에서 실행할 수 있습니다:

#### npx 사용

```bash
# 새로운 Agent Chat UI 프로젝트 생성
npx create-agent-chat-app --project-name my-chat-ui
cd my-chat-ui

# 종속성 설치 및 시작
pnpm install
pnpm dev
```

#### 저장소 복제

```bash
# 저장소 복제
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui

# 종속성 설치 및 시작
pnpm install
pnpm dev
```

## Agent에 연결

Agent Chat UI는 **로컬** 및 [**배포된 Agent**](https://docs.langchain.com/oss/python/langchain/deploy)에 모두 연결할 수 있습니다.

Agent Chat UI를 시작한 후 Agent에 연결하도록 구성해야 합니다:

1. **그래프 ID**: 그래프 이름을 입력합니다 (langgraph.json 파일의 `graphs`에서 찾음).
2. **배포 URL**: Agent 서버의 엔드포인트 (로컬 개발의 경우 `http://localhost:2024` 또는 배포된 Agent의 URL)
3. **LangSmith API 키 (선택 사항)**: LangSmith API 키를 추가합니다 (로컬 Agent 서버를 사용하는 경우 필수 아님)

구성되면 Agent Chat UI는 자동으로 Agent에서 중단된 스레드를 가져오고 표시합니다.

> [!TIP]
> Agent Chat UI는 Tool 호출 및 Tool 결과 메시지 렌더링에 대한 즉시 지원을 제공합니다. 표시되는 메시지를 사용자 정의하려면 [채팅에서 메시지 숨기기](https://github.com/langchain-ai/agent-chat-ui#hiding-messages-in-the-chat)를 참조합니다.
