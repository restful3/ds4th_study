# 프론트엔드

LangChain Agent, LangGraph 그래프 및 사용자 정의 API로부터의 실시간 스트리밍으로 생성 UI를 구축합니다.

`useStream` React Hook은 LangGraph 스트리밍 기능과 원활한 통합을 제공합니다. 스트리밍, 상태 관리 및 분기 로직의 복잡함을 모두 처리하므로, 훌륭한 생성 UI 환경을 구축하는 데 집중할 수 있습니다.

주요 기능:

*   **메시지 스트리밍** — 메시지 청크 스트림을 처리하여 완전한 메시지 형성
*   **자동 상태 관리** — 메시지, 인터럽트, 로드 상태 및 오류
*   **대화 분기** — 채팅 기록의 모든 지점에서 대체 대화 경로 생성
*   **UI 불가지론적 설계** — 자신의 컴포넌트와 스타일링 가져오기

## 설치

React 응용 프로그램에서 useStream Hook을 사용하려면 LangGraph SDK를 설치합니다:

## 기본 사용법

`useStream` Hook은 로컬 끝점에서 실행 중이거나 LangSmith 배포를 사용하여 배포된 모든 LangGraph 그래프에 연결됩니다.

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";

function Chat() {
  const stream = useStream({
    assistantId: "agent",
    // 로컬 개발
    apiUrl: "http://localhost:2024",
    // 프로덕션 배포(LangSmith 호스팅)
    // apiUrl: "https://your-deployment.us.langgraph.app"
  });

  const handleSubmit = (message: string) => {
    stream.submit({
      messages: [
        { content: message, type: "human" }
      ],
    });
  };

  return (
    <div>
      {stream.messages.map((message, idx) => (
        <div key={message.id ?? idx}>
          {message.type}: {message.content}
        </div>
      ))}

      {stream.isLoading && <div>Loading...</div>}
      {stream.error && <div>Error: {stream.error.message}</div>}
    </div>
  );
}
```

프로덕션 준비 호스팅, 기본 제공 관찰성, 인증 및 확장 기능을 포함한 [Agent를 LangSmith에 배포하는 방법](https://docs.langchain.com/oss/python/langchain/deploy)을 알아봅니다.

<details>
<summary><code>useStream</code> 매개변수</summary>

*   **assistantId** (문자열, 필수): 연결할 Agent의 ID입니다. LangSmith 배포를 사용할 때, 이는 배포 대시보드에 표시된 Agent ID와 일치해야 합니다. 사용자 정의 API 배포 또는 로컬 개발의 경우, 서버가 Agent를 식별하는 데 사용하는 모든 문자열이 될 수 있습니다.
*   **apiUrl** (문자열): LangGraph 서버의 URL입니다. 로컬 개발의 경우 기본값은 `http://localhost:2024`입니다.
*   **apiKey** (문자열): 인증을 위한 API 키입니다. LangSmith에 배포된 Agent에 연결할 때 필수입니다.
*   **threadId** (문자열): 새로운 것을 만드는 대신 기존 스레드에 연결합니다. 대화를 재개하는 데 유용합니다.
*   **onThreadId** `(id: string) => void`: 새 스레드가 생성될 때 호출되는 콜백입니다. 스레드 ID를 나중에 사용하도록 유지하려면 이를 사용하세요.
*   **reconnectOnMount** `boolean | (() => Storage)`: 컴포넌트가 마운트될 때 진행 중인 실행을 자동으로 재개합니다. 세션 저장소를 사용하려면 true로 설정하거나 사용자 정의 저장소 함수를 제공하세요.
*   **onCreated** `(run: Run) => void`: 새 실행이 생성될 때 호출되는 콜백입니다. 재개를 위해 실행 메타데이터를 유지하는 데 유용합니다.
*   **onError** `(error: Error) => void`: 스트리밍 중에 오류가 발생할 때 호출되는 콜백입니다.
*   **onFinish** `(state: StateType, run?: Run) => void`: 새 실행이 최종 상태로 성공적으로 완료될 때 호출되는 콜백입니다.
*   **onCustomEvent** `(data: unknown, context: { mutate }) => void`: writer를 사용하는 Agent에서 발생한 사용자 정의 이벤트를 처리합니다. 사용자 정의 스트리밍 이벤트를 참조하세요.
*   **onUpdateEvent** `(data: unknown, context: { mutate }) => void`: 각 그래프 단계 후 상태 업데이트 이벤트를 처리합니다.
*   **onMetadataEvent** `(metadata: { run_id, thread_id }) => void`: 실행 및 스레드 정보가 포함된 메타데이터 이벤트를 처리합니다.
*   **messagesKey** (문자열, 기본값: "messages"): 메시지 배열을 포함하는 그래프 상태의 키입니다.
*   **throttle** (boolean, 기본값: "true"): 더 나은 렌더링 성능을 위해 상태 업데이트를 일괄 처리합니다. 즉시 업데이트를 위해 비활성화하세요.
*   **initialValues** (StateType | null): 첫 번째 스트림이 로드되는 동안 표시할 초기 상태 값입니다. 캐시된 스레드 데이터를 즉시 표시하는 데 유용합니다.

</details>

<details>
<summary><code>useStream</code> 반환 값</summary>

*   **messages** (Message[]): 현재 스레드의 모든 메시지(Human 및 AI 메시지 포함)입니다.
*   **values** (StateType): 현재 그래프 상태 값입니다. 유형은 Agent 또는 그래프 유형 매개변수에서 유추됩니다.
*   **isLoading** (boolean): 스트림이 현재 진행 중인지 여부입니다. 로드 지표를 표시하려면 이를 사용하세요.
*   **error** (Error | null): 스트리밍 중에 발생한 오류입니다. 오류가 없으면 null입니다.
*   **interrupt** (Interrupt | undefined): Human-in-the-loop 승인 요청과 같은 사용자 입력이 필요한 현재 인터럽트입니다.
*   **toolCalls** (ToolCallWithResult[]): 모든 메시지에서 모든 Tool 호출, 결과 및 상태(보류 중, 완료 또는 오류)입니다.
*   **submit** `(input, options?) => Promise<void>`: Agent로 새 입력을 제출합니다. 인터럽트에서 재개할 때 입력으로 null을 전달하세요. 옵션은 분기에 대한 checkpoint, 낙관적 업데이트에 대한 optimisticValues, 낙관적 스레드 생성에 대한 threadId를 포함합니다.
*   **stop** `() => void`: 현재 스트림을 즉시 중지합니다.
*   **joinStream** `(runId: string) => void`: 실행 ID로 기존 스트림을 재개합니다. 수동 스트림 재개를 위해 onCreated와 함께 사용하세요.
*   **setBranch** `(branch: string) => void`: 대화 기록의 다른 분기로 전환합니다.
*   **getToolCalls** `(message) => ToolCall[]`: 특정 AI 메시지의 모든 Tool 호출을 가져옵니다.
*   **getMessagesMetadata** `(message) => MessageMetadata`: 메시지의 메타데이터를 가져오며, 소스 노드를 식별하기 위한 langgraph_node와 분기를 위한 firstSeenState 같은 스트리밍 정보를 포함합니다.
*   **experimental_branchTree** (BranchTree): 메시지 기반이 아닌 그래프에서 고급 분기 컨트롤을 위한 스레드의 트리 표현입니다.

</details>

## 스레드 관리

기본 제공 스레드 관리로 대화를 추적합니다. 현재 스레드 ID에 접근하고 새 스레드가 생성될 때 알림을 받을 수 있습니다:

```typescript
import { useState } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";

function Chat() {
  const [threadId, setThreadId] = useState<string | null>(null);

  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId: threadId,
    onThreadId: setThreadId,
  });

  // 새 스레드가 생성되면 threadId가 업데이트됩니다
  // URL 매개변수 또는 localStorage에 저장하여 지속성 유지
}
```
페이지 새로 고침 후 사용자가 대화를 재개할 수 있도록 threadId를 저장하는 것을 권장합니다.

### 페이지 새로 고침 후 재개

`useStream` Hook은 `reconnectOnMount: true`를 설정하여 마운트 시 진행 중인 실행을 자동으로 재개할 수 있습니다. 이는 페이지 새로 고침 후 스트림을 계속하는 데 유용하며, 다운타임 중에 생성된 메시지와 이벤트가 손실되지 않도록 합니다.

```typescript
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  reconnectOnMount: true,
});
```

기본적으로 생성된 실행의 ID는 `window.sessionStorage`에 저장되며, 사용자 정의 저장소 함수를 전달하여 바꿀 수 있습니다:

```typescript
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  reconnectOnMount: () => window.localStorage,
});
```

재개 프로세스에 대한 수동 제어를 위해, 실행 콜백을 사용하여 메타데이터를 유지하고 `joinStream`을 사용하여 재개하세요:

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import { useEffect, useRef } from "react";

function Chat({ threadId }: { threadId: string | null }) {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    onCreated: (run) => {
      // 스트림 시작 시 실행 ID 유지
      window.sessionStorage.setItem(`resume:${run.thread_id}`, run.run_id);
    },
    onFinish: (_, run) => {
      // 스트림이 완료되면 정리
      window.sessionStorage.removeItem(`resume:${run?.thread_id}`);
    },
  });

  // 저장된 실행 ID가 있으면 마운트 시 스트림 재개
  const joinedThreadId = useRef<string | null>(null);
  useEffect(() => {
    if (!threadId) return;
    const runId = window.sessionStorage.getItem(`resume:${threadId}`);
    if (runId && joinedThreadId.current !== threadId) {
      stream.joinStream(runId);
      joinedThreadId.current = threadId;
    }
  }, [threadId]);

  const handleSubmit = (text: string) => {
    // streamResumable를 사용하여 이벤트가 손실되지 않도록 보장
    stream.submit(
      { messages: [{ type: "human", content: text }] },
      { streamResumable: true }
    );
  };
}
```

> **세션 지속성 예시를 시도하세요**
>
> [session-persistence 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/session-persistence)에서 `reconnectOnMount` 및 스레드 지속성을 사용한 스트림 재개의 완전한 구현을 확인하세요.

## 낙관적 업데이트

네트워크 요청을 수행하기 전에 클라이언트 상태를 낙관적으로 업데이트하여 사용자에게 즉각적인 피드백을 제공할 수 있습니다:

```typescript
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
});

const handleSubmit = (text: string) => {
  const newMessage = { type: "human" as const, content: text };

  stream.submit(
    { messages: [newMessage] },
    {
      optimisticValues(prev) {
        const prevMessages = prev.messages ?? [];
        return { ...prev, messages: [...prevMessages, newMessage] };
      },
    }
  );
};
```

### 낙관적 스레드 생성

스레드가 생성되기 전에 스레드 ID를 알아야 하는 낙관적 UI 패턴을 활성화하려면 `submit`의 `threadId` 옵션을 사용하세요:

```typescript
import { useState } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";

function Chat() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [optimisticThreadId] = useState(() => crypto.randomUUID());

  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    onThreadId: setThreadId,
  });

  const handleSubmit = (text: string) => {
    // 스레드 생성을 기다리지 않고 즉시 탐색
    window.history.pushState({}, "", `/threads/${optimisticThreadId}`);

    // 미리 결정된 ID로 스레드 생성
    stream.submit(
      { messages: [{ type: "human", content: text }] },
      { threadId: optimisticThreadId }
    );
  };
}
```

### 캐시된 스레드 표시

`initialValues` 옵션을 사용하여 서버에서 기록을 로드하는 동안 캐시된 스레드 데이터를 즉시 표시합니다:

```typescript
function Chat({ threadId, cachedData }) {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    initialValues: cachedData?.values,
  });

  // 캐시된 메시지를 즉시 표시한 후 서버가 응답하면 업데이트
}
```

## 분기

이전 메시지를 편집하거나 AI 응답을 재생성하여 대체 대화 경로를 만듭니다. `getMessagesMetadata()`를 사용하여 분기에 대한 checkpoint 정보에 접근합니다:

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import { BranchSwitcher } from "./BranchSwitcher";

function Chat() {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
  });

  return (
    <div>
      {stream.messages.map((message) => {
        const meta = stream.getMessagesMetadata(message);
        const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;

        return (
          <div key={message.id}>
            <div>{message.content as string}</div>

            {/* Human 메시지 편집 */}
            {message.type === "human" && (
              <button
                onClick={() => {
                  const newContent = prompt("Edit message:", message.content as string);
                  if (newContent) {
                    stream.submit(
                      { messages: [{ type: "human", content: newContent }] },
                      { checkpoint: parentCheckpoint }
                    );
                  }
                }}
              >
                Edit
              </button>
            )}

            {/* AI 메시지 재생성 */}
            {message.type === "ai" && (
              <button
                onClick={() => stream.submit(undefined, { checkpoint: parentCheckpoint })}
              >
                Regenerate
              </button>
            )}

            {/* 분기 간 전환 */}
            <BranchSwitcher
              branch={meta?.branch}
              branchOptions={meta?.branchOptions}
              onSelect={(branch) => stream.setBranch(branch)}
            />
          </div>
        );
      })}
    </div>
  );
}
```

#### BranchSwitcher.tsx

```typescript
/**
 * 대화 분기 간 탐색을 위한 컴포넌트입니다.
 * 현재 분기 위치를 표시하고 대체 항목 간 전환을 허용합니다.
 */
export function BranchSwitcher({
  branch,
  branchOptions,
  onSelect,
}: {
  branch: string | undefined;
  branchOptions: string[] | undefined;
  onSelect: (branch: string) => void;
}) {
  if (!branchOptions || !branch) return null;
  const index = branchOptions.indexOf(branch);

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        disabled={index <= 0}
        onClick={() => onSelect(branchOptions[index - 1])}
      >
        ←
      </button>
      <span>{index + 1} / {branchOptions.length}</span>
      <button
        type="button"
        disabled={index >= branchOptions.length - 1}
        onClick={() => onSelect(branchOptions[index + 1])}
      >
        →
      </button>
    </div>
  );
}
```

고급 사용 사례의 경우, experimental_branchTree 속성을 사용하여 메시지 기반이 아닌 그래프의 스레드의 트리 표현을 가져옵니다.

> **분기 예시를 시도하세요**
>
> [branching-chat 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/branching-chat)에서 편집, 재생성 및 분기 전환을 사용한 대화 분기의 완전한 구현을 확인하세요.

## 타입 안전 스트리밍

`useStream` Hook은 `@[createAgent]` 또는 `StateGraph`로 생성된 그래프로 생성된 Agent와 함께 사용할 때 완전한 타입 추론을 지원합니다. `typeof agent` 또는 `typeof graph`를 타입 매개변수로 전달하여 Tool 호출 유형을 자동으로 추론하세요.

### createAgent로

`@[createAgent]`를 사용할 때, Tool 호출 유형은 Agent에 등록한 Tool에서 자동으로 추론됩니다:

#### agent.py

```python
from langchain import create_agent, tool

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
)
```

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { AgentState } from "./types";

function Chat() {
  // 수동으로 정의한 상태 유형 사용
  const stream = useStream<AgentState>({
    assistantId: "agent",
    apiUrl: "http://localhost:2024",
  });

  // stream.toolCalls[0].call.name은 "get_weather"로 타입화됨
  // stream.toolCalls[0].call.args는 { location: string }으로 타입화됨
}
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

// Python Agent와 일치하도록 Tool 호출 유형 정의
export type GetWeatherToolCall = {
  name: "get_weather";
  args: { location: string };
  id?: string;
};

export type AgentToolCalls = GetWeatherToolCall;

export interface AgentState {
  messages: Message<AgentToolCalls>[];
}
```

### StateGraph로

사용자 정의 `StateGraph` 응용 프로그램의 경우, 상태 유형은 그래프의 주석에서 유추됩니다:

#### graph.py

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatOpenAI(model="gpt-4o-mini")

async def agent(state: State) -> dict:
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(State)
workflow.add_node("agent", agent)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
graph = workflow.compile()
```

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { State } from "./types";

function Chat() {
  const stream = useStream<State>({
    assistantId: "agent",
    apiUrl: "http://localhost:2024",
  });
}
```

#### types.ts

```typescript
import { BaseMessage } from "@langchain/core/messages";

export interface State {
  messages: BaseMessage[];
}
```

### 주석 유형으로

LangGraph.js를 사용 중인 경우, 그래프의 주석 유형을 재사용할 수 있습니다. 전체 LangGraph.js 런타임을 가져오는 것을 피하려면 유형만 가져왔는지 확인하세요:

### 고급 타입 구성

인터럽트, 사용자 정의 이벤트 및 구성 가능한 옵션에 대해 추가 타입 매개변수를 지정할 수 있습니다:

## Tool 호출 렌더링

`getToolCalls`를 사용하여 AI 메시지에서 Tool 호출을 추출하고 렌더링합니다. Tool 호출은 호출 세부 정보, 결과(완료된 경우) 및 상태를 포함합니다.

#### agent.py

```python
from langchain import create_agent, tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f'{{"status": "success", "content": "Weather in {location}: Sunny, 72°F"}}'

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
)
```

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { AgentState, AgentToolCalls } from "./types";
import { ToolCallCard } from "./ToolCallCard";
import { MessageBubble } from "./MessageBubble";

function Chat() {
  const stream = useStream<AgentState>({
    assistantId: "agent",
    apiUrl: "http://localhost:2024",
  });

  return (
    <div className="flex flex-col gap-4">
      {stream.messages.map((message, idx) => {
        if (message.type === "ai") {
          const toolCalls = stream.getToolCalls(message);

          if (toolCalls.length > 0) {
            return (
              <div key={message.id ?? idx} className="flex flex-col gap-2">
                {toolCalls.map((toolCall) => (
                  <ToolCallCard key={toolCall.id} toolCall={toolCall} />
                ))}
              </div>
            );
          }
        }

        return <MessageBubble key={message.id ?? idx} message={message} />;
      })}
    </div>
  );
}
```

#### ToolCallCard.tsx

```typescript
import type { ToolCallWithResult, ToolCallState } from "@langchain/langgraph-sdk/react";
import type { ToolMessage } from "@langchain/langgraph-sdk";
import type { AgentToolCalls, GetWeatherToolCall } from "./types";
import { parseToolResult } from "./utils";
import { WeatherCard } from "./WeatherCard";
import { GenericToolCallCard } from "./GenericToolCallCard";

export function ToolCallCard({
  toolCall,
}: {
  toolCall: ToolCallWithResult<AgentToolCalls>;
}) {
  const { call, result, state } = toolCall;

  if (call.name === "get_weather") {
    return <WeatherCard call={call} result={result} state={state} />;
  }

  return <GenericToolCallCard call={call} result={result} state={state} />;
}
```

#### WeatherCard.tsx

```typescript
import type { ToolCallState } from "@langchain/langgraph-sdk/react";
import type { ToolMessage } from "@langchain/langgraph-sdk";
import type { GetWeatherToolCall } from "./types";
import { parseToolResult } from "./utils";

export function WeatherCard({
  call,
  result,
  state,
}: {
  call: GetWeatherToolCall;
  result?: ToolMessage;
  state: ToolCallState;
}) {
  const isLoading = state === "pending";
  const parsedResult = parseToolResult(result);

  return (
    <div className="relative overflow-hidden rounded-xl">
      <div className="absolute inset-0 bg-gradient-to-br from-sky-600 to-indigo-600" />
      <div className="relative p-4">
        <div className="flex items-center gap-2 text-white/80 text-xs mb-3">
          <span className="font-medium">{call.args.location}</span>
          {isLoading && <span className="ml-auto">Loading...</span>}
        </div>
        {parsedResult.status === "error" ? (
          <div className="bg-red-500/20 rounded-lg p-3 text-red-200 text-sm">
            {parsedResult.content}
          </div>
        ) : (
          <div className="text-white text-lg font-medium">
            {parsedResult.content || "Fetching weather..."}
          </div>
        )}
      </div>
    </div>
  );
}
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

// Python Agent Tool과 일치하도록 Tool 호출 유형 정의
export type GetWeatherToolCall = {
  name: "get_weather";
  args: { location: string };
  id?: string;
};

// Agent의 모든 Tool 호출의 합집합
export type AgentToolCalls = GetWeatherToolCall;

// Tool 호출과 함께 상태 유형 정의
export interface AgentState {
  messages: Message<AgentToolCalls>[];
}
```

#### utils.ts

```typescript
import type { ToolMessage } from "@langchain/langgraph-sdk";

export function parseToolResult(result?: ToolMessage): {
  status: string;
  content: string;
} {
  if (!result) return { status: "pending", content: "" };
  try {
    return JSON.parse(result.content as string);
  } catch {
    return { status: "success", content: result.content as string };
  }
}
```

> **Tool 호출 예시를 시도하세요**
>
> [tool-calling-agent 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/tool-calling-agent)에서 날씨, 계산기 및 메모 작성 Tool을 사용한 Tool 호출 렌더링의 완전한 구현을 확인하세요.

## 사용자 정의 스트리밍 이벤트

Tool 또는 노드에서 writer를 사용하여 Agent에서 사용자 정의 데이터를 스트리밍합니다. `onCustomEvent` 콜백으로 UI에서 이러한 이벤트를 처리합니다.

#### agent.py

```python
import asyncio
import time
from langchain import create_agent, tool
from langchain.types import ToolRuntime

@tool
async def analyze_data(data_source: str, *, config: ToolRuntime) -> str:
    """Analyze data with progress updates."""
    steps = ["Connecting...", "Fetching...", "Processing...", "Done!"]

    for i, step in enumerate(steps):
        # 실행 중에 진행 상황 이벤트 발송
        if config.writer:
            config.writer({
                "type": "progress",
                "id": f"analysis-{int(time.time() * 1000)}",
                "message": step,
                "progress": ((i + 1) / len(steps)) * 100,
            })
        await asyncio.sleep(0.5)

    return '{"result": "Analysis complete"}'

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[analyze_data],
)
```

#### Chat.tsx

```typescript
import asyncio
import time
from langchain import create_agent, tool
from langchain.types import ToolRuntime

@tool
async def analyze_data(data_source: str, *, config: ToolRuntime) -> str:
    """Analyze data with progress updates."""
    steps = ["Connecting...", "Fetching...", "Processing...", "Done!"]

    for i, step in enumerate(steps):
        # 실행 중에 진행 상황 이벤트 발송
        if config.writer:
            config.writer({
                "type": "progress",
                "id": f"analysis-{int(time.time() * 1000)}",
                "message": step,
                "progress": ((i + 1) / len(steps)) * 100,
            })
        await asyncio.sleep(0.5)

    return '{"result": "Analysis complete"}'

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[analyze_data],
)
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

// Python Agent와 일치하도록 Tool 호출 정의
export type AnalyzeDataToolCall = {
  name: "analyze_data";
  args: { data_source: string };
  id?: string;
};

export type AgentToolCalls = AnalyzeDataToolCall;

export interface AgentState {
  messages: Message<AgentToolCalls>[];
}
```

> **사용자 정의 스트리밍 예시를 시도하세요**
>
> [custom-streaming 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/custom-streaming)에서 진행률 표시줄, 상태 배지 및 파일 작업 카드가 있는 사용자 정의 이벤트의 완전한 구현을 확인하세요.

## 이벤트 처리

`useStream` Hook은 다양한 유형의 스트리밍 이벤트에 접근할 수 있는 콜백 옵션을 제공합니다. 스트림 모드를 명시적으로 구성할 필요가 없습니다. 처리하려는 이벤트 유형에 대한 콜백만 전달하세요:

### 사용 가능한 콜백

| 콜백 | 설명 | 스트림 모드 |
| :--- | :--- | :--- |
| `onUpdateEvent` | 각 그래프 단계 후 상태 업데이트를 받을 때 호출됨 | `updates` |
| `onCustomEvent` | 그래프에서 사용자 정의 이벤트를 받을 때 호출됨 | `custom` |
| `onMetadataEvent` | 실행 및 스레드 메타데이터와 함께 호출됨 | `metadata` |
| `onError` | 오류 발생 시 호출됨 | - |
| `onFinish` | 스트림 완료 시 호출됨 | - |

## 다중 Agent 스트리밍

다중 Agent 시스템 또는 여러 노드가 있는 그래프로 작업할 때, 메시지 메타데이터를 사용하여 각 메시지를 생성한 노드를 식별합니다. 이는 여러 LLM이 병렬로 실행되고 명확한 시각적 스타일링으로 출력을 표시하려는 경우 특히 유용합니다.

#### agent.py
```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, Send
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import BaseMessage, AIMessage
from typing import TypedDict, Annotated
import operator

# 다양성을 위해 다양한 모델 인스턴스 사용
analytical_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
creative_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
practical_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    topic: str
    analytical_research: str
    creative_research: str
    practical_research: str

def fan_out_to_researchers(state: State) -> list[Send]:
    return [
        Send("researcher_analytical", state),
        Send("researcher_creative", state),
        Send("researcher_practical", state),
    ]

def dispatcher(state: State) -> dict:
    last_message = state["messages"][-1] if state["messages"] else None
    topic = last_message.content if last_message else ""
    return {"topic": topic}

async def researcher_analytical(state: State) -> dict:
    response = await analytical_model.ainvoke([
        {"role": "system", "content": "You are an analytical research expert."},
        {"role": "user", "content": f"Research: {state['topic']}"},
    ])
    return {
        "analytical_research": response.content,
        "messages": [AIMessage(content=response.content, name="researcher_analytical")],
    }

# 창의적 및 실용적 연구원을 위한 유사한 노드...

workflow = StateGraph(State)
workflow.add_node("dispatcher", dispatcher)
workflow.add_node("researcher_analytical", researcher_analytical)
workflow.add_node("researcher_creative", researcher_creative)
workflow.add_node("researcher_practical", researcher_practical)
workflow.add_edge(START, "dispatcher")
workflow.add_conditional_edges("dispatcher", fan_out_to_researchers)
workflow.add_edge("researcher_analytical", END)
workflow.add_edge("researcher_creative", END)
workflow.add_edge("researcher_practical", END)

agent: CompiledStateGraph = workflow.compile()
```

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { AgentState } from "./types";
import { MessageBubble } from "./MessageBubble";

// 시각 표시를 위한 노드 구성
const NODE_CONFIG: Record<string, { label: string; color: string }> = {
  researcher_analytical: { label: "Analytical Research", color: "cyan" },
  researcher_creative: { label: "Creative Research", color: "purple" },
  researcher_practical: { label: "Practical Research", color: "emerald" },
};

function MultiAgentChat() {
  const stream = useStream<AgentState>({
    assistantId: "parallel-research",
    apiUrl: "http://localhost:2024",
  });

  return (
    <div className="flex flex-col gap-4">
      {stream.messages.map((message, idx) => {
        if (message.type !== "ai") {
          return <MessageBubble key={message.id ?? idx} message={message} />;
        }

        const metadata = stream.getMessagesMetadata?.(message);
        const nodeName =
          (metadata?.streamMetadata?.langgraph_node as string) ||
          (message as { name?: string }).name;

        const config = nodeName ? NODE_CONFIG[nodeName] : null;

        if (!config) {
          return <MessageBubble key={message.id ?? idx} message={message} />;
        }

        return (
          <div
            key={message.id ?? idx}
            className={`bg-${config.color}-950/30 border border-${config.color}-500/30 rounded-xl p-4`}
          >
            <div className={`text-sm font-semibold text-${config.color}-400 mb-2`}>
              {config.label}
            </div>
            <div className="text-neutral-200 whitespace-pre-wrap">
              {typeof message.content === "string" ? message.content : ""}
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

// 상태는 Python Agent State TypedDict와 일치
export interface AgentState {
  messages: Message[];
  topic: string;
  analytical_research: string;
  creative_research: string;
  practical_research: string;
}
```

> **병렬 연구 예시를 시도하세요**
>
> [parallel-research 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/parallel-research)에서 3개의 병렬 연구원과 명확한 시각적 스타일링을 사용한 다중 Agent 스트리밍의 완전한 구현을 확인하세요.

## Human-in-the-loop

Tool 실행을 위해 Agent가 Human 승인이 필요할 때 인터럽트를 처리합니다. [인터럽트 처리 방법](https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt) 가이드에서 더 알아봅니다.

#### agent.py

```python
import type { Message } from "@langchain/langgraph-sdk";

// Python Agent와 일치하는 Tool 호출 유형
export type SendEmailToolCall = {
  name: "send_email";
  args: { to: string; subject: string; body: string };
  id?: string;
};

export type DeleteFileToolCall = {
  name: "delete_file";
  args: { path: string };
  id?: string;
};

export type ReadFileToolCall = {
  name: "read_file";
  args: { path: string };
  id?: string;
};

export type AgentToolCalls = SendEmailToolCall | DeleteFileToolCall | ReadFileToolCall;

export interface AgentState {
  messages: Message<AgentToolCalls>[];
}

// HITL 유형
export interface HITLRequest {
  actionRequests: Array<{
    name: string;
    args: Record<string, unknown>;
  }>;
}

export interface HITLResponse {
  decisions: Array<
    | { type: "approve" }
    | { type: "reject"; message: string }
    | { type: "edit"; newArgs: Record<string, unknown> }
  >;
}
```

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { AgentState } from "./types";
import { MessageBubble } from "./MessageBubble";

// 시각 표시를 위한 노드 구성
const NODE_CONFIG: Record<string, { label: string; color: string }> = {
  researcher_analytical: { label: "Analytical Research", color: "cyan" },
  researcher_creative: { label: "Creative Research", color: "purple" },
  researcher_practical: { label: "Practical Research", color: "emerald" },
};

function MultiAgentChat() {
  const stream = useStream<AgentState>({
    assistantId: "parallel-research",
    apiUrl: "http://localhost:2024",
  });

  return (
    <div className="flex flex-col gap-4">
      {stream.messages.map((message, idx) => {
        if (message.type !== "ai") {
          return <MessageBubble key={message.id ?? idx} message={message} />;
        }

        const metadata = stream.getMessagesMetadata?.(message);
        const nodeName =
          (metadata?.streamMetadata?.langgraph_node as string) ||
          (message as { name?: string }).name;

        const config = nodeName ? NODE_CONFIG[nodeName] : null;

        if (!config) {
          return <MessageBubble key={message.id ?? idx} message={message} />;
        }

        return (
          <div
            key={message.id ?? idx}
            className={`bg-${config.color}-950/30 border border-${config.color}-500/30 rounded-xl p-4`}
          >
            <div className={`text-sm font-semibold text-${config.color}-400 mb-2`}>
              {config.label}
            </div>
            <div className="text-neutral-200 whitespace-pre-wrap">
              {typeof message.content === "string" ? message.content : ""}
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

// 상태는 Python Agent State TypedDict와 일치
export interface AgentState {
  messages: Message[];
  topic: string;
  analytical_research: string;
  creative_research: string;
  practical_research: string;
}
```

> **Human-in-the-loop 예시를 시도하세요**
>
> [human-in-the-loop 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/human-in-the-loop)에서 승인, 거부 및 편집 작업이 있는 승인 워크플로우의 완전한 구현을 확인하세요.

## 추론 모델

최종 답변 전에 "추론" 또는 "생각" 토큰을 스트리밍하는 모델 지원(예: OpenAI o1, 확장 생각이 있는 Anthropic Claude)

확장 추론/생각 지원은 현재 실험적입니다. 추론 토큰의 스트리밍 인터페이스는 제공자(OpenAI 대 Anthropic)에 따라 다르며, 추상화가 개발됨에 따라 변경될 수 있습니다.

확장 추론 기능이 있는 모델(OpenAI의 추론 모델 또는 Anthropic의 확장 생각)을 사용할 때, 생각 프로세스는 메시지 콘텐츠에 포함됩니다. 별도로 추출하고 표시해야 합니다.

#### agent.py

```python
from langchain import create_agent
from langchain_openai import ChatOpenAI

# 추론 가능 모델 사용
# OpenAI: o1, o1-mini, o1-preview
# Anthropic: claude-sonnet-4-20250514 with extended thinking enabled
model = ChatOpenAI(model="o1-mini")

agent = create_agent(
    model=model,
    tools=[],  # 추론 모델은 복잡한 추론 작업에 가장 좋습니다
)
```

#### Chat.tsx

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import type { AgentState } from "./types";
import { getReasoningFromMessage, getTextContent } from "./utils";
import { MessageBubble } from "./MessageBubble";

function ReasoningChat() {
  const stream = useStream<AgentState>({
    assistantId: "reasoning-agent",
    apiUrl: "http://localhost:2024",
  });

  return (
    <div className="flex flex-col gap-4">
      {stream.messages.map((message, idx) => {
        if (message.type === "ai") {
          const reasoning = getReasoningFromMessage(message);
          const textContent = getTextContent(message);

          return (
            <div key={message.id ?? idx}>
              {reasoning && (
                <div className="mb-4">
                  <div className="text-xs font-medium text-amber-400/80 mb-2">
                    Reasoning
                  </div>
                  <div className="bg-amber-950/50 border border-amber-500/20 rounded-2xl px-4 py-3">
                    <div className="text-sm text-amber-100/90 whitespace-pre-wrap">
                      {reasoning}
                    </div>
                  </div>
                </div>
              )}

              {textContent && (
                <div className="text-neutral-100 whitespace-pre-wrap">
                  {textContent}
                </div>
              )}
            </div>
          );
        }

        return <MessageBubble key={message.id ?? idx} message={message} />;
      })}

      {stream.isLoading && (
        <div className="flex items-center gap-2 text-amber-400/70">
          <span className="text-sm">Thinking...</span>
        </div>
      )}
    </div>
  );
}
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

export interface AgentState {
  messages: Message[];
}
```

#### utils.ts

```typescript
import type { Message, AIMessage } from "@langchain/langgraph-sdk";

/**
 * AI 메시지에서 추론/생각 콘텐츠를 추출합니다.
 * OpenAI 추론과 Anthropic 확장 생각을 모두 지원합니다.
 */
export function getReasoningFromMessage(message: Message): string | undefined {
  type MessageWithExtras = AIMessage & {
    additional_kwargs?: {
      reasoning?: {
        summary?: Array<{ type: string; text: string }>;
      };
    };
    contentBlocks?: Array<{ type: string; thinking?: string }>;
  };

  const msg = message as MessageWithExtras;

  // additional_kwargs의 OpenAI 추론 확인
  if (msg.additional_kwargs?.reasoning?.summary) {
    const content = msg.additional_kwargs.reasoning.summary
      .filter((item) => item.type === "summary_text")
      .map((item) => item.text)
      .join("");
    if (content.trim()) return content;
  }

  // contentBlocks의 Anthropic 생각 확인
  if (msg.contentBlocks?.length) {
    const thinking = msg.contentBlocks
      .filter((b) => b.type === "thinking" && b.thinking)
      .map((b) => b.thinking)
      .join("\n");
    if (thinking) return thinking;
  }

  // message.content 배열의 생각 확인
  if (Array.isArray(msg.content)) {
    const thinking = msg.content
      .filter((b): b is { type: "thinking"; thinking: string } =>
        typeof b === "object" && b?.type === "thinking" && "thinking" in b
      )
      .map((b) => b.thinking)
      .join("\n");
    if (thinking) return thinking;
  }

  return undefined;
}

/**
 * 메시지에서 텍스트 콘텐츠를 추출합니다.
 */
export function getTextContent(message: Message): string {
  if (typeof message.content === "string") return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .filter((c): c is { type: "text"; text: string } => c.type === "text")
      .map((c) => c.text)
      .join("");
  }
  return "";
}
```

> **추론 예시를 시도하세요**
>
> [reasoning-agent 예시](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/reasoning-agent)에서 OpenAI 및 Anthropic 모델을 사용한 추론 토큰 표시의 완전한 구현을 확인하세요.

## 사용자 정의 상태 유형

사용자 정의 LangGraph 응용 프로그램의 경우, Tool 호출 유형을 상태의 메시지 속성에 포함시킵니다.

## 사용자 정의 전송

사용자 정의 API 끝점 또는 비표준 배포의 경우, 전송 옵션과 함께 `FetchStreamTransport`를 사용하여 모든 스트리밍 API에 연결합니다.

## 관련 항목

*   [스트리밍 개요](https://docs.langchain.com/oss/python/langchain/streaming/overview) — LangChain Agent를 사용한 서버 측 스트리밍
*   [useStream API 참조](https://reference.langchain.com/javascript/functions/_langchain_langgraph-sdk.react.useStream.html) — 전체 API 문서
*   [Agent Chat UI](https://docs.langchain.com/oss/python/langchain/ui) — LangGraph Agent용 사전 구축된 Chat 인터페이스
*   [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) — Human 검토를 위한 인터럽트 구성
*   [다중 Agent 시스템](https://docs.langchain.com/oss/python/langchain/multi-agent) — 여러 LLM이 있는 Agent 구축
