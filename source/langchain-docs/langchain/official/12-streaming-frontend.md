# Frontend

Build generative UIs with real-time streaming from LangChain agents, LangGraph graphs, and custom APIs

The `useStream` React hook provides seamless integration with LangGraph streaming capabilities. It handles all the complexities of streaming, state management, and branching logic, letting you focus on building great generative UI experiences.

Key features:

*   **Messages streaming** — Handle a stream of message chunks to form a complete message
*   **Automatic state management** — for messages, interrupts, loading states, and errors
*   **Conversation branching** — Create alternate conversation paths from any point in the chat history
*   **UI-agnostic design** — Bring your own components and styling

## Installation

Install the LangGraph SDK to use the useStream hook in your React application:

## Basic usage

The `useStream` hook connects to any LangGraph graph, whether that’s running on from your own endpoint, or deployed using LangSmith deployments.

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";

function Chat() {
  const stream = useStream({
    assistantId: "agent",
    // Local development
    apiUrl: "http://localhost:2024",
    // Production deployment (LangSmith hosted)
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

Learn how to [deploy your agents to LangSmith](https://docs.langchain.com/oss/python/langchain/deploy) for production-ready hosting with built-in observability, authentication, and scaling.

<details>
<summary><code>useStream</code> parameters</summary>

*   **assistantId** (string, required): The ID of the agent to connect to. When using LangSmith deployments, this must match the agent ID shown in your deployment dashboard. For custom API deployments or local development, this can be any string that your server uses to identify the agent.
*   **apiUrl** (string): The URL of the LangGraph server. Defaults to `http://localhost:2024` for local development.
*   **apiKey** (string): API key for authentication. Required when connecting to deployed agents on LangSmith.
*   **threadId** (string): Connect to an existing thread instead of creating a new one. Useful for resuming conversations.
*   **onThreadId** `(id: string) => void`: Callback invoked when a new thread is created. Use this to persist the thread ID for later use.
*   **reconnectOnMount** `boolean | (() => Storage)`: Automatically resume an ongoing run when the component mounts. Set to true to use session storage, or provide a custom storage function.
*   **onCreated** `(run: Run) => void`: Callback invoked when a new run is created. Useful for persisting run metadata for resumption.
*   **onError** `(error: Error) => void`: Callback invoked when an error occurs during streaming.
*   **onFinish** `(state: StateType, run?: Run) => void`: Callback invoked when a new run completes successfully with the final state.
*   **onCustomEvent** `(data: unknown, context: { mutate }) => void`: Handle custom events emitted from your agent using the writer. See Custom streaming events.
*   **onUpdateEvent** `(data: unknown, context: { mutate }) => void`: Handle state update events after each graph step.
*   **onMetadataEvent** `(metadata: { run_id, thread_id }) => void`: Handle metadata events with run and thread information.
*   **messagesKey** (string, default: "messages"): The key in the graph state that contains the messages array.
*   **throttle** (boolean, default: "true"): Batch state updates for better rendering performance. Disable for immediate updates.
*   **initialValues** (StateType | null): Initial state values to display while the first stream is loading. Useful for showing cached thread data immediately.

</details>

<details>
<summary><code>useStream</code> return values</summary>

*   **messages** (Message[]): All messages in the current thread, including both human and AI messages.
*   **values** (StateType): The current graph state values. Type is inferred from the agent or graph type parameter.
*   **isLoading** (boolean): Whether a stream is currently in progress. Use this to show loading indicators.
*   **error** (Error | null): Any error that occurred during streaming. null when no error.
*   **interrupt** (Interrupt | undefined): Current interrupt requiring user input, such as human-in-the-loop approval requests.
*   **toolCalls** (ToolCallWithResult[]): All tool calls across all messages, with their results and state (pending, completed, or error).
*   **submit** `(input, options?) => Promise<void>`: Submit new input to the agent. Pass null as input when resuming from an interrupt with a command. Options include checkpoint for branching, optimisticValues for optimistic updates, and threadId for optimistic thread creation.
*   **stop** `() => void`: Stop the current stream immediately.
*   **joinStream** `(runId: string) => void`: Resume an existing stream by run ID. Use with onCreated for manual stream resumption.
*   **setBranch** `(branch: string) => void`: Switch to a different branch in the conversation history.
*   **getToolCalls** `(message) => ToolCall[]`: Get all tool calls for a specific AI message.
*   **getMessagesMetadata** `(message) => MessageMetadata`: Get metadata for a message, including streaming info like langgraph_node for identifying the source node, and firstSeenState for branching.
*   **experimental_branchTree** (BranchTree): Tree representation of the thread for advanced branching controls in non-message based graphs.

</details>

## Thread management

Keep track of conversations with built-in thread management. You can access the current thread ID and get notified when new threads are created:

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

  // threadId is updated when a new thread is created
  // Store it in URL params or localStorage for persistence
}
```
We recommend storing the threadId to let users resume conversations after page refreshes.

### Resume after page refresh

The `useStream` hook can automatically resume an ongoing run upon mounting by setting `reconnectOnMount: true`. This is useful for continuing a stream after a page refresh, ensuring no messages and events generated during the downtime are lost.

```typescript
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  reconnectOnMount: true,
});
```

By default the ID of the created run is stored in `window.sessionStorage`, which can be swapped by passing a custom storage function:

```typescript
const stream = useStream({
  apiUrl: "http://localhost:2024",
  assistantId: "agent",
  reconnectOnMount: () => window.localStorage,
});
```

For manual control over the resumption process, use the run callbacks to persist metadata and `joinStream` to resume:

```typescript
import { useStream } from "@langchain/langgraph-sdk/react";
import { useEffect, useRef } from "react";

function Chat({ threadId }: { threadId: string | null }) {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    onCreated: (run) => {
      // Persist run ID when stream starts
      window.sessionStorage.setItem(`resume:${run.thread_id}`, run.run_id);
    },
    onFinish: (_, run) => {
      // Clean up when stream completes
      window.sessionStorage.removeItem(`resume:${run?.thread_id}`);
    },
  });

  // Resume stream on mount if there's a stored run ID
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
    // Use streamResumable to ensure events aren't lost
    stream.submit(
      { messages: [{ type: "human", content: text }] },
      { streamResumable: true }
    );
  };
}
```

> **Try the session persistence example**
>
> See a complete implementation of stream resumption with `reconnectOnMount` and thread persistence in the [session-persistence example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/session-persistence).


## Optimistic updates

You can optimistically update the client state before performing a network request, providing immediate feedback to the user:

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

### Optimistic thread creation

Use the `threadId` option in `submit` to enable optimistic UI patterns where you need to know the thread ID before the thread is created:

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
    // Navigate immediately without waiting for thread creation
    window.history.pushState({}, "", `/threads/${optimisticThreadId}`);
    
    // Create thread with the predetermined ID
    stream.submit(
      { messages: [{ type: "human", content: text }] },
      { threadId: optimisticThreadId }
    );
  };
}
```

### Cached thread display

Use the `initialValues` option to display cached thread data immediately while the history is being loaded from the server:

```typescript
function Chat({ threadId, cachedData }) {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
    threadId,
    initialValues: cachedData?.values,
  });
  
  // Shows cached messages instantly, then updates when server responds
}
```

## Branching

Create alternate conversation paths by editing previous messages or regenerating AI responses. Use `getMessagesMetadata()` to access checkpoint information for branching:

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

            {/* Edit human messages */}
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

            {/* Regenerate AI messages */}
            {message.type === "ai" && (
              <button
                onClick={() => stream.submit(undefined, { checkpoint: parentCheckpoint })}
              >
                Regenerate
              </button>
            )}

            {/* Switch between branches */}
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
 * Component for navigating between conversation branches.
 * Shows the current branch position and allows switching between alternatives.
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

For advanced use cases, use the experimental_branchTree property to get the tree representation of the thread for non-message based graphs.

> **Try the branching example**
>
> See a complete implementation of conversation branching with edit, regenerate, and branch switching in the [branching-chat example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/branching-chat).

## Type-safe streaming

The `useStream` hook supports full type inference when used with agents created via `@[createAgent]` or graphs created with `StateGraph`. Pass `typeof agent` or `typeof graph` as the type parameter to automatically infer tool call types.

### With createAgent

When using `@[createAgent]`, tool call types are automatically inferred from the tools you register to your agent:

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
  // Use the manually defined state type
  const stream = useStream<AgentState>({
    assistantId: "agent",
    apiUrl: "http://localhost:2024",
  });

  // stream.toolCalls[0].call.name is typed as "get_weather"
  // stream.toolCalls[0].call.args is typed as { location: string }
}
```

#### types.ts

```typescript
import type { Message } from "@langchain/langgraph-sdk";

// Define tool call types to match your Python agent
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

### With StateGraph

For custom `StateGraph` applications, the state types are inferred from the graph’s annotation:

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

### With Annotation types

If you’re using LangGraph.js, you can reuse your graph’s annotation types. Make sure to only import types to avoid importing the entire LangGraph.js runtime:

### Advanced type configuration

You can specify additional type parameters for interrupts, custom events, and configurable options:

## Rendering tool calls

Use `getToolCalls` to extract and render tool calls from AI messages. Tool calls include the call details, result (if completed), and state.

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

// Define tool call types to match your Python agent's tools
export type GetWeatherToolCall = {
  name: "get_weather";
  args: { location: string };
  id?: string;
};

// Union of all tool calls in your agent
export type AgentToolCalls = GetWeatherToolCall;

// Define state type with your tool calls
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

> **Try the tool calling example**
>
> See a complete implementation of tool call rendering with weather, calculator, and note-taking tools in the [tool-calling-agent example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/tool-calling-agent).

## Custom streaming events

Stream custom data from your agent using the writer in your tools or nodes. Handle these events in the UI with the `onCustomEvent` callback.

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
        # Emit progress events during execution
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
        # Emit progress events during execution
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

// Define tool calls to match your Python agent
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

> **Try the custom streaming example**
>
> See a complete implementation of custom events with progress bars, status badges, and file operation cards in the [custom-streaming example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/custom-streaming).

## Event handling

The `useStream` hook provides callback options that give you access to different types of streaming events. You don’t need to explicitly configure stream modes—just pass callbacks for the event types you want to handle:

### Available callbacks

| Callback | Description | Stream mode |
| :--- | :--- | :--- |
| `onUpdateEvent` | Called when a state update is received after each graph step | `updates` |
| `onCustomEvent` | Called when a custom event is received from your graph | `custom` |
| `onMetadataEvent` | Called with run and thread metadata | `metadata` |
| `onError` | Called when an error occurs | - |
| `onFinish` | Called when the stream completes | - |

## Multi-agent streaming

When working with multi-agent systems or graphs with multiple nodes, use message metadata to identify which node generated each message. This is particularly useful when multiple LLMs run in parallel and you want to display their outputs with distinct visual styling.

#### agent.py
```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, Send
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import BaseMessage, AIMessage
from typing import TypedDict, Annotated
import operator

# Use different model instances for variety
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

# Similar nodes for creative and practical researchers...

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

// Node configuration for visual display
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

// State matches your Python agent's State TypedDict
export interface AgentState {
  messages: Message[];
  topic: string;
  analytical_research: string;
  creative_research: string;
  practical_research: string;
}
```

> **Try the parallel research example**
>
> See a complete implementation of multi-agent streaming with three parallel researchers and distinct visual styling in the [parallel-research example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/parallel-research).

## Human-in-the-loop

Handle interrupts when the agent requires human approval for tool execution. Learn more in the [How to handle interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt) guide.

#### agent.py

```python
import type { Message } from "@langchain/langgraph-sdk";

// Tool call types matching your Python agent
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

// HITL types
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

// Node configuration for visual display
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

// State matches your Python agent's State TypedDict
export interface AgentState {
  messages: Message[];
  topic: string;
  analytical_research: string;
  creative_research: string;
  practical_research: string;
}
```

> **Try the human-in-the-loop example**
>
> See a complete implementation of approval workflows with approve, reject, and edit actions in the [human-in-the-loop example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/human-in-the-loop).

## Reasoning models

Support for models that stream "reasoning" or "thought" tokens before the final answer (e.g., OpenAI o1, Anthropic Claude with extended thinking).

Extended reasoning/thinking support is currently experimental. The streaming interface for reasoning tokens varies by provider (OpenAI vs. Anthropic) and may change as abstractions are developed.

When using models with extended reasoning capabilities (like OpenAI’s reasoning models or Anthropic’s extended thinking), the thinking process is embedded in the message content. You’ll need to extract and display it separately.

#### agent.py

```python
from langchain import create_agent
from langchain_openai import ChatOpenAI

# Use a reasoning-capable model
# For OpenAI: o1, o1-mini, o1-preview
# For Anthropic: claude-sonnet-4-20250514 with extended thinking enabled
model = ChatOpenAI(model="o1-mini")

agent = create_agent(
    model=model,
    tools=[],  # Reasoning models work best for complex reasoning tasks
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
 * Extracts reasoning/thinking content from an AI message.
 * Supports both OpenAI reasoning and Anthropic extended thinking.
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

  // Check for OpenAI reasoning in additional_kwargs
  if (msg.additional_kwargs?.reasoning?.summary) {
    const content = msg.additional_kwargs.reasoning.summary
      .filter((item) => item.type === "summary_text")
      .map((item) => item.text)
      .join("");
    if (content.trim()) return content;
  }

  // Check for Anthropic thinking in contentBlocks
  if (msg.contentBlocks?.length) {
    const thinking = msg.contentBlocks
      .filter((b) => b.type === "thinking" && b.thinking)
      .map((b) => b.thinking)
      .join("\n");
    if (thinking) return thinking;
  }

  // Check for thinking in message.content array
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
 * Extracts text content from a message.
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

> **Try the reasoning example**
>
> See a complete implementation of reasoning token display with OpenAI and Anthropic models in the [reasoning-agent example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ui-react/src/examples/reasoning-agent).

## Custom state types

For custom LangGraph applications, embed your tool call types in your state’s messages property.

## Custom transport

For custom API endpoints or non-standard deployments, use the transport option with `FetchStreamTransport` to connect to any streaming API.



## Related

*   [Streaming overview](https://docs.langchain.com/oss/python/langchain/streaming/overview) — Server-side streaming with LangChain agents
*   [useStream API Reference](https://reference.langchain.com/javascript/functions/_langchain_langgraph-sdk.react.useStream.html) — Full API documentation
*   [Agent Chat UI](https://docs.langchain.com/oss/python/langchain/ui) — Pre-built chat interface for LangGraph agents
*   [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) — Configuring interrupts for human review
*   [Multi-agent systems](https://docs.langchain.com/oss/python/langchain/multi-agent) — Building agents with multiple LLMs