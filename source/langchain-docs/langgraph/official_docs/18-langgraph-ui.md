# Agent Chat UI

[Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) is a Next.js application that provides a conversational interface for any LangChain agent. It supports real-time chat, tool visualization, and advanced features like time-travel debugging and state forking.

Agent Chat UI is open source and can be found on [GitHub](https://github.com/langchain-ai/agent-chat-ui). You can also use [`create_agent`](/oss/python/langgraph/create_agent) to quickly set up a new agent with Agent Chat UI, or connect it to an existing project in [LangSmith](https://smith.langchain.com/).

Agent Chat UI is open source and can be adapted to your application needs.

[![Introducing Agent Chat UI](https://img.youtube.com/vi/lInrwVnZ83o/0.jpg)](https://www.youtube.com/watch?v=lInrwVnZ83o)

> [!TIP]
> You can use generative UI in the Agent Chat UI. For more information, see [Implement generative user interfaces with LangGraph](/langsmith/generative-ui-react).

## Quick start

The fastest way to get started is using the hosted version:

1. **Visit [Agent Chat UI](https://agent-chat.vercel.app)**
2. **Connect your agent** by entering your deployment URL or local server address
3. **Start chatting** - the UI will automatically detect and render tool calls and interrupts

## Local development

For customization or local development, you can run Agent Chat UI locally:

### Option 1: Use npx (Recommended)

```bash
# Create a new Agent Chat UI project
npx create-agent-chat-app --project-name my-chat-ui
cd my-chat-ui

# Install dependencies and start
pnpm install
pnpm dev
```

### Option 2: Clone repository

```bash
# Clone the repository
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui

# Install dependencies and start
pnpm install
pnpm dev
```

## Connect to your agent

Agent Chat UI can connect to both **local** and **deployed agents**.

After starting Agent Chat UI, you'll need to configure it to connect to your agent:

1. Click the **Connect** button in the top right.
2. Enter the URL of your agent. For local development, this is typically `http://localhost:2024`.
3. The UI will automatically detect the agent's schema and capabilities.

> [!INFO]
> Agent Chat UI has out-of-the-box support for rendering tool calls and tool result messages. To customize what messages are shown, see [Hiding Messages in the Chat](https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#hiding-messages-in-the-chat).
