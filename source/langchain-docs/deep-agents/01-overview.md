# Deep Agents Overview

> Build agents that can plan, use subagents, and leverage file systems for complex tasks

[deepagents](https://pypi.org/project/deepagents/) is a standalone library for building agents that can handle complex, multi-step tasks. Built on LangGraph and inspired by applications like Claude Code, Deep Research, and Manus, deep agents feature planning capabilities, a file system for context management, and the ability to spawn subagents.

---

## When to Use Deep Agents

Use deep agents when you need an agent that can:

- Handle **complex, multi-step tasks** that require planning and decomposition
- **Manage large amounts of context** through file system tools
- **Delegate work** to specialized subagents for context isolation
- **Persist memory** across conversations and threads

For simpler use cases, consider using LangChain's [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents) or building a custom [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) workflow.

---

## Core Capabilities

### Planning and Task Decomposition

Deep agents include a built-in `write_todos` tool that allows agents to break down complex tasks into individual steps, track progress, and adjust plans as new information emerges.

### Context Management

Through file system tools (`ls`, `read_file`, `write_file`, `edit_file`), agents can offload large amounts of context to memory, preventing context window overflow and working with variable-length tool results.

### Subagent Spawning

Through the built-in `task` tool, agents can spawn specialized subagents for context isolation. This allows the main agent to keep its context clean while still being able to go deep on specific subtasks.

### Long-term Memory

Extend the agent with persistent memory across threads using LangGraph's Store. Agents can store and retrieve information from previous conversations.

---

## Relationship to the LangChain Ecosystem

Deep agents are built on top of:

- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) - Provides the underlying graph execution and state management
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview) - Tools and model integrations work seamlessly with deep agents
- [LangSmith](https://docs.langchain.com/langsmith/home) - Observability, evaluation, and deployment

Deep agents applications can be deployed via [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments) and monitored with [LangSmith Observability](https://docs.langchain.com/langsmith/observability).

---

## Get Started

| Guide | Description |
|-------|-------------|
| [Quickstart](https://docs.langchain.com/oss/python/deepagents/quickstart) | Build your first deep agent |
| [Customization](https://docs.langchain.com/oss/python/deepagents/customization) | Learn about customization options |
| [Middleware](https://docs.langchain.com/oss/python/deepagents/middleware) | Understand the middleware architecture |
| [CLI](https://docs.langchain.com/oss/python/deepagents/cli) | Use the Deep Agents CLI |
| [Reference](https://reference.langchain.com/python/deepagents/) | See the deepagents API reference |

---

*Source: [https://docs.langchain.com/oss/python/deepagents/overview](https://docs.langchain.com/oss/python/deepagents/overview)*

---

<p align="center">
  <a href="README.md">Table of Contents</a> • <a href="02-quickstart.md">Next: Quickstart →</a>
</p>
