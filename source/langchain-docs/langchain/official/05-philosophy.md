# Philosophy

LangChain exists to be the easiest place to start building with LLMs, while also being flexible and production-ready.

LangChain is driven by a few core beliefs:

*   Large Language Models (LLMs) are great, powerful new technology.
*   LLMs are even better when you combine them with external sources of data.
*   LLMs will transform what the applications of the future look like. Specifically, the applications of the future will look more and more agentic.
*   It is still very early on in that transformation.
*   While it’s easy to build a prototype of those agentic applications, it’s still really hard to build agents that are reliable enough to put into production.

With LangChain, we have two core focuses:

1.  **We want to enable developers to build with the best models.**
    Different providers expose different APIs, with different model parameters and different message formats. Standardizing these model inputs and outputs is a core focus, making it easy for developer to easily change to the most recent state-of-the-art model, avoiding lock-in.
2.  **We want to make it easy to use models to orchestrate more complex flows that interact with other data and computation.**
    Models should be used for more than just *text generation* - they should also be used to orchestrate more complex flows that interact with other data. LangChain makes it easy to define [tools](/oss/python/langchain/tools) that LLMs can use dynamically, as well as help with parsing of and access to unstructured data.

## History

Given the constant rate of change in the field, LangChain has also evolved over time. Below is a brief timeline of how LangChain has changed over the years, evolving alongside what it means to build with LLMs:

### 2022-10-24
**v0.0.1**

A month before ChatGPT, **LangChain was launched as a Python package**. It consisted of two main components:
*   LLM abstractions
*   “Chains”, or predetermined steps of computation to run, for common use cases. For example - RAG: run a retrieval step, then run a generation step.

The name LangChain comes from “Language” (like Language models) and “Chains”.

### 2022-12
The first general purpose agents were added to LangChain.

These general purpose agents were based on the [ReAct paper](https://arxiv.org/abs/2210.03629) (ReAct standing for Reasoning and Acting). They used LLMs to generate JSON that represented tool calls, and then parsed that JSON to determine what tools to call.

### 2023-01
OpenAI releases 'Chat Completion' API. This was the first API that supported a chat-like interface (messages including System, User, and Assistant roles) rather than just a completion interface. LangChain added support for this.

### 2023-01
LangChain releases a JavaScript version.

### 2023-02
LangChain Inc. was formed as a company.

### 2023-03
OpenAI releases 'function calling' in their API. This allowed LLMs to return better structured output than just JSON. LangChain added support for this.

### 2023-06
LangSmith is released in private beta. LangSmith provides observability and evaluation for LLM applications.

### 2024-01
**v0.1.0**

LangChain releases 0.1.0. At the same time, the industry was maturing from just prototypes to production apps. This release focused on making LangChain more stable and ready for production.

### 2024-02
LangGraph released as an open-source library. LangGraph is a low-level orchestration layer for building agents. It supports first-class streaming, durable execution, short-term memory, and human-in-the-loop.

### 2024-06
Over 700 integrations. To make LangChain easier to maintain and use, we started splitting out integrations from the core package into their own standalone packages or `langchain-community`.

### 2024-10
As developers tried to improve the reliability of their applications, they needed more control than the high-level interfaces provided. LangGraph provided that low-level flexibility. Most chains and agents were marked as deprecated in LangChain with guides on how to migrate them to LangGraph. There is still one high-level abstraction created in LangGraph: an agent abstraction. It is built on top of low-level LangGraph and has the same interface as the ReAct agents from LangChain.

### 2025-04
Model APIs become more multimodal.

Models started to accept files, images, videos, and more. We updated the `langchain-core` message format accordingly to allow developers to specify these multimodal inputs in a standard way.

### 2025-10-20
**v1.0.0**

LangChain releases 1.0 with two major changes:

1.  **Complete revamp of all chains and agents in `langchain`.** All chains and agents are now replaced with only one high level abstraction: an agent abstraction built on top of LangGraph. This was the high-level abstraction that was originally created in LangGraph, but just moved to LangChain. For users still using old LangChain chains/agents who do NOT want to upgrade (note: we recommend you do), you can continue using old LangChain by installing the `langchain-classic` package.
2.  **A standard message content format.** Model APIs evolved from returning messages with a simple content string to more complex output types - reasoning blocks, citations, server-side tool calls, etc. LangChain evolved its message formats to standardize these across providers.
