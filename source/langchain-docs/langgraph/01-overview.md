# LangGraph overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for agentic behaviors, where you can call an LLM in a loop to ask it what to do next. It also provides fine-grained control over both the human-in-the-loop and the agent's state, and allows you to save and resume the state of the agent at any point.

## install

```bash
pip install -U langgraph
```

```python
from langgraph.graph import StateGraph, START, END

# Define the graph
graph_builder = StateGraph(dict)
graph_builder.add_node("node_a", lambda state: {"messages": ["Hello"]})
graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", END)

# Compile and run
graph = graph_builder.compile()
graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

## Core benefits

LangGraph provides low-level supporting infrastructure for any long-running, stateful workflow or agent. LangGraph does not abstract prompts or architecture, and provides the following central benefits:

- **Durable execution**: Build agents that persist through failures and can run for extended periods, resuming from where they left off.
- **Human-in-the-loop**: Incorporate human oversight by inspecting and modifying agent state at any point.
- **Comprehensive memory**: Create stateful agents with both short-term working memory for ongoing reasoning and long-term memory across sessions.
- **Debugging with LangSmith**: Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
- **Production-ready deployment**: Deploy sophisticated agent systems confidently with scalable infrastructure designed to handle the unique challenges of stateful, long-running workflows.

## LangGraph ecosystem

While LangGraph can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools for building agents. To improve your LLM application development, pair LangGraph with:

### LangSmith

Trace requests, evaluate outputs, and monitor deployments in one place. Prototype locally with LangGraph, then move to production with integrated observability and evaluation to build more reliable agent systems.

### LangSmith Agent Server

Deploy and scale agents effortlessly with a purpose-built deployment platform for long running, stateful workflows. Discover, reuse, configure, and share agents across teams — and iterate quickly with visual prototyping in Studio.

### LangChain

Provides integrations and composable components to streamline LLM application development. Contains agent abstractions built on top of LangGraph.

## Acknowledgements

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.
