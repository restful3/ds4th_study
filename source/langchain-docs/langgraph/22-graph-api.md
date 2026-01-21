# Graph API overview

## Graphs

1. **[State](https://docs.langchain.com/oss/python/langgraph/graph-api#state)**: A shared data structure that represents the current snapshot of your application. It can be any data type, but is typically defined using a shared state schema.
2. **[Nodes](https://docs.langchain.com/oss/python/langgraph/graph-api#nodes)**: Functions that encode the logic of your agents. They receive the current state as input, perform some computation or side-effect, and return an updated state.
3. **[Edges](https://docs.langchain.com/oss/python/langgraph/graph-api#edges)**: Functions that determine which Node to execute next based on the current state. They can be conditional branches or fixed transitions.

## StateGraph

[StateGraph](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.StateGraph) class is the main entrypoint for building a graph.

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)
```

## Compiling your graph

The step of compiling your graph creates a `CompiledGraph` from the graph definition. This provides an interface to [invoke](https://docs.langchain.com/oss/python/langgraph/graph-api#invoke), [stream](https://docs.langchain.com/oss/python/langgraph/graph-api#stream) and [inspect](https://docs.langchain.com/oss/python/langgraph/graph-api#inspect) the graph.

```python
graph = graph_builder.compile(...)
```

## State

The state is the interface between the graph and the outside world, as well as between nodes in the graph.

### Schema

The schema of the graph is the shape of the state that the graph maintains. It can be defined using:

*   [TypedDict](https://docs.python.org/3/library/typing.html#typing.TypedDict) (recommended)
*   [dataclass](https://docs.python.org/3/library/dataclasses.html)
*   [Pydantic BaseModel](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state)

See this [guide](https://docs.langchain.com/oss/python/langgraph/use-graph-api#define-input-and-output-schemas) for more details.

### Multiple schemas

*   Internal nodes can pass information that is not required in the graph’s input / output.
*   We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

See [this guide](https://docs.langchain.com/oss/python/langgraph/graph-api#define-input-and-output-schemas) for more details.

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input": "My"})
# {'graph_output': 'My name is Lance'}
```

1.  We pass `state: InputState` as the input schema to `node_1`. But, we write out to `foo`, a channel in `OverallState`. How can we write out to a state channel that is not included in the input schema? This is because a node can write to ANY state channel in the graph state. The graph state is the union of the state channels defined at initialization, which includes `OverallState` and the filters `InputState` and `OutputState`.
2.  We initialize the graph with:
    ```python
    StateGraph(
        OverallState,
        input_schema=InputState,
        output_schema=OutputState
    )
    ```
    So, how can we write to `PrivateState` in `node_2`? How does the graph gain access to this schema if it was not passed in the `StateGraph` initialization? We can do this because _nodes can also declare additional state channels_ as long as the state schema definition exists. In this case, the `PrivateState` schema is defined, so we can add `bar` as a new state channel in the graph and write to it.

## Reducers

Reducers allow you to define how updates to the state are applied.

### Default reducer

By default, the state is replaced with the new value.

```python
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: list[str]
```

If we have a state `{"foo": 1, "bar": ["hi"]}` and a node returns `{"foo": 2}`, the new state will be `{"foo": 2}`.

If a node returns `{"bar": ["bye"]}`, the new state will be `{"foo": 2, "bar": ["bye"]}`.

If we want to append to a list instead of replacing it, we can use `Annotated` with a reducer function.

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

Now, if we have a state `{"foo": 1, "bar": ["hi"]}` and a node returns `{"bar": ["bye"]}`, the new state will be `{"foo": 1, "bar": ["hi", "bye"]}`.

### Overwrite

[Overwrite](https://reference.langchain.com/python/langgraph/types/) is a special value that can be used to bypass reducers and overwrite the state value.

[Learn how to use Overwrite here](https://docs.langchain.com/oss/python/langgraph/use-graph-api#bypass-reducers-with-overwrite).

## Working with messages in graph state

### Why use messages?

Most modern LLM interfaces accept a list of messages as input. LangChain provides standard message objects like [HumanMessage](https://reference.langchain.com/python/langchain/messages/#langchain.messages.HumanMessage) and [AIMessage](https://reference.langchain.com/python/langchain/messages/#langchain.messages.AIMessage).

See [Messages conceptual guide](https://docs.langchain.com/oss/python/langchain/messages) for more context.

### Using messages in your graph

To use messages in your graph, you can define a key in your state schema that is a list of `Message` objects. Typically, you will want to append to this list rather than overwrite it, so you should use the `add_messages` reducer.

```python
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

### Serialization

The `add_messages` reducer also handles serialization of messages. You can pass a list of dicts to `add_messages` and it will convert them to `Message` objects.

```python
# this is supported
{"messages": [HumanMessage(content="message")]}

# and this is also supported
{"messages": [{"type": "human", "content": "message"}]}
```

See [here](https://python.langchain.com/docs/how_to/serialization/) for more details on serialization.

### MessagesState

Since having a list of messages is so common, LangGraph provides a pre-built state class called `MessagesState` that makes it easy to use messages.

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
```

## Nodes

Nodes are python functions (or `Runnable`s) that perform some logic. They receive the current `state` as input and return an update to the `state`.

Nodes can also receive a second argument, `config` (or `runtime` which wraps config).

1.  `state` – The state of the graph
2.  `config` – A [RunnableConfig](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig) object that contains configuration information like `thread_id` and tracing information like `tags`
3.  `runtime` – A Runtime object that contains [runtime context](https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context) and other information like `store` and `stream_writer`

```python
from dataclasses import dataclass
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

class State(TypedDict):
    input: str
    results: str

@dataclass
class Context:
    user_id: str

builder = StateGraph(State)

def plain_node(state: State):
    return state

def node_with_runtime(state: State, runtime: Runtime[Context]):
    print("In node: ", runtime.context.user_id)
    return {"results": f"Hello, {state['input']}!"}

def node_with_config(state: State, config: RunnableConfig):
    print("In node with thread_id: ", config["configurable"]["thread_id"])
    return {"results": f"Hello, {state['input']}!"}

builder.add_node("plain_node", plain_node)
builder.add_node("node_with_runtime", node_with_runtime)
builder.add_node("node_with_config", node_with_config)
```

### START node

`START` is a special node that represents the beginning of the graph execution.

```python
from langgraph.graph import START
graph.add_edge(START, "node_a")
```

### END node

`END` is a special node that represents the end of the graph execution.

```python
from langgraph.graph import END
graph.add_edge("node_a", END)
```

### Node caching

You can cache the results of nodes to avoid re-execution.
*   Specify a cache when compiling a graph (or specifying an entrypoint)
*   Specify a cache policy for nodes. Each cache policy supports:
    *   `key_func`: used to generate a cache key based on the input to a node, which defaults to a hash of the input with pickle.
    *   `ttl`: the time to live for the cache in seconds. If not specified, the cache will never expire.

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

class State(TypedDict):
    x: int
    result: int

builder = StateGraph(State)

def expensive_node(state: State) -> dict[str, int]:
    # expensive computation
    time.sleep(2)
    return {"result": state["x"] * 2}

builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")
graph = builder.compile(cache=InMemoryCache())

print(graph.invoke({"x": 5}, stream_mode='updates'))
# [{'expensive_node': {'result': 10}}]
# First run takes two seconds to run (due to mocked expensive computation).

print(graph.invoke({"x": 5}, stream_mode='updates'))
# [{'expensive_node': {'result': 10}, '__metadata__': {'cached': True}}]
# Second run utilizes cache and returns quickly.
```

## Edges

Edges determine how the graph progresses from one node to another.

*   **Normal Edges**: Go directly from one node to the next.
*   **Conditional Edges**: Call a function to determine which node(s) to go to next.
*   **Entry Point**: Which node to call first when user input arrives.
*   **Conditional Entry Point**: Call a function to determine which node(s) to call first when user input arrives.

### Normal edges

[add_edge](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.StateGraph.add_edge) creates a direct transition.

```python
graph.add_edge("node_a", "node_b")
```

### Conditional edges

[add_conditional_edges](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.state.StateGraph.add_conditional_edges) creates a transition based on a function.

```python
graph.add_conditional_edges("node_a", routing_function)
```

You can also provide a mapping from the output of the routing function to the name of the next node.

```python
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

### Entry point

Defines the first node to execute.

```python
from langgraph.graph import START
graph.add_edge(START, "node_a")
```

### Conditional entry point

Defines a dynamic start based on input.

```python
from langgraph.graph import START
graph.add_conditional_edges(START, routing_function)
```

## Send

[Send](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Send) is used to map-reduce or parallelize execution.

```python
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

## Command

[Command](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Command) is used to combine state updates and control flow.

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

You can use it inside conditional edges too:

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```

### When should I use command instead of conditional edges?

*   Use [Command](https://reference.langchain.com/python/langgraph/types/#langgraph.types.Command) when you need to both update the graph state and route to a different node. For example, when implementing [multi-agent handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs) where it’s important to route to a different agent and pass some information to that agent.
*   Use [conditional edges](https://docs.langchain.com/oss/python/langgraph/graph-api#conditional-edges) to route between nodes conditionally without updating the state.

### Navigating to a node in a parent graph

You can route to a node in a parent graph using `graph=Command.PARENT`.

```python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph", # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```

### Using inside tools

See [this guide](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-inside-tools).

### Human-in-the-loop

`Command` can also be used to resume execution after an interrupt.

```python
Command(resume="User input")
```

See [this conceptual guide](https://docs.langchain.com/oss/python/langgraph/interrupts) for more.

## Graph migrations

*   For threads at the end of the graph (i.e. not interrupted) you can change the entire topology of the graph (i.e. all nodes and edges, remove, add, rename, etc)
*   For threads currently interrupted, we support all topology changes other than renaming / removing nodes (as that thread could now be about to enter a node that no longer exists) — if this is a blocker please reach out and we can prioritize a solution.
*   For modifying state, we have full backwards and forwards compatibility for adding and removing keys
*   State keys that are renamed lose their saved state in existing threads
*   State keys whose types change in incompatible ways could currently cause issues in threads with state from before the change — if this is a blocker please reach out and we can prioritize a solution.

## Runtime context

You can pass runtime configuration and context to your graph.

```python
@dataclass
class ContextSchema:
    llm_provider: str = "openai"

graph = StateGraph(State, context_schema=ContextSchema)

graph.invoke(inputs, context={"llm_provider": "anthropic"})
```

Accessing context in a node:

```python
from langgraph.runtime import Runtime

def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)
    # ...
```

## Recursion limit

LangGraph prevents infinite loops by default using a recursion limit (default 25).

```python
graph.invoke(inputs, config={"recursion_limit": 5})
```

### Accessing and handling the recursion counter

You can access the current step via `config["metadata"]["langgraph_step"]`.

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

def my_node(state: dict, config: RunnableConfig) -> dict:
    current_step = config["metadata"]["langgraph_step"]
    print(f"Currently on step: {current_step}")
    return state
```

### Proactive recursion handling

You can use `RemainingSteps` to proactively manage the recursion limit.

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps
from langgraph.errors import GraphRecursionError

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps # Managed value - tracks steps until limit

# Proactive Approach (recommended) - using RemainingSteps
def agent_with_monitoring(state: State) -> dict:
    """Proactively monitor and handle recursion within the graph"""
    remaining = state["remaining_steps"]
    
    # Early detection - route to internal handling
    if remaining <= 2:
        return {
            "messages": ["Approaching limit, returning partial result"]
        }
    
    # Normal processing
    return {"messages": [f"Processing... ({remaining} steps remaining)"]}

def route_decision(state: State) -> Literal["agent", END]:
    if state["remaining_steps"] <= 2:
        return END
    return "agent"

# Build graph
builder = StateGraph(State)
builder.add_node("agent", agent_with_monitoring)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_decision)
graph = builder.compile()

# Proactive: Graph completes gracefully
result = graph.invoke({"messages": []}, {"recursion_limit": 10})

# Reactive Approach (fallback) - catching error externally
try:
    result = graph.invoke({"messages": []}, {"recursion_limit": 10})
except GraphRecursionError as e:
    # Handle externally after graph execution fails
    result = {"messages": ["Fallback: recursion limit exceeded"]}
```

*   Graceful degradation within the graph
*   Can save intermediate state in checkpoints
*   Better user experience with partial results
*   Graph completes normally (no exception)

### Other available metadata

You can inspect other metadata in the config:

```python
def inspect_metadata(state: dict, config: RunnableConfig) -> dict:
    metadata = config["metadata"]
    print(f"Step: {metadata['langgraph_step']}")
    print(f"Node: {metadata['langgraph_node']}")
    print(f"Triggers: {metadata['langgraph_triggers']}")
    print(f"Path: {metadata['langgraph_path']}")
    print(f"Checkpoint NS: {metadata['langgraph_checkpoint_ns']}")
    return state
```

## Visualization

See [this how-to guide](https://docs.langchain.com/oss/python/langgraph/use-graph-api#visualize-your-graph) for more on visualization.
