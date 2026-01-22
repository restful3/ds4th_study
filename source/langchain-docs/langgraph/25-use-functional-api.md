# Use the functional API

The [**Functional API**](https://docs.langchain.com/oss/python/langgraph/functional-api) allows you to add LangGraph’s key features — [persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/interrupts), and [streaming](https://docs.langchain.com/oss/python/langgraph/streaming) — to your applications with minimal changes to your existing code.

> **Tip**
> For conceptual information on the functional API, see [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api).

## Creating a simple workflow

When defining an `entrypoint`, input is restricted to the first argument of the function. To pass multiple inputs, you can use a dictionary.

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = inputs["value"]
    another_value = inputs["another_value"]
    ...

my_workflow.invoke({"value": 1, "another_value": 2})
```

<details>
<summary>Extended example: simple workflow</summary>

```python
import uuid
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

# Task that checks if a number is even
@task
def is_even(number: int) -> bool:
    return number % 2 == 0

# Task that formats a message
@task
def format_message(is_even: bool) -> str:
    return "The number is even." if is_even else "The number is odd."

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(inputs: dict) -> str:
    """Simple workflow to classify a number."""
    even = is_even(inputs["number"]).result()
    return format_message(even).result()

# Run the workflow with a unique thread ID
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke({"number": 7}, config=config)
print(result)
```
</details>

<details>
<summary>Extended example: Compose an essay with an LLM</summary>

This example demonstrates how to use the `@task` and `@entrypoint` decorators syntactically. Given that a checkpointer is provided, the workflow results will be persisted in the checkpointer.

```python
import uuid
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model('gpt-3.5-turbo')

# Task: generate essay using an LLM
@task
def compose_essay(topic: str) -> str:
    """Generate an essay about the given topic."""
    return model.invoke([
        {"role": "system", "content": "You are a helpful assistant that writes essays."},
        {"role": "user", "content": f"Write an essay about {topic}."}
    ]).content

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(topic: str) -> str:
    """Simple workflow that generates an essay with an LLM."""
    return compose_essay(topic).result()

# Execute the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke("the history of flight", config=config)
print(result)
```
</details>

## Parallel execution

Tasks can be executed in parallel by invoking them concurrently and waiting for the results. This is useful for improving performance in IO bound tasks (e.g., calling APIs for LLMs).

```python
@task
def add_one(number: int) -> int:
    return number + 1

@entrypoint(checkpointer=checkpointer)
def graph(numbers: list[int]) -> list[str]:
    futures = [add_one(i) for i in numbers]
    return [f.result() for f in futures]
```

<details>
<summary>Extended example: parallel LLM calls</summary>

This example demonstrates how to run multiple LLM calls in parallel using `@task`. Each call generates a paragraph on a different topic, and results are joined into a single text output.

```python
import uuid
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

# Initialize the LLM model
model = init_chat_model("gpt-3.5-turbo")

# Task that generates a paragraph about a given topic
@task
def generate_paragraph(topic: str) -> str:
    response = model.invoke([
        {"role": "system", "content": "You are a helpful assistant that writes educational paragraphs."},
        {"role": "user", "content": f"Write a paragraph about {topic}."}
    ])
    return response.content

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(topics: list[str]) -> str:
    """Generates multiple paragraphs in parallel and combines them."""
    futures = [generate_paragraph(topic) for topic in topics]
    paragraphs = [f.result() for f in futures]
    return "\n\n".join(paragraphs)

# Run the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke(["quantum computing", "climate change", "history of aviation"], config=config)
print(result)
```
This example uses LangGraph’s concurrency model to improve execution time, especially when tasks involve I/O like LLM completions.
</details>

## Calling graphs

The **Functional API** and the [**Graph API**](https://docs.langchain.com/oss/python/langgraph/graph-api) can be used together in the same application as they share the same underlying runtime.

```python
from langgraph.func import entrypoint
from langgraph.graph import StateGraph

builder = StateGraph()
...
some_graph = builder.compile()

@entrypoint()
def some_workflow(some_input: dict) -> int:
    # Call a graph defined using the graph API
    result_1 = some_graph.invoke(...)
    # Call another graph defined using the graph API
    result_2 = another_graph.invoke(...)
    return {
        "result_1": result_1,
        "result_2": result_2
    }
```

<details>
<summary>Extended example: calling a simple graph from the functional API</summary>

```python
import uuid
from typing import TypedDict
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

# Define the shared state type
class State(TypedDict):
    foo: int

# Define a simple transformation node
def double(state: State) -> State:
    return {"foo": state["foo"] * 2}

# Build the graph using the Graph API
builder = StateGraph(State)
builder.add_node("double", double)
builder.set_entry_point("double")
graph = builder.compile()

# Define the functional API workflow
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(x: int) -> dict:
    result = graph.invoke({"foo": x})
    return {"bar": result["foo"]}

# Execute the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(workflow.invoke(5, config=config))  # Output: {'bar': 10}
```
</details>

## Call other entrypoints

You can call other **entrypoints** from within an **entrypoint** or a **task**.

```python
@entrypoint() # Will automatically use the checkpointer from the parent entrypoint
def some_other_workflow(inputs: dict) -> int:
    return inputs["value"]

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    value = some_other_workflow.invoke({"value": 1})
    return value
```

<details>
<summary>Extended example: calling another entrypoint</summary>

```python
import uuid
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

# Initialize a checkpointer
checkpointer = InMemorySaver()

# A reusable sub-workflow that multiplies a number
@entrypoint()
def multiply(inputs: dict) -> int:
    return inputs["a"] * inputs["b"]

# Main workflow that invokes the sub-workflow
@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> dict:
    result = multiply.invoke({"a": inputs["x"], "b": inputs["y"]})
    return {"product": result}

# Execute the main workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(main.invoke({"x": 6, "y": 7}, config=config))  # Output: {'product': 42}
```
</details>

## Streaming

The **Functional API** uses the same streaming mechanism as the **Graph API**. Please read the [**streaming guide**](https://docs.langchain.com/oss/python/langgraph/streaming) section for more details.

Example of using the streaming API to stream both updates and custom data.

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer   

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> int:
    writer = get_stream_writer()   
    writer("Started processing")   
    result = inputs["x"] * 2
    writer(f"Result is {result}")   
    return result

config = {"configurable": {"thread_id": "abc"}}

for mode, chunk in main.stream(   
    {"x": 5},
    stream_mode=["custom", "updates"],   
    config=config
):
    print(f"{mode}: {chunk}")
```

1. Import `get_stream_writer` from `langgraph.config`.
2. Obtain a stream writer instance within the entrypoint.
3. Emit custom data before computation begins.
4. Emit another custom message after computing the result.
5. Use `.stream()` to process streamed output.
6. Specify which streaming modes to use.

```python
('updates', {'add_one': 2})
('updates', {'add_two': 3})
('custom', 'hello')
('custom', 'world')
('updates', {'main': 5})
```

> **Warning: Async with Python < 3.11**
> If using Python < 3.11 and writing async code, using `get_stream_writer` will not work. Instead please use the `StreamWriter` class directly. See [Async with Python < 3.11](https://docs.langchain.com/oss/python/langgraph/streaming#async) for more details.

```python
from langgraph.types import StreamWriter

@entrypoint(checkpointer=checkpointer)
async def main(inputs: dict, writer: StreamWriter) -> int:  
...
```

## Retry policy

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

# Simulating network failure for demonstration
attempts = 0

# Configure RetryPolicy to retry on ValueError
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def get_info():
    global attempts
    attempts += 1
    if attempts < 2:
        raise ValueError('Failure')
    return "OK"

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer):
    return get_info().result()

config = {"configurable": {"thread_id": "1"}}
main.invoke({'any_input': 'foobar'}, config=config)
# Output: 'OK'
```

## Caching tasks

```python
import time
from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy

@task(cache_policy=CachePolicy(ttl=120))    
def slow_add(x: int) -> int:
    time.sleep(1)
    return x * 2

@entrypoint(cache=InMemoryCache())
def main(inputs: dict) -> dict[str, int]:
    result1 = slow_add(inputs["x"]).result()
    result2 = slow_add(inputs["x"]).result()
    return {"result1": result1, "result2": result2}

for chunk in main.stream({"x": 5}, stream_mode="updates"):
    print(chunk)
```

## Resuming after an error

```python
import time
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter

attempts = 0

@task()
def get_info():
    global attempts
    attempts += 1
    if attempts < 2:
        raise ValueError("Failure")
    return "OK"

checkpointer = InMemorySaver()

@task
def slow_task():
    time.sleep(1)
    return "Ran slow task."

@entrypoint(checkpointer=checkpointer)
def main(inputs, writer: StreamWriter):
    slow_task_result = slow_task().result()
    get_info().result()
    return slow_task_result

config = {"configurable": {"thread_id": "1"}}

try:
    main.invoke({'any_input': 'foobar'}, config=config)
except ValueError:
    pass

# Resuming execution
main.invoke(None, config=config)
# Output: 'Ran slow task.'
```

## Human-in-the-loop

The functional API supports human-in-the-loop workflows using the `interrupt` function and the `Command` primitive.

### Basic human-in-the-loop workflow

```python
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

@task
def step_1(input_query):
    return f"{input_query} bar"

@task
def human_feedback(input_query):
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"

@task
def step_3(input_query):
    return f"{input_query} qux"

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def graph(input_query):
    result_1 = step_1(input_query).result()
    result_2 = human_feedback(result_1).result()
    result_3 = step_3(result_2).result()
    return result_3

# Usage
config = {"configurable": {"thread_id": "1"}}
# First stream hits interrupt
# Continue with Command(resume="baz")
```

### Review tool calls

```python
from typing import Union

def review_tool_call(tool_call: ToolCall) -> Union[ToolCall, ToolMessage]:
    human_review = interrupt({"question": "Is this correct?", "tool_call": tool_call})
    # Process return action and data...
```

## Short-term memory

Short-term memory allows storing information across different invocations of the same thread id.

### Manage checkpoints
Checkpoints can be managed via `graph.get_state(config)` and `graph.get_state_history(config)`.

### Decouple return value from saved value
Use `entrypoint.final` to separate the value returned to the caller from the state saved in the checkpoint.

```python
@entrypoint(checkpointer=checkpointer)
def accumulate(n: int, *, previous: int | None) -> entrypoint.final[int, int]:
    previous = previous or 0
    total = previous + n
    return entrypoint.final(value=previous, save=total)
```

### Chatbot example
A full chatbot implementation using `add_messages` and `InMemorySaver` is supported through the functional API by passing the `previous` state directly into the workflow.

## Long-term memory
Allows storing information across different thread ids for cross-conversation learning.

## Workflows
See the workflows and agent guide for more examples.
