# Use time-travel

When working with non-deterministic systems that make model-based decisions (e.g., agents powered by LLMs), it can be useful to examine their decision-making process in detail:

1.  💡 **Understand reasoning**: Analyze the steps that led to a successful result.
2.  🐞 **Debug mistakes**: Identify where and why errors occurred.
3.  🔍 **Explore alternatives**: Test different paths to uncover better solutions.

LangGraph provides [time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) functionality to support these use cases. Specifically, you can resume execution from a prior checkpoint — either replaying the same state or modifying it to explore alternatives. In all cases, resuming past execution produces a new fork in the history.

To use [time-travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel) in LangGraph:

1.  **[Run the graph](#1-run-the-graph)** with initial inputs using `invoke` or `stream` methods.
2.  **[Identify a checkpoint in an existing thread](#2-identify-a-checkpoint)**: Use the `get_state_history` method to retrieve the execution history for a specific `thread_id` and locate the desired `checkpoint_id`. Alternatively, set an [interrupt](https://docs.langchain.com/oss/python/langgraph/interrupts) before the node(s) where you want execution to pause. You can then find the most recent checkpoint recorded up to that interrupt.
3.  **[Update the graph state (optional)](#3-update-the-state)**: Use the `update_state` method to modify the graph’s state at the checkpoint and resume execution from alternative state.
4.  **[Resume execution from the checkpoint](#4-resume-execution-from-the-checkpoint)**: Use the `invoke` or `stream` methods with an input of `None` and a configuration containing the appropriate `thread_id` and `checkpoint_id`.

> [!TIP]
> For a conceptual overview of time-travel, see [Time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel).

---

## In a workflow

This example builds a simple LangGraph workflow that generates a joke topic and writes a joke using an LLM. It demonstrates how to run the graph, retrieve past execution checkpoints, optionally modify the state, and resume execution from a chosen checkpoint to explore alternate outcomes.

### Setup

#### Python
```python
%%capture --no-stderr
pip install --quiet -U langgraph langchain_anthropic
```

```python
import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("ANTHROPIC_API_KEY")
```

```python
import uuid
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0,
)

def generate_topic(state: State):
    """LLM call to generate a topic for the joke"""
    msg = model.invoke("Give me a funny topic for a joke")
    return {"topic": msg.content}

def write_joke(state: State):
    """LLM call to write a joke based on the topic"""
    msg = model.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

# Build workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_topic", generate_topic)
workflow.add_node("write_joke", write_joke)

# Add edges to connect nodes
workflow.add_edge(START, "generate_topic")
workflow.add_edge("generate_topic", "write_joke")
workflow.add_edge("write_joke", END)

# Compile
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
graph
```

#### TypeScript
```bash
npm install @langchain/langgraph @langchain/anthropic
```

```javascript
process.env.ANTHROPIC_API_KEY = "YOUR_API_KEY";
```

```javascript
import { v4 as uuidv4 } from "uuid";
import * as z from "zod";
import { StateGraph, START, END } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { MemorySaver } from "@langchain/langgraph";

const State = z.object({
  topic: z.string().optional(),
  joke: z.string().optional(),
});

const model = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
  temperature: 0,
});

// Build workflow
const workflow = new StateGraph(State)
  // Add nodes
  .addNode("generateTopic", async (state) => {
    // LLM call to generate a topic for the joke
    const msg = await model.invoke("Give me a funny topic for a joke");
    return { topic: msg.content };
  })
  .addNode("writeJoke", async (state) => {
    // LLM call to write a joke based on the topic
    const msg = await model.invoke(`Write a short joke about ${state.topic}`);
    return { joke: msg.content };
  })
  // Add edges to connect nodes
  .addEdge(START, "generateTopic")
  .addEdge("generateTopic", "writeJoke")
  .addEdge("writeJoke", END);

// Compile
const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });
```

### 1. Run the graph

#### Python
```python
config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}
state = graph.invoke({}, config)

print(state["topic"])
print()
print(state["joke"])
```

#### TypeScript
```javascript
const config = {
  configurable: {
    thread_id: uuidv4(),
  },
};

const state = await graph.invoke({}, config);

console.log(state.topic);
console.log();
console.log(state.joke);
```

**Output:**
```
How about "The Secret Life of Socks in the Dryer"? You know, exploring the mysterious phenomenon of how socks go into the laundry as pairs but come out as singles. Where do they go? Are they starting new lives elsewhere? Is there a sock paradise we don't know about? There's a lot of comedic potential in the everyday mystery that unites us all!

# The Secret Life of Socks in the Dryer

I finally discovered where all my missing socks go after the dryer. Turns out they're not missing at all—they've just eloped with someone else's socks from the laundromat to start new lives together.

My blue argyle is now living in Bermuda with a red polka dot, posting vacation photos on Sockstagram and sending me lint as alimony.
```

### 2. Identify a checkpoint

#### Python
```python
# The states are returned in reverse chronological order.
states = list(graph.get_state_history(config))

for state in states:
    print(state.next)
    print(state.config["configurable"]["checkpoint_id"])
    print()
```

#### TypeScript
```javascript
// The states are returned in reverse chronological order.
const states = [];
for await (const state of graph.getStateHistory(config)) {
  states.push(state);
}

for (const state of states) {
  console.log(state.next);
  console.log(state.config.configurable?.checkpoint_id);
  console.log();
}
```

**Output:**
```
()
1f02ac4a-ec9f-6524-8002-8f7b0bbeed0e

('write_joke',)
1f02ac4a-ce2a-6494-8001-cb2e2d651227

('generate_topic',)
1f02ac4a-a4e0-630d-8000-b73c254ba748

('__start__',)
1f02ac4a-a4dd-665e-bfff-e6c8c44315d9
```

#### Python
```python
# This is the state before last (states are listed in chronological order)
selected_state = states[1]
print(selected_state.next)
print(selected_state.values)
```

#### TypeScript
```javascript
// This is the state before last (states are listed in chronological order)
const selectedState = states[1];
console.log(selectedState.next);
console.log(selectedState.values);
```

**Output:**
```
('write_joke',)
{'topic': 'How about "The Secret Life of Socks in the Dryer"? You know, exploring the mysterious phenomenon of how socks go into the laundry as pairs but come out as singles. ...'}
```

### 3. Update the state

#### Python
```python
new_config = graph.update_state(selected_state.config, values={"topic": "chickens"})
print(new_config)
```

#### TypeScript
```javascript
const newConfig = await graph.updateState(selectedState.config, {
  topic: "chickens",
});
console.log(newConfig);
```

**Output:**
```
{'configurable': {'thread_id': '...', 'checkpoint_ns': '', 'checkpoint_id': '1f02ac4a-ecee-600b-8002-a1d21df32e4c'}}
```

### 4. Resume execution from the checkpoint

#### Python
```python
graph.invoke(None, new_config)
```

#### TypeScript
```javascript
await graph.invoke(null, newConfig);
```

**Output:**
```python
{'topic': 'chickens',
 'joke': 'Why did the chicken join a band?\n\nBecause it had excellent drumsticks!'}
```
