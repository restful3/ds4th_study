# Run a local server

This guide shows you how to run a LangGraph application locally.

## Prerequisites

Before you begin, ensure you have the following:

- An API key for [LangSmith](https://smith.langchain.com/settings) - free to sign up

## 1. Install the LangGraph CLI

**pip**
```bash
# Python >= 3.11 is required.
pip install -U "langgraph-cli[inmem]"
```

**uv**
```bash
# Python >= 3.11 is required.
uv add "langgraph-cli[inmem]"
```

## 2. Create a LangGraph app

Create a new app from the [new-langgraph-project-python template](https://github.com/langchain-ai/new-langgraph-project-python). This template demonstrates a single-node application you can extend with your own logic.

```bash
langgraph new path/to/your/app --template new-langgraph-project-python
```

> [!TIP]
> **Additional templates**
> If you use `langgraph new` without specifying a template, you will be presented with an interactive menu that will allow you to choose from a list of available templates.

## 3. Install dependencies

In the root of your new LangGraph app, install the dependencies in `edit` mode so your local changes are used by the server:

**pip**
```bash
pip install -e .
```

**uv**
```bash
uv sync
```

## 4. Create a `.env` file

You will find a `.env.example` in the root of your new LangGraph app. Create a `.env` file in the root of your new LangGraph app and copy the contents of the `.env.example` file into it, filling in the necessary API keys:

```text
LANGSMITH_API_KEY=lsv2...
```

## 5. Launch Agent server

Start the LangGraph API server locally:

```bash
langgraph dev
```

Sample output:

```text
INFO:langgraph_api.cli:

   Welcome to
   ╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
   ║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
   ╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

 - 🚀 API: http://127.0.0.1:2024
 - 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
 - 📚 API Docs: http://127.0.0.1:2024/docs

 This in-memory server is designed for development and testing.
 For production use, please use LangSmith Deployment.
```

The `langgraph dev` command starts Agent Server in an in-memory mode. This mode is suitable for development and testing purposes. For production use, deploy Agent Server with access to a persistent storage backend. For more information, see the [Platform setup overview](https://docs.langchain.com/langsmith/platform-setup).

## 6. Test your application in Studio

[Studio](https://docs.langchain.com/langsmith/studio) is a specialized UI that you can connect to LangGraph API server to visualize, interact with, and debug your application locally. Test your graph in Studio by visiting the URL provided in the output of the `langgraph dev` command:

```text
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

For an Agent Server running on a custom host/port, update the `baseUrl` query parameter in the URL. For example, if your server is running on `http://myhost:3000`:

```text
https://smith.langchain.com/studio/?baseUrl=http://myhost:3000
```

<details>
<summary>Safari compatibility</summary>

Currently, LangGraph Studio is not fully compatible with Safari. We recommend using Chrome, Brave or any other Chromium based browser.

If you must use Safari, you may need to enable "Prevent Cross-Site Tracking" in Safari's Settings > Privacy.

</details>

## 7. Test the API

### Python SDK (async)

1. Install the LangGraph Python SDK:

```bash
pip install langgraph-sdk
```

2. Send a message to the assistant (threadless run):

```python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

asyncio.run(main())
```

### Python SDK (sync)

1. Install the LangGraph Python SDK:

```bash
pip install langgraph-sdk
```

2. Send a message to the assistant (threadless run):

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

for chunk in client.runs.stream(
    None,  # Threadless run
    "agent", # Name of assistant. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
        }],
    },
    stream_mode="messages-tuple",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

### Rest API

```bash
curl -s --request POST \
    --url "http://localhost:2024/runs/stream" \
    --header 'Content-Type: application/json' \
    --data "{
        \"assistant_id\": \"agent\",
        \"input\": {
            \"messages\": [
                {
                    \"role\": \"human\",
                    \"content\": \"What is LangGraph?\"
                }
            ]
        },
        \"stream_mode\": \"messages-tuple\"
    }"
```

## Next steps

Now that you have a LangGraph app running locally, take your journey further by exploring deployment and advanced features:

- **Deployment quickstart**: Deploy your LangGraph app using LangSmith.
- **LangSmith**: Learn about foundational LangSmith concepts.
- **SDK Reference**: Explore the SDK API Reference.
