# LangSmith Deployment

When you're ready to deploy your LangChain agent to production, LangSmith provides a managed hosting platform designed for agent workloads. Traditional hosting platforms are built for stateless, short-lived web applications, while LangGraph is **purpose-built for stateful, long-running agents** that require persistent state and background execution. LangSmith handles the infrastructure, scaling, and operational concerns so you can deploy directly from your repository.

## Prerequisites

Before you begin, ensure you have the following:

- A [**GitHub account**](https://github.com/)
- A [**LangSmith account**](https://smith.langchain.com/) (free to sign up)

## Deploy your agent

### 1. Create a repository on GitHub

Your application's code must reside in a GitHub repository to be deployed on LangSmith. Both public and private repositories are supported. For this quickstart, first make sure your app is LangGraph-compatible by following the [**local server setup guide**](https://docs.langchain.com/oss/python/langchain/studio). Then, push your code to the repository.

### 2. Deploy to LangSmith

1. **Navigate to LangSmith Deployment**

   Log in to [LangSmith](https://smith.langchain.com/). In the left sidebar, select **Deployments**.

2. **Create new deployment**

   Click the **+ New Deployment** button. A pane will open where you can fill in the required fields.

3. **Link repository**

   If you are a first time user or adding a private repository that has not been previously connected, click the **Add new account** button and follow the instructions to connect your GitHub account.

4. **Deploy repository**

   Select your application's repository. Click **Submit** to deploy. This may take about 15 minutes to complete. You can check the status in the **Deployment details** view.

### 3. Test your application in Studio

Once your application is deployed:

1. Select the deployment you just created to view more details.
2. Click the **Studio** button in the top right corner. Studio will open to display your graph.

### 4. Get the API URL for your deployment

1. In the **Deployment details** view in LangGraph, click the **API URL** to copy it to your clipboard.
2. Click the `URL` to copy it to the clipboard.

### 5. Test the API

You can now test the API:

#### Python

1. Install LangGraph Python:

```bash
pip install langgraph-sdk
```

2. Send a message to the agent:

```python
from langgraph_sdk import get_sync_client  # or get_client for async

client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")

for chunk in client.runs.stream(
    None,       # Threadless run
    "agent",    # Name of agent. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?",
        }],
    },
    stream_mode="updates",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

#### Rest API

```bash
curl -s --request POST \
  --url <DEPLOYMENT_URL>/runs/stream \
  --header 'Content-Type: application/json' \
  --header "X-Api-Key: <LANGSMITH API KEY>" \
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
    \"stream_mode\": \"updates\"
  }"
```

> [!TIP]
> LangSmith offers additional hosting options, including self-hosted and hybrid. For more information, please see the [Platform setup overview](https://docs.smith.langchain.com/self_hosting).
