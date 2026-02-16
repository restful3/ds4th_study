# Application structure

A LangGraph application consists of one or more graphs, a configuration file (`langgraph.json`), a file that specifies dependencies, and an optional `.env` file that specifies environment variables.

This guide shows a typical structure of an application and shows you how to provide the required configuration to deploy an application with [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments).

> [!NOTE]
> LangSmith Deployment is a managed hosting platform for deploying and scaling LangGraph agents. It handles the infrastructure, scaling, and operational concerns so you can deploy your stateful, long-running agents directly from your repository. Learn more in the [Deployment documentation](https://docs.langchain.com/langsmith/deployments).

## Key concepts

To deploy using the LangSmith, the following information should be provided:

1.  A [LangGraph configuration file](https://docs.langchain.com/oss/python/langgraph/application-structure#configuration-file-concepts) (`langgraph.json`) that specifies the dependencies, graphs, and environment variables to use for the application.
2.  The [graphs](https://docs.langchain.com/oss/python/langgraph/application-structure#graphs) that implement the logic of the application.
3.  A file that specifies [dependencies](https://docs.langchain.com/oss/python/langgraph/application-structure#dependencies) required to run the application.
4.  [Environment variables](https://docs.langchain.com/oss/python/langgraph/application-structure#environment-variables) that are required for the application to run.

## File structure

Below are examples of directory structures for applications:

<details open>
<summary>Python (requirements.txt)</summary>

```bash
my-app/
├── my_agent             # all project code lies within here
│   ├── utils            # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py     # tools for your graph
│   │   ├── nodes.py     # node functions for your graph
│   │   └── state.py     # state definition of your graph
│   ├── __init__.py
│   └── agent.py         # code for constructing your graph
├── .env                 # environment variables
├── requirements.txt     # project dependencies
└── langgraph.json       # configuration file for LangGraph
```
</details>

<details>
<summary>Python (pyproject.toml)</summary>

```bash
my-app/
├── my_agent             # all project code lies within here
│   ├── utils            # utilities for your graph
│   │   ├── __init__.py
│   │   ├── tools.py     # tools for your graph
│   │   ├── nodes.py     # node functions for your graph
│   │   └── state.py     # state definition of your graph
│   ├── __init__.py
│   └── agent.py         # code for constructing your graph
├── .env                 # environment variables
├── langgraph.json       # configuration file for LangGraph
└── pyproject.toml       # dependencies for your project
```
</details>

> The directory structure of a LangGraph application can vary depending on the programming language and the package manager used.

## Configuration file

The `langgraph.json` file is a JSON file that specifies the dependencies, graphs, environment variables, and other settings required to deploy a LangGraph application.

See the [LangGraph configuration file reference](https://docs.langchain.com/langsmith/cli#configuration-file) for details on all supported keys in the JSON file.

> [!TIP]
> The [LangGraph CLI](https://docs.langchain.com/langsmith/cli) defaults to using the configuration file `langgraph.json` in the current directory.

### Examples

-   The dependencies involve a custom local package and the `langchain_openai` package.
-   A single graph will be loaded from the file `./your_package/your_file.py` with the variable `variable` .
-   The environment variables are loaded from the `.env` file.

```json
{
  "dependencies": ["langchain_openai", "./your_package"],
  "graphs": {
    "my_agent": "./your_package/your_file.py:agent"
  },
  "env": "./.env"
}
```

## Dependencies

A LangGraph application may depend on other Python packages. You will generally need to specify the following information for dependencies to be set up correctly:

1.  A file in the directory that specifies the dependencies (e.g. `requirements.txt`, `pyproject.toml`, or `package.json`).

2.  A `dependencies` key in the [LangGraph configuration file](https://docs.langchain.com/oss/python/langgraph/application-structure#configuration-file-concepts) that specifies the dependencies required to run the LangGraph application.

3.  Any additional binaries or system libraries can be specified using `dockerfile_lines` key in the [LangGraph configuration file](https://docs.langchain.com/oss/python/langgraph/application-structure#configuration-file-concepts).

## Graphs

Use the `graphs` key in the [LangGraph configuration file](https://docs.langchain.com/oss/python/langgraph/application-structure#configuration-file-concepts) to specify which graphs will be available in the deployed LangGraph application.

You can specify one or more graphs in the configuration file. Each graph is identified by a name (which should be unique) and a path for either: (1) the compiled graph or (2) a function that makes a graph is defined.

## Environment variables

If you’re working with a deployed LangGraph application locally, you can configure environment variables in the `env` key of the [LangGraph configuration file](https://docs.langchain.com/oss/python/langgraph/application-structure#configuration-file-concepts).

For a production deployment, you will typically want to configure the environment variables in the deployment environment.
