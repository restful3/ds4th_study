# Install LangGraph

To install the base LangGraph package:

**pip**
```bash
pip install -U langgraph
```

**uv**
```bash
uv add langgraph
```

To use LangGraph you will usually want to access LLMs and define tools. You can do this however you see fit.

One way to do this (which we will use in the docs) is to use [LangChain](https://docs.langchain.com/oss/python/langchain/overview).

Install LangChain with:

**pip**
```bash
pip install -U langchain
# Requires Python 3.10+
```

**uv**
```bash
uv add langchain
# Requires Python 3.10+
```

To work with specific LLM provider packages, you will need install them separately.

Refer to the [integrations](https://docs.langchain.com/oss/python/integrations/providers/overview) page for provider-specific installation instructions.
