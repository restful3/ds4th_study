# Install LangChain

To install the LangChain package:

#### pip

```bash
pip install -U langchain
# Requires Python 3.10+
```

#### uv

```bash
uv add langchain
# Requires Python 3.10+
```

LangChain provides integrations to hundreds of LLMs and thousands of other integrations. These live in independent provider packages.

#### pip

```bash
# Installing the OpenAI integration
pip install -U langchain-openai

# Installing the Anthropic integration
pip install -U langchain-anthropic
```

#### uv

```bash
# Installing the OpenAI integration
uv add langchain-openai

# Installing the Anthropic integration
uv add langchain-anthropic
```

---

> See the [Integrations tab](/oss/python/integrations/providers/overview) for a full list of available integrations.

Now that you have LangChain installed, you can get started by following the [Quickstart guide](/oss/python/langchain/quickstart).
