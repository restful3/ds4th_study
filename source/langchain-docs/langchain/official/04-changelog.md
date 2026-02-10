# Changelog

Log of updates and improvements to our Python packages.

> Subscribe: Our changelog includes an RSS feed that can integrate with Slack, email, Discord bots like Readybot or RSS Feeds to Discord Bot, and other subscription tools.

---

## Dec 15, 2025

**langchain**, **integrations**

### **langchain v1.2.0**

*   **`create_agent`**: Simplified support for provider-specific tool parameters and definitions via a new `extras` attribute on `tools`. Examples:
    *   Provider-specific configuration such as Anthropic’s [programmatic tool calling](/oss/python/integrations/chat/anthropic#programmatic-tool-calling) and [tool search](/oss/python/integrations/chat/anthropic#tool-search).
    *   Built-in tools that are executed client-side, as supported by [Anthropic](/oss/python/integrations/chat/anthropic#built-in-tools), [OpenAI](/oss/python/integrations/chat/openai#responses-api), and other providers.
*   Support for strict schema-adherence in agent `response_format` (see [ProviderStrategy](/oss/python/langchain/structured-output#provider-strategy) docs).

---

## Dec 8, 2025

**langchain**, **integrations**

#### langchain-google-genai v4.0.0

We’re re-written the Google GenAI integration to use Google’s consolidated Generative AI SDK, which provides access to the Gemini API and Vertex AI Platform under the same interface. This includes minimal breaking changes as well as deprecated packages in `langchain-google-vertexai`.

See the full [release notes and migration guide](https://github.com/langchain-ai/langchain-google/discussions/1422) for details.

---

## Nov 25, 2025

**langchain**

#### langchain v1.1.0

*   **[Model profiles](/oss/python/langchain/models#model-profiles)**: Chat models now expose supported features and capabilities through a `.profile` attribute. These data are derived from [models.dev](https://models.dev), an open source project providing model capability data.
*   **Summarization middleware**: Updated to support flexible trigger points using model profiles for context-aware summarization.
*   **[Structured output](/oss/python/langchain/structured-output)**: `ProviderStrategy` support (native structured output) can now be inferred from model profiles.
*   **[SystemMessage for create_agent](/oss/python/langchain/middleware/custom#working-with-system-messages)**: Support for passing `SystemMessage` instances directly to `create_agent`’s `system_prompt` parameter, enabling advanced features like cache control and structured content blocks.
*   **[Model retry middleware](/oss/python/langchain/middleware/built-in#model-retry)**: New middleware for automatically retrying failed model calls with configurable exponential backoff.
*   **[Content moderation middleware](/oss/python/langchain/middleware/built-in#content-moderation)**: OpenAI content moderation middleware for detecting and handling unsafe content in agent interactions. Supports checking user input, model output, and tool results.

---

## Oct 20, 2025

**langchain**, **langgraph**

#### v1.0.0

##### langchain
*   [Release notes](/oss/python/releases/langchain-v1)
*   [Migration guide](/oss/python/migrate/langchain-v1)

##### langgraph
*   [Release notes](/oss/python/releases/langgraph-v1)
*   [Migration guide](/oss/python/migrate/langgraph-v1)

> If you encounter any issues or have feedback, please [open an issue](https://github.com/langchain-ai/docs/issues/new/choose) so we can improve. To view v0.x documentation, [go to the archived content](https://github.com/langchain-ai/langchain/tree/v0.3/docs/docs) and [API reference](https://reference.langchain.com/v0.3/python/).
