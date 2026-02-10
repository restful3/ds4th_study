# LangChain 설치

LangChain 패키지를 설치하려면:

#### pip

```bash
pip install -U langchain
# Python 3.10 이상 필요
```

#### uv

```bash
uv add langchain
# Python 3.10 이상 필요
```

LangChain은 수백 개의 LLM과 수천 개의 다른 통합을 제공합니다. 이들은 독립적인 공급자 패키지에 있습니다.

#### pip

```bash
# OpenAI 통합 설치
pip install -U langchain-openai

# Anthropic 통합 설치
pip install -U langchain-anthropic
```

#### uv

```bash
# OpenAI 통합 설치
uv add langchain-openai

# Anthropic 통합 설치
uv add langchain-anthropic
```

---

> 사용 가능한 모든 통합의 목록은 [통합 탭](/oss/python/integrations/providers/overview)을 참조합니다.

이제 LangChain을 설치했으므로 [빠른 시작 가이드](/oss/python/langchain/quickstart)를 따라 시작할 수 있습니다.
