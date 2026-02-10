# LangChain 개요

LangChain은 사전 구축된 Agent 아키텍처와 모든 모델 또는 Tool에 대한 통합을 갖춘 오픈소스 프레임워크입니다. 생태계가 진화하는 만큼 빠르게 적응하는 Agent를 구축할 수 있습니다.

LangChain은 LLM으로 구동되는 Agent와 애플리케이션을 구축하기 시작하는 가장 간단한 방법입니다. 10줄 이하의 코드로 OpenAI, Anthropic, Google 등에 연결할 수 있습니다. LangChain은 사전 구축된 Agent 아키텍처와 모델 통합을 제공하여 빠르게 시작할 수 있고 LLM을 Agent와 애플리케이션에 원활하게 통합할 수 있습니다.

빠르게 Agent와 자율형 애플리케이션을 구축하려면 LangChain을 사용하기를 권장합니다. 결정적 워크플로우와 Agent 워크플로우의 조합, 대폭적인 사용자 정의, 신중하게 제어된 레이턴시가 필요한 고급 요구사항이 있는 경우 저수준 Agent 오케스트레이션 프레임워크 및 런타임인 LangGraph를 사용합니다.

LangChain Agent는 지속 가능한 실행, 스트리밍, 인간-루프 상호작용, 지속성 등을 제공하기 위해 LangGraph 위에 구축되었습니다. 기본 LangChain Agent 사용을 위해 LangGraph를 알 필요가 없습니다.

---
## Agent 만들기

```python
# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 가져옵니다."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Agent 실행
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

LangChain으로 고유한 Agent와 애플리케이션을 구축하기 시작하려면 [설치 지침](/oss/python/langchain/install)과 [빠른 시작 가이드](/oss/python/langchain/quickstart)를 참조합니다.

## 핵심 이점

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>표준 모델 인터페이스</h3>
      <p>서로 다른 공급자는 응답 형식을 포함한 모델과 상호작용하기 위해 고유한 API를 가지고 있습니다. LangChain은 모델과의 상호작용 방식을 표준화하여 공급자를 원활하게 교체할 수 있고 종속성을 피할 수 있습니다.</p>
      <a href="/oss/python/langchain/models">자세히 알아보기 ></a>
    </td>
    <td valign="top" width="50%">
      <h3>사용하기 쉽고 매우 유연한 Agent</h3>
      <p>LangChain의 Agent 추상화는 사용하기 쉽도록 설계되어 10줄 이하의 코드로 간단한 Agent를 구축할 수 있습니다. 또한 원하는 모든 문맥 엔지니어링을 수행할 수 있을 만큼 충분한 유연성을 제공합니다.</p>
      <a href="/oss/python/langchain/agents">자세히 알아보기 ></a>
    </td>
  </tr>
  <tr>
    <td valign="top" width="50%">
      <h3>LangGraph 위에 구축</h3>
      <p>LangChain의 Agent는 LangGraph 위에 구축되었습니다. 이를 통해 LangGraph의 지속 가능한 실행, 인간-루프 상호작용 지원, 지속성 등을 활용할 수 있습니다.</p>
      <a href="/oss/python/langgraph/overview">자세히 알아보기 ></a>
    </td>
    <td valign="top" width="50%">
      <h3>LangSmith로 디버깅</h3>
      <p>실행 경로를 추적하고, 상태 전환을 캡처하며, 상세한 런타임 메트릭을 제공하는 시각화 도구로 복잡한 Agent 동작에 대한 심층적인 가시성을 확보합니다.</p>
      <a href="/langsmith/home">자세히 알아보기 ></a>
    </td>
  </tr>
</table>
