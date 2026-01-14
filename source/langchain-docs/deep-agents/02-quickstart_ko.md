# 빠른 시작

몇 분 만에 첫 번째 deep agent를 만들어보세요

이 가이드는 계획 수립, 파일 시스템 도구, 서브에이전트 기능을 갖춘 첫 번째 deep agent를 만드는 과정을 안내합니다. 리서치를 수행하고 보고서를 작성할 수 있는 리서치 에이전트를 만들게 됩니다.

## 사전 요구사항

시작하기 전에 모델 제공업체(예: Anthropic, OpenAI)로부터 API 키가 있는지 확인하세요.

> **Note**
> Deep agents는 [도구 호출](https://docs.langchain.com/oss/python/concepts/tool-calling)을 지원하는 모델이 필요합니다. 모델 구성 방법은 [커스터마이징](https://docs.langchain.com/oss/python/deepagents/customization)을 참조하세요.

## 1단계: 의존성 설치

```bash
pip install deepagents tavily-python
```

> **Note**
> 이 가이드는 [Tavily](https://tavily.com/)를 예제 검색 제공업체로 사용하지만, 다른 검색 API(예: DuckDuckGo, SerpAPI, Brave Search)로 대체할 수 있습니다.

## 2단계: API 키 설정

```bash
export ANTHROPIC_API_KEY="your-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

## 3단계: 검색 도구 만들기

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
```

## 4단계: Deep agent 만들기

```python
# 에이전트를 전문 리서처로 유도하는 시스템 프롬프트
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.
You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    tools=[internet_search],
    system_prompt=research_instructions
)
```

## 5단계: 에이전트 실행

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is langgraph?"}]
})

# 에이전트의 응답 출력
print(result["messages"][-1].content)
```

## 무슨 일이 일어났나요?

당신의 deep agent는 자동으로:

1. **접근 방식을 계획함**: 내장된 `write_todos` 도구를 사용하여 리서치 작업을 분해
2. **리서치 수행**: `internet_search` 도구를 호출하여 정보 수집
3. **컨텍스트 관리**: 파일 시스템 도구(`write_file`, `read_file`)를 사용하여 큰 검색 결과를 오프로드
4. **서브에이전트 생성** (필요시): 복잡한 하위 작업을 전문화된 서브에이전트에게 위임
5. **보고서 종합**: 발견한 내용을 일관된 응답으로 컴파일

## 다음 단계

첫 번째 deep agent를 만들었으니:

- **에이전트 커스터마이징**: 커스텀 시스템 프롬프트, 도구, 서브에이전트를 포함한 [커스터마이징 옵션](https://docs.langchain.com/oss/python/deepagents/customization)에 대해 알아보세요.
- **미들웨어 이해**: deep agents를 구동하는 [미들웨어 아키텍처](https://docs.langchain.com/oss/python/deepagents/middleware)를 자세히 살펴보세요.
- **장기 메모리 추가**: 대화 간 [영구 메모리](https://docs.langchain.com/oss/python/deepagents/long-term-memory)를 활성화하세요.
- **프로덕션 배포**: LangGraph 애플리케이션의 [배포 옵션](https://docs.langchain.com/oss/python/langgraph/deploy)에 대해 알아보세요.

---

<p align="center">
  <a href="01-overview_ko.md">← 이전: 개요</a> • <a href="README.md">목차</a> • <a href="03-customization_ko.md">다음: 커스터마이징 →</a>
</p>
