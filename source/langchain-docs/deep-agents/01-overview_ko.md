# Deep Agents 개요

> 계획을 수립하고, 서브에이전트를 사용하며, 복잡한 작업을 위해 파일 시스템을 활용할 수 있는 에이전트 구축

[deepagents](https://pypi.org/project/deepagents/)는 복잡한 다단계 작업을 처리할 수 있는 에이전트를 구축하기 위한 독립형 라이브러리입니다. LangGraph를 기반으로 구축되었으며 Claude Code, Deep Research, Manus와 같은 애플리케이션에서 영감을 받아, deep agents는 계획 수립 기능, 컨텍스트 관리를 위한 파일 시스템, 그리고 서브에이전트를 생성하는 기능을 갖추고 있습니다.

---

## Deep agents를 사용해야 할 때

다음과 같은 기능이 필요한 에이전트가 필요할 때 deep agents를 사용하세요:

- 계획 수립과 분해가 필요한 **복잡한 다단계 작업 처리**
- 파일 시스템 도구를 통한 **대량의 컨텍스트 관리**
- 컨텍스트 격리를 위해 전문화된 서브에이전트에 **작업 위임**
- 대화 및 스레드 간 **메모리 유지**

더 간단한 사용 사례의 경우, LangChain의 [`create_agent`](https://docs.langchain.com/oss/python/langchain/agents)를 사용하거나 커스텀 [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) 워크플로우를 구축하는 것을 고려하세요.

---

## 핵심 기능

### 계획 수립 및 작업 분해

Deep agents는 에이전트가 복잡한 작업을 개별 단계로 분해하고, 진행 상황을 추적하며, 새로운 정보가 나타나면 계획을 조정할 수 있게 해주는 내장 `write_todos` 도구를 포함합니다.

### 컨텍스트 관리

파일 시스템 도구(`ls`, `read_file`, `write_file`, `edit_file`)를 통해 에이전트는 대량의 컨텍스트를 메모리로 오프로드하여 컨텍스트 윈도우 오버플로우를 방지하고 가변 길이 도구 결과로 작업할 수 있습니다.

### 서브에이전트 생성

내장 `task` 도구를 통해 에이전트는 컨텍스트 격리를 위한 전문화된 서브에이전트를 생성할 수 있습니다. 이를 통해 메인 에이전트의 컨텍스트를 깔끔하게 유지하면서도 특정 하위 작업을 깊이 있게 처리할 수 있습니다.

### 장기 메모리

LangGraph의 Store를 사용하여 스레드 간 지속적인 메모리로 에이전트를 확장하세요. 에이전트는 이전 대화의 정보를 저장하고 검색할 수 있습니다.

---

## LangChain 생태계와의 관계

Deep agents는 다음을 기반으로 구축되었습니다:

- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) - 기본 그래프 실행 및 상태 관리 제공
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview) - 도구 및 모델 통합이 deep agents와 원활하게 작동
- [LangSmith](https://smith.langchain.com/) - 관찰성, 평가 및 배포

Deep agents 애플리케이션은 [LangSmith Deployment](https://docs.smith.langchain.com/deployment)를 통해 배포하고 [LangSmith Observability](https://docs.smith.langchain.com/observability)로 모니터링할 수 있습니다.

---

## 시작하기

| 가이드 | 설명 |
|-------|------|
| [빠른 시작](https://docs.langchain.com/oss/python/deepagents/quickstart) | 첫 번째 deep agent 구축 |
| [커스터마이징](https://docs.langchain.com/oss/python/deepagents/customization) | 커스터마이징 옵션 알아보기 |
| [미들웨어](https://docs.langchain.com/oss/python/deepagents/middleware) | 미들웨어 아키텍처 이해 |
| [CLI](https://docs.langchain.com/oss/python/deepagents/cli) | Deep Agents CLI 사용 |
| [레퍼런스](https://reference.langchain.com/python/deepagents/) | deepagents API 레퍼런스 보기 |

---

*출처: [https://docs.langchain.com/oss/python/deepagents/overview](https://docs.langchain.com/oss/python/deepagents/overview)*

---

<p align="center">
  <a href="README.md">목차</a> • <a href="02-quickstart_ko.md">다음: 빠른 시작 →</a>
</p>
