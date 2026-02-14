# LangChain 버전 변경 사항

> 📖 **공식 문서**: [04-changelog.md](../../official/04-changelog_ko.md)

이 문서는 LangChain의 주요 버전 변경 사항을 한국어로 요약합니다. LangChain은 빠르게 발전하고 있으며, 각 버전마다 중요한 기능 추가와 API 변경이 있습니다.

---

## 📋 목차

1. [LangChain 1.0 (2025)](#langchain-10-2025)
2. [LangChain 0.3 (2024)](#langchain-03-2024)
3. [LangChain 0.2 (2024)](#langchain-02-2024)
4. [마이그레이션 가이드](#-마이그레이션-가이드)
5. [Breaking Changes 요약](#-breaking-changes-요약)

---

## LangChain 1.0 (2025)

**릴리스 날짜**: 2025년 1월

### 🎉 주요 변경사항

LangChain 1.0은 프로덕션 준비 완료를 선언하는 메이저 릴리스입니다.

#### 1. `create_agent()` API 안정화

**변경 전 (0.x)**:
```python
from langchain.agents import AgentExecutor, create_structured_chat_agent

agent_executor = AgentExecutor(
    agent=create_structured_chat_agent(...),
    tools=tools,
    memory=memory,
    verbose=True
)
```

**변경 후 (1.0)**:
```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,  # 새로운 메모리 시스템
)
```

**주요 개선**:
- 더 간단하고 직관적인 API
- `AgentExecutor` deprecated → `create_agent()`로 통일
- LangGraph 기반으로 내부 재설계

#### 2. LangGraph 기반 재설계

모든 Agent가 이제 LangGraph의 그래프 구조를 기반으로 합니다:

- **유연성**: 커스텀 워크플로우 쉽게 구성
- **가시성**: 실행 흐름을 그래프로 시각화
- **제어**: 각 단계를 세밀하게 제어 가능

#### 3. 통합 메모리 시스템

**Checkpointer 도입**:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)

# 대화 이력이 자동으로 저장됨
result = agent.invoke(
    {"messages": [{"role": "user", "content": "안녕"}]},
    config={"configurable": {"thread_id": "user-123"}}
)
```

**주요 특징**:
- 단기 메모리(대화 이력)와 장기 메모리(Store) 분리
- PostgreSQL, SQLite 등 다양한 백엔드 지원
- Thread 기반 대화 관리

#### 4. 향상된 스트리밍 지원

**새로운 Stream Modes**:

```python
# 1. updates 모드: 각 단계의 업데이트
for chunk in agent.stream(input, stream_mode="updates"):
    print(chunk)

# 2. messages 모드: 메시지만 스트리밍
for chunk in agent.stream(input, stream_mode="messages"):
    print(chunk)

# 3. custom 모드: 커스텀 이벤트
for chunk in agent.stream(input, stream_mode="custom"):
    print(chunk)
```

#### 5. 미들웨어 시스템

**새로운 기능**:

```python
from langgraph.prebuilt import create_agent

def logging_middleware(state, next):
    print(f"Before: {state}")
    result = next(state)
    print(f"After: {result}")
    return result

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[logging_middleware],
)
```

**내장 미들웨어**:
- Summarization: 대화 요약
- Human-in-the-Loop: 사람 승인 필요
- Tool Retry: 도구 실패 시 재시도
- Guardrails: 안전 가드레일

### 🔧 API 변경사항

| 기능 | 0.x | 1.0 | 상태 |
|------|-----|-----|------|
| Agent 생성 | `AgentExecutor` | `create_agent()` | ✅ Stable |
| 메모리 | `ConversationBufferMemory` | `Checkpointer` | ✅ Stable |
| 스트리밍 | `astream()` | `stream(mode=...)` | ✅ Stable |
| 도구 정의 | `@tool` | `@tool` | ✅ 변경 없음 |
| 모델 초기화 | 개별 import | `init_chat_model()` | ✅ 새로 추가 |

### 🚨 Breaking Changes

#### 1. `AgentExecutor` 제거

**이전 코드**:
```python
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

**마이그레이션**:
```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt,
)
```

#### 2. Memory API 변경

**이전 코드**:
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(agent=agent, memory=memory)
```

**마이그레이션**:
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)
```

#### 3. 실행 방식 변경

**이전 코드**:
```python
result = agent_executor.run("질문")
```

**마이그레이션**:
```python
result = agent.invoke({"messages": [{"role": "user", "content": "질문"}]})
answer = result["messages"][-1].content
```

### 📦 새로운 패키지 구조

LangChain 1.0은 모듈화된 패키지 구조를 채택합니다:

```
langchain (메타 패키지)
├── langchain-core (핵심 추상화)
├── langchain-openai (OpenAI 통합)
├── langchain-anthropic (Anthropic 통합)
├── langchain-google-genai (Google 통합)
├── langchain-community (커뮤니티 통합)
└── langgraph (Agent 프레임워크)
```

**설치 예시**:
```bash
# 최소 설치
pip install langchain-core langgraph

# OpenAI 사용
pip install langchain-openai

# 모든 기능
pip install langchain
```

### 🎯 1.0 마이그레이션 체크리스트

- [ ] `AgentExecutor` → `create_agent()` 변경
- [ ] `ConversationBufferMemory` → `Checkpointer` 변경
- [ ] `.run()` → `.invoke()` 변경
- [ ] 메시지 형식 확인 (dict → Message 객체)
- [ ] 스트리밍 코드 업데이트 (`stream_mode` 사용)
- [ ] 테스트 실행 및 검증

---

## LangChain 0.3 (2024)

**릴리스 날짜**: 2024년 9월

### 주요 변경사항

#### 1. LangGraph 통합 시작

0.3부터 LangGraph가 Agent의 기본 엔진으로 채택되기 시작했습니다.

#### 2. Tool Calling 표준화

모든 주요 LLM 프로바이더에서 Tool Calling이 표준화되었습니다:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools(tools)
```

#### 3. Structured Output 지원

```python
from langchain_core.pydantic_v1 import BaseModel

class Person(BaseModel):
    name: str
    age: int

model.with_structured_output(Person)
```

#### 4. 향상된 메시지 시스템

`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` 타입 추가

### Breaking Changes

- `LLMChain` deprecated
- 일부 레거시 Agent 타입 제거
- 메시지 형식 변경

---

## LangChain 0.2 (2024)

**릴리스 날짜**: 2024년 5월

### 주요 변경사항

#### 1. Pydantic v2 지원

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(description="검색어")
    max_results: int = Field(default=10, description="최대 결과 수")
```

#### 2. 비동기 지원 강화

```python
async def main():
    result = await agent.ainvoke({"input": "질문"})
```

#### 3. 런타임 설정

```python
result = agent.invoke(
    input,
    config={
        "callbacks": [handler],
        "tags": ["production"],
        "metadata": {"user_id": "123"}
    }
)
```

---

## 🔄 마이그레이션 가이드

### 0.2 → 0.3

**주요 변경사항**:
1. Tool Calling API 업데이트
2. 메시지 타입 변경
3. `LLMChain` 사용 중단

**마이그레이션 단계**:
```python
# 이전 (0.2)
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(input)

# 이후 (0.3)
from langchain_core.runnables import RunnablePassthrough

chain = prompt | llm
result = chain.invoke(input)
```

### 0.3 → 1.0

**주요 변경사항**:
1. `AgentExecutor` 제거
2. Checkpointer 도입
3. `create_agent()` API

**마이그레이션 단계**:

1. **Agent 생성 코드 변경**
   ```python
   # 이전
   agent_executor = AgentExecutor(agent=agent, tools=tools)

   # 이후
   agent = create_agent(model=model, tools=tools)
   ```

2. **메모리 시스템 변경**
   ```python
   # 이전
   memory = ConversationBufferMemory()

   # 이후
   checkpointer = InMemorySaver()
   ```

3. **실행 방식 변경**
   ```python
   # 이전
   result = agent_executor.run("질문")

   # 이후
   result = agent.invoke({"messages": [{"role": "user", "content": "질문"}]})
   ```

---

## ⚠️ Breaking Changes 요약

### LangChain 1.0

| 변경 사항 | 영향 | 대응 방법 |
|----------|------|----------|
| `AgentExecutor` 제거 | **높음** | `create_agent()` 사용 |
| Memory API 변경 | **높음** | `Checkpointer` 사용 |
| `.run()` 제거 | **중간** | `.invoke()` 사용 |
| 메시지 형식 변경 | **중간** | dict → Message 객체 |
| 스트리밍 API 변경 | **낮음** | `stream_mode` 파라미터 추가 |

### LangChain 0.3

| 변경 사항 | 영향 | 대응 방법 |
|----------|------|----------|
| `LLMChain` deprecated | **중간** | LCEL (파이프) 사용 |
| Tool Calling 변경 | **중간** | `.bind_tools()` 사용 |
| 일부 Agent 타입 제거 | **낮음** | `create_agent()` 사용 |

---

## 📚 추가 리소스

### 공식 문서
- [LangChain Changelog](https://github.com/langchain-ai/langchain/releases)
- [LangGraph Changelog](https://github.com/langchain-ai/langgraph/releases)
- [마이그레이션 가이드](https://docs.langchain.com/oss/python/migrate/langchain-v1)

### 교안 관련 섹션
- [Part 3: 첫 번째 Agent](../part03_first_agent.md) - `create_agent()` API
- [Part 4: 메모리 시스템](../part04_memory.md) - Checkpointer 사용법
- [Part 9: 프로덕션](../part09_production.md) - 스트리밍 및 고급 기능

### 커뮤니티
- [LangChain Discord](https://discord.gg/langchain) - 마이그레이션 질문
- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

---

## 🔍 버전 확인 방법

### Python에서 버전 확인

```python
import langchain
import langgraph

print(f"LangChain: {langchain.__version__}")
print(f"LangGraph: {langgraph.__version__}")
```

### CLI에서 버전 확인

```bash
pip show langchain langgraph
```

---

## ❓ FAQ

<details>
<summary>Q1: 0.x에서 1.0으로 업그레이드해야 하나요?</summary>

**A**: 네, 강력히 권장합니다.

**이유**:
- 더 안정적인 API
- 성능 개선
- 프로덕션 준비 완료
- 향후 기능은 1.0 기반으로 개발됨

**주의**: 마이그레이션 시간을 충분히 확보하세요 (소규모: 1-2일, 대규모: 1-2주)
</details>

<details>
<summary>Q2: 기존 코드가 여전히 작동하나요?</summary>

**A**: 단기적으로는 작동하지만, deprecated 경고가 표시됩니다.

LangChain 팀은 0.x 지원을 점진적으로 축소할 예정이므로, 가능한 빨리 마이그레이션하는 것이 좋습니다.
</details>

<details>
<summary>Q3: 마이그레이션 중 문제가 발생하면?</summary>

**A**: 다음 리소스를 활용하세요:

1. [Troubleshooting 가이드](./troubleshooting.md)
2. [LangChain Discord](https://discord.gg/langchain)
3. [GitHub Issues](https://github.com/langchain-ai/langchain/issues)
4. 교안 예제 코드 참고
</details>

---

*이 문서는 공식 changelog의 요약본입니다. 전체 내용은 [공식 문서](../../official/04-changelog_ko.md)를 참고하세요.*

*마지막 업데이트: 2025-02-05*
*버전: 1.0*
