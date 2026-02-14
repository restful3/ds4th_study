# 용어 사전

> 📖 LangChain AI Agent 교안에서 사용되는 주요 용어 정리

이 문서는 LangChain과 AI Agent 관련 기술 용어를 한국어로 설명합니다. 알파벳순으로 정리되어 있습니다.

---

## A

### Agent (에이전트)
LLM을 추론 엔진으로 사용하여 도구(Tool)를 호출하고 작업을 수행하는 자율적인 시스템입니다.

**특징**:
- 사용자 목표를 이해하고 실행 계획 수립
- 필요한 도구를 선택하고 호출
- 결과를 종합하여 최종 답변 생성

**예시**:
```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[search_tool, calculator_tool],
)
```

**관련 파트**: Part 3 (첫 번째 Agent)

---

### AgentExecutor (에이전트 실행기) [Deprecated]
LangChain 0.x에서 Agent를 실행하는 클래스입니다. LangChain 1.0에서는 `create_agent()`로 대체되었습니다.

**마이그레이션**:
```python
# 이전 (0.x)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 이후 (1.0)
agent = create_agent(model=model, tools=tools)
```

---

### Agentic RAG
Agent가 검색 전략을 스스로 결정하는 고급 RAG 시스템입니다.

**일반 RAG**:
- 고정된 검색 쿼리 사용
- 단일 검색 수행

**Agentic RAG**:
- Agent가 검색 쿼리를 동적으로 생성
- 여러 번 검색하여 정보 수집
- 검색 결과를 분석하고 추가 검색 결정

**관련 파트**: Part 8.3 (Agentic RAG)

---

### AIMessage
LLM이 생성한 메시지를 나타내는 클래스입니다.

```python
from langchain_core.messages import AIMessage

message = AIMessage(content="안녕하세요!")
```

**관련 파트**: Part 2.2 (Messages)

---

## B

### Batch Processing (배치 처리)
여러 입력을 한 번에 처리하는 방식입니다.

```python
inputs = ["질문1", "질문2", "질문3"]
results = model.batch(inputs)
```

**장점**:
- 효율적인 리소스 사용
- 처리 속도 향상

---

## C

### Callback (콜백)
Agent 실행 중 특정 이벤트 발생 시 호출되는 함수입니다.

```python
from langchain.callbacks import StdOutCallbackHandler

agent.invoke(input, config={"callbacks": [StdOutCallbackHandler()]})
```

**용도**:
- 로깅
- 트레이싱
- 성능 모니터링

**관련 파트**: Part 10.2 (Tracing)

---

### Checkpointer (체크포인터)
Agent의 상태(메모리)를 저장하고 복원하는 시스템입니다.

**종류**:
- `InMemorySaver`: 메모리에만 저장 (테스트용)
- `PostgresSaver`: PostgreSQL 데이터베이스에 저장
- `SQLiteSaver`: SQLite 파일에 저장

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)
```

**관련 파트**: Part 4 (메모리 시스템)

---

### Chat Models (채팅 모델)
대화형 인터페이스를 제공하는 LLM입니다.

**예시**:
- OpenAI: GPT-4o, GPT-4o-mini
- Anthropic: Claude 3.5 Sonnet
- Google: Gemini 1.5 Pro

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

**관련 파트**: Part 2.1 (Chat Models)

---

### Completion Models (완성 모델)
텍스트 완성을 수행하는 LLM입니다. (Deprecated)

**Note**: LangChain 1.0에서는 Chat Models 사용을 권장합니다.

---

### Context (컨텍스트)
Agent나 도구가 실행 중 접근할 수 있는 정보입니다.

**종류**:
- Model Context: LLM에 전달되는 프롬프트
- Tool Context: 도구가 접근 가능한 런타임 정보
- User Context: 사용자 세션 정보

**관련 파트**: Part 6 (컨텍스트 엔지니어링)

---

### `create_agent()`
LangChain 1.0에서 Agent를 생성하는 메인 API입니다.

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="당신은 유용한 AI 어시스턴트입니다.",
    checkpointer=checkpointer,
)
```

**관련 파트**: Part 3.1 (create_agent API)

---

## D

### Dependency Injection (의존성 주입)
도구나 미들웨어에 런타임 정보를 전달하는 패턴입니다.

```python
@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """사용자 정보 조회"""
    user_id = runtime.context.get("user_id")
    return f"User ID: {user_id}"
```

**관련 파트**: Part 6.4 (Runtime & Context)

---

## E

### Embeddings (임베딩)
텍스트를 벡터(숫자 배열)로 변환한 것입니다.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("안녕하세요")
# [0.123, -0.456, 0.789, ...]
```

**용도**:
- 의미 유사도 계산
- Vector Store에 저장
- RAG 시스템

**관련 파트**: Part 8.2 (Vector Store)

---

## F

### Function Calling → Tool Calling 참조

---

## G

### Guardrails (가드레일)
Agent의 출력을 검증하고 제한하는 안전 장치입니다.

**예시**:
- 유해 콘텐츠 필터링
- PII(개인정보) 탐지 및 제거
- 출력 형식 검증

```python
def content_filter(state):
    # 유해 콘텐츠 필터링 로직
    if is_harmful(state["messages"][-1].content):
        raise ValueError("유해 콘텐츠 감지됨")
    return state
```

**관련 파트**: Part 5.4 (Guardrails)

---

## H

### Handoffs (핸드오프)
한 Agent가 다른 Agent에게 제어를 넘기는 패턴입니다.

```python
# Agent A -> Agent B로 제어 전달
result = agent_a.invoke(input)
if needs_specialist:
    result = agent_b.invoke(result)
```

**사용 사례**:
- 티어 1 상담원 → 티어 2 전문가
- 일반 Agent → 전문 Agent

**관련 파트**: Part 7.3 (Handoffs)

---

### HITL (Human-in-the-Loop)
Agent 실행 중 사람의 승인이나 입력을 받는 패턴입니다.

```python
# Agent가 중요한 작업 전에 승인 요청
result = agent.invoke(input, interrupt_before=["execute_payment"])

# 사용자 승인 후 계속 실행
approved_result = agent.invoke(None, config={"configurable": {"thread_id": thread_id}})
```

**용도**:
- 중요한 결정 승인
- 위험한 작업 검토
- 사용자 선택 수집

**관련 파트**: Part 9.4-9.5 (Human-in-the-Loop)

---

### HumanMessage
사용자가 보낸 메시지를 나타내는 클래스입니다.

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(content="안녕하세요!")
```

**관련 파트**: Part 2.2 (Messages)

---

## I

### `init_chat_model()`
프로바이더에 관계없이 Chat Model을 초기화하는 통합 API입니다.

```python
from langchain.chat_models import init_chat_model

# 환경변수 기반 자동 선택
model = init_chat_model()

# 또는 명시적 지정
model = init_chat_model(model="gpt-4o-mini", provider="openai")
```

**관련 파트**: Part 2.1 (Chat Models)

---

### InMemorySaver
메모리에만 상태를 저장하는 Checkpointer입니다.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

**특징**:
- 빠른 속도
- 프로세스 종료 시 데이터 손실
- 테스트 및 개발용

**관련 파트**: Part 4.1 (기본 메모리)

---

## L

### LangChain
LLM 애플리케이션 개발을 위한 오픈소스 프레임워크입니다.

**주요 구성 요소**:
- Models: LLM 통합
- Agents: 자율 시스템
- Chains: 작업 파이프라인
- Memory: 대화 이력 관리

**공식 사이트**: https://python.langchain.com

---

### LangGraph
LangChain의 기반이 되는 그래프 기반 워크플로우 프레임워크입니다.

```python
from langgraph.graph import StateGraph

graph = StateGraph(state_schema)
graph.add_node("node1", function1)
graph.add_node("node2", function2)
graph.add_edge("node1", "node2")
```

**특징**:
- 유연한 워크플로우 정의
- 순환 그래프 지원 (Agent 루프)
- 체크포인팅 내장

**관련 파트**: Part 7.6 (Custom Workflow)

---

### LangSmith
LangChain Agent의 트레이싱, 디버깅, 평가 플랫폼입니다.

**주요 기능**:
- 실행 흐름 시각화
- 성능 모니터링
- A/B 테스팅
- 데이터셋 관리

```python
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "lsv2_pt_..."
```

**관련 파트**: Part 10.1-10.2 (LangSmith)

---

### LCEL (LangChain Expression Language)
LangChain의 파이프라인 구성 언어입니다.

```python
# 파이프(|)로 체인 구성
chain = prompt | model | output_parser

result = chain.invoke({"input": "질문"})
```

**특징**:
- 간결한 문법
- 자동 스트리밍 지원
- 비동기 실행

---

### LLM (Large Language Model)
대규모 언어 모델을 의미합니다.

**예시**:
- GPT-4, GPT-4o (OpenAI)
- Claude 3.5 (Anthropic)
- Gemini 1.5 (Google)

---

## M

### MCP (Model Context Protocol)
외부 도구와 데이터 소스를 LLM에 연결하는 표준 프로토콜입니다.

**특징**:
- 표준화된 인터페이스
- 다양한 데이터 소스 통합
- 플러그인 방식

```python
from langchain_mcp import MCPClient

mcp_client = MCPClient(server_url="http://localhost:8000")
agent = create_agent(model=model, tools=[mcp_client])
```

**관련 파트**: Part 8.4-8.6 (MCP)

---

### Messages (메시지)
LLM과 주고받는 대화의 기본 단위입니다.

**타입**:
- `SystemMessage`: 시스템 지시사항
- `HumanMessage`: 사용자 메시지
- `AIMessage`: AI 응답
- `ToolMessage`: 도구 실행 결과

```python
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="안녕하세요"),
    AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
]
```

**관련 파트**: Part 2.2 (Messages)

---

### Middleware (미들웨어)
Agent 실행 파이프라인에 끼워넣을 수 있는 커스텀 로직입니다.

**종류**:
- `before_model`: 모델 호출 전
- `after_model`: 모델 호출 후
- `wrap_model_call`: 모델 호출 감싸기
- `wrap_tool_call`: 도구 호출 감싸기

```python
def logging_middleware(state):
    print(f"Before: {state}")
    result = yield  # 모델 실행
    print(f"After: {result}")
    return result
```

**관련 파트**: Part 5 (미들웨어)

---

### Multimodal (멀티모달)
텍스트뿐만 아니라 이미지, 오디오 등 여러 형태의 입력을 처리하는 기능입니다.

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "이 이미지는 무엇인가요?"},
        {"type": "image_url", "image_url": {"url": "https://..."}}
    ]
)
```

**지원 모델**:
- GPT-4o, GPT-4o-mini (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- Gemini 1.5 Pro (Google)

**관련 파트**: Part 2.1 (Chat Models)

---

## P

### PostgresSaver
PostgreSQL 데이터베이스에 상태를 저장하는 Checkpointer입니다.

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg2 import pool

connection_pool = pool.SimpleConnectionPool(1, 10, "postgresql://...")
checkpointer = PostgresSaver(connection_pool)
```

**특징**:
- 영구 저장
- 확장 가능
- 프로덕션 준비

**관련 파트**: Part 4.2 (PostgreSQL Memory)

---

### Prompt (프롬프트)
LLM에게 전달하는 지시사항입니다.

**종류**:
- System Prompt: Agent의 역할과 규칙
- User Prompt: 사용자 질문
- Few-shot Prompt: 예시 포함

```python
system_prompt = """
당신은 유용한 AI 어시스턴트입니다.
사용자의 질문에 정확하고 친절하게 답변하세요.
"""
```

---

### Pydantic
Python의 데이터 검증 라이브러리입니다. LangChain에서 스키마 정의에 사용됩니다.

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(description="검색어")
    max_results: int = Field(default=10, description="최대 결과 수")
```

**용도**:
- Tool 입력 검증
- Structured Output 정의
- 설정 관리

---

## R

### RAG (Retrieval Augmented Generation)
외부 지식 베이스를 검색하여 LLM에 컨텍스트를 제공하는 기법입니다.

**프로세스**:
1. 사용자 질문을 임베딩으로 변환
2. Vector Store에서 관련 문서 검색
3. 검색된 문서를 LLM에 제공
4. LLM이 답변 생성

```python
from langchain_community.vectorstores import Chroma

# 문서를 Vector Store에 저장
vectorstore = Chroma.from_documents(documents, embeddings)

# 검색 및 생성
retriever = vectorstore.as_retriever()
chain = retriever | llm
```

**관련 파트**: Part 8.1-8.3 (RAG)

---

### ReAct
Reasoning (추론) + Acting (행동)을 결합한 Agent 패턴입니다.

**프로세스**:
1. **Thought**: 무엇을 해야 할지 생각
2. **Action**: 도구 호출
3. **Observation**: 도구 결과 관찰
4. 1-3 반복

```
Thought: 서울 날씨를 알기 위해 날씨 API를 호출해야겠다
Action: get_weather("서울")
Observation: 서울의 날씨는 맑음, 22도
Thought: 이제 사용자에게 답변할 수 있다
Answer: 서울의 날씨는 맑고 기온은 22도입니다.
```

**논문**: https://arxiv.org/abs/2210.03629

**관련 파트**: Part 3.3 (ReAct Pattern)

---

### Router (라우터)
입력을 분석하여 적절한 Agent나 도구로 라우팅하는 패턴입니다.

```python
def router(input):
    if "날씨" in input:
        return weather_agent
    elif "계산" in input:
        return calculator_agent
    else:
        return general_agent
```

**관련 파트**: Part 7.5 (Router Pattern)

---

### Runnable
LangChain의 실행 가능한 구성 요소를 나타내는 인터페이스입니다.

**메서드**:
- `invoke()`: 단일 실행
- `batch()`: 배치 실행
- `stream()`: 스트리밍 실행
- `ainvoke()`: 비동기 실행

---

## S

### Skills (스킬)
온디맨드로 로드되는 Agent의 능력입니다.

```python
skills = {
    "weather": weather_tool,
    "calculator": calculator_tool,
}

# 필요할 때만 로드
if needs_weather:
    agent.load_skill("weather")
```

**관련 파트**: Part 7.4 (Skills Pattern)

---

### State (상태)
Agent의 현재 실행 상태를 나타냅니다.

**포함 정보**:
- 메시지 이력
- 변수
- 도구 실행 결과

```python
state = {
    "messages": [...],
    "user_id": "123",
    "iteration": 1,
}
```

---

### Store (스토어)
Agent의 장기 메모리를 저장하는 시스템입니다.

**용도**:
- 사용자 선호도 저장
- 이전 대화 요약 저장
- 학습된 정보 저장

```python
from langgraph.store import InMemoryStore

store = InMemoryStore()
store.put("user-123", "preferences", {"language": "ko"})
```

**관련 파트**: Part 4.4 (Long-term Memory)

---

### Streaming (스트리밍)
Agent 응답을 실시간으로 받는 방식입니다.

```python
# 전체 응답 대기 (기본)
result = agent.invoke(input)

# 스트리밍
for chunk in agent.stream(input, stream_mode="messages"):
    print(chunk, end="", flush=True)
```

**장점**:
- 즉각적인 피드백
- 사용자 경험 개선
- 긴 응답도 빠르게 시작

**관련 파트**: Part 9.1-9.3 (Streaming)

---

### Structured Output
LLM 출력을 정해진 구조(스키마)로 받는 기능입니다.

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

model_with_structure = model.with_structured_output(Person)
result = model_with_structure.invoke("김철수는 30살입니다")
# Person(name="김철수", age=30)
```

**관련 파트**: Part 9.6 (Structured Output)

---

### Subagents (서브에이전트)
메인 Agent의 도구로 사용되는 Agent입니다.

```python
# Subagent 정의
weather_agent = create_agent(model=model, tools=[weather_tool])

# 메인 Agent의 도구로 사용
@tool
def get_weather_analysis(city: str) -> str:
    """날씨 분석을 수행합니다"""
    return weather_agent.invoke({"messages": [{"role": "user", "content": f"{city} 날씨 분석"}]})

main_agent = create_agent(model=model, tools=[get_weather_analysis])
```

**관련 파트**: Part 7.2 (Subagents)

---

### SystemMessage
시스템 지시사항을 나타내는 메시지 클래스입니다.

```python
from langchain_core.messages import SystemMessage

message = SystemMessage(content="당신은 유용한 AI 어시스턴트입니다.")
```

**관련 파트**: Part 2.2 (Messages)

---

## T

### Thread (스레드)
독립적인 대화 세션을 나타냅니다.

```python
# 사용자 A의 대화
config_a = {"configurable": {"thread_id": "user-a"}}
result_a = agent.invoke(input, config=config_a)

# 사용자 B의 대화 (독립적)
config_b = {"configurable": {"thread_id": "user-b"}}
result_b = agent.invoke(input, config=config_b)
```

---

### Token (토큰)
LLM이 처리하는 텍스트의 기본 단위입니다.

**예시**:
- "안녕하세요" → 약 3-4 토큰
- 1 토큰 ≈ 0.75 단어 (영어 기준)
- 1 토큰 ≈ 1-2 글자 (한국어 기준)

**중요성**:
- API 비용 계산 기준
- 모델 입력 제한 (예: 128K 토큰)

---

### Tool (도구)
Agent가 호출할 수 있는 함수 또는 API입니다.

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 알려줍니다"""
    return f"{city}의 날씨는 맑습니다"
```

**특징**:
- 명확한 docstring 필요 (LLM이 읽음)
- 타입 힌트 권장
- 에러 핸들링 포함

**관련 파트**: Part 2.3 (Tools)

---

### Tool Calling (도구 호출)
LLM이 함수를 호출하는 기능입니다. (이전: Function Calling)

```python
model_with_tools = model.bind_tools([get_weather])
result = model_with_tools.invoke("서울 날씨는?")
# LLM이 get_weather("서울") 호출 결정
```

**지원 모델**:
- GPT-4o, GPT-4o-mini
- Claude 3.5 Sonnet
- Gemini 1.5 Pro

**관련 파트**: Part 2.4 (Tool Calling)

---

### ToolMessage
도구 실행 결과를 나타내는 메시지 클래스입니다.

```python
from langchain_core.messages import ToolMessage

message = ToolMessage(
    content="서울의 날씨는 맑습니다",
    tool_call_id="call_123",
)
```

**관련 파트**: Part 2.2 (Messages)

---

### ToolRuntime
도구에서 Agent의 런타임 컨텍스트에 접근하는 인터페이스입니다.

```python
from langgraph.prebuilt import ToolRuntime

@tool
def get_user_data(runtime: ToolRuntime) -> str:
    """현재 사용자 정보를 조회합니다"""
    user_id = runtime.context.get("user_id")
    return f"User: {user_id}"
```

**관련 파트**: Part 6.5 (Tool Runtime)

---

### Tracing (트레이싱)
Agent 실행 흐름을 기록하고 시각화하는 기능입니다.

```python
import os

os.environ["LANGSMITH_TRACING"] = "true"

# 실행 시 자동으로 LangSmith에 기록됨
result = agent.invoke(input)
```

**관련 파트**: Part 10.2 (Tracing)

---

## V

### Vector Store (벡터 스토어)
임베딩 벡터를 저장하고 검색하는 데이터베이스입니다.

**종류**:
- Chroma: 로컬, 오픈소스
- Pinecone: 관리형, 클라우드
- Weaviate: 오픈소스, 확장 가능

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
)
```

**관련 파트**: Part 8.2 (Vector Store)

---

## W

### Workflow (워크플로우)
Agent의 실행 흐름을 정의하는 그래프입니다.

```python
from langgraph.graph import StateGraph

graph = StateGraph(state_schema)
graph.add_node("start", start_node)
graph.add_node("process", process_node)
graph.add_edge("start", "process")
```

**관련 파트**: Part 7.6 (Custom Workflow)

---

## 📚 추가 리소스

### 공식 문서
- [LangChain 용어집](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph 문서](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangSmith 문서](https://docs.langchain.com/langsmith)

### 교안 관련
- [Changelog](./changelog.md) - 버전 변경 사항
- [Troubleshooting](./troubleshooting.md) - 문제 해결
- [Resources](./resources.md) - 추가 학습 자료

---

## ❓ FAQ

<details>
<summary>Q: Agent와 LLM의 차이는?</summary>

**A**:
- **LLM**: 텍스트 입력을 받아 텍스트 출력을 생성
- **Agent**: LLM을 사용하여 도구를 호출하고 작업을 수행하는 시스템

LLM은 "뇌"이고, Agent는 "뇌 + 손발"입니다.
</details>

<details>
<summary>Q: RAG와 Fine-tuning의 차이는?</summary>

**A**:
- **RAG**: 외부 지식을 검색하여 프롬프트에 추가 (모델 수정 없음)
- **Fine-tuning**: 모델 자체를 새로운 데이터로 재학습

**장단점**:
- RAG: 빠르고 저렴, 지식 업데이트 쉬움
- Fine-tuning: 더 깊은 학습, 비용 높음
</details>

<details>
<summary>Q: Checkpointer와 Store의 차이는?</summary>

**A**:
- **Checkpointer**: 단기 메모리 (대화 이력, 최근 상태)
- **Store**: 장기 메모리 (사용자 선호도, 학습된 정보)

예: Checkpointer는 "오늘 나눈 대화", Store는 "사용자가 채식주의자라는 정보"
</details>

---

*더 많은 용어는 [공식 문서](https://docs.langchain.com/oss/python/langchain/overview)를 참고하세요.*

*마지막 업데이트: 2025-02-05*
*버전: 1.0*
