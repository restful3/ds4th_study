# 문제 해결 가이드

> 🔧 LangChain AI Agent 교안에서 자주 발생하는 문제와 해결 방법

이 문서는 교안을 학습하면서 발생할 수 있는 일반적인 문제들의 해결 방법을 정리합니다.

---

## 📋 목차

1. [설치 관련 문제](#-설치-관련-문제)
2. [API 키 및 인증 문제](#-api-키-및-인증-문제)
3. [Agent 동작 문제](#-agent-동작-문제)
4. [메모리 및 데이터베이스 문제](#-메모리-및-데이터베이스-문제)
5. [성능 및 최적화 문제](#-성능-및-최적화-문제)
6. [도구 및 통합 문제](#-도구-및-통합-문제)
7. [스트리밍 및 비동기 문제](#-스트리밍-및-비동기-문제)
8. [프로덕션 배포 문제](#-프로덕션-배포-문제)

---

## 🔧 설치 관련 문제

### Python 버전 문제

#### 증상
```
SyntaxError: invalid syntax
ModuleNotFoundError: No module named 'langchain'
```

#### 원인
- Python 버전이 3.10 미만
- 시스템에 여러 Python 버전 설치됨

#### 해결 방법

**1. Python 버전 확인**
```bash
python --version
python3 --version
python3.11 --version
```

**2. 올바른 버전 사용**
```bash
# 특정 버전으로 가상환경 생성
python3.11 -m venv .venv

# 또는 pyenv 사용 (권장)
pyenv install 3.11
pyenv local 3.11
```

**3. 가상환경 재생성**
```bash
# 기존 가상환경 삭제
rm -rf .venv

# 새로 생성
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 패키지 설치 실패

#### 증상
```
ERROR: Could not find a version that satisfies the requirement langchain>=0.3.0
ERROR: No matching distribution found for langchain
```

#### 원인
- pip 버전이 오래됨
- 네트워크 문제
- 잘못된 패키지 이름

#### 해결 방법

**1. pip 업그레이드**
```bash
pip install --upgrade pip setuptools wheel
```

**2. 캐시 삭제 후 재설치**
```bash
pip cache purge
pip install langchain --no-cache-dir
```

**3. 특정 버전 설치**
```bash
# 정확한 버전 지정
pip install langchain==0.3.1

# 또는 최소 버전만 지정
pip install "langchain>=0.3.0"
```

**4. 수동 다운로드 (네트워크 문제 시)**
```bash
# PyPI에서 whl 파일 다운로드 후
pip install langchain-0.3.1-py3-none-any.whl
```

---

### 의존성 충돌

#### 증상
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
ERROR: langchain 0.3.1 requires pydantic>=2.0, but you have pydantic 1.10.0
```

#### 원인
- 다른 패키지와 버전 충돌
- 이전 버전의 패키지가 설치되어 있음

#### 해결 방법

**1. 가상환경 재생성 (권장)**
```bash
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. 강제 업그레이드**
```bash
pip install --upgrade --force-reinstall pydantic
```

**3. pip-tools 사용**
```bash
pip install pip-tools
pip-compile requirements.txt
pip-sync requirements.txt
```

---

### uv 설치 또는 사용 문제

#### 증상
```
command not found: uv
```

#### 원인
- uv가 설치되지 않음
- PATH에 uv가 없음

#### 해결 방법

**1. uv 재설치**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 설치 후 셸 재시작
source ~/.bashrc  # 또는 ~/.zshrc
```

**2. PATH 확인**
```bash
echo $PATH | grep .cargo/bin
```

**3. 수동으로 PATH 추가**
```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export PATH="$HOME/.cargo/bin:$PATH"
```

---

## 🔑 API 키 및 인증 문제

### "OPENAI_API_KEY not found"

#### 증상
```python
openai.OpenAIError: The api_key client option must be set
ValidationError: OPENAI_API_KEY is required
```

#### 원인
- 환경변수가 설정되지 않음
- `.env` 파일이 로드되지 않음
- API 키 형식 오류

#### 해결 방법

**1. .env 파일 확인**
```bash
# .env 파일 존재 확인
ls -la src/.env

# 내용 확인
cat src/.env | grep OPENAI_API_KEY
```

**2. 환경변수 수동 설정**
```bash
# 임시로 설정 (현재 세션에만 유효)
export OPENAI_API_KEY="sk-proj-..."

# 확인
echo $OPENAI_API_KEY
```

**3. Python 코드에서 확인**
```python
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경변수 확인
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✅ API 키 설정됨: {api_key[:10]}...")
else:
    print("❌ API 키가 없습니다")
```

**4. API 키 형식 확인**
```python
# OpenAI 키 형식: sk-proj-... 또는 sk-...
# Anthropic 키 형식: sk-ant-...
# Google 키 형식: AI...
```

---

### "Rate limit exceeded"

#### 증상
```
openai.RateLimitError: Rate limit reached for gpt-4o
anthropic.RateLimitError: rate_limit_error
```

#### 원인
- API 사용량 한도 초과
- 너무 빠른 요청 속도

#### 해결 방법

**1. 사용량 확인**
```bash
# OpenAI Platform에서 확인
open https://platform.openai.com/usage
```

**2. 더 저렴한 모델 사용**
```python
# 비용이 높은 모델
model = ChatOpenAI(model="gpt-4o")  # $2.50 / 1M 토큰

# 비용이 낮은 모델
model = ChatOpenAI(model="gpt-4o-mini")  # $0.15 / 1M 토큰
```

**3. 재시도 로직 추가**
```python
from langchain_openai import ChatOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def call_llm(prompt):
    model = ChatOpenAI(model="gpt-4o-mini")
    return model.invoke(prompt)
```

**4. 요청 속도 제한**
```python
import time

for prompt in prompts:
    result = model.invoke(prompt)
    time.sleep(1)  # 1초 대기
```

---

### "Invalid API key" 또는 "Unauthorized"

#### 증상
```
openai.AuthenticationError: Incorrect API key provided
anthropic.AuthenticationError: invalid x-api-key
```

#### 원인
- API 키가 잘못됨
- API 키가 만료됨
- 키에 공백이나 특수문자 포함

#### 해결 방법

**1. API 키 재확인**
```bash
# 공백 제거
OPENAI_API_KEY=$(echo $OPENAI_API_KEY | tr -d ' ')
export OPENAI_API_KEY
```

**2. 새 API 키 발급**
- [OpenAI Platform](https://platform.openai.com/api-keys)
- [Anthropic Console](https://console.anthropic.com/settings/keys)

**3. .env 파일 형식 확인**
```env
# 올바른 형식 (따옴표 없음)
OPENAI_API_KEY=sk-proj-xxxxx

# 잘못된 형식
OPENAI_API_KEY="sk-proj-xxxxx"  # 따옴표 포함 ❌
OPENAI_API_KEY = sk-proj-xxxxx  # 공백 포함 ❌
```

---

## 🤖 Agent 동작 문제

### Agent가 도구를 호출하지 않음

#### 증상
- Agent가 도구를 사용하지 않고 직접 답변함
- "도구를 사용해야 하는데 사용하지 않음"

#### 원인
- 도구 설명(docstring)이 불명확
- 질문이 모호함
- 모델이 도구 호출을 지원하지 않음

#### 해결 방법

**1. 도구 설명 개선**
```python
# 나쁜 예
@tool
def search(query):
    """검색"""
    return search_api(query)

# 좋은 예
@tool
def search(query: str) -> str:
    """
    웹에서 정보를 검색합니다.

    Args:
        query: 검색할 키워드 또는 질문

    Returns:
        검색 결과 요약

    Examples:
        - "파이썬이란?" -> Python 프로그래밍 언어 정보 반환
        - "오늘 날씨" -> 현재 날씨 정보 반환
    """
    return search_api(query)
```

**2. 질문을 명확하게 작성**
```python
# 모호한 질문
"파이썬 알려줘"

# 명확한 질문
"파이썬 공식 문서에서 파이썬의 정의를 검색해줘"
```

**3. Tool Calling을 지원하는 모델 사용**
```python
# Tool Calling을 지원하지 않는 모델
model = ChatOpenAI(model="gpt-3.5-turbo-instruct")  # ❌

# Tool Calling을 지원하는 모델
model = ChatOpenAI(model="gpt-4o-mini")  # ✅
model = ChatAnthropic(model="claude-sonnet-4-5-20250929")  # ✅
```

**4. System Prompt 개선**
```python
system_prompt = """
당신은 도움이 되는 AI 어시스턴트입니다.
사용 가능한 도구를 적극적으로 활용하여 정확한 답변을 제공하세요.
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)
```

---

### "Tool not found" 또는 "Invalid tool name"

#### 증상
```
ValueError: Tool 'search_web' not found in tool list
KeyError: 'search_web'
```

#### 원인
- 도구 이름 오타
- 도구가 Agent에 등록되지 않음

#### 해결 방법

**1. 도구 이름 확인**
```python
# 등록된 도구 확인
for tool in tools:
    print(f"도구 이름: {tool.name}")

# Agent에 등록된 도구 확인
print(agent.tools)
```

**2. 도구를 명시적으로 등록**
```python
@tool
def search_web(query: str) -> str:
    """웹 검색"""
    return "검색 결과"

# 도구 리스트에 추가
tools = [search_web]  # 함수 자체를 전달 (tool 데코레이터 적용됨)

agent = create_agent(model=model, tools=tools)
```

---

### Agent 응답이 너무 느림

#### 증상
- Agent가 응답하는데 10초 이상 걸림
- 타임아웃 발생

#### 원인
- 느린 모델 사용 (GPT-4)
- 도구 실행이 느림
- 네트워크 지연

#### 해결 방법

**1. 더 빠른 모델 사용**
```python
# 느림
model = ChatOpenAI(model="gpt-4o")

# 빠름
model = ChatOpenAI(model="gpt-4o-mini")
model = ChatAnthropic(model="claude-haiku-4-5-20251001")
```

**2. 스트리밍 모드 활성화**
```python
# 전체 응답 대기
result = agent.invoke(input)

# 스트리밍 (즉시 응답 시작)
for chunk in agent.stream(input):
    print(chunk, end="", flush=True)
```

**3. 타임아웃 설정 증가**
```python
model = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=60,  # 60초로 증가
)
```

**4. 도구 성능 최적화**
```python
import time

@tool
def slow_tool(query: str) -> str:
    """느린 도구"""
    time.sleep(5)  # ❌ 5초 대기
    return "결과"

@tool
def fast_tool(query: str) -> str:
    """빠른 도구"""
    # 캐싱, 비동기 처리 등 최적화
    return "결과"
```

---

## 💾 메모리 및 데이터베이스 문제

### PostgreSQL 연결 실패

#### 증상
```
sqlalchemy.exc.OperationalError: could not connect to server: Connection refused
psycopg2.OperationalError: connection to server at "localhost", port 5432 failed
```

#### 원인
- PostgreSQL이 실행되지 않음
- 연결 문자열이 잘못됨
- 방화벽 또는 네트워크 문제

#### 해결 방법

**1. PostgreSQL 실행 확인**
```bash
# macOS (Homebrew)
brew services list | grep postgresql

# Ubuntu/Debian
sudo systemctl status postgresql

# Docker
docker ps | grep postgres
```

**2. PostgreSQL 시작**
```bash
# Homebrew
brew services start postgresql@15

# systemd
sudo systemctl start postgresql

# Docker
docker start langchain-postgres
```

**3. 연결 문자열 확인**
```python
# .env 파일
DATABASE_URL=postgresql://postgres:password@localhost:5432/langchain

# 형식: postgresql://[사용자]:[비밀번호]@[호스트]:[포트]/[데이터베이스]
```

**4. 수동 연결 테스트**
```bash
# psql 클라이언트로 연결
psql -h localhost -U postgres -d langchain

# 또는 Docker
docker exec -it langchain-postgres psql -U postgres -d langchain
```

---

### "Checkpointer" 관련 오류

#### 증상
```
TypeError: Checkpointer() missing 1 required positional argument: 'conn_string'
AttributeError: 'InMemorySaver' object has no attribute 'get_state'
```

#### 원인
- Checkpointer 초기화 오류
- 잘못된 Checkpointer 타입 사용

#### 해결 방법

**1. InMemorySaver 사용 (간단한 테스트)**
```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)
```

**2. PostgresSaver 사용 (프로덕션)**
```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg2 import pool

# 연결 풀 생성
connection_pool = pool.SimpleConnectionPool(
    1, 10,  # 최소, 최대 연결 수
    "postgresql://postgres:password@localhost:5432/langchain"
)

checkpointer = PostgresSaver(connection_pool)
agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)
```

---

### 메모리가 저장되지 않음

#### 증상
- Agent가 이전 대화를 기억하지 못함
- 매번 새로운 대화로 시작

#### 원인
- Thread ID를 지정하지 않음
- Checkpointer가 설정되지 않음

#### 해결 방법

**1. Thread ID 지정**
```python
# Thread ID 없이 실행 (매번 새로운 대화)
result = agent.invoke({"messages": [{"role": "user", "content": "안녕"}]})

# Thread ID 지정 (대화 이력 유지)
config = {"configurable": {"thread_id": "user-123"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "안녕"}]},
    config=config
)
```

**2. Checkpointer 확인**
```python
# Checkpointer 없음 (메모리 저장 안 됨)
agent = create_agent(model=model, tools=tools)

# Checkpointer 있음 (메모리 저장됨)
checkpointer = InMemorySaver()
agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)
```

---

## 🚀 성능 및 최적화 문제

### 메모리 사용량이 너무 높음

#### 증상
- 프로세스가 수 GB 메모리 사용
- OOM (Out of Memory) 에러

#### 원인
- 대화 이력이 너무 길어짐
- 임베딩 벡터가 메모리에 쌓임

#### 해결 방법

**1. 요약 미들웨어 사용 (메시지 자동 트리밍)**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages", 10),  # 10개 이상이면 요약
            keep=("messages", 5),      # 최근 5개는 유지
        ),
    ],
)
```

---

## 🔗 도구 및 통합 문제

### Vector Store 연결 실패

#### 증상
```
chromadb.errors.ConnectionError: Could not connect to Chroma
pinecone.exceptions.PineconeException: Invalid API key
```

#### 해결 방법

**1. Chroma (로컬)**
```python
from langchain_chroma import Chroma

# 영구 저장소 설정
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)
```

**2. Pinecone (클라우드)**
```python
from langchain_pinecone import PineconeVectorStore
import os

# API 키 확인
assert os.getenv("PINECONE_API_KEY"), "PINECONE_API_KEY가 없습니다"

vectorstore = PineconeVectorStore(
    index_name="langchain-index",
    embedding=embeddings,
)
```

---

## 📡 스트리밍 및 비동기 문제

### 스트리밍이 작동하지 않음

#### 증상
- `agent.stream()`이 전체 응답을 한 번에 반환
- 실시간 업데이트가 없음

#### 해결 방법

**1. Stream 모드 지정**
```python
# updates 모드로 스트리밍
for chunk in agent.stream(input, stream_mode="updates"):
    print(chunk)

# messages 모드로 스트리밍
for chunk in agent.stream(input, stream_mode="messages"):
    print(chunk)
```

**2. 모델 스트리밍 지원 확인**
```python
# 스트리밍 지원 모델
model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
```

---

## 🌐 프로덕션 배포 문제

### LangSmith 연결 안 됨

#### 증상
```
LangSmithConnectionError: Could not connect to LangSmith
```

#### 해결 방법

**1. API 키 확인**
```bash
# .env 파일
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-project
```

**2. 수동으로 트레이싱 활성화**
```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
```

---

## 🔍 디버깅 팁

### 상세 로깅 활성화

```python
import logging

# LangChain 로깅
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain")
logger.setLevel(logging.DEBUG)
```

### Agent 실행 흐름 확인

```python
from langchain.globals import set_debug

set_debug(True)  # 디버그 모드 활성화

result = agent.invoke(input)
```

---

## 📞 추가 도움말

문제가 해결되지 않으면:

1. **공식 문서**: [LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)
2. **Discord**: [LangChain Discord](https://discord.gg/langchain)
3. **GitHub Issues**: [LangChain GitHub](https://github.com/langchain-ai/langchain/issues)
4. **Stack Overflow**: [langchain 태그](https://stackoverflow.com/questions/tagged/langchain)

**교안 관련 문제는**:
- [changelog.md](./changelog.md) - 버전 변경 사항
- [glossary.md](./glossary.md) - 용어 설명
- [resources.md](./resources.md) - 추가 학습 자료

---

*마지막 업데이트: 2025-02-18*
*버전: 1.1*
