# LangChain 교안 환경 설정 가이드

> 📖 LangChain AI Agent 마스터 교안을 시작하기 위한 완벽한 환경 설정 가이드

이 가이드는 LangChain 1.0 기반의 AI Agent 교안을 학습하기 위한 개발 환경을 단계별로 설정하는 방법을 안내합니다.

---

## 📋 목차

1. [시스템 요구사항](#-시스템-요구사항)
2. [Python 설치](#-python-설치)
3. [프로젝트 클론 및 설정](#-프로젝트-클론-및-설정)
4. [의존성 설치](#-의존성-설치)
5. [환경변수 설정](#-환경변수-설정)
6. [LLM 프로바이더 설정](#-llm-프로바이더-설정)
7. [설치 확인](#-설치-확인)
8. [데이터베이스 설정 (선택)](#-데이터베이스-설정-선택-사항)
9. [문제 해결](#-문제-해결)

---

## 📋 시스템 요구사항

### 필수 요구사항

- **Python**: 3.10 이상 (3.11 권장)
- **운영체제**:
  - macOS 10.15 (Catalina) 이상
  - Linux (Ubuntu 20.04+, Debian 11+)
  - Windows 10/11 (WSL2 권장)
- **메모리**: 최소 8GB RAM (16GB 권장)
- **디스크**: 최소 5GB 여유 공간
- **인터넷**: API 호출을 위한 안정적인 인터넷 연결

### 권장 도구

- **코드 에디터**: VS Code, PyCharm, 또는 Cursor
- **터미널**: bash, zsh, 또는 fish
- **Git**: 버전 관리 및 협업
- **Docker**: 데이터베이스 실습 (Part 4 이후)

---

## 🐍 Python 설치

### Python 버전 확인

먼저 현재 시스템에 Python이 설치되어 있는지 확인합니다:

```bash
python --version
# 또는
python3 --version
```

**결과 예시**:
```
Python 3.11.5
```

Python 3.10 이상이 설치되어 있다면 다음 단계로 진행하세요.

### Python 설치 방법

#### macOS

**Option A: Homebrew 사용 (권장)**

```bash
# Homebrew 설치 (아직 설치되지 않은 경우)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치
brew install python@3.11
```

**Option B: 공식 설치 파일**

[python.org](https://www.python.org/downloads/)에서 macOS용 설치 파일을 다운로드하여 설치합니다.

#### Linux (Ubuntu/Debian)

```bash
# 시스템 패키지 업데이트
sudo apt update

# Python 3.11 설치
sudo apt install python3.11 python3.11-venv python3-pip

# 기본 python3를 3.11로 설정 (선택 사항)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
```

#### Windows

**Option A: WSL2 사용 (강력 권장)**

1. WSL2 설치:
   ```powershell
   wsl --install
   ```

2. Ubuntu 설치 후, 위의 Linux 설치 방법을 따릅니다.

**Option B: Windows 네이티브**

[python.org](https://www.python.org/downloads/)에서 Windows용 설치 파일을 다운로드하여 설치합니다.

⚠️ **주의**: "Add Python to PATH" 옵션을 반드시 선택하세요.

---

## 📦 프로젝트 클론 및 설정

### 프로젝트 다운로드

```bash
# GitHub에서 클론 (실제 리포지토리 URL로 변경)
git clone https://github.com/your-org/langchain-curriculum.git
cd langchain-curriculum

# 또는 ZIP 파일로 다운로드한 경우
unzip langchain-curriculum.zip
cd langchain-curriculum
```

### 디렉토리 구조 확인

```bash
ls -la
```

**예상 출력**:
```
langchain-curriculum/
├── README.md
├── CURRICULUM_PLAN.md
├── SETUP_GUIDE.md (이 파일)
├── docs/
├── src/
├── datasets/
├── assets/
├── projects/
└── official/
```

---

## 📥 의존성 설치

LangChain 교안은 두 가지 패키지 관리 도구를 지원합니다:

1. **uv** (권장 - 빠르고 현대적)
2. **pip** (전통적 방식)

### Option A: uv 사용 (권장)

uv는 Rust로 작성된 빠른 Python 패키지 매니저입니다.

#### uv 설치

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 의존성 설치

```bash
# src 디렉토리로 이동
cd src

# uv로 가상환경 생성 및 패키지 설치
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv pip install -r requirements.txt

# 선택 사항: 개발 도구 포함
uv pip install -e ".[dev]"
```

### Option B: pip 사용

#### 가상환경 생성

```bash
# src 디렉토리로 이동
cd src

# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# macOS/Linux:
source .venv/bin/activate

# Windows (CMD):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

#### 의존성 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt
```

### 설치 확인

```bash
# LangChain 설치 확인
python -c "import langchain; print(langchain.__version__)"

# 기대 출력: 0.3.x 이상
```

---

## 🔑 환경변수 설정

### .env 파일 생성

```bash
# src 디렉토리에서
cp .env.example .env
```

### API 키 설정

`.env` 파일을 텍스트 에디터로 열어 실제 API 키를 입력합니다:

```bash
# macOS/Linux
nano .env
# 또는
code .env  # VS Code가 설치된 경우
```

### 최소 필수 설정

최소한 **하나의 LLM 프로바이더** API 키가 필요합니다:

```env
# OpenAI 사용 시 (추천: 초보자에게 가장 쉬움)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 또는 Anthropic (Claude) 사용 시
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# LangSmith (선택 사항, Part 10에서 사용)
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LANGSMITH_TRACING=false
```

### 환경변수 로드 확인

```bash
# Python에서 환경변수 확인
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OPENAI_API_KEY:', os.getenv('OPENAI_API_KEY')[:10] + '...')"
```

---

## 🤖 LLM 프로바이더 설정

### OpenAI 설정 (권장 - 초보자용)

1. **API 키 발급**
   - [OpenAI Platform](https://platform.openai.com/api-keys) 방문
   - "Create new secret key" 클릭
   - 키를 복사하여 `.env` 파일의 `OPENAI_API_KEY`에 붙여넣기

2. **설치 확인**
   ```bash
   python -c "from langchain_openai import ChatOpenAI; model = ChatOpenAI(model='gpt-4o-mini'); print(model.invoke('안녕하세요!').content)"
   ```

3. **권장 모델**
   - **학습/개발**: `gpt-4o-mini` (저렴, 빠름)
   - **프로덕션**: `gpt-4o` (정확, 안정적)
   - **고급**: `gpt-4-turbo` (최고 성능)

### Anthropic (Claude) 설정

1. **API 키 발급**
   - [Anthropic Console](https://console.anthropic.com/settings/keys) 방문
   - "Create Key" 클릭
   - 키를 복사하여 `.env` 파일의 `ANTHROPIC_API_KEY`에 붙여넣기

2. **설치 확인**
   ```bash
   python -c "from langchain_anthropic import ChatAnthropic; model = ChatAnthropic(model='claude-3-5-sonnet-20241022'); print(model.invoke('안녕하세요!').content)"
   ```

3. **권장 모델**
   - **학습/개발**: `claude-3-5-haiku-20241022` (빠름)
   - **프로덕션**: `claude-3-5-sonnet-20241022` (균형)
   - **고급**: `claude-3-opus-20240229` (최고 품질)

### Google (Gemini) 설정

1. **API 키 발급**
   - [Google AI Studio](https://aistudio.google.com/app/apikey) 방문
   - "Create API Key" 클릭
   - 키를 복사하여 `.env` 파일의 `GOOGLE_API_KEY`에 붙여넣기

2. **설치 확인**
   ```bash
   python -c "from langchain_google_genai import ChatGoogleGenerativeAI; model = ChatGoogleGenerativeAI(model='gemini-1.5-flash'); print(model.invoke('안녕하세요!').content)"
   ```

3. **권장 모델**
   - **학습/개발**: `gemini-1.5-flash` (무료 할당량)
   - **프로덕션**: `gemini-1.5-pro` (고성능)

### 비용 가이드

| 프로바이더 | 모델 | 입력 (1M 토큰) | 출력 (1M 토큰) | 용도 |
|----------|------|--------------|--------------|------|
| OpenAI | gpt-4o-mini | $0.15 | $0.60 | 학습 |
| OpenAI | gpt-4o | $2.50 | $10.00 | 프로덕션 |
| Anthropic | claude-3-5-haiku | $0.80 | $4.00 | 학습 |
| Anthropic | claude-3-5-sonnet | $3.00 | $15.00 | 프로덕션 |
| Google | gemini-1.5-flash | 무료* | 무료* | 학습 |

*Google은 일일 무료 할당량 제공 (2024년 기준)

📖 **상세 정보**: [공식 문서 - 07-models.md](official/07-models.md)

---

## ✅ 설치 확인

### 자동 확인 스크립트

교안에 포함된 환경 확인 스크립트를 실행합니다:

```bash
# src 디렉토리에서
python part01_introduction/02_environment_check.py
```

**예상 출력**:
```
================================================================================
🔍 LangChain 환경 확인
================================================================================

✅ Python 버전: 3.11.5
✅ LangChain 버전: 0.3.1
✅ LangGraph 버전: 0.2.3
✅ OPENAI_API_KEY: 설정됨 (sk-proj-xxxx...)
✅ OpenAI API 연결: 성공

================================================================================
🎉 환경 설정이 완료되었습니다!
================================================================================
```

### 수동 확인

각 구성 요소를 개별적으로 확인할 수도 있습니다:

```bash
# 1. Python 버전
python --version

# 2. LangChain 임포트
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"

# 3. 환경변수 확인
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ API 키가 설정되었습니다' if os.getenv('OPENAI_API_KEY') else '❌ API 키가 없습니다')"

# 4. 간단한 Agent 테스트
python -c "
from langchain.agents import create_agent, tool
from langchain_openai import ChatOpenAI

@tool
def greet(name: str) -> str:
    '''사용자에게 인사합니다'''
    return f'안녕하세요, {name}님!'

model = ChatOpenAI(model='gpt-4o-mini')
agent = create_agent(model=model, tools=[greet])
result = agent.invoke({'messages': [{'role': 'user', 'content': '김철수에게 인사해줘'}]})
print('✅ Agent가 정상 작동합니다!')
"
```

---

## 🗄️ 데이터베이스 설정 (선택 사항)

Part 4 (메모리 시스템)부터는 데이터베이스가 필요합니다. 미리 설정하거나 Part 4에 도달했을 때 설정할 수 있습니다.

### PostgreSQL 설정

#### Docker 사용 (권장)

```bash
# PostgreSQL 컨테이너 실행
docker run -d \
  --name langchain-postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=langchain \
  -p 5432:5432 \
  postgres:15

# 연결 확인
docker exec -it langchain-postgres psql -U postgres -c "SELECT version();"
```

#### 네이티브 설치

**macOS (Homebrew)**:
```bash
brew install postgresql@15
brew services start postgresql@15
createdb langchain
```

**Ubuntu/Debian**:
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb langchain
```

### .env 파일 업데이트

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/langchain
```

### 연결 테스트

```bash
python -c "
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    result = conn.execute('SELECT version()')
    print('✅ PostgreSQL 연결 성공!')
    print(result.fetchone()[0])
"
```

---

## 🐛 문제 해결

### 일반적인 문제

#### 1. "ModuleNotFoundError: No module named 'langchain'"

**원인**: LangChain이 설치되지 않았거나 가상환경이 활성화되지 않음

**해결**:
```bash
# 가상환경 활성화 확인
which python  # .venv 경로가 표시되어야 함

# 재설치
pip install langchain langchain-core
```

#### 2. "OPENAI_API_KEY not found"

**원인**: 환경변수가 설정되지 않음

**해결**:
```bash
# .env 파일 확인
cat src/.env

# 환경변수 수동 설정 (임시)
export OPENAI_API_KEY="sk-proj-..."

# Python에서 확인
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

#### 3. "Rate limit exceeded" 또는 "Insufficient quota"

**원인**: API 사용량 한도 초과

**해결**:
- OpenAI Platform에서 사용량 확인
- 결제 방법 등록
- 더 저렴한 모델 사용 (`gpt-4o-mini`)

#### 4. Python 버전 충돌

**원인**: 시스템에 여러 Python 버전 설치됨

**해결**:
```bash
# pyenv 사용 (권장)
curl https://pyenv.run | bash
pyenv install 3.11
pyenv local 3.11

# 또는 절대 경로 사용
/usr/bin/python3.11 -m venv .venv
```

#### 5. Windows에서 "Activate.ps1을 로드할 수 없습니다"

**원인**: PowerShell 실행 정책

**해결**:
```powershell
# 관리자 권한으로 PowerShell 실행
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 또는 CMD 사용
.venv\Scripts\activate.bat
```

### 추가 도움말

문제가 해결되지 않으면:

1. **문제 해결 가이드**: [docs/appendix/troubleshooting.md](docs/appendix/troubleshooting.md)
2. **용어 사전**: [docs/appendix/glossary.md](docs/appendix/glossary.md)
3. **LangChain Discord**: https://discord.gg/langchain
4. **GitHub Issues**: 프로젝트 이슈 트래커

---

## 🚀 다음 단계

환경 설정이 완료되었습니다! 이제 학습을 시작할 준비가 되었습니다.

### 추천 학습 경로

1. **Part 1: AI Agent의 이해** ([docs/part01_introduction.md](docs/part01_introduction.md))
   - LangChain과 Agent의 기본 개념 학습
   - 첫 번째 "Hello, World!" Agent 만들기

2. **Part 2: LangChain 기초** ([docs/part02_fundamentals.md](docs/part02_fundamentals.md))
   - Chat Models, Messages, Tools 이해
   - 기본 구성 요소 실습

3. **Part 3: 첫 번째 Agent** ([docs/part03_first_agent.md](docs/part03_first_agent.md))
   - `create_agent()` API 마스터
   - 날씨 Agent 프로젝트

### 빠른 시작 예제

간단한 Agent를 만들어 보세요:

```python
# quick_start.py
from langchain.agents import create_agent, tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 날씨를 알려줍니다"""
    # 실제로는 API를 호출하지만, 여기서는 더미 데이터
    return f"{city}의 날씨는 맑고 기온은 22도입니다."

model = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="당신은 친절한 날씨 도우미입니다."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "서울 날씨 어때?"}]
})

print(result["messages"][-1].content)
```

실행:
```bash
python quick_start.py
```

---

## 📚 참고 자료

- **공식 문서**: [LangChain Python 문서](https://python.langchain.com/docs/)
- **API 레퍼런스**: [LangChain API](https://api.python.langchain.com/en/latest/)
- **교안 구조**: [CURRICULUM_PLAN.md](CURRICULUM_PLAN.md)
- **추가 학습 자료**: [docs/appendix/resources.md](docs/appendix/resources.md)

---

**환경 설정을 완료하신 것을 축하합니다! 🎉**

질문이나 문제가 있다면 [troubleshooting.md](docs/appendix/troubleshooting.md)를 참고하거나 커뮤니티에 문의하세요.

*마지막 업데이트: 2025-02-05*
*버전: 1.0.0*
*기반: LangChain 1.0*
