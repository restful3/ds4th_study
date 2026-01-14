# Deep Agents CLI

Deep Agents로 빌드하기 위한 대화형 명령줄 인터페이스

Deep Agents와 상호작용하기 위한 터미널 인터페이스로, 전체 기능에 직접 접근하여 에이전트를 빌드하고 테스트할 수 있습니다:

- **파일 작업** - 에이전트가 코드와 문서를 관리하고 수정할 수 있는 도구로 프로젝트의 파일을 읽고, 쓰고, 편집합니다.
- **셸 명령 실행** - 테스트 실행, 프로젝트 빌드, 의존성 관리, 버전 관리 시스템과의 상호작용을 위해 셸 명령을 실행합니다.
- **웹 검색** - 최신 정보와 문서를 위해 웹 검색 (Tavily API 키 필요).
- **HTTP 요청** - 데이터 가져오기 및 통합 작업을 위해 API와 외부 서비스에 HTTP 요청을 보냅니다.
- **작업 계획 및 추적** - 복잡한 작업을 개별 단계로 분해하고 내장된 todo 시스템을 통해 진행 상황을 추적합니다.
- **메모리 저장 및 검색** - 세션 간에 정보를 저장하고 검색하여, 에이전트가 프로젝트 규칙과 학습된 패턴을 기억할 수 있게 합니다.
- **Human-in-the-loop** - 민감한 도구 작업에 대해 사람의 승인을 요구합니다.

[데모 비디오 보기](https://youtu.be/IrnacLa9PJc?si=3yUnPbxnm2yaqVQb)

## 빠른 시작

### 1. API 키 설정

환경 변수로 내보내기:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

또는 프로젝트 루트에 `.env` 파일 생성:

```text
ANTHROPIC_API_KEY=your-api-key
```

### 2. CLI 실행

```bash
uvx deepagents-cli
```

### 3. 에이전트에게 작업 제공

```text
> Create a Python script that prints "Hello, World!"
```

에이전트는 파일을 수정하기 전에 diff와 함께 변경 사항을 제안하여 승인을 받습니다.

<details>
<summary>트레이싱 구성 (선택사항)</summary>

LangSmith에서 에이전트를 추적하려면 다음 환경 변수를 설정합니다:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
```

에이전트 트레이스에 대한 커스텀 프로젝트 이름을 선택적으로 설정할 수 있습니다:

```bash
export DEEPAGENTS_LANGSMITH_PROJECT="my-agent-project"
```

에이전트가 실행하는 사용자 코드의 경우, 별도의 프로젝트를 설정할 수 있습니다:

```bash
export LANGSMITH_PROJECT="my-user-code-project"
```

</details>

<details>
<summary>추가 설치 및 구성 옵션</summary>

**pip로 설치**

`uvx`를 사용하지 않으려면 패키지를 직접 설치할 수 있습니다:

```bash
pip install deepagents-cli
```

**OpenAI 구성 (Anthropic 대안)**

```bash
export OPENAI_API_KEY="your-key"
```

**웹 검색 구성**

```bash
export TAVILY_API_KEY="your-key"
```

이것들을 `.env` 파일에 넣을 수도 있습니다.

</details>

## 구성

### 명령줄 옵션

| 옵션 | 설명 |
| :--- | :--- |
| `--agent NAME` | 사용할 에이전트 프로필 이름 |
| `--auto-approve` | 승인 프롬프트 건너뛰기 (또는 `Ctrl+T`로 토글) |
| `--sandbox TYPE` | 원격 샌드박스 사용 (`modal`, `daytona`, 또는 `runloop`) |
| `--sandbox-id ID` | 연결할 기존 샌드박스의 ID |
| `--sandbox-setup PATH` | 샌드박스 설정 스크립트 경로 |

### CLI 명령

| 명령 | 설명 |
| :--- | :--- |
| `deepagents list` | 사용 가능한 에이전트 프로필 나열 |
| `deepagents help` | 도움말 정보 표시 |
| `deepagents reset --agent NAME` | 에이전트 상태 초기화 |
| `deepagents reset --agent NAME --target SOURCE` | 에이전트 상태를 특정 소스로 초기화 |

## 대화형 모드

### 슬래시 명령

- `/tokens` - 토큰 사용량 표시
- `/clear` - 대화 기록 지우기
- `/exit` 또는 `/quit` - CLI 종료

### Bash 명령

명령 앞에 `!`를 붙여 셸에서 직접 실행:

```bash
!git status
!npm test
!ls -la
```

### 키보드 단축키

| 단축키 | 동작 |
| :--- | :--- |
| `Enter` | 메시지 전송 (단일 줄) |
| `Option+Enter` / `Alt+Enter` | 줄바꿈 삽입 |
| `Ctrl+E` | 외부 편집기 열기 |
| `Ctrl+T` | 자동 승인 모드 토글 |
| `Ctrl+C` | 현재 작업 취소 |
| `Ctrl+D` | CLI 종료 |

## 메모리로 프로젝트 규칙 설정

CLI는 `~/.deepagents/AGENT_NAME/memories/`에 프로젝트별 메모리를 저장합니다. 에이전트는 이 메모리를 세 가지 방식으로 사용합니다:

1. **리서치**: 작업 시작 전 관련 컨텍스트를 메모리에서 검색
2. **응답**: 실행 중 불확실할 때 메모리 확인
3. **학습**: 향후 세션을 위해 새 정보를 자동 저장

메모리 구조 예시:

```text
~/.deepagents/backend-dev/memories/
├── api-conventions.md
├── database-schema.md
└── deployment-process.md
```

규칙을 설정하려면 에이전트에게 그냥 말하면 됩니다:

```bash
uvx deepagents-cli --agent backend-dev
> Our API uses snake_case and includes created_at/updated_at timestamps
```

이후 작업에서는 이 규칙을 자동으로 따릅니다:

```text
> Create a /users endpoint
# 프롬프트 없이 규칙 적용
```

## 원격 샌드박스 사용

원격 샌드박스는 여러 이점을 가진 격리된 코드 실행 환경을 제공합니다:

- **안전성**: 잠재적으로 해로운 코드 실행으로부터 로컬 머신 보호
- **깨끗한 환경**: 로컬 설정 없이 특정 의존성이나 OS 구성 사용
- **병렬 실행**: 격리된 환경에서 여러 에이전트를 동시에 실행
- **장기 실행 작업**: 머신을 차단하지 않고 시간이 오래 걸리는 작업 실행
- **재현성**: 팀 간 일관된 실행 환경 보장

### 원격 샌드박스를 사용하려면 다음 단계를 따르세요:

1. **샌드박스 제공업체 구성** ([Runloop](https://www.runloop.ai/), [Daytona](https://www.daytona.io/), 또는 [Modal](https://modal.com/)):

```bash
# Runloop
export RUNLOOP_API_KEY="your-key"

# Daytona
export DAYTONA_API_KEY="your-key"

# Modal
modal setup
```

2. **샌드박스와 함께 CLI 실행:**

```bash
uvx deepagents-cli --sandbox runloop --sandbox-setup ./setup.sh
```

에이전트는 로컬에서 실행되지만 모든 코드 작업은 원격 샌드박스에서 실행됩니다. 선택적 설정 스크립트는 환경 변수 구성, 저장소 복제, 의존성 준비를 수행할 수 있습니다.

3. **(선택사항) 샌드박스 환경을 구성하기 위한 `setup.sh` 파일 생성:**

```bash
#!/bin/bash
set -e

# GitHub 토큰을 사용하여 저장소 복제
git clone https://x-access-token:${GITHUB_TOKEN}@github.com/username/repo.git $HOME/workspace
cd $HOME/workspace

# 환경 변수를 영구적으로 만들기
cat >> ~/.bashrc <<'EOF'
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
cd $HOME/workspace
EOF

source ~/.bashrc
```

설정 스크립트가 접근할 수 있도록 로컬 `.env` 파일에 시크릿을 저장합니다.

---

<p align="center">
  <a href="09-middleware_ko.md">← 이전: 미들웨어</a> • <a href="README.md">목차</a>
</p>
