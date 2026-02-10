# LangChain 문서 → 마크다운 변환 작업 계획서

## 📊 현재 진행 상황

### ✅ 완료된 문서 (34개) - 전체 완료!
| 번호 | 파일명 | 원본 URL |
|------|--------|----------|
| 01 | overview.md | /oss/python/langchain/overview |
| 02 | install.md | /oss/python/langchain/install |
| 03 | quickstart.md | /oss/python/langchain/quickstart |
| 04 | changelog.md | /oss/python/langchain/changelog |
| 05 | philosophy.md | /oss/python/langchain/philosophy |
| 06 | agents.md | /oss/python/langchain/agents |
| 07 | models.md | /oss/python/langchain/models |
| 08 | messages.md | /oss/python/langchain/messages |
| 09 | tools.md | /oss/python/langchain/tools |
| 10 | short-term-memory.md | /oss/python/langchain/short-term-memory |
| 11 | streaming-overview.md | /oss/python/langchain/streaming |
| 12 | streaming-frontend.md | /oss/python/langchain/streaming/frontend |
| 13 | structured-output.md | /oss/python/langchain/structured-output |
| 14 | middleware-overview.md | /oss/python/langchain/middleware/overview |
| 15 | built-in-middleware.md | /oss/python/langchain/middleware/built-in |
| 16 | custom-middleware.md | /oss/python/langchain/middleware/custom |
| 17 | guardrails.md | /oss/python/langchain/guardrails |
| 18 | runtime.md | /oss/python/langchain/runtime |
| 19 | context-engineering.md | /oss/python/langchain/context-engineering |
| 20 | model-context-protocol.md | /oss/python/langchain/mcp |
| 21 | human-in-the-loop.md | /oss/python/langchain/human-in-the-loop |
| 22 | multi-agent.md (Overview) | /oss/python/langchain/multi-agent |
| 23 | subagents.md | /oss/python/langchain/multi-agent/subagents |
| 24 | handoffs.md | /oss/python/langchain/multi-agent/handoffs |
| 25 | skills.md | /oss/python/langchain/multi-agent/skills |
| 26 | router.md | /oss/python/langchain/multi-agent/router |
| 27 | custom-workflow.md | /oss/python/langchain/multi-agent/custom-workflow |
| 28 | retrieval.md | /oss/python/langchain/retrieval |
| 29 | long-term-memory.md | /oss/python/langchain/long-term-memory |
| 30 | langsmith-studio.md | /oss/python/langchain/studio |
| 31 | test.md | /oss/python/langchain/test |
| 32 | agent-chat-ui.md | /oss/python/langchain/ui |
| 33 | deployment.md | /oss/python/langchain/deploy |
| 34 | observability.md | /oss/python/langchain/observability |

#### Middleware
- [x] 14-middleware-overview.md ✅
- [x] 15-built-in-middleware.md ✅
- [x] 16-custom-middleware.md ✅

#### Advanced usage
- [x] 17-guardrails.md ✅
- [x] 18-runtime.md ✅
- [x] 19-context-engineering.md ✅
- [x] 20-model-context-protocol.md ✅
- [x] 21-human-in-the-loop.md ✅
- [x] 22-multi-agent.md (Overview) ✅
  - [x] 23-subagents.md ✅
  - [x] 24-handoffs.md ✅
  - [x] 25-skills.md ✅
  - [x] 26-router.md ✅
  - [x] 27-custom-workflow.md ✅
- [x] 28-retrieval.md ✅
- [x] 29-long-term-memory.md ✅

#### Agent development
- [x] 30-langsmith-studio.md ✅
- [x] 31-test.md ✅
- [x] 32-agent-chat-ui.md ✅

#### Deploy with LangSmith
- [x] 33-deployment.md ✅
- [x] 34-observability.md ✅

---

## 🎯 변환 원칙

### 1. 텍스트 처리
- **절대 요약 금지**: 원본 텍스트 100% 그대로 유지
- **누락 금지**: 모든 문장, 단어 포함
- **링크 보존**: 모든 내부/외부 링크 그대로 유지
  - 내부 링크: `/oss/python/langchain/...` 형식 유지
  - 외부 링크: 전체 URL 유지

### 2. 이미지 처리
```
다운로드 위치: ./images/
임베딩 방식: ![설명](./images/파일명.확장자)
```
- 이미지 파일명은 의미있는 이름으로 저장
- avif, png, jpg, svg 등 원본 포맷 유지

### 3. 코드 블록
```markdown
\`\`\`python
# 코드 그대로 복사
\`\`\`

\`\`\`text
# 출력 결과 그대로 복사
\`\`\`
```

### 4. 차트/다이어그램 → Mermaid
원본 차트를 정확히 분석하여 Mermaid로 재현:
```markdown
\`\`\`mermaid
flowchart TD
    input([input])
    model{model}
    ...
\`\`\`
```
- 노드 형태, 연결선, 레이블 정확히 일치
- 화살표 방향 정확히 구현

### 5. 탭(Tab) 구현
마크다운에서 탭 불가 → **연속 섹션**으로 대체:
```markdown
#### Python
\`\`\`python
# Python 코드
\`\`\`

#### JavaScript
\`\`\`javascript
// JavaScript 코드
\`\`\`
```

### 6. 펼침(Collapsible) 기능
```markdown
<details>
<summary>제목 (클릭하여 펼치기)</summary>

펼쳤을 때 보이는 전체 내용
- 코드 블록 포함
- 이미지 포함
- 모든 내용 누락 없이

</details>
```

### 7. 정보 박스 / 경고
```markdown
> 일반 인용/노트

> [!TIP]
> 팁 내용

> [!INFO]
> 정보 내용

> [!WARNING]
> 경고 내용
```

### 8. 테이블
```markdown
| 컬럼1 | 컬럼2 | 컬럼3 |
|-------|-------|-------|
| 값1   | 값2   | 값3   |
```
복잡한 테이블은 HTML `<table>` 사용 가능

---

## 🔧 변환 작업 프로세스

### Phase 1: 페이지 분석
1. Chrome으로 해당 페이지 접속
2. 전체 스크린샷 캡처 (스크롤 필요시 여러 장)
3. 페이지 구조 파악:
   - 헤딩 구조 (h1, h2, h3...)
   - 탭 존재 여부
   - 펼침 섹션 존재 여부
   - 차트/다이어그램 존재 여부
   - 이미지 존재 여부

### Phase 2: 콘텐츠 추출

> ⚠️ **핵심 교훈 (2024년 작업에서 발견된 문제)**
>
> `get_page_text`, `read_page`, "Copy Page" 버튼 등 **자동 텍스트 추출은 절대 신뢰하지 말 것!**
>
> **발견된 문제점:**
> - 탭(Tab) 내용: 현재 선택된 탭만 추출됨, 다른 탭 내용은 누락
> - 코드 블록: `result = agent.invoke(...)` 같은 실행/출력 부분이 잘림
> - 펼침 섹션: 접힌 상태로 추출되어 내용 누락
>
> **올바른 방법:**
> 1. 각 탭을 **직접 클릭** → **스크린샷 캡쳐** → **시각적으로 확인 후 수동 입력**
> 2. 코드 블록은 **전체가 보일 때까지 스크롤** → **끝까지 확인**
> 3. 자동 추출 결과를 그대로 쓰지 말고, **원본 화면과 대조 필수**

1. ~~`get_page_text` 또는 `read_page`로 텍스트 추출~~ ❌ 위험!
2. **각 탭 클릭 → 스크린샷 → 시각적 확인 후 수동 입력** ✅
3. **코드 블록 전체 스크롤하여 끝까지 확인** (특히 result, print, 출력 주석)
4. **펼침 섹션 클릭하여 펼친 후 스크린샷** → 수동 입력
5. 자동 추출은 참고용으로만, 최종 확인은 시각적으로

### Phase 3: 특수 요소 처리
1. 이미지 다운로드 → `images/` 폴더
2. 차트 분석 → Mermaid 코드 작성
3. 링크 URL 확인 및 보존

### Phase 4: 마크다운 작성
1. 기존 파일 스타일 참고하여 작성
2. 원본과 비교 검증
3. 링크 동작 확인

### Phase 5: 검증
1. 마크다운 렌더링 확인
2. 원본 페이지와 1:1 비교
3. 누락 사항 체크리스트:
   - [ ] 모든 텍스트 포함?
   - [ ] 모든 코드 블록 포함?
   - [ ] 모든 이미지 포함?
   - [ ] 모든 링크 동작?
   - [ ] 펼침 섹션 내용 완전?
   - [ ] 탭 내용 모두 포함?

---

## 📁 폴더 구조

```
langchain/
├── 01-overview.md
├── 02-install.md
├── ...
├── images/
│   ├── deepagents-langsmith.avif
│   ├── summary.avif
│   └── ... (새 이미지들)
├── CONVERSION_PLAN.md (이 파일)
└── 명령어.txt
```

---

## ⚠️ 주의사항

1. **AI의 요약 본능 억제**
   - "이 내용은 중요하지 않아서 생략" ❌
   - "비슷한 내용이라 하나로 합침" ❌
   - 모든 내용 100% 포함 ✅

2. **차트 정확도**
   - 노드 개수 정확히 일치
   - 연결선 방향 정확히 일치
   - 레이블 텍스트 정확히 일치

3. **코드 블록 무결성**
   - 들여쓰기 그대로 유지
   - 주석 포함
   - 출력 예시도 포함
   - ⚠️ **`result = agent.invoke(...)` 같은 실행 코드 누락 주의!**

4. **검증 필수**
   - 변환 후 반드시 원본과 비교
   - 글자 수 대략적으로 비슷한지 확인

5. **🚨 자동 추출 함정 (신규 추가)**

   | 자동 추출 방식 | 문제점 |
   |---------------|--------|
   | `get_page_text` | 탭 내용 일부만 추출, 코드 잘림 |
   | `read_page` | 동적 콘텐츠 누락 |
   | Copy Page 버튼 | 시각적 콘텐츠와 다른 내용 반환 |

   **해결책**: 반드시 **클릭 → 스크린샷 → 시각적 확인** 후 수동 입력

6. **🎨 시각적 구분 요소 처리 (2026-02 추가)**

   | 원본 요소 | 마크다운 처리 |
   |-----------|---------------|
   | 비디오 가이드 링크 박스 | `>` 인용 블록으로 감싸기 |
   | 정보/팁 박스 | `> [!TIP]`, `> [!INFO]` 등 사용 |
   | 펼침 섹션 (▸ 화살표) | `<details><summary>제목</summary>` 태그 |
   | 단순 링크 목록 | 테이블 ❌ → 불릿 리스트 ✅ (원본 형태 유지) |

   **핵심**: 원본의 시각적 형태를 그대로 따라가기. 임의로 테이블/리스트 변환 금지!

7. **📦 펼침 섹션 내 탭 주의**

   펼침 섹션 안에 탭(Decorator/Class 등)이 있을 수 있음
   - 펼침 섹션 클릭 → 탭 각각 클릭 → 모든 내용 확인 필수
   - 탭은 `#### 탭이름` + 코드 블록으로 연속 구현

8. **🖼️ 이미지 다운로드 및 삽입 (2026-02-05 추가)**

   **문제**: CDN에서 호스팅된 이미지를 직접 다운로드할 때 프록시/네트워크 제한으로 실패할 수 있음

   **해결책**:
   1. 브라우저에서 이미지 클릭 → 확대된 뷰 열기
   2. `gif_creator` 도구로 녹화 시작 → 스크롤 등 액션 수행 → 녹화 중지
   3. `export` 액션으로 `download: true` 옵션과 함께 파일 저장
   4. 저장된 파일을 `images/` 폴더로 이동

   **이미지 URL 추출 방법**:
   ```javascript
   // JavaScript로 이미지 경로 추출
   const img = document.querySelector('img[alt="이미지 설명"]');
   const url = new URL(img.src);
   console.log(url.origin + url.pathname);
   ```

   **마크다운에서 이미지 참조**:
   ```markdown
   ![이미지 설명](images/파일명.png)
   ```

9. **🔀 Mermaid 차트 실수 방지 (2026-02-05 추가)**

   **문제**: 차트를 텍스트 설명이나 기억에 의존해서 작성하면 연결 방향, 노드 형태가 틀림

   | 흔한 실수 | 올바른 처리 |
   |-----------|-------------|
   | 단방향으로 추정 | 원본 스크린샷에서 **양방향 화살표(↔)** 확인 필수 |
   | 모든 노드를 사각형으로 | **다이아몬드{...}** (조건/분기), **둥근괄호([...])** (시작/끝) 구분 |
   | 출력 노드 위치 추정 | 원본에서 **User response가 어디서 나오는지** 정확히 확인 |
   | 중간 노드 생략 | task, research 같은 **중간 라벨 노드** 누락 금지 |

   **해결책**:
   - 차트가 있는 페이지는 **반드시 스크린샷 캡처** 후 시각적으로 확인
   - 노드 형태: `([...])` = 시작/끝, `[...]` = 일반, `{...}` = 조건/분기
   - 화살표: `-->` = 단방향, `<-->` = 양방향
   - **한 페이지에 여러 차트가 있으면 각각 스크롤하여 모두 확인!**

   **⚠️ 양방향 화살표 vs 별도 두 화살표 구분**:
   ```
   # 하나의 양방향 화살표 (A ↔ B)
   A <--> B

   # 별도의 두 화살표 (A → B, B → A 각각 존재)
   A --> B
   B --> A
   ```
   원본에서 화살표가 **하나로 연결**되어 있는지, **두 개의 별도 화살표**인지 꼼꼼히 확인!

   **예시 - Single dispatch tool 차트**:
   ```
   ❌ 잘못된 버전 (양방향으로 단순화):
   Task <-->|research| ResearchAgent

   ✅ 올바른 버전 (별도 두 화살표):
   Task -->|research| ResearchAgent
   ResearchAgent --> Task
   ```

   **노드 형태 정리**:
   | Mermaid 문법 | 형태 | 용도 |
   |-------------|------|------|
   | `[텍스트]` | 사각형 | 일반 프로세스 |
   | `([텍스트])` | 라운드/스타디움 | 시작/끝, 입출력 |
   | `{텍스트}` | 다이아몬드 | 조건/분기 |
   | `[(텍스트)]` | 실린더 | 데이터베이스/저장소 |
   | `((텍스트))` | 원 | 에이전틱 스텝 |

   **차트 방향**:
   - `flowchart LR`: 가로 (왼쪽→오른쪽) - 원본이 가로일 때
   - `flowchart TD`: 세로 (위→아래) - 페이지 스크롤에 더 적합
   - 사용자 요청에 따라 세로(TD)로 통일 가능

---

## 🚀 다음 작업

**🎉 모든 문서 변환 완료!**

34개 문서 전체 변환이 완료되었습니다.

---

## 📝 작업 기록

### 2026-02-04: 17-guardrails.md 완료

**포함된 요소:**
- 미들웨어 흐름 다이어그램 → Mermaid flowchart로 변환
- PII detection 코드 블록 (PIIMiddleware 사용 예제)
- 펼침 섹션: "Built-in PII types and configuration" (테이블 2개 포함)
- Human-in-the-loop 코드 블록 (HumanInTheLoopMiddleware 예제)
- Custom guardrails 섹션 - Before/After agent guardrails
  - 각각 Class syntax / Decorator syntax 탭 → `####` 섹션으로 변환
- Combine multiple guardrails 코드 블록 (4개 레이어 예제)
- Additional resources 링크 목록

**배운점:**
1. 탭이 있는 코드 블록은 각 탭을 클릭하여 모든 버전의 코드 확인 필수
2. 펼침 섹션 내 테이블은 `<details>` 태그 안에 마크다운 테이블로 작성
3. 다이어그램의 점선 화살표(`-.->`)는 비동기/조건부 흐름을 나타냄
4. **Mermaid 중첩 박스**: 박스 안에 박스가 있는 구조는 `subgraph`로 구현
   ```mermaid
   subgraph wrap_tool_call[wrap_tool_call]
       tools([tools])
   end
   ```
5. **카드형 레이아웃**: 나란히 배치된 카드 UI는 2열 테이블로 변환
   ```markdown
   | 제목1 | 제목2 |
   |-------|-------|
   | 내용1 | 내용2 |
   ```

### 2026-02-04: 18-runtime.md 완료

**포함된 요소:**
- Overview: Runtime 객체 설명 (Context, Store, Stream writer)
- TIP 박스: dependency injection 설명
- Access 섹션: context_schema 사용법 코드 블록
- Inside tools 섹션: ToolRuntime 파라미터 사용법 코드 블록
- Inside middleware 섹션: Runtime/ModelRequest 파라미터 사용법 코드 블록
  - dynamic_prompt, before_model, after_model 데코레이터 예제

**특이사항:**
- 탭/펼침 섹션 없음 - 단순 구조의 페이지
- 번호 매김 목록 (1, 2, 3) 사용
- 굵은 글씨 링크 (tools, middleware, long-term memory 등) 보존

### 2026-02-04: 19-context-engineering.md 완료

**포함된 요소:**
- Overview: 에이전트 실패 원인 설명
- The agent loop: 간단한 다이어그램 (request → model → tools → result)
- What you can control: 3x3 테이블 + 2열 카드 (Transient/Persistent context)
- Data sources: 4열 테이블
- How it works: 미들웨어 설명
- Model context: 5개 카드 (System Prompt, Messages, Tools, Model, Response Format)
  - System Prompt 탭: State, Store, Runtime Context (3개 코드 블록)
  - Messages 탭: State, Store, Runtime Context (3개 코드 블록)
  - Tools > Defining tools: 코드 블록
  - Tools > Selecting tools 탭: State, Store, Runtime Context (3개 코드 블록)
  - Model 탭: State, Store, Runtime Context (3개 코드 블록)
  - Response format > Defining formats: 코드 블록
  - Response format > Selecting formats 탭: State, Store, Runtime Context (3개 코드 블록)
- Tool context
  - Reads 탭: State, Store, Runtime Context (3개 코드 블록)
  - Writes 탭: State, Store (2개 코드 블록)
- Life-cycle context: 미들웨어 흐름 다이어그램 (subgraph 사용)
  - Example: Summarization 코드 블록
- Best practices: 6개 항목 번호 목록
- Related resources: 5개 링크

**특이사항:**
- 매우 긴 페이지 (가장 긴 페이지 중 하나)
- 다수의 탭 구조 (각 섹션에 State/Store/Runtime Context 탭)
- `get_page_text`가 모든 탭 내용을 포함함 (수동 클릭 불필요)
- 2개의 Mermaid 다이어그램 포함
- TIP/INFO 박스 여러 개 포함

### 2026-02-04: 20-model-context-protocol.md 완료

**포함된 요소:**
- Quickstart: langchain-mcp-adapters 설치 (pip/uv 탭)
- MultiServerMCPClient 사용 예제 코드 블록
- INFO 박스: stateless by default 설명
- Custom servers: FastMCP 라이브러리 사용
  - Math server (stdio transport) / Weather server (streamable HTTP transport) 탭
- Transports 섹션
  - HTTP: 코드 블록
  - Passing headers: 코드 블록
  - Authentication: 코드 블록 + 2개 펼침 섹션 (Example custom auth, Built-in OAuth flow)
  - stdio: 코드 블록
- Stateful sessions: client.session() 사용 예제
- Core features
  - Tools > Loading tools, Structured content (2개 펼침 섹션), Multimodal tool content
  - Resources > Loading resources (2개 코드 블록)
  - Prompts > Loading prompts (2개 코드 블록)
- Advanced features
  - Tool interceptors: 테이블
  - Accessing runtime context: Runtime context/Store/State/Tool call ID 탭 (4개 코드 블록)
  - State updates and commands: 2개 코드 블록
  - Custom interceptors: Basic pattern, Modifying requests, Modifying headers, Composing interceptors, Error handling
  - Progress notifications: 콜백 코드 블록
  - Logging: 콜백 코드 블록
  - Elicitation: Server setup, Client setup, Response actions 섹션
- Additional resources: 3개 링크

**특이사항:**
- 매우 긴 페이지 (MCP 전체 기능 문서화)
- URL이 `/model-context-protocol`이 아닌 `/mcp`로 변경됨
- 다수의 탭 구조 (pip/uv, Math/Weather server, Runtime context 관련 4개 탭)
- 다수의 펼침 섹션 (Authentication, Structured content 등)
- 테이블 2개 (Tool interceptors, Response actions)

### 2026-02-04: 21-human-in-the-loop.md 완료

**포함된 요소:**
- 소개: Human-in-the-Loop (HITL) 미들웨어 설명
- Interrupt decision types: 3행 테이블 (approve, edit, reject)
- TIP 박스: editing tool arguments 주의사항
- Configuring interrupts: 코드 블록 + INFO 박스 (checkpointer 필수)
  - Configuration options: interrupt_on, description_prefix
  - InterruptOnConfig options: allowed_decisions, description
- Responding to interrupts: 코드 블록
  - Decision types 탭: approve/edit/reject (3개 코드 블록)
  - Multiple decisions: 코드 블록
- Streaming with human-in-the-loop: 코드 블록
- Execution lifecycle: 5단계 번호 목록
- Custom HITL logic: 설명

**특이사항:**
- 비교적 간단한 구조의 페이지
- Decision types 섹션에 탭 (approve/edit/reject)
- 여러 개의 Configuration options 서브섹션 (테이블 형태로 변환)
- INFO/TIP 박스 2개 포함

### 2026-02-04: 22-multi-agent.md (Overview) 완료

**포함된 요소:**
- 소개: Multi-agent systems 설명
- Why multi-agent?: 3개 항목 (Context management, Distributed development, Parallelization)
- TIP 박스: context engineering
- Patterns: 5행 테이블 (Subagents, Handoffs, Skills, Router, Custom workflow)
- Choosing a pattern: 별점 테이블 (4개 특성 x 4개 패턴)
- TIP 박스: 패턴 혼합 가능
- Visual overview: 4개 탭 (Subagents, Handoffs, Skills, Router)
- Performance comparison:
  - Key metrics: Model calls, Tokens processed 설명
  - One-shot request: 테이블 + 4개 탭 설명
  - Repeat request: 테이블 + 4개 탭 설명
  - Multi-domain: 테이블 + 4개 탭 설명
  - Summary: 2개 테이블 (비교표, 최적화 목적별)

**특이사항:**
- Overview 페이지 (하위 페이지 Subagents, Handoffs, Skills, Router, Custom workflow 있음)
- 많은 탭 구조 (Visual overview, Performance comparison 각 섹션에 4개 탭)
- 다수의 테이블 (Patterns, Choosing a pattern, Performance comparison)
- 별점 평가 (⭐) 사용
- TIP 박스 2개

### 2026-02-04: 23-retrieval.md 완료

**포함된 요소:**
- 소개: LLM 제한사항 (Finite context, Static knowledge), RAG 소개
- Building a knowledge base: INFO 박스 (기존 knowledge base 사용 가능)
- Tutorial 카드: Semantic search
- From retrieval to RAG: 소개
- Retrieval pipeline: Mermaid flowchart (Sources → Loaders → Documents → Split → Embeddings → Store → Retriever → LLM)
- Building blocks: 5행 테이블 (Document loaders, Text splitters, Embedding models, Vector stores, Retrievers)
- RAG architectures: 테이블 (2-Step RAG, Agentic RAG, Hybrid) + INFO 박스 (Latency)
- 2-step RAG: 설명 + Mermaid flowchart + Tutorial 카드
- Agentic RAG: 설명 + Mermaid flowchart (decision 노드 포함) + 코드 블록 + 펼침 섹션 (Extended example) + Tutorial 카드
- Hybrid RAG: 설명 + 3개 항목 (Query enhancement, Retrieval validation, Answer validation) + Mermaid flowchart + Tutorial 카드

**특이사항:**
- 4개의 Mermaid 다이어그램 포함
- 색상 스타일링된 Mermaid (startend, process, decision classDef)
- 1개의 펼침 섹션 (Extended example: Agentic RAG)
- 여러 개의 Tutorial 카드 (인용 블록으로 변환)
- INFO 박스 2개

### 2026-02-04: 24-long-term-memory.md 완료

**포함된 요소:**
- Overview: LangGraph persistence 소개
- Memory storage: namespace, key 설명 + 코드 블록 (InMemoryStore 사용)
- Read long-term memory in tools: 코드 블록 (ToolRuntime, store.get 사용)
- Write long-term memory from tools: 코드 블록 (TypedDict, store.put 사용)

**특이사항:**
- 비교적 간단한 페이지
- 탭/펼침 섹션 없음
- 3개의 코드 블록
- LangGraph persistence 링크 포함

### 2026-02-04: 25-langsmith-studio.md 완료

**포함된 요소:**
- 소개: LangSmith Studio (무료 시각적 인터페이스) 설명
- Prerequisites: 3개 항목 (LangSmith account, API key, LANGSMITH_TRACING 환경변수)
- Set up local Agent server:
  - 1. Install the LangGraph CLI: 코드 블록
  - 2. Prepare your agent: agent.py 코드 블록 (send_email 함수 + create_agent 사용)
  - 3. Environment variables: .env 파일 설명 + CAUTION 박스
  - 4. Create a LangGraph config file: langgraph.json 코드 블록 + INFO 박스 + 프로젝트 구조 코드 블록
  - 5. Install dependencies: pip/uv 탭 (2개 코드 블록)
  - 6. View your agent in Studio: langgraph dev 명령 + TIP 박스 (Safari 제한) + Studio 기능 설명
- Video guide: Set up local Agent Server, Deploy 링크

**특이사항:**
- 비교적 간단한 페이지
- pip/uv 탭 1개 (Install dependencies 섹션)
- CAUTION, INFO, TIP 박스 3개
- 프로젝트 구조 코드 블록 포함
- LangSmith docs 링크 목록 6개

### 2026-02-04: 26-test.md 완료

**포함된 요소:**
- 소개: Unit tests vs Integration tests 설명
- Unit testing
  - Mocking chat model: GenericFakeChatModel 코드 블록
  - InMemorySaver checkpointer: 다중 턴 테스트 코드 블록
- Integration testing
  - 2개 카드: Trajectory match, LLM-as-judge
  - Installing AgentEvals: pip install agentevals
  - Trajectory match evaluator: 4개 모드 테이블 (strict, unordered, subset, superset)
    - 4개 펼침 섹션 (Strict match, Unordered match, Subset and superset match 코드 블록)
  - LLM-as-Judge evaluator:
    - 2개 펼침 섹션 (Without reference trajectory, With reference trajectory)
  - Async support: 펼침 섹션 (async_judge, async_evaluator 코드 블록)
- LangSmith integration:
  - 환경변수 설정 코드 블록
  - 2개 펼침 섹션 (Using pytest integration, Using the evaluate function)
- Recording & replaying HTTP calls:
  - conftest.py 코드 블록
  - pytest.ini/pyproject.toml 탭 (2개 코드 블록)
  - WARNING 박스 (cassettes 주의)

**특이사항:**
- 매우 긴 페이지
- 8개의 펼침 섹션 (`<details>`)
- pytest.ini/pyproject.toml 탭
- 2개 카드 레이아웃 (Trajectory match, LLM-as-judge)
- 테이블 2개 (evaluator modes, approach comparison)
- 다수의 코드 블록 (테스트 함수 예제)

### 2026-02-04: 27-agent-chat-ui.md 완료

**포함된 요소:**
- 소개: Agent Chat UI (Next.js application) 설명
- 비디오 임베드: Introducing Agent Chat UI (YouTube)
- TIP 박스: generative UI 설명 + LangGraph 링크
- Quick start: 3개 번호 목록 (Visit, Connect, Start chatting)
- Local development: Use npx / Clone repository 탭 (2개 코드 블록)
- Connect to your agent: 3개 설정 항목 (Graph ID, Deployment URL, LangSmith API key)
- TIP 박스: tool calls/tool result messages 렌더링 설명

**특이사항:**
- 비교적 간단한 페이지
- Use npx / Clone repository 탭 1개
- TIP 박스 2개
- YouTube 비디오 임베드 (인용 블록으로 변환)
- Agent Chat UI 호스트 버전 링크

### 2026-02-04: 28-deployment.md 완료

**포함된 요소:**
- 소개: LangSmith managed hosting platform 설명
- Prerequisites: 2개 항목 (GitHub account, LangSmith account)
- Deploy your agent:
  - 1. Create a repository on GitHub: 설명 + local server setup guide 링크
  - 2. Deploy to LangSmith: 4개 서브스텝 (Navigate, Create, Link, Deploy)
  - 3. Test your application in Studio: 2개 번호 목록
  - 4. Get the API URL: 2개 번호 목록
  - 5. Test the API: Python / Rest API 탭 (2개 코드 블록)
- TIP 박스: self-hosted and hybrid 옵션 안내

**특이사항:**
- 비교적 간단한 페이지
- Python / Rest API 탭 1개
- TIP 박스 1개
- 번호 매김 단계별 가이드 (1-5)
- 서브스텝 (2. Deploy to LangSmith 안에 4개 항목)

### 2026-02-05: Multi-agent 하위 페이지 5개 전체 완료! 🎉

**23-subagents.md 포함된 요소:**
- 소개: subagents 아키텍처 설명 (supervisor + subagents via tools)
- Mermaid flowchart: User → MainAgent → SubagentA/B/C → UserResponse
- Key characteristics: 4개 항목
- INFO 박스: Supervisor vs. Router 비교
- When to use: 설명 + TIP 박스 (user interaction within subagent)
- Basic implementation: 코드 블록 + Tutorial 카드
- Design decisions: 테이블 (5가지 결정 사항)
- Sync vs. async: 테이블 + 2개 Mermaid sequence diagram
- Tool patterns: 테이블 + Mermaid flowchart + 코드 블록 + 펼침 섹션
- Context engineering: 테이블
- Subagent specs: 3가지 방법 (System prompt, Enum constraint, Tool-based discovery)
- Subagent inputs/outputs: 펼침 섹션 2개

**24-handoffs.md 포함된 요소:**
- 소개: handoffs 아키텍처 (state-driven behavior)
- TIP 박스: OpenAI 용어 설명
- Mermaid sequence diagram: warranty support flow
- Key characteristics: 4개 항목
- When to use: 설명
- Basic implementation: 코드 블록 + TIP 박스 (ToolMessage 필요성) + Tutorial 카드
- Implementation approaches: 2가지 (Single agent with middleware, Multiple agent subgraphs)
- Single agent with middleware: 코드 블록 + 펼침 섹션 (Complete example)
- Multiple agent subgraphs: 코드 블록 + 펼침 섹션 (Complete example) + TIP 박스
- Context engineering: 설명 + 코드 블록 + TIP 박스
- Implementation considerations: 3개 항목

**25-skills.md 포함된 요소:**
- 소개: skills 아키텍처 설명
- TIP 박스: llms.txt 비교
- Mermaid flowchart: User → Agent → SkillA/B/C
- Key characteristics: 4개 항목
- When to use: 설명
- Basic implementation: 코드 블록 + Tutorial 카드
- Extending the pattern: 2가지 (Dynamic tool registration, Hierarchical skills)

**26-router.md 포함된 요소:**
- 소개: router 아키텍처 설명
- Mermaid flowchart: Query → Router → AgentA/B/C → Synthesize → Combined answer
- Key characteristics: 3개 항목
- When to use: 설명
- Basic implementation: 2개 탭 (Single agent, Multiple agents parallel) → 연속 섹션
- Stateless vs. stateful: 2가지
- Stateless: INFO 박스 (Router vs. Subagents 비교)
- Stateful: Tool wrapper, Full persistence + TIP 박스

**27-custom-workflow.md 포함된 요소:**
- 소개: custom workflow 아키텍처 설명
- Mermaid flowchart: Input → Conditional → path_a/path_b → Output
- Key characteristics: 4개 항목
- When to use: 설명 + Tutorial 카드
- Basic implementation: 코드 블록
- Example: RAG pipeline: 펼침 섹션 (Custom RAG workflow)
  - 3가지 노드 타입 설명
  - Mermaid flowchart: Query → Rewrite → Retrieve → Agent → Response
  - 전체 WNBA stats assistant 코드 블록

### 2026-02-04: 29-observability.md 완료 (최종 문서!)

**포함된 요소:**
- 소개: LangSmith tracing을 통한 에이전트 모니터링 설명
- Prerequisites: 2개 항목 (LangSmith account, API key)
- Enable tracing: 환경변수 설정 코드 블록 (LANGSMITH_TRACING, LANGSMITH_API_KEY)
- Quickstart: 전체 에이전트 코드 블록 (send_email, search_web 함수 + create_agent 사용)
- Trace selectively: tracing_context 컨텍스트 매니저 코드 블록
- Log to a project: 2개 펼침 섹션 (Statically, Dynamically)
  - Statically: LANGSMITH_PROJECT 환경변수 설명
  - Dynamically: tracing_context(project_name=...) 코드 블록
- Add metadata to traces: 2개 코드 블록 (config 사용, tracing_context 사용)
- TIP 박스: LangSmith documentation 링크

**특이사항:**
- 비교적 간단한 페이지
- 2개의 펼침 섹션 (Statically, Dynamically)
- 탭 없음
- TIP 박스 1개
- LangSmith integration을 위한 최종 문서

---

## 📈 프로젝트 진행 요약

**🎉 총 변환 완료 문서: 34개 (100%)**

| 카테고리 | 문서 수 | 상태 |
|---------|---------|------|
| Getting started (Overview, Install, Quickstart) | 3 | ✅ |
| Learn (Changelog, Philosophy) | 2 | ✅ |
| Core concepts (Agents, Models, Messages, Tools, Memory, Streaming, Structured output) | 8 | ✅ |
| Middleware | 3 | ✅ |
| Advanced usage (Guardrails, Runtime, Context, MCP, HITL, Retrieval, Long-term memory) | 7 | ✅ |
| Multi-agent (Overview + 5 sub-pages) | 6 | ✅ |
| Agent development (Studio, Test, Chat UI) | 3 | ✅ |
| Deploy with LangSmith (Deployment, Observability) | 2 | ✅ |

**변환 원칙 준수:**
- ✅ 원본 텍스트 100% 유지 (요약/누락 없음)
- ✅ 모든 링크 보존
- ✅ 탭 → `####` 연속 섹션으로 변환
- ✅ 펼침 섹션 → `<details><summary>` 태그 사용
- ✅ 차트/다이어그램 → Mermaid 코드로 변환
- ✅ 정보 박스 → `> [!TIP]`, `> [!INFO]`, `> [!WARNING]` 등 사용

**⚠️ 발견된 문제 (2026-02-05):**
- Multi-agent 섹션에 숨겨진 하위 페이지 5개가 있었음
- 사이드바의 확장 가능한 메뉴(▸) 항상 확인 필요!
- → 해결됨: 23-subagents, 24-handoffs, 25-skills, 26-router, 27-custom-workflow 추가 완료
