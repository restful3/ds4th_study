# **ds4th study**

### **1. 스터디 목적**

* **AI 에이전트(Deep Agents, LangGraph 등)의 핵심 개념과 구현 방법을 학습** 하고, 이를 **'퀀트 에이전트' 공통 프로젝트** 에 적용해 봄으로써 실제 에이전트 설계·구현 역량을 강화하는 것을 목표로 한다.
* **핵심 도구**: Deep Agents, LangGraph (LangChain 은 필요 시 참조용으로 활용)
* **주요 주제**: MCP, Memory, Human-in-the-loop, Skills 등 실제 에이전트 설계에 필수적인 영역
* **Pydantic** 등은 별도 메인 주제로 다루지 않고, 필요 시 참조하는 형태로만 활용

---

### **2. 스터디 시간**

* **격주 로테이션** 으로 운영
* **홀수 주 (이론)**: 매주 토요일 **오전 8:30 \~** — 문서 학습 중심 스터디
* **짝수 주 (구현)**: 퀀트 스터디 종료 후 **오전 9:30 \~** — 자유 주제 시간을 활용한 구현 공유

---

### **3. 스터디 장소**

* Webex

---

### **4. 역할 분담**

* **Deep Agents 리드**: 종훈(S)
* **LangGraph 리드**: 수경
* **리드 역할**: 스케줄 관리, 핵심 주제 선정, 파트원별 영역 배정, 최종 결과물(발표 및 구현) 관리

---

### **5. 스터디 운영 계획 (2026년 5월 \~ 2026년 7월)**

---

#### **5월: Deep Agents 학습 및 실습**



#### **2026년 5월 9일 (토) 오전 8:30**: 1주차 이론 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m661ff64edcb8b55c76ab6d5011cb7f27)

* **태영**: Overview, Quickstart, Customization, Models
  * 원문: [`01-overview_ko.md`](https://github.com/restful3/langchain-docs/blob/main/deep-agents/01-overview_ko.md), [`02-quickstart_ko.md`](https://github.com/restful3/langchain-docs/blob/main/deep-agents/02-quickstart_ko.md), [`03-customization_ko.md`](https://github.com/restful3/langchain-docs/blob/main/deep-agents/03-customization_ko.md)
  * 발표 자료: [week1-overview-taeyoung](https://github.com/restful3/langchain-docs/tree/main/deep-agents/week1-overview-taeyoung)
* **종훈(S)**: Context engineering, Memory, Skills
  * `04-harness_ko.md`, `08-long-term-memory_ko.md`
* **종훈(L)**: Backends, Sandboxes, Permissions
  * `05-backends_ko.md`



#### **2026년 5월 16일 (토) 오전 9:30**: 2주차 실습 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=madde6f8079f99b17c91a793e35b8de59)

* **태영**: Quickstart 및 CLI 기본 구동 실습
  * `01-overview_ko.md`, `02-quickstart_ko.md`, `03-customization_ko.md`
* **종훈(S)**: Memory 와 Skills 적용 실습
  * `04-harness_ko.md`, `08-long-term-memory_ko.md`
* **종훈(L)**: 샌드박스와 권한(Permissions) 제어 실습
  * `05-backends_ko.md`



#### **2026년 5월 23일 (토) 오전 8:30**: 3주차 이론 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=md89a6dc15f39b828835ebc9b32024bd6)

* **보현**: Subagents, Async subagents, HITL
  * `06-subagents_ko.md`, `07-human-in-the-loop_ko.md`
* **수경**: 배포(Deploy), Production, Streaming
  * `10-cli_ko.md` (Deploy 파트), `09-middleware_ko.md`
* **세훈**: Protocols (MCP, A2A, ACP), Frontend
  * `09-middleware_ko.md`, `10-cli_ko.md` (MCP 파트)



#### **2026년 5월 30일 (토) 오전 9:30**: 4주차 실습 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=mbf76dd0c34fa02a98416d13adcae86e2)

* **보현**: Subagents, Async subagents, HITL
  * `06-subagents_ko.md`, `07-human-in-the-loop_ko.md`
* **수경**: 배포(Deploy), Production, Streaming
  * `10-cli_ko.md` (Deploy 파트), `09-middleware_ko.md`
* **세훈**: Protocols (MCP, A2A, ACP), Frontend
  * `09-middleware_ko.md`, `10-cli_ko.md` (MCP 파트)

---

#### **6월 \~ 7월: LangGraph 학습 및 실습** *(분량이 많아 2개월 할당)*



#### **2026년 6월 6일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m9a194682e586905462cf452240ce3574)

* 발표자 - TBD
* 주제: **LangGraph** 입문 / StateGraph 기초



#### **2026년 6월 13일 (토) 오전 9:30**: 구현 및 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m87f9a7efa706eae278720a8f458f10a5)

* 발표자 - 자유 주제
* 주제: **LangGraph 기반 퀀트 에이전트 구현 공유**



#### **2026년 6월 20일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=maec40b0601bf91499a6c59760bd5340b)

* 발표자 - TBD
* 주제: **LangGraph 심화** (Persistence / HITL / Multi-agent 등)



#### **2026년 6월 27일 (토) 오전 9:30**: 구현 및 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m15e64a942ca773ba9ff3893b375b65b5)

* 발표자 - 자유 주제
* 주제: **LangGraph 구현 확장 및 공유**



#### **2026년 7월 4일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m039bd4fb3e1d60792aaf56f8ec2e85e9)

* 발표자 - TBD
* 주제: **LangGraph 고급 주제** (Streaming / Subgraph 등)



#### **2026년 7월 11일 (토) 오전 9:30**: 구현 및 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m245f52cbecc6367e56a150b271a78fbd)

* 발표자 - 자유 주제
* 주제: **LangGraph 구현 통합 및 공유**



#### **2026년 7월 18일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m2757d0104b346d8532d4d18d4a6ed952)

* 발표자 - TBD
* 주제: **LangGraph 마무리 / 베스트 프랙티스**



#### **2026년 7월 25일 (토) 오전 9:30**: 최종 구현 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m30772a3f3b895e39b31b3d369e35e71c)

* 발표자 - 전원
* 주제: **퀀트 에이전트 최종 결과물 공유 및 회고**

---

## 📅 6월 ~ 7월 LangGraph 세미나 일정

AI 에이전트 스터디 6월~7월 LangGraph 파트 상세 운영 계획 및 발표자 배정 현황입니다.

* **이론 주차 (홀수 회차):** LangGraph 공식 문서 (`01.md` ~ `26.md`)를 분할하여 세부 항목별 담당자가 발표 진행 (오전 08:30 ~)
* **구현 주차 (짝수 회차):** 퀀트 에이전트 구현을 공통 주제로 삼아 실습 결과물 공유 (오후 09:30 ~)
* **참여 인원 (총 9명):** 수경, 송태영, 재익, 종훈(S), 보현, 정훈, 종훈(L), 세훈, 태호

---

### 🗓️ 주차별 상세 일정

#### 🔹 [1회차] 2026년 6월 6일 (토) 오전 8:30: 이론 발표 (1차) — [Webex 링크]()
* **주제:** LangGraph 패러다임과 핵심 구조 (Fundamentals)
* **발표자 및 담당 문서:**
  * 👤 **수경**: `01.md` ~ `03.md` (LangGraph 개요, 기본 철학 및 에이전트 아키텍처 패러다임)
  * 👤 **송태영**: `04.md` ~ `06.md` (핵심 3요소 정의: `State` 스키마, `Nodes` 작업자, `Edges` 라우팅)
  * 👤 **재익**: `07.md` ~ `09.md` (기본 흐름 제어, 기초 그래프 빌드, 컴파일 및 시각화)

#### 🔹 [2회차] 2026년 6월 13일 (토) 오후 9:30: 구현 및 공유 (1차) — [Webex 링크]()
* **주제:** LangGraph 기초 그래프 빌드 및 퀀트 에이전트 프로토타입 공유
* **발표자:** 👤 **수경**, 👤 **종훈(S)**

---

#### 🔹 [3회차] 2026년 6월 20일 (토) 오전 8:30: 이론 발표 (2차) — [Webex 링크]()
* **주제:** 상태 지속성 및 인간 개입 워크플로우 (Advanced Control)
* **발표자 및 담당 문서:**
  * 👤 **보현**: `10.md` ~ `12.md` (**Persistence** - Checkpointer 메커니즘을 통한 세션 유지, 상태 저장 및 복구)
  * 👤 **정훈**: `13.md` ~ `14.md` (**Memory** - 단기 대화 맥락 및 장기 기억 시스템 설계)
  * 👤 **종훈(L)**: `15.md` ~ `18.md` (**HITL & Streaming** - 인간 개입 루프 구현 및 실시간 LLM/노드 상태 스트리밍)

#### 🔹 [4회차] 2026년 6월 27일 (토) 오후 9:30: 구현 및 공유 (2차) — [Webex 링크]()
* **주제:** MCP 기반 멀티 에이전트 도구 연동 및 환경 구축 실습 공유
* **발표자:** 👤 **세훈**, 👤 **태호**

---

#### 🔹 [5회차] 2026년 7월 4일 (토) 오전 8:30: 이론 발표 (3차) — [Webex 링크]()
* **주제:** 멀티 에이전트 오케스트레이션 및 컴포넌트화 (Multi-Agent Systems)
* **발표자 및 담당 문서:**
  * 👤 **종훈(S)**: `19.md` ~ `21.md` (**Subgraphs** - 하위 그래프를 메인 그래프에 플러그인 형태로 조립·모듈화)
  * 👤 **종훈(L)**: `22.md` ~ `23.md` (**Multi-Agent** - Supervisor 패턴, 계층적 구조 및 P2P 패턴 구현)
  * 👤 **보현**: `24.md` ~ `26.md` (**Error Handling & Config** - 예외 추적, 자동 재시도 및 런타임 설정 관리)

#### 🔹 [6회차] 2026년 7월 11일 (토) 오후 9:30: 구현 및 공유 (3차) — [Webex 링크]()
* **주제:** 메모리(Memory) 및 휴먼인더루프(HITL) 워크플로우 적용 실습 공유
* **발표자:** 👤 **송태영**, 👤 **재익**

---

#### 🔹 [7회차] 2026년 7월 18일 (토) 오전 8:30: 이론 발표 (4차) — [Webex 링크]()
* **주제:** 총 마무리 정리 및 엔지니어링 최적화 (Wrap-up & Production)
* **발표자 및 담당 주제 (`01`~`26.md` 종합 및 응용):**
  * 👤 **수경**: **종합 아키텍처 & 상태 최적화** (핵심 디자인 패턴 요약 및 대규모 환경에서의 State 클린업 기법)
  * 👤 **세훈**: **디버깅 & 테스트** (LangSmith 연동 에이전트 트레이싱 및 그래프 단위 유닛 테스트)
  * 👤 **태호**: **실무 연동 및 구현 전략** (내재화 중인 시스템 및 MCP 환경에 LangGraph 아키텍처 결합 제언)

#### 🔹 [8회차] 2026년 7월 25일 (토) 오후 9:30: 최종 구현 공유 (4차) — [Webex 링크]()
* **주제:** 퀀트 에이전트 최종 결과물 공유 및 회고
* **발표자:** 👥 **전원** (메인 구현 공유: 👤 **보현**, 👤 **정훈**)


---

### **6. 스터디 운영 방법**

* **학습 대상**:
  * Deep Agents, LangGraph 공식 문서 및 관련 자료 (LangChain 은 필요 시 참조)
  * 참고 교재: [langchain-docs (한글 번역)](https://github.com/restful3/langchain-docs)
* **학습 방법**:
  * 방대한 문서량을 효율적으로 소화하기 위해 담당자를 지정하여 해당 파트만 깊게 학습
  * 발표는 **NotebookLM 등 AI 도구로 생성한 요약 자료** 를 활용하여 핵심 위주로 진행 (1시간 내)
  * 나머지 인원은 AI 요약과 발표를 통해 전체 내용을 파악
* **구현 주제**:
  * **'퀀트 에이전트 구현'** 을 공통 주제로 선정하여 각자의 관심사에 따라 자유롭게 기능을 추가하며 학습 내용 적용
* **학습 공유**: 매주 학습한 내용을 발표자료와 함께 GitHub 에 공유
* **운영 규칙**:
  * [스터디 운영 규칙](https://github.com/restful3/ds4th_study/blob/main/source/etc/%EC%8A%A4%ED%84%B0%EB%94%94_%EC%9A%B4%EC%98%81_%EA%B7%9C%EC%B9%99_v01.pdf)

---

### **7. 향후 계획**

* 각 파트 리드는 **5월 9일 첫 발표를 위해 5월 2일까지 멤버별 담당 영역 할당을 완료** 하고 공유할 예정
* 발표자는 **NotebookLM** 등 AI 도구를 활용해 슬라이드 및 인포그래픽을 생성함으로써 준비 부담을 최소화

---

### **8. 기타**

* **참가 희망 요청**: [Email](mailto:restful3@gmail.com)

* **이제 까지 다룬 내용**: [archive 확인](https://github.com/restful3/ds4th_study/tree/main/archive)
