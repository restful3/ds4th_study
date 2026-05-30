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

모든 참여자가 각자 LangGraph 기반 퀀트 에이전트(Quant Agent)를 개발하며, 배정된 발표자는 **당일 담당한 이론(마크다운 문서) 발표와 본인의 퀀트 에이전트 내 해당 기능 구현 코드를 함께 공유**합니다.

* **운영 방식:** 홀수 회차는 2명, 짝수 회차는 1명이 발표를 전담하여 실시하며, 마지막 회차는 전원이 함께 참여합니다.
* **참여 인원 (총 9명):** 수경, 송태영, 재익, 종훈(S), 보현, 정훈, 종훈(L), 세훈, 태호

#### 🗓️ 주차별 상세 일정 (이론 + 구현 통합형)

#### 🔹 [1회차] 2026년 6월 6일 (토) 오전 8:30 (홀수 회차 - 2명) — [Webex 링크]()
* **주제:** LangGraph 패러다임과 핵심 아키텍처 기초
* **발표자 및 담당 내용:**
  * 👤 **수경**: `01.md` ~ `03.md` 이론 + 본인의 퀀트 에이전트 기본 상태 구조(State) 및 노드 초기 설계 공유
  * 👤 **송태영**: `04.md` ~ `06.md` 이론 + 본인의 퀀트 에이전트 라우팅(Edges) 및 컴파일 구조 공유

#### 🔹 [2회차] 2026년 6월 13일 (토) 오후 9:30 (짝수 회차 - 1명) — [Webex 링크]()
* **주제:** 그래프 빌드 및 워크플로우 시각화
* **발표자 및 담당 내용:**
  * 👤 **재익**: `07.md` ~ `09.md` 이론 + 조건부 에지 라우팅 및 퀀트 그래프 시각화(Visualization) 구현 공유

#### 🔹 [3회차] 2026년 6월 20일 (토) 오전 8:30 (홀수 회차 - 2명) — [Webex 링크]()
* **주제:** 상태 지속성(Persistence)과 에이전트 메모리 시스템
* **발표자 및 담당 내용:**
  * 👤 **종훈(S)**: `10.md` ~ `12.md` 이론 + Checkpointer를 이용한 퀀트 에이전트 세션 저장 및 복구 구현 공유
  * 👤 **보현**: `13.md` ~ `14.md` 이론 + 퀀트 거래 맥락 유지를 위한 단기/장기 메모리(Memory) 데이터베이스 연동 공유

#### 🔹 [4회차] 2026년 6월 27일 (토) 오후 9:30 (짝수 회차 - 1명) — [Webex 링크]()
* **주제:** 인간 개입(HITL) 루프 및 실시간 스트리밍
* **발표자 및 담당 내용:**
  * 👤 **정훈**: `15.md` ~ `18.md` 이론 + 매수/매도 최종 승인 단계를 위한 인간 개입 루프 및 토큰 단위 스트리밍 구현 공유

#### 🔹 [5회차] 2026년 7월 4일 (토) 오전 8:30 (홀수 회차 - 2명) — [Webex 링크]()
* **주제:** 하위 그래프 모듈화 및 멀티 에이전트 기초
* **발표자 및 담당 내용:**
  * 👤 **종훈(L)**: `19.md` ~ `21.md` 이론 + 복잡한 자산 배분/분석 로직을 분리하는 하위 그래프(Subgraphs) 모듈화 구현 공유
  * 👤 **세훈**: `22.md` ~ `23.md` 이론 + Supervisor 패턴을 적용한 멀티 에이전트(Multi-Agent) 조율 구조 공유

#### 🔹 [6회차] 2026년 7월 11일 (토) 오후 9:30 (짝수 회차 - 1명) — [Webex 링크]()
* **주제:** 예외 처리 및 런타임 설정 제어
* **발표자 및 담당 내용:**
  * 👤 **태호**: `24.md` ~ `26.md` 이론 + 에이전트 실행 오류 자동 재시도(Retry) 및 시장 상황별 런타임 설정(Configuration) 관리 공유

#### 🔹 [7회차] 2026년 7월 18일 (토) 오전 8:30 (홀수 회차 - 2명) — [Webex 링크]()
* **주제:** 종합 아키텍처 최적화 및 에이전트 테스트 엔지니어링
* **발표자 및 담당 내용:**
  * 👤 **수경**: 종합 디자인 패턴 요약 이론 + 대규모 트래픽 대비 퀀트 에이전트 State 클린업 및 메모리 최적화 구조 공유
  * 👤 **송태영**: 디버깅 및 프로덕션 배포 이론 + LangSmith 연동 트레이싱(Tracing), 그래프 단위 유닛 테스트 검증 및 MCP(Model Context Protocol) 환경 결합 구현 공유

#### 🔹 [8회차] 2026년 7월 25일 (토) 오후 9:30 (최종 회차 - 전원) — [Webex 링크]()
* **주제:** 퀀트 에이전트 최종 결과물 공유 및 스터디 회고
* **발표자:** 👥 **전원 참여**
* **내용:** 각자 개발 완료한 LangGraph 기반 퀀트 에이전트 최종 시스템 시연, 성과 공유 및 2개월간의 스터디 총평/회고

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
