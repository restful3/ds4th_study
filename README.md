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
* **짝수 주 (구현)**: 퀀트 스터디 종료 후 **오후 9:30 \~** — 자유 주제 시간을 활용한 구현 공유

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
  * `01-overview_ko.md`, `02-quickstart_ko.md`, `03-customization_ko.md`
* **종훈(S)**: Context engineering, Memory, Skills
  * `04-harness_ko.md`, `08-long-term-memory_ko.md`
* **종훈(L)**: Backends, Sandboxes, Permissions
  * `05-backends_ko.md`



#### **2026년 5월 16일 (토) 오후 9:30**: 2주차 실습 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=madde6f8079f99b17c91a793e35b8de59)

* **종훈(S)**: Quickstart 및 CLI 기본 구동 실습
  * `02-quickstart_ko.md`, `10-cli_ko.md`
* **재익**: Memory 와 Skills 적용 실습
  * `04-harness_ko.md`, `08-long-term-memory_ko.md`
* **태영**: 샌드박스와 권한(Permissions) 제어 실습
  * `05-backends_ko.md`



#### **2026년 5월 23일 (토) 오전 8:30**: 3주차 이론 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=md89a6dc15f39b828835ebc9b32024bd6)

* **보현**: Subagents, Async subagents, HITL
  * `06-subagents_ko.md`, `07-human-in-the-loop_ko.md`
* **수경**: 배포(Deploy), Production, Streaming
  * `10-cli_ko.md` (Deploy 파트), `09-middleware_ko.md`
* **세훈**: Protocols (MCP, A2A, ACP), Frontend
  * `09-middleware_ko.md`, `10-cli_ko.md` (MCP 파트)



#### **2026년 5월 30일 (토) 오후 9:30**: 4주차 실습 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=mbf76dd0c34fa02a98416d13adcae86e2)

* **TBD**: Subagent 위임과 HITL (승인) 시스템 실습
  * `06-subagents_ko.md`, `07-human-in-the-loop_ko.md`
* **TBD**: MCP 연결 및 CLI 배포 실습
  * `10-cli_ko.md`

---

#### **6월 \~ 7월: LangGraph 학습 및 실습** *(분량이 많아 2개월 할당)*



#### **2026년 6월 6일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m9a194682e586905462cf452240ce3574)

* 발표자 - TBD
* 주제: **LangGraph** 입문 / StateGraph 기초



#### **2026년 6월 13일 (토) 오후 9:30**: 구현 및 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m87f9a7efa706eae278720a8f458f10a5)

* 발표자 - 자유 주제
* 주제: **LangGraph 기반 퀀트 에이전트 구현 공유**



#### **2026년 6월 20일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=maec40b0601bf91499a6c59760bd5340b)

* 발표자 - TBD
* 주제: **LangGraph 심화** (Persistence / HITL / Multi-agent 등)



#### **2026년 6월 27일 (토) 오후 9:30**: 구현 및 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m15e64a942ca773ba9ff3893b375b65b5)

* 발표자 - 자유 주제
* 주제: **LangGraph 구현 확장 및 공유**



#### **2026년 7월 4일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m039bd4fb3e1d60792aaf56f8ec2e85e9)

* 발표자 - TBD
* 주제: **LangGraph 고급 주제** (Streaming / Subgraph 등)



#### **2026년 7월 11일 (토) 오후 9:30**: 구현 및 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m245f52cbecc6367e56a150b271a78fbd)

* 발표자 - 자유 주제
* 주제: **LangGraph 구현 통합 및 공유**



#### **2026년 7월 18일 (토) 오전 8:30**: 이론 발표 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m2757d0104b346d8532d4d18d4a6ed952)

* 발표자 - TBD
* 주제: **LangGraph 마무리 / 베스트 프랙티스**



#### **2026년 7월 25일 (토) 오후 9:30**: 최종 구현 공유 — [Webex 링크](https://lgehq.webex.com/lgehq-en/j.php?MTID=m30772a3f3b895e39b31b3d369e35e71c)

* 발표자 - 전원
* 주제: **퀀트 에이전트 최종 결과물 공유 및 회고**

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
