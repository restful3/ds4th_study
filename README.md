# **ds4th study**

AI, 데이터 과학, 머신러닝, 에이전트 기술을 함께 학습하고 발표 자료와 실습 코드를 공유하는 스터디 저장소입니다.

현재는 **Knowledge Graphs and LLMs in Action** 교재를 중심으로 지식 그래프와 LLM을 결합한 하이브리드 지능형 시스템을 학습합니다.

---

## **1. 현재 스터디**

### **Knowledge Graphs and LLMs in Action**

* **교재**: Alessandro Negro 외, *Knowledge Graphs and LLMs in Action*
* **자료 위치**: [`source/books/Alessandro Negro - Knowledge Graphs and LLMs in Action`](source/books/Alessandro%20Negro%20-%20Knowledge%20Graphs%20and%20LLMs%20in%20Action)
* **학습 목표**
  * Knowledge Graph와 LLM의 결합 방식 이해
  * 구조화/비구조화 데이터에서 Knowledge Graph를 구축하는 방법 학습
  * Graph ML, GNN, Graph RAG, 자연어 기반 KG 질의, LangGraph 기반 QA 에이전트 구현 흐름 학습
  * 이후 에이전트/검색/RAG 프로젝트에 적용 가능한 공통 설계 감각 확보

---

## **2. 스터디 시간**

* **운영 주기**: 격주 토요일
* **AI 스터디**: 토요일 **09:00-10:00**
* **참고**: 같은 날 퀀트 스터디는 **08:00-09:00**에 진행
* **휴식일**: **2026년 7월 18일(토)** 은 휴식
* **재개일**: **2026년 7월 25일(토)**

---

## **3. 스터디 장소**

* Webex

---

## **4. 운영 계획 (2026년 7월-10월)**

> 최대 8회 발표 기준으로 구성했습니다.  
> 쪽수는 교재의 인쇄 쪽수 기준이며, 챕터 경계를 우선해 일부 회차는 40-60쪽 범위에서 조금 벗어납니다.

| 회차 | 일자 | 범위 | 주제 | 페이지 | 발표자 | 자료 |
|---:|---|---|---|---:|---|---|
| - | 2026-07-18 | 휴식 | 다음 교재 준비 | - | - | - |
| 1 | 2026-07-25 | Ch1-Ch3 | KG와 LLM의 결합, 하이브리드 지능형 시스템, 온톨로지 기반 첫 KG 구축 | pp. 1-64 (64쪽) | 수경 | 추후 공유 |
| 2 | 2026-08-08 | Ch4-Ch6 | 멀티소스 통합, 비정형 데이터 지식 추출, LLM 기반 KG 구축 | pp. 65-128 (64쪽) | 추후 확정 | 추후 공유 |
| 3 | 2026-08-22 | Ch7 | Named Entity Disambiguation 기초와 도메인 기반 KG 활용 | pp. 129-179 (51쪽) | 추후 확정 | 추후 공유 |
| 4 | 2026-09-05 | Ch8-Ch9 | Open LLM/도메인 온톨로지 기반 NED, KG 위의 머신러닝 개요 | pp. 180-232 (53쪽) | 추후 확정 | 추후 공유 |
| 5 | 2026-09-19 | Ch10 | 그래프 특성 공학: 수동/반자동 접근 | pp. 233-271 (39쪽) | 추후 확정 | 추후 공유 |
| 6 | 2026-10-03 | Ch11-Ch12 | 그래프 임베딩, GNN, 노드 분류와 링크 예측 | pp. 272-334 (63쪽) | 추후 확정 | 추후 공유 |
| 7 | 2026-10-17 | Ch13-Ch14 | Graph RAG, 자연어 기반 Knowledge Graph 질의 | pp. 335-396 (62쪽) | 추후 확정 | 추후 공유 |
| 8 | 2026-10-31 | Ch15 + Appendix A | LangGraph 기반 QA 에이전트와 그래프 기초 정리 | pp. 397-446 (50쪽) | 추후 확정 | 추후 공유 |

### **부록 운영**

* **Appendix B. Neo4j**, **Appendix C. Building knowledge graphs from structured sources** 는 발표 필수 범위에서 제외합니다.
* 필요 시 Ch15 발표 또는 별도 실습 시간에서 참고 자료로 활용합니다.

---

## **5. 스터디 운영 방법**

* **발표 방식**: 50분 발표 + 10분 Q&A
* **발표 분량**: 회당 대략 40-60쪽
* **발표 준비**
  * 챕터별 원문 Markdown과 한국어 번역본을 우선 활용
  * NotebookLM 등 AI 도구로 요약, 슬라이드, 인포그래픽을 생성해 발표 부담을 줄임
  * 발표자는 핵심 개념, 구현 흐름, 적용 가능성을 중심으로 정리
* **자료 공유**
  * 발표 자료, 코드, 추가 정리 문서는 GitHub에 공유
  * 챕터별 자료는 가능하면 해당 교재 폴더 아래에 함께 정리
* **운영 규칙**: [스터디 운영 규칙](source/etc/%EC%8A%A4%ED%84%B0%EB%94%94_%EC%9A%B4%EC%98%81_%EA%B7%9C%EC%B9%99_v01.pdf)

---

## **6. 자료 구조**

| 경로 | 설명 |
|---|---|
| [`source/books`](source/books) | 현재 진행 중이거나 신규 학습 대상인 책 자료 |
| [`source/papers`](source/papers) | 논문, 번역, 해설판, 발표 자료 |
| [`source/deep_agents`](source/deep_agents) | 2026년 5월 Deep Agents 스터디 자료 |
| [`source/langgraph`](source/langgraph) | 2026년 6-7월 LangGraph 스터디 자료와 회의록 |
| [`archive`](archive) | 이전 스터디 자료 아카이브 |
| [`official`](official) | 공식 문서 또는 원본 참고 자료 |

---

## **7. 이전 진행 이력**

### **Deep Agents / LangGraph 기반 퀀트 에이전트 스터디**

* **기간**: 2026년 5월-2026년 7월 11일
* **주제**: Deep Agents, LangGraph, MCP, Memory, Human-in-the-loop, Skills, Multi-agent, Streaming
* **공통 프로젝트**: 퀀트 에이전트 구현
* **자료**
  * [`source/deep_agents`](source/deep_agents)
  * [`source/langgraph`](source/langgraph)

---

## **8. 회의록**

* [2026-07-11 랭그래프 스터디 종료 및 차기 교재 논의](source/langgraph/%ED%9A%8C%EC%9D%98%EB%A1%9D/2026-07-11_%EB%9E%AD%EA%B7%B8%EB%9E%98%ED%94%84_%EC%8A%A4%ED%84%B0%EB%94%94_%EC%A2%85%EB%A3%8C_%EB%B0%8F_%EC%B0%A8%EA%B8%B0_%EA%B5%90%EC%9E%AC_%EB%85%BC%EC%9D%98.md)

---

## **9. 기타**

* **참가 희망 요청**: [Email](mailto:restful3@gmail.com)
* **이전 자료**: [archive 확인](archive)
