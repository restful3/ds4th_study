# **ds4th study**

### **1. 스터디 목적**

* **『Knowledge Graphs and LLMs in Action』(Alessandro Negro 외, Manning)를 통해 지식 그래프와 LLM의 결합 방식부터 온톨로지 기반 KG 구축·비정형 데이터 지식 추출·개체명 중의성 해소·Graph ML과 GNN·Graph RAG·LangGraph 기반 QA 에이전트까지, 하이브리드 지능형 시스템을 직접 구현하는 데 필요한 방법론을 체계적으로 학습하는 것** 을 목표로 한다.

---

### **2. 스터디 시간**

* 매주 토요일 **09:00–10:00** (기존 08:30 시작에서 09:00 시작으로 변경)

---

### **3. 스터디 장소**

* Webex

---

### **4. 스터디 운영 계획 (2026년 7월 \~ 2026년 10월)**

> 2026년 7월 18일(토)은 휴식하고, 7월 25일(토)부터 재개한다.
>
> 쪽수는 교재의 인쇄 쪽수 기준이다. 첫 회차는 Chapter 1·2를 함께 다루고, 이후에는 매주 한 챕터씩 진행한다. 발표 담당은 확정되는 대로 갱신한다.

---

#### **2026년 7월 25일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=me1d2791ff716182dd6d9d2d0029b6697)

* 발표자 - 수경
* **Chapter 1. 지식 그래프와 LLM: 강력한 조합 (Knowledge Graphs and LLMs: A Killer Combination)** — pp. 1–16
    * 지식 그래프(KG)와 대규모 언어 모델(LLM) 소개
    * LLM의 환각·맥락 부족 한계와 KG의 보완 관계
    * KG와 LLM을 결합한 데이터 기반 애플리케이션 구축
* Chapter 1 자료: [상세 리포트](https://restful3.github.io/ds4th_study/studies/knowledge-graphs-and-llms-in-action/presentations/2026-07-25-ch01/report.html) · [발표자료](https://restful3.github.io/ds4th_study/studies/knowledge-graphs-and-llms-in-action/presentations/2026-07-25-ch01/)
* **Chapter 2. 지능형 시스템: 하이브리드 접근 (Intelligent Systems: A Hybrid Approach)** — pp. 17–36
    * 지능형 자문 시스템(IAS)을 위한 설계 개념과 아키텍처
    * 하이브리드 시스템이 KG와 LLM의 상호 보완적 강점을 활용하는 방식
* Chapter 2 자료: 추후 공유


#### **2026년 8월 1일**: [Webex](#)

* 발표자 - 태영
* **Chapter 3. 온톨로지로 첫 지식 그래프 만들기 (Create Your First Knowledge Graph from Ontologies)** — pp. 37–64
    * 사용 사례에 기반한 KG 기술 선택
    * 인간 표현형 온톨로지(HPO) 기반 임상 지원 KG 구축
    * KG 위에서의 분석과 온톨로지 기반 추론
* 자료: 추후 공유


#### **2026년 8월 8일**: [Webex](#)

* 발표자 - 종훈(S)
* **Chapter 4. 단순 네트워크에서 멀티소스 통합으로 (From Simple Networks to Multisource Integration)** — pp. 65–94
    * 여러 구조화 데이터 소스로부터 복잡한 KG 구축·통합
    * 엔터티·관계 병합 후처리와 분석·질의 기법
    * LLM을 활용한 KG 결과 분석
* 자료: 추후 공유


#### **2026년 8월 15일**: [Webex](#)

* 발표자 - 두균
* **Chapter 5. 비정형 데이터에서 도메인 특화 지식 추출 (Extracting Domain-Specific Knowledge from Unstructured Data)** — pp. 95–114
    * 비정형 데이터로부터 지식 그래프 구축
    * 록펠러 아카이브 센터 사례로 보는 아카이브 관리의 복잡성
    * LLM을 사용한 개체·관계 추출
* 자료: 추후 공유


#### **2026년 8월 22일**: [Webex](#)

* 발표자 - 종훈(L)
* **Chapter 6. LLM으로 지식 그래프 구축하기 (Building Knowledge Graphs with Large Language Models)** — pp. 115–128
    * 아카이브를 지식 그래프로 변환하는 그래프 모델링
    * 데이터 정규화·정제와 엔터티 해소(entity resolution)
    * 지적 네트워크 분석
* 자료: 추후 공유


#### **2026년 8월 29일**: [Webex](#)

* 발표자 - 재익
* **Chapter 7. 개체명 중의성 해소 (Named Entity Disambiguation)** — pp. 129–179
    * 개체명 인식(NER)에서 개체명 중의성 해소(NED)로
    * NED와 지식 그래프 기술의 결합
    * 여러 출처로부터의 KG 구축과 고급 분석
* 자료: 추후 공유


#### **2026년 9월 5일**: [Webex](#)

* 발표자 - 태호
* **Chapter 8. 오픈 LLM과 도메인 온톨로지를 활용한 NED (NED with Open LLMs and Domain Ontologies)** — pp. 180–206
    * 전통적인 NED 도구의 한계
    * 범용 LLM과 도메인 온톨로지를 결합한 중의성 해소
    * 최단 경로 탐지·경로-텍스트 변환 기반 다단계 중의성 해소
* 자료: 추후 공유


#### **2026년 9월 12일**: [Webex](#)

* 발표자 - 정훈
* **Chapter 9. 지식 그래프 위의 머신러닝 입문 (Machine Learning on Knowledge Graphs: A Primer Approach)** — pp. 207–232
    * 지식 그래프에서의 머신러닝 이해
    * 그래프에서 수행되는 일반적인 ML 과제
    * 노드·관계 표현의 역할
* 자료: 추후 공유


#### **2026년 9월 19일**: [Webex](#)

* 발표자 - 추후 확정
* **Chapter 10. 그래프 특성 공학: 수동·반자동 접근 (Graph Feature Engineering: Manual and Semiautomated Approaches)** — pp. 233–271
    * 노드·관계에 대한 수동 특성 공학 기법
    * 도메인 전문성과 반자동 추출의 결합
    * 특성 공학의 실제 응용
* 자료: 추후 공유


#### **2026년 9월 26일**: [Webex](#)

* 발표자 - 추후 확정
* **Chapter 11. 그래프 표현 학습과 그래프 신경망 (Graph Representation Learning and Graph Neural Networks)** — pp. 272–301
    * 그래프 표현 학습(GRL)과 그래프 ML의 확장
    * 딥러닝을 통한 특성 공학 자동화와 그래프 임베딩
    * 그래프 신경망(GNN) 기초
* 자료: 추후 공유


#### **2026년 10월 3일**: [Webex](#)

* 발표자 - 추후 확정
* **Chapter 12. GNN 기반 노드 분류와 링크 예측 (Node Classification and Link Prediction with GNNs)** — pp. 302–334
    * 실제 시나리오에서의 GNN 활용
    * 자금세탁방지(AML) 노드 분류 시스템 구축
    * 추천 시스템을 위한 링크 예측 시스템 구축
* 자료: 추후 공유


#### **2026년 10월 10일**: [Webex](#)

* 발표자 - 추후 확정
* **Chapter 13. 지식 그래프 기반 검색 증강 생성 (Knowledge Graph-Powered Retrieval-Augmented Generation)** — pp. 335–355
    * LLM을 AI 에이전트로서 유용하게 만드는 방법
    * 검색 증강 생성(RAG)을 통한 LLM 기반화(grounding)
    * KG 기반 RAG 시스템 구축
* 자료: 추후 공유


#### **2026년 10월 17일**: [Webex](#)

* 발표자 - 추후 확정
* **Chapter 14. 자연어로 지식 그래프에 질문하기 (Asking a KG Questions with Natural Language)** — pp. 356–396
    * 복잡한 시나리오에서 RAG의 한계
    * 도메인 전문성을 모방하는 고급 질의응답 시스템 구축
    * 쿼리 결과를 의미 있고 실행 가능한 요약으로 변환
* 자료: 추후 공유


#### **2026년 10월 24일**: [Webex](#)

* 발표자 - 추후 확정
* **Chapter 15. LangGraph로 QA 에이전트 구축 (Building a QA Agent with LangGraph)** — pp. 397–428
    * 전문가 모방 접근법 구현
    * LangGraph 오케스트레이션과 Streamlit 프런트엔드 기반 질의응답 조사 구현
    * 시스템 조정과 개선
* 자료: 추후 공유

---

### **5. 스터디 운영 방법**

* **교재**: [Knowledge Graphs and LLMs in Action (Alessandro Negro 외, Manning)](https://www.manning.com/books/knowledge-graphs-and-llms-in-action)
* **학습 자료**: [`source/Alessandro Negro - Knowledge Graphs and LLMs in Action`](source/Alessandro%20Negro%20-%20Knowledge%20Graphs%20and%20LLMs%20in%20Action) — 챕터별 원문 Markdown·한국어 번역본·PDF
* **학습 공유**: 매주 학습한 내용을 발표자료와 함께 GitHub에 공유
* **AI 에이전트 활용**: [Codex·Claude Code 시작 가이드](QUICKSTART.md) — 담당 날짜와 챕터를 알려주면 자료 탐색부터 HTML 발표자료 생성·검증까지 지원
* **웹 발표자료**: [`docs/studies`](docs/studies) — 스터디 종료 후 학습자료가 `archive/`로 이동해도 공개 발표 URL은 유지
* **HTML 산출물 템플릿**: [공통 청사진](agent-support/templates/STUDY_SESSION_BLUEPRINT.md) + [`study-report-v1`](agent-support/templates/study-report/DESIGN.md) + [`study-deck-v1`](agent-support/templates/study-deck/DESIGN.md) — Chapter 1 완성본의 상세 리포트·28장 리포트 파생 발표 흐름·자동 목차·표/도형·전체화면 이미지 줌을 기준으로, 에이전트가 원자료 → 리포트 품질 게이트 → 발표자료 순서로 생성
* **발표 방식**: 담당 범위에 대한 50분 발표 + 10분 Q&A (총 1시간 진행)
* **운영 규칙**: [스터디 운영 규칙](source/etc/%EC%8A%A4%ED%84%B0%EB%94%94_%EC%9A%B4%EC%98%81_%EA%B7%9C%EC%B9%99_v01.pdf)

---

### **6. 기타**

* **참가 희망 요청**: [Email](mailto:restful3@gmail.com)
* **아카이브**: [GitHub Archive](archive)
    * [Deep Agents 스터디 자료](source/deep_agents) - 2026년 5월 완료
    * [LangGraph 스터디 자료](source/langgraph) - 2026년 6\~7월 완료

---

### **7. 회의록**

* [2026-07-11 랭그래프 스터디 종료 및 차기 교재 논의](source/langgraph/%ED%9A%8C%EC%9D%98%EB%A1%9D/2026-07-11_%EB%9E%AD%EA%B7%B8%EB%9E%98%ED%94%84_%EC%8A%A4%ED%84%B0%EB%94%94_%EC%A2%85%EB%A3%8C_%EB%B0%8F_%EC%B0%A8%EA%B8%B0_%EA%B5%90%EC%9E%AC_%EB%85%BC%EC%9D%98.md)
