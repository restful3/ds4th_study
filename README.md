# **ds4th study**

---

### **1. 스터디 목적**

* **Sebastian Raschka의 "Build a Large Language Model (From Scratch)" 책을 통해 대형 언어 모델(LLM)의 내부 작동 원리를 완전히 이해하고, GPT 유사 모델을 처음부터 구현하는 실무 능력을 강화하는 것**을 목표로 한다.

---

### **2. 스터디 시간**

* 매주 토요일 오전 9시부터 1시간

---

### **3. 스터디 장소**

* Webex

---

### **4. 스터디 운영 계획 (2025년 9월 ~ 2025년 11월)**

---

#### **2025년 9월 13일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=m8a945c481103c4a418d4670b103f74be	)

* 발표자 - C성진
  * **Chapter 1:** Understanding Large Language Models
    * LLM 기본 개념 및 트랜스포머 아키텍처 소개
    * 사전 훈련과 파인튜닝 단계 이해
    * GPT 시리즈 모델의 발전 과정

#### **2025년 9월 20일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=m7ce242843e71db115e8b2369986a04aa	)

* 발표자 - 경연
  * **Chapter 2:** Working with Text Data
    * 텍스트 토큰화 및 토큰 ID 변환
    * 바이트 페어 인코딩(BPE) 구현
    * 토큰 임베딩과 위치 인코딩 생성
    * 슬라이딩 윈도우를 통한 훈련 데이터 생성

* 발표자 - K성진, 태호
  * **Kaggle**
    * MAP - Charting Student Math Misunderstandings(태호)
    * ARC Prize 2025 (K성진)
       
#### **2025년 9월 27일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=m794c4e50f5ff0e9aa332e21523485655	)

* 발표자 - S종훈
  * **Chapter 3:** Coding Attention Mechanisms
    * 셀프 어텐션 메커니즘 구현
    * 인과적 어텐션 마스크 적용
    * 멀티헤드 어텐션 구조 구축
    * 드롭아웃을 통한 정규화
   
* 발표자 - 재익
  * **Kaggle**
    * ARC Prize 2025

#### **✅ 2025년 10월 4일**: 휴일 (추석 연휴 - 스터디 없음)

#### **✅ 2025년 10월 11일**: 휴일 (추석 연휴 - 스터디 없음)

#### **2025년 10월 18일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=me926d907650c9b1a34a5cf2a56fbea7c	)

* 발표자 - 민호
  * **Chapter 4:** Implementing a GPT Model from Scratch to Generate Text
    * GPT 아키텍처 전체 구현
    * 레이어 정규화 및 피드포워드 네트워크
    * 트랜스포머 블록 조립
    * 텍스트 생성 및 디코딩 전략
   
* 발표자 - 영재
  * **Kaggle**
    * ARC Prize 2025

#### **2025년 10월 25일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=m18b99f88356090f9f2ce9d29adefecc3	)

* 발표자 - 우석
  * **Chapter 5:** Pretraining on Unlabeled Data
    * 모델 성능 평가 지표 구현
    * 훈련 루프 및 검증 프로세스
    * OpenAI 사전 훈련 가중치 로드
    * 온도 스케일링 및 top-k 샘플링
    * 모델 저장 및 로드 방법
   
* 발표자 - K성진, 태호
  * **Kaggle**
    * ARC Prize 2025


#### **2025년 11월 1일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=m54969345967403cc4743ce696daeb9f0	)

* 발표자 - 태호
  * **Chapter 6:** Finetuning for Text Classification
    * 분류 작업을 위한 모델 헤드 수정
    * 파인튜닝 데이터셋 준비 방법
    * 다양한 파인튜닝 전략 비교
    * 성능 평가 및 결과 분석
    * 스팸 분류기 구현 실습

#### **2025년 11월 8일**: [Webex](https://lgehq.webex.com/lgehq-en/j.php?MTID=me13617ac007993e928d591c2d3dc036e	)

* 발표자 - 태영
  * **Chapter 7:** Finetuning to Follow Instructions
    * 지시사항 파인튜닝 데이터셋 구성
    * 인간 피드백 학습(RLHF) 개념
    * 직접 선호도 최적화(DPO) 구현
    * 모델 정렬 및 안전성 확보
    * 대화형 AI 구축

---

### **5. 스터디 운영 방법**

* **주교재**:
  * [Build a Large Language Model (From Scratch) - Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  * **공식 GitHub 저장소**: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

* **참고 자료**:
  * [Hands-On Large Language Models - Jay Alammar & Maarten Grootendorst](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)
  * [LLM을 활용한 실전 AI 애플리케이션 개발 - 허정준](https://github.com/onlybooks/llm)

* **학습 공유**: 매주 학습한 내용을 발표자료와 함께 GitHub에 공유
* **발표 방식**:
  * 각 챕터에 대한 50분 발표 + 10분 Q&A
  * 이론 설명과 실제 구현 코드 시연
  * 참고 자료의 관련 내용도 함께 다룰 수 있음

* **운영 규칙**:
  * 스터디 운영 규칙 (별도 문서 참조)

---

### **6. 학습 목표 및 성과물**

* **기초 단계 (9월)**: LLM 기본 개념과 텍스트 처리 파이프라인 이해
* **핵심 단계 (10월)**: 어텐션 메커니즘과 GPT 모델 완전 구현  
* **응용 단계 (11월)**: 모델 훈련, 파인튜닝 및 실용화

* **최종 성과물**: 
  * 처음부터 구현한 완전한 GPT 유사 모델
  * 개인별 특화된 LLM 프로젝트
  * 학습 과정과 구현 코드가 체계적으로 정리된 GitHub 저장소

---

### **7. 기타**

* **참가 희망 요청**: [Email](mailto:restful3@gmail.com)
* **이제까지 다룬 내용**: [archive 확인](https://github.com/restful3/ds4th_study/tree/main/archive)
* **교재 구매 링크**: 
  * [Manning 주교재](https://www.manning.com/books/build-a-large-language-model-from-scratch)
  * [O'Reilly 참고서](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)
  * [국내서 참고자료](https://github.com/onlybooks/llm)


---







