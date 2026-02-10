# Part 3: First Agent - 첫 번째 에이전트 만들기

> 📚 **학습 시간**: 약 2-3시간
> 🎯 **난이도**: ⭐⭐⭐☆☆ (중급)
> 📖 **공식 문서**: [06-agents.md](/official/06-agents.md), [03-quickstart.md](/official/03-quickstart.md)
> 📄 **교안 문서**: [part03_first_agent.md](/docs/part03_first_agent.md)
> 🎯 **미니 프로젝트**: [Weather Assistant](/projects/01_weather_assistant/)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] `create_agent()` API로 Agent 생성
- [x] 실전 Weather Agent 구현
- [x] ReAct 패턴의 원리 이해
- [x] Custom System Prompt 작성
- [x] Streaming Agent 구현

---

## 📚 개요

드디어 **첫 번째 Agent**를 만듭니다! Agent는 LLM을 "두뇌"로 사용하여 도구를 호출하고 문제를 해결하는 자율적인 시스템입니다.

**왜 중요한가?**
- Agent는 LangChain의 핵심 개념입니다
- ReAct 패턴은 현대 AI Agent의 표준입니다
- 실전 프로젝트로 이어지는 핵심 스킬입니다

**실무 활용 사례**
- 고객 지원 챗봇
- 데이터 분석 Assistant
- 자동화 워크플로우

---

## 📁 예제 파일

### 01_basic_agent.py
**난이도**: ⭐⭐☆☆☆ | **예상 시간**: 30분

LangChain의 `create_agent()` API를 사용해 첫 Agent를 만듭니다.

**학습 내용**:
- `create_agent()` 함수 사용
- 모델과 도구 연결
- Agent 실행 및 응답 처리
- 기본 Agent 구조 이해

**실행 방법**:
```bash
python 01_basic_agent.py
```

**주요 개념**:
- Agent = Model + Tools + Execution Loop
- `invoke()` 메서드로 Agent 실행
- 메시지 히스토리 관리

---

### 02_weather_agent.py
**난이도**: ⭐⭐☆☆☆ | **예상 시간**: 40분

실전 날씨 조회 Agent를 구현합니다. (공식 Quickstart 예제 기반)

**학습 내용**:
- 실전 도구 구현 (날씨 API 연동)
- 에러 핸들링
- 사용자 친화적 응답 생성
- 도구 실행 로그 확인

**실행 방법**:
```bash
python 02_weather_agent.py
```

**주요 개념**:
- 외부 API와 LLM 연결
- Agent의 추론 과정 관찰
- 실제 사용 가능한 Agent 구축

---

### 03_react_pattern.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 50분

ReAct (Reasoning + Acting) 패턴의 원리를 깊이 이해합니다.

**학습 내용**:
- ReAct 루프의 동작 원리
- Thought → Action → Observation 사이클
- Agent의 추론 과정 시각화
- Step-by-step 실행

**실행 방법**:
```bash
python 03_react_pattern.py
```

**주요 개념**:
- **Thought**: Agent의 내부 추론
- **Action**: 도구 호출 결정
- **Observation**: 도구 실행 결과
- 반복적 문제 해결

---

### 04_custom_prompt.py
**난이도**: ⭐⭐☆☆☆ | **예상 시간**: 35분

Agent의 성격과 행동을 제어하는 Custom System Prompt를 작성합니다.

**학습 내용**:
- System Prompt의 중요성
- Agent 페르소나 정의
- 응답 스타일 제어
- 제약 조건 설정

**실행 방법**:
```bash
python 04_custom_prompt.py
```

**주요 개념**:
- System Prompt = Agent의 "지시서"
- 명확한 역할 정의
- 출력 형식 제어

---

### 05_streaming_agent.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 45분

실시간 스트리밍으로 Agent의 응답을 받습니다.

**학습 내용**:
- `stream()` 메서드 사용
- 실시간 토큰 출력
- 중간 단계 스트리밍
- 프론트엔드 통합 준비

**실행 방법**:
```bash
python 05_streaming_agent.py
```

**주요 개념**:
- 사용자 경험 개선
- 긴 응답의 즉각적 피드백
- 각 단계별 진행 상황 표시

---

## 🎓 실습 과제

### 과제 1: 계산기 Agent (⭐⭐☆)

**목표**: 수학 계산을 수행하는 Agent를 만드세요.

**요구사항**:
1. 기본 사칙연산 도구 4개
2. "123 + 456을 계산해줘" 같은 자연어 질문에 응답
3. 복잡한 수식도 처리 (예: "(10 + 5) * 2")

**힌트**:
- `eval()` 사용 시 보안 주의!
- 안전한 계산을 위해 `ast.literal_eval()` 사용 고려

**해답**: [solutions/exercise_01.py](/src/part03_first_agent/solutions/exercise_01.py)

---

### 과제 2: 정보 검색 Agent (⭐⭐⭐)

**목표**: 인터넷 검색 기능을 가진 Agent를 만드세요.

**요구사항**:
1. Wikipedia 검색 도구 (또는 다른 검색 API)
2. 요약 생성 도구
3. "파이썬의 역사를 알려줘" 같은 질문에 응답

**해답**: [solutions/exercise_02.py](/src/part03_first_agent/solutions/exercise_02.py)

---

### 과제 3: 멀티스텝 Agent (⭐⭐⭐⭐)

**목표**: 여러 단계의 추론이 필요한 복잡한 작업을 수행하는 Agent를 만드세요.

**요구사항**:
1. 3개 이상의 도구
2. 도구를 순차적으로 사용하여 문제 해결
3. 각 단계의 추론 과정을 명확하게 출력

**예시 질문**: "서울과 부산의 날씨를 비교하고, 더 따뜻한 곳을 추천해줘"

**해답**: [solutions/exercise_03.py](/src/part03_first_agent/solutions/exercise_03.py)

---

## 💡 실전 팁

### Tip 1: 효과적인 System Prompt 작성

```python
SYSTEM_PROMPT = """
당신은 전문 날씨 상담 Assistant입니다.

역할:
- 사용자에게 정확한 날씨 정보 제공
- 간결하고 친절한 답변
- 필요시 옷차림 조언 포함

제약사항:
- 날씨 정보가 없으면 추측하지 마세요
- 항상 도구를 사용하여 최신 정보를 확인하세요

응답 형식:
1. 현재 날씨 요약
2. 세부 정보 (온도, 습도 등)
3. 한줄 조언
"""
```

### Tip 2: 도구 실행 로깅

Agent의 동작을 이해하려면 로깅이 중요합니다:

```python
import langchain
langchain.debug = True  # 상세 로그 활성화

# 또는
from langchain.callbacks import StdOutCallbackHandler

agent = create_agent(
    model=model,
    tools=tools,
    callbacks=[StdOutCallbackHandler()]
)
```

### Tip 3: 에러 핸들링

```python
try:
    response = agent.invoke({"messages": [user_message]})
except Exception as e:
    print(f"Agent 실행 중 오류: {e}")
    # 폴백 응답 또는 재시도 로직
```

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: Agent가 무한 루프에 빠졌어요</summary>

**A**: ReAct 루프가 수렴하지 않을 수 있습니다:
1. **Max iterations 설정**: Agent의 최대 반복 횟수 제한
2. **더 나은 도구 설명**: LLM이 도구 사용을 명확히 이해하도록
3. **System Prompt 개선**: "최대 3단계 안에 답변하세요" 등의 지시

```python
from langgraph.prebuilt import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    max_iterations=5  # 최대 5번 반복
)
```
</details>

<details>
<summary>Q2: Agent가 불필요한 도구를 계속 호출해요</summary>

**A**:
1. **System Prompt 개선**: "이미 정보가 충분하면 바로 답변하세요"
2. **도구 설명 명확화**: 언제 사용해야 하는지 명시
3. **Few-shot 예시 추가**: 올바른 사용 패턴 보여주기
</details>

<details>
<summary>Q3: Streaming이 작동하지 않아요</summary>

**A**:
- 모든 모델이 streaming을 지원하는 것은 아닙니다
- OpenAI, Anthropic은 대부분 지원
- 로컬 모델은 확인 필요

```python
# Streaming 지원 확인
if hasattr(model, 'stream'):
    for chunk in agent.stream(input):
        print(chunk)
```
</details>

---

## 🚀 미니 프로젝트

### Project 1: Weather Assistant

이제 배운 내용을 종합하여 완전한 날씨 비서를 만들어보세요!

**프로젝트 링크**: [Weather Assistant](/projects/01_weather_assistant/)

**주요 기능**:
- 도시별 현재 날씨 조회
- 일주일 예보
- 여행 추천
- 옷차림 조언

**예상 소요 시간**: 2-3시간
**난이도**: ⭐⭐⭐☆☆

---

## 🔗 심화 학습

1. **공식 문서 심화**
   - [06-agents.md](/official/06-agents.md) - Agent 전체 가이드
   - [03-quickstart.md](/official/03-quickstart.md) - 빠른 시작 예제

2. **관련 논문**
   - [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
   - [WebGPT: Browser-assisted question-answering](https://arxiv.org/abs/2112.09332)

3. **커뮤니티 리소스**
   - [LangChain Agent 갤러리](https://python.langchain.com/docs/use_cases/agent_applications/)
   - [Agent 디버깅 가이드](https://python.langchain.com/docs/how_to/debugging/)

4. **다음 단계**
   - [Part 4: Memory Systems](/src/part04_memory/README.md) - 대화 기억하기

---

## ✅ 체크리스트

Part 3을 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다 (5개)
- [ ] 실습 과제를 완료했다 (3개)
- [ ] `create_agent()`의 사용법을 이해했다
- [ ] ReAct 패턴의 동작 원리를 설명할 수 있다
- [ ] Custom System Prompt를 작성할 수 있다
- [ ] Streaming Agent를 구현할 수 있다
- [ ] Weather Assistant 프로젝트를 시작했다

---

**이전**: [← Part 2 - Fundamentals](/src/part02_fundamentals/README.md)
**다음**: [Part 4 - Memory Systems로 이동](/src/part04_memory/README.md) →

---

**학습 진도**: ▓▓▓░░░░░░░ 30% (Part 3/10 완료)

*마지막 업데이트: 2025-02-06*
