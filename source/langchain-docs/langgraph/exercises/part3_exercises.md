# Part 3: Agent 연습 문제

> Tool Calling과 Agent 패턴을 익히는 연습 문제입니다.

---

## 문제 1: 간단한 도구 정의 ⭐⭐

### 설명
`@tool` 데코레이터를 사용하여 세 가지 도구를 정의하세요.

### 요구사항
1. `add(a: int, b: int)`: 덧셈
2. `multiply(a: int, b: int)`: 곱셈
3. `get_length(text: str)`: 문자열 길이

### 힌트
- docstring이 도구 설명이 됨
- 타입 힌트 필수

---

## 문제 2: ReAct Agent 기본 ⭐⭐⭐

### 설명
날씨 조회 도구를 사용하는 ReAct Agent를 만드세요.

### 요구사항
1. `get_weather(city: str)` 도구 정의
2. ReAct 루프 구현 (Reason -> Act -> Observe)
3. 최대 3회 반복 후 응답

### State 구조
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: list
    iterations: int
```

---

## 문제 3: Multi-Tool Agent ⭐⭐⭐

### 설명
여러 도구를 상황에 맞게 선택하는 Agent를 만드세요.

### 도구 목록
1. `search(query: str)`: 검색
2. `calculator(expression: str)`: 계산
3. `translator(text: str, target_lang: str)`: 번역

### 테스트 케이스
- "파이썬이란 무엇인가?" → search
- "123 * 456은?" → calculator
- "Hello를 한국어로" → translator

---

## 문제 4: Supervisor Agent ⭐⭐⭐⭐

### 설명
다른 Agent를 관리하는 Supervisor를 구현하세요.

### 요구사항
1. `research_agent`: 정보 조사
2. `writer_agent`: 글 작성
3. `supervisor`: 작업 분배 및 결과 검토

### 플로우
```
User Request
     ↓
[Supervisor] ─→ [Research Agent] ─→ [Writer Agent]
     ↑___________________|__________________|
```

---

## 문제 5: 서브그래프 활용 ⭐⭐⭐⭐

### 설명
복잡한 작업을 서브그래프로 캡슐화하세요.

### 요구사항
1. `data_pipeline` 서브그래프: 수집 → 정제 → 변환
2. `analysis` 서브그래프: 통계 → 시각화
3. 부모 그래프에서 두 서브그래프 연결

### 힌트
- 서브그래프도 `StateGraph`로 정의
- `add_node`에 컴파일된 서브그래프 전달

---

## 해답

해답은 [solutions/part3_solutions.py](./solutions/part3_solutions.py)를 참조하세요.
