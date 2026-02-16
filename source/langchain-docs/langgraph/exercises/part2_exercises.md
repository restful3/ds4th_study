# Part 2: Workflows 연습 문제

> 워크플로우 패턴을 익히는 연습 문제입니다.

---

## 문제 1: 텍스트 처리 파이프라인 ⭐⭐

### 설명
텍스트를 단계별로 처리하는 파이프라인을 만드세요.

### 요구사항
1. `clean` 노드: 공백 정리, 소문자 변환
2. `tokenize` 노드: 단어 리스트로 분리
3. `count` 노드: 단어 수 계산

### State 구조
```python
class TextState(TypedDict):
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    word_count: int
```

---

## 문제 2: 스마트 라우터 ⭐⭐

### 설명
입력 유형에 따라 다른 처리 경로로 라우팅하는 그래프를 만드세요.

### 요구사항
- 숫자만 포함: `number_handler`로 라우팅
- 이메일 형식: `email_handler`로 라우팅
- URL 형식: `url_handler`로 라우팅
- 그 외: `text_handler`로 라우팅

### 힌트
- 정규표현식 활용
- `add_conditional_edges` 사용

---

## 문제 3: 병렬 데이터 수집기 ⭐⭐⭐

### 설명
여러 소스에서 동시에 데이터를 수집하고 통합하는 그래프를 만드세요.

### 요구사항
1. `source_a`, `source_b`, `source_c` 노드가 병렬 실행
2. `aggregator` 노드가 모든 결과를 통합

### 힌트
- Fan-out / Fan-in 패턴
- `Send` API 또는 여러 노드에서 같은 타겟으로 연결

---

## 문제 4: Orchestrator-Worker 구현 ⭐⭐⭐

### 설명
작업을 분배하고 결과를 수집하는 Orchestrator-Worker 패턴을 구현하세요.

### 요구사항
1. `orchestrator`: 작업 목록을 worker에게 분배
2. `worker`: 개별 작업 처리
3. `collector`: 모든 결과 수집

### 예시
```python
# 입력: {"tasks": ["task1", "task2", "task3"]}
# 각 task가 worker에서 처리됨
# 출력: {"results": ["result1", "result2", "result3"]}
```

---

## 문제 5: Evaluator-Optimizer 루프 ⭐⭐⭐⭐

### 설명
결과를 평가하고 기준 미달이면 다시 시도하는 루프를 구현하세요.

### 요구사항
1. `generator`: 결과 생성
2. `evaluator`: 품질 평가 (0-100점)
3. 80점 이상이면 통과, 아니면 재시도
4. 최대 3회 시도

### State 구조
```python
class EvalState(TypedDict):
    prompt: str
    result: str
    score: int
    attempts: int
```

---

## 해답

해답은 [solutions/part2_solutions.py](./solutions/part2_solutions.py)를 참조하세요.
