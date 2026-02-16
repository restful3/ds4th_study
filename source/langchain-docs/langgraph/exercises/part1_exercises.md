# Part 1: Foundation 연습 문제

> State, Node, Edge, Reducer의 기초를 다지는 연습 문제입니다.

---

## 문제 1: 간단한 카운터 그래프 ⭐

### 설명
정수 값을 증가, 감소, 두 배로 만드는 세 개의 노드를 가진 그래프를 만드세요.

### 요구사항
- `increment` 노드: 값을 1 증가
- `decrement` 노드: 값을 1 감소
- `double` 노드: 값을 2배로

### 입력/출력
```python
# 입력: {"value": 5}
# 실행 순서: increment -> double -> decrement
# 출력: {"value": 11}  # (5+1)*2-1 = 11
```

### 힌트
- `TypedDict`로 State 정의
- `add_edge`로 순차 연결

---

## 문제 2: 조건부 인사 그래프 ⭐⭐

### 설명
시간대에 따라 다른 인사말을 반환하는 그래프를 만드세요.

### 요구사항
- 입력: 시간 (0-23)
- 6-11시: "좋은 아침입니다!"
- 12-17시: "좋은 오후입니다!"
- 18-21시: "좋은 저녁입니다!"
- 그 외: "안녕하세요!"

### State 구조
```python
class GreetingState(TypedDict):
    hour: int
    greeting: str
```

### 힌트
- `add_conditional_edges` 사용
- 라우터 함수에서 시간대 판별

---

## 문제 3: 메시지 누적 그래프 ⭐⭐

### 설명
여러 노드에서 메시지를 누적하는 그래프를 만드세요. `add_messages` reducer를 사용합니다.

### 요구사항
- 세 개의 노드가 각각 메시지를 추가
- 모든 메시지가 순서대로 누적

### 예상 결과
```python
# 최종 messages: [
#   AIMessage(content="Step 1 완료"),
#   AIMessage(content="Step 2 완료"),
#   AIMessage(content="Step 3 완료")
# ]
```

### 힌트
- `MessagesState` 상속 또는 `Annotated[list, add_messages]` 사용

---

## 문제 4: 커스텀 Reducer ⭐⭐⭐

### 설명
중복을 제거하면서 리스트를 병합하는 커스텀 Reducer를 만드세요.

### 요구사항
```python
# 예: current = ["a", "b"], new = ["b", "c"]
# 결과: ["a", "b", "c"]
```

### 힌트
- `set`을 활용한 중복 제거
- 순서 유지 필요

---

## 문제 5: 입출력 스키마 분리 ⭐⭐⭐

### 설명
입력 스키마와 출력 스키마가 다른 그래프를 만드세요.

### 요구사항
- 입력: `{"raw_data": "hello world"}`
- 출력: `{"word_count": 2, "char_count": 11}`

### 힌트
- 내부 State는 모든 필드 포함
- `input` / `output` 파라미터 활용

---

## 해답

해답은 [solutions/part1_solutions.py](./solutions/part1_solutions.py)를 참조하세요.
