# Part 4: Production 연습 문제

> 프로덕션 기능을 익히는 연습 문제입니다.

---

## 문제 1: Checkpointer 기본 ⭐⭐

### 설명
MemorySaver를 사용하여 상태를 저장하고 복구하는 그래프를 만드세요.

### 요구사항
1. 3단계 처리 그래프 구현
2. 2단계 후 상태 저장
3. 새 세션에서 저장된 상태 로드

### 테스트
```python
# 첫 번째 실행 (thread_id: "session_1")
result1 = app.invoke(input, config)

# 두 번째 실행 (같은 thread_id)
result2 = app.invoke(None, config)  # 이어서 실행
```

---

## 문제 2: 메시지 관리 ⭐⭐⭐

### 설명
긴 대화에서 메시지를 효율적으로 관리하는 그래프를 만드세요.

### 요구사항
1. 메시지가 10개 초과하면 오래된 메시지 삭제
2. System 메시지는 항상 유지
3. 삭제 시 `RemoveMessage` 사용

### 힌트
```python
# RemoveMessage 사용법
return {"messages": [RemoveMessage(id=msg.id) for msg in old_msgs]}
```

---

## 문제 3: Human-in-the-Loop ⭐⭐⭐

### 설명
특정 조건에서 사용자 승인을 요청하는 그래프를 만드세요.

### 요구사항
1. 금액이 100만원 이상이면 `interrupt()`로 승인 요청
2. 승인되면 계속, 거부되면 취소
3. `Command(resume=...)` 으로 재개

### State
```python
class ApprovalState(TypedDict):
    amount: int
    approved: Optional[bool]
    result: str
```

---

## 문제 4: 스트리밍 구현 ⭐⭐⭐

### 설명
다양한 스트리밍 모드를 테스트하는 코드를 작성하세요.

### 요구사항
1. `values` 모드: 전체 상태 스트리밍
2. `updates` 모드: 노드별 업데이트만
3. `messages` 모드: 메시지만 스트리밍

### 테스트 코드
```python
# 각 모드별로 출력 비교
for chunk in app.stream(input, config, stream_mode="values"):
    print(f"[values] {chunk}")

for chunk in app.stream(input, config, stream_mode="updates"):
    print(f"[updates] {chunk}")
```

---

## 문제 5: Time Travel ⭐⭐⭐⭐

### 설명
상태 히스토리를 탐색하고 특정 시점으로 복원하는 기능을 구현하세요.

### 요구사항
1. 5단계 처리 그래프 실행
2. `get_state_history()`로 히스토리 조회
3. 3단계 시점으로 복원
4. 다른 경로로 재실행 (Fork)

### 힌트
```python
# 히스토리에서 특정 checkpoint 찾기
for snapshot in app.get_state_history(config):
    if snapshot.values.get("step") == 3:
        # 이 시점에서 fork
        pass
```

---

## 해답

해답은 [solutions/part4_solutions.py](./solutions/part4_solutions.py)를 참조하세요.
