# 간단한 챗봇 (Simple Chatbot)

메모리 기능이 있는 간단한 대화형 챗봇 예제입니다.

## 기능

- 대화 기록 유지 (세션별 메모리)
- 스트리밍 응답 지원
- 인터랙티브 모드

## 학습 포인트

이 예제에서 배울 수 있는 LangGraph 개념:

1. **MessagesState**: 메시지 기반 상태 관리
2. **Checkpointer**: 대화 기록 저장 및 복원
3. **Thread ID**: 세션별 대화 분리
4. **Streaming**: 실시간 응답 스트리밍

## 실행 방법

```bash
# 기본 데모 실행
python -m examples.01_simple_chatbot.main

# 인터랙티브 모드
python -m examples.01_simple_chatbot.main interactive

# 스트리밍 예제
python -m examples.01_simple_chatbot.main stream
```

## 코드 구조

```
01_simple_chatbot/
├── main.py      # 메인 실행 파일
└── README.md    # 이 파일
```

## 핵심 코드 설명

### 1. 그래프 구성

```python
from langgraph.graph import StateGraph, MessagesState

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
```

### 2. 메모리 설정

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
```

### 3. 세션 관리

```python
# 각 세션은 고유한 thread_id를 가짐
config = {"configurable": {"thread_id": "session_1"}}
result = app.invoke({"messages": [HumanMessage(content="안녕!")]}, config=config)
```

## 확장 아이디어

- 시스템 프롬프트 커스터마이징
- 대화 내보내기/가져오기
- 다중 사용자 지원
- 웹 인터페이스 추가
