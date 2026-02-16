# 연구 어시스턴트 (Research Assistant)

도구(Tool)를 사용하는 ReAct 스타일의 연구 어시스턴트 예제입니다.

## 기능

- 웹 검색 (시뮬레이션)
- 수학 계산기
- 메모 저장/조회
- 현재 시간 확인

## 학습 포인트

이 예제에서 배울 수 있는 LangGraph 개념:

1. **Tool Calling**: @tool 데코레이터로 도구 정의
2. **ToolNode**: 도구 실행 노드
3. **ReAct 패턴**: 추론-행동-관찰 루프
4. **조건부 엣지**: 도구 호출 여부에 따른 분기

## 실행 방법

```bash
# 기본 데모 실행
python -m examples.02_research_assistant.main

# 인터랙티브 모드
python -m examples.02_research_assistant.main interactive
```

## 코드 구조

```
02_research_assistant/
├── main.py      # 메인 실행 파일
└── README.md    # 이 파일
```

## 핵심 코드 설명

### 1. 도구 정의

```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    # 검색 로직
    return f"검색 결과: {query}"
```

### 2. ReAct 그래프 구성

```python
graph = StateGraph(MessagesState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,  # 도구 호출이 있으면 tools, 없으면 END
    {"tools": "tools", "end": END}
)
graph.add_edge("tools", "agent")  # 순환 구조
```

### 3. 도구 호출 결정

```python
def should_continue(state: MessagesState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return "end"
```

## 그래프 구조

```
START → agent → (도구 필요?) → tools → agent → ... → END
                    ↓ (아니오)
                   END
```

## 확장 아이디어

- 실제 웹 검색 API 연동 (Google, Bing)
- 파일 읽기/쓰기 도구 추가
- 데이터베이스 쿼리 도구
- 이메일 전송 도구
- RAG (Retrieval Augmented Generation) 통합
