# 멀티 에이전트 팀 (Multi-Agent Team)

여러 전문 에이전트가 협업하는 시스템 예제입니다.

## 에이전트 구성

- **Supervisor**: 작업 분배 및 조율
- **Researcher**: 정보 수집 및 분석
- **Writer**: 콘텐츠 작성
- **Critic**: 품질 검토 및 피드백

## 워크플로우

```
요청 → Supervisor → Researcher → Supervisor → Writer → Supervisor → Critic
                                                   ↑                    ↓
                                                   └──── (수정 필요) ────┘
                                                              ↓ (승인)
                                                            완료
```

## 학습 포인트

이 예제에서 배울 수 있는 LangGraph 개념:

1. **Multi-Agent 시스템**: 여러 에이전트 협업
2. **Supervisor 패턴**: 중앙 조율자
3. **조건부 라우팅**: 상태에 따른 동적 분기
4. **피드백 루프**: 반복적 개선

## 실행 방법

```bash
# 기본 데모 실행
python -m examples.03_multi_agent_team.main

# 인터랙티브 모드
python -m examples.03_multi_agent_team.main interactive
```

## 코드 구조

```
03_multi_agent_team/
├── main.py      # 메인 실행 파일
└── README.md    # 이 파일
```

## 핵심 코드 설명

### 1. State 정의

```python
class TeamState(TypedDict):
    request: str           # 원본 요청
    current_agent: str     # 현재 단계
    research_result: str   # 리서치 결과
    draft: str             # 작성된 초안
    feedback: str          # 피드백
    final_output: str      # 최종 결과물
    history: Annotated[List[str], operator.add]  # 작업 히스토리
    revision_count: int    # 수정 횟수
```

### 2. Supervisor 라우팅

```python
def route_to_agent(state: TeamState) -> str:
    current = state.get("current_agent", "")

    if current == "researcher":
        return "researcher"
    elif current == "writer":
        return "writer"
    # ...
```

### 3. 피드백 루프

```python
# Supervisor에서 피드백 처리
if "수정 필요" in feedback and revision_count < 2:
    return {
        "current_agent": "writer",
        "revision_count": revision_count + 1,
    }
else:
    return {"current_agent": "done"}
```

## 확장 아이디어

- 더 많은 전문 에이전트 추가 (Editor, Fact-checker 등)
- 병렬 에이전트 실행
- 에이전트별 도구 할당
- 작업 우선순위 시스템
- 에이전트 간 직접 통신
