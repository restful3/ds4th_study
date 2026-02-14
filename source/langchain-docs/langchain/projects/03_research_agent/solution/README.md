# Research Agent 솔루션 가이드

## 구현 핵심 포인트

### 1. 멀티에이전트 아키텍처
- **Planner Agent**: 연구 주제를 하위 질문으로 분해
- **Searcher Agent**: 웹 검색으로 정보 수집
- **Analyst Agent**: 수집된 정보를 분석 및 요약
- **Writer Agent**: 최종 보고서 작성

### 2. 에이전트 간 협업 패턴
- Subagents 패턴으로 순차 실행
- 각 에이전트의 출력이 다음 에이전트의 입력이 됨

### 3. 주요 학습 포인트
- `create_agent()`로 각 역할별 에이전트 생성
- 에이전트 간 데이터 전달 구조 설계
- 에러 처리 및 재시도 로직

## 참고 자료

- [Part 7: 멀티에이전트](/docs/part07_multi_agent.md) - 멀티에이전트 패턴
- [Part 2: 도구](/docs/part02_fundamentals.md) - 도구 정의
- [23-subagents.md](/official/23-subagents.md) - Subagents 공식 문서
