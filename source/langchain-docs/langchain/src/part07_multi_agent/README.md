# Part 7: Multi-Agent Systems - 여러 Agent의 협업

> 📚 **학습 시간**: 약 4-5시간
> 🎯 **난이도**: ⭐⭐⭐⭐⭐ (전문가)
> 📖 **공식 문서**: [22-multi-agent.md](/official/22-multi-agent.md), [23-subagents.md](/official/23-subagents.md), [24-handoffs.md](/official/24-handoffs.md), [25-skills.md](/official/25-skills.md), [26-router.md](/official/26-router.md), [27-custom-workflow.md](/official/27-custom-workflow.md)
> 📄 **교안 문서**: [part07_multi_agent.md](/docs/part07_multi_agent.md)
> 🎯 **미니 프로젝트**: [Research Agent](/projects/03_research_agent/)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] 멀티에이전트 시스템의 필요성 이해
- [x] Subagents 패턴 구현 (동기/비동기)
- [x] Handoffs 패턴으로 제어 전달
- [x] Skills 패턴으로 온디맨드 Agent 로딩
- [x] Router 패턴으로 입력 분류
- [x] LangGraph로 복잡한 워크플로우 구축

---

## 📚 개요

복잡한 문제는 **여러 Agent의 협업**으로 해결합니다. 각 Agent는 전문 분야를 담당하고, 협력하여 더 큰 목표를 달성합니다.

**왜 중요한가?**
- 단일 Agent의 한계 극복
- 전문화와 분업으로 성능 향상
- 복잡한 워크플로우 구현

**실무 활용 사례**
- 고객 서비스 (라우팅 → 전문 Agent)
- 리서치 시스템 (검색 → 분석 → 요약)
- 소프트웨어 개발 (계획 → 코딩 → 테스트)

---

## 📁 예제 파일

### 01_why_multi_agent.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 30분

멀티에이전트가 필요한 이유와 주요 패턴을 비교합니다.

**학습 내용**:
- 단일 Agent의 한계
- 멀티에이전트 패턴 개요
- 각 패턴의 장단점
- 언제 어떤 패턴을 사용할지

**실행 방법**:
```bash
python 01_why_multi_agent.py
```

**주요 개념**:
- **Subagents**: Agent가 다른 Agent를 도구로 사용
- **Handoffs**: Agent 간 제어 전달
- **Router**: 입력에 따라 적절한 Agent 선택

---

### 02_subagents_basic.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 50분

Subagents 패턴을 구현합니다 (동기 실행).

**학습 내용**:
- Agent를 도구로 래핑
- 메인 Agent → Sub Agent 호출
- 결과 통합
- 중첩 Agent 구조

**실행 방법**:
```bash
python 02_subagents_basic.py
```

**주요 개념**:
- Sub Agent = 전문가 Agent
- 메인 Agent = 조정자 (Coordinator)
- 순차적 실행

---

### 03_subagents_async.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

비동기 Subagents로 병렬 실행을 구현합니다.

**학습 내용**:
- 여러 Sub Agent 동시 실행
- 결과 대기 및 통합
- 에러 핸들링
- 성능 향상

**실행 방법**:
```bash
python 03_subagents_async.py
```

**주요 개념**:
- 병렬 실행으로 속도 향상
- `asyncio` 활용
- 독립적인 작업에 적합

---

### 04_handoffs.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

Handoffs 패턴으로 Agent 간 제어를 전달합니다.

**학습 내용**:
- Transfer Tool 사용
- Agent 간 컨텍스트 전달
- 대화 이어받기
- 순환 방지

**실행 방법**:
```bash
python 04_handoffs.py
```

**주요 개념**:
- Agent A → Agent B로 전달
- 대화 히스토리 유지
- 명시적 제어 흐름

---

### 05_skills_pattern.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 55분

Skills 패턴으로 필요할 때만 Agent를 로딩합니다.

**학습 내용**:
- Lazy Loading
- Skill Registry
- 온디맨드 Agent 생성
- 리소스 최적화

**실행 방법**:
```bash
python 05_skills_pattern.py
```

**주요 개념**:
- 모든 Agent를 항상 로딩하지 않음
- 필요 시 동적 로딩
- 메모리 및 비용 절약

---

### 06_router_pattern.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 55분

Router 패턴으로 입력을 분류하여 적절한 Agent에 전달합니다.

**학습 내용**:
- 입력 분류 (Classification)
- 라우팅 로직
- 여러 전문 Agent 관리
- 폴백 처리

**실행 방법**:
```bash
python 06_router_pattern.py
```

**주요 개념**:
- 고객 지원: 문의 타입별 Agent
- 멀티도메인 시스템
- 효율적 리소스 활용

---

### 07_custom_workflow.py
**난이도**: ⭐⭐⭐⭐⭐ | **예상 시간**: 90분

LangGraph로 복잡한 멀티에이전트 워크플로우를 구축합니다.

**학습 내용**:
- StateGraph 정의
- Node와 Edge 설정
- 조건부 라우팅
- 상태 관리

**실행 방법**:
```bash
python 07_custom_workflow.py
```

**주요 개념**:
- 그래프 기반 워크플로우
- 유연한 제어 흐름
- 복잡한 협업 패턴

---

## 🎓 실습 과제

### 과제 1: 이중 전문가 시스템 (⭐⭐⭐⭐)

**목표**: 두 명의 전문가 Agent가 협력하는 시스템을 만드세요.

**요구사항**:
1. **검색 전문가**: 인터넷 검색 담당
2. **요약 전문가**: 검색 결과 요약 담당
3. Subagents 패턴 사용
4. "파이썬의 최신 트렌드는?" 질문에 응답

**해답**: [solutions/exercise_01.py](/src/part07_multi_agent/solutions/exercise_01.py)

---

### 과제 2: 고객 서비스 라우터 (⭐⭐⭐⭐)

**목표**: 문의 타입별로 전문 Agent에 라우팅하는 시스템을 만드세요.

**요구사항**:
1. Router Agent: 문의 분류
2. **기술 지원** Agent
3. **환불** Agent
4. **일반 문의** Agent
5. 각 Agent는 다른 도구와 프롬프트 사용

**해답**: [solutions/exercise_02.py](/src/part07_multi_agent/solutions/exercise_02.py)

---

### 과제 3: 리서치 파이프라인 (⭐⭐⭐⭐⭐)

**목표**: 복잡한 리서치 작업을 수행하는 멀티에이전트 시스템을 만드세요.

**요구사항**:
1. **Planner** Agent: 리서치 계획 수립
2. **Searcher** Agent: 정보 수집 (병렬)
3. **Analyst** Agent: 데이터 분석
4. **Writer** Agent: 최종 보고서 작성
5. LangGraph로 워크플로우 구축

**예시 질문**: "인공지능의 윤리적 이슈에 대한 보고서를 작성해줘"

**해답**: [solutions/exercise_03.py](/src/part07_multi_agent/solutions/exercise_03.py)

---

## 💡 실전 팁

### Tip 1: 패턴 선택 가이드

```python
# Subagents: 전문가 조합, 명확한 분업
if task_requires_multiple_experts:
    use_subagents_pattern()

# Handoffs: 순차적 처리, 컨텍스트 전달
if agents_need_conversation_context:
    use_handoffs_pattern()

# Router: 입력 타입별 처리
if multiple_domains_or_types:
    use_router_pattern()

# Skills: 동적 로딩, 리소스 최적화
if many_agents_not_all_needed:
    use_skills_pattern()

# Custom Workflow: 복잡한 로직
if complex_conditional_flow:
    use_langgraph()
```

### Tip 2: Agent 간 통신

```python
# 방법 1: 상태 공유 (LangGraph)
class SharedState(TypedDict):
    messages: list
    research_data: dict
    current_phase: str

# 방법 2: Tool 결과 전달 (Subagents)
@tool
def call_expert_agent(query: str) -> str:
    """전문가 Agent 호출"""
    result = expert_agent.invoke({"messages": [query]})
    return result["messages"][-1].content
```

### Tip 3: 성능 최적화

```python
import asyncio

# 병렬 실행으로 속도 향상
async def parallel_research(topics: list):
    tasks = [
        search_agent.ainvoke({"topic": topic})
        for topic in topics
    ]
    results = await asyncio.gather(*tasks)
    return results
```

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: Subagents vs Handoffs 차이는?</summary>

**A**:
- **Subagents**: 메인 Agent가 Sub Agent를 **도구처럼** 호출
  - 결과를 받아서 계속 진행
  - 메인 Agent가 전체 제어

- **Handoffs**: Agent가 다른 Agent에게 **제어를 넘김**
  - 대화가 이어짐
  - 명시적 전환

```python
# Subagent
main_agent → sub_agent → main_agent

# Handoff
agent_A → agent_B (제어 종료)
```
</details>

<details>
<summary>Q2: 언제 멀티에이전트를 사용하나요?</summary>

**A**: 다음 상황에서 고려하세요:
- 문제가 명확하게 분리 가능
- 각 부분에 전문성 필요
- 병렬 처리로 성능 향상 가능
- 유지보수성 향상 (모듈화)

**사용하지 말아야 할 때**:
- 간단한 작업 (오버엔지니어링)
- Agent 간 통신 비용이 큼
- 복잡도 증가가 이득보다 큼
</details>

<details>
<summary>Q3: LangGraph는 언제 필요한가요?</summary>

**A**: 다음 경우에 사용:
- 복잡한 조건부 흐름
- 루프나 순환 필요
- 상태 관리가 복잡
- 시각화 및 디버깅 필요

**간단한 경우**는 기본 패턴(Subagents, Handoffs)으로 충분합니다.
</details>

---

## 🚀 미니 프로젝트

### Project 3: Research Agent System

멀티에이전트 협업으로 완전한 리서치 시스템을 구축하세요!

**프로젝트 링크**: [Research Agent](/projects/03_research_agent/)

**주요 기능**:
- 질문 분석 및 리서치 계획
- 병렬 정보 수집
- 데이터 통합 및 분석
- 구조화된 보고서 생성

**예상 소요 시간**: 4-6시간
**난이도**: ⭐⭐⭐⭐☆

---

## 🔗 심화 학습

1. **공식 문서 심화**
   - [22-multi-agent.md](/official/22-multi-agent.md) - 멀티에이전트 개요
   - [23-subagents.md](/official/23-subagents.md) - Subagents 패턴
   - [24-handoffs.md](/official/24-handoffs.md) - Handoffs 패턴
   - [25-skills.md](/official/25-skills.md) - Skills 패턴
   - [26-router.md](/official/26-router.md) - Router 패턴
   - [27-custom-workflow.md](/official/27-custom-workflow.md) - LangGraph

2. **관련 논문**
   - [Generative Agents: Interactive Simulacra](https://arxiv.org/abs/2304.03442)
   - [MetaGPT: Meta Programming for Multi-Agent Systems](https://arxiv.org/abs/2308.00352)
   - [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)

3. **커뮤니티 리소스**
   - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
   - [Multi-Agent Examples](https://python.langchain.com/docs/use_cases/multi_agent/)

4. **다음 단계**
   - [Part 8: RAG & MCP](/src/part08_rag_mcp/README.md) - 지식 통합

---

## ✅ 체크리스트

Part 7을 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다 (7개)
- [ ] 실습 과제를 완료했다 (3개)
- [ ] 멀티에이전트의 필요성을 이해했다
- [ ] Subagents 패턴을 구현할 수 있다
- [ ] Handoffs로 제어를 전달할 수 있다
- [ ] Router 패턴을 사용할 수 있다
- [ ] LangGraph로 워크플로우를 만들 수 있다

---

**이전**: [← Part 6 - Context Engineering](/src/part06_context/README.md)
**다음**: [Part 8 - RAG & MCP로 이동](/src/part08_rag_mcp/README.md) →

---

**학습 진도**: ▓▓▓▓▓▓▓░░░ 70% (Part 7/10 완료)

*마지막 업데이트: 2025-02-06*
