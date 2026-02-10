# Project 3: 연구 에이전트 시스템 (Research Agent System)

> 난이도: 고급
> 예상 소요 시간: 4-5시간
> 관련 파트: Part 5 (Multi-Agent 시스템), Part 7 (실전 프로젝트)

---

## 프로젝트 개요

다중 에이전트 협업을 통해 자동화된 리서치 보고서를 생성하는 시스템을 구축합니다.

### 학습 목표

- Multi-Agent 시스템 설계 및 구현
- Agent 간 협업 및 통신
- 병렬 처리 및 작업 조율
- 구조화된 출력 생성
- 웹 검색 통합

---

## 시스템 아키텍처

```
사용자 질문
    ↓
┌─────────────────────┐
│  Planner Agent      │ ← 연구 계획 수립
│  (계획 수립자)       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Searcher Agent     │ ← 정보 수집 (병렬)
│  (정보 수집자)       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Analyst Agent      │ ← 데이터 분석
│  (분석가)           │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Writer Agent       │ ← 보고서 작성
│  (작성자)           │
└─────────────────────┘
    ↓
  최종 보고서
```

---

## 기능 요구사항

### 1. Planner Agent (계획 수립)
- 연구 주제 분석
- 하위 질문 생성 (3-5개)
- 검색 키워드 추출
- 연구 우선순위 설정

### 2. Searcher Agent (정보 수집)
- 웹 검색 실행
- 병렬 정보 수집
- 결과 필터링
- 소스 검증

### 3. Analyst Agent (데이터 분석)
- 수집된 정보 통합
- 핵심 인사이트 추출
- 모순 사항 식별
- 신뢰도 평가

### 4. Writer Agent (보고서 작성)
- 구조화된 보고서 생성
- 마크다운 형식
- 참고 문헌 포함
- 요약 및 결론

---

## 프로젝트 구조

```
03_research_agent/
├── README.md                    # 프로젝트 문서
├── main.py                      # 메인 실행 파일
├── multi_agent_system.py        # Multi-Agent 시스템
├── agents/                      # Agent 구현
│   ├── __init__.py
│   ├── planner.py              # Planner Agent
│   ├── searcher.py             # Searcher Agent
│   ├── analyst.py              # Analyst Agent
│   └── writer.py               # Writer Agent
├── tools/                       # Agent Tools
│   ├── __init__.py
│   ├── web_search.py           # 웹 검색 도구
│   └── data_processor.py       # 데이터 처리 도구
├── utils/                       # 유틸리티
│   ├── __init__.py
│   ├── prompts.py              # 프롬프트 템플릿
│   └── formatting.py           # 출력 포맷팅
├── requirements.txt             # 의존성
├── .env.example                # 환경 변수 예시
├── tests/                       # 테스트
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_system.py
│   └── test_tools.py
└── solution/                    # 참고 솔루션
    └── README_SOLUTION.md
```

---

## 시작하기

### 1. 의존성 설치

```bash
cd /Users/restful3/Desktop/langchain/projects/03_research_agent
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key  # 웹 검색용
```

### 3. 실행

```bash
# 대화형 모드
python main.py

# 단일 쿼리 모드
python main.py --query "인공지능의 미래 전망"

# 상세 모드 (각 단계 출력)
python main.py --query "기후 변화 대응 기술" --verbose
```

---

## 사용 예시

```bash
$ python main.py

🔬 연구 에이전트 시스템에 오신 것을 환영합니다!

연구 주제를 입력하세요: 양자 컴퓨팅의 최신 동향

📋 [Planner] 연구 계획 수립 중...
   - 하위 질문 1: 양자 컴퓨팅의 기본 원리는?
   - 하위 질문 2: 최신 양자 컴퓨터 개발 현황은?
   - 하위 질문 3: 실용화 가능성과 과제는?

🔍 [Searcher] 정보 수집 중... (3개 질문 병렬 처리)
   ✓ 15개 소스에서 정보 수집 완료

📊 [Analyst] 데이터 분석 중...
   - 핵심 인사이트 5개 추출
   - 신뢰도 평가 완료

✍️  [Writer] 보고서 작성 중...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 연구 보고서: 양자 컴퓨팅의 최신 동향
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 요약
양자 컴퓨팅은 양자역학의 원리를 활용하여 기존 컴퓨터로는
불가능한 계산을 수행할 수 있는 차세대 컴퓨팅 기술입니다...

## 주요 발견 사항

### 1. 기본 원리
- 큐비트(Qubit)를 사용한 중첩 상태
- 얽힘(Entanglement) 현상 활용
...

### 2. 개발 현황
- IBM Quantum: 127큐비트 시스템 공개
- Google: 양자 우월성 달성
...

### 3. 실용화 전망
- 향후 5-10년 내 실용화 예상
- 주요 과제: 오류 정정, 온도 유지
...

## 결론
양자 컴퓨팅은 암호학, 신약 개발, 최적화 문제 등
다양한 분야에서 혁신을 가져올 것으로 기대됩니다...

## 참고 문헌
1. IBM Quantum Blog - https://...
2. Nature: Quantum Computing - https://...
3. MIT Technology Review - https://...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 보고서가 'reports/quantum_computing.md'에 저장되었습니다.
```

---

## 구현 가이드

### Step 1: Agent 베이스 클래스

```python
# agents/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Agent 베이스 클래스"""

    def __init__(self, name: str, llm):
        self.name = name
        self.llm = llm

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 실행 메서드"""
        pass

    def log(self, message: str):
        """로그 출력"""
        print(f"[{self.name}] {message}")
```

### Step 2: Planner Agent 구현

```python
# agents/planner.py
from typing import List, Dict
from langchain.prompts import PromptTemplate

class PlannerAgent(BaseAgent):
    """연구 계획을 수립하는 Agent"""

    def run(self, input_data: Dict) -> Dict:
        topic = input_data["topic"]
        self.log(f"연구 주제 분석 중: {topic}")

        # 하위 질문 생성
        sub_questions = self._generate_sub_questions(topic)

        # 검색 키워드 추출
        keywords = self._extract_keywords(topic)

        return {
            "topic": topic,
            "sub_questions": sub_questions,
            "keywords": keywords,
        }

    def _generate_sub_questions(self, topic: str) -> List[str]:
        prompt = PromptTemplate(
            template="""주제: {topic}

이 주제를 깊이 연구하기 위한 3-5개의 하위 질문을 생성하세요.

하위 질문:""",
            input_variables=["topic"],
        )

        response = self.llm.invoke(prompt.format(topic=topic))
        # 파싱 로직...
        return questions
```

### Step 3: Multi-Agent 시스템

```python
# multi_agent_system.py
from typing import Dict
from agents.planner import PlannerAgent
from agents.searcher import SearcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent

class ResearchAgentSystem:
    """Multi-Agent 연구 시스템"""

    def __init__(self, llm):
        self.planner = PlannerAgent("Planner", llm)
        self.searcher = SearcherAgent("Searcher", llm)
        self.analyst = AnalystAgent("Analyst", llm)
        self.writer = WriterAgent("Writer", llm)

    def research(self, topic: str) -> str:
        """전체 연구 프로세스 실행"""

        # 1. 계획 수립
        plan = self.planner.run({"topic": topic})

        # 2. 정보 수집
        search_results = self.searcher.run(plan)

        # 3. 데이터 분석
        analysis = self.analyst.run(search_results)

        # 4. 보고서 작성
        report = self.writer.run(analysis)

        return report
```

---

## 고급 기능

### 1. 병렬 처리

```python
import asyncio
from typing import List

async def parallel_search(questions: List[str]):
    """질문들을 병렬로 검색"""
    tasks = [search_async(q) for q in questions]
    results = await asyncio.gather(*tasks)
    return results
```

### 2. Agent 간 통신

```python
class AgentMessage:
    """Agent 간 메시지"""
    def __init__(self, sender: str, receiver: str, content: Dict):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.timestamp = time.time()

class MessageBus:
    """Agent 통신을 위한 메시지 버스"""
    def __init__(self):
        self.messages = []

    def send(self, message: AgentMessage):
        self.messages.append(message)

    def receive(self, agent_name: str) -> List[AgentMessage]:
        return [m for m in self.messages if m.receiver == agent_name]
```

### 3. 상태 관리

```python
from typing import TypedDict

class ResearchState(TypedDict):
    """연구 진행 상태"""
    topic: str
    plan: Dict
    search_results: List
    analysis: Dict
    report: str
    status: str
```

---

## 웹 검색 통합

### Tavily API 사용

```python
# tools/web_search.py
from langchain_community.tools.tavily_search import TavilySearchResults

def create_search_tool():
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
    )
```

### DuckDuckGo 사용 (무료 대안)

```python
from langchain_community.tools import DuckDuckGoSearchResults

def create_ddg_search_tool():
    return DuckDuckGoSearchResults(
        max_results=5,
        backend="news",
    )
```

---

## 테스트

### 단위 테스트

```bash
pytest tests/ -v
```

### 통합 테스트

```bash
pytest tests/test_system.py -v
```

### 성능 테스트

```bash
python -m pytest tests/ --benchmark-only
```

---

## 최적화 전략

### 1. 캐싱
- 검색 결과 캐싱
- LLM 응답 캐싱
- 중간 결과 저장

### 2. 비용 절감
- GPT-4o-mini 사용 (대부분 작업)
- GPT-4o 사용 (최종 보고서만)
- 프롬프트 최적화

### 3. 속도 개선
- 병렬 처리 활용
- 스트리밍 응답
- 배치 처리

---

## 도전 과제

### 1. 인터랙티브 모드
사용자가 중간에 피드백을 제공할 수 있는 시스템

### 2. 다국어 지원
한국어, 영어 등 다양한 언어로 보고서 생성

### 3. 시각화
검색 결과를 차트나 그래프로 표현

### 4. 팩트 체킹
정보의 신뢰성 자동 검증

### 5. 협업 기능
여러 사용자가 함께 리서치 수행

---

## 평가 기준

### 기능 완성도 (40점)
- [ ] 4개 Agent 모두 구현
- [ ] 웹 검색 통합
- [ ] 구조화된 보고서 생성
- [ ] 에러 핸들링

### 코드 품질 (30점)
- [ ] Agent 간 느슨한 결합
- [ ] 재사용 가능한 컴포넌트
- [ ] 타입 힌트 사용
- [ ] 문서화

### 성능 (20점)
- [ ] 병렬 처리 구현
- [ ] 적절한 캐싱
- [ ] 비용 효율성

### 창의성 (10점)
- [ ] 독창적인 기능 추가
- [ ] 사용자 경험 개선

---

## 문제 해결

### Q: Tavily API 키가 없어요
A: DuckDuckGo를 대신 사용하세요 (무료)

### Q: Agent가 너무 느려요
A: 병렬 처리와 GPT-4o-mini 사용

### Q: 검색 결과 품질이 낮아요
A: 검색 키워드 개선 및 결과 필터링 강화

### Q: 보고서 형식이 일정하지 않아요
A: Structured Output이나 Pydantic 모델 사용

---

## 참고 자료

- [LangChain Multi-Agent](https://python.langchain.com/docs/concepts/agents/)
- [LangGraph 튜토리얼](https://langchain-ai.github.io/langgraph/)
- [Tavily API 문서](https://tavily.com/)
- Part 5: Multi-Agent 시스템
- Part 7: 실전 프로젝트

---

## 다음 단계

프로젝트 완료 후:
1. Project 4: Customer Service Agent
2. LangGraph로 마이그레이션
3. 프로덕션 배포

**행운을 빕니다!**
