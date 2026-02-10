"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent - 실습 과제 3 해답
================================================================================

과제: 리서치 파이프라인 (Planner→Searcher→Analyst→Writer)
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. Planner: 리서치 계획 수립
2. Searcher: 정보 수집
3. Analyst: 데이터 분석
4. Writer: 최종 보고서 작성
5. 4단계 파이프라인 구축

학습 목표:
- 복잡한 Multi-Agent 파이프라인
- Agent 간 데이터 전달
- 순차적 협업 플로우

================================================================================
"""

from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from typing_extensions import TypedDict
import operator

# ============================================================================
# 리서치 State
# ============================================================================

class ResearchState(TypedDict):
    """리서치 파이프라인 상태"""
    messages: Annotated[list, operator.add]
    topic: str  # 리서치 주제
    research_plan: str  # 리서치 계획
    raw_data: list  # 수집된 데이터
    analysis: str  # 분석 결과
    final_report: str  # 최종 보고서

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def search_academic_papers(query: str) -> str:
    """학술 논문을 검색합니다."""
    return f"""
    [논문 1] {query}에 관한 최신 연구
    - 저자: Dr. Smith et al.
    - 발행: 2024
    - 핵심: 혁신적인 접근법 제시
    
    [논문 2] {query} 응용 사례
    - 저자: Prof. Johnson
    - 발행: 2023
    - 핵심: 실제 적용 결과 분석
    """

@tool
def search_industry_reports(topic: str) -> str:
    """산업 보고서를 검색합니다."""
    return f"""
    [보고서 1] {topic} 시장 동향 2024
    - 출처: Tech Research Inc.
    - 규모: $50B (전년 대비 +15%)
    - 전망: 지속 성장 예상
    
    [보고서 2] {topic} 기술 트렌드
    - 출처: Industry Insights
    - 주요 트렌드: AI 통합, 클라우드화
    """

# ============================================================================
# 각 단계의 Agent들
# ============================================================================

def create_planner_agent():
    """Planner Agent: 리서치 계획 수립"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def plan(topic: str) -> str:
        prompt = f"""당신은 리서치 Planner입니다.
        
주제: {topic}

다음 형식으로 리서치 계획을 수립하세요:

1. 리서치 목표
2. 주요 조사 항목 (3-5개)
3. 데이터 소스
4. 예상 결과물

간결하고 체계적으로 작성하세요."""

        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    
    return plan

def create_searcher_agent():
    """Searcher Agent: 정보 수집"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_academic_papers, search_industry_reports]
    
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(
        model, tools,
        state_modifier="당신은 정보 수집 전문 Agent입니다. 학술 논문과 산업 보고서를 검색합니다."
    )

def create_analyst_agent():
    """Analyst Agent: 데이터 분석"""
    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    def analyze(plan: str, data: list) -> str:
        prompt = f"""당신은 데이터 분석 전문가입니다.

리서치 계획:
{plan}

수집된 데이터:
{chr(10).join([str(d) for d in data])}

다음 형식으로 분석하세요:

1. 데이터 요약
2. 주요 발견사항
3. 패턴 및 트렌드
4. 시사점

전문적이고 통찰력 있게 분석하세요."""

        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    
    return analyze

def create_writer_agent():
    """Writer Agent: 보고서 작성"""
    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    def write_report(topic: str, plan: str, analysis: str) -> str:
        prompt = f"""당신은 전문 리포트 작성자입니다.

주제: {topic}

리서치 계획:
{plan}

분석 결과:
{analysis}

다음 형식으로 최종 보고서를 작성하세요:

# {topic} 리서치 보고서

## 요약
(2-3 문장)

## 배경
(리서치 필요성)

## 주요 발견사항
(핵심 내용)

## 결론 및 제언
(향후 방향)

## 참고자료

전문적이고 읽기 쉽게 작성하세요."""

        response = model.invoke([HumanMessage(content=prompt)])
        return response.content
    
    return write_report

# ============================================================================
# 리서치 파이프라인 구축
# ============================================================================

def create_research_pipeline():
    """4단계 리서치 파이프라인"""
    
    planner = create_planner_agent()
    searcher = create_searcher_agent()
    analyst = create_analyst_agent()
    writer = create_writer_agent()
    
    # 1. Planning 노드
    def planning_node(state: ResearchState) -> dict:
        print("\n📋 [Planner] 리서치 계획 수립 중...")
        
        topic = state["topic"]
        research_plan = planner(topic)
        
        print(f"✅ 계획 수립 완료\n{research_plan[:200]}...")
        
        return {"research_plan": research_plan}
    
    # 2. Searching 노드
    def searching_node(state: ResearchState) -> dict:
        print("\n🔍 [Searcher] 정보 수집 중...")
        
        # 계획에 기반하여 검색
        search_query = f"{state['topic']} 관련 자료"
        
        result = searcher.invoke({
            "messages": [HumanMessage(content=f"다음 주제로 정보를 수집하세요: {search_query}")]
        })
        
        # 데이터 수집
        raw_data = [msg.content for msg in result["messages"] if isinstance(msg, AIMessage)]
        
        print(f"✅ 데이터 수집 완료 ({len(raw_data)}개 항목)")
        
        return {"raw_data": raw_data}
    
    # 3. Analysis 노드
    def analysis_node(state: ResearchState) -> dict:
        print("\n📊 [Analyst] 데이터 분석 중...")
        
        analysis = analyst(state["research_plan"], state["raw_data"])
        
        print(f"✅ 분석 완료\n{analysis[:200]}...")
        
        return {"analysis": analysis}
    
    # 4. Writing 노드
    def writing_node(state: ResearchState) -> dict:
        print("\n✍️  [Writer] 보고서 작성 중...")
        
        final_report = writer(
            state["topic"],
            state["research_plan"],
            state["analysis"]
        )
        
        print("✅ 보고서 작성 완료")
        
        return {"final_report": final_report}
    
    # 그래프 구축
    graph_builder = StateGraph(ResearchState)
    
    # 노드 추가
    graph_builder.add_node("planner", planning_node)
    graph_builder.add_node("searcher", searching_node)
    graph_builder.add_node("analyst", analysis_node)
    graph_builder.add_node("writer", writing_node)
    
    # 순차 파이프라인
    graph_builder.add_edge(START, "planner")
    graph_builder.add_edge("planner", "searcher")
    graph_builder.add_edge("searcher", "analyst")
    graph_builder.add_edge("analyst", "writer")
    graph_builder.add_edge("writer", END)
    
    return graph_builder.compile()

# ============================================================================
# 테스트
# ============================================================================

def test_research_pipeline():
    """리서치 파이프라인 테스트"""
    print("=" * 70)
    print("🔬 리서치 파이프라인 테스트")
    print("=" * 70)
    
    pipeline = create_research_pipeline()
    
    topics = [
        "인공지능의 윤리적 이슈",
        "양자 컴퓨팅의 미래",
    ]
    
    for topic in topics:
        print(f"\n{'=' * 70}")
        print(f"📚 리서치 주제: {topic}")
        print("=" * 70)
        
        result = pipeline.invoke({
            "topic": topic,
            "messages": []
        })
        
        print(f"\n{'=' * 70}")
        print("📄 최종 보고서")
        print("=" * 70)
        print(result["final_report"])
        print("\n")

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🔬 Part 7: 리서치 파이프라인 - 실습 과제 3 해답")
    print("=" * 70)
    
    try:
        test_research_pipeline()
        
        print("\n💡 학습 포인트:")
        print("  1. 4단계 순차 파이프라인")
        print("  2. Agent 간 데이터 전달")
        print("  3. 전문화된 역할 분담")
        print("  4. 복잡한 워크플로우 구성")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
