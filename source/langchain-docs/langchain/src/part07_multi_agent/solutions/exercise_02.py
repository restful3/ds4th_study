"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent - 실습 과제 2 해답
================================================================================

과제: 고객 서비스 라우터 (문의 타입별 전문 Agent)
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. Router Agent: 문의 타입 분류
2. Technical Support Agent: 기술 지원
3. Sales Agent: 영업 문의
4. General Agent: 일반 문의

학습 목표:
- 조건부 라우팅
- 전문화된 Multiple Agents
- 동적 Agent 선택

================================================================================
"""

from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def check_system_status() -> str:
    """시스템 상태를 확인합니다."""
    return """
    ✅ 시스템 상태
    - 웹 서버: 정상
    - 데이터베이스: 정상
    - API 서버: 정상
    - 응답 시간: 45ms
    """

@tool
def search_documentation(topic: str) -> str:
    """문서를 검색합니다."""
    return f"'{topic}'에 대한 기술 문서를 찾았습니다. (시뮬레이션)"

@tool
def get_pricing() -> str:
    """가격 정보를 조회합니다."""
    return """
    💰 가격 플랜
    - Basic: $9/월
    - Pro: $29/월
    - Enterprise: 문의
    """

@tool
def create_ticket(issue: str) -> str:
    """지원 티켓을 생성합니다."""
    return f"티켓 #T-2024-001 생성됨: {issue}"

# ============================================================================
# 문의 분류기
# ============================================================================

class InquiryClassifier:
    """문의 타입 분류기"""
    
    TECHNICAL_KEYWORDS = ["오류", "에러", "error", "bug", "문제", "작동", "설치", "설정"]
    SALES_KEYWORDS = ["가격", "구매", "플랜", "결제", "할인", "견적", "price"]
    
    @classmethod
    def classify(cls, text: str) -> Literal["technical", "sales", "general"]:
        """문의 타입 분류"""
        text_lower = text.lower()
        
        tech_count = sum(1 for kw in cls.TECHNICAL_KEYWORDS if kw in text_lower)
        sales_count = sum(1 for kw in cls.SALES_KEYWORDS if kw in text_lower)
        
        if tech_count > sales_count and tech_count > 0:
            return "technical"
        elif sales_count > 0:
            return "sales"
        else:
            return "general"

# ============================================================================
# 전문 Agents
# ============================================================================

def create_technical_agent():
    """기술 지원 Agent"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [check_system_status, search_documentation, create_ticket]
    
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(
        model, tools,
        state_modifier="당신은 기술 지원 전문 Agent입니다. 기술적 문제를 해결하고 문서를 안내합니다."
    )

def create_sales_agent():
    """영업 Agent"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [get_pricing]
    
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(
        model, tools,
        state_modifier="당신은 영업 전문 Agent입니다. 제품 가격과 플랜을 안내합니다."
    )

def create_general_agent():
    """일반 문의 Agent"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def respond(messages):
        return model.invoke([
            SystemMessage(content="당신은 친절한 고객 서비스 Agent입니다."),
            *messages
        ])
    return respond

# ============================================================================
# 라우터 시스템
# ============================================================================

class RouterState(MessagesState):
    """라우터 상태"""
    inquiry_type: str

def create_customer_service_router():
    """고객 서비스 라우터"""
    
    tech_agent = create_technical_agent()
    sales_agent = create_sales_agent()
    general_agent = create_general_agent()
    
    # 분류 노드
    def classify_inquiry(state: RouterState) -> dict:
        user_message = state["messages"][-1].content
        inquiry_type = InquiryClassifier.classify(user_message)
        
        icons = {"technical": "🔧", "sales": "💰", "general": "💬"}
        print(f"{icons[inquiry_type]} 분류: {inquiry_type}")
        
        return {"inquiry_type": inquiry_type}
    
    # 라우팅 결정
    def route_to_agent(state: RouterState) -> Literal["technical", "sales", "general"]:
        return state["inquiry_type"]
    
    # Agent 노드들
    def technical_node(state: RouterState) -> dict:
        print("🔧 [Technical Agent] 처리 중...")
        result = tech_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}
    
    def sales_node(state: RouterState) -> dict:
        print("💰 [Sales Agent] 처리 중...")
        result = sales_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}
    
    def general_node(state: RouterState) -> dict:
        print("💬 [General Agent] 처리 중...")
        response = general_agent(state["messages"])
        return {"messages": [response]}
    
    # 그래프 구축
    graph_builder = StateGraph(RouterState)
    
    graph_builder.add_node("classify", classify_inquiry)
    graph_builder.add_node("technical", technical_node)
    graph_builder.add_node("sales", sales_node)
    graph_builder.add_node("general", general_node)
    
    graph_builder.add_edge(START, "classify")
    graph_builder.add_conditional_edges(
        "classify",
        route_to_agent,
        {"technical": "technical", "sales": "sales", "general": "general"}
    )
    graph_builder.add_edge("technical", END)
    graph_builder.add_edge("sales", END)
    graph_builder.add_edge("general", END)
    
    return graph_builder.compile()

# ============================================================================
# 테스트
# ============================================================================

def test_router_system():
    """라우터 시스템 테스트"""
    print("=" * 70)
    print("📞 고객 서비스 라우터 테스트")
    print("=" * 70)
    
    router = create_customer_service_router()
    
    test_cases = [
        ("로그인 오류가 발생해요", "technical"),
        ("가격 플랜을 알고 싶어요", "sales"),
        ("서비스 소개 부탁드립니다", "general"),
    ]
    
    for question, expected_type in test_cases:
        print(f"\n{'=' * 70}")
        print(f"👤 문의: {question}")
        print(f"📋 기대 타입: {expected_type}")
        print("=" * 70)
        
        result = router.invoke({"messages": [HumanMessage(content=question)]})
        
        print(f"✅ 실제 타입: {result['inquiry_type']}")
        print(f"\n🤖 응답:\n{result['messages'][-1].content}\n")

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📞 Part 7: 고객 서비스 라우터 - 실습 과제 2 해답")
    print("=" * 70)
    
    try:
        test_router_system()
        
        print("\n💡 학습 포인트:")
        print("  1. 조건부 라우팅 (문의 타입별)")
        print("  2. 전문화된 Multiple Agents")
        print("  3. 동적 Agent 선택")
        print("  4. 확장 가능한 라우터 설계")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
