"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: Production - 실습 과제 2 해답
================================================================================

과제: 승인 기반 작업 Agent (HITL - Human-in-the-Loop)
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. 중요한 작업은 사용자 승인 필요
2. Interrupt를 통한 작업 중단
3. 승인 후 재개 기능

학습 목표:
- Human-in-the-Loop 패턴
- Interrupt 사용
- 상태 저장 및 재개

================================================================================
"""

from typing import Literal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# ============================================================================
# 도구 정의 (위험도 포함)
# ============================================================================

@tool
def read_data(source: str) -> str:
    """데이터를 읽습니다. (안전)"""
    return f"데이터 읽기 완료: {source}에서 1,234개 레코드 로드"

@tool
def analyze_data(data: str) -> str:
    """데이터를 분석합니다. (안전)"""
    return f"분석 완료: {data}에 대한 통계 생성"

@tool
def delete_records(table: str, condition: str) -> str:
    """레코드를 삭제합니다. (위험 - 승인 필요)"""
    return f"⚠️  삭제 작업: {table} 테이블에서 {condition} 조건의 레코드 삭제"

@tool
def send_email_blast(recipient_count: int, message: str) -> str:
    """대량 이메일을 발송합니다. (위험 - 승인 필요)"""
    return f"⚠️  이메일 발송: {recipient_count}명에게 메시지 전송"

@tool
def update_production_config(setting: str, value: str) -> str:
    """프로덕션 설정을 변경합니다. (위험 - 승인 필요)"""
    return f"⚠️  설정 변경: {setting} = {value}"

# 위험한 작업 목록
DANGEROUS_ACTIONS = {
    "delete_records",
    "send_email_blast",
    "update_production_config"
}

# ============================================================================
# HITL State
# ============================================================================

class HITLState(MessagesState):
    """HITL Agent 상태"""
    pending_action: dict  # 승인 대기 중인 작업
    approved: bool  # 승인 여부

# ============================================================================
# HITL Agent 구축
# ============================================================================

def create_hitl_agent():
    """Human-in-the-Loop Agent 생성"""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [
        read_data,
        analyze_data,
        delete_records,
        send_email_blast,
        update_production_config
    ]
    
    # Agent 노드
    def agent_node(state: HITLState) -> dict:
        """Agent 작업 수행"""
        
        # 승인이 필요한 경우 (pending_action이 있는 경우)
        if state.get("pending_action"):
            if state.get("approved"):
                # 승인됨 - 작업 실행
                action = state["pending_action"]
                print(f"✅ 승인됨 - 작업 실행: {action['tool']}")
                
                # 실제 도구 실행 시뮬레이션
                tool_map = {t.name: t for t in tools}
                tool = tool_map[action["tool"]]
                result = tool.invoke(action["args"])
                
                response_msg = AIMessage(content=f"작업 완료:\n{result}")
                
                return {
                    "messages": [response_msg],
                    "pending_action": None,
                    "approved": False
                }
            else:
                # 거부됨
                print("❌ 거부됨 - 작업 취소")
                response_msg = AIMessage(content="사용자가 작업을 거부했습니다.")
                return {
                    "messages": [response_msg],
                    "pending_action": None,
                    "approved": False
                }
        
        # 일반 Agent 실행
        basic_agent = create_react_agent(model, tools[:2])  # 안전한 도구만
        result = basic_agent.invoke({"messages": state["messages"]})
        
        # 도구 호출 확인
        last_message = result["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            
            # 위험한 작업인지 확인
            if tool_name in DANGEROUS_ACTIONS:
                print(f"⚠️  위험한 작업 감지: {tool_name}")
                print(f"📋 승인이 필요합니다!")
                
                return {
                    "messages": result["messages"],
                    "pending_action": {
                        "tool": tool_name,
                        "args": tool_call["args"]
                    }
                }
        
        return {"messages": result["messages"]}
    
    # 승인 필요 여부 판단
    def needs_approval(state: HITLState) -> Literal["approval", "complete"]:
        """승인이 필요한지 판단"""
        if state.get("pending_action") and not state.get("approved"):
            return "approval"
        return "complete"
    
    # 승인 노드 (Interrupt 발생)
    def approval_node(state: HITLState) -> dict:
        """승인 요청 (여기서 Interrupt)"""
        action = state["pending_action"]
        
        # Interrupt - 여기서 실행이 중단되고 사용자 입력 대기
        print(f"\n{'=' * 70}")
        print("⚠️  승인 요청")
        print("=" * 70)
        print(f"작업: {action['tool']}")
        print(f"인자: {action['args']}")
        print("=" * 70)
        
        # 실제로는 여기서 interrupt()를 호출하여 실행 중단
        # 사용자가 승인/거부 후 재개
        
        return {}
    
    # 그래프 구축
    graph_builder = StateGraph(HITLState)
    
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("approval", approval_node)
    
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        needs_approval,
        {
            "approval": "approval",
            "complete": END
        }
    )
    graph_builder.add_edge("approval", END)  # 승인 후 종료 (재개 시 agent로)
    
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["approval"]  # approval 전에 중단
    )
    
    return graph

# ============================================================================
# 승인 워크플로우
# ============================================================================

def run_with_approval(agent, question: str, thread_id: str = "approval_demo"):
    """승인 워크플로우 실행"""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n질문: {question}\n")
    
    # 1단계: 초기 실행
    print("1️⃣ Agent 실행 중...")
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )
    
    # Interrupt 확인
    state = agent.get_state(config)
    
    if state.next:  # 다음 노드가 있으면 중단된 것
        print("\n⏸️  실행이 중단되었습니다 (승인 대기)")
        
        # 사용자 입력 받기
        print("\n승인하시겠습니까? (y/n): ", end="")
        user_input = input().strip().lower()
        
        approved = user_input in ['y', 'yes', '예']
        
        # 2단계: 승인/거부와 함께 재개
        print(f"\n2️⃣ 재개 중... (승인: {approved})")
        
        result = agent.invoke(
            {"approved": approved},
            config
        )
    
    # 최종 결과
    print(f"\n✅ 완료")
    if result.get("messages"):
        print(f"결과: {result['messages'][-1].content}")
    
    return result

# ============================================================================
# 테스트
# ============================================================================

def test_hitl_agent():
    """HITL Agent 테스트"""
    print("=" * 70)
    print("🤝 Human-in-the-Loop Agent 테스트")
    print("=" * 70)
    
    agent = create_hitl_agent()
    
    # 테스트 1: 안전한 작업 (승인 불필요)
    print("\n[테스트 1] 안전한 작업")
    print("-" * 70)
    
    run_with_approval(
        agent,
        "users 테이블에서 데이터를 읽고 분석해줘",
        "test1"
    )
    
    # 테스트 2: 위험한 작업 (승인 필요)
    print("\n\n[테스트 2] 위험한 작업")
    print("-" * 70)
    
    run_with_approval(
        agent,
        "inactive_users 테이블에서 is_active=false 조건으로 레코드를 삭제해줘",
        "test2"
    )

def test_multiple_approvals():
    """다중 승인 시나리오"""
    print("\n" + "=" * 70)
    print("🔄 다중 승인 시나리오")
    print("=" * 70)
    
    # 시뮬레이션만 (실제 다중 승인은 복잡함)
    print("""
시나리오: 여러 단계에서 승인이 필요한 경우

1. 데이터 백업 (승인 필요)
   → 사용자 승인
   
2. 레코드 삭제 (승인 필요)
   → 사용자 승인
   
3. 이메일 알림 (승인 필요)
   → 사용자 승인

이런 복잡한 워크플로우도 HITL 패턴으로 구현 가능합니다.
    """)

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🤝 Part 9: 승인 기반 작업 Agent - 실습 과제 2 해답")
    print("=" * 70)
    
    try:
        test_hitl_agent()
        test_multiple_approvals()
        
        print("\n💡 학습 포인트:")
        print("  1. Human-in-the-Loop 패턴")
        print("  2. interrupt_before 사용")
        print("  3. 상태 저장 및 재개")
        print("  4. 위험한 작업 안전장치")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
