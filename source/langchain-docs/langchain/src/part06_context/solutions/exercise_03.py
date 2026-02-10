"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context - 실습 과제 3 해답
================================================================================

과제: 적응형 Agent
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. 작업 복잡도에 따라 모델 자동 선택
2. 간단한 작업은 gpt-4o-mini, 복잡한 작업은 gpt-4o
3. 비용 효율성과 성능의 균형

학습 목표:
- 복잡도 분석 로직
- 모델 동적 선택
- 적응형 시스템 설계

================================================================================
"""

from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================================================
# 복잡도 분류기
# ============================================================================

class ComplexityAnalyzer:
    """작업 복잡도 분석기"""

    # 복잡도 키워드
    SIMPLE_KEYWORDS = [
        "안녕", "hello", "hi", "뭐", "누구", "이름",
        "간단", "빠르게", "짧게"
    ]

    COMPLEX_KEYWORDS = [
        "설명", "분석", "비교", "차이", "장단점",
        "자세히", "상세히", "깊이", "전문적",
        "어떻게", "왜", "구현", "코드", "알고리즘"
    ]

    VERY_COMPLEX_KEYWORDS = [
        "최적화", "아키텍처", "설계", "디자인패턴",
        "성능", "확장", "분산", "트레이드오프",
        "심층", "고급", "전문가", "복잡한"
    ]

    @classmethod
    def analyze(cls, text: str) -> Literal["simple", "medium", "complex"]:
        """텍스트 복잡도 분석"""

        text_lower = text.lower()

        # 길이 기반 판단
        word_count = len(text.split())

        # 키워드 매칭
        simple_matches = sum(1 for kw in cls.SIMPLE_KEYWORDS if kw in text_lower)
        complex_matches = sum(1 for kw in cls.COMPLEX_KEYWORDS if kw in text_lower)
        very_complex_matches = sum(1 for kw in cls.VERY_COMPLEX_KEYWORDS if kw in text_lower)

        # 점수 계산
        complexity_score = 0

        # 길이 점수 (단어 수)
        if word_count < 5:
            complexity_score += 0
        elif word_count < 15:
            complexity_score += 1
        else:
            complexity_score += 2

        # 키워드 점수
        if very_complex_matches > 0:
            complexity_score += 3
        elif complex_matches > 0:
            complexity_score += 2
        elif simple_matches > 0:
            complexity_score += 0
        else:
            complexity_score += 1

        # 질문 복잡도 (?, 여러 문장)
        if text.count('?') > 1 or text.count('.') > 1:
            complexity_score += 1

        # 최종 판단
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 4:
            return "medium"
        else:
            return "complex"


# ============================================================================
# 도구 정의
# ============================================================================

@tool
def calculate(expression: str) -> float:
    """수식을 계산합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 2")
    """
    try:
        # 안전한 계산을 위해 제한된 eval
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "허용되지 않은 문자가 포함되어 있습니다."

        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def get_info(topic: str) -> str:
    """주제에 대한 정보를 제공합니다.

    Args:
        topic: 정보를 얻고자 하는 주제
    """
    return f"'{topic}'에 대한 정보를 제공합니다. (이것은 데모 도구입니다)"


# ============================================================================
# 적응형 Agent 상태
# ============================================================================

class AdaptiveState(MessagesState):
    """적응형 Agent 상태"""
    complexity: str  # simple, medium, complex
    model_used: str  # 사용된 모델


# ============================================================================
# 적응형 Agent 구축
# ============================================================================

def create_adaptive_agent():
    """적응형 Agent 생성"""

    # 모델들
    simple_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    complex_model = ChatOpenAI(model="gpt-4o", temperature=0.7)

    tools = [calculate, get_info]

    # 복잡도 분석 노드
    def analyze_complexity(state: AdaptiveState) -> dict:
        """질문의 복잡도 분석"""
        last_message = state["messages"][-1]

        if isinstance(last_message, HumanMessage):
            complexity = ComplexityAnalyzer.analyze(last_message.content)

            print(f"🔍 복잡도 분석: {complexity}")

            return {"complexity": complexity}

        return {"complexity": "medium"}

    # 모델 선택 노드
    def select_model(state: AdaptiveState) -> Literal["simple_agent", "complex_agent"]:
        """복잡도에 따라 모델 선택"""
        complexity = state.get("complexity", "medium")

        if complexity == "simple":
            print(f"📱 선택된 모델: gpt-4o-mini (간단한 작업)")
            return "simple_agent"
        else:
            print(f"🚀 선택된 모델: gpt-4o (복잡한 작업)")
            return "complex_agent"

    # 간단한 Agent 노드
    def simple_agent_node(state: AdaptiveState) -> dict:
        """gpt-4o-mini로 처리"""
        model_with_tools = simple_model.bind_tools(tools)
        response = model_with_tools.invoke(state["messages"])

        return {
            "messages": [response],
            "model_used": "gpt-4o-mini"
        }

    # 복잡한 Agent 노드
    def complex_agent_node(state: AdaptiveState) -> dict:
        """gpt-4o로 처리"""
        model_with_tools = complex_model.bind_tools(tools)
        response = model_with_tools.invoke(state["messages"])

        return {
            "messages": [response],
            "model_used": "gpt-4o"
        }

    # 도구 실행 필요 여부 판단
    def should_use_tool(state: AdaptiveState) -> Literal["use_tool", "finish"]:
        """도구 호출이 필요한지 판단"""
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "use_tool"

        return "finish"

    # 도구 실행 노드
    def tool_node(state: AdaptiveState) -> dict:
        """도구 실행"""
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"🔧 도구 호출: {tool_name}({tool_args})")

            # 도구 실행
            tool_map = {t.name: t for t in tools}
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                print(f"📊 결과: {result}")

                from langchain_core.messages import ToolMessage
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

        return {"messages": tool_messages}

    # 최종 응답 노드
    def final_response(state: AdaptiveState) -> dict:
        """최종 응답 생성"""
        # 도구 결과를 바탕으로 최종 응답
        complexity = state.get("complexity", "medium")

        if complexity == "simple":
            response = simple_model.invoke(state["messages"])
        else:
            response = complex_model.invoke(state["messages"])

        return {"messages": [response]}

    # 그래프 구축
    graph_builder = StateGraph(AdaptiveState)

    # 노드 추가
    graph_builder.add_node("analyze", analyze_complexity)
    graph_builder.add_node("simple_agent", simple_agent_node)
    graph_builder.add_node("complex_agent", complex_agent_node)
    graph_builder.add_node("tool", tool_node)
    graph_builder.add_node("final_response", final_response)

    # 엣지 추가
    graph_builder.add_edge(START, "analyze")
    graph_builder.add_conditional_edges(
        "analyze",
        select_model,
        {
            "simple_agent": "simple_agent",
            "complex_agent": "complex_agent"
        }
    )

    # simple_agent -> 도구 판단
    graph_builder.add_conditional_edges(
        "simple_agent",
        should_use_tool,
        {
            "use_tool": "tool",
            "finish": END
        }
    )

    # complex_agent -> 도구 판단
    graph_builder.add_conditional_edges(
        "complex_agent",
        should_use_tool,
        {
            "use_tool": "tool",
            "finish": END
        }
    )

    # 도구 실행 후 최종 응답
    graph_builder.add_edge("tool", "final_response")
    graph_builder.add_edge("final_response", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


# ============================================================================
# 통계 추적
# ============================================================================

class AdaptiveStats:
    """적응형 Agent 통계"""

    def __init__(self):
        self.stats = {
            "simple": {"count": 0, "cost": 0},
            "medium": {"count": 0, "cost": 0},
            "complex": {"count": 0, "cost": 0}
        }

        # 모델별 비용 (1K 토큰당, 대략적)
        self.costs = {
            "gpt-4o-mini": 0.0003,
            "gpt-4o": 0.010
        }

    def record(self, complexity: str, model: str, tokens: int = 500):
        """기록"""
        self.stats[complexity]["count"] += 1

        cost = (tokens / 1000) * self.costs.get(model, 0.001)
        self.stats[complexity]["cost"] += cost

    def print_report(self):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("📊 적응형 Agent 통계")
        print("=" * 70)

        total_count = sum(s["count"] for s in self.stats.values())
        total_cost = sum(s["cost"] for s in self.stats.values())

        print(f"\n전체:")
        print(f"  총 요청: {total_count}")
        print(f"  총 비용: ${total_cost:.6f}")

        print(f"\n복잡도별:")
        for complexity, data in self.stats.items():
            if data["count"] > 0:
                print(f"  {complexity}: {data['count']}회, ${data['cost']:.6f}")

        print("=" * 70)


# ============================================================================
# 테스트 함수
# ============================================================================

def test_adaptive_agent():
    """적응형 Agent 테스트"""
    print("=" * 70)
    print("🔄 적응형 Agent 테스트")
    print("=" * 70)

    agent = create_adaptive_agent()
    config = {"configurable": {"thread_id": "test_adaptive"}}
    stats = AdaptiveStats()

    test_cases = [
        ("안녕하세요!", "simple"),
        ("2 + 2는 얼마인가요?", "simple"),
        ("Python의 리스트와 튜플의 차이점을 설명해주세요.", "medium"),
        ("머신러닝 모델의 성능 최적화 전략과 하이퍼파라미터 튜닝 방법을 자세히 설명해주세요.", "complex"),
        ("분산 시스템의 아키텍처 설계 시 고려해야 할 트레이드오프를 분석해주세요.", "complex"),
    ]

    for question, expected_complexity in test_cases:
        print(f"\n{'=' * 70}")
        print(f"👤 질문: {question}")
        print(f"📋 기대 복잡도: {expected_complexity}")
        print("=" * 70)

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

        actual_complexity = result.get("complexity", "unknown")
        model_used = result.get("model_used", "unknown")

        print(f"✅ 실제 복잡도: {actual_complexity}")
        print(f"🤖 사용된 모델: {model_used}")

        final_message = result["messages"][-1]
        print(f"\n🤖 응답:\n{final_message.content[:200]}...")

        # 통계 기록
        stats.record(actual_complexity, model_used)

    # 통계 출력
    stats.print_report()


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 적응형 Agent - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /stats - 통계 보기")
    print("  /test <text> - 복잡도만 테스트")
    print("  /quit - 종료")
    print("=" * 70)

    agent = create_adaptive_agent()
    config = {"configurable": {"thread_id": "interactive"}}
    stats = AdaptiveStats()

    while True:
        try:
            user_input = input("\n👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                stats.print_report()
                print("\n👋 종료합니다.")
                break

            elif user_input == "/stats":
                stats.print_report()
                continue

            elif user_input.startswith("/test"):
                text = user_input[6:].strip()
                if text:
                    complexity = ComplexityAnalyzer.analyze(text)
                    print(f"🔍 복잡도: {complexity}")
                continue

            # 일반 질문
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            complexity = result.get("complexity", "unknown")
            model_used = result.get("model_used", "unknown")

            final_message = result["messages"][-1]
            print(f"\n🤖 [{model_used}] {final_message.content}")

            stats.record(complexity, model_used)

        except KeyboardInterrupt:
            stats.print_report()
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🔄 Part 6: 적응형 Agent - 실습 과제 3 해답")
    print("=" * 70)

    try:
        # 테스트
        test_adaptive_agent()

        # 대화형 모드
        print("\n대화형 모드를 실행하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예']:
            interactive_mode()

    except Exception as e:
        print(f"\n⚠️  오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # 학습 포인트
    print("\n" + "=" * 70)
    print("💡 학습 포인트:")
    print("  1. 작업 복잡도 자동 분석")
    print("  2. 동적 모델 선택")
    print("  3. 비용 효율성 최적화")
    print("  4. 적응형 시스템 설계")
    print("\n💡 추가 학습:")
    print("  1. ML 기반 복잡도 예측")
    print("  2. 실시간 A/B 테스팅")
    print("  3. 사용자 피드백 학습")
    print("  4. 멀티 모델 앙상블")
    print("=" * 70)


if __name__ == "__main__":
    main()
