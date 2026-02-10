"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent - 실습 과제 1 해답
================================================================================

과제: 이중 전문가 시스템 (검색 + 요약 Agent 협업)
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. Searcher Agent: 정보 검색 전담
2. Summarizer Agent: 요약 전담
3. 두 Agent의 순차적 협업

학습 목표:
- Multi-Agent 패턴
- Agent 간 정보 전달
- 전문화된 Agent 설계

================================================================================
"""

from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
import operator

# ============================================================================
# 도구 정의
# ============================================================================

# 검색 도구들
@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색 쿼리
    """
    # 실제로는 웹 검색 API를 호출
    # 여기서는 시뮬레이션
    results = f"""
    검색 결과: '{query}'

    1. Python은 1991년 Guido van Rossum이 개발한 고급 프로그래밍 언어입니다.
       간결하고 읽기 쉬운 문법으로 초보자도 쉽게 배울 수 있습니다.

    2. Python의 주요 특징:
       - 인터프리터 언어
       - 동적 타이핑
       - 풍부한 라이브러리
       - 다양한 패러다임 지원 (절차적, 객체지향, 함수형)

    3. 주요 활용 분야:
       - 웹 개발 (Django, Flask)
       - 데이터 분석 (Pandas, NumPy)
       - 머신러닝 (TensorFlow, PyTorch, Scikit-learn)
       - 자동화 스크립팅

    4. Python의 인기:
       - TIOBE Index에서 상위권 유지
       - GitHub에서 가장 많이 사용되는 언어 중 하나
       - AI/ML 분야에서 사실상 표준 언어

    5. 커뮤니티와 생태계:
       - PyPI(Python Package Index)에 30만개 이상의 패키지
       - 활발한 커뮤니티와 풍부한 문서
       - 정기적인 버전 업데이트 (3.12, 3.13 등)
    """
    return results


@tool
def database_search(query: str) -> str:
    """데이터베이스에서 정보를 검색합니다.

    Args:
        query: 검색 쿼리
    """
    # 데이터베이스 검색 시뮬레이션
    results = f"""
    데이터베이스 검색 결과: '{query}'

    [문서 1] Python 기초 가이드
    - 작성일: 2024-01-15
    - 내용: Python 설치, 기본 문법, 데이터 타입
    - 조회수: 1,245

    [문서 2] Python 고급 기능
    - 작성일: 2024-01-10
    - 내용: 데코레이터, 제너레이터, 컨텍스트 매니저
    - 조회수: 892

    [문서 3] Python 실전 프로젝트
    - 작성일: 2024-01-05
    - 내용: 웹 크롤러, API 서버, 데이터 분석
    - 조회수: 2,103
    """
    return results


# ============================================================================
# Multi-Agent State
# ============================================================================

class MultiAgentState(TypedDict):
    """Multi-Agent 상태"""
    messages: Annotated[list, operator.add]  # 대화 메시지
    search_results: str  # 검색 결과
    summary: str  # 요약 결과
    next_agent: str  # 다음 Agent


# ============================================================================
# Searcher Agent 생성
# ============================================================================

def create_searcher_agent():
    """검색 Agent 생성"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [web_search, database_search]

    system_prompt = """당신은 정보 검색 전문 Agent입니다.

역할:
1. 사용자 질문을 분석하여 검색 쿼리 생성
2. 적절한 검색 도구 선택 및 실행
3. 검색 결과를 다음 Agent에게 전달

검색 전략:
- 웹 검색: 일반적인 정보, 최신 정보
- 데이터베이스 검색: 기존 문서, 내부 자료

검색 결과는 가능한 한 많은 정보를 포함하세요."""

    agent = create_react_agent(model, tools, state_modifier=system_prompt)
    return agent


# ============================================================================
# Summarizer Agent 생성
# ============================================================================

def create_summarizer_agent():
    """요약 Agent 생성"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 요약은 도구 없이 순수 LLM으로
    def summarize(messages: list) -> AIMessage:
        """검색 결과를 요약"""

        system_prompt = """당신은 정보 요약 전문 Agent입니다.

역할:
1. 검색 Agent가 수집한 정보를 분석
2. 핵심 내용을 명확하고 간결하게 요약
3. 사용자가 쉽게 이해할 수 있도록 구조화

요약 가이드:
- 핵심 포인트 3-5개로 정리
- 불필요한 세부사항 제거
- 논리적 흐름 유지
- 명확한 문장 사용

형식:
📌 핵심 요약
- 포인트 1
- 포인트 2
- 포인트 3

💡 추가 정보
- 상세 내용
"""

        full_messages = [SystemMessage(content=system_prompt)] + messages
        response = model.invoke(full_messages)
        return response

    return summarize


# ============================================================================
# Multi-Agent 그래프 구축
# ============================================================================

def create_dual_expert_system():
    """이중 전문가 시스템 생성"""

    searcher_agent = create_searcher_agent()
    summarizer = create_summarizer_agent()

    # Searcher 노드
    def searcher_node(state: MultiAgentState) -> dict:
        """검색 Agent 실행"""
        print("\n🔍 [Searcher Agent] 정보 검색 중...")

        result = searcher_agent.invoke({"messages": state["messages"]})

        # 검색 결과 추출
        search_results = ""
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                search_results += msg.content + "\n"

        print(f"✅ [Searcher Agent] 검색 완료")

        return {
            "messages": result["messages"],
            "search_results": search_results,
            "next_agent": "summarizer"
        }

    # Summarizer 노드
    def summarizer_node(state: MultiAgentState) -> dict:
        """요약 Agent 실행"""
        print("\n📝 [Summarizer Agent] 요약 생성 중...")

        # 검색 결과를 메시지로 구성
        summary_messages = [
            HumanMessage(
                content=f"""다음은 검색된 정보입니다. 사용자 질문에 답하기 위해 이 정보를 요약해주세요.

원래 질문: {state['messages'][0].content}

검색 결과:
{state['search_results']}

위 정보를 바탕으로 사용자 질문에 대한 명확하고 간결한 답변을 작성해주세요."""
            )
        ]

        summary_response = summarizer(summary_messages)

        print(f"✅ [Summarizer Agent] 요약 완료")

        return {
            "messages": [summary_response],
            "summary": summary_response.content,
            "next_agent": "end"
        }

    # 그래프 구축
    graph_builder = StateGraph(MultiAgentState)

    # 노드 추가
    graph_builder.add_node("searcher", searcher_node)
    graph_builder.add_node("summarizer", summarizer_node)

    # 엣지: 순차 실행
    graph_builder.add_edge(START, "searcher")
    graph_builder.add_edge("searcher", "summarizer")
    graph_builder.add_edge("summarizer", END)

    graph = graph_builder.compile()

    return graph


# ============================================================================
# 테스트 함수
# ============================================================================

def test_dual_expert_system():
    """이중 전문가 시스템 테스트"""
    print("=" * 70)
    print("👥 이중 전문가 시스템 테스트")
    print("=" * 70)

    system = create_dual_expert_system()

    test_questions = [
        "Python 프로그래밍 언어에 대해 알려줘",
        "머신러닝이란 무엇인가?",
        "웹 개발의 최신 트렌드는?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 70}")
        print(f"📝 질문 {i}: {question}")
        print("=" * 70)

        result = system.invoke({
            "messages": [HumanMessage(content=question)]
        })

        # 최종 요약 출력
        final_summary = result["summary"]
        print(f"\n{'=' * 70}")
        print("📊 최종 결과")
        print("=" * 70)
        print(final_summary)


def test_with_visualization():
    """시각화와 함께 테스트"""
    print("\n" + "=" * 70)
    print("🎨 Agent 협업 과정 시각화")
    print("=" * 70)

    system = create_dual_expert_system()
    question = "인공지능의 역사를 간단히 설명해줘"

    print(f"\n👤 사용자 질문: {question}")
    print("\n" + "=" * 70)
    print("Agent 협업 흐름")
    print("=" * 70)
    print("""
    👤 User
     │
     ├─ 질문 입력
     │
     ▼
    🔍 Searcher Agent
     │
     ├─ 검색 도구 선택
     ├─ 웹 검색 실행
     ├─ DB 검색 실행
     ├─ 결과 수집
     │
     ▼
    📝 Summarizer Agent
     │
     ├─ 정보 분석
     ├─ 핵심 추출
     ├─ 요약 생성
     │
     ▼
    👤 User (최종 답변)
    """)

    print("\n실행 중...\n")

    result = system.invoke({
        "messages": [HumanMessage(content=question)]
    })

    print("\n" + "=" * 70)
    print("✅ 협업 완료")
    print("=" * 70)
    print(f"\n최종 답변:\n{result['summary']}")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 이중 전문가 시스템 - 대화형 모드")
    print("=" * 70)
    print("두 전문 Agent가 협업하여 답변합니다:")
    print("  🔍 Searcher: 정보 검색")
    print("  📝 Summarizer: 요약 생성")
    print("\n명령어: /quit - 종료")
    print("=" * 70)

    system = create_dual_expert_system()

    while True:
        try:
            user_input = input("\n👤 질문: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 종료합니다.")
                break

            print("\n" + "=" * 70)
            print("처리 중...")
            print("=" * 70)

            result = system.invoke({
                "messages": [HumanMessage(content=user_input)]
            })

            print("\n" + "=" * 70)
            print("💡 답변")
            print("=" * 70)
            print(result["summary"])

        except KeyboardInterrupt:
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
    print("👥 Part 7: 이중 전문가 시스템 - 실습 과제 1 해답")
    print("=" * 70)

    try:
        # 테스트 1: 기본 테스트
        test_dual_expert_system()

        # 테스트 2: 시각화
        test_with_visualization()

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
    print("  1. Multi-Agent 패턴 (검색 + 요약)")
    print("  2. Agent 간 순차적 정보 전달")
    print("  3. 전문화된 Agent 설계")
    print("  4. StateGraph로 협업 플로우 구성")
    print("\n💡 추가 학습:")
    print("  1. 병렬 검색 (여러 소스 동시)")
    print("  2. Agent 간 피드백 루프")
    print("  3. 검증 Agent 추가")
    print("  4. 동적 Agent 선택")
    print("=" * 70)


if __name__ == "__main__":
    main()
