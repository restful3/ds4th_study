"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: First Agent - 실습 과제 3 해답
================================================================================

과제: 멀티 에이전트 대화
난이도: ⭐⭐⭐⭐☆ (중상급)

요구사항:
1. Agent A: 낙관적이고 긍정적인 성격
2. Agent B: 현실적이고 신중한 성격
3. 같은 질문("서울 날씨 좋은데 소풍 갈까?")을 두 Agent에게 물어보고 답변 비교
4. 두 Agent의 답변을 종합하여 최종 의사결정을 내리는 로직 구현

학습 목표:
- 같은 도구, 다른 System Prompt로 다른 성격의 Agent 생성
- 여러 Agent의 응답 수집 및 비교
- ReAct 패턴의 이해와 활용

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# ============================================================================
# 공통 도구 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회합니다.

    Args:
        city: 도시 이름 (예: 서울, 부산)

    Returns:
        해당 도시의 현재 날씨 정보
    """
    weather_data = {
        "서울": "맑음, 24°C, 습도 55%, 미세먼지 보통",
        "부산": "흐림, 21°C, 습도 70%, 미세먼지 좋음",
        "제주": "비, 19°C, 습도 85%, 미세먼지 좋음",
    }
    return weather_data.get(city, f"{city}: 맑음, 22°C (기본값)")


@tool
def get_forecast(city: str, days: int = 3) -> str:
    """며칠간의 날씨 예보를 조회합니다.

    Args:
        city: 도시 이름
        days: 예보 일수 (기본 3일)

    Returns:
        해당 도시의 날씨 예보
    """
    forecast_data = {
        "서울": "오늘: 맑음 24°C -> 내일: 흐림 20°C -> 모레: 비 17°C",
        "부산": "오늘: 흐림 21°C -> 내일: 비 18°C -> 모레: 맑음 22°C",
    }
    return forecast_data.get(city, f"{city}: 맑음 -> 흐림 -> 맑음")


# ============================================================================
# Agent 생성 - 두 개의 다른 성격
# ============================================================================

def create_optimistic_agent():
    """낙관적이고 긍정적인 Agent 생성 (Agent A)"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    tools = [get_weather, get_forecast]

    system_prompt = """당신은 매우 낙관적이고 긍정적인 성격의 야외 활동 전문가입니다.

성격:
- 항상 밝고 긍정적인 관점에서 바라봅니다
- "할 수 있다!", "좋은 기회다!" 같은 긍정적 표현을 자주 사용합니다
- 약간의 불확실성이 있어도 긍정적으로 해석합니다
- 날씨가 약간 흐려도 "구름이 좋은 그늘을 만들어줘요!" 식으로 표현합니다

답변 원칙:
- 도구를 사용하여 날씨 정보를 확인한 후 답변합니다
- 야외 활동의 장점과 즐거움을 강조합니다
- 준비물이나 대안도 제시하지만, 긍정적 톤으로 전달합니다
- 답변 마지막에 "[추천: 가세요!]" 또는 "[추천: 좋아요!]" 같은 태그를 붙입니다"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


def create_realistic_agent():
    """현실적이고 신중한 Agent 생성 (Agent B)"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    tools = [get_weather, get_forecast]

    system_prompt = """당신은 현실적이고 신중한 성격의 안전 전문가입니다.

성격:
- 모든 가능성을 고려하여 신중하게 판단합니다
- 리스크와 대비책을 먼저 생각합니다
- 데이터를 기반으로 객관적인 분석을 제공합니다
- "하지만", "고려해야 할 점은" 같은 균형잡힌 표현을 사용합니다

답변 원칙:
- 도구를 사용하여 날씨 정보와 예보를 모두 확인합니다
- 날씨 변화 가능성, 미세먼지, 안전 요소를 꼼꼼히 점검합니다
- 장단점을 모두 명확히 제시합니다
- 대안과 플랜B를 항상 준비합니다
- 답변 마지막에 "[추천: 신중하게]" 또는 "[추천: 재고하세요]" 같은 태그를 붙입니다"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


# ============================================================================
# 의사결정 로직
# ============================================================================

def make_final_decision(optimistic_response: str, realistic_response: str) -> str:
    """두 Agent의 답변을 종합하여 최종 의사결정을 내립니다.

    Args:
        optimistic_response: 낙관적 Agent의 답변
        realistic_response: 현실적 Agent의 답변

    Returns:
        종합 의사결정 결과
    """
    # 간단한 키워드 기반 분석
    positive_keywords = ["추천: 가세요", "추천: 좋아요", "좋은", "완벽", "최적"]
    negative_keywords = ["추천: 재고", "추천: 신중", "위험", "주의", "비"]

    optimistic_score = sum(1 for kw in positive_keywords if kw in optimistic_response)
    realistic_concern = sum(1 for kw in negative_keywords if kw in realistic_response)

    decision = "\n" + "=" * 70
    decision += "\n  최종 의사결정 (Agent A + Agent B 종합)\n"
    decision += "=" * 70

    if realistic_concern == 0:
        decision += "\n\n  결론: 소풍 가세요!"
        decision += "\n  이유: 낙관적 Agent와 현실적 Agent 모두 긍정적입니다."
        decision += "\n  두 전문가 모두 동의하므로 안심하고 야외 활동을 즐기세요."
    elif optimistic_score > realistic_concern:
        decision += "\n\n  결론: 소풍 가되, 준비를 철저히 하세요!"
        decision += "\n  이유: 전반적으로 긍정적이지만 일부 주의사항이 있습니다."
        decision += "\n  현실적 Agent의 조언을 참고하여 만반의 준비를 하세요."
    else:
        decision += "\n\n  결론: 실내 활동을 고려해 보세요."
        decision += "\n  이유: 현실적 Agent가 여러 우려사항을 제기했습니다."
        decision += "\n  날씨가 확실히 좋아질 때까지 기다리는 것도 좋은 방법입니다."

    decision += "\n\n  팁: 두 관점을 모두 참고하여 자신에게 맞는 결정을 내리세요."
    decision += "\n" + "=" * 70

    return decision


# ============================================================================
# 테스트 함수
# ============================================================================

def test_multi_agent_conversation():
    """멀티 에이전트 대화 테스트"""
    print("=" * 70)
    print("  멀티 에이전트 대화 테스트")
    print("=" * 70)

    # Agent 생성
    agent_optimistic = create_optimistic_agent()
    agent_realistic = create_realistic_agent()

    # 테스트 질문
    question = "서울 날씨 좋은데 소풍 갈까?"

    print(f"\n  공통 질문: {question}")
    print("=" * 70)

    # Agent A (낙관적) 실행
    print("\n  Agent A (낙관적/긍정적):")
    print("-" * 70)

    try:
        result_a = agent_optimistic.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        response_a = result_a["messages"][-1].content
        print(f"\n{response_a}")
    except Exception as e:
        response_a = f"(Agent A 오류: {e})"
        print(response_a)

    print("\n" + "=" * 70)

    # Agent B (현실적) 실행
    print("\n  Agent B (현실적/신중):")
    print("-" * 70)

    try:
        result_b = agent_realistic.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        response_b = result_b["messages"][-1].content
        print(f"\n{response_b}")
    except Exception as e:
        response_b = f"(Agent B 오류: {e})"
        print(response_b)

    # 최종 의사결정
    decision = make_final_decision(response_a, response_b)
    print(decision)


def test_multiple_questions():
    """여러 질문으로 멀티 에이전트 테스트"""
    print("\n" + "=" * 70)
    print("  여러 질문으로 멀티 에이전트 비교")
    print("=" * 70)

    agent_optimistic = create_optimistic_agent()
    agent_realistic = create_realistic_agent()

    questions = [
        "서울 날씨 좋은데 소풍 갈까?",
        "이번 주말에 부산으로 여행 갈까?",
        "비 오는 날 제주도 가도 괜찮을까?",
    ]

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"  질문: {question}")
        print("=" * 70)

        try:
            result_a = agent_optimistic.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            result_b = agent_realistic.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            print(f"\n  Agent A (긍정적): {result_a['messages'][-1].content[:200]}...")
            print(f"\n  Agent B (신중한): {result_b['messages'][-1].content[:200]}...")

        except Exception as e:
            print(f"\n  오류 발생: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("  Part 3: 멀티 에이전트 대화 - 실습 과제 3 해답")
    print("=" * 70)

    try:
        test_multi_agent_conversation()
    except Exception as e:
        print(f"\n  테스트 실패: {e}")
        print("(API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    print("\n" + "=" * 70)
    print("  추가 학습 포인트:")
    print("  1. LLM 기반 의사결정 (종합 Agent 추가)")
    print("  2. 더 많은 페르소나 추가 (모험가, 경제전문가 등)")
    print("  3. 가중 투표 시스템 구현")
    print("  4. Part 7의 Multi-Agent 시스템으로 확장")
    print("=" * 70)


if __name__ == "__main__":
    main()
