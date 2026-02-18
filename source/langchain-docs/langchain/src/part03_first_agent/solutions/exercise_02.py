"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: First Agent - 실습 과제 2 해답
================================================================================

과제: 여행 플래너 Agent
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. 다음 도구 구현:
   - get_weather(city, date): 날씨 조회
   - search_attractions(city): 관광지 검색
   - estimate_budget(city, days): 예산 추정
2. System Prompt로 Agent를 "경험 많은 여행 가이드"로 설정
3. 사용자가 "파리 3일 여행 계획해줘"라고 하면:
   - 날씨 확인
   - 인기 관광지 추천
   - 예산 추정
   - 종합 여행 계획 제시
4. Streaming 모드로 실시간 계획 과정 표시

학습 목표:
- 여러 도구의 순차적 조합
- System Prompt로 전문가 페르소나 설정
- Streaming을 활용한 실시간 출력

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


# ============================================================================
# 도구 정의
# ============================================================================

@tool
def get_weather(city: str, date: str = "오늘") -> str:
    """여행지의 날씨를 조회합니다.

    Args:
        city: 도시 이름 (예: 파리, 도쿄, 서울)
        date: 날짜 (예: 오늘, 내일, 2025-03-15)

    Returns:
        해당 도시의 날씨 정보
    """
    weather_data = {
        "파리": "맑음, 18°C, 습도 55% - 야외 관광하기 좋은 날씨",
        "도쿄": "흐림, 20°C, 습도 65% - 우산 준비 권장",
        "서울": "맑음, 22°C, 습도 60% - 나들이하기 좋은 날씨",
        "뉴욕": "비, 15°C, 습도 85% - 실내 관광 추천",
        "런던": "흐림, 14°C, 습도 75% - 겉옷 필수",
        "방콕": "맑음, 33°C, 습도 80% - 자외선 차단제 필수",
    }
    weather = weather_data.get(city, f"{city}: 맑음, 20°C (일반적 날씨)")
    return f"{city}의 {date} 날씨: {weather}"


@tool
def search_attractions(city: str) -> str:
    """도시의 인기 관광지를 검색합니다.

    Args:
        city: 도시 이름

    Returns:
        해당 도시의 인기 관광지 목록
    """
    attractions_data = {
        "파리": """파리 인기 관광지:
1. 에펠탑 - 파리의 상징, 야경 필수 (입장료: 약 26유로)
2. 루브르 박물관 - 모나리자, 밀로의 비너스 (입장료: 약 17유로)
3. 개선문 - 샹젤리제 거리의 끝 (입장료: 약 13유로)
4. 몽마르뜨 언덕 - 사크레쾨르 대성당, 예술가 거리
5. 세느강 유람선 - 파리 전경 감상 (약 15유로)""",
        "도쿄": """도쿄 인기 관광지:
1. 센소지 - 아사쿠사의 역사적 사찰 (무료)
2. 시부야 스크램블 교차로 - 도쿄의 아이콘
3. 메이지 신궁 - 도심 속 평화로운 신사 (무료)
4. 도쿄 타워 - 전망대 (약 1,200엔)
5. 아키하바라 - 전자제품과 서브컬처의 중심""",
        "서울": """서울 인기 관광지:
1. 경복궁 - 조선 왕조의 정궁 (입장료: 3,000원)
2. 명동 - 쇼핑과 길거리 음식
3. 남산타워 - 서울 전경 (입장료: 16,000원)
4. 북촌 한옥마을 - 전통 한옥 체험
5. 이태원 - 다국적 음식과 문화""",
    }
    result = attractions_data.get(city, f"{city}의 관광지 정보: 현지 관광 안내소를 방문해 보세요.")
    return result


@tool
def estimate_budget(city: str, days: int = 3) -> str:
    """여행 예산을 추정합니다.

    Args:
        city: 여행 도시
        days: 여행 일수 (기본 3일)

    Returns:
        예상 여행 예산 내역
    """
    budget_data = {
        "파리": {"accommodation": 150, "food": 60, "transport": 20, "attractions": 40, "currency": "유로"},
        "도쿄": {"accommodation": 120, "food": 50, "transport": 15, "attractions": 30, "currency": "유로 환산"},
        "서울": {"accommodation": 80, "food": 30, "transport": 10, "attractions": 15, "currency": "유로 환산"},
    }

    budget = budget_data.get(city, {"accommodation": 100, "food": 40, "transport": 15, "attractions": 25, "currency": "유로 환산"})

    daily_total = budget["accommodation"] + budget["food"] + budget["transport"] + budget["attractions"]
    total = daily_total * days

    return f"""{city} {days}일 여행 예상 예산 ({budget['currency']}):

  1일 예산:
  - 숙박: {budget['accommodation']}유로/일
  - 식비: {budget['food']}유로/일
  - 교통: {budget['transport']}유로/일
  - 관광: {budget['attractions']}유로/일
  - 소계: {daily_total}유로/일

  총 예산: {daily_total}유로 x {days}일 = {total}유로

  팁: 환율 변동과 개인 쇼핑 비용은 별도입니다."""


# ============================================================================
# Agent 생성
# ============================================================================

def create_travel_planner_agent():
    """여행 플래너 Agent 생성"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    tools = [get_weather, search_attractions, estimate_budget]

    # System Prompt - "경험 많은 여행 가이드" 페르소나
    system_prompt = """당신은 20년 경력의 전문 여행 가이드입니다.

사용자의 여행을 완벽하게 계획해 주세요.

여행 계획 순서 (항상 이 순서를 따르세요):
1. 먼저 get_weather로 여행지 날씨를 확인합니다
2. search_attractions로 인기 관광지를 검색합니다
3. estimate_budget으로 예산을 추정합니다
4. 모든 정보를 종합하여 완성된 여행 계획을 제시합니다

답변 형식:
- 날씨 정보와 옷차림 조언
- 추천 관광지와 방문 순서
- 예산 요약
- 여행 팁과 주의사항

말투:
- 전문적이면서 친근한 톤
- 실용적인 팁 위주"""

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


# ============================================================================
# 테스트 함수
# ============================================================================

def test_travel_planner():
    """여행 플래너 Agent 테스트"""
    print("=" * 70)
    print("  여행 플래너 Agent 테스트")
    print("=" * 70)

    agent = create_travel_planner_agent()

    test_cases = [
        "파리 3일 여행 계획해줘",
        "도쿄 2일 여행 추천해줘",
        "서울 1일 관광 코스를 짜줘",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"  테스트 {i}: {question}")
        print("=" * 70)

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )

            final_message = result["messages"][-1]
            print(f"\n  Agent 응답:\n{final_message.content}\n")

        except Exception as e:
            print(f"\n  오류 발생: {e}\n")

    print("=" * 70)
    print("  모든 테스트 완료!")
    print("=" * 70)


def test_streaming_mode():
    """Streaming 모드 테스트"""
    print("\n" + "=" * 70)
    print("  Streaming 모드 테스트")
    print("=" * 70)

    agent = create_travel_planner_agent()

    question = "파리 3일 여행 계획해줘"
    print(f"\n  사용자: {question}")
    print("  Agent: (실시간 스트리밍)\n")

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values"
        ):
            latest_message = chunk["messages"][-1]

            # 도구 호출 표시
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    print(f"   [도구 호출] {tc['name']}({tc['args']})")

            # 도구 결과 표시
            elif latest_message.__class__.__name__ == "ToolMessage":
                content_preview = latest_message.content[:80].replace("\n", " ")
                print(f"   [도구 결과] {content_preview}...")

            # 최종 답변 표시
            elif hasattr(latest_message, "content") and latest_message.content:
                if not (hasattr(latest_message, "tool_calls") and latest_message.tool_calls):
                    print(f"\n   [최종 답변]\n{latest_message.content}")

    except Exception as e:
        print(f"\n  스트리밍 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("  Part 3: 여행 플래너 Agent - 실습 과제 2 해답")
    print("=" * 70)

    try:
        test_travel_planner()
        test_streaming_mode()
    except Exception as e:
        print(f"\n  테스트 실패: {e}")
        print("(API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    print("\n" + "=" * 70)
    print("  추가 학습 포인트:")
    print("  1. 실제 날씨 API 연동 (OpenWeatherMap)")
    print("  2. Google Places API로 실제 관광지 검색")
    print("  3. 환율 API로 실시간 예산 계산")
    print("  4. 사용자 선호도 기반 맞춤 추천")
    print("=" * 70)


if __name__ == "__main__":
    main()
