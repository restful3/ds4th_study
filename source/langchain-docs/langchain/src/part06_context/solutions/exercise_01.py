"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context - 실습 과제 1 해답
================================================================================

과제: 시간대별 Agent
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. 현재 시간에 따라 다른 도구 제공
2. 오전/오후/저녁 시간대별 Agent 동작 변경
3. Context로 시간 정보 전달

학습 목표:
- Context 기반 동적 도구 선택
- 시간대별 로직 분기
- state_modifier 활용

================================================================================
"""

from datetime import datetime, time
from typing import Literal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ============================================================================
# 시간대별 도구 정의
# ============================================================================

# 아침 도구들 (6:00 - 12:00)
@tool
def morning_briefing() -> str:
    """아침 브리핑을 제공합니다. 날씨, 뉴스, 일정 요약을 포함합니다."""
    return """
    ☀️ 아침 브리핑
    ━━━━━━━━━━━━━━━━━━━━━━
    🌤️  날씨: 맑음, 15°C
    📰 주요 뉴스: AI 기술 발전 가속화
    📅 오늘 일정:
       - 09:00 팀 미팅
       - 14:00 프로젝트 리뷰
    💪 오늘의 목표: 생산적인 하루 보내기!
    """


@tool
def breakfast_recipe() -> str:
    """간단한 아침 식사 레시피를 추천합니다."""
    return """
    🍳 추천 아침 메뉴: 에그 베네딕트

    재료:
    - 잉글리시 머핀 2개
    - 계란 2개
    - 베이컨 2줄
    - 홀랜다이즈 소스

    조리 시간: 15분
    간단하고 영양가 높은 아침 식사입니다!
    """


@tool
def morning_exercise() -> str:
    """아침 운동 루틴을 제공합니다."""
    return """
    🏃 아침 운동 루틴 (20분)

    1. 스트레칭 (5분)
    2. 조깅 (10분)
    3. 복근 운동 (5분)

    ⚡ 상쾌한 하루를 시작하세요!
    """


# 오후 도구들 (12:00 - 18:00)
@tool
def productivity_tips() -> str:
    """업무 생산성 향상 팁을 제공합니다."""
    return """
    💼 생산성 향상 팁

    1. ⏰ 포모도로 기법 (25분 집중 + 5분 휴식)
    2. 📝 To-Do 리스트 우선순위화
    3. 🚫 알림 끄고 딥워크 시간 확보
    4. ☕ 적절한 휴식과 간식
    5. 🎯 하나의 작업에 집중
    """


@tool
def lunch_recommendation() -> str:
    """점심 메뉴를 추천합니다."""
    return """
    🍱 오늘의 점심 추천

    1. 연어 샐러드 볼 - 건강하고 가벼운 선택
    2. 치킨 샌드위치 - 간편하고 든든한 한 끼
    3. 비빔밥 - 영양 균형이 완벽한 메뉴

    💡 오후 업무를 위해 과식은 피하세요!
    """


@tool
def meeting_scheduler() -> str:
    """회의 일정을 관리합니다."""
    return """
    📅 오후 회의 일정

    - 14:00 프로젝트 진행 상황 리뷰
    - 15:30 클라이언트 미팅 (온라인)
    - 17:00 팀 브레인스토밍

    ✅ 회의 전 준비사항을 확인하세요!
    """


# 저녁 도구들 (18:00 - 22:00)
@tool
def dinner_recipe() -> str:
    """저녁 식사 레시피를 추천합니다."""
    return """
    🍝 저녁 메뉴 추천: 크림 파스타

    재료:
    - 파스타 면 200g
    - 생크림 200ml
    - 베이컨, 마늘, 양파
    - 파마산 치즈

    조리 시간: 25분
    따뜻하고 든든한 저녁 식사!
    """


@tool
def evening_relaxation() -> str:
    """저녁 휴식 활동을 추천합니다."""
    return """
    🌙 저녁 휴식 활동

    1. 📚 독서 (30분) - 자기계발서 추천
    2. 🎵 음악 감상 - 잔잔한 재즈/클래식
    3. 🧘 명상 (10분) - 하루 정리
    4. 🛀 따뜻한 샤워
    5. ☕ 허브티 한 잔

    편안한 저녁 보내세요!
    """


@tool
def tomorrow_preparation() -> str:
    """내일을 위한 준비 체크리스트를 제공합니다."""
    return """
    📋 내일 준비 체크리스트

    ✅ 내일 일정 확인
    ✅ 필요한 자료 준비
    ✅ 옷과 가방 준비
    ✅ 아침 식사 재료 체크
    ✅ 충분한 수면 (7-8시간)

    좋은 밤 되세요! 🌟
    """


# 심야 도구들 (22:00 - 06:00)
@tool
def sleep_guide() -> str:
    """수면 가이드를 제공합니다."""
    return """
    😴 숙면을 위한 가이드

    1. 💡 조명 낮추기
    2. 📱 전자기기 멀리하기
    3. 🌡️  적정 온도 유지 (18-20°C)
    4. 🧘 호흡 명상 (4-7-8 호흡법)
    5. 📚 가벼운 독서

    편안한 밤 되세요! 🌙
    """


@tool
def emergency_contact() -> str:
    """긴급 연락처 정보를 제공합니다."""
    return """
    🚨 긴급 연락처

    - 응급의료: 119
    - 경찰: 112
    - 소방: 119
    - 24시간 병원: 1339

    안전한 밤 되세요!
    """


# ============================================================================
# 시간대 판단
# ============================================================================

def get_time_period() -> Literal["morning", "afternoon", "evening", "night"]:
    """현재 시간대 반환"""
    now = datetime.now().time()

    if time(6, 0) <= now < time(12, 0):
        return "morning"
    elif time(12, 0) <= now < time(18, 0):
        return "afternoon"
    elif time(18, 0) <= now < time(22, 0):
        return "evening"
    else:
        return "night"


def get_tools_for_period(period: str) -> list:
    """시간대별 도구 반환"""
    tools_map = {
        "morning": [morning_briefing, breakfast_recipe, morning_exercise],
        "afternoon": [productivity_tips, lunch_recommendation, meeting_scheduler],
        "evening": [dinner_recipe, evening_relaxation, tomorrow_preparation],
        "night": [sleep_guide, emergency_contact],
    }
    return tools_map.get(period, [])


def get_greeting_for_period(period: str) -> str:
    """시간대별 인사말"""
    greetings = {
        "morning": "☀️ 좋은 아침입니다!",
        "afternoon": "🌤️  좋은 오후입니다!",
        "evening": "🌆 좋은 저녁입니다!",
        "night": "🌙 늦은 시간이네요!",
    }
    return greetings.get(period, "안녕하세요!")


# ============================================================================
# 시간대별 Agent 생성
# ============================================================================

def create_time_based_agent(custom_time: datetime = None):
    """시간대별 Agent 생성"""

    # 시간 설정 (테스트용)
    if custom_time:
        target_time = custom_time.time()
        if time(6, 0) <= target_time < time(12, 0):
            period = "morning"
        elif time(12, 0) <= target_time < time(18, 0):
            period = "afternoon"
        elif time(18, 0) <= target_time < time(22, 0):
            period = "evening"
        else:
            period = "night"
    else:
        period = get_time_period()

    # 시간대별 도구 선택
    tools = get_tools_for_period(period)

    # 시간대별 시스템 프롬프트
    period_names = {
        "morning": "아침",
        "afternoon": "오후",
        "evening": "저녁",
        "night": "심야"
    }

    system_prompt = f"""{get_greeting_for_period(period)}

현재 시간대: {period_names[period]}

저는 시간대에 맞는 도움을 드리는 AI 어시스턴트입니다.
{period_names[period]} 시간대에 유용한 도구들을 사용할 수 있습니다:

"""

    for tool_func in tools:
        system_prompt += f"- {tool_func.name}: {tool_func.description}\n"

    system_prompt += "\n무엇을 도와드릴까요?"

    # 모델 생성
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Agent 생성
    agent = create_react_agent(
        model,
        tools,
        state_modifier=system_prompt
    )

    return agent, period


# ============================================================================
# 테스트 함수
# ============================================================================

def test_time_based_agent():
    """시간대별 Agent 테스트"""
    print("=" * 70)
    print("⏰ 시간대별 Agent 테스트")
    print("=" * 70)

    # 각 시간대별 테스트
    test_times = [
        (datetime(2024, 1, 1, 8, 0), "morning", "아침 브리핑 해줘"),
        (datetime(2024, 1, 1, 14, 0), "afternoon", "생산성 향상 팁을 알려줘"),
        (datetime(2024, 1, 1, 19, 0), "evening", "저녁 메뉴 추천해줘"),
        (datetime(2024, 1, 1, 23, 0), "night", "잠을 잘 자는 방법을 알려줘"),
    ]

    for test_time, expected_period, question in test_times:
        print(f"\n{'=' * 70}")
        print(f"🕐 테스트 시간: {test_time.strftime('%H:%M')}")
        print(f"📍 기대 시간대: {expected_period}")
        print("=" * 70)

        agent, period = create_time_based_agent(custom_time=test_time)

        print(f"✅ 실제 시간대: {period}")
        assert period == expected_period, f"시간대 불일치: {period} != {expected_period}"

        print(f"\n👤 질문: {question}")

        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        final_message = result["messages"][-1]
        print(f"\n🤖 Agent 응답:\n{final_message.content}\n")


def test_current_time():
    """현재 시간으로 테스트"""
    print("\n" + "=" * 70)
    print("🕐 현재 시간 Agent 테스트")
    print("=" * 70)

    now = datetime.now()
    print(f"\n현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    agent, period = create_time_based_agent()

    period_names = {
        "morning": "아침",
        "afternoon": "오후",
        "evening": "저녁",
        "night": "심야"
    }

    print(f"시간대: {period_names[period]}")
    print(f"인사말: {get_greeting_for_period(period)}")

    # 시간대별 추천 질문
    recommended_questions = {
        "morning": "아침 브리핑을 해줘",
        "afternoon": "생산성 팁을 알려줘",
        "evening": "저녁 메뉴를 추천해줘",
        "night": "숙면 가이드를 알려줘",
    }

    question = recommended_questions[period]
    print(f"\n👤 추천 질문: {question}")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    final_message = result["messages"][-1]
    print(f"\n🤖 Agent 응답:\n{final_message.content}")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 시간대별 Agent - 대화형 모드")
    print("=" * 70)

    now = datetime.now()
    agent, period = create_time_based_agent()

    period_names = {
        "morning": "아침",
        "afternoon": "오후",
        "evening": "저녁",
        "night": "심야"
    }

    print(f"\n현재 시간: {now.strftime('%H:%M')}")
    print(f"시간대: {period_names[period]}")
    print(f"{get_greeting_for_period(period)}")

    # 사용 가능한 도구 표시
    tools = get_tools_for_period(period)
    print(f"\n사용 가능한 기능:")
    for tool_func in tools:
        print(f"  - {tool_func.name}")

    print("\n명령어:")
    print("  /time - 현재 시간대 정보")
    print("  /tools - 사용 가능한 도구")
    print("  /quit - 종료")
    print("=" * 70)

    while True:
        try:
            user_input = input(f"\n[{period_names[period]}] 👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 좋은 하루 보내세요!")
                break

            elif user_input == "/time":
                now = datetime.now()
                print(f"\n⏰ 현재 시간: {now.strftime('%H:%M:%S')}")
                print(f"📍 시간대: {period_names[period]}")
                continue

            elif user_input == "/tools":
                print(f"\n🔧 사용 가능한 도구:")
                for tool_func in tools:
                    print(f"  - {tool_func.name}: {tool_func.description}")
                continue

            # 일반 질문
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

            final_message = result["messages"][-1]
            print(f"\n🤖 {final_message.content}")

        except KeyboardInterrupt:
            print("\n👋 좋은 하루 보내세요!")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("⏰ Part 6: 시간대별 Agent - 실습 과제 1 해답")
    print("=" * 70)

    try:
        # 테스트 1: 시간대별 테스트
        test_time_based_agent()

        # 테스트 2: 현재 시간
        test_current_time()

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
    print("  1. Context 기반 동적 도구 선택")
    print("  2. 시간대별 로직 분기")
    print("  3. state_modifier로 시스템 프롬프트 커스터마이징")
    print("  4. 조건부 Agent 생성")
    print("\n💡 추가 학습:")
    print("  1. 위치 기반 Agent (GPS)")
    print("  2. 날씨 기반 도구 선택")
    print("  3. 사용자 활동 패턴 학습")
    print("  4. 다중 Context 조합")
    print("=" * 70)


if __name__ == "__main__":
    main()
