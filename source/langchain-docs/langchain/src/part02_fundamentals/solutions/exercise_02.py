"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: Fundamentals - 실습 과제 2 해답
================================================================================

과제: 날씨 조회 도구
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. Pydantic 모델로 파라미터 정의
2. 도시, 단위(섭씨/화씨) 파라미터
3. Mock 데이터 또는 실제 API 연동

학습 목표:
- Pydantic 모델 사용
- 복잡한 파라미터 구조화
- 선택적 파라미터 처리

================================================================================
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import random

# ============================================================================
# Pydantic 모델 정의
# ============================================================================

class WeatherQuery(BaseModel):
    """날씨 조회 요청 모델"""

    city: str = Field(
        description="조회할 도시 이름 (예: 서울, 부산, Tokyo)"
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="온도 단위. 'celsius' (섭씨) 또는 'fahrenheit' (화씨)"
    )
    include_forecast: bool = Field(
        default=False,
        description="3일 예보 포함 여부"
    )


# ============================================================================
# Mock 날씨 데이터
# ============================================================================

MOCK_WEATHER_DATA = {
    "서울": {"temp_c": 15, "condition": "맑음", "humidity": 45},
    "부산": {"temp_c": 18, "condition": "흐림", "humidity": 60},
    "대구": {"temp_c": 16, "condition": "비", "humidity": 75},
    "인천": {"temp_c": 14, "condition": "맑음", "humidity": 50},
    "광주": {"temp_c": 17, "condition": "구름 많음", "humidity": 55},
    "Tokyo": {"temp_c": 20, "condition": "Sunny", "humidity": 40},
    "New York": {"temp_c": 10, "condition": "Cloudy", "humidity": 65},
    "London": {"temp_c": 8, "condition": "Rainy", "humidity": 80},
}


def celsius_to_fahrenheit(celsius: float) -> float:
    """섭씨를 화씨로 변환"""
    return (celsius * 9/5) + 32


def generate_forecast(base_temp: float, days: int = 3) -> list:
    """간단한 예보 생성 (Mock)"""
    forecast = []
    for i in range(1, days + 1):
        temp_variation = random.randint(-3, 3)
        forecast.append({
            "day": f"Day {i}",
            "temp": base_temp + temp_variation,
            "condition": random.choice(["맑음", "흐림", "비", "구름 많음"])
        })
    return forecast


# ============================================================================
# 날씨 도구 정의
# ============================================================================

@tool(args_schema=WeatherQuery)
def get_weather(city: str, unit: str = "celsius", include_forecast: bool = False) -> str:
    """지정된 도시의 현재 날씨 정보를 조회합니다.

    이 도구는 도시 이름을 받아 현재 날씨, 온도, 습도 정보를 반환합니다.
    선택적으로 3일 예보도 포함할 수 있습니다.

    Args:
        city: 조회할 도시 이름
        unit: 온도 단위 ('celsius' 또는 'fahrenheit')
        include_forecast: 3일 예보 포함 여부

    Returns:
        날씨 정보 문자열
    """
    # 도시 데이터 조회
    weather_data = MOCK_WEATHER_DATA.get(
        city,
        {"temp_c": 20, "condition": "알 수 없음", "humidity": 50}
    )

    # 온도 변환
    temp = weather_data["temp_c"]
    if unit == "fahrenheit":
        temp = celsius_to_fahrenheit(temp)
        temp_unit = "°F"
    else:
        temp_unit = "°C"

    # 기본 정보 구성
    result = f"""
🌤️  {city} 날씨 정보
━━━━━━━━━━━━━━━━━━━━
📍 위치: {city}
🌡️  온도: {temp:.1f}{temp_unit}
☁️  날씨: {weather_data['condition']}
💧 습도: {weather_data['humidity']}%
"""

    # 예보 포함
    if include_forecast:
        result += "\n📅 3일 예보:\n"
        forecast = generate_forecast(weather_data["temp_c"])
        for day_data in forecast:
            day_temp = day_data["temp"]
            if unit == "fahrenheit":
                day_temp = celsius_to_fahrenheit(day_temp)
            result += f"  {day_data['day']}: {day_temp:.1f}{temp_unit}, {day_data['condition']}\n"

    return result.strip()


# ============================================================================
# 테스트 코드
# ============================================================================

def test_weather_tool():
    """날씨 도구 테스트"""
    print("=" * 70)
    print("🧪 날씨 도구 테스트")
    print("=" * 70)

    # 테스트 1: 기본 조회 (섭씨)
    print("\n📝 테스트 1: 서울 날씨 (섭씨)")
    result = get_weather.invoke({"city": "서울", "unit": "celsius"})
    print(result)

    # 테스트 2: 화씨 단위
    print("\n📝 테스트 2: 부산 날씨 (화씨)")
    result = get_weather.invoke({"city": "부산", "unit": "fahrenheit"})
    print(result)

    # 테스트 3: 예보 포함
    print("\n📝 테스트 3: 대구 날씨 (예보 포함)")
    result = get_weather.invoke({
        "city": "대구",
        "unit": "celsius",
        "include_forecast": True
    })
    print(result)

    # 테스트 4: 없는 도시
    print("\n📝 테스트 4: 알 수 없는 도시")
    result = get_weather.invoke({"city": "Unknown City"})
    print(result)

    print("\n" + "=" * 70)
    print("✅ 모든 테스트 통과!")
    print("=" * 70)


def test_with_llm():
    """LLM과 통합 테스트"""
    print("\n" + "=" * 70)
    print("🤖 LLM 통합 테스트")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools([get_weather])

    test_questions = [
        "서울의 날씨를 알려줘",
        "부산 날씨를 화씨로 알려줘",
        "도쿄의 3일 예보를 보여줘",
    ]

    for question in test_questions:
        print(f"\n👤 질문: {question}")
        response = model_with_tools.invoke([{"role": "user", "content": question}])

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"🔧 도구 호출: {tool_call['name']}")
            print(f"   파라미터: {tool_call['args']}")

            result = get_weather.invoke(tool_call["args"])
            print(f"\n{result}")
        else:
            print(f"🤖 응답: {response.content}")

    print("\n" + "=" * 70)
    print("✅ LLM 통합 테스트 완료!")
    print("=" * 70)


# ============================================================================
# 실제 API 연동 예시 (주석)
# ============================================================================

"""
실제 OpenWeatherMap API를 사용하려면:

1. API 키 발급: https://openweathermap.org/api

2. 코드 수정:

import requests

def get_real_weather(city: str, api_key: str, unit: str = "celsius"):
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric" if unit == "celsius" else "imperial"
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if response.status_code == 200:
        return {
            "temp": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        }
    else:
        raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
"""


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🌤️  Part 2: 날씨 조회 도구 - 실습 과제 2 해답")
    print("=" * 70)

    # 1. 도구 단위 테스트
    test_weather_tool()

    # 2. LLM 통합 테스트
    try:
        test_with_llm()
    except Exception as e:
        print(f"\n⚠️  LLM 통합 테스트 스킵: {e}")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("💡 추가 학습 포인트:")
    print("  1. 실제 OpenWeatherMap API 연동")
    print("  2. 캐싱으로 API 호출 최소화")
    print("  3. 여러 도시 동시 조회")
    print("  4. 날씨 알림 기능 (비 예보 시)")
    print("=" * 70)


if __name__ == "__main__":
    main()
