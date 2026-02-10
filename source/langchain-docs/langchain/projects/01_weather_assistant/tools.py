"""
================================================================================
Project 1: 날씨 비서 Agent - 도구 모듈
================================================================================

파일명: tools.py
설명: OpenWeatherMap API를 사용하는 날씨 조회 도구

================================================================================
"""

import os
import requests
from typing import Dict, Optional
from langchain.tools import tool


# ============================================================================
# 한글 도시명 매핑
# ============================================================================

CITY_NAME_MAP = {
    "서울": "Seoul",
    "부산": "Busan",
    "대구": "Daegu",
    "인천": "Incheon",
    "광주": "Gwangju",
    "대전": "Daejeon",
    "울산": "Ulsan",
    "제주": "Jeju",
    "뉴욕": "New York",
    "런던": "London",
    "파리": "Paris",
    "도쿄": "Tokyo",
    "베이징": "Beijing",
    "상하이": "Shanghai",
}


# ============================================================================
# 날씨 API 헬퍼 함수
# ============================================================================

def get_weather_data(city: str) -> Optional[Dict]:
    """
    OpenWeatherMap API로부터 날씨 데이터를 가져옵니다.

    Args:
        city: 도시 이름 (한글 또는 영문)

    Returns:
        날씨 데이터 딕셔너리 또는 None (오류 시)
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENWEATHER_API_KEY가 설정되지 않았습니다. "
            ".env 파일에 API 키를 추가해주세요."
        )

    # 한글 도시명을 영문으로 변환
    city_english = CITY_NAME_MAP.get(city, city)

    # API 엔드포인트
    url = "http://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city_english,
        "appid": api_key,
        "units": "metric",  # 섭씨 온도
        "lang": "kr",       # 한국어 설명
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        print(f"⚠️ 날씨 API 응답 시간 초과: {city}")
        return None

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"⚠️ 도시를 찾을 수 없습니다: {city}")
        else:
            print(f"⚠️ HTTP 오류: {e}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"⚠️ 네트워크 오류: {e}")
        return None


def format_weather_response(data: Dict, city: str) -> str:
    """
    날씨 데이터를 사람이 읽기 쉬운 형식으로 변환합니다.

    Args:
        data: OpenWeatherMap API 응답 데이터
        city: 도시 이름

    Returns:
        포맷된 날씨 정보 문자열
    """
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    description = data["weather"][0]["description"]

    # 날씨 이모지 매핑
    weather_emoji = {
        "맑음": "☀️",
        "구름": "☁️",
        "비": "🌧️",
        "눈": "❄️",
        "안개": "🌫️",
        "천둥": "⚡",
    }

    emoji = "🌤️"  # 기본 이모지
    for keyword, emo in weather_emoji.items():
        if keyword in description:
            emoji = emo
            break

    # 응답 포맷팅
    response = f"""
{emoji} {city} 날씨 정보:
━━━━━━━━━━━━━━━━━━━━━━
🌡️ 기온: {temp}°C (체감 {feels_like}°C)
💧 습도: {humidity}%
🌈 날씨: {description}
    """.strip()

    return response


# ============================================================================
# LangChain 도구
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """
    주어진 도시의 현재 날씨를 조회합니다.

    Args:
        city: 도시 이름 (예: "서울", "Seoul", "뉴욕", "New York")

    Returns:
        현재 날씨 정보 (온도, 습도, 날씨 상태)

    Examples:
        >>> get_weather("서울")
        "☀️ 서울 날씨 정보:\n🌡️ 기온: 22°C..."

        >>> get_weather("New York")
        "☁️ New York 날씨 정보:\n🌡️ 기온: 15°C..."
    """
    # 날씨 데이터 가져오기
    data = get_weather_data(city)

    if data is None:
        return f"❌ '{city}'의 날씨 정보를 가져올 수 없습니다. 도시 이름을 확인해주세요."

    # 응답 포맷팅
    return format_weather_response(data, city)


@tool
def get_forecast(city: str) -> str:
    """
    주어진 도시의 5일 날씨 예보를 조회합니다. (도전 과제)

    Args:
        city: 도시 이름

    Returns:
        5일 날씨 예보 정보

    Note:
        이 기능은 도전 과제입니다. 구현하려면 OpenWeatherMap의
        5 day forecast API를 사용하세요:
        https://openweathermap.org/forecast5
    """
    return "🚧 5일 예보 기능은 아직 구현되지 않았습니다. 도전 과제로 구현해보세요!"


@tool
def compare_weather(city1: str, city2: str) -> str:
    """
    두 도시의 날씨를 비교합니다. (도전 과제)

    Args:
        city1: 첫 번째 도시 이름
        city2: 두 번째 도시 이름

    Returns:
        두 도시의 날씨 비교 정보
    """
    data1 = get_weather_data(city1)
    data2 = get_weather_data(city2)

    if data1 is None or data2 is None:
        return "❌ 두 도시의 날씨 정보를 가져올 수 없습니다."

    temp1 = data1["main"]["temp"]
    temp2 = data2["main"]["temp"]
    desc1 = data1["weather"][0]["description"]
    desc2 = data2["weather"][0]["description"]

    temp_diff = abs(temp1 - temp2)
    warmer = city1 if temp1 > temp2 else city2

    comparison = f"""
📊 {city1} vs {city2} 날씨 비교:
━━━━━━━━━━━━━━━━━━━━━━
{city1}: {temp1}°C, {desc1}
{city2}: {temp2}°C, {desc2}

🌡️ 온도 차이: {temp_diff:.1f}°C
더 따뜻한 곳: {warmer}
    """.strip()

    return comparison


# ============================================================================
# 유틸리티 함수
# ============================================================================

def check_api_key() -> bool:
    """
    OPENWEATHER_API_KEY가 설정되어 있는지 확인합니다.

    Returns:
        API 키가 설정되어 있으면 True, 아니면 False
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    return api_key is not None and len(api_key) > 0


if __name__ == "__main__":
    # 도구 테스트
    print("=" * 70)
    print("🧪 날씨 도구 테스트")
    print("=" * 70)

    # API 키 확인
    if not check_api_key():
        print("❌ OPENWEATHER_API_KEY가 설정되지 않았습니다.")
        print("📝 .env 파일에 API 키를 추가하세요:")
        print("   OPENWEATHER_API_KEY=your-api-key-here")
        exit(1)

    print("✅ API 키가 설정되어 있습니다.\n")

    # 테스트 1: 서울 날씨
    print("테스트 1: 서울 날씨")
    print(get_weather.invoke({"city": "서울"}))
    print()

    # 테스트 2: 뉴욕 날씨
    print("\n테스트 2: 뉴욕 날씨")
    print(get_weather.invoke({"city": "뉴욕"}))
    print()

    # 테스트 3: 존재하지 않는 도시
    print("\n테스트 3: 존재하지 않는 도시")
    print(get_weather.invoke({"city": "아무도시"}))
