"""
================================================================================
Project 1: 날씨 비서 Agent - 도구 테스트
================================================================================

파일명: test_tools.py
설명: tools.py의 날씨 도구 함수들을 테스트합니다.

실행 방법:
    pytest tests/test_tools.py -v

================================================================================
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import sys

# 상위 디렉토리의 tools 모듈 임포트
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import (
    get_weather_data,
    format_weather_response,
    get_weather,
    CITY_NAME_MAP,
    check_api_key,
)


# ============================================================================
# 테스트 픽스처
# ============================================================================

@pytest.fixture
def mock_weather_data():
    """모의 날씨 API 응답 데이터"""
    return {
        "main": {
            "temp": 22.5,
            "feels_like": 21.0,
            "humidity": 65,
        },
        "weather": [
            {
                "description": "맑음",
                "main": "Clear",
            }
        ],
    }


@pytest.fixture
def mock_api_response(mock_weather_data):
    """모의 API 응답 객체"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_weather_data
    return mock_response


# ============================================================================
# 테스트: 도시명 매핑
# ============================================================================

def test_city_name_map():
    """한글 도시명 매핑이 올바른지 테스트"""
    assert CITY_NAME_MAP["서울"] == "Seoul"
    assert CITY_NAME_MAP["부산"] == "Busan"
    assert CITY_NAME_MAP["뉴욕"] == "New York"


# ============================================================================
# 테스트: API 키 확인
# ============================================================================

def test_check_api_key_present():
    """API 키가 있을 때"""
    with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}):
        assert check_api_key() is True


def test_check_api_key_absent():
    """API 키가 없을 때"""
    with patch.dict(os.environ, {}, clear=True):
        assert check_api_key() is False


# ============================================================================
# 테스트: 날씨 데이터 가져오기
# ============================================================================

@patch("tools.requests.get")
def test_get_weather_data_success(mock_get, mock_api_response):
    """날씨 데이터를 성공적으로 가져오는 경우"""
    mock_get.return_value = mock_api_response

    with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}):
        data = get_weather_data("서울")

    assert data is not None
    assert data["main"]["temp"] == 22.5
    assert data["weather"][0]["description"] == "맑음"


@patch("tools.requests.get")
def test_get_weather_data_city_not_found(mock_get):
    """존재하지 않는 도시"""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_get.return_value = mock_response

    with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}):
        data = get_weather_data("아무도시")

    assert data is None


@patch("tools.requests.get")
def test_get_weather_data_timeout(mock_get):
    """API 타임아웃"""
    mock_get.side_effect = TimeoutError("Timeout")

    with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}):
        data = get_weather_data("서울")

    assert data is None


def test_get_weather_data_no_api_key():
    """API 키가 없을 때"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OPENWEATHER_API_KEY가 설정되지 않았습니다"):
            get_weather_data("서울")


# ============================================================================
# 테스트: 응답 포맷팅
# ============================================================================

def test_format_weather_response(mock_weather_data):
    """날씨 응답이 올바르게 포맷되는지 테스트"""
    formatted = format_weather_response(mock_weather_data, "서울")

    assert "서울" in formatted
    assert "22.5°C" in formatted
    assert "65%" in formatted
    assert "맑음" in formatted


# ============================================================================
# 테스트: LangChain 도구
# ============================================================================

@patch("tools.get_weather_data")
def test_get_weather_tool_success(mock_get_data, mock_weather_data):
    """get_weather 도구가 성공적으로 작동하는지 테스트"""
    mock_get_data.return_value = mock_weather_data

    result = get_weather.invoke({"city": "서울"})

    assert "서울" in result
    assert "22.5°C" in result


@patch("tools.get_weather_data")
def test_get_weather_tool_failure(mock_get_data):
    """get_weather 도구가 오류를 처리하는지 테스트"""
    mock_get_data.return_value = None

    result = get_weather.invoke({"city": "아무도시"})

    assert "가져올 수 없습니다" in result


# ============================================================================
# 통합 테스트
# ============================================================================

@pytest.mark.skipif(
    not os.getenv("OPENWEATHER_API_KEY"),
    reason="실제 API 키가 필요합니다"
)
def test_integration_real_api():
    """실제 API를 호출하는 통합 테스트 (API 키 필요)"""
    data = get_weather_data("Seoul")

    assert data is not None
    assert "main" in data
    assert "temp" in data["main"]


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
