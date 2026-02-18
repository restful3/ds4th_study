"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: Fundamentals - 실습 과제 3 해답
================================================================================

과제: 실용적인 Tool 만들기
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. ReservationInput Pydantic 스키마 정의:
   - restaurant_name: 식당 이름
   - date: 예약 날짜 (YYYY-MM-DD 형식)
   - time: 예약 시간 (HH:MM 형식)
   - party_size: 인원 수 (1-20명)
   - special_requests: 특별 요청사항 (선택)
2. @tool 데코레이터로 make_reservation 도구 구현
3. 도구를 직접 호출하여 테스트

학습 목표:
- Pydantic BaseModel로 복잡한 입력 스키마 정의
- Field descriptions로 파라미터 가이드 제공
- 검증(validation)이 작동하는 도구 구현
- LLM과의 Tool Calling 연동

================================================================================
"""

import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# Pydantic 스키마 정의
# ============================================================================

class ReservationInput(BaseModel):
    """식당 예약을 위한 입력 스키마"""

    restaurant_name: str = Field(
        description="예약할 식당 이름 (예: '스시 오마카세', '라 피자')"
    )
    date: str = Field(
        description="예약 날짜 (YYYY-MM-DD 형식, 예: '2025-03-15')"
    )
    time: str = Field(
        description="예약 시간 (HH:MM 형식, 예: '18:30')"
    )
    party_size: int = Field(
        ge=1,
        le=20,
        description="예약 인원 수 (1-20명)"
    )
    special_requests: Optional[str] = Field(
        default=None,
        description="특별 요청사항 (예: '창가 자리', '알레르기 있음', '생일 케이크')"
    )

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        """날짜 형식 검증 (YYYY-MM-DD)"""
        try:
            parsed_date = datetime.strptime(v, "%Y-%m-%d")
            if parsed_date.date() < datetime.now().date():
                raise ValueError("과거 날짜로는 예약할 수 없습니다")
            return v
        except ValueError as e:
            if "과거 날짜" in str(e):
                raise
            raise ValueError(f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요: {v}")

    @field_validator("time")
    @classmethod
    def validate_time(cls, v):
        """시간 형식 검증 (HH:MM)"""
        try:
            parsed_time = datetime.strptime(v, "%H:%M")
            hour = parsed_time.hour
            if hour < 10 or hour > 22:
                raise ValueError("영업 시간(10:00~22:00) 내에서만 예약 가능합니다")
            return v
        except ValueError as e:
            if "영업 시간" in str(e):
                raise
            raise ValueError(f"시간 형식이 올바르지 않습니다. HH:MM 형식을 사용하세요: {v}")


# ============================================================================
# 도구 정의
# ============================================================================

@tool(args_schema=ReservationInput)
def make_reservation(
    restaurant_name: str,
    date: str,
    time: str,
    party_size: int,
    special_requests: Optional[str] = None,
) -> str:
    """식당 예약을 생성합니다.

    식당 이름, 날짜, 시간, 인원 수를 받아 예약을 처리합니다.
    특별 요청사항이 있으면 함께 전달합니다.

    Args:
        restaurant_name: 예약할 식당 이름
        date: 예약 날짜 (YYYY-MM-DD)
        time: 예약 시간 (HH:MM)
        party_size: 예약 인원 수 (1-20명)
        special_requests: 특별 요청사항 (선택)

    Returns:
        예약 확인 정보
    """
    # 예약 ID 생성 (간단한 Mock)
    reservation_id = f"RSV-{abs(hash(f'{restaurant_name}{date}{time}')) % 10000:04d}"

    result = f"""
예약 확인
{'=' * 40}
  예약 번호: {reservation_id}
  식당: {restaurant_name}
  날짜: {date}
  시간: {time}
  인원: {party_size}명"""

    if special_requests:
        result += f"\n  특별 요청: {special_requests}"

    result += f"\n{'=' * 40}\n  상태: 예약 완료"

    return result.strip()


# ============================================================================
# 테스트 코드
# ============================================================================

def test_basic_reservation():
    """기본 예약 테스트"""
    print("=" * 70)
    print("테스트 1: 기본 예약")
    print("=" * 70)

    result = make_reservation.invoke({
        "restaurant_name": "스시 오마카세",
        "date": "2026-03-15",
        "time": "18:30",
        "party_size": 4,
    })
    print(result)


def test_reservation_with_requests():
    """특별 요청사항 포함 예약 테스트"""
    print("\n" + "=" * 70)
    print("테스트 2: 특별 요청사항 포함")
    print("=" * 70)

    result = make_reservation.invoke({
        "restaurant_name": "라 피자",
        "date": "2026-04-01",
        "time": "19:00",
        "party_size": 6,
        "special_requests": "창가 자리, 생일 케이크 준비 부탁드립니다",
    })
    print(result)


def test_validation():
    """입력 검증 테스트"""
    print("\n" + "=" * 70)
    print("테스트 3: 입력 검증")
    print("=" * 70)

    # party_size 범위 초과 테스트
    print("\n  party_size=25 (범위 초과) 테스트:")
    try:
        make_reservation.invoke({
            "restaurant_name": "테스트 식당",
            "date": "2026-03-15",
            "time": "18:00",
            "party_size": 25,  # 범위 초과
        })
        print("  [FAIL] 검증이 통과되었습니다 (예상: 실패)")
    except Exception as e:
        print(f"  [PASS] 검증 실패 감지: {type(e).__name__}")

    # 잘못된 시간 형식 테스트
    print("\n  time='25:00' (잘못된 시간) 테스트:")
    try:
        make_reservation.invoke({
            "restaurant_name": "테스트 식당",
            "date": "2026-03-15",
            "time": "25:00",  # 잘못된 시간
            "party_size": 2,
        })
        print("  [FAIL] 검증이 통과되었습니다 (예상: 실패)")
    except Exception as e:
        print(f"  [PASS] 검증 실패 감지: {type(e).__name__}")


def test_tool_schema():
    """도구 스키마 확인"""
    print("\n" + "=" * 70)
    print("테스트 4: 도구 스키마 확인")
    print("=" * 70)

    print(f"\n  도구 이름: {make_reservation.name}")
    print(f"  도구 설명: {make_reservation.description[:60]}...")
    print(f"  파라미터 스키마:")
    for name, info in make_reservation.args.items():
        required = "필수" if info.get("default") is None else "선택"
        desc = info.get("description", "설명 없음")[:50]
        print(f"    - {name} ({required}): {desc}")


def test_with_llm():
    """LLM과 Tool Calling 통합 테스트"""
    print("\n" + "=" * 70)
    print("테스트 5: LLM Tool Calling 통합")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools([make_reservation])

    question = "내일 저녁 7시에 '한우마을'에 4명 예약해줘. 창가 자리로 부탁해."
    print(f"\n  질문: {question}")

    # 모델 호출
    response = model_with_tools.invoke([HumanMessage(question)])

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"\n  도구 호출: {tool_call['name']}")
        print(f"  파라미터: {tool_call['args']}")

        # 도구 실행
        result = make_reservation.invoke(tool_call["args"])
        print(f"\n{result}")

        # 최종 응답 생성
        messages = [
            HumanMessage(question),
            response,
            ToolMessage(content=result, tool_call_id=tool_call["id"]),
        ]
        final_response = model_with_tools.invoke(messages)
        print(f"\n  AI 최종 응답: {final_response.content}")
    else:
        print(f"\n  AI 응답 (도구 미사용): {response.content}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("Part 2: 실용적인 Tool 만들기 - 실습 과제 3 해답")
    print("=" * 70)

    # 1. 기본 예약
    test_basic_reservation()

    # 2. 특별 요청사항
    test_reservation_with_requests()

    # 3. 입력 검증
    test_validation()

    # 4. 스키마 확인
    test_tool_schema()

    # 5. LLM 통합 테스트
    try:
        test_with_llm()
    except Exception as e:
        print(f"\n  LLM 통합 테스트 스킵: {e}")
        print("  (API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다)")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("추가 학습 포인트:")
    print("  1. 예약 취소/변경 도구를 추가해보세요")
    print("  2. 예약 가능 시간 조회 도구를 만들어보세요")
    print("  3. 여러 식당을 검색하고 예약하는 멀티 도구 시나리오를 구현해보세요")
    print("  4. 중첩 Pydantic 모델로 주소, 연락처 등을 추가해보세요")
    print("=" * 70)


if __name__ == "__main__":
    main()
