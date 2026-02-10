"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: LangChain 기초
================================================================================

파일명: 04_tools_advanced.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 25분

📚 학습 목표:
  - Pydantic BaseModel을 사용한 Tool 스키마 정의
  - Field를 사용한 파라미터 검증 및 설명
  - Optional, Required 파라미터 처리
  - Enum/Literal 타입으로 선택지 제한
  - 중첩된 Pydantic 모델로 복잡한 데이터 구조 다루기

📖 공식 문서:
  • Tools: /official/09-tools.md

🔧 필요한 패키지:
  pip install langchain pydantic

🚀 실행 방법:
  python 04_tools_advanced.py

================================================================================
"""

from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from enum import Enum


# ============================================================================
# 예제 1: Pydantic BaseModel로 Tool 입력 스키마 정의
# ============================================================================

class WeatherInput(BaseModel):
    """날씨 조회를 위한 입력 스키마"""
    city: str = Field(description="날씨를 조회할 도시 이름 (예: 서울, 부산)")
    country: str = Field(default="한국", description="국가 이름")


@tool(args_schema=WeatherInput)
def get_weather_advanced(city: str, country: str = "한국") -> str:
    """주어진 도시의 날씨를 상세하게 조회합니다."""
    # 실제로는 API를 호출
    weather_data = {
        ("서울", "한국"): "맑음, 22도, 습도 60%",
        ("부산", "한국"): "흐림, 20도, 습도 75%",
        ("뉴욕", "미국"): "비, 15도, 습도 85%",
    }

    weather = weather_data.get((city, country), "날씨 정보를 찾을 수 없습니다")
    return f"{country} {city}의 날씨: {weather}"


def example_1_pydantic_schema():
    """Pydantic BaseModel을 사용한 스키마 정의"""
    print("=" * 70)
    print("📌 예제 1: Pydantic BaseModel로 Tool 입력 스키마 정의")
    print("=" * 70)

    # Tool 정보 확인
    print(f"\n🔧 도구 이름: {get_weather_advanced.name}")
    print(f"📝 도구 설명: {get_weather_advanced.description}")
    print(f"\n📋 입력 스키마:")
    print(f"   {get_weather_advanced.args_schema.schema()}")

    # Tool 실행
    result1 = get_weather_advanced.invoke({"city": "서울"})
    print(f"\n🌤️  {result1}")

    result2 = get_weather_advanced.invoke({"city": "뉴욕", "country": "미국"})
    print(f"🌤️  {result2}")

    print("\n💡 Pydantic으로 타입 검증, 기본값, 설명을 한번에 정의!\n")


# ============================================================================
# 예제 2: Field 설명과 검증
# ============================================================================

class UserProfileInput(BaseModel):
    """사용자 프로필 생성 입력"""
    name: str = Field(description="사용자 이름", min_length=2, max_length=50)
    age: int = Field(description="사용자 나이", ge=0, le=150)  # ge=greater or equal
    email: str = Field(description="이메일 주소")
    bio: Optional[str] = Field(default=None, description="자기소개 (선택사항)")

    @validator('email')
    def validate_email(cls, v):
        """이메일 형식 검증"""
        if '@' not in v:
            raise ValueError('올바른 이메일 형식이 아닙니다')
        return v


@tool(args_schema=UserProfileInput)
def create_user_profile(name: str, age: int, email: str, bio: Optional[str] = None) -> str:
    """사용자 프로필을 생성합니다."""
    profile = f"👤 이름: {name}\n   나이: {age}세\n   이메일: {email}"
    if bio:
        profile += f"\n   소개: {bio}"
    return profile


def example_2_field_validation():
    """Field를 사용한 상세 검증"""
    print("=" * 70)
    print("📌 예제 2: Field 설명과 검증")
    print("=" * 70)

    # 정상 케이스
    print("\n✅ 정상 케이스:")
    result1 = create_user_profile.invoke({
        "name": "김철수",
        "age": 30,
        "email": "kim@example.com",
        "bio": "파이썬 개발자입니다."
    })
    print(result1)

    # bio 없이 (Optional)
    print("\n✅ bio 없이 (Optional):")
    result2 = create_user_profile.invoke({
        "name": "이영희",
        "age": 25,
        "email": "lee@example.com"
    })
    print(result2)

    # 에러 케이스 처리
    print("\n❌ 잘못된 입력 (나이 음수):")
    try:
        result3 = create_user_profile.invoke({
            "name": "박민수",
            "age": -5,  # 잘못된 나이
            "email": "park@example.com"
        })
    except Exception as e:
        print(f"   오류 발생: {str(e)}")

    print("\n💡 Field로 최소/최대값, 길이 등을 자동으로 검증!\n")


# ============================================================================
# 예제 3: Optional과 Required 파라미터
# ============================================================================

class SearchInput(BaseModel):
    """검색 도구 입력 스키마"""
    query: str = Field(description="검색어 (필수)")
    max_results: int = Field(default=10, description="최대 결과 개수", ge=1, le=100)
    filter_date: Optional[str] = Field(default=None, description="날짜 필터 (예: 2024-01-01)")
    include_images: bool = Field(default=False, description="이미지 포함 여부")


@tool(args_schema=SearchInput)
def search_with_options(
    query: str,
    max_results: int = 10,
    filter_date: Optional[str] = None,
    include_images: bool = False
) -> str:
    """다양한 옵션으로 검색을 수행합니다."""
    result = f"🔍 '{query}' 검색 결과 (최대 {max_results}개)"

    if filter_date:
        result += f"\n   날짜 필터: {filter_date} 이후"

    if include_images:
        result += "\n   이미지 포함"

    return result


def example_3_optional_required():
    """Optional과 Required 파라미터 사용"""
    print("=" * 70)
    print("📌 예제 3: Optional과 Required 파라미터")
    print("=" * 70)

    # 필수 파라미터만
    print("\n1️⃣ 필수 파라미터만:")
    result1 = search_with_options.invoke({"query": "LangChain"})
    print(f"   {result1}")

    # 일부 옵션 사용
    print("\n2️⃣ 일부 옵션 사용:")
    result2 = search_with_options.invoke({
        "query": "Python",
        "max_results": 20
    })
    print(f"   {result2}")

    # 모든 옵션 사용
    print("\n3️⃣ 모든 옵션 사용:")
    result3 = search_with_options.invoke({
        "query": "AI",
        "max_results": 50,
        "filter_date": "2024-01-01",
        "include_images": True
    })
    print(f"   {result3}")

    print("\n💡 Optional은 None이 가능, Required는 반드시 필요!\n")


# ============================================================================
# 예제 4: Enum/Literal 타입으로 선택지 제한
# ============================================================================

class Priority(str, Enum):
    """우선순위 Enum"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskInput(BaseModel):
    """작업 생성 입력 스키마"""
    title: str = Field(description="작업 제목")
    priority: Priority = Field(description="우선순위 (low, medium, high, urgent)")
    status: Literal["todo", "in_progress", "done"] = Field(
        default="todo",
        description="작업 상태"
    )
    assignee: Optional[str] = Field(default=None, description="담당자")


@tool(args_schema=TaskInput)
def create_task(
    title: str,
    priority: Priority,
    status: Literal["todo", "in_progress", "done"] = "todo",
    assignee: Optional[str] = None
) -> str:
    """새로운 작업을 생성합니다."""
    priority_emoji = {
        Priority.LOW: "🟢",
        Priority.MEDIUM: "🟡",
        Priority.HIGH: "🟠",
        Priority.URGENT: "🔴"
    }

    task = f"📋 작업 생성됨\n"
    task += f"   제목: {title}\n"
    task += f"   우선순위: {priority_emoji[priority]} {priority.value}\n"
    task += f"   상태: {status}"

    if assignee:
        task += f"\n   담당자: {assignee}"

    return task


def example_4_enum_literal():
    """Enum과 Literal으로 선택지 제한"""
    print("=" * 70)
    print("📌 예제 4: Enum/Literal 타입으로 선택지 제한")
    print("=" * 70)

    # 정상 케이스
    print("\n✅ 정상 케이스 1:")
    result1 = create_task.invoke({
        "title": "LangChain 문서 작성",
        "priority": "high",
        "assignee": "김철수"
    })
    print(result1)

    print("\n✅ 정상 케이스 2:")
    result2 = create_task.invoke({
        "title": "버그 수정",
        "priority": "urgent",
        "status": "in_progress"
    })
    print(result2)

    # 잘못된 값 시도
    print("\n❌ 잘못된 우선순위:")
    try:
        result3 = create_task.invoke({
            "title": "테스트",
            "priority": "super_high"  # 존재하지 않는 값
        })
    except Exception as e:
        print(f"   오류: {str(e)[:100]}...")

    print("\n💡 Enum/Literal로 허용된 값만 사용하도록 강제!\n")


# ============================================================================
# 예제 5: 중첩된 Pydantic 모델
# ============================================================================

class Address(BaseModel):
    """주소 정보"""
    street: str = Field(description="도로명")
    city: str = Field(description="도시")
    zipcode: str = Field(description="우편번호")


class ContactInfo(BaseModel):
    """연락처 정보"""
    email: str = Field(description="이메일")
    phone: Optional[str] = Field(default=None, description="전화번호")


class CompanyInput(BaseModel):
    """회사 정보 입력 스키마"""
    name: str = Field(description="회사명")
    address: Address = Field(description="회사 주소")
    contact: ContactInfo = Field(description="연락처 정보")
    employees: int = Field(description="직원 수", ge=1)


@tool(args_schema=CompanyInput)
def register_company(
    name: str,
    address: Address,
    contact: ContactInfo,
    employees: int
) -> str:
    """회사 정보를 등록합니다."""
    result = f"🏢 회사 등록 완료\n"
    result += f"   회사명: {name}\n"
    result += f"   주소: {address.city} {address.street} ({address.zipcode})\n"
    result += f"   이메일: {contact.email}\n"

    if contact.phone:
        result += f"   전화: {contact.phone}\n"

    result += f"   직원 수: {employees}명"

    return result


def example_5_nested_models():
    """중첩된 Pydantic 모델로 복잡한 구조"""
    print("=" * 70)
    print("📌 예제 5: 중첩된 Pydantic 모델")
    print("=" * 70)

    # 복잡한 중첩 구조
    company_data = {
        "name": "테크 스타트업",
        "address": {
            "street": "테헤란로 123",
            "city": "서울",
            "zipcode": "06234"
        },
        "contact": {
            "email": "info@techstartup.com",
            "phone": "02-1234-5678"
        },
        "employees": 50
    }

    result = register_company.invoke(company_data)
    print(f"\n{result}")

    # 전화번호 없이
    print("\n📞 전화번호 없이 등록:")
    company_data2 = {
        "name": "AI 연구소",
        "address": {
            "street": "강남대로 456",
            "city": "서울",
            "zipcode": "06789"
        },
        "contact": {
            "email": "contact@ailab.com"
            # phone은 Optional이므로 생략 가능
        },
        "employees": 20
    }

    result2 = register_company.invoke(company_data2)
    print(f"\n{result2}")

    print("\n💡 중첩 모델로 복잡한 데이터 구조를 체계적으로 관리!\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n🎓 Part 2: LangChain 기초 - Tools (고급)\n")

    example_1_pydantic_schema()
    input("⏎ 계속하려면 Enter...")

    example_2_field_validation()
    input("⏎ 계속하려면 Enter...")

    example_3_optional_required()
    input("⏎ 계속하려면 Enter...")

    example_4_enum_literal()
    input("⏎ 계속하려면 Enter...")

    example_5_nested_models()

    print("=" * 70)
    print("🎉 Tools 고급 학습 완료!")
    print("📖 다음: 05_tool_calling.py - Tool Calling")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. Pydantic BaseModel 장점:
#    - 자동 타입 검증
#    - 명확한 스키마 정의
#    - IDE 자동완성 지원
#    - 복잡한 검증 로직 추가 가능
#
# 2. Field 주요 파라미터:
#    - description: 필드 설명 (LLM이 읽음)
#    - default: 기본값
#    - ge/le: 숫자의 최소/최대값
#    - min_length/max_length: 문자열 길이 제한
#
# 3. Optional vs Required:
#    - Optional[str]: None 가능
#    - str: 반드시 필요
#    - default 값이 있으면 Optional
#
# 4. Enum vs Literal:
#    - Enum: 재사용 가능한 선택지 그룹
#    - Literal: 인라인으로 바로 정의
#    - 둘 다 허용된 값만 입력 가능
#
# 5. 중첩 모델 활용:
#    - 복잡한 데이터는 여러 모델로 분리
#    - 재사용성 향상
#    - 유지보수 용이
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "validation error"가 계속 발생
# 해결: Pydantic 스키마와 실제 함수 시그니처가 일치하는지 확인
#
# 문제: Optional 필드인데 None을 받으면 오류
# 해결: 함수 시그니처에도 Optional[T] = None 으로 명시
#
# 문제: Enum 값이 문자열로 전달되는데 오류
# 해결: Enum은 자동으로 문자열에서 변환됨, str을 상속받았는지 확인
#
# 문제: 중첩 모델이 딕셔너리로 전달되는데 오류
# 해결: Pydantic이 자동으로 파싱함, 구조만 맞으면 OK
#
# ============================================================================
