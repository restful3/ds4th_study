"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 (Production)
================================================================================

파일명: 06_structured_output.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 25분

📚 학습 목표:
  - Structured Output의 필요성 이해
  - Pydantic으로 출력 구조 정의
  - Agent에서 구조화된 데이터 생성
  - 실전 활용 패턴

📖 공식 문서:
  • Structured Output: /official/12-structured-output.md

📄 교안 문서:
  • Part 9 개요: /docs/part09_production.md

🔧 필요한 패키지:
  pip install langchain langchain-openai pydantic

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 06_structured_output.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: Structured Output 개념
# ============================================================================

def example_1_structured_output_concept():
    """Structured Output의 필요성과 개념"""
    print("=" * 70)
    print("📌 예제 1: Structured Output 개념")
    print("=" * 70)

    print("""
📊 Structured Output이란?

정의:
  LLM의 자유 형식 텍스트 출력을 정해진 구조(스키마)로 강제하는 기법

왜 필요한가?
  • 일관된 데이터 형식
  • 타입 안전성 (Type Safety)
  • 자동 검증 (Validation)
  • 다운스트림 시스템 연동 용이
  • 에러 감소

문제: 일반 LLM 출력
  "서울의 날씨는 맑고 기온은 22도이며 습도는 60%입니다."
  → 파싱이 어렵고, 형식이 불안정

해결: Structured Output
  {
    "city": "서울",
    "condition": "맑음",
    "temperature": 22,
    "humidity": 60
  }
  → JSON 형식, 타입 명확, 자동 검증

구현 방법:
  1️⃣ Pydantic 모델 정의
  2️⃣ LLM에 구조 지정
  3️⃣ 자동 파싱 및 검증

💡 프로덕션 시스템에서는 필수!
    """)


# ============================================================================
# 예제 2: Pydantic 기본 사용법
# ============================================================================

def example_2_pydantic_basics():
    """Pydantic으로 데이터 구조 정의"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Pydantic 기본 사용법")
    print("=" * 70)

    # Pydantic 모델 정의
    class WeatherData(BaseModel):
        """날씨 데이터 구조"""
        city: str = Field(description="도시 이름")
        temperature: float = Field(description="온도 (섭씨)")
        condition: str = Field(description="날씨 상태")
        humidity: int = Field(ge=0, le=100, description="습도 (0-100%)")

    class UserProfile(BaseModel):
        """사용자 프로필 구조"""
        name: str = Field(description="사용자 이름")
        age: int = Field(ge=0, le=150, description="나이")
        email: str = Field(description="이메일 주소")
        interests: List[str] = Field(description="관심사 목록")

    print("\n🔹 Pydantic 모델 예시:")
    print("-" * 70)

    # 날씨 데이터 생성
    weather = WeatherData(
        city="서울",
        temperature=22.5,
        condition="맑음",
        humidity=60
    )

    print("\n✅ 날씨 데이터:")
    print(weather.model_dump_json(indent=2))

    # 사용자 프로필 생성
    user = UserProfile(
        name="김철수",
        age=30,
        email="kim@example.com",
        interests=["AI", "Python", "독서"]
    )

    print("\n✅ 사용자 프로필:")
    print(user.model_dump_json(indent=2))

    # 검증 실패 예시
    print("\n🔹 검증 실패 예시:")
    print("-" * 70)

    try:
        WeatherData(
            city="부산",
            temperature=25.0,
            condition="흐림",
            humidity=150  # 잘못된 값 (0-100 범위 초과)
        )
    except Exception as e:
        print(f"❌ 검증 오류: {e}")

    print("\n" + "-" * 70)
    print("💡 Pydantic은 자동으로 타입과 제약 조건을 검증합니다.")


# ============================================================================
# 예제 3: LLM에서 Structured Output 생성
# ============================================================================

def example_3_llm_structured_output():
    """LLM이 구조화된 데이터를 생성하도록 설정"""
    print("\n" + "=" * 70)
    print("📌 예제 3: LLM에서 Structured Output 생성")
    print("=" * 70)

    # 출력 구조 정의
    class Product(BaseModel):
        """제품 정보 구조"""
        name: str = Field(description="제품명")
        price: float = Field(description="가격")
        category: str = Field(description="카테고리")
        in_stock: bool = Field(description="재고 여부")
        rating: float = Field(ge=0, le=5, description="평점 (0-5)")

    # Structured Output 지원 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # with_structured_output()으로 구조 지정
    structured_llm = llm.with_structured_output(Product)

    print("\n🔹 제품 정보 추출:")
    print("-" * 70)

    product_text = """
    MacBook Pro는 애플의 프리미엄 노트북입니다.
    가격은 2,390,000원이며, 컴퓨터 카테고리에 속합니다.
    현재 재고가 있으며, 고객 평점은 4.8점입니다.
    """

    print(f"입력 텍스트:\n{product_text}")

    # 구조화된 출력 생성
    result = structured_llm.invoke(product_text)

    print(f"\n✅ 구조화된 출력:")
    print(f"  제품명: {result.name}")
    print(f"  가격: {result.price:,}원")
    print(f"  카테고리: {result.category}")
    print(f"  재고: {'있음' if result.in_stock else '없음'}")
    print(f"  평점: {result.rating}점")

    print(f"\nJSON 형식:")
    print(result.model_dump_json(indent=2))

    print("\n" + "-" * 70)
    print("💡 LLM이 자동으로 텍스트를 구조화된 데이터로 변환합니다.")


# ============================================================================
# 예제 4: 복잡한 중첩 구조
# ============================================================================

def example_4_nested_structures():
    """중첩된 복잡한 데이터 구조"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 복잡한 중첩 구조")
    print("=" * 70)

    # 중첩 구조 정의
    class Address(BaseModel):
        """주소 정보"""
        street: str = Field(description="도로명")
        city: str = Field(description="도시")
        zipcode: str = Field(description="우편번호")

    class OrderItem(BaseModel):
        """주문 항목"""
        product_name: str = Field(description="제품명")
        quantity: int = Field(ge=1, description="수량")
        price: float = Field(ge=0, description="단가")

    class Order(BaseModel):
        """주문 정보"""
        order_id: str = Field(description="주문 번호")
        customer_name: str = Field(description="고객명")
        shipping_address: Address = Field(description="배송 주소")
        items: List[OrderItem] = Field(description="주문 항목 목록")
        total_amount: float = Field(ge=0, description="총 금액")
        status: Literal["pending", "shipped", "delivered"] = Field(
            description="주문 상태"
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Order)

    print("\n🔹 주문 정보 파싱:")
    print("-" * 70)

    order_text = """
    주문번호 ORD-2024-001의 고객 김철수님이
    서울특별시 강남구 테헤란로 123, 우편번호 06234로 배송 요청했습니다.
    주문 내역:
    - MacBook Pro 1대, 2,390,000원
    - Magic Mouse 2개, 각 99,000원
    총 금액은 2,588,000원이며 배송 중입니다.
    """

    print(f"입력 텍스트:\n{order_text}")

    result = structured_llm.invoke(order_text)

    print(f"\n✅ 파싱된 주문 정보:")
    print(f"  주문번호: {result.order_id}")
    print(f"  고객명: {result.customer_name}")
    print(f"  배송지: {result.shipping_address.street}, {result.shipping_address.city}")
    print(f"  우편번호: {result.shipping_address.zipcode}")
    print(f"\n  주문 항목:")
    for item in result.items:
        print(f"    • {item.product_name}: {item.quantity}개 x {item.price:,}원")
    print(f"\n  총 금액: {result.total_amount:,}원")
    print(f"  상태: {result.status}")

    print("\n" + "-" * 70)
    print("💡 복잡한 중첩 구조도 자동으로 파싱됩니다.")


# ============================================================================
# 예제 5: Agent에서 Structured Output 활용
# ============================================================================

def example_5_agent_structured_output():
    """Agent가 구조화된 출력을 생성"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Agent에서 Structured Output 활용")
    print("=" * 70)

    # 분석 보고서 구조
    class AnalysisReport(BaseModel):
        """분석 보고서 구조"""
        title: str = Field(description="보고서 제목")
        summary: str = Field(description="요약")
        findings: List[str] = Field(description="주요 발견 사항")
        recommendations: List[str] = Field(description="권장 사항")
        confidence_score: float = Field(ge=0, le=1, description="신뢰도 (0-1)")
        generated_at: str = Field(description="생성 시각")

    @tool
    def analyze_data(data_source: str) -> str:
        """데이터를 분석합니다."""
        return f"{data_source} 데이터 분석 완료: 평균 85점, 증가 추세"

    # Structured Output을 생성하는 함수
    def generate_structured_report(user_query: str) -> AnalysisReport:
        """구조화된 보고서 생성"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        structured_llm = llm.with_structured_output(AnalysisReport)

        prompt = f"""
다음 분석 요청에 대한 보고서를 작성하세요:
{user_query}

보고서에는 제목, 요약, 주요 발견 사항(3개), 권장 사항(3개), 신뢰도가 포함되어야 합니다.
        """

        report = structured_llm.invoke(prompt)
        report.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return report

    print("\n🔹 구조화된 보고서 생성:")
    print("-" * 70)

    user_query = "2024년 Q4 매출 데이터를 분석하고 개선 방안을 제시해주세요."
    print(f"요청: {user_query}\n")

    report = generate_structured_report(user_query)

    print("📄 생성된 보고서:")
    print("=" * 70)
    print(f"\n제목: {report.title}")
    print(f"\n요약:\n{report.summary}")
    print(f"\n주요 발견 사항:")
    for i, finding in enumerate(report.findings, 1):
        print(f"  {i}. {finding}")
    print(f"\n권장 사항:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    print(f"\n신뢰도: {report.confidence_score:.0%}")
    print(f"생성 시각: {report.generated_at}")

    # JSON 저장 예시
    print("\n💾 JSON 형식으로 저장:")
    json_output = report.model_dump_json(indent=2)
    print(json_output[:200] + "...")

    print("\n" + "-" * 70)
    print("💡 Agent의 출력을 구조화하여 다운스트림 시스템에 연동 가능")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 9: 프로덕션 - Structured Output")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_structured_output_concept()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_pydantic_basics()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_llm_structured_output()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_nested_structures()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_agent_structured_output()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 9: 프로덕션을 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. Part 10: Deployment")
    print("  2. LangSmith로 모니터링")
    print("  3. 프로덕션 배포")
    print("\n📚 Part 9 핵심 요약:")
    print("  • Streaming: 실시간 응답 처리")
    print("  • Stream Modes: values, updates, messages")
    print("  • Custom Streaming: 데이터 가공 및 필터링")
    print("  • HITL: 사람의 개입으로 안전성 확보")
    print("  • Structured Output: 일관된 데이터 형식")
    print("\n🎯 프로덕션 준비 완료!")
    print("  이제 Part 10에서 실제 배포를 학습합니다.")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. Structured Output 고급:
#    - Optional 필드
#    - Union 타입
#    - Custom Validators
#    - JSON Schema 직접 정의
#
# 2. 실전 활용:
#    - API 응답 구조화
#    - 데이터베이스 저장
#    - 다른 시스템 연동
#    - 자동 문서화
#
# 3. 성능 최적화:
#    - 스키마 캐싱
#    - 병렬 처리
#    - 스트리밍 + 구조화
#
# 4. 에러 처리:
#    - 검증 실패 시 재시도
#    - 부분 파싱
#    - 폴백 전략
#
# ============================================================================
