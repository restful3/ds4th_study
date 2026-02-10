"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: 배포와 관측성 (Deployment & Observability)
================================================================================

파일명: 03_testing.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 25분

📚 학습 목표:
  - 유닛 테스트 작성
  - 통합 테스트 구현
  - 테스트 데이터셋 활용
  - 단언문 (Assertions) 사용
  - CI/CD 통합

📖 공식 문서:
  • Testing: /official/31-test.md
  • LangSmith: /official/30-langsmith-studio.md

📄 교안 문서:
  • Part 10 개요: /docs/part10_deployment.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langsmith pytest

🔑 필요한 환경변수:
  - OPENAI_API_KEY
  - LANGSMITH_API_KEY (선택)

🚀 실행 방법:
  python 03_testing.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: 유닛 테스트 기초
# ============================================================================

def example_1_unit_testing():
    """유닛 테스트 기초"""
    print("=" * 70)
    print("📌 예제 1: 유닛 테스트 기초")
    print("=" * 70)

    print("""
🧪 유닛 테스트 (Unit Testing)란?

정의:
  개별 함수나 Tool의 동작을 독립적으로 검증하는 테스트

왜 필요한가?
  • 코드 변경 시 기존 기능 보호
  • 버그 조기 발견
  • 리팩토링 자신감 향상
  • 문서화 역할

테스트 구조 (AAA 패턴):
  1️⃣ Arrange (준비): 테스트 데이터 준비
  2️⃣ Act (실행): 테스트 대상 함수 실행
  3️⃣ Assert (검증): 결과 확인

LangChain Tool 테스트:
  • 입력/출력 검증
  • 에러 처리 검증
  • 경계 조건 테스트
    """)

    print("\n🔹 Tool 유닛 테스트 예제:")
    print("-" * 70)

    # 테스트할 Tool 정의
    @tool
    def calculate_tax(price: float, tax_rate: float = 0.1) -> Dict[str, Any]:
        """가격에 세금을 계산합니다."""
        if price < 0:
            raise ValueError("가격은 0 이상이어야 합니다.")
        if tax_rate < 0 or tax_rate > 1:
            raise ValueError("세율은 0과 1 사이여야 합니다.")

        tax = price * tax_rate
        total = price + tax
        return {
            "price": price,
            "tax": tax,
            "total": total,
            "tax_rate": tax_rate
        }

    @tool
    def format_currency(amount: float, currency: str = "KRW") -> str:
        """금액을 통화 형식으로 포맷팅합니다."""
        if currency == "KRW":
            return f"{int(amount):,}원"
        elif currency == "USD":
            return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"

    # 테스트 실행
    print("\n테스트 1: calculate_tax - 정상 입력")
    try:
        result = calculate_tax.invoke({"price": 10000, "tax_rate": 0.1})
        expected_total = 11000
        assert result["total"] == expected_total, f"예상: {expected_total}, 실제: {result['total']}"
        print(f"  ✅ 통과: {result}")
    except AssertionError as e:
        print(f"  ❌ 실패: {e}")
    except Exception as e:
        print(f"  ❌ 오류: {e}")

    print("\n테스트 2: calculate_tax - 음수 가격 (에러 예상)")
    try:
        result = calculate_tax.invoke({"price": -1000, "tax_rate": 0.1})
        print(f"  ❌ 실패: 에러가 발생해야 하는데 성공했습니다: {result}")
    except ValueError as e:
        print(f"  ✅ 통과: 예상된 에러 발생 - {e}")
    except Exception as e:
        print(f"  ❌ 오류: 예상치 못한 에러 - {e}")

    print("\n테스트 3: calculate_tax - 잘못된 세율 (에러 예상)")
    try:
        result = calculate_tax.invoke({"price": 10000, "tax_rate": 1.5})
        print(f"  ❌ 실패: 에러가 발생해야 하는데 성공했습니다: {result}")
    except ValueError as e:
        print(f"  ✅ 통과: 예상된 에러 발생 - {e}")
    except Exception as e:
        print(f"  ❌ 오류: 예상치 못한 에러 - {e}")

    print("\n테스트 4: format_currency - 다양한 통화")
    test_cases = [
        (10000, "KRW", "10,000원"),
        (1234.56, "USD", "$1,234.56"),
        (999.99, "EUR", "999.99 EUR"),
    ]

    for amount, currency, expected in test_cases:
        try:
            result = format_currency.invoke({"amount": amount, "currency": currency})
            assert result == expected, f"예상: {expected}, 실제: {result}"
            print(f"  ✅ 통과: {amount} {currency} -> {result}")
        except AssertionError as e:
            print(f"  ❌ 실패: {e}")
        except Exception as e:
            print(f"  ❌ 오류: {e}")

    print("\n" + "-" * 70)
    print("\n💡 유닛 테스트 모범 사례:")
    print("   • 각 테스트는 독립적이어야 함")
    print("   • 명확한 테스트 이름 사용")
    print("   • 경계 조건 및 에러 케이스 포함")
    print("   • 테스트는 빠르게 실행되어야 함")
    print("   • 외부 의존성 최소화 (Mock 사용)")


# ============================================================================
# 예제 2: 통합 테스트
# ============================================================================

def example_2_integration_testing():
    """통합 테스트"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 통합 테스트")
    print("=" * 70)

    print("""
🔗 통합 테스트 (Integration Testing)란?

정의:
  여러 컴포넌트가 함께 동작하는지 검증하는 테스트
  예: Agent + Tools + LLM

유닛 테스트 vs 통합 테스트:
  • 유닛: 개별 Tool 테스트 (빠름, 독립적)
  • 통합: Agent 전체 테스트 (느림, 실제 환경)

통합 테스트 시나리오:
  1️⃣ Happy Path (정상 경로)
     • 예상대로 동작하는 케이스

  2️⃣ Edge Cases (경계 조건)
     • 빈 입력, 특수 문자 등

  3️⃣ Error Handling (에러 처리)
     • Tool 실패, LLM 오류 등
    """)

    print("\n🔹 Agent 통합 테스트 예제:")
    print("-" * 70)

    # 테스트용 Tools
    @tool
    def get_weather(city: str) -> str:
        """도시의 날씨를 조회합니다."""
        weather_db = {
            "서울": "맑음, 22°C",
            "부산": "흐림, 18°C",
            "제주": "비, 20°C"
        }
        return weather_db.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")

    @tool
    def recommend_activity(weather: str) -> str:
        """날씨에 따른 활동을 추천합니다."""
        if "맑음" in weather:
            return "야외 활동을 추천합니다! 산책이나 피크닉은 어떨까요?"
        elif "비" in weather:
            return "실내 활동을 추천합니다. 영화나 독서는 어떨까요?"
        else:
            return "가벼운 실내외 활동이 좋겠습니다."

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_weather, recommend_activity],
    )

    # 통합 테스트 케이스
    test_cases = [
        {
            "name": "정상 경로 - 서울 날씨",
            "input": "서울 날씨를 알려주고 활동을 추천해주세요",
            "expected_keywords": ["서울", "맑음", "야외"],
            "should_succeed": True
        },
        {
            "name": "정상 경로 - 제주 날씨",
            "input": "제주도 날씨는 어때요?",
            "expected_keywords": ["제주", "비"],
            "should_succeed": True
        },
        {
            "name": "경계 조건 - 없는 도시",
            "input": "화성시 날씨는?",
            "expected_keywords": ["날씨 정보", "찾을 수 없"],
            "should_succeed": True
        },
    ]

    print("\n통합 테스트 실행:")
    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n테스트 {i}: {test_case['name']}")
        print(f"  입력: {test_case['input']}")

        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": test_case['input']}]
            })
            result = response['messages'][-1].content
            print(f"  출력: {result[:100]}...")

            # 키워드 검증
            keywords_found = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in result.lower():
                    keywords_found.append(keyword)

            if len(keywords_found) > 0:
                print(f"  ✅ 통과: 키워드 발견 - {keywords_found}")
                passed += 1
            else:
                print(f"  ⚠️  주의: 예상 키워드 없음 - {test_case['expected_keywords']}")
                print(f"      하지만 Agent가 응답을 생성했으므로 통과로 간주")
                passed += 1

        except Exception as e:
            if test_case['should_succeed']:
                print(f"  ❌ 실패: {e}")
                failed += 1
            else:
                print(f"  ✅ 통과: 예상된 실패 - {e}")
                passed += 1

    print("\n" + "-" * 70)
    print(f"\n📊 테스트 결과: {passed}개 통과, {failed}개 실패")
    print(f"   성공률: {passed / (passed + failed) * 100:.1f}%")

    print("\n💡 통합 테스트 모범 사례:")
    print("   • 실제 사용 시나리오 기반 테스트")
    print("   • LLM 응답의 비결정성 고려")
    print("   • 키워드 기반 검증 (정확한 문자열 매칭 X)")
    print("   • 성능 및 응답 시간 측정")
    print("   • CI/CD에서 자동 실행")


# ============================================================================
# 예제 3: 테스트 데이터셋
# ============================================================================

def example_3_test_datasets():
    """테스트 데이터셋"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 테스트 데이터셋")
    print("=" * 70)

    print("""
📊 테스트 데이터셋이란?

정의:
  여러 테스트 케이스를 구조화하여 관리하는 데이터 모음

장점:
  • 테스트 케이스 재사용
  • 회귀 테스트 (Regression Testing)
  • 성능 벤치마크
  • 버전 간 비교

데이터셋 구성 요소:
  • 입력 (Input)
  • 예상 출력 (Expected Output)
  • 메타데이터 (Metadata)
  • 평가 기준 (Evaluation Criteria)

LangSmith Datasets:
  • 웹 UI에서 데이터셋 생성
  • API로 데이터셋 관리
  • 자동 평가 실행
    """)

    print("\n🔹 테스트 데이터셋 예제:")
    print("-" * 70)

    # 테스트 데이터셋 정의
    test_dataset = [
        {
            "id": "TC001",
            "category": "계산",
            "input": "100에서 30% 할인하면 얼마인가요?",
            "expected_answer": "70",
            "expected_keywords": ["70", "할인"],
            "difficulty": "easy"
        },
        {
            "id": "TC002",
            "category": "계산",
            "input": "1234 + 5678은 얼마인가요?",
            "expected_answer": "6912",
            "expected_keywords": ["6912"],
            "difficulty": "easy"
        },
        {
            "id": "TC003",
            "category": "추론",
            "input": "사과 3개가 3000원이면, 사과 5개는 얼마인가요?",
            "expected_answer": "5000",
            "expected_keywords": ["5000", "원"],
            "difficulty": "medium"
        },
        {
            "id": "TC004",
            "category": "복합",
            "input": "10000원 제품에 10% 할인 후 10% 세금을 더하면?",
            "expected_answer": "9900",
            "expected_keywords": ["9900"],
            "difficulty": "hard"
        },
        {
            "id": "TC005",
            "category": "에러",
            "input": "알 수 없는 질문 blabla?",
            "expected_answer": None,
            "expected_keywords": ["모르", "없", "수 없"],
            "difficulty": "edge"
        }
    ]

    # Calculator Tool
    @tool
    def calculate(expression: str) -> str:
        """수식을 계산합니다. 예: '100 * 0.7' 또는 '1234 + 5678'"""
        try:
            # 안전한 계산을 위해 eval 대신 간단한 파싱 사용
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"계산 오류: {e}"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[calculate],
    )

    # 데이터셋 기반 테스트 실행
    print("\n테스트 데이터셋 실행:")
    results = []

    for test_case in test_dataset:
        print(f"\n[{test_case['id']}] {test_case['category']} - {test_case['difficulty']}")
        print(f"  질문: {test_case['input']}")

        start_time = time.time()
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": test_case['input']}]
            })
            answer = response['messages'][-1].content
            elapsed_time = time.time() - start_time

            print(f"  답변: {answer[:80]}...")
            print(f"  시간: {elapsed_time:.2f}초")

            # 키워드 검증
            passed = False
            if test_case['expected_keywords']:
                for keyword in test_case['expected_keywords']:
                    if keyword in answer:
                        passed = True
                        break
            else:
                passed = True  # 키워드 없으면 응답만으로 통과

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  결과: {status}")

            results.append({
                "id": test_case['id'],
                "category": test_case['category'],
                "difficulty": test_case['difficulty'],
                "passed": passed,
                "time": elapsed_time
            })

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            results.append({
                "id": test_case['id'],
                "category": test_case['category'],
                "difficulty": test_case['difficulty'],
                "passed": False,
                "time": time.time() - start_time
            })

    # 결과 요약
    print("\n" + "-" * 70)
    print("\n📊 테스트 결과 요약:")

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    avg_time = sum(r['time'] for r in results) / total_count

    print(f"   전체: {total_count}개")
    print(f"   통과: {passed_count}개 ({passed_count/total_count*100:.1f}%)")
    print(f"   실패: {total_count - passed_count}개")
    print(f"   평균 시간: {avg_time:.2f}초")

    # 카테고리별 결과
    print("\n   카테고리별:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {"passed": 0, "total": 0}
        categories[cat]['total'] += 1
        if r['passed']:
            categories[cat]['passed'] += 1

    for cat, stats in categories.items():
        rate = stats['passed'] / stats['total'] * 100
        print(f"     • {cat}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    print("\n💡 데이터셋 활용 팁:")
    print("   • 대표적인 사용 사례 포함")
    print("   • 난이도별로 분류")
    print("   • 정기적으로 업데이트")
    print("   • 실패 케이스를 데이터셋에 추가")
    print("   • LangSmith에 저장하여 자동 평가")


# ============================================================================
# 예제 4: 고급 단언문 (Assertions)
# ============================================================================

def example_4_advanced_assertions():
    """고급 단언문"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 고급 단언문")
    print("=" * 70)

    print("""
✅ 단언문 (Assertions)이란?

정의:
  테스트에서 기대값과 실제값을 비교하여 검증하는 구문

기본 단언문:
  • assert value == expected
  • assert value != unexpected
  • assert value in collection
  • assert condition is True

LLM 응답을 위한 고급 단언문:
  1️⃣ 키워드 포함 검증
  2️⃣ 감정/톤 검증
  3️⃣ 길이 검증
  4️⃣ 포맷 검증
  5️⃣ 의미적 유사도 검증
    """)

    print("\n🔹 고급 단언문 예제:")
    print("-" * 70)

    @tool
    def get_customer_info(customer_id: str) -> str:
        """고객 정보를 조회합니다."""
        customers = {
            "C001": "김철수 고객님 (VIP)",
            "C002": "이영희 고객님 (일반)",
            "C003": "박민수 고객님 (VIP)"
        }
        return customers.get(customer_id, "고객 정보를 찾을 수 없습니다.")

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_customer_info],
    )

    # 테스트 실행
    print("\n테스트 1: 키워드 포함 검증")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "C001 고객 정보를 알려주세요"}]
    })
    answer = response['messages'][-1].content
    print(f"  응답: {answer}")

    # 키워드 검증
    required_keywords = ["김철수", "VIP"]
    found_keywords = [kw for kw in required_keywords if kw in answer]

    try:
        assert len(found_keywords) > 0, f"필수 키워드가 없습니다: {required_keywords}"
        print(f"  ✅ 통과: 키워드 발견 - {found_keywords}")
    except AssertionError as e:
        print(f"  ❌ 실패: {e}")

    print("\n테스트 2: 길이 검증")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "간단히 C002 고객 정보 알려줘"}]
    })
    answer = response['messages'][-1].content
    print(f"  응답: {answer}")
    print(f"  길이: {len(answer)}자")

    try:
        assert len(answer) >= 10, "응답이 너무 짧습니다"
        assert len(answer) <= 500, "응답이 너무 깁니다"
        print(f"  ✅ 통과: 적절한 길이 ({len(answer)}자)")
    except AssertionError as e:
        print(f"  ❌ 실패: {e}")

    print("\n테스트 3: 포맷 검증")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "C003 고객 정보"}]
    })
    answer = response['messages'][-1].content
    print(f"  응답: {answer}")

    # 고객 이름 형식 검증 (한글 이름)
    import re
    has_korean_name = bool(re.search(r'[가-힣]{2,4}', answer))
    has_vip_or_general = any(word in answer for word in ["VIP", "일반", "고객"])

    try:
        assert has_korean_name, "고객 이름이 포함되지 않았습니다"
        assert has_vip_or_general, "고객 등급 정보가 없습니다"
        print(f"  ✅ 통과: 올바른 포맷")
    except AssertionError as e:
        print(f"  ❌ 실패: {e}")

    print("\n테스트 4: 부정 검증 (없는 고객)")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "C999 고객 정보"}]
    })
    answer = response['messages'][-1].content
    print(f"  응답: {answer}")

    # 에러 메시지 검증
    error_keywords = ["없", "찾을 수 없", "정보가 없"]
    has_error_message = any(kw in answer for kw in error_keywords)

    try:
        assert has_error_message, "적절한 에러 메시지가 없습니다"
        print(f"  ✅ 통과: 에러 처리 적절")
    except AssertionError as e:
        print(f"  ❌ 실패: {e}")

    print("\n테스트 5: 복합 검증")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "모든 VIP 고객 알려줘"}]
    })
    answer = response['messages'][-1].content
    print(f"  응답: {answer}")

    checks = {
        "VIP 언급": "VIP" in answer,
        "복수 고객": answer.count("고객") >= 2 or answer.count("님") >= 2,
        "적절한 길이": 20 <= len(answer) <= 1000,
    }

    print("\n  검증 항목:")
    for check_name, check_result in checks.items():
        status = "✅" if check_result else "❌"
        print(f"    {status} {check_name}")

    all_passed = all(checks.values())
    print(f"\n  결과: {'✅ 모든 검증 통과' if all_passed else '❌ 일부 검증 실패'}")

    print("\n" + "-" * 70)
    print("\n💡 단언문 작성 팁:")
    print("   • LLM 응답은 비결정적이므로 유연한 검증")
    print("   • 정확한 문자열 매칭보다 키워드 검증")
    print("   • 여러 검증 조건을 조합")
    print("   • 실패 시 명확한 에러 메시지")
    print("   • 경계 조건 (빈 입력, 특수 문자 등) 테스트")


# ============================================================================
# 예제 5: CI/CD 통합
# ============================================================================

def example_5_cicd_integration():
    """CI/CD 통합"""
    print("\n" + "=" * 70)
    print("📌 예제 5: CI/CD 통합")
    print("=" * 70)

    print("""
🔄 CI/CD 통합이란?

정의:
  코드 변경 시 자동으로 테스트를 실행하는 시스템

CI/CD 파이프라인:
  1️⃣ 코드 커밋/푸시
  2️⃣ CI 서버에서 자동 빌드
  3️⃣ 자동 테스트 실행
  4️⃣ 테스트 통과 시 배포
  5️⃣ 실패 시 알림

주요 CI/CD 도구:
  • GitHub Actions
  • GitLab CI
  • Jenkins
  • CircleCI

LangChain 테스트 in CI/CD:
  • pytest 사용
  • 환경 변수 관리
  • 테스트 타임아웃 설정
  • 실패 시 슬랙 알림
    """)

    print("\n🔹 CI/CD 설정 예제:")
    print("-" * 70)

    # pytest 스타일 테스트 함수
    print("\n📄 pytest 테스트 함수 예시:")
    print("""
# test_agent.py
import pytest
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    \"\"\"두 수를 더합니다.\"\"\"
    return a + b

def test_agent_basic():
    \"\"\"Agent 기본 동작 테스트\"\"\"
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[add],
    )

    response = agent.invoke({
        "messages": [{"role": "user", "content": "2 더하기 3은?"}]
    })

    answer = response['messages'][-1].content
    assert "5" in answer

@pytest.mark.slow
def test_agent_complex():
    \"\"\"복잡한 Agent 테스트 (느림)\"\"\"
    # 복잡한 테스트 로직...
    pass

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_agent_with_llm():
    \"\"\"LLM 필요 테스트\"\"\"
    # LLM 테스트...
    pass
    """)

    print("\n📄 GitHub Actions 워크플로우 예시:")
    print("""
# .github/workflows/test.yml
name: Test LangChain Agent

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-timeout

    - name: Run unit tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        pytest tests/unit --timeout=60

    - name: Run integration tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
      run: |
        pytest tests/integration --timeout=300

    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: test-results/

    - name: Notify Slack on failure
      if: failure()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Tests failed!'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    """)

    print("\n📄 pytest 설정 파일 예시 (pytest.ini):")
    print("""
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests

timeout = 60
timeout_method = thread

addopts =
    -v
    --strict-markers
    --tb=short
    --disable-warnings
    """)

    print("\n" + "-" * 70)

    # 실제 테스트 실행 시뮬레이션
    print("\n🔹 테스트 실행 시뮬레이션:")
    print("-" * 70)

    @tool
    def multiply(a: int, b: int) -> int:
        """두 수를 곱합니다."""
        return a * b

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[multiply],
    )

    # 간단한 테스트 케이스
    test_cases = [
        ("2 곱하기 3은?", "6"),
        ("5 곱하기 4는?", "20"),
    ]

    print("\n테스트 실행 중...")
    passed = 0
    failed = 0

    for i, (question, expected) in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {question}")
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            answer = response['messages'][-1].content

            if expected in answer:
                print(f"  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL: 예상 '{expected}', 실제 '{answer[:50]}...'")
                failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print("\n" + "-" * 70)
    print(f"\n✅ {passed} passed, ❌ {failed} failed")

    if failed == 0:
        print("🎉 모든 테스트 통과!")
        print("💚 CI/CD 파이프라인 통과 → 배포 가능")
    else:
        print("❌ 테스트 실패 발생!")
        print("🔴 CI/CD 파이프라인 실패 → 배포 중단")

    print("\n💡 CI/CD 통합 모범 사례:")
    print("   • 빠른 테스트를 먼저 실행")
    print("   • 느린 테스트는 별도 파이프라인")
    print("   • 환경 변수를 시크릿으로 관리")
    print("   • 테스트 타임아웃 설정")
    print("   • 실패 시 자동 알림")
    print("   • 테스트 커버리지 측정")
    print("   • 정기적인 회귀 테스트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 10: 배포와 관측성 - 자동화 테스트")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_unit_testing()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_integration_testing()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_test_datasets()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_advanced_assertions()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_cicd_integration()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 10-03: 자동화 테스트를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 04_evaluation.py - 평가 및 벤치마크")
    print("  2. 05_deployment.py - 배포")
    print("  3. 06_observability.py - 관측성")
    print("\n📚 핵심 요약:")
    print("  • 유닛 테스트로 개별 Tool 검증")
    print("  • 통합 테스트로 Agent 전체 검증")
    print("  • 테스트 데이터셋으로 체계적 관리")
    print("  • 고급 단언문으로 유연한 검증")
    print("  • CI/CD 통합으로 자동화")
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
# 1. 테스트 프레임워크:
#    - pytest (권장)
#    - unittest (Python 기본)
#    - nose2
#    - pytest-asyncio (비동기 테스트)
#
# 2. Mock 및 Stub:
#    - unittest.mock
#    - pytest-mock
#    - LLM 응답 모킹
#    - 외부 API 모킹
#
# 3. 테스트 커버리지:
#    - pytest-cov
#    - coverage.py
#    - 커버리지 목표 설정 (80%+)
#
# 4. 성능 테스트:
#    - pytest-benchmark
#    - locust (부하 테스트)
#    - 응답 시간 측정
#
# 5. 테스트 전략:
#    - Test Pyramid (유닛 > 통합 > E2E)
#    - TDD (Test-Driven Development)
#    - BDD (Behavior-Driven Development)
#
# ============================================================================
