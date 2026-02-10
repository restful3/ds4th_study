"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: 배포와 관측성 (Deployment & Observability)
================================================================================

파일명: 02_tracing.py
난이도: ⭐⭐⭐⭐☆ (고급)
예상 시간: 25분

📚 학습 목표:
  - 커스텀 트레이싱 스팬 생성
  - 트레이스 필터링 및 검색
  - 성능 분석 및 최적화
  - 트레이스 메타데이터 활용
  - 디버깅 워크플로우

📖 공식 문서:
  • LangSmith: /official/30-langsmith-studio.md
  • Tracing: /official/31-test.md

📄 교안 문서:
  • Part 10 개요: /docs/part10_deployment.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langsmith

🔑 필요한 환경변수:
  - OPENAI_API_KEY
  - LANGSMITH_API_KEY (선택)

🚀 실행 방법:
  python 02_tracing.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langsmith import traceable, Client
from langsmith.run_helpers import trace

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# LangSmith 클라이언트 (선택적)
langsmith_enabled = bool(os.getenv("LANGSMITH_API_KEY"))
if langsmith_enabled:
    try:
        client = Client()
    except:
        langsmith_enabled = False

# ============================================================================
# 예제 1: 커스텀 트레이싱 스팬
# ============================================================================

def example_1_custom_spans():
    """커스텀 트레이싱 스팬 생성"""
    print("=" * 70)
    print("📌 예제 1: 커스텀 트레이싱 스팬")
    print("=" * 70)

    print("""
🔍 커스텀 스팬 (Custom Span)이란?

정의:
  LangChain 자동 트레이싱 외에 사용자 정의 함수/로직을
  추적하기 위한 트레이싱 단위

왜 필요한가?
  • 복잡한 로직의 세부 추적
  • 외부 API 호출 모니터링
  • 데이터 전처리/후처리 추적
  • 성능 병목 지점 파악

활용 방법:
  1️⃣ @traceable 데코레이터
  2️⃣ with trace() 컨텍스트 매니저
  3️⃣ 수동 span 생성
    """)

    print("\n🔹 커스텀 스팬 예제:")
    print("-" * 70)

    # 데코레이터 방식
    @traceable(name="데이터_전처리")
    def preprocess_data(text: str) -> str:
        """데이터 전처리 함수"""
        print(f"  [전처리] 입력: {text[:50]}...")
        time.sleep(0.1)  # 처리 시뮬레이션
        processed = text.lower().strip()
        print(f"  [전처리] 출력: {processed[:50]}...")
        return processed

    @traceable(
        name="외부_API_호출",
        metadata={"api": "weather", "version": "v2"}
    )
    def fetch_external_data(location: str) -> Dict[str, Any]:
        """외부 API 호출 시뮬레이션"""
        print(f"  [API] 위치 조회: {location}")
        time.sleep(0.2)  # API 호출 시뮬레이션
        result = {
            "location": location,
            "temperature": 22,
            "condition": "맑음",
            "timestamp": datetime.now().isoformat()
        }
        print(f"  [API] 결과: {result}")
        return result

    @traceable(name="데이터_후처리")
    def postprocess_result(data: Dict[str, Any]) -> str:
        """결과 후처리"""
        print("  [후처리] 데이터 포매팅 중...")
        time.sleep(0.1)
        formatted = f"{data['location']}: {data['temperature']}°C, {data['condition']}"
        print(f"  [후처리] 출력: {formatted}")
        return formatted

    # 전체 파이프라인 실행
    @traceable(name="날씨_조회_파이프라인")
    def weather_pipeline(user_input: str) -> str:
        """날씨 조회 전체 파이프라인"""
        print("\n🌟 파이프라인 시작")

        # 1. 전처리
        processed_input = preprocess_data(user_input)

        # 2. 외부 API 호출
        weather_data = fetch_external_data(processed_input)

        # 3. 후처리
        final_result = postprocess_result(weather_data)

        print("🌟 파이프라인 완료\n")
        return final_result

    # 실행
    user_query = "  서울 날씨 알려주세요  "
    print(f"👤 사용자 입력: '{user_query}'")

    result = weather_pipeline(user_query)

    print(f"\n✅ 최종 결과: {result}")
    print("-" * 70)

    if langsmith_enabled:
        print("\n💡 LangSmith에서 확인할 수 있는 것:")
        print("   • 각 함수의 실행 시간")
        print("   • 함수 간 호출 관계")
        print("   • 입력/출력 데이터")
        print("   • 커스텀 메타데이터")
        print("\n   🔗 https://smith.langchain.com에서 확인하세요!")
    else:
        print("\n💡 LANGSMITH_API_KEY를 설정하면 트레이스를 시각화할 수 있습니다.")


# ============================================================================
# 예제 2: 트레이스 필터링 및 검색
# ============================================================================

def example_2_trace_filtering():
    """트레이스 필터링 및 검색"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 트레이스 필터링 및 검색")
    print("=" * 70)

    print("""
🔍 트레이스 필터링 및 검색:

LangSmith에서 제공하는 필터링 옵션:
  • 태그 (Tags)
  • 메타데이터 (Metadata)
  • 실행 시간 (Duration)
  • 성공/실패 상태
  • 날짜 범위

코드에서 태그와 메타데이터 추가:
  ```python
  agent.invoke(
      input,
      config={
          "tags": ["production", "user-123"],
          "metadata": {
              "session_id": "abc-123",
              "user_tier": "premium"
          }
      }
  )
  ```
    """)

    print("\n🔹 태그와 메타데이터를 활용한 실행:")
    print("-" * 70)

    @tool
    def search_database(query: str) -> str:
        """데이터베이스를 검색합니다."""
        return f"'{query}'에 대한 검색 결과 5개"

    @tool
    def analyze_sentiment(text: str) -> str:
        """텍스트의 감정을 분석합니다."""
        return "긍정적"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_database, analyze_sentiment],
    )

    # 시나리오 1: 프리미엄 사용자
    print("\n시나리오 1: 프리미엄 사용자 요청")

    response_1 = agent.invoke(
        {"messages": [{"role": "user", "content": "인공지능 관련 최신 뉴스를 검색해주세요"}]},
        config={
            "tags": ["premium-user", "search-feature", "production"],
            "metadata": {
                "user_id": "user-001",
                "user_tier": "premium",
                "session_id": "session-abc-123",
                "feature": "news-search",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

    print(f"  🤖 응답: {response_1['messages'][-1].content[:100]}...")

    # 시나리오 2: 무료 사용자
    print("\n시나리오 2: 무료 사용자 요청")

    response_2 = agent.invoke(
        {"messages": [{"role": "user", "content": "이 문장의 감정을 분석해주세요: 오늘 정말 좋은 날이에요!"}]},
        config={
            "tags": ["free-user", "sentiment-feature", "production"],
            "metadata": {
                "user_id": "user-002",
                "user_tier": "free",
                "session_id": "session-xyz-789",
                "feature": "sentiment-analysis",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

    print(f"  🤖 응답: {response_2['messages'][-1].content[:100]}...")

    # 시나리오 3: 테스트 환경
    print("\n시나리오 3: 테스트 환경")

    response_3 = agent.invoke(
        {"messages": [{"role": "user", "content": "데이터베이스에서 'AI' 검색"}]},
        config={
            "tags": ["test", "qa", "staging"],
            "metadata": {
                "environment": "staging",
                "test_case": "TC-001",
                "tester": "QA-team",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

    print(f"  🤖 응답: {response_3['messages'][-1].content[:100]}...")

    print("\n" + "-" * 70)

    if langsmith_enabled:
        print("\n💡 LangSmith에서 필터링 방법:")
        print("   1. 태그로 필터: tag:premium-user")
        print("   2. 메타데이터로 필터: metadata.user_tier == 'premium'")
        print("   3. 여러 조건 조합: tag:production AND metadata.feature == 'search'")
        print("   4. 실패한 실행만: status:error")
        print("   5. 느린 실행만: latency > 5000ms")
        print("\n   📊 이를 통해 특정 사용자/기능/환경의 성능을 분석할 수 있습니다!")
    else:
        print("\n💡 LANGSMITH_API_KEY를 설정하면 필터링 기능을 사용할 수 있습니다.")


# ============================================================================
# 예제 3: 성능 분석 및 최적화
# ============================================================================

def example_3_performance_analysis():
    """성능 분석 및 최적화"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 성능 분석 및 최적화")
    print("=" * 70)

    print("""
⚡ 성능 분석 및 최적화:

트레이싱으로 파악할 수 있는 것:
  1️⃣ 병목 지점 (Bottleneck)
     • 어느 함수/Tool이 가장 느린가?
     • LLM 호출이 최적화되었는가?

  2️⃣ 불필요한 호출
     • 중복된 Tool 호출
     • 과도한 LLM 호출

  3️⃣ 비용 최적화
     • 토큰 사용량 분석
     • 모델 선택 최적화 (GPT-4 vs GPT-3.5)

  4️⃣ 캐싱 기회
     • 반복되는 질문/응답 패턴
     • 캐시 가능한 데이터
    """)

    print("\n🔹 성능 측정 예제:")
    print("-" * 70)

    # 느린 함수 시뮬레이션
    @traceable(name="느린_데이터베이스_쿼리")
    def slow_database_query(query: str) -> List[Dict]:
        """느린 데이터베이스 쿼리 시뮬레이션"""
        print(f"  [DB] 쿼리 실행: {query}")
        start = time.time()
        time.sleep(2.0)  # 2초 지연
        end = time.time()
        print(f"  [DB] 완료 ({end - start:.2f}초)")
        return [{"id": 1, "data": "결과"}]

    # 빠른 함수 (최적화 후)
    @traceable(name="최적화된_데이터베이스_쿼리")
    def optimized_database_query(query: str) -> List[Dict]:
        """최적화된 데이터베이스 쿼리 (인덱스, 캐싱 적용)"""
        print(f"  [DB-최적화] 쿼리 실행: {query}")
        start = time.time()
        time.sleep(0.2)  # 0.2초 (10배 빠름)
        end = time.time()
        print(f"  [DB-최적화] 완료 ({end - start:.2f}초)")
        return [{"id": 1, "data": "결과"}]

    # 캐싱 시뮬레이션
    cache: Dict[str, Any] = {}

    @traceable(name="캐시_지원_쿼리")
    def cached_query(query: str) -> List[Dict]:
        """캐시를 지원하는 쿼리"""
        if query in cache:
            print(f"  [캐시] 히트! {query}")
            return cache[query]

        print(f"  [캐시] 미스. DB 조회 중...")
        start = time.time()
        time.sleep(0.5)
        result = [{"id": 1, "data": "결과"}]
        cache[query] = result
        end = time.time()
        print(f"  [캐시] 완료 및 저장 ({end - start:.2f}초)")
        return result

    # 성능 비교
    print("\n1️⃣ 느린 쿼리:")
    slow_database_query("SELECT * FROM large_table")

    print("\n2️⃣ 최적화된 쿼리:")
    optimized_database_query("SELECT * FROM large_table WHERE indexed_column = 1")

    print("\n3️⃣ 캐시 적용 (첫 호출):")
    cached_query("SELECT * FROM cache_test")

    print("\n4️⃣ 캐시 적용 (두 번째 호출):")
    cached_query("SELECT * FROM cache_test")

    print("\n" + "-" * 70)

    # 성능 통계
    @traceable(name="성능_통계_분석")
    def analyze_performance():
        """성능 통계 분석"""
        stats = {
            "느린 쿼리": "2000ms",
            "최적화 쿼리": "200ms (10배 개선)",
            "캐시 첫 호출": "500ms",
            "캐시 두번째 호출": "~0ms (즉시)",
            "개선 효과": "최대 2000배 향상"
        }
        return stats

    stats = analyze_performance()

    print("\n📊 성능 분석 결과:")
    for key, value in stats.items():
        print(f"   • {key}: {value}")

    if langsmith_enabled:
        print("\n💡 LangSmith 성능 분석 기능:")
        print("   • 실행 시간 히스토그램")
        print("   • 느린 실행 자동 감지")
        print("   • 함수별 평균 실행 시간")
        print("   • 시간 경과에 따른 성능 추이")
        print("   • 토큰 사용량 및 비용 분석")
    else:
        print("\n💡 LANGSMITH_API_KEY를 설정하면 성능 분석을 시각화할 수 있습니다.")


# ============================================================================
# 예제 4: 트레이스 메타데이터 활용
# ============================================================================

def example_4_trace_metadata():
    """트레이스 메타데이터 활용"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 트레이스 메타데이터 활용")
    print("=" * 70)

    print("""
📝 메타데이터 (Metadata) 활용:

메타데이터란?
  트레이스에 첨부하는 추가 정보
  예: 사용자 ID, 세션 ID, 버전, 환경 등

활용 사례:
  1️⃣ 사용자별 분석
     • 특정 사용자의 모든 요청 추적
     • 사용자 경험 개선

  2️⃣ A/B 테스트
     • 버전별 성능 비교
     • 실험 그룹 추적

  3️⃣ 디버깅
     • 오류 발생 맥락 파악
     • 재현 가능한 조건 식별

  4️⃣ 비즈니스 인사이트
     • 기능별 사용량 분석
     • 고객 세그먼트별 패턴
    """)

    print("\n🔹 메타데이터 활용 예제:")
    print("-" * 70)

    @traceable(name="사용자_요청_처리")
    def process_user_request(
        user_id: str,
        request: str,
        user_profile: Dict[str, Any]
    ) -> str:
        """사용자 요청 처리"""
        print(f"\n  👤 사용자: {user_id}")
        print(f"  📝 요청: {request}")
        print(f"  📊 프로필: {user_profile}")

        # 사용자 티어에 따른 처리
        if user_profile.get("tier") == "premium":
            print("  ✨ 프리미엄 처리 적용")
            time.sleep(0.1)
            response = f"프리미엄 응답: {request}에 대한 상세 분석"
        else:
            print("  📌 기본 처리 적용")
            time.sleep(0.2)
            response = f"기본 응답: {request}에 대한 결과"

        return response

    # 시나리오 1: 프리미엄 사용자
    user_1 = {
        "user_id": "user-premium-001",
        "tier": "premium",
        "region": "KR",
        "signup_date": "2024-01-15"
    }

    @traceable(
        name="프리미엄_사용자_세션",
        metadata={
            "user_id": user_1["user_id"],
            "user_tier": user_1["tier"],
            "region": user_1["region"],
            "experiment_group": "A",
            "feature_flag": "new-ui-enabled"
        }
    )
    def premium_user_session():
        print("\n🔸 시나리오 1: 프리미엄 사용자")
        return process_user_request(
            user_1["user_id"],
            "시장 분석 보고서 작성",
            user_1
        )

    result_1 = premium_user_session()
    print(f"  ✅ 결과: {result_1}")

    # 시나리오 2: 무료 사용자
    user_2 = {
        "user_id": "user-free-002",
        "tier": "free",
        "region": "US",
        "signup_date": "2024-02-01"
    }

    @traceable(
        name="무료_사용자_세션",
        metadata={
            "user_id": user_2["user_id"],
            "user_tier": user_2["tier"],
            "region": user_2["region"],
            "experiment_group": "B",
            "feature_flag": "old-ui-enabled"
        }
    )
    def free_user_session():
        print("\n🔸 시나리오 2: 무료 사용자")
        return process_user_request(
            user_2["user_id"],
            "간단한 질문",
            user_2
        )

    result_2 = free_user_session()
    print(f"  ✅ 결과: {result_2}")

    # 시나리오 3: 오류 발생 케이스
    @traceable(
        name="오류_발생_케이스",
        metadata={
            "user_id": "user-error-003",
            "error_context": "database-connection-failed",
            "retry_count": 3,
            "environment": "production"
        }
    )
    def error_case():
        print("\n🔸 시나리오 3: 오류 발생")
        try:
            print("  ⚠️  데이터베이스 연결 시도...")
            time.sleep(0.1)
            # 오류 시뮬레이션
            raise ConnectionError("Database connection timeout")
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            return f"오류 발생: {str(e)}"

    result_3 = error_case()
    print(f"  📌 처리: {result_3}")

    print("\n" + "-" * 70)

    if langsmith_enabled:
        print("\n💡 메타데이터로 가능한 분석:")
        print("   • 프리미엄 vs 무료 사용자 성능 비교")
        print("   • A/B 테스트 그룹별 전환율")
        print("   • 지역별 사용 패턴")
        print("   • 오류 발생 맥락 분석")
        print("   • 기능 플래그별 성능 측정")
        print("\n   🔍 필터 예시: metadata.user_tier == 'premium' AND metadata.region == 'KR'")
    else:
        print("\n💡 LANGSMITH_API_KEY를 설정하면 메타데이터 분석이 가능합니다.")


# ============================================================================
# 예제 5: 디버깅 워크플로우
# ============================================================================

def example_5_debugging_workflow():
    """디버깅 워크플로우"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 트레이싱을 활용한 디버깅")
    print("=" * 70)

    print("""
🐛 트레이싱을 활용한 디버깅 워크플로우:

1️⃣ 문제 발견:
   • 사용자 보고: "AI가 이상한 답변을 했어요"
   • 모니터링 알림: 오류율 증가
   • 성능 저하: 응답 시간 증가

2️⃣ 트레이스 검색:
   • LangSmith에서 해당 시간대 트레이스 찾기
   • 실패한 실행만 필터링
   • 특정 사용자/세션 추적

3️⃣ 근본 원인 분석:
   • LLM 입력이 잘못되었나?
   • Tool이 예상과 다르게 동작했나?
   • 외부 API 오류인가?

4️⃣ 재현 및 수정:
   • 트레이스의 입력으로 로컬 재현
   • 수정 후 재테스트
   • 새 버전 배포

5️⃣ 검증:
   • 트레이싱으로 수정 확인
   • 동일 패턴의 다른 케이스 확인
    """)

    print("\n🔹 디버깅 시나리오:")
    print("-" * 70)

    @tool
    def get_product_price(product_id: str) -> str:
        """제품 가격을 조회합니다."""
        # 버그: 존재하지 않는 제품 처리 안 됨
        prices = {
            "P001": "10,000원",
            "P002": "20,000원"
        }

        if product_id not in prices:
            # 원래는 적절한 에러 처리가 필요
            return "가격 정보를 찾을 수 없습니다."

        return prices[product_id]

    @tool
    def calculate_discount(price_str: str, discount_rate: float) -> str:
        """할인가를 계산합니다."""
        try:
            # 버그 가능성: 문자열 파싱 오류
            price = int(price_str.replace(",", "").replace("원", ""))
            discounted = price * (1 - discount_rate)
            return f"{int(discounted):,}원"
        except Exception as e:
            return f"계산 오류: {str(e)}"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_product_price, calculate_discount],
    )

    print("\n🔸 시나리오 1: 정상 케이스")

    @traceable(
        name="정상_제품_조회",
        metadata={"scenario": "success", "product": "P001"}
    )
    def success_case():
        response = agent.invoke({
            "messages": [{"role": "user", "content": "P001 제품의 10% 할인가를 알려주세요"}]
        })
        return response['messages'][-1].content

    try:
        result = success_case()
        print(f"  ✅ 결과: {result}")
    except Exception as e:
        print(f"  ❌ 오류: {e}")

    print("\n🔸 시나리오 2: 존재하지 않는 제품")

    @traceable(
        name="존재하지_않는_제품",
        metadata={"scenario": "product-not-found", "product": "P999"}
    )
    def not_found_case():
        response = agent.invoke({
            "messages": [{"role": "user", "content": "P999 제품의 가격을 알려주세요"}]
        })
        return response['messages'][-1].content

    try:
        result = not_found_case()
        print(f"  ⚠️  결과: {result}")
    except Exception as e:
        print(f"  ❌ 오류: {e}")

    print("\n🔸 시나리오 3: 잘못된 할인율")

    @traceable(
        name="잘못된_할인율",
        metadata={"scenario": "invalid-discount", "discount": 150}
    )
    def invalid_discount_case():
        response = agent.invoke({
            "messages": [{"role": "user", "content": "P001 제품의 150% 할인가를 계산해주세요"}]
        })
        return response['messages'][-1].content

    try:
        result = invalid_discount_case()
        print(f"  🤔 결과: {result}")
    except Exception as e:
        print(f"  ❌ 오류: {e}")

    print("\n" + "-" * 70)

    print("\n💡 디버깅 팁:")
    print("   1. 각 시나리오에 명확한 메타데이터 추가")
    print("   2. 성공/실패 케이스를 태그로 구분")
    print("   3. 오류 메시지를 메타데이터에 포함")
    print("   4. 재현 가능한 입력 값 기록")
    print("   5. 수정 전/후 트레이스 비교")

    if langsmith_enabled:
        print("\n🔧 LangSmith 디버깅 워크플로우:")
        print("   1. Runs 탭에서 실패한 실행 필터")
        print("   2. 해당 트레이스 클릭하여 상세 확인")
        print("   3. LLM 입력/출력 및 Tool 호출 검토")
        print("   4. 'Playground'에서 즉시 재현")
        print("   5. 수정 후 동일 입력으로 재테스트")
        print("\n   🎯 문제 해결 시간을 10배 단축할 수 있습니다!")
    else:
        print("\n💡 LANGSMITH_API_KEY를 설정하면 강력한 디버깅 도구를 사용할 수 있습니다.")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 10: 배포와 관측성 - 트레이싱 심화")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_custom_spans()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_trace_filtering()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_performance_analysis()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_trace_metadata()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_debugging_workflow()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 10-02: 트레이싱 심화를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 03_testing.py - 자동화 테스트")
    print("  2. 04_evaluation.py - 평가 및 벤치마크")
    print("  3. 05_deployment.py - 배포")
    print("\n📚 핵심 요약:")
    print("  • @traceable 데코레이터로 커스텀 스팬 생성")
    print("  • 태그와 메타데이터로 트레이스 필터링")
    print("  • 성능 분석으로 병목 지점 파악")
    print("  • 메타데이터로 비즈니스 인사이트 추출")
    print("  • 트레이싱은 디버깅의 필수 도구")
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
# 1. 고급 트레이싱 패턴:
#    - Nested spans (중첩 스팬)
#    - Distributed tracing (분산 트레이싱)
#    - Span attributes (스팬 속성)
#    - Span events (스팬 이벤트)
#
# 2. 성능 최적화 전략:
#    - 병렬 처리 (parallel execution)
#    - 배치 처리 (batch processing)
#    - 캐싱 전략 (caching strategies)
#    - 레이지 로딩 (lazy loading)
#
# 3. 트레이스 데이터 활용:
#    - 커스텀 대시보드 생성
#    - 알림 규칙 설정
#    - 데이터 내보내기 및 분석
#    - 장기 트렌드 분석
#
# 4. 보안 및 프라이버시:
#    - 민감 정보 마스킹
#    - PII (개인식별정보) 처리
#    - 데이터 보존 정책
#    - 접근 권한 관리
#
# 5. 비용 관리:
#    - 트레이싱 샘플링
#    - 데이터 저장 최적화
#    - 불필요한 트레이스 필터링
#    - 비용 할당 및 추적
#
# ============================================================================
