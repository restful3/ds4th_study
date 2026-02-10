"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: 배포와 관측성 (Deployment & Observability)
================================================================================

파일명: 04_evaluation.py
난이도: ⭐⭐⭐⭐⭐ (전문가)
예상 시간: 30분

📚 학습 목표:
  - 평가 메트릭 이해 및 적용
  - 평가 데이터셋 관리
  - 벤치마킹 수행
  - A/B 테스트 실행
  - 커스텀 평가자 작성

📖 공식 문서:
  • LangSmith: /official/30-langsmith-studio.md
  • Testing: /official/31-test.md

📄 교안 문서:
  • Part 10 개요: /docs/part10_deployment.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langsmith

🔑 필요한 환경변수:
  - OPENAI_API_KEY
  - LANGSMITH_API_KEY (선택)

🚀 실행 방법:
  python 04_evaluation.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import time
import statistics
from typing import Dict, Any, List
from datetime import datetime
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
# 예제 1: 평가 메트릭 소개
# ============================================================================

def example_1_evaluation_metrics():
    """평가 메트릭 소개"""
    print("=" * 70)
    print("📌 예제 1: 평가 메트릭 소개")
    print("=" * 70)

    print("""
📊 평가 메트릭 (Evaluation Metrics)이란?

정의:
  AI Agent의 성능을 정량적으로 측정하는 지표

왜 중요한가?
  • 객관적인 성능 평가
  • 버전 간 비교
  • 개선 효과 검증
  • 프로덕션 준비도 판단

주요 메트릭 카테고리:

1️⃣ 정확도 (Accuracy Metrics)
   • Correctness: 답변이 정확한가?
   • Relevance: 답변이 질문과 관련 있는가?
   • Completeness: 필요한 정보를 모두 포함하는가?

2️⃣ 성능 (Performance Metrics)
   • Latency: 응답 시간
   • Throughput: 처리량 (QPS)
   • Token Usage: 토큰 사용량

3️⃣ 품질 (Quality Metrics)
   • Coherence: 일관성
   • Fluency: 자연스러움
   • Helpfulness: 유용성

4️⃣ 신뢰성 (Reliability Metrics)
   • Success Rate: 성공률
   • Error Rate: 오류율
   • Consistency: 일관성 (같은 입력, 같은 출력)

5️⃣ 비용 (Cost Metrics)
   • Cost per Query: 쿼리당 비용
   • Token Efficiency: 토큰 효율성
    """)

    print("\n🔹 기본 메트릭 측정 예제:")
    print("-" * 70)

    @tool
    def get_capital(country: str) -> str:
        """국가의 수도를 반환합니다."""
        capitals = {
            "대한민국": "서울",
            "일본": "도쿄",
            "미국": "워싱턴 D.C.",
            "프랑스": "파리",
            "영국": "런던"
        }
        return capitals.get(country, f"{country}의 수도 정보가 없습니다.")

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_capital],
    )

    # 평가 데이터셋
    eval_dataset = [
        {
            "question": "대한민국의 수도는 어디인가요?",
            "expected_answer": "서울",
            "category": "simple"
        },
        {
            "question": "일본 수도 알려줘",
            "expected_answer": "도쿄",
            "category": "simple"
        },
        {
            "question": "미국과 프랑스의 수도를 모두 알려주세요",
            "expected_answer": ["워싱턴", "파리"],
            "category": "multi"
        },
        {
            "question": "존재하지 않는 나라 xyz의 수도는?",
            "expected_answer": None,
            "category": "edge"
        }
    ]

    print("\n평가 실행:")
    results = []

    for i, case in enumerate(eval_dataset, 1):
        print(f"\n[{i}] {case['question']}")

        start_time = time.time()
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": case['question']}]
            })
            answer = response['messages'][-1].content
            latency = time.time() - start_time

            print(f"  응답: {answer[:80]}...")
            print(f"  지연시간: {latency:.2f}초")

            # 정확도 평가
            if case['expected_answer'] is None:
                # 에러 케이스: "정보가 없다"는 메시지 확인
                is_correct = any(word in answer for word in ["없", "정보가 없", "모르"])
            elif isinstance(case['expected_answer'], list):
                # 다중 답변: 모든 키워드 포함 확인
                is_correct = all(exp in answer for exp in case['expected_answer'])
            else:
                # 단일 답변: 키워드 포함 확인
                is_correct = case['expected_answer'] in answer

            status = "✅" if is_correct else "❌"
            print(f"  정확도: {status}")

            results.append({
                "question": case['question'],
                "category": case['category'],
                "is_correct": is_correct,
                "latency": latency,
                "answer_length": len(answer)
            })

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            results.append({
                "question": case['question'],
                "category": case['category'],
                "is_correct": False,
                "latency": time.time() - start_time,
                "answer_length": 0
            })

    # 메트릭 집계
    print("\n" + "-" * 70)
    print("\n📊 평가 결과 요약:")

    accuracy = sum(1 for r in results if r['is_correct']) / len(results) * 100
    avg_latency = statistics.mean(r['latency'] for r in results)
    avg_answer_length = statistics.mean(r['answer_length'] for r in results)

    print(f"   정확도 (Accuracy): {accuracy:.1f}%")
    print(f"   평균 지연시간 (Latency): {avg_latency:.2f}초")
    print(f"   평균 응답 길이: {avg_answer_length:.0f}자")

    # 카테고리별 정확도
    print("\n   카테고리별 정확도:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['is_correct'])

    for cat, correctness in categories.items():
        cat_accuracy = sum(correctness) / len(correctness) * 100
        print(f"     • {cat}: {cat_accuracy:.1f}%")

    print("\n💡 메트릭 선택 가이드:")
    print("   • 도메인에 맞는 메트릭 선택")
    print("   • 여러 메트릭을 조합하여 종합 평가")
    print("   • 자동화된 메트릭 + 사람 평가")
    print("   • 시간에 따른 메트릭 추이 모니터링")


# ============================================================================
# 예제 2: 평가 데이터셋 관리
# ============================================================================

def example_2_evaluation_datasets():
    """평가 데이터셋 관리"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 평가 데이터셋 관리")
    print("=" * 70)

    print("""
📂 평가 데이터셋 관리:

평가 데이터셋의 중요성:
  • 일관된 성능 측정
  • 회귀(regression) 방지
  • 버전 간 비교 가능

데이터셋 구성 요소:
  1️⃣ 입력 (Input/Question)
  2️⃣ 예상 출력 (Expected Output)
  3️⃣ 메타데이터 (Category, Difficulty, Tags)
  4️⃣ 평가 기준 (Evaluation Criteria)

데이터셋 설계 원칙:
  • 실제 사용 사례 반영
  • 난이도 분포 균형
  • Edge cases 포함
  • 정기적 업데이트

LangSmith 데이터셋:
  • UI에서 데이터셋 생성/관리
  • 버전 관리
  • 팀 공유
  • 자동 평가 실행
    """)

    print("\n🔹 체계적인 데이터셋 예제:")
    print("-" * 70)

    # 고급 평가 데이터셋
    evaluation_dataset = {
        "name": "Calculator Agent v1.0",
        "description": "기본 계산 기능 평가",
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "test_cases": [
            {
                "id": "CALC-001",
                "category": "기본계산",
                "difficulty": "easy",
                "input": "10 더하기 20은?",
                "expected": "30",
                "tags": ["addition", "basic"],
                "weight": 1.0
            },
            {
                "id": "CALC-002",
                "category": "기본계산",
                "difficulty": "easy",
                "input": "100 빼기 25는?",
                "expected": "75",
                "tags": ["subtraction", "basic"],
                "weight": 1.0
            },
            {
                "id": "CALC-003",
                "category": "응용계산",
                "difficulty": "medium",
                "input": "1000원의 15% 할인가는?",
                "expected": "850",
                "tags": ["percentage", "discount"],
                "weight": 1.5
            },
            {
                "id": "CALC-004",
                "category": "복합계산",
                "difficulty": "hard",
                "input": "5000원에 10% 할인 후 10% 세금 추가하면?",
                "expected": "4950",
                "tags": ["multi-step", "complex"],
                "weight": 2.0
            },
            {
                "id": "CALC-005",
                "category": "엣지케이스",
                "difficulty": "edge",
                "input": "0으로 나누기",
                "expected": None,  # 에러 처리 확인
                "tags": ["error-handling", "edge"],
                "weight": 1.5
            }
        ]
    }

    # Calculator Tool
    @tool
    def calculate(expression: str) -> str:
        """수식을 계산합니다."""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except ZeroDivisionError:
            return "0으로 나눌 수 없습니다."
        except Exception as e:
            return f"계산 오류: {e}"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[calculate],
    )

    # 데이터셋 실행
    print(f"\n📊 데이터셋: {evaluation_dataset['name']}")
    print(f"   버전: {evaluation_dataset['version']}")
    print(f"   케이스 수: {len(evaluation_dataset['test_cases'])}")
    print("\n" + "-" * 70)

    results = []
    total_weight = 0
    weighted_score = 0

    for test_case in evaluation_dataset['test_cases']:
        print(f"\n[{test_case['id']}] {test_case['difficulty'].upper()}")
        print(f"  질문: {test_case['input']}")
        print(f"  카테고리: {test_case['category']}")
        print(f"  태그: {', '.join(test_case['tags'])}")
        print(f"  가중치: {test_case['weight']}")

        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": test_case['input']}]
            })
            answer = response['messages'][-1].content
            print(f"  응답: {answer[:80]}...")

            # 평가
            if test_case['expected'] is None:
                # 에러 처리 확인
                is_correct = any(word in answer for word in ["오류", "나눌 수 없", "불가능"])
            else:
                is_correct = test_case['expected'] in answer

            score = test_case['weight'] if is_correct else 0
            weighted_score += score
            total_weight += test_case['weight']

            status = "✅ PASS" if is_correct else "❌ FAIL"
            print(f"  결과: {status} (점수: {score}/{test_case['weight']})")

            results.append({
                "id": test_case['id'],
                "passed": is_correct,
                "weight": test_case['weight'],
                "score": score
            })

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            total_weight += test_case['weight']
            results.append({
                "id": test_case['id'],
                "passed": False,
                "weight": test_case['weight'],
                "score": 0
            })

    # 최종 점수
    print("\n" + "=" * 70)
    print("\n📊 최종 평가 결과:")

    final_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0
    pass_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    print(f"   가중 점수: {final_score:.1f}/100")
    print(f"   통과율: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)")

    # 난이도별 분석
    print("\n   난이도별 결과:")
    difficulty_map = {tc['id']: tc['difficulty'] for tc in evaluation_dataset['test_cases']}
    difficulty_results = {}

    for r in results:
        diff = difficulty_map[r['id']]
        if diff not in difficulty_results:
            difficulty_results[diff] = {"passed": 0, "total": 0}
        difficulty_results[diff]['total'] += 1
        if r['passed']:
            difficulty_results[diff]['passed'] += 1

    for diff, stats in difficulty_results.items():
        rate = stats['passed'] / stats['total'] * 100
        print(f"     • {diff}: {stats['passed']}/{stats['total']} ({rate:.1f}%)")

    print("\n💡 데이터셋 관리 팁:")
    print("   • 버전 관리로 변경 추적")
    print("   • 가중치로 중요도 반영")
    print("   • 실패 케이스를 데이터셋에 추가")
    print("   • 정기적으로 데이터셋 검토 및 업데이트")
    print("   • LangSmith에 저장하여 팀과 공유")


# ============================================================================
# 예제 3: 벤치마킹
# ============================================================================

def example_3_benchmarking():
    """벤치마킹"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 벤치마킹")
    print("=" * 70)

    print("""
⚡ 벤치마킹 (Benchmarking)이란?

정의:
  시스템 성능을 체계적으로 측정하고 비교하는 프로세스

벤치마킹 목적:
  1️⃣ 성능 기준선(baseline) 수립
  2️⃣ 최적화 효과 측정
  3️⃣ 모델/버전 간 비교
  4️⃣ 병목 지점 파악
  5️⃣ SLA(Service Level Agreement) 검증

측정 항목:
  • 처리량 (Throughput): QPS (Queries Per Second)
  • 지연시간 (Latency): p50, p95, p99
  • 리소스 사용량: CPU, 메모리
  • 비용: 토큰 사용량, API 비용
  • 정확도: Accuracy, F1 Score

벤치마크 시나리오:
  • Cold Start: 첫 요청
  • Warm: 캐시 적용
  • Peak Load: 최대 부하
  • Sustained Load: 지속 부하
    """)

    print("\n🔹 성능 벤치마크 예제:")
    print("-" * 70)

    @tool
    def search_docs(query: str) -> str:
        """문서를 검색합니다."""
        time.sleep(0.1)  # 검색 시뮬레이션
        return f"{query}에 대한 문서 3개 발견"

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[search_docs],
    )

    # 벤치마크 쿼리
    benchmark_queries = [
        "인공지능이란?",
        "LangChain 사용법",
        "Python 튜토리얼",
        "머신러닝 알고리즘",
        "데이터 분석 방법"
    ]

    print("\n벤치마크 실행 (5개 쿼리 x 3회 반복):")
    print("-" * 70)

    all_latencies = []
    all_token_counts = []

    for iteration in range(1, 4):
        print(f"\n🔄 Iteration {iteration}/3:")

        iteration_latencies = []

        for i, query in enumerate(benchmark_queries, 1):
            start = time.time()

            try:
                response = agent.invoke({
                    "messages": [{"role": "user", "content": query}]
                })
                latency = time.time() - start

                # 토큰 사용량 추정 (실제로는 response에서 가져옴)
                answer = response['messages'][-1].content
                estimated_tokens = len(answer.split()) * 1.3  # 대략적인 추정

                iteration_latencies.append(latency)
                all_latencies.append(latency)
                all_token_counts.append(estimated_tokens)

                print(f"  [{i}] {query[:30]:30s} | {latency:5.2f}s | ~{int(estimated_tokens):3d} tokens")

            except Exception as e:
                print(f"  [{i}] {query[:30]:30s} | ERROR: {e}")

        avg_latency = statistics.mean(iteration_latencies)
        print(f"  → 평균 지연: {avg_latency:.2f}s")

    # 통계 분석
    print("\n" + "=" * 70)
    print("\n📊 벤치마크 결과 분석:")

    latencies_sorted = sorted(all_latencies)
    n = len(latencies_sorted)

    p50 = latencies_sorted[int(n * 0.50)]
    p95 = latencies_sorted[int(n * 0.95)]
    p99 = latencies_sorted[int(n * 0.99)] if n > 100 else latencies_sorted[-1]

    print(f"\n   지연시간 (Latency):")
    print(f"     • 평균 (Mean): {statistics.mean(all_latencies):.2f}s")
    print(f"     • 중앙값 (Median): {statistics.median(all_latencies):.2f}s")
    print(f"     • 최소 (Min): {min(all_latencies):.2f}s")
    print(f"     • 최대 (Max): {max(all_latencies):.2f}s")
    print(f"     • p50: {p50:.2f}s")
    print(f"     • p95: {p95:.2f}s")
    print(f"     • p99: {p99:.2f}s")

    if len(all_latencies) > 1:
        print(f"     • 표준편차: {statistics.stdev(all_latencies):.2f}s")

    print(f"\n   처리량 (Throughput):")
    total_time = sum(all_latencies)
    qps = len(all_latencies) / total_time if total_time > 0 else 0
    print(f"     • QPS (Queries/Second): {qps:.2f}")

    print(f"\n   토큰 사용량:")
    print(f"     • 평균: {statistics.mean(all_token_counts):.0f} tokens")
    print(f"     • 총합: {sum(all_token_counts):.0f} tokens")

    # 성능 등급 판정
    print(f"\n   성능 등급:")
    if p95 < 2.0:
        grade = "🟢 우수 (Excellent)"
    elif p95 < 5.0:
        grade = "🟡 양호 (Good)"
    else:
        grade = "🔴 개선 필요 (Needs Improvement)"
    print(f"     {grade}")

    print("\n💡 벤치마킹 모범 사례:")
    print("   • 여러 번 반복 측정 (통계적 유의성)")
    print("   • Cold start와 warm start 구분")
    print("   • 다양한 쿼리 패턴 테스트")
    print("   • 시간대별, 날짜별 추이 모니터링")
    print("   • p95, p99 같은 백분위수 중요")
    print("   • 기준선 대비 회귀 방지")


# ============================================================================
# 예제 4: A/B 테스트
# ============================================================================

def example_4_ab_testing():
    """A/B 테스트"""
    print("\n" + "=" * 70)
    print("📌 예제 4: A/B 테스트")
    print("=" * 70)

    print("""
🔬 A/B 테스트란?

정의:
  두 가지 이상의 버전을 비교하여 어느 것이 더 나은지 검증

사용 사례:
  • 프롬프트 최적화
  • 모델 선택 (GPT-4 vs GPT-3.5)
  • Tool 구성 변경
  • 파라미터 튜닝

A/B 테스트 프로세스:
  1️⃣ 가설 수립
  2️⃣ 변형 A, B 정의
  3️⃣ 평가 메트릭 선정
  4️⃣ 테스트 실행
  5️⃣ 통계적 유의성 검증
  6️⃣ 의사결정

통계적 유의성:
  • 충분한 샘플 수
  • p-value < 0.05
  • 실질적 차이 (practical significance)
    """)

    print("\n🔹 A/B 테스트 예제:")
    print("-" * 70)

    @tool
    def get_info(topic: str) -> str:
        """주제에 대한 정보를 제공합니다."""
        return f"{topic}은(는) 중요한 주제입니다. 자세한 내용은 문서를 참조하세요."

    # 버전 A: 간결한 프롬프트
    print("\n버전 A: 간결한 시스템 프롬프트")
    agent_a = create_agent(
        model="gpt-4o-mini",
        tools=[get_info],
    )

    # 버전 B: 상세한 프롬프트 (동일 agent 사용, 실제로는 다른 설정)
    print("버전 B: 상세한 시스템 프롬프트")
    agent_b = create_agent(
        model="gpt-4o-mini",
        tools=[get_info],
    )

    # 테스트 쿼리
    test_queries = [
        "인공지능에 대해 알려줘",
        "머신러닝이 뭐야?",
        "Python 장점은?",
        "데이터 분석 방법",
        "클라우드 컴퓨팅 설명"
    ]

    print("\n" + "-" * 70)
    print("A/B 테스트 실행:")
    print("-" * 70)

    results_a = []
    results_b = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[테스트 {i}] {query}")

        # 버전 A 실행
        print("  🅰️  버전 A:")
        start = time.time()
        try:
            response_a = agent_a.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            answer_a = response_a['messages'][-1].content
            latency_a = time.time() - start
            length_a = len(answer_a)

            print(f"     응답: {answer_a[:60]}...")
            print(f"     지연: {latency_a:.2f}s | 길이: {length_a}자")

            results_a.append({
                "query": query,
                "latency": latency_a,
                "length": length_a,
                "success": True
            })
        except Exception as e:
            print(f"     ❌ 오류: {e}")
            results_a.append({
                "query": query,
                "latency": 0,
                "length": 0,
                "success": False
            })

        # 버전 B 실행
        print("  🅱️  버전 B:")
        start = time.time()
        try:
            response_b = agent_b.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            answer_b = response_b['messages'][-1].content
            latency_b = time.time() - start
            length_b = len(answer_b)

            print(f"     응답: {answer_b[:60]}...")
            print(f"     지연: {latency_b:.2f}s | 길이: {length_b}자")

            results_b.append({
                "query": query,
                "latency": latency_b,
                "length": length_b,
                "success": True
            })
        except Exception as e:
            print(f"     ❌ 오류: {e}")
            results_b.append({
                "query": query,
                "latency": 0,
                "length": 0,
                "success": False
            })

    # 결과 비교
    print("\n" + "=" * 70)
    print("\n📊 A/B 테스트 결과 비교:")

    success_rate_a = sum(1 for r in results_a if r['success']) / len(results_a) * 100
    success_rate_b = sum(1 for r in results_b if r['success']) / len(results_b) * 100

    avg_latency_a = statistics.mean(r['latency'] for r in results_a if r['success'])
    avg_latency_b = statistics.mean(r['latency'] for r in results_b if r['success'])

    avg_length_a = statistics.mean(r['length'] for r in results_a if r['success'])
    avg_length_b = statistics.mean(r['length'] for r in results_b if r['success'])

    print(f"\n   성공률:")
    print(f"     🅰️  버전 A: {success_rate_a:.1f}%")
    print(f"     🅱️  버전 B: {success_rate_b:.1f}%")

    print(f"\n   평균 지연시간:")
    print(f"     🅰️  버전 A: {avg_latency_a:.2f}s")
    print(f"     🅱️  버전 B: {avg_latency_b:.2f}s")
    latency_diff = ((avg_latency_b - avg_latency_a) / avg_latency_a * 100)
    print(f"     → 차이: {latency_diff:+.1f}%")

    print(f"\n   평균 응답 길이:")
    print(f"     🅰️  버전 A: {avg_length_a:.0f}자")
    print(f"     🅱️  버전 B: {avg_length_b:.0f}자")
    length_diff = ((avg_length_b - avg_length_a) / avg_length_a * 100)
    print(f"     → 차이: {length_diff:+.1f}%")

    # 권장사항
    print(f"\n   💡 권장사항:")
    if abs(latency_diff) < 5:
        print("     • 지연시간 차이가 미미함 (< 5%)")
    elif latency_diff < 0:
        print(f"     • 버전 A가 {abs(latency_diff):.1f}% 더 빠름 ✅")
    else:
        print(f"     • 버전 B가 {abs(latency_diff):.1f}% 더 느림 ⚠️")

    print("\n💡 A/B 테스트 모범 사례:")
    print("   • 하나의 변수만 변경")
    print("   • 충분한 샘플 수 확보 (최소 30+)")
    print("   • 여러 메트릭 종합 판단")
    print("   • 통계적 + 실질적 유의성 고려")
    print("   • 프로덕션 환경에서 점진적 롤아웃")


# ============================================================================
# 예제 5: 커스텀 평가자
# ============================================================================

def example_5_custom_evaluators():
    """커스텀 평가자"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 커스텀 평가자")
    print("=" * 70)

    print("""
🎯 커스텀 평가자 (Custom Evaluator)란?

정의:
  도메인 특화 요구사항에 맞는 평가 로직

기본 평가자의 한계:
  • 범용적, 도메인 특화 X
  • 비즈니스 규칙 반영 어려움
  • 복잡한 평가 기준 표현 제한

커스텀 평가자 사용 사례:
  1️⃣ 톤 앤 매너 검증
     • 친절한가? 전문적인가?

  2️⃣ 규정 준수 확인
     • 금지 단어 포함 여부
     • 필수 정보 포함 여부

  3️⃣ 도메인 정확도
     • 의료: 정확한 의학 용어 사용
     • 금융: 리스크 경고 포함

  4️⃣ 비즈니스 규칙
     • 특정 제품 우선 추천
     • 가격 범위 준수
    """)

    print("\n🔹 커스텀 평가자 예제:")
    print("-" * 70)

    # 평가자 정의
    class CustomerServiceEvaluator:
        """고객 서비스 응답 평가자"""

        def __init__(self):
            self.required_phrases = ["감사합니다", "도와드리겠습니다"]
            self.forbidden_words = ["모르겠", "안 돼", "불가능"]
            self.professional_keywords = ["고객님", "확인", "처리"]

        def evaluate(self, answer: str) -> Dict[str, Any]:
            """응답을 평가합니다."""
            scores = {}

            # 1. 예의 점수
            politeness_score = sum(
                1 for phrase in self.required_phrases if phrase in answer
            ) / len(self.required_phrases) * 100
            scores['politeness'] = politeness_score

            # 2. 금지어 확인
            has_forbidden = any(word in answer for word in self.forbidden_words)
            scores['no_forbidden_words'] = 0 if has_forbidden else 100

            # 3. 전문성 점수
            professional_count = sum(
                1 for keyword in self.professional_keywords if keyword in answer
            )
            scores['professionalism'] = min(professional_count / 2 * 100, 100)

            # 4. 길이 적절성
            length = len(answer)
            if 50 <= length <= 300:
                scores['appropriate_length'] = 100
            elif length < 50:
                scores['appropriate_length'] = 50
            else:
                scores['appropriate_length'] = 70

            # 종합 점수
            overall_score = statistics.mean(scores.values())

            return {
                "scores": scores,
                "overall": overall_score,
                "passed": overall_score >= 70,
                "details": {
                    "has_forbidden": has_forbidden,
                    "length": length,
                    "professional_count": professional_count
                }
            }

    @tool
    def handle_complaint(issue: str) -> str:
        """고객 불만을 처리합니다."""
        responses = {
            "배송 지연": "고객님, 배송 지연에 대해 사과드립니다. 즉시 확인하여 처리하겠습니다. 감사합니다.",
            "제품 불량": "고객님, 불편을 끼쳐 죄송합니다. 교환 또는 환불 처리를 도와드리겠습니다.",
            "기타": "고객님의 문의사항을 확인했습니다. 최선을 다해 도와드리겠습니다. 감사합니다."
        }
        return responses.get(issue, responses["기타"])

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[handle_complaint],
    )

    # 평가자 초기화
    evaluator = CustomerServiceEvaluator()

    # 테스트 케이스
    test_cases = [
        "배송이 너무 늦어요. 언제 오나요?",
        "받은 제품이 파손되었습니다.",
        "환불 가능한가요?",
        "서비스가 형편없네요!"
    ]

    print("\n커스텀 평가 실행:")
    print("-" * 70)

    evaluation_results = []

    for i, user_query in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] 사용자: {user_query}")

        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_query}]
            })
            answer = response['messages'][-1].content

            print(f"  응답: {answer}")

            # 커스텀 평가 실행
            eval_result = evaluator.evaluate(answer)

            print(f"\n  📊 평가 결과:")
            print(f"     종합 점수: {eval_result['overall']:.1f}/100")
            print(f"     통과 여부: {'✅ PASS' if eval_result['passed'] else '❌ FAIL'}")
            print(f"\n     세부 점수:")
            for metric, score in eval_result['scores'].items():
                status = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
                print(f"       {status} {metric}: {score:.1f}/100")

            evaluation_results.append({
                "query": user_query,
                "answer": answer,
                "evaluation": eval_result,
                "passed": eval_result['passed']
            })

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            evaluation_results.append({
                "query": user_query,
                "passed": False
            })

    # 최종 요약
    print("\n" + "=" * 70)
    print("\n📊 커스텀 평가 요약:")

    pass_count = sum(1 for r in evaluation_results if r['passed'])
    total_count = len(evaluation_results)
    pass_rate = pass_count / total_count * 100

    print(f"   통과율: {pass_count}/{total_count} ({pass_rate:.1f}%)")

    # 평균 점수
    avg_scores = {}
    for r in evaluation_results:
        if 'evaluation' in r:
            for metric, score in r['evaluation']['scores'].items():
                if metric not in avg_scores:
                    avg_scores[metric] = []
                avg_scores[metric].append(score)

    print(f"\n   평균 메트릭 점수:")
    for metric, scores in avg_scores.items():
        avg = statistics.mean(scores)
        print(f"     • {metric}: {avg:.1f}/100")

    print("\n💡 커스텀 평가자 작성 팁:")
    print("   • 명확한 평가 기준 정의")
    print("   • 도메인 전문가와 협업")
    print("   • 점수와 함께 설명 제공")
    print("   • 정기적으로 평가 기준 업데이트")
    print("   • 사람 평가와 병행")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 10: 배포와 관측성 - 평가 및 벤치마크")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_evaluation_metrics()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_evaluation_datasets()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_benchmarking()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_ab_testing()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_custom_evaluators()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 10-04: 평가 및 벤치마크를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 05_deployment.py - 배포")
    print("  2. 06_observability.py - 관측성")
    print("\n📚 핵심 요약:")
    print("  • 평가 메트릭으로 성능 정량화")
    print("  • 평가 데이터셋으로 체계적 관리")
    print("  • 벤치마킹으로 성능 기준 수립")
    print("  • A/B 테스트로 최적 버전 선택")
    print("  • 커스텀 평가자로 도메인 특화 평가")
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
# 1. 고급 평가 메트릭:
#    - BLEU, ROUGE (텍스트 유사도)
#    - BERTScore (의미적 유사도)
#    - Perplexity
#    - Human Evaluation
#
# 2. 자동화된 평가:
#    - LangSmith Evaluators
#    - Continuous Evaluation
#    - Regression Detection
#    - Alert on Performance Drop
#
# 3. 통계적 분석:
#    - t-test, ANOVA
#    - Confidence Intervals
#    - Effect Size
#    - Sample Size Calculation
#
# 4. 프로덕션 평가:
#    - Online Evaluation
#    - Shadow Mode
#    - Canary Deployment
#    - Blue-Green Deployment
#
# 5. 비즈니스 메트릭:
#    - User Satisfaction
#    - Task Completion Rate
#    - Time to Resolution
#    - Cost per Interaction
#
# ============================================================================
