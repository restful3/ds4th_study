"""
================================================================================
LangChain AI Agent 마스터 교안
Part 7: Multi-Agent Systems
================================================================================

파일명: 03_subagents_async.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - 동기 vs 비동기 Subagent 비교
  - asyncio를 사용한 병렬 Subagent 실행
  - 여러 Subagent 동시 호출
  - 결과 수집 및 통합
  - 실전: 여러 소스 동시 검색

📖 공식 문서:
  • Subagents: /official/23-subagents.md

📄 교안 문서:
  • Part 7 Subagents: /docs/part07_multi_agent.md (Section 2.4)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 03_subagents_async.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import asyncio
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ============================================================================
# 예제 1: 동기 vs 비동기 비교
# ============================================================================

def example_1_sync_vs_async():
    """동기와 비동기 Subagent의 성능 비교"""
    print("=" * 70)
    print("📌 예제 1: 동기 vs 비동기 비교")
    print("=" * 70)

    print("""
💡 동기 (Synchronous):
   - Agent를 순차적으로 실행
   - 이전 Agent가 완료되어야 다음 Agent 시작
   - 총 실행 시간 = 각 Agent 시간의 합

⚡ 비동기 (Asynchronous):
   - 여러 Agent를 동시에 실행
   - 모든 Agent가 병렬로 작동
   - 총 실행 시간 = 가장 느린 Agent의 시간
    """)

    # 간단한 작업 시뮬레이션
    def slow_task(name: str, seconds: int) -> str:
        """시간이 걸리는 작업 시뮬레이션"""
        print(f"  [{name}] 시작...")
        time.sleep(seconds)
        print(f"  [{name}] 완료 ({seconds}초)")
        return f"{name} 결과"

    async def slow_task_async(name: str, seconds: int) -> str:
        """비동기 작업 시뮬레이션"""
        print(f"  [{name}] 시작...")
        await asyncio.sleep(seconds)
        print(f"  [{name}] 완료 ({seconds}초)")
        return f"{name} 결과"

    print("\n🐢 동기 방식 (순차 실행):")
    print("-" * 70)

    start = time.time()
    result1 = slow_task("Agent 1", 1)
    result2 = slow_task("Agent 2", 1)
    result3 = slow_task("Agent 3", 1)
    sync_time = time.time() - start

    print(f"\n총 실행 시간: {sync_time:.2f}초")

    print("\n⚡ 비동기 방식 (병렬 실행):")
    print("-" * 70)

    async def run_async():
        start = time.time()
        results = await asyncio.gather(
            slow_task_async("Agent 1", 1),
            slow_task_async("Agent 2", 1),
            slow_task_async("Agent 3", 1),
        )
        async_time = time.time() - start
        print(f"\n총 실행 시간: {async_time:.2f}초")
        return async_time

    async_time = asyncio.run(run_async())

    print(f"\n📊 성능 향상:")
    print("-" * 70)
    print(f"동기: {sync_time:.2f}초")
    print(f"비동기: {async_time:.2f}초")
    print(f"향상: {(sync_time / async_time):.1f}배 빠름!")

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 2: asyncio로 병렬 Subagent 실행
# ============================================================================

def example_2_parallel_subagents():
    """asyncio를 사용하여 여러 Subagent를 병렬로 실행"""
    print("\n" + "=" * 70)
    print("📌 예제 2: asyncio로 병렬 Subagent 실행")
    print("=" * 70)

    print("""
💡 asyncio.gather() 사용:
   - 여러 비동기 함수를 동시에 실행
   - 모든 작업이 완료될 때까지 대기
   - 결과를 리스트로 반환
    """)

    # 비동기 Subagent들
    async def research_subagent_async(topic: str) -> str:
        """비동기 리서치 Subagent"""
        prompt = f"""
당신은 리서치 전문가입니다.
'{topic}'에 대한 간단한 정보를 제공하세요 (2-3문장).
"""
        # 비동기 LLM 호출
        response = await llm.ainvoke(prompt)
        return response.content

    async def analysis_subagent_async(topic: str) -> str:
        """비동기 분석 Subagent"""
        prompt = f"""
당신은 분석 전문가입니다.
'{topic}'의 장단점을 분석하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        return response.content

    async def summary_subagent_async(topic: str) -> str:
        """비동기 요약 Subagent"""
        prompt = f"""
당신은 요약 전문가입니다.
'{topic}'를 한 문장으로 요약하세요.
"""
        response = await llm.ainvoke(prompt)
        return response.content

    # 병렬 실행
    async def run_parallel(topic: str):
        print(f"\n주제: {topic}")
        print("\n🚀 3개 Subagent를 병렬로 실행합니다...")
        print("-" * 70)

        start = time.time()

        # 모든 Subagent를 동시에 실행
        results = await asyncio.gather(
            research_subagent_async(topic),
            analysis_subagent_async(topic),
            summary_subagent_async(topic),
        )

        elapsed = time.time() - start

        print(f"\n✅ 완료! (총 {elapsed:.2f}초)")
        print("\n결과:")
        print("-" * 70)
        print(f"\n1. 리서치:\n{results[0]}")
        print(f"\n2. 분석:\n{results[1]}")
        print(f"\n3. 요약:\n{results[2]}")

    topic = input("\n주제를 입력하세요: ").strip() or "클라우드 컴퓨팅"
    asyncio.run(run_parallel(topic))

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 3: 여러 Subagent 동시 호출
# ============================================================================

def example_3_multiple_concurrent_calls():
    """다양한 Subagent를 동시에 호출하여 데이터 수집"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 여러 Subagent 동시 호출")
    print("=" * 70)

    print("""
💡 독립적인 Subagent들을 병렬로 실행:
   - 뉴스 검색 Subagent
   - 소셜 미디어 분석 Subagent
   - 블로그 검색 Subagent
   - 학술 논문 검색 Subagent
    """)

    # 다양한 소스의 Subagent들
    async def news_subagent_async(topic: str) -> str:
        """뉴스 검색 Subagent"""
        prompt = f"""
당신은 뉴스 검색 전문가입니다.
'{topic}'에 대한 최신 뉴스 2개를 요약하세요.
"""
        response = await llm.ainvoke(prompt)
        return f"[뉴스]\n{response.content}"

    async def social_subagent_async(topic: str) -> str:
        """소셜 미디어 분석 Subagent"""
        prompt = f"""
당신은 소셜 미디어 분석가입니다.
'{topic}'에 대한 SNS 반응을 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        return f"[SNS]\n{response.content}"

    async def blog_subagent_async(topic: str) -> str:
        """블로그 검색 Subagent"""
        prompt = f"""
당신은 블로그 검색 전문가입니다.
'{topic}'에 대한 블로그 글 2개를 요약하세요.
"""
        response = await llm.ainvoke(prompt)
        return f"[블로그]\n{response.content}"

    async def academic_subagent_async(topic: str) -> str:
        """학술 논문 검색 Subagent"""
        prompt = f"""
당신은 학술 연구 전문가입니다.
'{topic}'에 대한 주요 연구 결과를 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        return f"[학술]\n{response.content}"

    # 모든 소스에서 동시에 검색
    async def search_all_sources(topic: str):
        print(f"\n주제: {topic}")
        print("\n🔍 4개 소스에서 동시에 검색 중...")
        print("-" * 70)

        start = time.time()

        # 병렬 실행
        results = await asyncio.gather(
            news_subagent_async(topic),
            social_subagent_async(topic),
            blog_subagent_async(topic),
            academic_subagent_async(topic),
        )

        elapsed = time.time() - start

        print(f"\n✅ 검색 완료! (총 {elapsed:.2f}초)")
        print("\n통합 결과:")
        print("=" * 70)
        for result in results:
            print(f"\n{result}")
            print("-" * 70)

    topic = input("\n검색 주제: ").strip() or "인공지능 윤리"
    asyncio.run(search_all_sources(topic))

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 4: 결과 수집 및 통합
# ============================================================================

def example_4_result_aggregation():
    """여러 Subagent의 결과를 수집하고 통합"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 결과 수집 및 통합")
    print("=" * 70)

    print("""
💡 결과 통합 패턴:
   1. 여러 Subagent를 병렬로 실행
   2. 모든 결과 수집
   3. 통합 Subagent가 결과를 종합
    """)

    # 데이터 수집 Subagent들
    async def source1_subagent(query: str) -> str:
        """소스 1 데이터 수집"""
        prompt = f"'{query}'에 대한 정보 1 (간단히)"
        response = await llm.ainvoke(prompt)
        return response.content

    async def source2_subagent(query: str) -> str:
        """소스 2 데이터 수집"""
        prompt = f"'{query}'에 대한 정보 2 (간단히)"
        response = await llm.ainvoke(prompt)
        return response.content

    async def source3_subagent(query: str) -> str:
        """소스 3 데이터 수집"""
        prompt = f"'{query}'에 대한 정보 3 (간단히)"
        response = await llm.ainvoke(prompt)
        return response.content

    # 통합 Subagent
    async def aggregator_subagent(all_data: list[str]) -> str:
        """여러 소스의 데이터를 통합"""
        combined = "\n\n".join([f"소스 {i+1}:\n{data}" for i, data in enumerate(all_data)])

        prompt = f"""
당신은 정보 통합 전문가입니다.
다음 여러 소스의 정보를 종합하여 하나의 일관된 요약을 작성하세요:

{combined}

핵심 내용을 3-4문장으로 정리하세요.
"""
        response = await llm.ainvoke(prompt)
        return response.content

    # 전체 프로세스
    async def collect_and_aggregate(query: str):
        print(f"\n쿼리: {query}")
        print("\n[1/2] 데이터 수집 중...")
        print("-" * 70)

        # 병렬로 데이터 수집
        data_results = await asyncio.gather(
            source1_subagent(query),
            source2_subagent(query),
            source3_subagent(query),
        )

        print("✅ 데이터 수집 완료")

        print("\n[2/2] 데이터 통합 중...")
        print("-" * 70)

        # 결과 통합
        final_result = await aggregator_subagent(data_results)

        print("✅ 통합 완료")
        print("\n최종 결과:")
        print("=" * 70)
        print(final_result)

    query = input("\n검색어: ").strip() or "양자 암호화"
    asyncio.run(collect_and_aggregate(query))

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 예제 5: 실전 - 여러 소스 동시 검색
# ============================================================================

def example_5_multi_source_search():
    """실전: 여러 소스에서 동시에 정보를 검색하고 종합하는 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 여러 소스 동시 검색")
    print("=" * 70)

    print("""
🎯 실전 시나리오: 종합 정보 검색 시스템

단계:
   1. 사용자 질문 입력
   2. 5개 Subagent가 병렬로 검색:
      - 웹 검색
      - 뉴스 검색
      - 학술 자료 검색
      - 비디오 검색
      - Q&A 검색
   3. 결과 통합 및 정리
    """)

    # 각 소스별 검색 Subagent
    async def web_search_subagent(query: str) -> dict:
        """웹 검색 Subagent"""
        start = time.time()
        prompt = f"""
웹에서 '{query}'를 검색한 결과를 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        elapsed = time.time() - start

        return {
            "source": "웹 검색",
            "content": response.content,
            "time": elapsed
        }

    async def news_search_subagent(query: str) -> dict:
        """뉴스 검색 Subagent"""
        start = time.time()
        prompt = f"""
'{query}' 관련 최신 뉴스를 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        elapsed = time.time() - start

        return {
            "source": "뉴스",
            "content": response.content,
            "time": elapsed
        }

    async def academic_search_subagent(query: str) -> dict:
        """학술 자료 검색 Subagent"""
        start = time.time()
        prompt = f"""
'{query}' 관련 학술 연구를 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        elapsed = time.time() - start

        return {
            "source": "학술 자료",
            "content": response.content,
            "time": elapsed
        }

    async def video_search_subagent(query: str) -> dict:
        """비디오 검색 Subagent"""
        start = time.time()
        prompt = f"""
'{query}' 관련 교육 영상 내용을 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        elapsed = time.time() - start

        return {
            "source": "비디오",
            "content": response.content,
            "time": elapsed
        }

    async def qa_search_subagent(query: str) -> dict:
        """Q&A 검색 Subagent"""
        start = time.time()
        prompt = f"""
'{query}' 관련 Q&A를 요약하세요 (2-3문장).
"""
        response = await llm.ainvoke(prompt)
        elapsed = time.time() - start

        return {
            "source": "Q&A",
            "content": response.content,
            "time": elapsed
        }

    # 결과 통합
    async def synthesize_results(results: list[dict]) -> str:
        """모든 검색 결과를 종합"""
        combined = "\n\n".join([
            f"[{r['source']}]\n{r['content']}"
            for r in results
        ])

        prompt = f"""
당신은 정보 종합 전문가입니다.
다음 여러 소스의 검색 결과를 종합하여 일관된 답변을 작성하세요:

{combined}

종합 답변 (5-6문장):
"""
        response = await llm.ainvoke(prompt)
        return response.content

    # 전체 검색 시스템
    async def comprehensive_search(query: str):
        print(f"\n질문: {query}")
        print("\n" + "=" * 70)
        print("🔍 5개 소스에서 동시에 검색 중...")
        print("-" * 70)

        start_total = time.time()

        # 모든 검색을 병렬로 실행
        results = await asyncio.gather(
            web_search_subagent(query),
            news_search_subagent(query),
            academic_search_subagent(query),
            video_search_subagent(query),
            qa_search_subagent(query),
        )

        search_time = time.time() - start_total

        print(f"\n✅ 검색 완료 (총 {search_time:.2f}초)")
        print("\n개별 결과:")
        print("=" * 70)

        for result in results:
            print(f"\n[{result['source']}] ({result['time']:.2f}초)")
            print(result['content'])
            print("-" * 70)

        print("\n🔄 결과 종합 중...")
        synthesized = await synthesize_results(results)

        total_time = time.time() - start_total

        print("\n📊 종합 답변:")
        print("=" * 70)
        print(synthesized)
        print("\n" + "=" * 70)
        print(f"⏱️  총 소요 시간: {total_time:.2f}초")

    query = input("\n질문을 입력하세요: ").strip()
    if not query:
        query = "블록체인 기술의 미래는 어떻게 될까요?"

    asyncio.run(comprehensive_search(query))

    input("\n⏎ Enter를 눌러 계속...")

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("=" * 70)
    print("Part 7: Multi-Agent Systems")
    print("03. Subagents (비동기)")
    print("=" * 70)

    while True:
        print("\n")
        print("📚 실행할 예제를 선택하세요:")
        print("-" * 70)
        print("1. 동기 vs 비동기 비교")
        print("2. asyncio로 병렬 Subagent 실행")
        print("3. 여러 Subagent 동시 호출")
        print("4. 결과 수집 및 통합")
        print("5. 실전: 여러 소스 동시 검색")
        print("0. 종료")
        print("-" * 70)

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_sync_vs_async()
        elif choice == "2":
            example_2_parallel_subagents()
        elif choice == "3":
            example_3_multiple_concurrent_calls()
        elif choice == "4":
            example_4_result_aggregation()
        elif choice == "5":
            example_5_multi_source_search()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다.")

    print("\n" + "=" * 70)
    print("📚 학습 완료!")
    print("=" * 70)
    print("""
✅ 배운 내용:
   - 동기 vs 비동기 Subagent 비교
   - asyncio.gather()로 병렬 실행
   - 여러 Subagent 동시 호출
   - 결과 수집 및 통합
   - 실전 종합 검색 시스템

💡 핵심 요약:
   ┌─────────────────────────────────────────────────────────────────┐
   │ 비동기 Subagent는 여러 작업을 동시에 실행하여 성능 향상        │
   │                                                                   │
   │ 주요 이점:                                                       │
   │ • 3-5배 빠른 실행 속도                                          │
   │ • 독립적인 작업을 병렬로 처리                                   │
   │ • asyncio.gather()로 간단한 구현                                │
   │ • 결과를 리스트로 쉽게 수집                                     │
   │                                                                   │
   │ 사용 시점:                                                       │
   │ • 여러 소스에서 데이터 수집                                     │
   │ • 독립적인 분석 작업                                            │
   │ • 응답 속도가 중요한 경우                                       │
   └─────────────────────────────────────────────────────────────────┘
    """)

if __name__ == "__main__":
    main()
