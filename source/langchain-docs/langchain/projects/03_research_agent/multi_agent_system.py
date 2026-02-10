"""
Multi-Agent Research System
다중 에이전트 연구 시스템
"""

import time
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI

from agents.planner import PlannerAgent
from agents.searcher import SearcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent


class ResearchAgentSystem:
    """Multi-Agent 연구 시스템"""

    def __init__(self, verbose: bool = False):
        """
        초기화

        Args:
            verbose: 상세 로그 출력 여부
        """
        self.verbose = verbose

        # LLM 초기화 (각 Agent별로 최적 모델 선택 가능)
        self.planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.searcher_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.analyst_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.writer_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  # 고품질 작성용

        # Agent 초기화
        self.planner = PlannerAgent("Planner", self.planner_llm, verbose=verbose)
        self.searcher = SearcherAgent("Searcher", self.searcher_llm, verbose=verbose)
        self.analyst = AnalystAgent("Analyst", self.analyst_llm, verbose=verbose)
        self.writer = WriterAgent("Writer", self.writer_llm, verbose=verbose)

        # 상태 관리
        self.state: Dict[str, Any] = {}

    def research(self, topic: str) -> str:
        """
        전체 연구 프로세스 실행

        Args:
            topic: 연구 주제

        Returns:
            str: 최종 보고서
        """
        start_time = time.time()

        # 초기 상태 설정
        self.state = {
            "topic": topic,
            "status": "started",
            "start_time": start_time,
        }

        try:
            # 1단계: 연구 계획 수립
            self._log_step(1, "연구 계획 수립", "📋")
            plan = self.planner.run({"topic": topic})
            self.state["plan"] = plan

            if self.verbose:
                self._print_plan(plan)

            # 2단계: 정보 수집
            self._log_step(2, "정보 수집", "🔍")
            search_results = self.searcher.run(plan)
            self.state["search_results"] = search_results

            if self.verbose:
                self._print_search_results(search_results)

            # 3단계: 데이터 분석
            self._log_step(3, "데이터 분석", "📊")
            analysis = self.analyst.run({
                "plan": plan,
                "search_results": search_results
            })
            self.state["analysis"] = analysis

            if self.verbose:
                self._print_analysis(analysis)

            # 4단계: 보고서 작성
            self._log_step(4, "보고서 작성", "✍️")
            report = self.writer.run({
                "topic": topic,
                "plan": plan,
                "analysis": analysis
            })
            self.state["report"] = report

            # 완료
            elapsed_time = time.time() - start_time
            self.state["status"] = "completed"
            self.state["elapsed_time"] = elapsed_time

            self._log(f"\n✅ 연구 완료! (소요 시간: {elapsed_time:.1f}초)")

            return report

        except Exception as e:
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            raise

    def _log_step(self, step: int, description: str, icon: str):
        """단계 로그 출력"""
        print(f"\n{icon} 단계 {step}/4: {description}")
        if not self.verbose:
            print("   처리 중...", end="", flush=True)

    def _log(self, message: str):
        """일반 로그 출력"""
        if self.verbose or "완료" in message or "실패" in message:
            print(message)

    def _print_plan(self, plan: Dict):
        """계획 출력 (verbose 모드)"""
        print("\n   📋 연구 계획:")
        print(f"   주제: {plan['topic']}")
        print(f"\n   하위 질문 ({len(plan['sub_questions'])}개):")
        for i, q in enumerate(plan['sub_questions'], 1):
            print(f"   {i}. {q}")
        print(f"\n   검색 키워드: {', '.join(plan['keywords'][:5])}")

    def _print_search_results(self, results: Dict):
        """검색 결과 출력 (verbose 모드)"""
        total_sources = len(results.get('sources', []))
        print(f"\n   🔍 검색 결과: {total_sources}개 소스")
        for i, source in enumerate(results.get('sources', [])[:3], 1):
            print(f"   {i}. {source.get('title', 'Unknown')[:50]}...")

    def _print_analysis(self, analysis: Dict):
        """분석 결과 출력 (verbose 모드)"""
        insights = analysis.get('insights', [])
        print(f"\n   📊 핵심 인사이트: {len(insights)}개")
        for i, insight in enumerate(insights[:3], 1):
            print(f"   {i}. {insight[:60]}...")

    def get_state(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return self.state.copy()

    def reset(self):
        """상태 초기화"""
        self.state = {}


class AsyncResearchAgentSystem(ResearchAgentSystem):
    """
    비동기 Multi-Agent 연구 시스템
    (도전 과제용)
    """

    async def research_async(self, topic: str) -> str:
        """
        비동기 연구 프로세스 실행

        Args:
            topic: 연구 주제

        Returns:
            str: 최종 보고서
        """
        import asyncio

        start_time = time.time()

        # 1단계: 계획 수립
        plan = await self._run_agent_async(self.planner, {"topic": topic})

        # 2단계: 병렬 정보 수집
        search_tasks = []
        for question in plan['sub_questions']:
            task = self._search_question_async(question)
            search_tasks.append(task)

        search_results_list = await asyncio.gather(*search_tasks)

        # 결과 통합
        search_results = {
            "sources": [item for sublist in search_results_list for item in sublist]
        }

        # 3단계: 분석
        analysis = await self._run_agent_async(
            self.analyst,
            {"plan": plan, "search_results": search_results}
        )

        # 4단계: 보고서 작성
        report = await self._run_agent_async(
            self.writer,
            {"topic": topic, "plan": plan, "analysis": analysis}
        )

        elapsed_time = time.time() - start_time
        print(f"\n✅ 비동기 연구 완료! (소요 시간: {elapsed_time:.1f}초)")

        return report

    async def _run_agent_async(self, agent, input_data: Dict) -> Dict:
        """Agent 비동기 실행"""
        import asyncio
        # 실제 구현에서는 Agent의 run 메서드를 비동기로 변경 필요
        return await asyncio.to_thread(agent.run, input_data)

    async def _search_question_async(self, question: str) -> list:
        """질문 비동기 검색"""
        import asyncio
        # 실제 구현에서는 searcher의 검색 메서드를 비동기로 변경 필요
        result = await asyncio.to_thread(
            self.searcher.search_single_question,
            question
        )
        return result
