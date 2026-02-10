"""
Searcher Agent
정보를 수집하는 Agent
"""

from typing import Dict, List, Any
from .base import BaseAgent


class SearcherAgent(BaseAgent):
    """정보 수집 Agent"""

    def __init__(self, name: str, llm, verbose: bool = False):
        super().__init__(name, llm, verbose)
        self.search_tool = self._init_search_tool()

    def _init_search_tool(self):
        """검색 도구 초기화"""
        try:
            # Tavily 검색 시도
            import os
            if os.getenv("TAVILY_API_KEY"):
                from langchain_community.tools.tavily_search import TavilySearchResults
                return TavilySearchResults(
                    max_results=5,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=False,
                )
        except Exception:
            pass

        # DuckDuckGo 대체
        try:
            from langchain_community.tools import DuckDuckGoSearchResults
            self.log("Tavily를 사용할 수 없어 DuckDuckGo를 사용합니다.")
            return DuckDuckGoSearchResults(max_results=5)
        except Exception as e:
            self.log(f"검색 도구 초기화 실패: {e}", force=True)
            return None

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        정보 수집 실행

        Args:
            input_data: {"sub_questions": List[str], "keywords": List[str]}

        Returns:
            Dict: 검색 결과
        """
        sub_questions = input_data.get("sub_questions", [])
        keywords = input_data.get("keywords", [])

        self.log("정보 수집 시작...", force=True)

        all_sources = []

        # 하위 질문별로 검색
        for i, question in enumerate(sub_questions, 1):
            self.log(f"질문 {i}/{len(sub_questions)} 검색 중: {question[:50]}...")
            sources = self.search_single_question(question)
            all_sources.extend(sources)

        # 중복 제거
        unique_sources = self._deduplicate_sources(all_sources)

        self.log(f"총 {len(unique_sources)}개 소스 수집 완료", force=True)

        return {
            "sources": unique_sources,
            "total_count": len(unique_sources),
        }

    def search_single_question(self, question: str) -> List[Dict]:
        """
        단일 질문에 대한 검색

        Args:
            question: 검색 질문

        Returns:
            List[Dict]: 검색 결과
        """
        if not self.search_tool:
            # 검색 도구가 없으면 더미 데이터 반환
            return [{
                "title": f"검색 결과: {question}",
                "content": "검색 도구를 사용할 수 없습니다.",
                "url": "N/A",
            }]

        try:
            results = self.search_tool.invoke(question)

            # 결과 파싱
            sources = []
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        sources.append({
                            "title": result.get("title", "제목 없음"),
                            "content": result.get("content", result.get("snippet", "")),
                            "url": result.get("url", result.get("link", "")),
                        })
                    elif isinstance(result, str):
                        sources.append({
                            "title": question,
                            "content": result,
                            "url": "N/A",
                        })
            elif isinstance(results, str):
                sources.append({
                    "title": question,
                    "content": results,
                    "url": "N/A",
                })

            return sources

        except Exception as e:
            self.log(f"검색 오류: {e}")
            return [{
                "title": f"검색 실패: {question}",
                "content": str(e),
                "url": "N/A",
            }]

    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """
        중복 소스 제거

        Args:
            sources: 소스 리스트

        Returns:
            List[Dict]: 중복 제거된 소스 리스트
        """
        seen_urls = set()
        unique_sources = []

        for source in sources:
            url = source.get("url", "")
            if url and url != "N/A" and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(source)
            elif url == "N/A" and source.get("content"):
                # URL이 없어도 유용한 콘텐츠가 있으면 포함
                unique_sources.append(source)

        return unique_sources
