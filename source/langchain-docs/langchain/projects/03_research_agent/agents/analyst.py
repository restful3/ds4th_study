"""
Analyst Agent
수집된 데이터를 분석하는 Agent
"""

from typing import Dict, List, Any
from .base import BaseAgent


class AnalystAgent(BaseAgent):
    """데이터 분석 Agent"""

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        수집된 정보를 분석

        Args:
            input_data: {"plan": Dict, "search_results": Dict}

        Returns:
            Dict: 분석 결과
        """
        plan = input_data.get("plan", {})
        search_results = input_data.get("search_results", {})
        sources = search_results.get("sources", [])

        self.log("데이터 분석 시작...", force=True)

        # 핵심 인사이트 추출
        insights = self._extract_insights(plan["topic"], sources)
        self.log(f"{len(insights)}개 핵심 인사이트 추출 완료")

        # 정보 요약
        summary = self._summarize_information(sources)
        self.log("정보 요약 완료")

        # 신뢰도 평가
        credibility = self._assess_credibility(sources)
        self.log("신뢰도 평가 완료")

        analysis = {
            "insights": insights,
            "summary": summary,
            "credibility": credibility,
            "source_count": len(sources),
        }

        return analysis

    def _extract_insights(self, topic: str, sources: List[Dict]) -> List[str]:
        """
        핵심 인사이트 추출

        Args:
            topic: 연구 주제
            sources: 소스 리스트

        Returns:
            List[str]: 인사이트 리스트
        """
        # 소스 내용 결합
        content_summary = self._combine_sources(sources, max_length=3000)

        prompt = f"""주제: {topic}

수집된 정보:
{content_summary}

위 정보에서 핵심 인사이트 5-7개를 추출하세요.

요구사항:
- 가장 중요하고 흥미로운 발견 사항
- 구체적인 데이터나 사실 포함
- 각 인사이트는 1-2문장으로 간결하게

인사이트를 다음 형식으로 작성하세요:
1. [첫 번째 인사이트]
2. [두 번째 인사이트]
...

핵심 인사이트:"""

        response = self.invoke_llm(prompt)

        # 인사이트 파싱
        insights = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                insight = line.split('.', 1)[-1].strip()
                insight = insight.lstrip('-•').strip()
                if insight and len(insight) > 10:
                    insights.append(insight)

        return insights[:7]

    def _summarize_information(self, sources: List[Dict]) -> str:
        """
        정보 요약

        Args:
            sources: 소스 리스트

        Returns:
            str: 요약 텍스트
        """
        content_summary = self._combine_sources(sources, max_length=2000)

        prompt = f"""다음 정보를 3-4문장으로 요약하세요:

{content_summary}

핵심 내용을 중심으로 간결하게 요약해주세요.

요약:"""

        return self.invoke_llm(prompt)

    def _assess_credibility(self, sources: List[Dict]) -> Dict[str, Any]:
        """
        소스 신뢰도 평가

        Args:
            sources: 소스 리스트

        Returns:
            Dict: 신뢰도 정보
        """
        # 간단한 신뢰도 지표
        total_sources = len(sources)
        sources_with_url = sum(1 for s in sources if s.get("url") and s.get("url") != "N/A")

        credibility_score = (sources_with_url / total_sources * 100) if total_sources > 0 else 0

        return {
            "total_sources": total_sources,
            "verified_sources": sources_with_url,
            "credibility_score": round(credibility_score, 1),
            "assessment": self._get_credibility_assessment(credibility_score),
        }

    def _get_credibility_assessment(self, score: float) -> str:
        """신뢰도 점수에 따른 평가"""
        if score >= 80:
            return "높음 - 대부분의 정보가 검증 가능한 소스에서 수집됨"
        elif score >= 50:
            return "중간 - 일부 정보는 추가 검증이 필요함"
        else:
            return "낮음 - 정보의 신뢰성을 확인할 수 없음"

    def _combine_sources(self, sources: List[Dict], max_length: int = 3000) -> str:
        """
        소스들을 하나의 텍스트로 결합

        Args:
            sources: 소스 리스트
            max_length: 최대 길이

        Returns:
            str: 결합된 텍스트
        """
        combined = []
        current_length = 0

        for source in sources:
            title = source.get("title", "")
            content = source.get("content", "")

            source_text = f"\n[{title}]\n{content}\n"
            source_length = len(source_text)

            if current_length + source_length > max_length:
                # 남은 공간만큼만 추가
                remaining = max_length - current_length
                if remaining > 100:
                    source_text = source_text[:remaining] + "..."
                    combined.append(source_text)
                break

            combined.append(source_text)
            current_length += source_length

        return "\n".join(combined)
