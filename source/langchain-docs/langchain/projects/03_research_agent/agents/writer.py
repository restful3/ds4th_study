"""
Writer Agent
최종 보고서를 작성하는 Agent
"""

from typing import Dict, Any
from .base import BaseAgent


class WriterAgent(BaseAgent):
    """보고서 작성 Agent"""

    def run(self, input_data: Dict[str, Any]) -> str:
        """
        최종 보고서 작성

        Args:
            input_data: {"topic": str, "plan": Dict, "analysis": Dict}

        Returns:
            str: 마크다운 형식 보고서
        """
        topic = input_data.get("topic")
        plan = input_data.get("plan", {})
        analysis = input_data.get("analysis", {})

        self.log("보고서 작성 중...", force=True)

        # 보고서 섹션 생성
        report_sections = []

        # 제목 및 메타데이터
        report_sections.append(self._generate_header(topic, analysis))

        # 요약
        report_sections.append(self._generate_summary(analysis))

        # 주요 발견 사항
        report_sections.append(self._generate_findings(plan, analysis))

        # 상세 분석
        report_sections.append(self._generate_detailed_analysis(topic, analysis))

        # 결론
        report_sections.append(self._generate_conclusion(topic, analysis))

        # 참고 문헌 (분석에서 sources가 있는 경우)
        if "source_count" in analysis and analysis["source_count"] > 0:
            report_sections.append(self._generate_references(analysis))

        # 전체 보고서 조합
        report = "\n\n".join(report_sections)

        self.log("보고서 작성 완료", force=True)

        return report

    def _generate_header(self, topic: str, analysis: Dict) -> str:
        """보고서 헤더 생성"""
        from datetime import datetime

        date_str = datetime.now().strftime("%Y년 %m월 %d일")
        source_count = analysis.get("source_count", 0)

        header = f"""# {topic}

**연구 보고서**

- 작성일: {date_str}
- 참조 소스: {source_count}개
- 작성자: AI 연구 에이전트 시스템

---"""
        return header

    def _generate_summary(self, analysis: Dict) -> str:
        """요약 섹션 생성"""
        summary = analysis.get("summary", "요약 정보가 없습니다.")

        return f"""## 요약

{summary}"""

    def _generate_findings(self, plan: Dict, analysis: Dict) -> str:
        """주요 발견 사항 섹션 생성"""
        insights = analysis.get("insights", [])

        if not insights:
            return "## 주요 발견 사항\n\n정보가 충분하지 않습니다."

        findings = ["## 주요 발견 사항\n"]

        for i, insight in enumerate(insights, 1):
            findings.append(f"{i}. {insight}")

        return "\n".join(findings)

    def _generate_detailed_analysis(self, topic: str, analysis: Dict) -> str:
        """상세 분석 섹션 생성"""
        insights = analysis.get("insights", [])
        summary = analysis.get("summary", "")

        prompt = f"""주제: {topic}

핵심 인사이트:
{chr(10).join([f'- {i}' for i in insights])}

요약: {summary}

위 정보를 바탕으로 3-4개의 상세 분석 섹션을 작성하세요.
각 섹션은 다음 형식을 따르세요:

### [섹션 제목]

[2-3문단의 상세 설명]

마크다운 형식으로 작성하되, 헤더는 ### (h3)를 사용하세요.

상세 분석:"""

        detailed_analysis = self.invoke_llm(prompt)

        return f"""## 상세 분석

{detailed_analysis}"""

    def _generate_conclusion(self, topic: str, analysis: Dict) -> str:
        """결론 섹션 생성"""
        insights = analysis.get("insights", [])
        credibility = analysis.get("credibility", {})

        prompt = f"""주제: {topic}

핵심 인사이트:
{chr(10).join([f'- {i}' for i in insights])}

위 연구 내용을 바탕으로 2-3문단의 결론을 작성하세요.

결론에는 다음 내용이 포함되어야 합니다:
- 주요 발견 사항의 의미
- 향후 전망이나 시사점
- 추가 연구가 필요한 영역

결론:"""

        conclusion = self.invoke_llm(prompt)

        # 신뢰도 정보 추가
        credibility_note = ""
        if credibility:
            score = credibility.get("credibility_score", 0)
            assessment = credibility.get("assessment", "")
            credibility_note = f"\n\n*신뢰도 평가: {score}% - {assessment}*"

        return f"""## 결론

{conclusion}{credibility_note}"""

    def _generate_references(self, analysis: Dict) -> str:
        """참고 문헌 섹션 생성"""
        source_count = analysis.get("source_count", 0)

        references = f"""## 참고 문헌

본 보고서는 {source_count}개의 온라인 소스를 참조하여 작성되었습니다.

*주: 이 보고서는 AI 에이전트가 자동으로 생성한 것으로,
정보의 정확성을 보장하지 않습니다. 중요한 결정을 내리기 전에
반드시 원본 소스를 확인하시기 바랍니다.*"""

        return references
