"""
Planner Agent
연구 계획을 수립하는 Agent
"""

from typing import Dict, List, Any
from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """연구 계획 수립 Agent"""

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        연구 주제를 분석하고 계획을 수립

        Args:
            input_data: {"topic": str}

        Returns:
            Dict: 연구 계획 (주제, 하위 질문, 키워드)
        """
        topic = input_data["topic"]
        self.log("연구 계획 수립 중...", force=True)

        # 하위 질문 생성
        sub_questions = self._generate_sub_questions(topic)
        self.log(f"{len(sub_questions)}개 하위 질문 생성 완료")

        # 검색 키워드 추출
        keywords = self._extract_keywords(topic, sub_questions)
        self.log(f"{len(keywords)}개 검색 키워드 추출 완료")

        plan = {
            "topic": topic,
            "sub_questions": sub_questions,
            "keywords": keywords,
        }

        return plan

    def _generate_sub_questions(self, topic: str) -> List[str]:
        """
        주제에 대한 하위 질문 생성

        Args:
            topic: 연구 주제

        Returns:
            List[str]: 하위 질문 리스트
        """
        prompt = f"""주제: {topic}

이 주제를 깊이 있게 연구하기 위한 3-5개의 하위 질문을 생성하세요.

요구사항:
- 각 질문은 서로 다른 관점을 다뤄야 합니다
- 구체적이고 답변 가능한 질문이어야 합니다
- 논리적 순서로 배열하세요

질문을 다음 형식으로 작성하세요:
1. [첫 번째 질문]
2. [두 번째 질문]
3. [세 번째 질문]
...

하위 질문:"""

        response = self.invoke_llm(prompt)

        # 질문 파싱
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # 번호나 불릿 제거
                question = line.split('.', 1)[-1].strip()
                question = question.lstrip('-•').strip()
                if question:
                    questions.append(question)

        return questions[:5]  # 최대 5개

    def _extract_keywords(self, topic: str, sub_questions: List[str]) -> List[str]:
        """
        주제와 하위 질문에서 검색 키워드 추출

        Args:
            topic: 연구 주제
            sub_questions: 하위 질문 리스트

        Returns:
            List[str]: 검색 키워드 리스트
        """
        questions_text = '\n'.join([f"- {q}" for q in sub_questions])

        prompt = f"""주제: {topic}

하위 질문:
{questions_text}

이 연구를 위한 효과적인 검색 키워드를 5-10개 추출하세요.

요구사항:
- 구체적이고 검색하기 좋은 키워드
- 영어와 한국어 모두 포함 가능
- 너무 일반적이지 않은 키워드

키워드를 쉼표로 구분하여 나열하세요:"""

        response = self.invoke_llm(prompt)

        # 키워드 파싱
        keywords = []
        for keyword in response.replace('\n', ',').split(','):
            keyword = keyword.strip().strip('"\'')
            if keyword and len(keyword) > 1:
                keywords.append(keyword)

        return keywords[:10]  # 최대 10개
