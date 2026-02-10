"""
Router Agent - 문의를 분류하고 라우팅
"""

from typing import Dict, Any
from .base import BaseAgent


class RouterAgent(BaseAgent):
    """문의를 적절한 Agent로 라우팅하는 Agent"""

    # 카테고리별 키워드
    CATEGORIES = {
        "support": [
            "오류", "에러", "버그", "작동", "실행", "설치",
            "문제", "고장", "안돼", "안 돼", "기술", "설정"
        ],
        "billing": [
            "결제", "요금", "비용", "환불", "구독", "취소",
            "카드", "계좌", "청구", "금액", "가격", "할인"
        ],
        "general": [
            "문의", "정보", "질문", "알려", "설명", "어떻게",
            "무엇", "언제", "어디", "왜", "사용법", "가이드"
        ],
    }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """실행하지 않고 route 메서드 사용"""
        raise NotImplementedError("RouterAgent는 route() 메서드를 사용하세요")

    def route(self, message: str) -> Dict[str, str]:
        """
        메시지를 분류하고 라우팅

        Args:
            message: 고객 메시지

        Returns:
            Dict: 라우팅 정보 (category, confidence, agent)
        """
        self.log("문의 분석 중...", force=True)

        # 1. 키워드 기반 빠른 분류
        keyword_category = self._classify_by_keywords(message)

        # 2. LLM 기반 정확한 분류
        llm_category = self._classify_by_llm(message)

        # 3. 결합
        category = llm_category if llm_category else keyword_category
        confidence = self._calculate_confidence(message, category)

        self.log(f"➜ {category.title()} Agent로 전달", force=True)

        return {
            "category": category,
            "confidence": confidence,
        }

    def _classify_by_keywords(self, message: str) -> str:
        """키워드 기반 분류"""
        message_lower = message.lower()
        scores = {category: 0 for category in self.CATEGORIES}

        for category, keywords in self.CATEGORIES.items():
            for keyword in keywords:
                if keyword in message_lower:
                    scores[category] += 1

        # 점수가 가장 높은 카테고리 반환
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)

        return "general"

    def _classify_by_llm(self, message: str) -> str:
        """LLM 기반 정확한 분류"""
        prompt = f"""다음 고객 문의를 분류하세요.

고객 문의: "{message}"

카테고리:
- support: 기술 지원, 오류, 문제 해결
- billing: 결제, 요금, 환불 관련
- general: 일반 문의, 정보 요청

가장 적절한 카테고리 하나만 답변하세요 (support, billing, general 중 하나).

카테고리:"""

        response = self.invoke_llm(prompt).strip().lower()

        # 응답에서 카테고리 추출
        for category in self.CATEGORIES.keys():
            if category in response:
                return category

        return "general"

    def _calculate_confidence(self, message: str, category: str) -> float:
        """분류 신뢰도 계산"""
        message_lower = message.lower()
        keywords = self.CATEGORIES[category]

        # 키워드 매칭 비율
        matches = sum(1 for keyword in keywords if keyword in message_lower)
        keyword_ratio = matches / len(keywords) if keywords else 0

        # 기본 신뢰도 (0.5) + 키워드 보너스
        confidence = 0.5 + (keyword_ratio * 0.5)

        return min(confidence, 1.0)
