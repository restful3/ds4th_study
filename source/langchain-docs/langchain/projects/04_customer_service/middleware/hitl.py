"""
Human-in-the-Loop Middleware
사람 개입 미들웨어
"""

from typing import Dict, Any, List


class HumanInTheLoop:
    """Human-in-the-Loop 구현"""

    # 승인이 필요한 중요 작업
    CRITICAL_ACTIONS = [
        "환불", "refund",
        "취소", "cancel",
        "삭제", "delete",
        "해지", "terminate",
    ]

    def __init__(self, confidence_threshold: float = 0.7):
        """
        초기화

        Args:
            confidence_threshold: 자동 처리 신뢰도 임계값
        """
        self.confidence_threshold = confidence_threshold
        self.approval_log = []

    def requires_approval(self, message: str, confidence: float = 1.0) -> bool:
        """
        승인 필요 여부 확인

        Args:
            message: 사용자 메시지
            confidence: 응답 신뢰도

        Returns:
            bool: 승인 필요 여부
        """
        # 중요 작업 키워드 확인
        has_critical_action = any(
            action in message for action in self.CRITICAL_ACTIONS
        )

        # 신뢰도가 낮은 경우
        low_confidence = confidence < self.confidence_threshold

        return has_critical_action or low_confidence

    def request_approval(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        사람의 승인 요청

        Args:
            message: 메시지
            context: 컨텍스트 정보

        Returns:
            bool: 승인 여부
        """
        print("\n" + "=" * 60)
        print("⚠️  Human-in-the-Loop: 승인이 필요합니다")
        print("=" * 60)
        print(f"\n요청 내용: {message}")
        print(f"응답 예정: {context.get('answer', 'N/A')[:100]}...")

        response = input("\n✅ 승인하시겠습니까? (y/n): ").strip().lower()

        approved = response in ['y', 'yes', '예', 'ㅇ']

        # 로그 기록
        self.approval_log.append({
            "message": message,
            "context": context,
            "approved": approved,
        })

        return approved

    def get_approval_history(self) -> List[Dict]:
        """승인 기록 조회"""
        return self.approval_log.copy()
