"""
Escalation Agent - 에스컬레이션 처리
"""

from typing import Dict, Any
from .base import BaseAgent


class EscalationAgent(BaseAgent):
    """에스컬레이션 Agent"""

    def __init__(self, name: str, llm, hitl, verbose: bool = False):
        super().__init__(name, llm, verbose)
        self.hitl = hitl

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        에스컬레이션 처리

        Args:
            input_data: {"message": str, "response": dict, "context": list}

        Returns:
            Dict: 에스컬레이션 결과
        """
        message = input_data["message"]
        response = input_data["response"]

        self.log("에스컬레이션 검토 중...")

        # HITL 승인 필요 여부 확인
        if self.hitl.requires_approval(message):
            approved = self.hitl.request_approval(message, response)

            if not approved:
                return {
                    "answer": "죄송합니다. 해당 요청은 승인되지 않았습니다.",
                    "escalated": True,
                    "approved": False,
                }

        # 사람 담당자에게 전달
        return {
            "answer": "담당자에게 전달하였습니다. 곧 연락드리겠습니다.",
            "escalated": True,
            "approved": True,
        }
