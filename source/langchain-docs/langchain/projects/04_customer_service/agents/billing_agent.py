"""
Billing Agent - 결제 관련
"""

from typing import Dict, Any
from .base import BaseAgent


class BillingAgent(BaseAgent):
    """결제 관련 Agent"""

    def __init__(self, name: str, llm, rag, verbose: bool = False):
        super().__init__(name, llm, verbose)
        self.rag = rag

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        결제 관련 문의 처리

        Args:
            input_data: {"message": str, "context": list, "session_id": str}

        Returns:
            Dict: 응답 정보
        """
        message = input_data["message"]

        self.log("결제 관련 문의 처리 중...")

        # RAG로 관련 정책 검색
        relevant_docs = self.rag.search(message, category="billing")

        # 답변 생성
        answer = self._generate_answer(message, relevant_docs)

        return {
            "answer": answer,
            "sources": relevant_docs,
            "needs_escalation": self._check_critical_action(message),
        }

    def _generate_answer(self, message: str, docs: list) -> str:
        """답변 생성"""
        doc_content = "\n\n".join([doc.page_content for doc in docs[:2]])

        prompt = f"""당신은 친절한 결제 담당자입니다.

결제 정책:
{doc_content}

고객 문의: {message}

위 정책을 바탕으로 명확하게 답변하세요.
결제 관련 정보는 정확해야 하므로 불확실하면 확인 후 답변하겠다고 하세요.

답변:"""

        return self.invoke_llm(prompt)

    def _check_critical_action(self, message: str) -> bool:
        """중요 작업 여부 확인"""
        critical_keywords = ["환불", "취소", "삭제", "해지"]
        return any(keyword in message for keyword in critical_keywords)
