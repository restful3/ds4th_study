"""
Support Agent - 기술 지원
"""

from typing import Dict, Any
from .base import BaseAgent


class SupportAgent(BaseAgent):
    """기술 지원 Agent"""

    def __init__(self, name: str, llm, rag, verbose: bool = False):
        super().__init__(name, llm, verbose)
        self.rag = rag

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        기술 지원 제공

        Args:
            input_data: {"message": str, "context": list, "session_id": str}

        Returns:
            Dict: 응답 정보
        """
        message = input_data["message"]
        context = input_data.get("context", [])

        self.log("기술 지원 처리 중...")

        # RAG로 관련 문서 검색
        relevant_docs = self.rag.search(message, category="support")

        # 답변 생성
        answer = self._generate_answer(message, relevant_docs, context)

        return {
            "answer": answer,
            "sources": relevant_docs,
            "needs_escalation": self._check_escalation(answer),
        }

    def _generate_answer(self, message: str, docs: list, context: list) -> str:
        """답변 생성"""
        # 문서 내용 결합
        doc_content = "\n\n".join([doc.page_content for doc in docs[:2]])

        prompt = f"""당신은 친절한 기술 지원 담당자입니다.

참고 문서:
{doc_content}

고객 문의: {message}

위 정보를 바탕으로 친절하고 명확하게 답변하세요.
단계별로 설명하고, 필요하다면 추가 정보를 요청하세요.

답변:"""

        return self.invoke_llm(prompt)

    def _check_escalation(self, answer: str) -> bool:
        """에스컬레이션 필요 여부 확인"""
        escalation_keywords = ["확인이 어렵", "도움이 필요", "전문가"]
        return any(keyword in answer for keyword in escalation_keywords)
