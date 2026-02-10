"""
General Agent - 일반 문의
"""

from typing import Dict, Any
from .base import BaseAgent


class GeneralAgent(BaseAgent):
    """일반 문의 Agent"""

    def __init__(self, name: str, llm, rag, verbose: bool = False):
        super().__init__(name, llm, verbose)
        self.rag = rag

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        일반 문의 처리

        Args:
            input_data: {"message": str, "context": list, "session_id": str}

        Returns:
            Dict: 응답 정보
        """
        message = input_data["message"]

        self.log("일반 문의 처리 중...")

        # RAG로 관련 정보 검색
        relevant_docs = self.rag.search(message)

        # 답변 생성
        answer = self._generate_answer(message, relevant_docs)

        return {
            "answer": answer,
            "sources": relevant_docs,
            "needs_escalation": False,
        }

    def _generate_answer(self, message: str, docs: list) -> str:
        """답변 생성"""
        doc_content = "\n\n".join([doc.page_content for doc in docs[:2]])

        prompt = f"""당신은 친절한 고객 서비스 담당자입니다.

참고 정보:
{doc_content}

고객 문의: {message}

친절하고 도움이 되는 답변을 제공하세요.

답변:"""

        return self.invoke_llm(prompt)
