"""
Base Agent Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """모든 고객 서비스 Agent의 베이스 클래스"""

    def __init__(self, name: str, llm, verbose: bool = False):
        """
        초기화

        Args:
            name: Agent 이름
            llm: LLM 인스턴스
            verbose: 상세 로그 출력 여부
        """
        self.name = name
        self.llm = llm
        self.verbose = verbose

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 실행 (서브클래스에서 구현)

        Args:
            input_data: 입력 데이터

        Returns:
            Dict[str, Any]: 출력 데이터
        """
        pass

    def log(self, message: str, force: bool = False):
        """로그 출력"""
        if self.verbose or force:
            print(f"[{self.name}] {message}")

    def invoke_llm(self, prompt: str) -> str:
        """LLM 호출"""
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
