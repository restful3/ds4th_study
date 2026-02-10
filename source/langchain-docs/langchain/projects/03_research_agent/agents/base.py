"""
Base Agent Class
Agent 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    """모든 Agent의 베이스 클래스"""

    def __init__(self, name: str, llm, verbose: bool = False):
        """
        초기화

        Args:
            name: Agent 이름
            llm: 사용할 LLM 인스턴스
            verbose: 상세 로그 출력 여부
        """
        self.name = name
        self.llm = llm
        self.verbose = verbose

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 실행 메서드 (서브클래스에서 구현 필요)

        Args:
            input_data: 입력 데이터

        Returns:
            Dict[str, Any]: 출력 데이터
        """
        pass

    def log(self, message: str, force: bool = False):
        """
        로그 출력

        Args:
            message: 로그 메시지
            force: verbose 설정과 무관하게 강제 출력
        """
        if self.verbose or force:
            print(f"   [{self.name}] {message}")

    def format_prompt(self, template: str, **kwargs) -> str:
        """
        프롬프트 템플릿 포맷팅

        Args:
            template: 프롬프트 템플릿
            **kwargs: 템플릿 변수

        Returns:
            str: 포맷된 프롬프트
        """
        return template.format(**kwargs)

    def invoke_llm(self, prompt: str) -> str:
        """
        LLM 호출

        Args:
            prompt: 프롬프트

        Returns:
            str: LLM 응답
        """
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
