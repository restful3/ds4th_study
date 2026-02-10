"""
Configuration Management
설정 관리
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI


class Config:
    """시스템 설정"""

    def __init__(self, verbose: bool = False):
        """
        초기화

        Args:
            verbose: 상세 로그 출력 여부
        """
        self.verbose = verbose

        # LLM 설정
        self.llm_config = {
            "router": {
                "model": "gpt-4o-mini",
                "temperature": 0,
            },
            "support": {
                "model": "gpt-4o-mini",
                "temperature": 0.3,
            },
            "billing": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
            "general": {
                "model": "gpt-4o-mini",
                "temperature": 0.5,
            },
            "escalation": {
                "model": "gpt-4o",
                "temperature": 0.3,
            },
        }

        # RAG 설정
        self.rag_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 3,
            "score_threshold": 0.7,
        }

        # HITL 설정
        self.hitl_config = {
            "enabled": True,
            "critical_actions": ["refund", "cancel_subscription", "delete_account"],
            "confidence_threshold": 0.7,
        }

        # 모니터링 설정
        self.monitoring_config = {
            "enabled": True,
            "log_level": "INFO",
            "metrics_enabled": True,
        }

    def get_llm(self, agent_type: str) -> ChatOpenAI:
        """
        Agent 타입에 맞는 LLM 반환

        Args:
            agent_type: Agent 타입

        Returns:
            ChatOpenAI: LLM 인스턴스
        """
        config = self.llm_config.get(agent_type, self.llm_config["general"])
        return ChatOpenAI(**config)

    def get_rag_config(self) -> Dict[str, Any]:
        """RAG 설정 반환"""
        return self.rag_config.copy()

    def get_hitl_config(self) -> Dict[str, Any]:
        """HITL 설정 반환"""
        return self.hitl_config.copy()

    def get_monitoring_config(self) -> Dict[str, Any]:
        """모니터링 설정 반환"""
        return self.monitoring_config.copy()


class ProductionConfig(Config):
    """프로덕션 설정"""

    def __init__(self):
        super().__init__(verbose=False)

        # 프로덕션용 최적화 설정
        self.llm_config["router"]["model"] = "gpt-4o-mini"
        self.llm_config["escalation"]["model"] = "gpt-4o"

        self.monitoring_config["log_level"] = "WARNING"
        self.monitoring_config["metrics_enabled"] = True


class DevelopmentConfig(Config):
    """개발 설정"""

    def __init__(self):
        super().__init__(verbose=True)

        # 개발용 빠른 모델
        for agent_type in self.llm_config:
            self.llm_config[agent_type]["model"] = "gpt-4o-mini"

        self.monitoring_config["log_level"] = "DEBUG"
