"""
Monitoring Middleware
모니터링 미들웨어
"""

import time
from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime


class Monitor:
    """시스템 모니터링"""

    def __init__(self):
        """초기화"""
        self.sessions: Dict[str, Dict] = {}
        self.metrics: Dict[str, List] = defaultdict(list)
        self.errors: List[Dict] = []

    def track_request(self, session_id: str, message: str):
        """
        요청 추적 시작

        Args:
            session_id: 세션 ID
            message: 메시지
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "start_time": time.time(),
                "messages": [],
                "agent_switches": 0,
                "errors": 0,
            }

        self.sessions[session_id]["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
        })

    def track_response(
        self,
        session_id: str,
        response: Dict,
        category: str,
        confidence: float
    ):
        """
        응답 추적

        Args:
            session_id: 세션 ID
            response: 응답 정보
            category: 카테고리
            confidence: 신뢰도
        """
        if session_id in self.sessions:
            elapsed = time.time() - self.sessions[session_id]["start_time"]

            # 메트릭 기록
            self.metrics["response_time"].append(elapsed)
            self.metrics["confidence"].append(confidence)
            self.metrics["category"].append(category)

    def track_error(self, session_id: str, error: str):
        """
        에러 추적

        Args:
            session_id: 세션 ID
            error: 에러 메시지
        """
        self.errors.append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "error": error,
        })

        if session_id in self.sessions:
            self.sessions[session_id]["errors"] += 1

    def track_satisfaction(self, session_id: str, rating: int):
        """
        만족도 추적

        Args:
            session_id: 세션 ID
            rating: 평점 (1-5)
        """
        self.metrics["satisfaction"].append(rating)

        if session_id in self.sessions:
            self.sessions[session_id]["satisfaction"] = rating

    def get_stats(self) -> Dict[str, Any]:
        """
        통계 조회

        Returns:
            Dict: 통계 정보
        """
        response_times = self.metrics.get("response_time", [])
        satisfactions = self.metrics.get("satisfaction", [])
        confidences = self.metrics.get("confidence", [])

        return {
            "total_sessions": len(self.sessions),
            "total_errors": len(self.errors),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "avg_satisfaction": sum(satisfactions) / len(satisfactions) if satisfactions else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        }

    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("📊 세션 통계")
        print("=" * 60)
        print(f"총 세션: {stats['total_sessions']}")
        print(f"평균 응답 시간: {stats['avg_response_time']:.2f}초")
        print(f"평균 만족도: {stats['avg_satisfaction']:.1f}/5")
        print(f"평균 신뢰도: {stats['avg_confidence']:.1%}")
        print(f"오류 수: {stats['total_errors']}")
        print("=" * 60)
