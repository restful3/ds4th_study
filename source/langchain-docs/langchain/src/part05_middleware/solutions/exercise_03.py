"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: Middleware - 실습 과제 3 해답
================================================================================

과제: 종합 모니터링 시스템
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. 로깅, 성능, 에러를 통합 모니터링
2. 여러 Middleware 체이닝
3. 대시보드급 리포트 생성

학습 목표:
- 복수 Middleware 조합
- 종합 모니터링 시스템 구축
- 프로덕션급 observability

================================================================================
"""

from typing import Optional, Any
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import time
import traceback
import json
from collections import defaultdict

# ============================================================================
# 로깅 Middleware
# ============================================================================

class LoggingMiddleware:
    """로깅 미들웨어"""

    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.logs = []

    def log(self, level: str, message: str, **kwargs):
        """로그 기록"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

        # 콘솔 출력
        icon = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }.get(level, "📝")

        print(f"{icon} [{level}] {message}")

    def get_logs(self, level: Optional[str] = None) -> list:
        """로그 조회"""
        if level:
            return [log for log in self.logs if log["level"] == level]
        return self.logs

    def clear_logs(self):
        """로그 초기화"""
        self.logs.clear()


# ============================================================================
# 성능 추적 Middleware
# ============================================================================

class PerformanceMiddleware:
    """성능 추적 미들웨어"""

    def __init__(self):
        self.metrics = []
        self.current_metric = None

    def start_tracking(self, operation: str):
        """성능 추적 시작"""
        self.current_metric = {
            "operation": operation,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "timestamp": datetime.now().isoformat(),
        }

    def stop_tracking(self, **extra_data):
        """성능 추적 종료"""
        if not self.current_metric:
            return

        self.current_metric["end_time"] = time.time()
        self.current_metric["duration"] = (
            self.current_metric["end_time"] - self.current_metric["start_time"]
        )
        self.current_metric.update(extra_data)

        self.metrics.append(self.current_metric)
        print(f"⏱️  [{self.current_metric['operation']}] {self.current_metric['duration']:.3f}초")

        self.current_metric = None

    def get_stats(self) -> dict:
        """통계 계산"""
        if not self.metrics:
            return {}

        durations = [m["duration"] for m in self.metrics]

        return {
            "total_calls": len(self.metrics),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
        }


# ============================================================================
# 에러 추적 Middleware
# ============================================================================

class ErrorTrackingMiddleware:
    """에러 추적 미들웨어"""

    def __init__(self):
        self.errors = []
        self.error_counts = defaultdict(int)

    def track_error(self, error: Exception, context: dict = None):
        """에러 기록"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
        }

        self.errors.append(error_entry)
        self.error_counts[error_entry["error_type"]] += 1

        print(f"❌ [ERROR] {error_entry['error_type']}: {error_entry['error_message']}")

    def get_errors(self, error_type: Optional[str] = None) -> list:
        """에러 조회"""
        if error_type:
            return [e for e in self.errors if e["error_type"] == error_type]
        return self.errors

    def get_error_summary(self) -> dict:
        """에러 요약"""
        return {
            "total_errors": len(self.errors),
            "error_types": dict(self.error_counts),
            "recent_errors": self.errors[-5:] if self.errors else []
        }


# ============================================================================
# 통합 모니터링 시스템
# ============================================================================

class MonitoringSystem:
    """통합 모니터링 시스템"""

    def __init__(self):
        self.logging = LoggingMiddleware()
        self.performance = PerformanceMiddleware()
        self.error_tracking = ErrorTrackingMiddleware()

        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.start_time = datetime.now()

    def before_request(self, state: MessagesState):
        """요청 전 처리"""
        self.request_count += 1

        user_msg = state["messages"][-1].content if state["messages"] else ""

        self.logging.log(
            "INFO",
            f"Request #{self.request_count} started",
            user_message=user_msg[:50]
        )

        self.performance.start_tracking(f"request_{self.request_count}")

    def after_request(self, result: dict, success: bool = True):
        """요청 후 처리"""
        if success:
            self.success_count += 1
            self.logging.log("INFO", f"Request #{self.request_count} succeeded")
        else:
            self.failure_count += 1
            self.logging.log("WARNING", f"Request #{self.request_count} failed")

        ai_msg = result["messages"][-1].content if result.get("messages") else ""

        self.performance.stop_tracking(
            success=success,
            response_length=len(ai_msg)
        )

    def on_error(self, error: Exception, context: dict = None):
        """에러 발생 시"""
        self.error_tracking.track_error(error, context)
        self.logging.log(
            "ERROR",
            f"Error in request #{self.request_count}",
            error=str(error)
        )

    def get_health_status(self) -> dict:
        """시스템 상태"""
        success_rate = (
            self.success_count / self.request_count
            if self.request_count > 0
            else 0
        )

        uptime = datetime.now() - self.start_time

        perf_stats = self.performance.get_stats()

        return {
            "status": "healthy" if success_rate > 0.95 else "degraded",
            "uptime_seconds": uptime.total_seconds(),
            "total_requests": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "avg_response_time": perf_stats.get("avg_duration", 0),
            "error_count": len(self.error_tracking.errors),
        }

    def print_dashboard(self):
        """대시보드 출력"""
        print("\n" + "=" * 70)
        print("📊 모니터링 대시보드")
        print("=" * 70)

        # 1. 시스템 상태
        health = self.get_health_status()
        status_icon = "✅" if health["status"] == "healthy" else "⚠️"

        print(f"\n{status_icon} 시스템 상태: {health['status'].upper()}")
        print(f"⏱️  가동 시간: {health['uptime_seconds']:.0f}초")

        # 2. 요청 통계
        print(f"\n📈 요청 통계:")
        print(f"  총 요청: {health['total_requests']}")
        print(f"  성공: {health['success_count']} ({health['success_rate']:.1%})")
        print(f"  실패: {health['failure_count']}")

        # 3. 성능 메트릭
        perf_stats = self.performance.get_stats()
        if perf_stats:
            print(f"\n⚡ 성능 메트릭:")
            print(f"  평균 응답 시간: {perf_stats['avg_duration']:.3f}초")
            print(f"  최소 응답 시간: {perf_stats['min_duration']:.3f}초")
            print(f"  최대 응답 시간: {perf_stats['max_duration']:.3f}초")

        # 4. 에러 요약
        error_summary = self.error_tracking.get_error_summary()
        print(f"\n❌ 에러 요약:")
        print(f"  총 에러: {error_summary['total_errors']}")

        if error_summary['error_types']:
            print(f"  에러 타입별:")
            for error_type, count in error_summary['error_types'].items():
                print(f"    - {error_type}: {count}")

        # 5. 최근 로그
        recent_logs = self.logging.get_logs()[-5:]
        if recent_logs:
            print(f"\n📜 최근 로그 (최대 5개):")
            for log in recent_logs:
                time_str = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
                print(f"  [{time_str}] {log['level']}: {log['message']}")

        print("\n" + "=" * 70)

    def export_report(self, filepath: str = None) -> dict:
        """리포트 내보내기"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "health": self.get_health_status(),
            "performance": self.performance.get_stats(),
            "errors": self.error_tracking.get_error_summary(),
            "logs": self.logging.get_logs(),
            "metrics": self.performance.metrics,
        }

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 리포트 저장됨: {filepath}")

        return report


# ============================================================================
# 모니터링 통합 챗봇 구축
# ============================================================================

def create_monitored_chatbot(model_name: str = "gpt-4o-mini"):
    """모니터링 통합 챗봇 생성"""

    model = ChatOpenAI(model=model_name, temperature=0.7)
    monitoring = MonitoringSystem()

    # 챗봇 노드 (모니터링 통합)
    def chatbot_node(state: MessagesState, config: RunnableConfig) -> dict:
        """모니터링되는 챗봇 노드"""

        # Before request
        monitoring.before_request(state)

        try:
            # LLM 호출
            response = model.invoke(state["messages"])
            result = {"messages": [response]}

            # After request (success)
            monitoring.after_request(result, success=True)

            return result

        except Exception as e:
            # Error handling
            monitoring.on_error(e, {"messages": len(state["messages"])})
            monitoring.after_request({}, success=False)

            # 에러 메시지 반환
            error_msg = AIMessage(
                content=f"죄송합니다. 오류가 발생했습니다: {str(e)}"
            )
            return {"messages": [error_msg]}

    # 그래프 구축
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph, monitoring


# ============================================================================
# 테스트 함수
# ============================================================================

def test_monitoring_system():
    """모니터링 시스템 테스트"""
    print("=" * 70)
    print("📊 종합 모니터링 시스템 테스트")
    print("=" * 70)

    chatbot, monitoring = create_monitored_chatbot()
    config = {"configurable": {"thread_id": "test_monitoring"}}

    # 정상 요청들
    normal_questions = [
        "안녕하세요!",
        "파이썬이란 무엇인가요?",
        "머신러닝의 기초를 설명해주세요.",
    ]

    print("\n✅ 정상 요청 테스트...")
    for question in normal_questions:
        print(f"\n{'=' * 70}")
        print(f"👤 {question}")
        print("=" * 70)

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

        print(f"🤖 {result['messages'][-1].content[:150]}...")

    # 대시보드 출력
    monitoring.print_dashboard()


def test_error_handling():
    """에러 처리 테스트"""
    print("\n" + "=" * 70)
    print("❌ 에러 처리 테스트")
    print("=" * 70)

    # 잘못된 API 키로 챗봇 생성 (에러 유발)
    import os
    original_key = os.environ.get("OPENAI_API_KEY")

    try:
        # 일시적으로 잘못된 키 설정
        os.environ["OPENAI_API_KEY"] = "invalid_key"

        chatbot, monitoring = create_monitored_chatbot()
        config = {"configurable": {"thread_id": "test_error"}}

        print("\n🧪 잘못된 API 키로 요청...")
        result = chatbot.invoke(
            {"messages": [HumanMessage(content="테스트")]},
            config
        )

        print(f"🤖 {result['messages'][-1].content}")

    finally:
        # 원래 키 복원
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key

    # 에러 요약
    error_summary = monitoring.error_tracking.get_error_summary()
    print(f"\n📊 에러 요약:")
    print(f"  총 에러: {error_summary['total_errors']}")
    print(f"  에러 타입: {list(error_summary['error_types'].keys())}")


def test_performance_tracking():
    """성능 추적 테스트"""
    print("\n" + "=" * 70)
    print("⚡ 성능 추적 테스트")
    print("=" * 70)

    chatbot, monitoring = create_monitored_chatbot()
    config = {"configurable": {"thread_id": "test_performance"}}

    # 다양한 길이의 요청
    questions = [
        "안녕",  # 짧은 질문
        "인공지능의 역사를 간단히 설명해주세요.",  # 중간 질문
        "머신러닝과 딥러닝의 차이점, 각각의 응용 사례, 그리고 미래 전망에 대해 상세히 설명해주세요.",  # 긴 질문
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n📝 요청 {i}: {question}")

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

    # 성능 통계
    perf_stats = monitoring.performance.get_stats()
    print(f"\n⚡ 성능 통계:")
    print(f"  총 호출: {perf_stats['total_calls']}")
    print(f"  평균: {perf_stats['avg_duration']:.3f}초")
    print(f"  최소: {perf_stats['min_duration']:.3f}초")
    print(f"  최대: {perf_stats['max_duration']:.3f}초")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 모니터링 챗봇 - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /dashboard - 대시보드 보기")
    print("  /health - 시스템 상태")
    print("  /logs [level] - 로그 조회")
    print("  /errors - 에러 목록")
    print("  /export <file> - 리포트 내보내기")
    print("  /quit - 종료")
    print("=" * 70)

    chatbot, monitoring = create_monitored_chatbot()
    config = {"configurable": {"thread_id": "interactive"}}

    while True:
        try:
            user_input = input("\n👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                monitoring.print_dashboard()
                print("\n👋 종료합니다.")
                break

            elif user_input == "/dashboard":
                monitoring.print_dashboard()
                continue

            elif user_input == "/health":
                health = monitoring.get_health_status()
                print(f"\n💚 시스템 상태:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
                continue

            elif user_input.startswith("/logs"):
                parts = user_input.split()
                level = parts[1] if len(parts) > 1 else None
                logs = monitoring.logging.get_logs(level)
                print(f"\n📜 로그 ({len(logs)}개):")
                for log in logs[-10:]:
                    print(f"  [{log['timestamp']}] {log['level']}: {log['message']}")
                continue

            elif user_input == "/errors":
                errors = monitoring.error_tracking.get_errors()
                print(f"\n❌ 에러 목록 ({len(errors)}개):")
                for error in errors:
                    print(f"  [{error['timestamp']}] {error['error_type']}: {error['error_message']}")
                continue

            elif user_input.startswith("/export"):
                parts = user_input.split()
                filepath = parts[1] if len(parts) > 1 else "monitoring_report.json"
                monitoring.export_report(filepath)
                continue

            # 일반 메시지
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            ai_response = result["messages"][-1].content
            print(f"\n🤖 {ai_response}")

        except KeyboardInterrupt:
            monitoring.print_dashboard()
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📊 Part 5: 종합 모니터링 시스템 - 실습 과제 3 해답")
    print("=" * 70)

    try:
        # 테스트 1: 기본 모니터링
        test_monitoring_system()

        # 테스트 2: 성능 추적
        test_performance_tracking()

        # 테스트 3: 에러 처리 (선택)
        print("\n에러 처리 테스트를 실행하시겠습니까? (y/n): ", end="")
        if input().strip().lower() in ['y', 'yes', '예']:
            test_error_handling()

        # 대화형 모드
        print("\n대화형 모드를 실행하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예']:
            interactive_mode()

    except Exception as e:
        print(f"\n⚠️  오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # 학습 포인트
    print("\n" + "=" * 70)
    print("💡 학습 포인트:")
    print("  1. 복수 Middleware 통합 (로깅, 성능, 에러)")
    print("  2. 종합 대시보드 구현")
    print("  3. 시스템 health check")
    print("  4. 리포트 내보내기")
    print("\n💡 추가 학습:")
    print("  1. Prometheus/Grafana 통합")
    print("  2. 실시간 알림 (Slack, Email)")
    print("  3. 분산 추적 (OpenTelemetry)")
    print("  4. 로그 집계 (ELK Stack)")
    print("=" * 70)


if __name__ == "__main__":
    main()
