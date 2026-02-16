"""
[Chapter 20] 배포 준비 (Deployment Ready)

📝 설명:
    LangGraph 애플리케이션을 프로덕션 환경에 배포하기 위한
    최종 점검 사항과 모범 사례를 정리합니다.

🎯 학습 목표:
    - 프로덕션 체크리스트 확인
    - 환경 구성 관리
    - 로깅 및 모니터링
    - 성능 최적화
    - 보안 고려사항

📚 관련 문서:
    - docs/Part5-Advanced/20-deployment-ready.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/deployment/

💻 실행 방법:
    python -m src.part5_advanced.20_deployment_ready

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
import logging
import time
from typing import TypedDict, Annotated, List, Optional
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator


# =============================================================================
# 1. 프로덕션 체크리스트
# =============================================================================

def explain_production_checklist():
    """프로덕션 체크리스트 설명"""
    print("\n" + "=" * 60)
    print("📘 프로덕션 배포 체크리스트")
    print("=" * 60)

    print("""
배포 전 체크리스트:

□ 환경 구성
  ├─ [ ] 환경 변수 설정 (.env 또는 환경 변수)
  ├─ [ ] API 키 보안 저장 (Secret Manager)
  ├─ [ ] 환경별 설정 분리 (dev/staging/prod)
  └─ [ ] 설정 검증 로직

□ 체크포인터
  ├─ [ ] 영구 저장소 선택 (PostgreSQL, Redis 등)
  ├─ [ ] 연결 풀 설정
  ├─ [ ] 백업 전략
  └─ [ ] 데이터 보존 정책

□ 에러 처리
  ├─ [ ] 예외 처리 완비
  ├─ [ ] 재시도 로직
  ├─ [ ] 폴백 전략
  └─ [ ] 에러 알림

□ 로깅
  ├─ [ ] 구조화된 로깅
  ├─ [ ] 로그 레벨 설정
  ├─ [ ] 로그 수집/집계
  └─ [ ] 민감 정보 마스킹

□ 모니터링
  ├─ [ ] 메트릭 수집
  ├─ [ ] 대시보드 구성
  ├─ [ ] 알림 설정
  └─ [ ] 헬스 체크

□ 성능
  ├─ [ ] 응답 시간 목표
  ├─ [ ] 동시성 제한
  ├─ [ ] 캐싱 전략
  └─ [ ] 리소스 제한

□ 보안
  ├─ [ ] 인증/인가
  ├─ [ ] 입력 검증
  ├─ [ ] Rate Limiting
  └─ [ ] 감사 로그
""")


# =============================================================================
# 2. 환경 구성 관리
# =============================================================================

class Config:
    """환경 설정 관리 클래스"""

    def __init__(self):
        load_dotenv()

        # 환경 식별
        self.env = os.getenv("ENV", "development")

        # API 키
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # 데이터베이스
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///langgraph.db")

        # 성능 설정
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))

        # 로깅
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> List[str]:
        """설정 검증"""
        errors = []

        if self.env == "production":
            if not self.anthropic_api_key:
                errors.append("ANTHROPIC_API_KEY가 설정되지 않았습니다")
            if "sqlite" in self.db_url:
                errors.append("프로덕션에서는 SQLite를 사용하지 마세요")

        return errors

    def __repr__(self):
        return f"Config(env={self.env}, db={self.db_url[:30]}...)"


def run_config_example():
    """환경 구성 예제"""
    print("\n" + "=" * 60)
    print("예제 1: 환경 구성 관리")
    print("=" * 60)

    config = Config()

    print(f"\n🔧 현재 설정:")
    print(f"   환경: {config.env}")
    print(f"   DB URL: {config.db_url[:50]}...")
    print(f"   최대 동시 요청: {config.max_concurrent_requests}")
    print(f"   요청 타임아웃: {config.request_timeout}초")
    print(f"   로그 레벨: {config.log_level}")

    # 설정 검증
    errors = config.validate()
    if errors:
        print(f"\n⚠️  설정 검증 오류:")
        for error in errors:
            print(f"   - {error}")
    else:
        print(f"\n✅ 설정 검증 통과")


# =============================================================================
# 3. 로깅 설정
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """구조화된 로깅 설정"""

    # 로거 생성
    logger = logging.getLogger("langgraph_app")
    logger.setLevel(getattr(logging, level))

    # 핸들러 설정
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level))

    # 포맷터 설정 (구조화된 형식)
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # 핸들러 추가
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


class LoggingState(TypedDict):
    """로깅 예제 State"""
    input_data: str
    result: str
    start_time: float
    duration: float


def create_logged_graph(logger: logging.Logger):
    """로깅이 포함된 그래프"""

    def start_processing(state: LoggingState) -> LoggingState:
        """처리 시작"""
        logger.info(f"Processing started: input='{state['input_data'][:50]}...'")
        return {"start_time": time.time()}

    def process_data(state: LoggingState) -> LoggingState:
        """데이터 처리"""
        logger.debug("Processing data...")
        time.sleep(0.1)  # 시뮬레이션
        result = state["input_data"].upper()
        logger.debug(f"Processing complete: result_length={len(result)}")
        return {"result": result}

    def finish_processing(state: LoggingState) -> LoggingState:
        """처리 완료"""
        duration = time.time() - state.get("start_time", time.time())
        logger.info(f"Processing finished: duration={duration:.3f}s")
        return {"duration": duration}

    graph = StateGraph(LoggingState)
    graph.add_node("start", start_processing)
    graph.add_node("process", process_data)
    graph.add_node("finish", finish_processing)

    graph.add_edge(START, "start")
    graph.add_edge("start", "process")
    graph.add_edge("process", "finish")
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=MemorySaver())


def run_logging_example():
    """로깅 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 구조화된 로깅")
    print("=" * 60)

    logger = setup_logging("DEBUG")

    app = create_logged_graph(logger)
    config = {"configurable": {"thread_id": "log_1"}}

    print("\n📋 로그 출력:")
    result = app.invoke({
        "input_data": "테스트 데이터입니다",
        "result": "",
        "start_time": 0,
        "duration": 0
    }, config=config)

    print(f"\n⏱️  처리 시간: {result['duration']:.3f}초")


# =============================================================================
# 4. 메트릭 수집
# =============================================================================

class Metrics:
    """간단한 메트릭 수집기"""

    def __init__(self):
        self.counters = {}
        self.timings = {}

    def increment(self, name: str, value: int = 1):
        """카운터 증가"""
        self.counters[name] = self.counters.get(name, 0) + value

    def record_timing(self, name: str, duration: float):
        """타이밍 기록"""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

    def get_stats(self, name: str) -> dict:
        """통계 반환"""
        timings = self.timings.get(name, [])
        if not timings:
            return {"count": 0}

        return {
            "count": len(timings),
            "avg": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings)
        }

    def report(self):
        """리포트 출력"""
        print("\n📊 메트릭 리포트:")
        print("\n   카운터:")
        for name, value in self.counters.items():
            print(f"      {name}: {value}")

        print("\n   타이밍:")
        for name in self.timings:
            stats = self.get_stats(name)
            print(f"      {name}:")
            print(f"         count: {stats['count']}")
            print(f"         avg: {stats['avg']:.3f}s")
            print(f"         min: {stats['min']:.3f}s")
            print(f"         max: {stats['max']:.3f}s")


def with_metrics(metrics: Metrics, name: str):
    """메트릭 수집 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics.increment(f"{name}_calls")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                metrics.increment(f"{name}_success")
                return result
            except Exception as e:
                metrics.increment(f"{name}_errors")
                raise
            finally:
                metrics.record_timing(f"{name}_duration", time.time() - start)
        return wrapper
    return decorator


def run_metrics_example():
    """메트릭 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 메트릭 수집")
    print("=" * 60)

    metrics = Metrics()

    class MetricState(TypedDict):
        value: int
        result: int

    def create_metric_graph():
        @with_metrics(metrics, "process")
        def process_node(state: MetricState) -> MetricState:
            time.sleep(0.05 + state["value"] * 0.01)
            return {"result": state["value"] * 2}

        graph = StateGraph(MetricState)
        graph.add_node("process", process_node)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        return graph.compile(checkpointer=MemorySaver())

    app = create_metric_graph()

    # 여러 번 실행
    print("\n🔄 5회 실행 중...")
    for i in range(5):
        config = {"configurable": {"thread_id": f"metric_{i}"}}
        app.invoke({"value": i, "result": 0}, config=config)

    # 메트릭 리포트
    metrics.report()


# =============================================================================
# 5. 에러 처리 및 복구
# =============================================================================

class RobustState(TypedDict):
    """견고한 State"""
    input: str
    retries: int
    max_retries: int
    result: Optional[str]
    error: Optional[str]
    recovered: bool


def create_robust_graph():
    """견고한 에러 처리 그래프"""

    def process_with_recovery(state: RobustState) -> RobustState:
        """복구 로직이 포함된 처리"""
        retries = state.get("retries", 0)

        try:
            # 처음 2번은 실패 시뮬레이션
            if retries < 2:
                raise ValueError(f"임시 오류 (시도 {retries + 1})")

            # 성공
            return {
                "result": f"처리됨: {state['input'].upper()}",
                "error": None,
                "recovered": retries > 0
            }

        except Exception as e:
            return {
                "retries": retries + 1,
                "error": str(e)
            }

    def should_retry(state: RobustState) -> str:
        """재시도 여부"""
        if state.get("result"):
            return "success"

        if state.get("retries", 0) >= state.get("max_retries", 3):
            return "failed"

        return "retry"

    def handle_success(state: RobustState) -> RobustState:
        """성공 처리"""
        return {}

    def handle_failure(state: RobustState) -> RobustState:
        """실패 처리"""
        return {"result": f"처리 실패: {state.get('error', 'Unknown')}"}

    graph = StateGraph(RobustState)
    graph.add_node("process", process_with_recovery)
    graph.add_node("success", handle_success)
    graph.add_node("failed", handle_failure)

    graph.add_edge(START, "process")
    graph.add_conditional_edges(
        "process",
        should_retry,
        {"retry": "process", "success": "success", "failed": "failed"}
    )
    graph.add_edge("success", END)
    graph.add_edge("failed", END)

    return graph.compile(checkpointer=MemorySaver())


def run_robust_example():
    """견고한 에러 처리 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 견고한 에러 처리")
    print("=" * 60)

    app = create_robust_graph()
    config = {"configurable": {"thread_id": "robust_1"}}

    print("\n🛡️  에러 복구 실행:")
    result = app.invoke({
        "input": "테스트 데이터",
        "retries": 0,
        "max_retries": 5,
        "result": None,
        "error": None,
        "recovered": False
    }, config=config)

    print(f"   재시도 횟수: {result['retries']}")
    print(f"   복구됨: {result['recovered']}")
    print(f"   결과: {result['result']}")


# =============================================================================
# 6. 헬스 체크
# =============================================================================

def health_check() -> dict:
    """헬스 체크"""
    checks = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }

    # 환경 변수 체크
    try:
        config = Config()
        errors = config.validate()
        checks["checks"]["config"] = {
            "status": "pass" if not errors else "fail",
            "errors": errors
        }
    except Exception as e:
        checks["checks"]["config"] = {"status": "fail", "error": str(e)}

    # 체크포인터 체크
    try:
        checkpointer = MemorySaver()
        checks["checks"]["checkpointer"] = {"status": "pass"}
    except Exception as e:
        checks["checks"]["checkpointer"] = {"status": "fail", "error": str(e)}

    # 전체 상태 결정
    all_pass = all(
        c.get("status") == "pass"
        for c in checks["checks"].values()
    )
    checks["status"] = "healthy" if all_pass else "unhealthy"

    return checks


def run_health_check_example():
    """헬스 체크 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 헬스 체크")
    print("=" * 60)

    result = health_check()

    print(f"\n🏥 헬스 체크 결과:")
    print(f"   상태: {result['status']}")
    print(f"   시간: {result['timestamp']}")
    print(f"\n   상세 체크:")
    for name, check in result["checks"].items():
        status_emoji = "✅" if check["status"] == "pass" else "❌"
        print(f"      {status_emoji} {name}: {check['status']}")
        if check.get("errors"):
            for error in check["errors"]:
                print(f"         - {error}")


# =============================================================================
# 7. 보안 고려사항
# =============================================================================

def explain_security():
    """보안 고려사항 설명"""
    print("\n" + "=" * 60)
    print("📘 보안 고려사항")
    print("=" * 60)

    print("""
보안 체크리스트:

1. API 키 관리
   - 환경 변수 또는 Secret Manager 사용
   - 코드에 API 키 하드코딩 금지
   - 키 로테이션 계획

2. 입력 검증
   - 모든 사용자 입력 검증
   - 길이 제한
   - 타입 검증
   - 인젝션 방지

3. 인증/인가
   - API 키 또는 토큰 인증
   - 역할 기반 접근 제어
   - 리소스별 권한 확인

4. Rate Limiting
   - IP 기반 제한
   - 사용자 기반 제한
   - 토큰 버킷 알고리즘

5. 데이터 보호
   - 민감 정보 암호화
   - PII 마스킹
   - 로그에서 민감 정보 제외

6. 감사 로그
   - 모든 작업 기록
   - 누가, 언제, 무엇을
   - 변경 불가능한 로그

예시: 입력 검증

def validate_input(data: dict) -> tuple[bool, list]:
    errors = []

    # 필수 필드
    if not data.get("message"):
        errors.append("message는 필수입니다")

    # 길이 제한
    if len(data.get("message", "")) > 10000:
        errors.append("message는 10000자 이하여야 합니다")

    # 타입 검증
    if not isinstance(data.get("count", 0), int):
        errors.append("count는 정수여야 합니다")

    return len(errors) == 0, errors
""")


# =============================================================================
# 8. 최종 정리
# =============================================================================

def final_summary():
    """최종 정리"""
    print("\n" + "=" * 60)
    print("📘 LangGraph 교육 과정 완료!")
    print("=" * 60)

    print("""
🎉 축하합니다! LangGraph 교육 과정을 모두 완료했습니다!

학습 내용 요약:

Part 1: Foundation (기초)
  - LangGraph 소개 및 설치
  - State, Node, Edge 개념
  - Reducer와 MessagesState

Part 2: Workflows (워크플로우)
  - Prompt Chaining
  - Routing (조건부 분기)
  - Parallelization (병렬 처리)
  - Orchestrator-Worker
  - Evaluator-Optimizer

Part 3: Agent (에이전트)
  - Tool Calling
  - ReAct Agent
  - Multi-Agent 시스템
  - Subgraph

Part 4: Production (프로덕션)
  - Checkpointer (상태 저장)
  - Memory (단기/장기 메모리)
  - Streaming (스트리밍)
  - Human-in-the-Loop
  - Time Travel

Part 5: Advanced (고급)
  - Functional API
  - Durable Execution
  - 배포 준비

다음 단계:

1. 실습 프로젝트
   - examples/ 폴더의 프로젝트 완성
   - 자신만의 Agent 구현

2. 심화 학습
   - LangGraph Cloud 탐색
   - 고급 패턴 연구

3. 커뮤니티 참여
   - GitHub Issues/Discussions
   - 블로그 포스팅

공식 문서:
  https://langchain-ai.github.io/langgraph/

감사합니다! 🙏
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 20] 배포 준비 (Deployment Ready)")
    print("=" * 60)

    load_dotenv()

    # 프로덕션 체크리스트
    explain_production_checklist()

    # 예제 실행
    run_config_example()
    run_logging_example()
    run_metrics_example()
    run_robust_example()
    run_health_check_example()

    # 보안 고려사항
    explain_security()

    # 최종 정리
    final_summary()

    print("\n" + "=" * 60)
    print("✅ LangGraph 교육 과정 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
