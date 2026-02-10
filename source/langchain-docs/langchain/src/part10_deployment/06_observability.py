"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: 배포와 관측성 (Deployment & Observability)
================================================================================

파일명: 06_observability.py
난이도: ⭐⭐⭐⭐⭐ (전문가)
예상 시간: 30분

📚 학습 목표:
  - 모니터링 시스템 구축
  - 구조화된 로깅
  - 알림 및 경보 설정
  - 대시보드 구축
  - 인시던트 대응

📖 공식 문서:
  • LangSmith: /official/30-langsmith-studio.md
  • Observability Best Practices

📄 교안 문서:
  • Part 10 개요: /docs/part10_deployment.md

🔧 필요한 패키지:
  pip install langchain langchain-openai

🔑 필요한 환경변수:
  - OPENAI_API_KEY
  - LANGSMITH_API_KEY (선택)

🚀 실행 방법:
  python 06_observability.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: 모니터링 시스템
# ============================================================================

def example_1_monitoring():
    """모니터링 시스템"""
    print("=" * 70)
    print("📌 예제 1: 모니터링 시스템")
    print("=" * 70)

    print("""
📊 모니터링 (Monitoring)이란?

정의:
  시스템의 상태, 성능, 가용성을 지속적으로 관찰하고 측정하는 것

왜 필요한가?
  • 문제 조기 발견
  • 성능 최적화
  • 용량 계획 (Capacity Planning)
  • SLA 준수 확인
  • 근본 원인 분석

관측성의 3가지 기둥 (Three Pillars):

1️⃣ 메트릭 (Metrics)
   • 정의: 시간에 따라 측정되는 숫자 값
   • 예: CPU 사용률, 요청 수, 응답 시간
   • 도구: Prometheus, CloudWatch, Datadog

2️⃣ 로그 (Logs)
   • 정의: 이벤트의 텍스트 기록
   • 예: 에러 메시지, 사용자 행동, 디버그 정보
   • 도구: ELK Stack, Loki, CloudWatch Logs

3️⃣ 트레이스 (Traces)
   • 정의: 요청의 전체 흐름 추적
   • 예: Agent → LLM → Tool → 응답
   • 도구: Jaeger, Zipkin, LangSmith

핵심 메트릭 (Golden Signals):
  • Latency: 응답 시간
  • Traffic: 요청 수
  • Errors: 오류율
  • Saturation: 리소스 사용률
    """)

    print("\n🔹 메트릭 수집 예제:")
    print("-" * 70)

    # 메트릭 수집 클래스
    class MetricsCollector:
        """간단한 메트릭 수집기"""

        def __init__(self):
            self.metrics = {
                "requests_total": 0,
                "requests_success": 0,
                "requests_failed": 0,
                "latencies": [],
            }

        def record_request(self, success: bool, latency: float):
            """요청 메트릭 기록"""
            self.metrics["requests_total"] += 1
            if success:
                self.metrics["requests_success"] += 1
            else:
                self.metrics["requests_failed"] += 1
            self.metrics["latencies"].append(latency)

        def get_metrics(self) -> Dict[str, Any]:
            """메트릭 반환"""
            latencies = self.metrics["latencies"]
            return {
                "requests_total": self.metrics["requests_total"],
                "requests_success": self.metrics["requests_success"],
                "requests_failed": self.metrics["requests_failed"],
                "success_rate": (
                    self.metrics["requests_success"] / self.metrics["requests_total"] * 100
                    if self.metrics["requests_total"] > 0 else 0
                ),
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
            }

    # 메트릭 수집기 초기화
    metrics = MetricsCollector()

    # Agent 생성
    @tool
    def demo_tool(query: str) -> str:
        """데모 도구"""
        return f"처리 완료: {query}"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[demo_tool],
    )

    # 여러 요청 시뮬레이션
    print("\n요청 처리 중...")
    test_queries = [
        "첫 번째 질문",
        "두 번째 질문",
        "세 번째 질문",
        "네 번째 질문",
        "다섯 번째 질문",
    ]

    for i, query in enumerate(test_queries, 1):
        start = time.time()
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            latency = time.time() - start
            metrics.record_request(success=True, latency=latency)
            print(f"  [{i}] ✅ 성공 ({latency:.2f}s)")
        except Exception as e:
            latency = time.time() - start
            metrics.record_request(success=False, latency=latency)
            print(f"  [{i}] ❌ 실패: {e}")

    # 메트릭 출력
    print("\n" + "-" * 70)
    print("\n📊 수집된 메트릭:")
    collected_metrics = metrics.get_metrics()
    for key, value in collected_metrics.items():
        if "latency" in key:
            print(f"   • {key}: {value:.3f}s")
        elif "rate" in key:
            print(f"   • {key}: {value:.1f}%")
        else:
            print(f"   • {key}: {value}")

    print("\n💡 메트릭 활용:")
    print("   • Prometheus로 메트릭 노출 (/metrics 엔드포인트)")
    print("   • Grafana로 대시보드 시각화")
    print("   • 임계값 초과 시 알림")
    print("   • 장기 추세 분석")


# ============================================================================
# 예제 2: 구조화된 로깅
# ============================================================================

def example_2_structured_logging():
    """구조화된 로깅"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 구조화된 로깅")
    print("=" * 70)

    print("""
📝 구조화된 로깅 (Structured Logging)이란?

정의:
  로그를 JSON 등 구조화된 포맷으로 기록하는 것

일반 로그 vs 구조화된 로그:

일반 로그:
  2024-01-15 10:30:45 INFO User john logged in from 192.168.1.1

구조화된 로그 (JSON):
  {
    "timestamp": "2024-01-15T10:30:45Z",
    "level": "INFO",
    "message": "User logged in",
    "user": "john",
    "ip": "192.168.1.1",
    "session_id": "abc123"
  }

장점:
  • 검색 및 필터링 용이
  • 자동 파싱 및 집계
  • 메타데이터 풍부
  • 로그 분석 도구와 통합 쉬움

로그 레벨:
  DEBUG    - 상세한 디버그 정보
  INFO     - 일반 정보
  WARNING  - 경고 (문제는 아니지만 주의)
  ERROR    - 오류 (기능 실패)
  CRITICAL - 심각한 오류 (시스템 장애)

베스트 프랙티스:
  • 민감 정보 (비밀번호, API Key) 로깅 금지
  • 요청 ID로 로그 추적
  • 컨텍스트 정보 포함 (user_id, session_id 등)
  • 일관된 포맷 사용
    """)

    print("\n🔹 구조화된 로깅 예제:")
    print("-" * 70)

    # 간단한 구조화 로거
    class StructuredLogger:
        """구조화된 로거"""

        def __init__(self, service_name: str):
            self.service_name = service_name

        def _log(self, level: str, message: str, **kwargs):
            """로그 기록"""
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "service": self.service_name,
                "level": level,
                "message": message,
                **kwargs
            }
            print(json.dumps(log_entry, ensure_ascii=False, indent=2))

        def info(self, message: str, **kwargs):
            self._log("INFO", message, **kwargs)

        def warning(self, message: str, **kwargs):
            self._log("WARNING", message, **kwargs)

        def error(self, message: str, **kwargs):
            self._log("ERROR", message, **kwargs)

        def debug(self, message: str, **kwargs):
            self._log("DEBUG", message, **kwargs)

    # 로거 초기화
    logger = StructuredLogger("langchain-agent")

    print("\n다양한 로그 예제:\n")

    # 1. 정상 요청
    logger.info(
        "Request received",
        request_id="req-001",
        user_id="user-123",
        endpoint="/chat",
        method="POST"
    )

    # 2. Tool 실행
    logger.debug(
        "Tool execution started",
        request_id="req-001",
        tool_name="search_database",
        parameters={"query": "AI"}
    )

    # 3. 경고
    logger.warning(
        "High latency detected",
        request_id="req-001",
        latency_ms=2500,
        threshold_ms=2000
    )

    # 4. 오류
    logger.error(
        "Tool execution failed",
        request_id="req-001",
        tool_name="search_database",
        error="Connection timeout",
        retry_count=3
    )

    # 5. 성공 응답
    logger.info(
        "Request completed",
        request_id="req-001",
        status_code=200,
        latency_ms=2500,
        tokens_used=150
    )

    print("\n" + "-" * 70)

    print("\n💡 로깅 모범 사례:")
    print("   • 요청 ID로 전체 흐름 추적")
    print("   • 적절한 로그 레벨 사용")
    print("   • 에러 시 스택 트레이스 포함")
    print("   • 프로덕션에서는 DEBUG 로그 비활성화")
    print("   • 로그 로테이션 설정")
    print("   • 중앙 집중식 로그 수집 (ELK, CloudWatch)")


# ============================================================================
# 예제 3: 알림 및 경보
# ============================================================================

def example_3_alerting():
    """알림 및 경보"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 알림 및 경보")
    print("=" * 70)

    print("""
🚨 알림 (Alerting)이란?

정의:
  특정 조건이 충족될 때 자동으로 팀에게 통지하는 시스템

왜 필요한가?
  • 장애 조기 감지
  • 24/7 모니터링
  • 빠른 대응
  • 서비스 품질 유지

알림 종류:

1️⃣ 임계값 기반 (Threshold-based)
   • 예: CPU > 80%, 에러율 > 5%

2️⃣ 이상 탐지 (Anomaly Detection)
   • 예: 평소 패턴과 다른 트래픽

3️⃣ 복합 조건 (Composite)
   • 예: 에러율 > 5% AND 지연시간 > 2s

알림 채널:
  • Slack, Discord
  • Email
  • PagerDuty, Opsgenie
  • SMS, 전화
  • Webhook

알림 설계 원칙:
  • 실행 가능한 알림만 (Actionable Alerts)
  • 알림 피로 방지 (Alert Fatigue)
  • 우선순위 지정 (P0, P1, P2, P3)
  • 에스컬레이션 정책
  • On-call 로테이션
    """)

    print("\n🔹 알림 규칙 예제:")
    print("-" * 70)

    # 알림 시스템 시뮬레이션
    class AlertManager:
        """알림 관리자"""

        def __init__(self):
            self.alerts = []

        def check_and_alert(self, metric_name: str, value: float, threshold: float,
                           severity: str = "WARNING"):
            """메트릭 확인 및 알림"""
            if value > threshold:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "severity": severity,
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold,
                    "message": f"{metric_name} exceeded threshold: {value} > {threshold}"
                }
                self.alerts.append(alert)
                self._send_alert(alert)
                return True
            return False

        def _send_alert(self, alert: Dict[str, Any]):
            """알림 전송 (시뮬레이션)"""
            severity_icon = {
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "🔥"
            }.get(alert["severity"], "🔔")

            print(f"\n{severity_icon} 알림 발생!")
            print(f"   심각도: {alert['severity']}")
            print(f"   메트릭: {alert['metric']}")
            print(f"   현재값: {alert['value']}")
            print(f"   임계값: {alert['threshold']}")
            print(f"   메시지: {alert['message']}")
            print(f"   시간: {alert['timestamp']}")

    # 알림 매니저 초기화
    alert_manager = AlertManager()

    # 시뮬레이션 메트릭
    print("\n모니터링 메트릭 확인 중...\n")

    metrics_to_check = [
        {"name": "응답시간", "value": 2.5, "threshold": 2.0, "severity": "WARNING"},
        {"name": "에러율", "value": 8.5, "threshold": 5.0, "severity": "ERROR"},
        {"name": "CPU 사용률", "value": 85.0, "threshold": 80.0, "severity": "WARNING"},
        {"name": "메모리 사용률", "value": 95.0, "threshold": 90.0, "severity": "CRITICAL"},
        {"name": "디스크 사용률", "value": 60.0, "threshold": 80.0, "severity": "INFO"},
    ]

    for metric in metrics_to_check:
        alerted = alert_manager.check_and_alert(
            metric["name"],
            metric["value"],
            metric["threshold"],
            metric["severity"]
        )
        if not alerted:
            print(f"✅ {metric['name']}: {metric['value']} (정상)")

    # 알림 요약
    print("\n" + "=" * 70)
    print(f"\n📊 알림 요약: 총 {len(alert_manager.alerts)}개 발생")

    if alert_manager.alerts:
        severity_counts = {}
        for alert in alert_manager.alerts:
            sev = alert["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        print("\n심각도별:")
        for severity, count in sorted(severity_counts.items()):
            print(f"   • {severity}: {count}개")

    print("\n📄 Slack 알림 예제:")
    print('''
import requests

def send_slack_alert(webhook_url: str, alert: dict):
    """Slack으로 알림 전송"""
    message = {
        "text": f"🚨 Alert: {alert['severity']}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 {alert['severity']} Alert"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Metric:*\\n{alert['metric']}"},
                    {"type": "mrkdwn", "text": f"*Value:*\\n{alert['value']}"},
                    {"type": "mrkdwn", "text": f"*Threshold:*\\n{alert['threshold']}"},
                    {"type": "mrkdwn", "text": f"*Time:*\\n{alert['timestamp']}"}
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert['message']
                }
            }
        ]
    }
    requests.post(webhook_url, json=message)
    ''')

    print("\n💡 알림 모범 사례:")
    print("   • 실행 가능한 알림만 설정")
    print("   • 심각도 명확히 구분")
    print("   • 런북(Runbook) 링크 포함")
    print("   • 자동 복구(Auto-remediation) 고려")
    print("   • 알림 피로 방지 (중복 제거, 그룹화)")
    print("   • 정기적으로 알림 규칙 검토")


# ============================================================================
# 예제 4: 대시보드 구축
# ============================================================================

def example_4_dashboards():
    """대시보드 구축"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 대시보드 구축")
    print("=" * 70)

    print("""
📈 대시보드 (Dashboard)란?

정의:
  시스템 상태와 메트릭을 시각화하는 UI

왜 필요한가?
  • 한눈에 시스템 상태 파악
  • 트렌드 분석
  • 문제 진단
  • 팀 간 커뮤니케이션

대시보드 종류:

1️⃣ 운영 대시보드 (Operational)
   • 실시간 메트릭
   • 현재 상태 모니터링
   • 예: 요청 수, 에러율, 지연시간

2️⃣ 분석 대시보드 (Analytical)
   • 장기 트렌드
   • 비즈니스 메트릭
   • 예: DAU, 사용 패턴, 비용

3️⃣ 전략 대시보드 (Strategic)
   • 고위 경영진용
   • KPI 및 목표 추적
   • 예: SLA 준수율, ROI

대시보드 설계 원칙:
  • 가장 중요한 메트릭을 상단에
  • 색상으로 상태 표시 (녹색/노랑/빨강)
  • 적절한 차트 유형 선택
  • 드릴다운 가능
  • 컨텍스트 제공 (목표, 임계값)

도구:
  • Grafana (오픈소스, 인기)
  • Datadog
  • New Relic
  • CloudWatch Dashboards
  • Kibana
    """)

    print("\n🔹 대시보드 구성 예제:")
    print("-" * 70)

    print("""
📊 LangChain Agent 대시보드 구성:

═══════════════════════════════════════════════════════════
🎯 핵심 메트릭 (Top Row)
═══════════════════════════════════════════════════════════

┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  요청 수/분     │   성공률        │   평균 지연     │   에러율        │
│   1,234         │   99.2%         │   1.2s          │   0.8%          │
│   ↑ 15%         │   ↓ 0.3%        │   ↓ 0.2s        │   ↑ 0.1%        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

═══════════════════════════════════════════════════════════
📈 시계열 차트 (Middle Row)
═══════════════════════════════════════════════════════════

요청 수 (시간별):
  [실시간 라인 차트]
  - 지난 24시간 트렌드
  - 피크 시간대 확인

응답 시간 분포 (백분위수):
  [히스토그램]
  - P50, P95, P99
  - SLA 목표선 표시

에러 타입별 분포:
  [파이 차트]
  - Tool 실패
  - LLM 타임아웃
  - 네트워크 오류

═══════════════════════════════════════════════════════════
🔧 시스템 리소스 (Bottom Row)
═══════════════════════════════════════════════════════════

CPU/메모리 사용률:
  [게이지 차트]
  - 현재 사용률
  - 임계값 경고

토큰 사용량:
  [영역 차트]
  - 누적 토큰 수
  - 비용 추정

활성 세션:
  [숫자 표시]
  - 현재 활성 사용자
  - 동시 요청 수
    """)

    print("\n📄 Grafana 대시보드 JSON 예제:")
    print('''
{
  "dashboard": {
    "title": "LangChain Agent Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Requests per Minute",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agent_requests_total[1m])",
            "legendFormat": "Requests/min"
          }
        ]
      },
      {
        "id": 2,
        "title": "Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(agent_requests_success) / sum(agent_requests_total) * 100",
            "legendFormat": "Success Rate"
          }
        ],
        "thresholds": [
          {"value": 95, "color": "red"},
          {"value": 99, "color": "yellow"},
          {"value": 99.5, "color": "green"}
        ]
      },
      {
        "id": 3,
        "title": "Response Time (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, agent_request_duration_seconds)",
            "legendFormat": "P95 Latency"
          }
        ]
      }
    ]
  }
}
    ''')

    print("\n💡 대시보드 모범 사례:")
    print("   • 청중에 맞는 대시보드 (개발자 vs 경영진)")
    print("   • 가장 중요한 메트릭 강조")
    print("   • 실시간 업데이트 (자동 새로고침)")
    print("   • 알림과 연동")
    print("   • 팀 전체가 접근 가능")
    print("   • 정기적으로 리뷰 및 업데이트")


# ============================================================================
# 예제 5: 인시던트 대응
# ============================================================================

def example_5_incident_response():
    """인시던트 대응"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 인시던트 대응")
    print("=" * 70)

    print("""
🚒 인시던트 대응 (Incident Response)이란?

정의:
  서비스 장애나 문제 발생 시 신속하게 대응하고 복구하는 프로세스

인시던트 심각도:

P0 (Critical)
  • 정의: 전체 서비스 다운
  • 예: API 완전 불통, 데이터 손실
  • 대응: 즉시, 24/7
  • SLA: 15분 내 대응, 1시간 내 복구

P1 (High)
  • 정의: 주요 기능 장애
  • 예: Agent 응답 없음, 높은 오류율
  • 대응: 30분 내
  • SLA: 4시간 내 복구

P2 (Medium)
  • 정의: 일부 기능 저하
  • 예: 느린 응답, 특정 Tool 오류
  • 대응: 1시간 내
  • SLA: 24시간 내 복구

P3 (Low)
  • 정의: 사소한 문제
  • 예: UI 버그, 로그 경고
  • 대응: 업무 시간 내
  • SLA: 1주일 내 수정

인시던트 대응 프로세스:

1️⃣ 감지 (Detection)
   • 모니터링 알림
   • 사용자 보고
   • 자동 감지 시스템

2️⃣ 대응 (Response)
   • On-call 엔지니어 호출
   • 인시던트 채널 생성
   • 초기 평가

3️⃣ 진단 (Diagnosis)
   • 로그/메트릭 분석
   • 트레이스 확인
   • 근본 원인 파악

4️⃣ 복구 (Resolution)
   • 임시 조치 (Workaround)
   • 롤백 또는 Hotfix
   • 서비스 정상화 확인

5️⃣ 사후 검토 (Post-Mortem)
   • 타임라인 정리
   • 근본 원인 문서화
   • 재발 방지 계획
   • Blameless Culture
    """)

    print("\n🔹 인시던트 대응 시나리오:")
    print("-" * 70)

    # 인시던트 시뮬레이션
    print("\n🚨 인시던트 발생!")
    print("=" * 70)

    incident = {
        "id": "INC-2024-001",
        "title": "Agent 응답 시간 급증",
        "severity": "P1",
        "status": "INVESTIGATING",
        "detected_at": "2024-02-06T14:30:00Z",
        "reporter": "monitoring-system",
        "impact": "사용자 50%가 2초 이상 지연 경험",
    }

    print(f"\n📋 인시던트 정보:")
    print(f"   ID: {incident['id']}")
    print(f"   제목: {incident['title']}")
    print(f"   심각도: {incident['severity']}")
    print(f"   상태: {incident['status']}")
    print(f"   발생 시간: {incident['detected_at']}")
    print(f"   영향: {incident['impact']}")

    print("\n" + "=" * 70)
    print("🔍 대응 타임라인:\n")

    timeline = [
        {
            "time": "14:30",
            "event": "알림 수신",
            "action": "P95 응답시간이 5초 초과",
            "owner": "monitoring"
        },
        {
            "time": "14:32",
            "event": "On-call 엔지니어 확인",
            "action": "인시던트 채널 생성, 초기 조사 시작",
            "owner": "engineer-1"
        },
        {
            "time": "14:35",
            "event": "근본 원인 파악",
            "action": "외부 API 응답 지연 확인 (데이터베이스 부하)",
            "owner": "engineer-1"
        },
        {
            "time": "14:38",
            "event": "임시 조치",
            "action": "타임아웃 설정 조정, 서킷 브레이커 활성화",
            "owner": "engineer-2"
        },
        {
            "time": "14:45",
            "event": "캐싱 적용",
            "action": "자주 사용되는 쿼리 캐싱 활성화",
            "owner": "engineer-1"
        },
        {
            "time": "14:50",
            "event": "복구 확인",
            "action": "P95 응답시간 1.5초로 정상화",
            "owner": "engineer-2"
        },
        {
            "time": "14:55",
            "event": "인시던트 종료",
            "action": "모니터링 지속, 사후 검토 예약",
            "owner": "engineer-1"
        },
    ]

    for entry in timeline:
        print(f"[{entry['time']}] {entry['event']}")
        print(f"          → {entry['action']}")
        print(f"          담당: {entry['owner']}\n")

    print("=" * 70)
    print("\n✅ 인시던트 해결 완료")
    print(f"   총 소요 시간: 25분")
    print(f"   다운타임: 없음 (서비스 저하만 발생)")

    print("\n📝 사후 검토 (Post-Mortem):")
    print("""
제목: Agent 응답 시간 급증 인시던트
날짜: 2024-02-06
심각도: P1
영향: 사용자 50%, 25분간 지연 경험

타임라인:
  [위 타임라인 참조]

근본 원인:
  • 외부 데이터베이스 API의 예상치 못한 부하
  • 캐싱 미적용으로 모든 요청이 API 호출

왜 발생했나:
  • 캐싱 전략이 구현되지 않음
  • 외부 API 의존성에 대한 모니터링 부족
  • 서킷 브레이커 미구현

잘 된 점:
  ✅ 모니터링 시스템이 즉시 감지
  ✅ 빠른 초기 대응 (2분 내)
  ✅ 효과적인 임시 조치

개선할 점:
  ❌ 캐싱이 미리 구현되었어야 함
  ❌ 외부 API 의존성 모니터링 부족
  ❌ 롤백 플랜 부재

액션 아이템:
  1. [ ] 캐싱 전략 구현 및 배포 (담당: engineer-1, 기한: 2024-02-10)
  2. [ ] 외부 API 모니터링 추가 (담당: engineer-2, 기한: 2024-02-08)
  3. [ ] 서킷 브레이커 패턴 구현 (담당: engineer-1, 기한: 2024-02-15)
  4. [ ] 롤백 플레이북 작성 (담당: team, 기한: 2024-02-12)
  5. [ ] 의존성 매핑 및 문서화 (담당: engineer-2, 기한: 2024-02-20)
    """)

    print("\n💡 인시던트 대응 모범 사례:")
    print("   • 명확한 에스컬레이션 정책")
    print("   • Runbook (대응 절차서) 준비")
    print("   • Blameless Post-Mortem 문화")
    print("   • 정기적인 장애 훈련 (Game Day)")
    print("   • 커뮤니케이션 채널 사전 정의")
    print("   • 인시던트 히스토리 데이터베이스")
    print("   • 학습한 내용을 시스템에 반영")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 10: 배포와 관측성 - 관측성")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_monitoring()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_structured_logging()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_alerting()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_dashboards()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_incident_response()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 10-06: 관측성을 완료했습니다!")
    print("=" * 70)
    print("\n🎊 축하합니다! Part 10: 배포와 관측성 전체를 완료했습니다!")
    print("\n📚 Part 10 핵심 요약:")
    print("  • LangSmith로 트레이싱 및 평가")
    print("  • 자동화된 테스트로 품질 보증")
    print("  • 평가 메트릭으로 성능 측정")
    print("  • Docker와 Kubernetes로 배포")
    print("  • 모니터링, 로깅, 알림으로 관측성 확보")
    print("\n🚀 이제 프로덕션 환경에 Agent를 배포할 준비가 되었습니다!")
    print("\n💡 계속 학습하세요:")
    print("  • 실제 프로젝트에 적용")
    print("  • 커뮤니티와 경험 공유")
    print("  • 최신 LangChain 업데이트 팔로우")
    print("  • 다양한 Agent 패턴 실험")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 모니터링 도구:
#    - Prometheus + Grafana
#    - Datadog
#    - New Relic
#    - CloudWatch
#
# 2. 로그 관리:
#    - ELK Stack (Elasticsearch, Logstash, Kibana)
#    - Loki + Grafana
#    - Splunk
#    - CloudWatch Logs Insights
#
# 3. 트레이싱:
#    - Jaeger
#    - Zipkin
#    - LangSmith
#    - Datadog APM
#
# 4. 알림 도구:
#    - PagerDuty
#    - Opsgenie
#    - Slack
#    - Email/SMS
#
# 5. SRE 개념:
#    - SLI, SLO, SLA
#    - Error Budget
#    - Toil Reduction
#    - Chaos Engineering
#
# 6. 인시던트 관리:
#    - Incident.io
#    - Blameless Post-Mortems
#    - On-call Rotation
#    - Runbooks
#
# ============================================================================
