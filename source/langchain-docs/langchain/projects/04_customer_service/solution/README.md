# Customer Service Agent 솔루션 가이드

## 구현 핵심 포인트

### 1. 에이전트 구조
- **RouterAgent**: 문의 유형 분류 (기술지원, 결제, 일반, 에스컬레이션)
- **SupportAgent**: 기술 지원 담당
- **BillingAgent**: 결제 관련 처리
- **GeneralAgent**: 일반 문의 응대
- **EscalationAgent**: 복잡한 케이스 상위 전달

### 2. 미들웨어 활용
- **HITL 미들웨어**: 중요 결정 시 사람 승인 요청
- **모니터링 미들웨어**: 응답 시간, 만족도 추적

### 3. 프로덕션 요소
- FastAPI 기반 API 서버
- 구조화된 로깅 (structlog)
- Docker 배포 설정

### 4. 주요 학습 포인트
- Router 패턴으로 문의 라우팅
- Handoffs 패턴으로 에이전트 간 전환
- 미들웨어로 횡단 관심사 처리

## 참고 자료

- [Part 5: 미들웨어](/docs/part05_middleware.md) - 미들웨어 시스템
- [Part 7: 멀티에이전트](/docs/part07_multi_agent.md) - Router/Handoffs 패턴
- [Part 9: 프로덕션](/docs/part09_production.md) - HITL, 스트리밍
- [Part 10: 배포](/docs/part10_deployment.md) - 배포 및 관측성
