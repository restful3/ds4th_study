# Code Assistant 예제

LangGraph를 활용한 코드 분석 및 생성 Assistant 구현 예제입니다.

## 개요

이 예제는 다음 기능을 포함합니다:
- 코드 분석 및 리뷰
- 코드 생성 및 수정
- Human-in-the-Loop 승인 프로세스
- 코드 실행 및 테스트 (시뮬레이션)

## 아키텍처

```
┌────────────────────────────────────────────────────────────────┐
│                      Code Assistant                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌───────────┐     ┌───────────────────┐    │
│   │  Analyze │────▶│ Plan Code │────▶│ Human Approval    │    │
│   │  Request │     │  Changes  │     │ (interrupt)       │    │
│   └──────────┘     └───────────┘     └─────────┬─────────┘    │
│                                                 │               │
│                    ┌────────────────────────────┴─────────┐    │
│                    │                                      │    │
│                    ▼ (approved)                 ▼ (rejected)   │
│           ┌───────────────┐              ┌─────────────────┐  │
│           │   Generate    │              │  Request More   │  │
│           │     Code      │              │    Details      │  │
│           └───────┬───────┘              └─────────────────┘  │
│                   │                                            │
│                   ▼                                            │
│           ┌───────────────┐                                   │
│           │  Validate &   │                                   │
│           │    Review     │                                   │
│           └───────────────┘                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 학습 포인트

1. **Human-in-the-Loop**: 코드 변경 전 승인 프로세스
2. **Multi-Step 워크플로우**: 분석 → 계획 → 생성 → 검증
3. **도구 통합**: 코드 실행, 린팅, 테스트 도구
4. **상태 관리**: 코드 변경 히스토리 추적

## 파일 구조

```
05_code_assistant/
├── README.md          # 이 파일
└── main.py            # 메인 구현
```

## 실행 방법

```bash
# 환경 변수 설정
export ANTHROPIC_API_KEY=your-api-key

# 기본 데모
python main.py

# 인터랙티브 모드
python main.py interactive
```

## 주요 컴포넌트

### 1. 요청 분석기 (Request Analyzer)
사용자 요청을 분석하여 작업 유형 결정

### 2. 코드 플래너 (Code Planner)
코드 변경 계획 수립

### 3. Human Approval (Human-in-the-Loop)
코드 변경 전 사용자 승인

### 4. 코드 생성기 (Code Generator)
계획에 따른 코드 생성

### 5. 코드 검증기 (Code Validator)
생성된 코드의 문법 및 품질 검증

## 지원하는 작업

- `analyze`: 코드 분석 및 설명
- `generate`: 새 코드 생성
- `modify`: 기존 코드 수정
- `review`: 코드 리뷰
- `explain`: 코드 설명

## 확장 아이디어

- 실제 코드 실행 환경 연동
- Git 통합 (커밋, 브랜치 관리)
- 테스트 자동 생성
- 다중 파일 수정 지원
- IDE 플러그인 연동

## 관련 챕터

- [Chapter 9: 도구와 에이전트](../../docs/Part3-Agent/09-tools-and-agents.md)
- [Chapter 15: Human-in-the-Loop](../../docs/Part4-Production/15-human-in-the-loop.md)
- [Chapter 16: 스트리밍](../../docs/Part4-Production/16-streaming.md)
