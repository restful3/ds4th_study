"""
[Chapter 18] Functional API

📝 설명:
    Functional API는 @entrypoint와 @task 데코레이터를 사용하여
    더 간결하고 직관적인 방식으로 그래프를 정의합니다.
    복잡한 워크플로우를 함수형 스타일로 표현할 수 있습니다.

🎯 학습 목표:
    - @entrypoint 데코레이터 이해
    - @task 데코레이터 이해
    - Functional API vs Graph API 비교
    - 복잡한 워크플로우의 함수형 표현

📚 관련 문서:
    - docs/Part5-Advanced/18-functional-api.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/functional_api/

💻 실행 방법:
    python -m src.part5_advanced.18_functional_api

📦 필요한 패키지:
    - langgraph>=0.2.0
"""

import os
from typing import TypedDict, List, Any
from dotenv import load_dotenv

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# =============================================================================
# 1. Functional API 개념 설명
# =============================================================================

def explain_functional_api():
    """Functional API 개념 설명"""
    print("\n" + "=" * 60)
    print("📘 Functional API")
    print("=" * 60)

    print("""
Functional API란?
    데코레이터 기반의 선언적 방식으로 LangGraph 워크플로우를 정의합니다.
    @entrypoint와 @task를 사용하여 그래프 구조를 더 직관적으로 표현합니다.

Graph API vs Functional API:

┌─────────────────┬────────────────────────────────────┐
│   Graph API     │         Functional API             │
├─────────────────┼────────────────────────────────────┤
│ StateGraph()    │ @entrypoint                        │
│ add_node()      │ @task                              │
│ add_edge()      │ 함수 호출로 자동 연결              │
│ compile()       │ 데코레이터가 자동 처리             │
│ 명시적 상태     │ 함수 인자/반환값 = 상태            │
└─────────────────┴────────────────────────────────────┘

주요 데코레이터:

1. @entrypoint
   - 워크플로우의 진입점 정의
   - Checkpointer 설정 가능
   - 그래프 컴파일 역할

2. @task
   - 개별 작업 단위 정의
   - 자동으로 노드로 변환
   - 병렬 실행 가능

장점:
    - 코드가 더 간결하고 읽기 쉬움
    - 일반 Python 함수처럼 작성
    - 타입 힌트와 자연스럽게 통합
    - 테스트하기 용이

제한사항:
    - 복잡한 조건부 엣지에는 Graph API가 유리
    - 일부 고급 기능은 Graph API만 지원
""")


# =============================================================================
# 2. @task 기본 사용법
# =============================================================================

@task
def fetch_data(url: str) -> dict:
    """데이터 가져오기 Task"""
    # 실제로는 HTTP 요청
    return {"url": url, "data": f"Data from {url}", "status": "success"}


@task
def process_data(data: dict) -> dict:
    """데이터 처리 Task"""
    content = data.get("data", "")
    processed = content.upper()
    return {"original": content, "processed": processed}


@task
def save_result(result: dict) -> str:
    """결과 저장 Task"""
    # 실제로는 DB 저장
    return f"Saved: {result.get('processed', '')[:30]}..."


def run_task_example():
    """@task 예제"""
    print("\n" + "=" * 60)
    print("예제 1: @task 기본 사용")
    print("=" * 60)

    # @task 함수는 일반 함수처럼 호출 가능
    # 하지만 entrypoint 내에서 호출하면 자동으로 노드로 변환됨

    @entrypoint(checkpointer=MemorySaver())
    def data_pipeline(url: str) -> str:
        """데이터 파이프라인"""
        # Task들을 순차적으로 호출
        fetched = fetch_data(url)
        processed = process_data(fetched)
        saved = save_result(processed)
        return saved

    # 실행
    config = {"configurable": {"thread_id": "pipeline_1"}}
    result = data_pipeline.invoke("https://api.example.com/data", config=config)

    print(f"\n📥 파이프라인 결과:")
    print(f"   {result}")


# =============================================================================
# 3. @entrypoint 기본 사용법
# =============================================================================

def run_entrypoint_example():
    """@entrypoint 예제"""
    print("\n" + "=" * 60)
    print("예제 2: @entrypoint 기본 사용")
    print("=" * 60)

    # Task 정의
    @task
    def greet(name: str) -> str:
        return f"안녕하세요, {name}님!"

    @task
    def add_emoji(text: str) -> str:
        return f"👋 {text} 🎉"

    @task
    def uppercase(text: str) -> str:
        return text.upper()

    # Entrypoint 정의
    @entrypoint(checkpointer=MemorySaver())
    def greeting_workflow(name: str) -> str:
        """인사 워크플로우"""
        greeting = greet(name)
        with_emoji = add_emoji(greeting)
        final = uppercase(with_emoji)
        return final

    # 실행
    config = {"configurable": {"thread_id": "greet_1"}}
    result = greeting_workflow.invoke("철수", config=config)

    print(f"\n💬 인사 결과:")
    print(f"   {result}")


# =============================================================================
# 4. 병렬 Task 실행
# =============================================================================

@task
def analyze_sentiment(text: str) -> dict:
    """감정 분석 (시뮬레이션)"""
    # 간단한 규칙 기반 분석
    positive_words = ["좋", "훌륭", "최고", "행복"]
    negative_words = ["나쁨", "싫", "최악", "슬픔"]

    score = 0
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1

    return {"type": "sentiment", "score": score}


@task
def extract_keywords(text: str) -> dict:
    """키워드 추출 (시뮬레이션)"""
    # 간단한 키워드 추출
    words = text.split()
    keywords = [w for w in words if len(w) > 2][:5]
    return {"type": "keywords", "keywords": keywords}


@task
def count_stats(text: str) -> dict:
    """통계 계산"""
    return {
        "type": "stats",
        "char_count": len(text),
        "word_count": len(text.split())
    }


def run_parallel_tasks_example():
    """병렬 Task 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 병렬 Task 실행")
    print("=" * 60)

    @task
    def combine_results(results: List[dict]) -> dict:
        """결과 병합"""
        combined = {}
        for result in results:
            result_type = result.get("type", "unknown")
            combined[result_type] = result
        return combined

    @entrypoint(checkpointer=MemorySaver())
    def text_analysis(text: str) -> dict:
        """텍스트 분석 워크플로우"""
        # 병렬 실행 (Functional API에서는 자동으로 최적화)
        sentiment = analyze_sentiment(text)
        keywords = extract_keywords(text)
        stats = count_stats(text)

        # 결과 병합
        results = combine_results([sentiment, keywords, stats])
        return results

    # 실행
    config = {"configurable": {"thread_id": "analysis_1"}}
    text = "오늘 날씨가 정말 좋아서 기분이 훌륭합니다"
    result = text_analysis.invoke(text, config=config)

    print(f"\n📊 분석 결과:")
    print(f"   입력: '{text}'")
    for key, value in result.items():
        print(f"   {key}: {value}")


# =============================================================================
# 5. 조건부 실행
# =============================================================================

@task
def quick_process(data: str) -> str:
    """빠른 처리"""
    return f"[Quick] {data}"


@task
def detailed_process(data: str) -> str:
    """상세 처리"""
    return f"[Detailed] Analysis of '{data}' with comprehensive results"


def run_conditional_example():
    """조건부 실행 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 조건부 Task 실행")
    print("=" * 60)

    @entrypoint(checkpointer=MemorySaver())
    def conditional_workflow(data: str, detailed: bool = False) -> str:
        """조건부 워크플로우"""
        # 일반적인 if/else로 조건부 실행
        if detailed:
            result = detailed_process(data)
        else:
            result = quick_process(data)
        return result

    # Quick 모드
    config = {"configurable": {"thread_id": "cond_1"}}
    result1 = conditional_workflow.invoke({"data": "샘플 데이터", "detailed": False}, config=config)

    # Detailed 모드
    config2 = {"configurable": {"thread_id": "cond_2"}}
    result2 = conditional_workflow.invoke({"data": "샘플 데이터", "detailed": True}, config=config2)

    print(f"\n🔀 조건부 실행 결과:")
    print(f"   Quick 모드: {result1}")
    print(f"   Detailed 모드: {result2}")


# =============================================================================
# 6. Human-in-the-Loop with Functional API
# =============================================================================

@task
def prepare_proposal(content: str) -> dict:
    """제안서 준비"""
    return {
        "title": "프로젝트 제안",
        "content": content,
        "status": "prepared"
    }


def run_hitl_functional_example():
    """HITL Functional API 예제"""
    print("\n" + "=" * 60)
    print("예제 5: Functional API에서 Human-in-the-Loop")
    print("=" * 60)

    @entrypoint(checkpointer=MemorySaver())
    def approval_workflow(content: str) -> dict:
        """승인 워크플로우"""
        # 제안서 준비
        proposal = prepare_proposal(content)

        # 승인 요청 (interrupt 사용)
        approval = interrupt({
            "type": "approval_request",
            "proposal": proposal,
            "message": "제안서 승인이 필요합니다."
        })

        # 승인 결과에 따라 처리
        if approval.get("approved"):
            return {
                **proposal,
                "status": "approved",
                "approver": approval.get("approver", "Unknown")
            }
        else:
            return {
                **proposal,
                "status": "rejected",
                "reason": approval.get("reason", "No reason provided")
            }

    # 실행
    config = {"configurable": {"thread_id": "approval_1"}}

    # 첫 번째 호출 - interrupt에서 멈춤
    result = approval_workflow.invoke("새로운 AI 프로젝트", config=config)

    # 상태 확인
    state = approval_workflow.get_state(config)
    print(f"\n⏸️  승인 대기 중...")
    print(f"   다음 단계: {state.next if state.next else '없음'}")

    if state.next:
        # 승인하고 재개
        print("   👤 관리자가 승인함")
        result = approval_workflow.invoke(
            Command(resume={"approved": True, "approver": "김관리자"}),
            config=config
        )

    print(f"\n📋 최종 결과:")
    print(f"   상태: {result.get('status')}")
    if result.get('approver'):
        print(f"   승인자: {result.get('approver')}")


# =============================================================================
# 7. 복합 워크플로우 예제
# =============================================================================

@task
def validate_input(data: dict) -> dict:
    """입력 검증"""
    errors = []
    if not data.get("name"):
        errors.append("이름이 필요합니다")
    if not data.get("email"):
        errors.append("이메일이 필요합니다")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": data
    }


@task
def enrich_data(validated: dict) -> dict:
    """데이터 보강"""
    data = validated["data"]
    return {
        **validated,
        "data": {
            **data,
            "created_at": "2024-01-01",
            "enriched": True
        }
    }


@task
def format_output(enriched: dict) -> str:
    """출력 포맷팅"""
    data = enriched["data"]
    return f"사용자: {data.get('name')} ({data.get('email')})"


def run_complex_workflow_example():
    """복합 워크플로우 예제"""
    print("\n" + "=" * 60)
    print("예제 6: 복합 워크플로우")
    print("=" * 60)

    @entrypoint(checkpointer=MemorySaver())
    def user_registration(user_data: dict) -> dict:
        """사용자 등록 워크플로우"""
        # 검증
        validated = validate_input(user_data)

        if not validated["valid"]:
            return {
                "success": False,
                "errors": validated["errors"]
            }

        # 데이터 보강
        enriched = enrich_data(validated)

        # 포맷팅
        output = format_output(enriched)

        return {
            "success": True,
            "message": output,
            "data": enriched["data"]
        }

    # 유효한 입력
    config1 = {"configurable": {"thread_id": "reg_1"}}
    result1 = user_registration.invoke({
        "name": "홍길동",
        "email": "hong@example.com"
    }, config=config1)

    # 유효하지 않은 입력
    config2 = {"configurable": {"thread_id": "reg_2"}}
    result2 = user_registration.invoke({
        "name": "",
        "email": ""
    }, config=config2)

    print(f"\n📝 등록 결과:")
    print(f"\n   유효한 입력:")
    print(f"   - 성공: {result1['success']}")
    print(f"   - 메시지: {result1.get('message', 'N/A')}")

    print(f"\n   유효하지 않은 입력:")
    print(f"   - 성공: {result2['success']}")
    print(f"   - 에러: {result2.get('errors', [])}")


# =============================================================================
# 8. Functional API 패턴 정리
# =============================================================================

def explain_functional_patterns():
    """Functional API 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 Functional API 패턴 정리")
    print("=" * 60)

    print("""
Functional API 사용 패턴:

1. 기본 파이프라인
   @task
   def step1(x): return process1(x)

   @task
   def step2(x): return process2(x)

   @entrypoint(checkpointer=MemorySaver())
   def pipeline(input):
       a = step1(input)
       b = step2(a)
       return b

2. 조건부 실행
   @entrypoint(checkpointer=...)
   def workflow(data, mode):
       if mode == "fast":
           return fast_process(data)
       else:
           return slow_process(data)

3. 병렬 처리
   @entrypoint(checkpointer=...)
   def parallel_workflow(data):
       # 병렬 실행 (자동 최적화)
       result1 = task1(data)
       result2 = task2(data)
       return combine(result1, result2)

4. HITL 통합
   @entrypoint(checkpointer=...)
   def approval_workflow(data):
       prepared = prepare(data)
       approval = interrupt({"message": "승인 필요"})
       return finalize(prepared, approval)

선택 가이드:

Graph API 사용 시:
    - 복잡한 조건부 엣지
    - 동적 라우팅
    - 세밀한 상태 관리

Functional API 사용 시:
    - 간단한 순차/병렬 워크플로우
    - 빠른 프로토타이핑
    - 테스트 용이성 중요
    - 코드 가독성 중요
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 18] Functional API")
    print("=" * 60)

    load_dotenv()

    # 개념 설명
    explain_functional_api()

    # 예제 실행
    run_task_example()
    run_entrypoint_example()
    run_parallel_tasks_example()
    run_conditional_example()
    run_hitl_functional_example()
    run_complex_workflow_example()

    # 패턴 정리
    explain_functional_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 19_durable_execution.py (Durable Execution)")
    print("=" * 60)


if __name__ == "__main__":
    main()
