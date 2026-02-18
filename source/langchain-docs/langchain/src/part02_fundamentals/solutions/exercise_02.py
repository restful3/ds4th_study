"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: Fundamentals - 실습 과제 2 해답
================================================================================

과제: 대화 기록 관리
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. 시스템 메시지로 "당신은 Python 튜터입니다" 설정
2. 3턴의 대화 구현:
   - 턴1: "변수란 무엇인가요?"
   - 턴2: "예제를 보여주세요"
   - 턴3: "그럼 상수는요?"
3. 각 턴마다 대화 기록을 누적하여 전달

학습 목표:
- SystemMessage, HumanMessage, AIMessage 사용법
- 대화 기록 누적을 통한 맥락 유지
- 모델이 이전 대화를 참조하는 방식 이해

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


# ============================================================================
# 1단계: 모델 및 시스템 메시지 설정
# ============================================================================

def setup_conversation():
    """모델 초기화 및 시스템 메시지 설정"""
    print("=" * 70)
    print("1단계: 대화 환경 설정")
    print("=" * 70)

    # 모델 초기화
    model = init_chat_model("gpt-4o-mini", temperature=0.7)

    # 시스템 메시지로 페르소나 설정
    system_message = SystemMessage("""당신은 친절한 Python 튜터입니다.

**역할**:
- 초보자가 이해하기 쉽게 프로그래밍 개념을 설명합니다
- 항상 간단한 코드 예제와 함께 설명합니다
- 핵심 포인트를 2-3줄로 요약합니다

**응답 형식**:
1. 개념 설명 (2-3문장)
2. 코드 예제 (간단한 것)
3. 핵심 포인트 요약
""")

    messages = [system_message]

    print("  시스템 메시지 설정 완료")
    print(f"  현재 메시지 수: {len(messages)}")

    return model, messages


# ============================================================================
# 2단계: 3턴 대화 구현
# ============================================================================

def run_conversation(model, messages: list):
    """3턴의 대화를 실행하며 기록을 누적합니다."""
    print("\n" + "=" * 70)
    print("2단계: 3턴 대화 실행")
    print("=" * 70)

    # 대화 턴 목록
    user_questions = [
        "변수란 무엇인가요?",
        "예제를 보여주세요",
        "그럼 상수는요?",
    ]

    for turn, question in enumerate(user_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"턴 {turn}")
        print(f"{'─' * 70}")

        # 사용자 메시지 추가
        user_msg = HumanMessage(question)
        messages.append(user_msg)
        print(f"\n  사용자: {question}")

        # 모델 호출 (전체 대화 기록 전달)
        response = model.invoke(messages)

        # AI 응답을 대화 기록에 추가
        messages.append(response)
        print(f"\n  AI 응답:\n{response.content}")

        # 현재 대화 기록 상태 출력
        print(f"\n  [대화 기록 길이: {len(messages)}개 메시지]")

    return messages


# ============================================================================
# 3단계: 대화 기록 분석
# ============================================================================

def analyze_conversation(messages: list):
    """대화 기록을 분석하고 맥락 유지를 확인합니다."""
    print("\n" + "=" * 70)
    print("3단계: 대화 기록 분석")
    print("=" * 70)

    print("\n전체 대화 흐름:")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        content_preview = msg.content[:60].replace('\n', ' ')
        if len(msg.content) > 60:
            content_preview += "..."
        print(f"  [{i}] {msg_type}: {content_preview}")

    # 맥락 유지 확인
    print("\n" + "-" * 70)
    print("맥락 유지 확인 포인트:")
    print("  1. 턴2에서 모델이 턴1의 '변수' 개념을 참조하여 예제를 제공했는가?")
    print("  2. 턴3에서 '변수'와 '상수'를 비교하며 설명했는가?")
    print("  3. SystemMessage의 응답 형식(개념/예제/요약)을 일관되게 따랐는가?")


# ============================================================================
# 보너스: Dictionary 포맷으로 동일 대화 구현
# ============================================================================

def bonus_dict_format(model):
    """Dictionary 포맷으로 동일한 대화를 구현합니다."""
    print("\n" + "=" * 70)
    print("보너스: Dictionary 포맷 비교")
    print("=" * 70)

    # Dictionary 형식으로 대화 구성
    messages = [
        {"role": "system", "content": "당신은 친절한 Python 튜터입니다. 간결하게 답변하세요."},
        {"role": "user", "content": "변수란 무엇인가요?"},
    ]

    response = model.invoke(messages)
    print(f"\n  Dictionary 포맷 응답 (일부):")
    print(f"  {response.content[:100]}...")

    print(f"\n  비교:")
    print(f"  - Message 객체: 타입 안전성, IDE 자동완성, 메타데이터 접근")
    print(f"  - Dictionary: 간단한 프로토타입, JSON 직렬화 용이")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("Part 2: 대화 기록 관리 - 실습 과제 2 해답")
    print("=" * 70)

    try:
        # 1. 대화 환경 설정
        model, messages = setup_conversation()

        # 2. 3턴 대화 실행
        messages = run_conversation(model, messages)

        # 3. 대화 기록 분석
        analyze_conversation(messages)

        # 4. 보너스: Dictionary 포맷
        bonus_dict_format(model)

    except Exception as e:
        print(f"\n  오류 발생: {e}")
        print("  API 키가 설정되지 않았거나 네트워크 문제일 수 있습니다")

    # 추가 학습 포인트
    print("\n" + "=" * 70)
    print("추가 학습 포인트:")
    print("  1. SystemMessage를 변경하면서 응답 스타일 차이를 관찰하세요")
    print("  2. 대화 턴을 늘려서 맥락이 언제까지 유지되는지 확인하세요")
    print("  3. 대화 기록을 JSON으로 직렬화/역직렬화 해보세요")
    print("  4. usage_metadata로 턴별 토큰 사용량 변화를 추적하세요")
    print("=" * 70)


if __name__ == "__main__":
    main()
