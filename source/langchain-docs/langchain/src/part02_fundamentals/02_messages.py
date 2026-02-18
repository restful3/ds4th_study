"""
================================================================================
LangChain AI Agent 마스터 교안
Part 2: LangChain 기초
================================================================================

파일명: 02_messages.py
난이도: ⭐⭐☆☆☆ (초급)
예상 시간: 20분

📚 학습 목표:
  - LangChain의 메시지 타입 이해
  - SystemMessage, HumanMessage, AIMessage 활용
  - 메시지 객체와 딕셔너리 포맷 비교
  - 메시지 메타데이터 활용법

📖 공식 문서:
  • Messages: /official/08-messages.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 02_messages.py

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)


# ============================================================================
# 예제 1: 기본 메시지 타입
# ============================================================================

def example_1_basic_messages():
    """SystemMessage, HumanMessage, AIMessage 기본 사용"""
    print("=" * 70)
    print("📌 예제 1: 기본 메시지 타입")
    print("=" * 70)

    # 세 가지 기본 메시지 타입
    system_msg = SystemMessage(content="당신은 친절한 AI 어시스턴트입니다.")
    human_msg = HumanMessage(content="안녕하세요!")
    ai_msg = AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")

    print("\n📨 메시지 타입:")
    print(f"\n1️⃣ SystemMessage:")
    print(f"   Type: {system_msg.type}")
    print(f"   Content: {system_msg.content}")

    print(f"\n2️⃣ HumanMessage:")
    print(f"   Type: {human_msg.type}")
    print(f"   Content: {human_msg.content}")

    print(f"\n3️⃣ AIMessage:")
    print(f"   Type: {ai_msg.type}")
    print(f"   Content: {ai_msg.content}")

    print("\n💡 각 메시지는 역할(role)과 내용(content)을 가집니다.\n")


# ============================================================================
# 예제 2: 메시지로 대화 구성하기
# ============================================================================

def example_2_building_conversation():
    """메시지를 사용하여 실제 대화 만들기"""
    print("=" * 70)
    print("📌 예제 2: 메시지로 대화 구성하기")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 시스템 메시지로 AI의 역할 정의
    messages = [
        SystemMessage(content="당신은 파이썬 프로그래밍 전문가입니다. 간단명료하게 답변합니다."),
        HumanMessage(content="리스트 컴프리헨션이 뭔가요?"),
    ]

    print("\n💬 대화 구성:")
    print(f"   System: {messages[0].content}")
    print(f"   Human: {messages[1].content}")

    # LLM 호출
    response = model.invoke(messages)

    print(f"\n🤖 AI 응답:")
    print(f"   {response.content}")

    print(f"\n📊 응답 타입: {type(response).__name__}")
    print(f"📊 응답 role: {response.type}\n")


# ============================================================================
# 예제 3: 딕셔너리 포맷 vs 메시지 객체
# ============================================================================

def example_3_dict_vs_objects():
    """두 가지 메시지 표현 방식 비교"""
    print("=" * 70)
    print("📌 예제 3: 딕셔너리 포맷 vs 메시지 객체")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 방법 1: 메시지 객체 사용
    messages_objects = [
        SystemMessage(content="당신은 친절한 번역가입니다."),
        HumanMessage(content="Hello를 한국어로 번역해주세요."),
    ]

    # 방법 2: 딕셔너리 사용 (간단한 경우)
    messages_dicts = [
        {"role": "system", "content": "당신은 친절한 번역가입니다."},
        {"role": "user", "content": "Hello를 한국어로 번역해주세요."},
    ]

    print("\n📋 방법 1: 메시지 객체")
    print(f"   {messages_objects}")
    response1 = model.invoke(messages_objects)
    print(f"   응답: {response1.content}")

    print("\n📋 방법 2: 딕셔너리")
    print(f"   {messages_dicts}")
    response2 = model.invoke(messages_dicts)
    print(f"   응답: {response2.content}")

    print("\n💡 둘 다 동일하게 작동하지만, 객체 방식이 더 많은 기능 제공!\n")


# ============================================================================
# 예제 4: 메시지 메타데이터와 속성
# ============================================================================

def example_4_message_metadata():
    """메시지의 추가 속성과 메타데이터 활용"""
    print("=" * 70)
    print("📌 예제 4: 메시지 메타데이터와 속성")
    print("=" * 70)

    # 메타데이터가 있는 메시지
    message_with_metadata = HumanMessage(
        content="중요한 질문입니다.",
        additional_kwargs={"priority": "high", "user_id": "12345"},
    )

    print("\n📦 메시지 속성:")
    print(f"   Content: {message_with_metadata.content}")
    print(f"   Type: {message_with_metadata.type}")
    print(f"   Additional kwargs: {message_with_metadata.additional_kwargs}")

    # ID로 메시지 추적 가능
    print(f"   Message ID: {message_with_metadata.id}")

    # 실제 대화에서 사용
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke([message_with_metadata])

    print(f"\n🤖 응답:")
    print(f"   Content: {response.content}")
    print(f"   Response ID: {response.id}")
    print(f"   Response metadata: {response.response_metadata}")

    print("\n💡 메타데이터로 메시지 추적, 우선순위 설정 등 가능!\n")


# ============================================================================
# 예제 5: 멀티턴 대화 (Multi-turn Conversation)
# ============================================================================

def example_5_multiturn_conversation():
    """여러 턴의 대화를 메시지로 구성"""
    print("=" * 70)
    print("📌 예제 5: 멀티턴 대화")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 대화 이력을 메시지 리스트로 관리
    conversation = [
        SystemMessage(content="당신은 수학 선생님입니다."),
    ]

    print("\n💬 대화 시작:")
    print(f"   System: {conversation[0].content}\n")

    # Turn 1
    conversation.append(HumanMessage(content="2 + 2는 얼마인가요?"))
    print(f"   👤 학생: {conversation[-1].content}")

    response1 = model.invoke(conversation)
    conversation.append(response1)
    print(f"   🤖 선생님: {response1.content}\n")

    # Turn 2 (이전 대화 기억)
    conversation.append(HumanMessage(content="그럼 여기에 3을 더하면요?"))
    print(f"   👤 학생: {conversation[-1].content}")

    response2 = model.invoke(conversation)
    conversation.append(response2)
    print(f"   🤖 선생님: {response2.content}\n")

    # Turn 3 (계속 대화 기억)
    conversation.append(HumanMessage(content="처음 답에서 1을 빼면요?"))
    print(f"   👤 학생: {conversation[-1].content}")

    response3 = model.invoke(conversation)
    conversation.append(response3)
    print(f"   🤖 선생님: {response3.content}\n")

    print(f"💡 전체 메시지 수: {len(conversation)}개")
    print("💡 모든 대화 이력이 메시지 리스트에 저장되어 문맥 유지!\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    print("\n🎓 Part 2: LangChain 기초 - Messages\n")

    example_1_basic_messages()
    input("⏎ 계속하려면 Enter...")

    example_2_building_conversation()
    input("⏎ 계속하려면 Enter...")

    example_3_dict_vs_objects()
    input("⏎ 계속하려면 Enter...")

    example_4_message_metadata()
    input("⏎ 계속하려면 Enter...")

    example_5_multiturn_conversation()

    print("=" * 70)
    print("🎉 Messages 학습 완료!")
    print("📖 다음: 03_tools_basic.py - Tool 기초")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 메시지 타입:
#    - SystemMessage: AI의 역할/행동 지침
#    - HumanMessage: 사용자 입력
#    - AIMessage: AI의 응답
#    - ToolMessage: 도구 실행 결과 (Part 3에서 학습)
#
# 2. 메시지 속성:
#    - content: 메시지 내용
#    - type: 메시지 타입 (system, human, ai)
#    - additional_kwargs: 추가 메타데이터
#    - id: 고유 식별자
#
# 3. 대화 이력 관리:
#    - 메시지 리스트로 전체 대화 유지
#    - SystemMessage는 보통 맨 앞에
#    - 턴마다 HumanMessage와 AIMessage 추가
#
# 4. 실전 팁:
#    - SystemMessage로 일관된 AI 성격 유지
#    - 메타데이터로 메시지 추적 및 필터링
#    - 너무 긴 대화는 메모리 문제 (Part 4에서 해결)
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: SystemMessage가 무시되는 것 같아요
# 해결: SystemMessage는 반드시 메시지 리스트의 맨 앞에 위치해야 합니다
#
# 문제: 대화 이력이 너무 길어서 오류 발생
# 해결: 토큰 제한 고려, 오래된 메시지 제거 또는 요약 (Part 4 참조)
#
# 문제: AIMessage를 직접 만들어도 되나요?
# 해결: 네! 미리 작성된 대화 예시를 만들 때 유용합니다 (Few-shot learning)
#
# ============================================================================
