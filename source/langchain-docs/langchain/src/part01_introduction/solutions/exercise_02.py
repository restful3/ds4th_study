"""
================================================================================
LangChain AI Agent 마스터 교안
Part 1: AI Agent의 이해 - 실습 과제 2 해답
================================================================================

과제: 온도(Temperature) 파라미터 실험

요구사항:
1. 같은 질문에 대해 다른 온도 값으로 응답 생성
2. Temperature 0.0, 0.5, 1.0, 1.5, 2.0으로 테스트
3. 각 온도에서의 응답 차이를 관찰하고 기록
4. 결과를 비교하여 출력

================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경 설정
load_dotenv()

# ============================================================================
# 솔루션: Temperature 실험
# ============================================================================

def test_temperature(question: str, temperatures=[0.0, 0.5, 1.0, 1.5, 2.0]):
    """
    다양한 temperature 값으로 응답을 생성하고 비교합니다.

    Args:
        question: 테스트할 질문
        temperatures: 테스트할 temperature 값 리스트
    """

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    print("=" * 80)
    print("🌡️ Temperature 실험")
    print("=" * 80)
    print()
    print(f"📝 질문: {question}")
    print()
    print("💡 Temperature란?")
    print("   - 0에 가까울수록: 결정적, 일관된 응답")
    print("   - 1에 가까울수록: 균형잡힌 창의성")
    print("   - 2에 가까울수록: 매우 창의적, 예측 불가능")
    print()

    responses = {}

    for temp in temperatures:
        print("=" * 80)
        print(f"🌡️ Temperature: {temp}")
        print("=" * 80)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temp,
        )

        try:
            # 같은 질문을 3번 반복해서 일관성 확인
            print("\n[3번 반복 테스트]")
            for i in range(3):
                response = llm.invoke([HumanMessage(content=question)])
                print(f"\n응답 {i+1}:")
                print(response.content)

                # 첫 번째 응답만 저장
                if i == 0:
                    responses[temp] = response.content

            print()

        except Exception as e:
            print(f"❌ 오류: {e}")
            print()

    # 결과 분석
    print("\n" + "=" * 80)
    print("📊 결과 분석")
    print("=" * 80)
    print()

    print("🔍 관찰 포인트:")
    print()

    print("1️⃣ Temperature 0.0:")
    print("   - 가장 결정적인 응답")
    print("   - 3번 반복해도 거의 동일한 답변")
    print("   - 사실 기반 질문에 적합")
    print()

    print("2️⃣ Temperature 1.0:")
    print("   - 균형잡힌 창의성")
    print("   - 약간의 변화는 있지만 일관성 유지")
    print("   - 일반적인 대화에 적합")
    print()

    print("3️⃣ Temperature 2.0:")
    print("   - 매우 창의적이고 다양한 응답")
    print("   - 반복할 때마다 크게 다른 답변")
    print("   - 창작 활동에 적합")
    print()

    return responses


def compare_creativity_tasks():
    """
    창의적 작업과 사실 기반 작업에서 temperature 차이 비교
    """
    print("=" * 80)
    print("🎨 창의적 작업 vs 사실 기반 작업")
    print("=" * 80)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    # 테스트 케이스
    tasks = [
        {
            "type": "사실 기반",
            "question": "대한민국의 수도는 어디인가요?",
            "recommended_temp": 0.0,
        },
        {
            "type": "창의적",
            "question": "고양이를 주인공으로 한 짧은 시를 작성해주세요.",
            "recommended_temp": 1.5,
        },
    ]

    for task in tasks:
        print(f"\n📌 {task['type']} 작업")
        print(f"질문: {task['question']}")
        print(f"권장 Temperature: {task['recommended_temp']}")
        print("-" * 80)

        # 낮은 온도 (0.0)
        llm_low = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        response_low = llm_low.invoke([HumanMessage(content=task['question'])])

        print(f"\n🌡️ Temperature 0.0:")
        print(response_low.content)

        # 높은 온도 (1.5)
        llm_high = ChatOpenAI(model="gpt-4o-mini", temperature=1.5)
        response_high = llm_high.invoke([HumanMessage(content=task['question'])])

        print(f"\n🌡️ Temperature 1.5:")
        print(response_high.content)
        print("\n" + "=" * 80)


def find_optimal_temperature():
    """
    다양한 작업 유형에 대한 최적 temperature 찾기
    """
    print("=" * 80)
    print("🎯 작업별 최적 Temperature")
    print("=" * 80)
    print()

    recommendations = [
        {
            "task": "수학 문제 풀이",
            "temp": 0.0,
            "reason": "정확한 계산과 논리가 필요"
        },
        {
            "task": "코드 생성",
            "temp": 0.2,
            "reason": "문법 정확성이 중요, 약간의 창의성"
        },
        {
            "task": "일반 대화",
            "temp": 0.7,
            "reason": "자연스러운 대화, 적당한 다양성"
        },
        {
            "task": "브레인스토밍",
            "temp": 1.2,
            "reason": "다양한 아이디어 생성"
        },
        {
            "task": "창작 (시, 소설)",
            "temp": 1.5,
            "reason": "높은 창의성과 독창성"
        },
    ]

    for rec in recommendations:
        print(f"📋 {rec['task']}")
        print(f"   🌡️ 권장 Temperature: {rec['temp']}")
        print(f"   💡 이유: {rec['reason']}")
        print()


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""

    print("\n" + "=" * 80)
    print("✅ Part 1 실습 과제 2 - 해답")
    print("=" * 80)
    print()

    # 실험 1: 같은 질문으로 temperature 비교
    print("🔬 실험 1: Temperature 값에 따른 응답 차이\n")

    test_temperature(
        question="AI Agent가 무엇인지 2문장으로 설명해주세요.",
        temperatures=[0.0, 0.5, 1.0, 1.5]
    )

    # 실험 2: 창의적 vs 사실 기반 작업
    print("\n" + "=" * 80)
    print("🔬 실험 2: 작업 유형별 Temperature 비교\n")

    compare_creativity_tasks()

    # 가이드: 최적 temperature
    print("\n")
    find_optimal_temperature()

    # 요약
    print("=" * 80)
    print("📝 핵심 정리")
    print("=" * 80)
    print()
    print("✅ Temperature는 모델의 창의성을 조절합니다")
    print("✅ 0.0 = 결정적, 2.0 = 매우 창의적")
    print("✅ 작업 유형에 따라 적절한 값을 선택하세요")
    print("✅ 일반적으로 0.7-1.0이 대화에 적합합니다")
    print()


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 학습 포인트
# ============================================================================
#
# 1. Temperature 파라미터:
#    - 모델 출력의 무작위성 조절
#    - 0.0 = 가장 확률 높은 토큰 선택 (결정적)
#    - 2.0 = 매우 다양한 토큰 선택 (창의적)
#
# 2. 작업별 최적값:
#    - 사실 기반: 0.0-0.3
#    - 대화: 0.7-1.0
#    - 창작: 1.2-2.0
#
# 3. 실험의 중요성:
#    - 항상 여러 값으로 테스트
#    - 작업에 맞는 최적값 찾기
#    - 일관성 vs 창의성 균형
#
# ============================================================================
# 🎓 추가 실험
# ============================================================================
#
# - top_p 파라미터도 함께 실험해보기
# - 다른 모델 (Claude, Gemini)에서도 테스트
# - 더 복잡한 작업으로 실험 (코드 생성, 번역 등)
#
# ============================================================================
