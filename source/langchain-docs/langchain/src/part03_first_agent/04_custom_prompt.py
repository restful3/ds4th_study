"""
================================================================================
LangChain AI Agent 마스터 교안
Part 3: 첫 번째 Agent 만들기
================================================================================

파일명: 04_custom_prompt.py
난이도: ⭐⭐☆☆☆ (초급)
예상 시간: 20분

📚 학습 목표:
  - System Prompt의 중요성과 역할 이해
  - Default vs Custom Prompt 비교
  - 역할 기반 Prompt 작성 (선생님, 과학자, 코미디언)
  - 제약사항과 가이드라인 명시
  - 도메인별 전문 Prompt 작성

📖 공식 문서:
  • System Prompt: /official/06-agents.md (라인 242-283)
  • Quickstart: /official/03-quickstart.md (라인 57-69)

📄 교안 문서:
  • Part 3 개요: /docs/part03_first_agent.md (섹션 4)

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 04_custom_prompt.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 공통 도구 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 조회합니다.

    Args:
        city: 도시 이름 (예: 서울, 부산, 뉴욕)
    """
    weather_data = {
        "서울": "맑음, 22°C, 습도 60%",
        "부산": "흐림, 20°C, 습도 70%",
        "뉴욕": "비, 15°C, 습도 85%",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")


# ============================================================================
# 예제 1: Default vs Custom System Prompt 비교
# ============================================================================

def example_1_default_vs_custom():
    """System Prompt 없음 vs 있음 비교"""
    print("=" * 70)
    print("📌 예제 1: Default vs Custom System Prompt")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Agent 1: System Prompt 없음 (기본)
    agent_default = create_agent(
        model=model,
        tools=[get_weather],
        # system_prompt 없음
    )

    # Agent 2: Custom System Prompt 있음
    agent_custom = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="""당신은 친근하고 다정한 날씨 안내원입니다.

항상 이모티콘을 사용하여 밝고 긍정적으로 답변하세요.
날씨가 좋으면 "외출하기 좋은 날씨네요!", 날씨가 나쁘면 "실내 활동을 추천드려요!" 같은 조언을 추가하세요."""
    )

    # 테스트
    question = {"messages": [{"role": "user", "content": "서울 날씨 어때?"}]}

    print("\n🔹 Agent 1 (System Prompt 없음):")
    result1 = agent_default.invoke(question)
    print(f"답변: {result1['messages'][-1].content}")

    print("\n🔹 Agent 2 (Custom System Prompt):")
    result2 = agent_custom.invoke(question)
    print(f"답변: {result2['messages'][-1].content}")

    print("\n💡 핵심 차이:")
    print("  - Agent 1: 정보만 전달하는 중립적 답변")
    print("  - Agent 2: 친근하고 조언을 포함한 답변")
    print("  - System Prompt로 Agent의 성격이 완전히 달라집니다!\n")


# ============================================================================
# 예제 2: 역할 기반 Prompt (선생님, 과학자, 코미디언)
# ============================================================================

def example_2_role_based_prompts():
    """같은 도구, 다른 역할의 Agent들"""
    print("=" * 70)
    print("📌 예제 2: 역할 기반 Prompt - 3가지 페르소나")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Persona 1: 초등학교 선생님
    teacher_prompt = """당신은 초등학교 선생님입니다.

성격:
- 아이들에게 설명하듯이 쉽고 친절하게 말합니다
- 어려운 용어는 피하고 비유를 많이 사용합니다
- "~구나", "~란다" 같은 부드러운 말투를 사용합니다

예시:
"온도가 22도란다. 반팔 입기 딱 좋은 날씨구나!" """

    # Persona 2: 기상학자
    scientist_prompt = """당신은 전문 기상학자입니다.

성격:
- 과학적이고 정확한 용어를 사용합니다
- 기온, 습도, 기압 등을 상세히 설명합니다
- 날씨 현상의 원인을 과학적으로 분석합니다

예시:
"현재 기온은 섭씨 22도이며, 상대습도는 60%입니다. 고기압의 영향으로..." """

    # Persona 3: 코미디언
    comedian_prompt = """당신은 유머러스한 코미디언입니다.

성격:
- 날씨를 재치있고 웃기게 표현합니다
- 말장난과 과장을 즐겨 사용합니다
- 긍정적이고 에너지 넘치는 톤입니다

예시:
"오늘 날씨는 '짱-창'해요! 태양님이 완전 '빛-나'고 계시네요!" """

    # Agent 생성
    agent_teacher = create_agent(model, tools=[get_weather], system_prompt=teacher_prompt)
    agent_scientist = create_agent(model, tools=[get_weather], system_prompt=scientist_prompt)
    agent_comedian = create_agent(model, tools=[get_weather], system_prompt=comedian_prompt)

    # 테스트
    question = {"messages": [{"role": "user", "content": "서울 날씨 알려줘"}]}

    print("\n👩‍🏫 선생님 Agent:")
    result1 = agent_teacher.invoke(question)
    print(f"{result1['messages'][-1].content}")

    print("\n🔬 기상학자 Agent:")
    result2 = agent_scientist.invoke(question)
    print(f"{result2['messages'][-1].content}")

    print("\n😄 코미디언 Agent:")
    result3 = agent_comedian.invoke(question)
    print(f"{result3['messages'][-1].content}")

    print("\n💡 핵심 포인트:")
    print("  - 같은 도구, 같은 데이터로 완전히 다른 답변!")
    print("  - System Prompt가 Agent의 '정체성'을 만듭니다")
    print("  - 타겟 사용자에 맞춰 적절한 페르소나를 선택하세요\n")


# ============================================================================
# 예제 3: 제약사항과 가이드라인 추가
# ============================================================================

def example_3_constraints_and_guidelines():
    """제약사항으로 Agent 행동 제어하기"""
    print("=" * 70)
    print("📌 예제 3: 제약사항과 가이드라인")
    print("=" * 70)

    @tool
    def calculate(expression: str) -> str:
        """수식을 계산합니다."""
        try:
            result = eval(expression)
            return f"계산 결과: {result}"
        except Exception as e:
            return f"계산 오류: {str(e)}"

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 제약사항이 많은 Prompt
    constrained_prompt = """당신은 안전한 계산기 Agent입니다.

역할:
- 사용자가 요청한 수학 계산을 수행합니다
- 항상 calculate 도구를 사용하여 계산합니다

제약사항 (반드시 지켜야 함):
1. 절대 임의로 답을 추측하지 않습니다
2. calculate 도구 없이는 답변하지 않습니다
3. 위험한 코드 실행을 요청하면 거부합니다
4. 수학 계산만 처리하고 다른 작업은 하지 않습니다

가이드라인:
- 계산 전에 수식을 사용자에게 확인시키세요
- 결과는 명확하고 간단하게 제시하세요
- 복잡한 수식은 단계별로 나눠서 계산하세요

거부 예시:
"죄송하지만, 저는 수학 계산만 도와드릴 수 있습니다." """

    agent = create_agent(model, tools=[calculate], system_prompt=constrained_prompt)

    # 테스트 1: 정상 요청
    print("\n✅ 테스트 1: 정상적인 계산 요청")
    print("👤 사용자: 25 곱하기 4는 얼마야?")
    result1 = agent.invoke({"messages": [{"role": "user", "content": "25 곱하기 4는 얼마야?"}]})
    print(f"🤖 Agent: {result1['messages'][-1].content}")

    # 테스트 2: 제약사항 위반 요청
    print("\n🚫 테스트 2: 제약사항 위반 (도구 없이 추측 요구)")
    print("👤 사용자: 대충 100 곱하기 3 정도면 얼마일까? 대략 말해줘")
    result2 = agent.invoke({"messages": [{"role": "user", "content": "대충 100 곱하기 3 정도면 얼마일까? 대략 말해줘"}]})
    print(f"🤖 Agent: {result2['messages'][-1].content}")

    print("\n💡 핵심 포인트:")
    print("  - 제약사항으로 Agent의 행동을 엄격하게 제어할 수 있습니다")
    print("  - '절대 ~하지 않습니다'같은 강력한 표현을 사용하세요")
    print("  - 금융, 의료 등 민감한 분야에서 특히 중요합니다\n")


# ============================================================================
# 예제 4: 도메인별 전문 Prompt (의료, 법률, 기술)
# ============================================================================

def example_4_domain_specific_prompts():
    """도메인 전문가 Agent 만들기"""
    print("=" * 70)
    print("📌 예제 4: 도메인별 전문 Prompt")
    print("=" * 70)

    @tool
    def search_info(query: str) -> str:
        """정보를 검색합니다."""
        # 더미 데이터
        info_db = {
            "두통": "일반적인 원인: 스트레스, 수면 부족, 탈수. 권장 조치: 충분한 휴식과 수분 섭취.",
            "계약서": "계약서 작성 시 주의사항: 계약 당사자 명시, 계약 기간, 위약 조건 등을 명확히 기재.",
            "Python": "Python은 1991년 Guido van Rossum이 개발한 고급 프로그래밍 언어입니다.",
        }
        for key, value in info_db.items():
            if key in query:
                return value
        return f"'{query}'에 대한 정보를 찾을 수 없습니다"

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 도메인 1: 의료 상담 Agent
    medical_prompt = """당신은 의료 정보 제공 Agent입니다. (주의: 실제 진단은 불가)

역할:
- 일반적인 건강 정보를 제공합니다
- 증상에 대한 일반적인 원인을 설명합니다

응답 구조 (항상 따르세요):
1. 증상 확인: 사용자가 말한 증상 요약
2. 일반적 원인: 가능한 원인 2-3가지
3. 권장 조치: 집에서 할 수 있는 조치
4. 병원 방문: 병원에 가야 하는 경우 안내

⚠️ 중요한 제약:
- 절대 확정 진단을 하지 않습니다
- 모든 답변 끝에 "정확한 진단은 의사와 상담하세요"를 추가합니다
- 심각한 증상은 즉시 병원 방문을 권유합니다"""

    # 도메인 2: 법률 자문 Agent
    legal_prompt = """당신은 법률 정보 제공 Agent입니다. (주의: 실제 법률 자문은 불가)

역할:
- 일반적인 법률 정보를 제공합니다
- 법률 문서 작성 시 주의사항을 안내합니다

응답 구조:
1. 질문 이해: 법률 문제 요약
2. 일반 원칙: 관련 법률의 일반적 내용
3. 주의사항: 특히 조심해야 할 사항
4. 전문가 상담: 변호사 상담 권유

⚠️ 중요한 제약:
- 절대 법률 조언을 하지 않습니다 (정보 제공만)
- "이것은 일반 정보이며, 법률 자문이 아닙니다"를 명시합니다
- 복잡한 사안은 변호사 상담을 권유합니다"""

    # 도메인 3: 기술 지원 Agent
    technical_prompt = """당신은 기술 지원 전문 Agent입니다.

역할:
- 프로그래밍 및 기술 질문에 답변합니다
- 명확하고 실용적인 해결책을 제시합니다

응답 구조:
1. 문제 파악: 사용자의 기술적 문제 요약
2. 해결 방법: 단계별 해결 가이드
3. 예제 코드: 필요시 코드 예시 제공
4. 추가 팁: 관련된 유용한 정보

응답 스타일:
- 전문 용어는 사용하되, 간단히 설명 추가
- 코드는 주석과 함께 제공
- 여러 해결 방법이 있으면 모두 제시"""

    # Agent 생성
    agent_medical = create_agent(model, tools=[search_info], system_prompt=medical_prompt)
    agent_legal = create_agent(model, tools=[search_info], system_prompt=legal_prompt)
    agent_technical = create_agent(model, tools=[search_info], system_prompt=technical_prompt)

    # 테스트
    print("\n🏥 의료 상담 Agent:")
    print("👤 사용자: 두통이 있어요. 어떻게 해야 하나요?")
    result1 = agent_medical.invoke({"messages": [{"role": "user", "content": "두통이 있어요. 어떻게 해야 하나요?"}]})
    print(f"🤖 Agent:\n{result1['messages'][-1].content}\n")

    print("-" * 70)

    print("\n⚖️ 법률 자문 Agent:")
    print("👤 사용자: 계약서 작성할 때 뭘 주의해야 하나요?")
    result2 = agent_legal.invoke({"messages": [{"role": "user", "content": "계약서 작성할 때 뭘 주의해야 하나요?"}]})
    print(f"🤖 Agent:\n{result2['messages'][-1].content}\n")

    print("-" * 70)

    print("\n💻 기술 지원 Agent:")
    print("👤 사용자: Python이 뭔가요?")
    result3 = agent_technical.invoke({"messages": [{"role": "user", "content": "Python이 뭔가요?"}]})
    print(f"🤖 Agent:\n{result3['messages'][-1].content}")

    print("\n💡 핵심 포인트:")
    print("  - 도메인별로 특화된 Prompt를 작성하세요")
    print("  - 응답 구조를 명시하면 일관된 품질을 유지할 수 있습니다")
    print("  - 책임 제한 문구를 반드시 포함하세요 (의료, 법률 등)\n")


# ============================================================================
# 예제 5: 다국어 Prompt (한국어, 영어, 일본어)
# ============================================================================

def example_5_multilingual_prompts():
    """다국어 지원 Agent"""
    print("=" * 70)
    print("📌 예제 5: 다국어 Prompt")
    print("=" * 70)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 다국어 Prompt
    multilingual_prompt = """당신은 다국어 날씨 안내 Agent입니다.

언어 감지:
- 사용자의 질문 언어를 자동으로 감지합니다
- 같은 언어로 답변합니다

지원 언어:
- 한국어: 존댓말 사용, 친근한 톤
- English: Professional and clear tone
- 日本語: 丁寧な言葉遣い

답변 형식:
1. 인사 (해당 언어로)
2. 날씨 정보
3. 마무리 인사

예시:
- 한국어: "안녕하세요! 서울은 맑고 22도입니다. 좋은 하루 되세요!"
- English: "Hello! Seoul is sunny and 22°C. Have a great day!"
- 日本語: "こんにちは！ソウルは晴れで22度です。良い一日を！" """

    agent = create_agent(model, tools=[get_weather], system_prompt=multilingual_prompt)

    # 다국어 테스트
    questions = [
        ("한국어", "서울 날씨 알려줘"),
        ("English", "What's the weather in Seoul?"),
        ("日本語", "ソウルの天気を教えてください"),
    ]

    for lang, question in questions:
        print(f"\n🌍 {lang}:")
        print(f"👤 사용자: {question}")
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(f"🤖 Agent: {result['messages'][-1].content}")

    print("\n💡 핵심 포인트:")
    print("  - LLM은 다국어를 자연스럽게 처리할 수 있습니다")
    print("  - Prompt에 각 언어별 톤과 스타일을 명시하세요")
    print("  - 글로벌 서비스에서 매우 유용합니다\n")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 3: 첫 번째 Agent 만들기 - System Prompt 커스터마이징")
    print("\n")

    # 모든 예제 실행
    example_1_default_vs_custom()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_role_based_prompts()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_constraints_and_guidelines()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_domain_specific_prompts()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_multilingual_prompts()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 System Prompt 커스터마이징 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 05_streaming_agent.py - 실시간 스트리밍 구현")
    print("  2. Part 4: Memory & Context Management")
    print("\n📚 System Prompt 작성 팁:")
    print("  • 역할을 명확히 정의하세요")
    print("  • 제약사항은 강력한 표현으로 ('절대', '반드시')")
    print("  • 응답 구조를 명시하면 일관성이 향상됩니다")
    print("  • 도메인별 전문 용어와 톤을 사용하세요")
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
# 1. 좋은 System Prompt의 특징:
#    - 명확성: 역할과 책임이 분명함
#    - 구체성: "친절하게"보다 "초등학생에게 설명하듯이"
#    - 일관성: 응답 형식을 명시
#    - 안전성: 제약사항과 책임 제한 포함
#
# 2. System Prompt 구조 템플릿:
#    """당신은 [역할]입니다.
#
#    역할:
#    - [주요 책임 1]
#    - [주요 책임 2]
#
#    성격/톤:
#    - [말투 스타일]
#    - [응답 스타일]
#
#    응답 구조:
#    1. [첫 번째 단계]
#    2. [두 번째 단계]
#
#    제약사항:
#    - 절대 [금지 사항 1]
#    - 반드시 [필수 사항 1]
#    """
#
# 3. Temperature 설정:
#    - 0.0: 일관적이고 결정적 (금융, 의료)
#    - 0.3-0.5: 균형잡힌 (일반적 사용)
#    - 0.7-1.0: 창의적 (마케팅, 크리에이티브)
#
# 4. Prompt Engineering 베스트 프랙티스:
#    - Few-shot learning: 예시를 포함
#    - Chain-of-Thought: "단계적으로 생각하세요"
#    - Role-playing: 특정 역할 부여
#    - Constraints: 명확한 제약 조건
#
# 5. A/B 테스트:
#    - 여러 Prompt 버전을 테스트하세요
#    - 사용자 피드백을 수집하세요
#    - 지속적으로 개선하세요
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: Prompt를 무시하고 다르게 행동함
# 해결:
#   - "반드시", "절대" 같은 강력한 표현 사용
#   - temperature를 낮춰서 일관성 향상
#   - 더 강력한 모델 사용 (gpt-4o)
#
# 문제: Prompt가 너무 길어서 비용이 많이 듦
# 해결:
#   - 핵심만 남기고 불필요한 설명 제거
#   - 반복되는 내용 통합
#   - 예시는 2-3개만 포함
#
# 문제: 다국어 지원이 안 됨
# 해결:
#   - Prompt 자체를 여러 언어로 작성
#   - "사용자 언어로 답변하세요" 명시
#   - gpt-4o 같은 다국어 성능 좋은 모델 사용
#
# 문제: Agent가 제약사항을 어김
# 해결:
#   - 제약사항을 더 구체적으로 명시
#   - 거부 예시 포함
#   - Middleware로 추가 검증 (Part 5에서 학습)
#
# ============================================================================
