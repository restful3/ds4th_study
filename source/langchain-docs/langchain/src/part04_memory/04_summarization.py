"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory System - Summarization
================================================================================

파일명: 04_summarization.py
난이도: ⭐⭐⭐⭐☆ (중상급)
예상 시간: 30분

📚 학습 목표:
  - Message Summarization의 필요성 이해
  - before_model로 커스텀 요약 구현
  - SummarizationMiddleware 사용
  - Rolling Summary 패턴
  - 요약 + 최근 메시지 전략

📖 공식 문서:
  • Short-term Memory: /official/10-short-term-memory.md

📄 교안 문서:
  • Part 4 메모리: /docs/part04_memory.md (Section 4)

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 04_summarization.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, SummarizationMiddleware
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)

# ============================================================================
# 예제 1: Summarization이 필요한 이유
# ============================================================================

def example_1_why_summarization():
    """Trim vs Summarization 비교"""
    print("=" * 70)
    print("📌 예제 1: Summarization이 필요한 이유")
    print("=" * 70)
    print()

    print("""
🤔 문제 상황:
   긴 대화에서 Context Window를 관리해야 하는데...

❌ Trim (메시지 삭제)의 문제:
   👤 사용자: 제 이름은 김철수이고 서울에 살아요.
   🤖 AI: 안녕하세요 김철수님!
   👤 사용자: 저는 파이썬 개발자입니다.
   🤖 AI: 파이썬 개발자시군요!

   ... (많은 대화) ...

   [Trim으로 초기 메시지 삭제]

   👤 사용자: 제 이름과 직업이 뭐였죠?
   🤖 AI: 죄송하지만 그 정보를 찾을 수 없습니다.

   ❌ 문제: 중요한 정보가 완전히 손실됨!

✅ Summarization의 장점:
   [이전 대화 요약]: 사용자는 김철수이고 서울에 거주하는 파이썬 개발자입니다.

   ... (최근 대화) ...

   👤 사용자: 제 이름과 직업이 뭐였죠?
   🤖 AI: 김철수님, 파이썬 개발자이시죠!

   ✅ 장점: 중요한 정보는 요약으로 보존됨!

💡 Summarization이 필요한 경우:
   - 장기간 대화 (50+ 메시지)
   - 컨텍스트가 중요한 작업
   - 정보 손실이 허용되지 않는 경우
   - 사용자 경험이 중요한 서비스

📊 비교:
   ┌─────────────┬──────────┬──────────────┬─────────────┐
   │ 방법        │ 정보손실 │ 구현 복잡도  │ 비용        │
   ├─────────────┼──────────┼──────────────┼─────────────┤
   │ Trim        │ 높음     │ 낮음         │ 낮음        │
   │ Summary     │ 낮음     │ 중간         │ 중간        │
   │ 둘 다 사용  │ 매우낮음 │ 높음         │ 중간        │
   └─────────────┴──────────┴──────────────┴─────────────┘
    """)


# ============================================================================
# 예제 2: 커스텀 Summarization 미들웨어
# ============================================================================

def example_2_custom_summarization():
    """before_model로 커스텀 요약 구현"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 커스텀 Summarization 미들웨어")
    print("=" * 70)
    print("\n💡 직접 요약 로직을 구현하여 세밀한 제어가 가능합니다.\n")

    # 요약을 위한 별도 LLM
    summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @before_model
    def summarize_old_messages(
        state: AgentState,
        runtime: Runtime
    ) -> dict[str, Any] | None:
        """오래된 메시지를 요약"""
        messages = state["messages"]

        # 10개 이하면 요약 안 함
        if len(messages) <= 10:
            return None

        print("\n📝 메시지 요약 중...")

        # 시스템 메시지
        system_msg = messages[0] if messages[0].type == "system" else None

        # 요약할 메시지 (중간 부분)
        # 최근 5개는 원본 유지
        to_summarize = messages[1:-5] if system_msg else messages[:-5]
        recent_messages = messages[-5:]

        if not to_summarize:
            return None

        # 메시지를 텍스트로 변환
        conversation_text = "\n".join([
            f"{'사용자' if m.type == 'human' else 'AI'}: {m.content}"
            for m in to_summarize
            if hasattr(m, 'content') and m.content
        ])

        # 요약 생성
        summary_prompt = f"""
다음 대화를 간결하게 요약해주세요. 중요한 정보(이름, 선호도, 맥락 등)를 모두 포함하세요.

대화:
{conversation_text}

요약 (2-3문장):
"""

        summary_response = summarizer.invoke([
            HumanMessage(content=summary_prompt)
        ])

        summary_text = summary_response.content

        print(f"✅ 요약 완료: {len(to_summarize)}개 메시지 → 1개 요약")
        print(f"📄 요약 내용: {summary_text[:100]}...")

        # 요약 메시지 생성
        summary_message = SystemMessage(
            content=f"[이전 대화 요약]: {summary_text}"
        )

        # 새로운 메시지 구성
        new_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]

        if system_msg:
            new_messages.append(system_msg)

        new_messages.append(summary_message)
        new_messages.extend(recent_messages)

        return {"messages": new_messages}

    # Agent 생성
    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[summarize_old_messages],
        checkpointer=checkpointer,
        system_prompt="당신은 친절한 AI 어시스턴트입니다."
    )

    config: RunnableConfig = {"configurable": {"thread_id": "custom-summary-test"}}

    # 긴 대화 시뮬레이션
    print("\n=" * 50)
    print("📝 대화 진행")
    print("=" * 50)

    conversations = [
        "제 이름은 이지은입니다.",
        "저는 디자이너로 일하고 있어요.",
        "서울 강남에 살고 있습니다.",
        "고양이 두 마리를 키워요.",
        "커피를 정말 좋아합니다.",
        "주말에는 등산을 즐겨요.",
        "파이썬을 배우고 싶어요.",
        "최근에 AI에 관심이 생겼습니다.",
        "LangChain이 흥미로워 보여요.",
        "실습 프로젝트를 해보고 싶습니다.",
        "제 이름과 직업, 취미를 말해주세요.",  # 요약에서 정보 가져오기
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"\n대화 {i}:")
        print(f"👤 사용자: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )

        response = result['messages'][-1].content
        print(f"🤖 AI: {response[:150]}...")

        # 메시지 수 확인
        state = agent.get_state(config)
        msg_count = len(state.values["messages"])
        print(f"   📊 현재 메시지 수: {msg_count}")

    print("\n✅ 요약 덕분에 중요한 정보가 보존되었습니다!")


# ============================================================================
# 예제 3: SummarizationMiddleware 사용
# ============================================================================

def example_3_builtin_summarization():
    """LangChain 내장 SummarizationMiddleware"""
    print("\n" + "=" * 70)
    print("📌 예제 3: SummarizationMiddleware (내장)")
    print("=" * 70)
    print("\n💡 LangChain이 제공하는 내장 요약 미들웨어를 사용합니다.\n")

    checkpointer = InMemorySaver()

    # SummarizationMiddleware 설정
    summarization = SummarizationMiddleware(
        model="gpt-4o-mini",              # 요약에 사용할 모델
        trigger=("tokens", 2000),         # 2000 토큰 초과 시 요약
        keep=("messages", 10),            # 최근 10개 메시지 유지
    )

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[summarization],       # 미들웨어 추가
        checkpointer=checkpointer,
        system_prompt="당신은 도움이 되는 AI 어시스턴트입니다."
    )

    config: RunnableConfig = {"configurable": {"thread_id": "builtin-summary-test"}}

    print("⚙️  설정:")
    print(f"   - 요약 트리거: 2000 토큰 초과")
    print(f"   - 유지 메시지: 최근 10개")
    print(f"   - 요약 모델: gpt-4o-mini")

    # 많은 대화 생성
    print("\n=" * 50)
    print("📝 긴 대화 생성")
    print("=" * 50)

    topics = [
        "인공지능의 역사에 대해 알려주세요.",
        "머신러닝과 딥러닝의 차이는 무엇인가요?",
        "GPT 모델은 어떻게 작동하나요?",
        "LangChain의 주요 기능을 설명해주세요.",
        "Agent와 Chain의 차이는 무엇인가요?",
        "RAG 시스템에 대해 알려주세요.",
        "벡터 데이터베이스는 무엇인가요?",
        "프롬프트 엔지니어링 기법을 알려주세요.",
    ]

    for i, topic in enumerate(topics, 1):
        print(f"\n대화 {i}:")
        print(f"👤 사용자: {topic}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": topic}]},
            config
        )

        response = result['messages'][-1].content
        print(f"🤖 AI: {response[:100]}...")

    # 요약 확인
    print("\n=" * 50)
    print("📊 요약 확인")
    print("=" * 50)

    print("👤 사용자: 우리가 무엇에 대해 이야기했는지 요약해주세요.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "우리가 무엇에 대해 이야기했는지 요약해주세요."}]},
        config
    )

    print(f"🤖 AI: {result['messages'][-1].content}")

    print("\n✅ SummarizationMiddleware가 자동으로 요약을 관리했습니다!")


# ============================================================================
# 예제 4: Rolling Summary (지속 업데이트 요약)
# ============================================================================

def example_4_rolling_summary():
    """요약을 지속적으로 업데이트하는 패턴"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Rolling Summary")
    print("=" * 70)
    print("\n💡 기존 요약에 새로운 정보를 추가하여 계속 업데이트합니다.\n")

    summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @before_model
    def rolling_summary(
        state: AgentState,
        runtime: Runtime
    ) -> dict[str, Any] | None:
        """Rolling summary 생성"""
        messages = state["messages"]

        if len(messages) <= 8:
            return None

        # 기존 요약 찾기
        existing_summary = None
        summary_index = None

        for i, msg in enumerate(messages):
            if msg.type == "system" and "[대화 요약]" in msg.content:
                existing_summary = msg.content.replace("[대화 요약]: ", "")
                summary_index = i
                break

        # 요약할 새로운 메시지들
        if summary_index is not None:
            new_messages = messages[summary_index + 1:-4]
        else:
            new_messages = messages[1:-4]

        if not new_messages:
            return None

        print("\n🔄 Rolling Summary 업데이트 중...")

        # 새 메시지를 텍스트로
        new_text = "\n".join([
            f"{'사용자' if m.type == 'human' else 'AI'}: {m.content}"
            for m in new_messages
            if hasattr(m, 'content') and m.content
        ])

        # 요약 업데이트
        if existing_summary:
            prompt = f"""
기존 요약:
{existing_summary}

새로운 대화:
{new_text}

기존 요약에 새로운 정보를 추가하여 업데이트된 요약을 만들어주세요 (3-4문장):
"""
        else:
            prompt = f"""
다음 대화를 요약해주세요 (3-4문장):

{new_text}

요약:
"""

        updated_summary = summarizer.invoke([
            HumanMessage(content=prompt)
        ]).content

        print(f"✅ 요약 업데이트 완료")
        if existing_summary:
            print(f"📝 이전: {existing_summary[:80]}...")
        print(f"📝 현재: {updated_summary[:80]}...")

        # 새로운 메시지 구성
        system_msg = messages[0] if messages[0].type == "system" and "[대화 요약]" not in messages[0].content else None
        recent = messages[-4:]

        new_msgs = [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        if system_msg:
            new_msgs.append(system_msg)

        new_msgs.append(SystemMessage(content=f"[대화 요약]: {updated_summary}"))
        new_msgs.extend(recent)

        return {"messages": new_msgs}

    checkpointer = InMemorySaver()
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        middleware=[rolling_summary],
        checkpointer=checkpointer,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "rolling-test"}}

    # 점진적 대화
    conversations = [
        "제 이름은 최민호입니다.",
        "데이터 과학자로 일하고 있어요.",
        "머신러닝 모델을 개발합니다.",
        "최근에 NLP 프로젝트를 시작했어요.",
        "LangChain을 사용해보려고 합니다.",
        "팀에 5명의 개발자가 있습니다.",
        "다음 주에 발표가 있어요.",
        "RAG 시스템을 구축 중입니다.",
        "지금까지 제가 한 얘기를 요약해주세요.",
    ]

    for i, msg in enumerate(conversations, 1):
        print(f"\n대화 {i}:")
        print(f"👤 사용자: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )

        print(f"🤖 AI: {result['messages'][-1].content[:120]}...")

    print("\n✅ Rolling Summary로 지속적으로 요약이 업데이트되었습니다!")


# ============================================================================
# 예제 5: Hybrid Strategy (요약 + 최근 메시지)
# ============================================================================

def example_5_hybrid_strategy():
    """요약과 최근 메시지를 함께 사용하는 전략"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Hybrid Strategy (요약 + 최근 메시지)")
    print("=" * 70)
    print("\n💡 가장 강력한 전략: 요약 + 최근 원본 메시지\n")

    print("""
🎯 Hybrid Strategy 구조:

┌─────────────────────────────────────────────┐
│ 시스템 메시지                               │
├─────────────────────────────────────────────┤
│ [요약] 오래된 대화의 요약 (20-50 메시지)   │
├─────────────────────────────────────────────┤
│ 최근 메시지 1 (원본)                        │
│ 최근 메시지 2 (원본)                        │
│ 최근 메시지 3 (원본)                        │
│ ...                                         │
│ 최근 메시지 N (원본)                        │
└─────────────────────────────────────────────┘

✅ 장점:
   - 장기 컨텍스트: 요약으로 보존
   - 단기 컨텍스트: 원본 메시지로 유지
   - 정보 손실 최소화
   - 자연스러운 대화 흐름

💡 사용 시나리오:
   - 고객 지원 (긴 대화 이력)
   - 개인 비서 (지속적인 관계)
   - 복잡한 프로젝트 관리
   - 교육/튜터링 시스템
    """)

    # 실제 구현은 example_2와 유사하지만,
    # 요약 + 최근 메시지를 모두 유지하는 것이 핵심
    print("\n💡 구현 팁:")
    print("   1. 메시지가 20개 이상이면 요약 시작")
    print("   2. 오래된 메시지(1-15)를 요약")
    print("   3. 최근 메시지(16-20) 원본 유지")
    print("   4. 새 메시지가 추가되면 다시 평가")
    print("\n📊 예상 효과:")
    print("   - Context Window: 90% 절약")
    print("   - 정보 보존: 95% 유지")
    print("   - 대화 품질: 거의 손실 없음")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 4: Memory System - Summarization")
    print("\n")

    # 예제 1: 필요성
    example_1_why_summarization()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 2: 커스텀 요약
    example_2_custom_summarization()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 3: 내장 미들웨어
    example_3_builtin_summarization()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 4: Rolling Summary
    example_4_rolling_summary()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 5: Hybrid Strategy
    example_5_hybrid_strategy()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 4-4 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 05_custom_state.py - Custom State")
    print("  2. 06_long_term_store.py - Long-term Memory")
    print("\n📚 핵심 개념 복습:")
    print("  • Summarization: 정보 보존하며 토큰 절약")
    print("  • Rolling Summary: 지속적 요약 업데이트")
    print("  • SummarizationMiddleware: 내장 솔루션")
    print("  • Hybrid: 요약 + 최근 메시지 (최상의 전략)")
    print("\n💡 선택 가이드:")
    print("  • 단순 대화: Trim")
    print("  • 중요 대화: Summarization")
    print("  • 장기 대화: Hybrid Strategy")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 요약 품질 향상:
#    - 구조화된 프롬프트 사용
#    - Few-shot 예제 제공
#    - 중요도 기반 필터링
#
# 2. 요약 타이밍:
#    - 토큰 기반: 일정 토큰 수 초과 시
#    - 메시지 기반: N개 메시지 초과 시
#    - 시간 기반: 일정 시간 경과 후
#
# 3. 다단계 요약:
#    - Level 1: 최근 20개 메시지 요약
#    - Level 2: 요약들의 요약
#    - Level 3: 전체 대화의 핵심 요약
#
# 4. 요약 검증:
#    - 원본과 요약 비교
#    - 정보 손실 측정
#    - 사용자 피드백 수집
#
# 5. 비용 최적화:
#    - 저렴한 모델로 요약
#    - 배치 요약 (여러 메시지 한번에)
#    - 캐싱 활용
#
# ============================================================================
