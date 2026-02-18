"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory - 실습 과제 3 해답
================================================================================

과제: 사용자 프로필 시스템
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. InMemoryStore에 사용자 프로필 저장 (이름, 선호도, 관심사)
2. 대화에서 사용자 정보를 자동 추출하여 프로필 업데이트
3. 프로필 기반 개인화 응답 생성

학습 목표:
- InMemoryStore 사용법
- 구조화된 데이터 저장/조회
- Store와 Checkpointer 통합

================================================================================
"""

from dotenv import load_dotenv
from typing import Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
import json

load_dotenv()

# ============================================================================
# State 정의
# ============================================================================

class ChatState(MessagesState):
    """채팅 상태"""
    user_id: str


# ============================================================================
# 프로필 관리자
# ============================================================================

class UserProfileManager:
    """사용자 프로필 관리"""

    def __init__(self, store: InMemoryStore):
        self.store = store

    def get_profile(self, user_id: str) -> dict:
        """사용자 프로필 조회"""
        namespace = ("users", user_id)
        items = self.store.search(namespace)

        if not items:
            # 기본 프로필 생성
            default_profile = {
                "user_id": user_id,
                "name": None,
                "preferences": {},
                "interests": [],
                "conversation_count": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            self.store.put(namespace, "profile", default_profile)
            return default_profile

        return items[0].value

    def update_profile(self, user_id: str, updates: dict):
        """프로필 업데이트"""
        namespace = ("users", user_id)
        profile = self.get_profile(user_id)

        # 업데이트 적용
        for key, value in updates.items():
            if key == "preferences":
                profile["preferences"].update(value)
            elif key == "interests":
                # 중복 제거하며 추가
                for interest in value:
                    if interest not in profile["interests"]:
                        profile["interests"].append(interest)
            else:
                profile[key] = value

        profile["updated_at"] = datetime.now().isoformat()

        # 저장
        self.store.put(namespace, "profile", profile)

    def add_conversation_history(self, user_id: str, topic: str, summary: str):
        """대화 기록 추가"""
        namespace = ("users", user_id)
        key = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        conversation_record = {
            "topic": topic,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        self.store.put(namespace, key, conversation_record)

        # 대화 횟수 증가
        profile = self.get_profile(user_id)
        profile["conversation_count"] += 1
        self.update_profile(user_id, {"conversation_count": profile["conversation_count"]})

    def get_conversation_history(self, user_id: str) -> list:
        """대화 기록 조회"""
        namespace = ("users", user_id)
        items = self.store.search(namespace)

        conversations = [
            item.value for item in items
            if item.key.startswith("conversation_")
        ]

        return sorted(conversations, key=lambda x: x["timestamp"], reverse=True)


# ============================================================================
# 프로필 기반 챗봇 구축
# ============================================================================

def create_profile_chatbot() -> tuple[CompiledStateGraph, InMemoryStore]:
    """프로필 기반 챗봇 생성"""

    # LLM 모델
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Store 생성
    store = InMemoryStore()
    profile_manager = UserProfileManager(store)

    # 챗봇 노드
    def chatbot_node(state: ChatState, config: dict, store: InMemoryStore) -> dict:
        """개인화된 응답 생성"""
        user_id = state["user_id"]

        # 프로필 로드
        profile = profile_manager.get_profile(user_id)

        # 시스템 프롬프트에 프로필 정보 포함
        system_prompt = "당신은 친절한 AI 어시스턴트입니다.\n\n"

        if profile.get("name"):
            system_prompt += f"사용자 이름: {profile['name']}\n"

        if profile.get("preferences"):
            system_prompt += "사용자 선호도:\n"
            for key, value in profile["preferences"].items():
                system_prompt += f"  - {key}: {value}\n"

        if profile.get("interests"):
            system_prompt += f"관심사: {', '.join(profile['interests'])}\n"

        if profile.get("conversation_count") > 0:
            system_prompt += f"\n이 사용자와 {profile['conversation_count']}번 대화했습니다.\n"

        # 최근 대화 기록
        recent_conversations = profile_manager.get_conversation_history(user_id)[:3]
        if recent_conversations:
            system_prompt += "\n최근 대화 주제:\n"
            for conv in recent_conversations:
                system_prompt += f"  - {conv['topic']}: {conv['summary']}\n"

        system_prompt += "\n위 정보를 바탕으로 개인화된 응답을 제공하세요."

        # 메시지 구성
        messages = [SystemMessage(content=system_prompt)] + state["messages"]

        # 응답 생성
        response = model.invoke(messages)

        return {"messages": [response]}

    # 프로필 추출 노드
    def extract_profile_node(state: ChatState, config: dict, store: InMemoryStore) -> dict:
        """대화에서 프로필 정보 추출"""
        user_id = state["user_id"]

        # 최근 대화 분석
        recent_messages = state["messages"][-4:]  # 최근 2턴
        conversation_text = "\n".join([
            f"{'사용자' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent_messages
        ])

        # 프로필 정보 추출 프롬프트
        extraction_prompt = f"""다음 대화에서 사용자에 대한 정보를 추출하세요.

대화:
{conversation_text}

다음 형식의 JSON으로 응답하세요:
{{
    "name": "사용자 이름 (언급된 경우)",
    "preferences": {{"키": "값"}},
    "interests": ["관심사1", "관심사2"],
    "topic": "이번 대화의 주요 주제",
    "summary": "이번 대화 요약 (한 문장)"
}}

정보가 없으면 해당 필드는 null이나 빈 배열로 설정하세요.
"""

        extraction_response = model.invoke([HumanMessage(content=extraction_prompt)])

        try:
            # JSON 파싱
            content = extraction_response.content
            # ```json ... ``` 형식 처리
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            extracted = json.loads(content)

            # 프로필 업데이트
            updates = {}
            if extracted.get("name"):
                updates["name"] = extracted["name"]
            if extracted.get("preferences"):
                updates["preferences"] = extracted["preferences"]
            if extracted.get("interests"):
                updates["interests"] = extracted["interests"]

            if updates:
                profile_manager.update_profile(user_id, updates)
                print(f"📝 프로필 업데이트: {updates}")

            # 대화 기록 저장
            if extracted.get("topic") and extracted.get("summary"):
                profile_manager.add_conversation_history(
                    user_id,
                    extracted["topic"],
                    extracted["summary"]
                )

        except Exception as e:
            print(f"⚠️  프로필 추출 실패: {e}")

        return {}

    # 그래프 구축
    graph_builder = StateGraph(ChatState)

    # 노드 추가
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("extract_profile", extract_profile_node)

    # 엣지 추가
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", "extract_profile")
    graph_builder.add_edge("extract_profile", END)

    # 컴파일 (Store 포함)
    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory, store=store)

    return graph, store


# ============================================================================
# 테스트 함수
# ============================================================================

def test_profile_system():
    """프로필 시스템 테스트"""
    print("=" * 70)
    print("👤 사용자 프로필 시스템 테스트")
    print("=" * 70)

    chatbot, store = create_profile_chatbot()
    profile_manager = UserProfileManager(store)

    # 사용자 1: Alice
    print("\n📝 Alice와의 대화...")
    alice_config = {
        "configurable": {
            "thread_id": "alice_session",
        }
    }

    alice_conversations = [
        "안녕하세요! 제 이름은 Alice입니다.",
        "저는 파이썬 프로그래밍을 좋아해요.",
        "특히 머신러닝에 관심이 많아요.",
        "커피는 아메리카노를 선호합니다.",
    ]

    for msg in alice_conversations:
        print(f"\n👤 Alice: {msg}")
        result = chatbot.invoke(
            {
                "messages": [HumanMessage(content=msg)],
                "user_id": "alice"
            },
            alice_config
        )
        ai_response = result["messages"][-1].content
        print(f"🤖 {ai_response}")

    # Alice 프로필 확인
    print("\n" + "=" * 70)
    print("📊 Alice 프로필 확인")
    print("=" * 70)
    alice_profile = profile_manager.get_profile("alice")
    print(json.dumps(alice_profile, indent=2, ensure_ascii=False))

    # 사용자 2: Bob
    print("\n" + "=" * 70)
    print("📝 Bob과의 대화...")
    print("=" * 70)

    bob_config = {
        "configurable": {
            "thread_id": "bob_session",
        }
    }

    bob_conversations = [
        "안녕! 나는 Bob이야.",
        "웹 개발을 하고 있어.",
        "React와 Node.js를 주로 써.",
        "차는 녹차를 좋아해.",
    ]

    for msg in bob_conversations:
        print(f"\n👤 Bob: {msg}")
        result = chatbot.invoke(
            {
                "messages": [HumanMessage(content=msg)],
                "user_id": "bob"
            },
            bob_config
        )
        ai_response = result["messages"][-1].content
        print(f"🤖 {ai_response}")

    # Bob 프로필 확인
    print("\n" + "=" * 70)
    print("📊 Bob 프로필 확인")
    print("=" * 70)
    bob_profile = profile_manager.get_profile("bob")
    print(json.dumps(bob_profile, indent=2, ensure_ascii=False))

    # 개인화 응답 테스트
    print("\n" + "=" * 70)
    print("🎯 개인화 응답 테스트")
    print("=" * 70)

    # Alice에게 추천
    print("\n👤 Alice: 공부할 만한 새로운 주제를 추천해줘")
    result = chatbot.invoke(
        {
            "messages": [HumanMessage(content="공부할 만한 새로운 주제를 추천해줘")],
            "user_id": "alice"
        },
        alice_config
    )
    print(f"🤖 {result['messages'][-1].content}")

    # Bob에게 추천
    print("\n👤 Bob: 공부할 만한 새로운 주제를 추천해줘")
    result = chatbot.invoke(
        {
            "messages": [HumanMessage(content="공부할 만한 새로운 주제를 추천해줘")],
            "user_id": "bob"
        },
        bob_config
    )
    print(f"🤖 {result['messages'][-1].content}")

    # 대화 기록 확인
    print("\n" + "=" * 70)
    print("📜 Alice 대화 기록")
    print("=" * 70)
    alice_history = profile_manager.get_conversation_history("alice")
    for i, conv in enumerate(alice_history, 1):
        print(f"\n{i}. {conv['topic']}")
        print(f"   요약: {conv['summary']}")
        print(f"   시간: {conv['timestamp']}")

    print("\n" + "=" * 70)
    print("✅ 프로필 시스템 테스트 완료!")
    print("=" * 70)


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 프로필 기반 챗봇 - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /profile - 내 프로필 보기")
    print("  /history - 대화 기록")
    print("  /switch <user_id> - 사용자 전환")
    print("  /quit - 종료")
    print("=" * 70)

    chatbot, store = create_profile_chatbot()
    profile_manager = UserProfileManager(store)

    current_user = input("\n사용자 ID를 입력하세요: ").strip()
    config = {"configurable": {"thread_id": f"{current_user}_session"}}

    while True:
        try:
            user_input = input(f"\n[{current_user}] 👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 종료합니다.")
                break

            elif user_input == "/profile":
                profile = profile_manager.get_profile(current_user)
                print("\n📊 내 프로필:")
                print(json.dumps(profile, indent=2, ensure_ascii=False))
                continue

            elif user_input == "/history":
                history = profile_manager.get_conversation_history(current_user)
                if not history:
                    print("📜 대화 기록이 없습니다.")
                else:
                    print(f"\n📜 대화 기록 ({len(history)}개):")
                    for i, conv in enumerate(history, 1):
                        print(f"\n{i}. {conv['topic']}")
                        print(f"   {conv['summary']}")
                continue

            elif user_input.startswith("/switch"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("❌ 사용법: /switch <user_id>")
                    continue
                current_user = parts[1]
                config = {"configurable": {"thread_id": f"{current_user}_session"}}
                print(f"✅ {current_user}로 전환했습니다.")
                continue

            # 일반 메시지
            result = chatbot.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": current_user
                },
                config
            )

            ai_response = result["messages"][-1].content
            print(f"🤖 {ai_response}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("👤 Part 4: 사용자 프로필 시스템 - 실습 과제 3 해답")
    print("=" * 70)

    try:
        # 자동 테스트
        test_profile_system()

        # 대화형 모드
        print("\n대화형 모드를 실행하시겠습니까? (y/n): ", end="")
        choice = input().strip().lower()

        if choice in ['y', 'yes', '예']:
            interactive_mode()

    except Exception as e:
        print(f"\n⚠️  오류 발생: {e}")
        import traceback
        traceback.print_exc()

    # 학습 포인트
    print("\n" + "=" * 70)
    print("💡 학습 포인트:")
    print("  1. InMemoryStore로 구조화된 데이터 저장")
    print("  2. Namespace로 사용자별 데이터 격리")
    print("  3. 대화에서 자동으로 프로필 정보 추출")
    print("  4. 프로필 기반 개인화 응답")
    print("\n💡 추가 학습:")
    print("  1. PostgreSQL Store로 영구 저장")
    print("  2. 엔티티 인식 (NER) 통합")
    print("  3. 선호도 학습 알고리즘")
    print("  4. GDPR 준수 (삭제 요청 처리)")
    print("=" * 70)


if __name__ == "__main__":
    main()
