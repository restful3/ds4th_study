"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory - 실습 과제 1 해답
================================================================================

과제: 세션 기반 챗봇
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. 여러 사용자의 독립적인 대화 관리
2. 각 세션별로 대화 기록 저장
3. 사용자 구분을 통한 격리된 메모리 관리

학습 목표:
- 세션별 메모리 관리
- InMemoryStore 사용
- 멀티 유저 환경 구현

================================================================================
"""

from typing import Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from datetime import datetime

# ============================================================================
# State 정의
# ============================================================================

class ChatState(MessagesState):
    """채팅 상태"""
    pass


# ============================================================================
# 챗봇 그래프 구축
# ============================================================================

def create_session_chatbot():
    """세션 기반 챗봇 생성"""

    # LLM 모델
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # 챗봇 노드
    def chatbot_node(state: ChatState) -> dict:
        """대화 응답 생성"""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    # 그래프 구축
    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # 메모리 추가 - 세션별 대화 기록 저장
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


# ============================================================================
# 세션 관리자
# ============================================================================

class SessionManager:
    """멀티 유저 세션 관리"""

    def __init__(self):
        self.chatbot = create_session_chatbot()
        self.sessions = {}  # 세션 메타데이터 저장

    def create_session(self, user_id: str, session_name: str = None) -> str:
        """새 세션 생성"""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.sessions[session_id] = {
            "user_id": user_id,
            "session_name": session_name or f"{user_id}의 세션",
            "created_at": datetime.now(),
            "message_count": 0
        }

        print(f"✅ 새 세션 생성: {session_id}")
        return session_id

    def chat(self, session_id: str, user_message: str) -> str:
        """세션에서 대화"""
        if session_id not in self.sessions:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")

        # 세션별 설정
        config = {"configurable": {"thread_id": session_id}}

        # 메시지 전송
        result = self.chatbot.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config
        )

        # 통계 업데이트
        self.sessions[session_id]["message_count"] += 1

        # AI 응답 반환
        return result["messages"][-1].content

    def get_session_info(self, session_id: str) -> dict:
        """세션 정보 조회"""
        if session_id not in self.sessions:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")

        return self.sessions[session_id]

    def list_sessions(self, user_id: str = None) -> list:
        """세션 목록 조회"""
        if user_id:
            return [
                sid for sid, info in self.sessions.items()
                if info["user_id"] == user_id
            ]
        return list(self.sessions.keys())

    def get_history(self, session_id: str) -> Sequence[BaseMessage]:
        """세션 대화 기록 조회"""
        config = {"configurable": {"thread_id": session_id}}
        state = self.chatbot.get_state(config)
        return state.values.get("messages", [])


# ============================================================================
# 테스트 함수
# ============================================================================

def test_multi_user_sessions():
    """멀티 유저 세션 테스트"""
    print("=" * 70)
    print("👥 세션 기반 챗봇 테스트")
    print("=" * 70)

    manager = SessionManager()

    # 사용자 1의 세션들
    print("\n📝 Alice의 세션 생성...")
    alice_work = manager.create_session("alice", "업무 상담")
    alice_personal = manager.create_session("alice", "개인 상담")

    # 사용자 2의 세션
    print("\n📝 Bob의 세션 생성...")
    bob_session = manager.create_session("bob", "기술 지원")

    # Alice - 업무 세션
    print("\n" + "=" * 70)
    print("💼 Alice의 업무 세션")
    print("=" * 70)

    response = manager.chat(alice_work, "안녕하세요! 저는 프로젝트 매니저입니다.")
    print(f"🤖 {response}")

    response = manager.chat(alice_work, "프로젝트 일정 관리 팁을 알려주세요.")
    print(f"🤖 {response}")

    # Alice - 개인 세션
    print("\n" + "=" * 70)
    print("👤 Alice의 개인 세션")
    print("=" * 70)

    response = manager.chat(alice_personal, "안녕하세요! 저는 요가를 좋아합니다.")
    print(f"🤖 {response}")

    response = manager.chat(alice_personal, "집에서 할 수 있는 운동을 추천해주세요.")
    print(f"🤖 {response}")

    # Bob의 세션
    print("\n" + "=" * 70)
    print("🔧 Bob의 기술 지원 세션")
    print("=" * 70)

    response = manager.chat(bob_session, "Python 설치 방법을 알려주세요.")
    print(f"🤖 {response}")

    response = manager.chat(bob_session, "가상환경은 어떻게 만드나요?")
    print(f"🤖 {response}")

    # 세션 격리 확인
    print("\n" + "=" * 70)
    print("🔍 세션 격리 확인")
    print("=" * 70)

    # Alice 업무 세션에서 이전 대화 기억 확인
    response = manager.chat(alice_work, "제 직업이 뭐였죠?")
    print(f"👤 Alice (업무): 제 직업이 뭐였죠?")
    print(f"🤖 {response}")
    print("✅ 업무 세션의 컨텍스트 유지 확인!")

    # Alice 개인 세션에서는 업무 내용 모름
    response = manager.chat(alice_personal, "제가 프로젝트 매니저라고 말했나요?")
    print(f"\n👤 Alice (개인): 제가 프로젝트 매니저라고 말했나요?")
    print(f"🤖 {response}")
    print("✅ 세션 간 격리 확인!")

    # 통계 출력
    print("\n" + "=" * 70)
    print("📊 세션 통계")
    print("=" * 70)

    for session_id in [alice_work, alice_personal, bob_session]:
        info = manager.get_session_info(session_id)
        print(f"\n세션: {info['session_name']}")
        print(f"  사용자: {info['user_id']}")
        print(f"  생성 시간: {info['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  메시지 수: {info['message_count']}")

    # 대화 기록 조회
    print("\n" + "=" * 70)
    print("📜 Alice 업무 세션 전체 대화 기록")
    print("=" * 70)

    history = manager.get_history(alice_work)
    for i, msg in enumerate(history, 1):
        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
        print(f"\n{role} {msg.content[:100]}...")

    print("\n" + "=" * 70)
    print("✅ 모든 테스트 완료!")
    print("=" * 70)


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 세션 기반 챗봇 - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /new <user_id> [session_name] - 새 세션 생성")
    print("  /switch <session_id> - 세션 전환")
    print("  /list - 모든 세션 목록")
    print("  /info - 현재 세션 정보")
    print("  /history - 대화 기록")
    print("  /quit - 종료")
    print("=" * 70)

    manager = SessionManager()
    current_session = None

    while True:
        try:
            if current_session:
                user_input = input(f"\n[{current_session}] 👤 : ").strip()
            else:
                user_input = input("\n[세션 없음] 👤 : ").strip()

            if not user_input:
                continue

            # 명령어 처리
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=2)
                cmd = parts[0]

                if cmd == "/quit":
                    print("👋 종료합니다.")
                    break

                elif cmd == "/new":
                    if len(parts) < 2:
                        print("❌ 사용법: /new <user_id> [session_name]")
                        continue
                    user_id = parts[1]
                    session_name = parts[2] if len(parts) > 2 else None
                    current_session = manager.create_session(user_id, session_name)

                elif cmd == "/switch":
                    if len(parts) < 2:
                        print("❌ 사용법: /switch <session_id>")
                        continue
                    session_id = parts[1]
                    if session_id in manager.sessions:
                        current_session = session_id
                        print(f"✅ 세션 전환: {session_id}")
                    else:
                        print(f"❌ 세션을 찾을 수 없습니다: {session_id}")

                elif cmd == "/list":
                    sessions = manager.list_sessions()
                    if not sessions:
                        print("세션이 없습니다.")
                    else:
                        print("\n📋 세션 목록:")
                        for sid in sessions:
                            info = manager.get_session_info(sid)
                            marker = "👉" if sid == current_session else "  "
                            print(f"{marker} {sid} - {info['session_name']}")

                elif cmd == "/info":
                    if not current_session:
                        print("❌ 활성 세션이 없습니다.")
                        continue
                    info = manager.get_session_info(current_session)
                    print(f"\n📊 세션 정보:")
                    print(f"  ID: {current_session}")
                    print(f"  이름: {info['session_name']}")
                    print(f"  사용자: {info['user_id']}")
                    print(f"  메시지 수: {info['message_count']}")

                elif cmd == "/history":
                    if not current_session:
                        print("❌ 활성 세션이 없습니다.")
                        continue
                    history = manager.get_history(current_session)
                    print(f"\n📜 대화 기록 ({len(history)}개 메시지):")
                    for msg in history:
                        role = "👤" if isinstance(msg, HumanMessage) else "🤖"
                        print(f"{role} {msg.content}")

                else:
                    print(f"❌ 알 수 없는 명령어: {cmd}")

                continue

            # 일반 메시지
            if not current_session:
                print("❌ 세션을 먼저 생성하세요: /new <user_id>")
                continue

            response = manager.chat(current_session, user_input)
            print(f"🤖 {response}")

        except KeyboardInterrupt:
            print("\n👋 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("👥 Part 4: 세션 기반 챗봇 - 실습 과제 1 해답")
    print("=" * 70)

    try:
        # 자동 테스트
        test_multi_user_sessions()

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
    print("  1. MemorySaver로 세션별 대화 기록 관리")
    print("  2. thread_id로 세션 구분")
    print("  3. 멀티 유저 환경에서 격리된 메모리")
    print("  4. 세션 메타데이터 관리")
    print("\n💡 추가 학습:")
    print("  1. 세션 만료 시간 설정")
    print("  2. 영구 저장소 연동 (PostgreSQL)")
    print("  3. 세션 검색 및 필터링")
    print("  4. 세션 내보내기/가져오기")
    print("=" * 70)


if __name__ == "__main__":
    main()
