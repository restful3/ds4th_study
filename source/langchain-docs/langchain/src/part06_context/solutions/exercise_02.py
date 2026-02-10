"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context - 실습 과제 2 해답
================================================================================

과제: 권한 기반 Agent
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. 사용자 권한 레벨에 따라 도구 제한
2. Admin, Manager, User 권한 구분
3. 권한 없는 작업 시도 시 적절한 안내

학습 목표:
- 권한 기반 접근 제어 (RBAC)
- Context로 권한 정보 전달
- 보안을 고려한 Agent 설계

================================================================================
"""

from typing import Literal
from enum import Enum
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ============================================================================
# 권한 레벨 정의
# ============================================================================

class PermissionLevel(str, Enum):
    """권한 레벨"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"


# ============================================================================
# 권한별 도구 정의
# ============================================================================

# Guest 도구 (모든 사용자)
@tool
def view_public_info() -> str:
    """공개 정보를 조회합니다."""
    return """
    📢 공개 정보
    ━━━━━━━━━━━━━━━━
    - 회사명: Tech Corp
    - 업종: IT 서비스
    - 설립: 2020년
    - 위치: 서울시 강남구
    """


@tool
def get_help() -> str:
    """도움말을 제공합니다."""
    return """
    ❓ 도움말
    ━━━━━━━━━━━━━━━━
    사용 가능한 명령:
    - 공개 정보 조회
    - 도움말 보기
    - 문의하기

    더 많은 기능을 사용하려면 로그인하세요!
    """


# User 도구
@tool
def view_personal_data() -> str:
    """개인 데이터를 조회합니다."""
    return """
    👤 개인 정보
    ━━━━━━━━━━━━━━━━
    - 이름: 홍길동
    - 부서: 개발팀
    - 입사일: 2023-01-15
    - 직급: 사원
    """


@tool
def submit_request() -> str:
    """요청사항을 제출합니다."""
    return """
    ✅ 요청이 제출되었습니다!

    요청 번호: REQ-2024-001
    상태: 대기 중
    담당자: 관리자

    처리까지 1-2일 소요될 예정입니다.
    """


@tool
def view_team_calendar() -> str:
    """팀 캘린더를 조회합니다."""
    return """
    📅 팀 캘린더
    ━━━━━━━━━━━━━━━━
    오늘:
    - 10:00 팀 스탠드업
    - 14:00 코드 리뷰

    내일:
    - 09:00 스프린트 계획
    - 15:00 1:1 미팅
    """


# Manager 도구
@tool
def view_team_data() -> str:
    """팀 데이터를 조회합니다."""
    return """
    👥 팀 현황
    ━━━━━━━━━━━━━━━━
    팀원 수: 8명
    진행 중인 프로젝트: 3개
    이번 달 성과: 95%

    팀원 목록:
    - 홍길동 (개발)
    - 김철수 (디자인)
    - 이영희 (QA)
    (외 5명)
    """


@tool
def approve_request(request_id: str = "REQ-2024-001") -> str:
    """요청을 승인합니다.

    Args:
        request_id: 요청 ID
    """
    return f"""
    ✅ 요청 승인 완료

    요청 ID: {request_id}
    승인자: Manager
    승인 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    요청자에게 알림이 전송되었습니다.
    """


@tool
def assign_task(assignee: str = "팀원", task: str = "업무") -> str:
    """팀원에게 업무를 할당합니다.

    Args:
        assignee: 담당자
        task: 업무 내용
    """
    return f"""
    📋 업무 할당 완료

    담당자: {assignee}
    업무: {task}
    기한: 3일
    우선순위: 중간

    담당자에게 알림이 전송되었습니다.
    """


# Admin 도구
@tool
def view_system_config() -> str:
    """시스템 설정을 조회합니다."""
    return """
    ⚙️  시스템 설정
    ━━━━━━━━━━━━━━━━
    - 서버 상태: 정상
    - DB 연결: 정상
    - CPU 사용률: 45%
    - 메모리 사용률: 62%
    - 활성 사용자: 127명
    """


@tool
def manage_users(action: str = "list") -> str:
    """사용자를 관리합니다.

    Args:
        action: 작업 (list, add, remove, modify)
    """
    return f"""
    👥 사용자 관리 [{action}]
    ━━━━━━━━━━━━━━━━
    전체 사용자: 156명
    활성: 127명
    비활성: 29명

    최근 가입:
    - user_123 (2024-01-20)
    - user_124 (2024-01-21)

    작업이 완료되었습니다.
    """


@tool
def system_backup() -> str:
    """시스템 백업을 실행합니다."""
    return """
    💾 백업 실행 중...

    ━━━━━━━━━━━━━━━━
    데이터베이스: ✅
    파일 시스템: ✅
    설정 파일: ✅

    백업 완료!
    위치: /backup/2024-01-22
    크기: 2.3 GB
    """


@tool
def view_audit_log() -> str:
    """감사 로그를 조회합니다."""
    return """
    📜 감사 로그
    ━━━━━━━━━━━━━━━━
    [2024-01-22 14:30] user_admin: 시스템 설정 변경
    [2024-01-22 14:25] user_manager: 요청 승인
    [2024-01-22 14:20] user_123: 개인 정보 조회
    [2024-01-22 14:15] user_admin: 사용자 추가

    총 1,234개 로그
    """


# ============================================================================
# 권한 기반 도구 선택
# ============================================================================

def get_tools_for_permission(permission: PermissionLevel) -> list:
    """권한에 따른 도구 목록 반환"""

    # 기본 도구 (모든 사용자)
    base_tools = [view_public_info, get_help]

    # 권한별 추가 도구
    permission_tools = {
        PermissionLevel.GUEST: [],
        PermissionLevel.USER: [
            view_personal_data,
            submit_request,
            view_team_calendar
        ],
        PermissionLevel.MANAGER: [
            view_personal_data,
            submit_request,
            view_team_calendar,
            view_team_data,
            approve_request,
            assign_task
        ],
        PermissionLevel.ADMIN: [
            view_personal_data,
            submit_request,
            view_team_calendar,
            view_team_data,
            approve_request,
            assign_task,
            view_system_config,
            manage_users,
            system_backup,
            view_audit_log
        ]
    }

    return base_tools + permission_tools.get(permission, [])


def get_permission_description(permission: PermissionLevel) -> str:
    """권한 설명"""
    descriptions = {
        PermissionLevel.GUEST: "제한된 공개 정보만 조회 가능",
        PermissionLevel.USER: "개인 정보 조회 및 요청 제출 가능",
        PermissionLevel.MANAGER: "팀 관리 및 요청 승인 가능",
        PermissionLevel.ADMIN: "전체 시스템 관리 가능"
    }
    return descriptions.get(permission, "")


# ============================================================================
# 권한 기반 Agent 생성
# ============================================================================

def create_permission_based_agent(
    user_id: str,
    permission: PermissionLevel
):
    """권한 기반 Agent 생성"""

    tools = get_tools_for_permission(permission)

    permission_names = {
        PermissionLevel.ADMIN: "관리자",
        PermissionLevel.MANAGER: "매니저",
        PermissionLevel.USER: "일반 사용자",
        PermissionLevel.GUEST: "게스트"
    }

    system_prompt = f"""당신은 권한 기반 접근 제어를 지원하는 AI 어시스턴트입니다.

현재 사용자:
- ID: {user_id}
- 권한: {permission_names[permission]}
- 설명: {get_permission_description(permission)}

사용 가능한 기능:
"""

    for tool_func in tools:
        system_prompt += f"- {tool_func.name}: {tool_func.description}\n"

    system_prompt += f"""
중요 규칙:
1. 사용자의 권한 내에서만 작업을 수행하세요
2. 권한이 없는 작업은 정중히 거절하세요
3. 더 높은 권한이 필요한 경우 안내하세요

무엇을 도와드릴까요?
"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    agent = create_react_agent(
        model,
        tools,
        state_modifier=system_prompt
    )

    return agent


# ============================================================================
# 사용자 데이터베이스 (모의)
# ============================================================================

from datetime import datetime

USERS_DB = {
    "admin001": {
        "name": "관리자",
        "permission": PermissionLevel.ADMIN,
        "department": "IT"
    },
    "manager001": {
        "name": "김매니저",
        "permission": PermissionLevel.MANAGER,
        "department": "개발팀"
    },
    "user001": {
        "name": "이사원",
        "permission": PermissionLevel.USER,
        "department": "개발팀"
    },
    "guest": {
        "name": "게스트",
        "permission": PermissionLevel.GUEST,
        "department": None
    }
}


# ============================================================================
# 테스트 함수
# ============================================================================

def test_permission_levels():
    """권한 레벨별 테스트"""
    print("=" * 70)
    print("🔐 권한 기반 Agent 테스트")
    print("=" * 70)

    test_cases = [
        ("guest", "공개 정보를 보여줘"),
        ("user001", "내 개인 정보를 보여줘"),
        ("user001", "시스템 설정을 보여줘"),  # 권한 없음
        ("manager001", "팀 현황을 보여줘"),
        ("manager001", "시스템 백업을 실행해줘"),  # 권한 없음
        ("admin001", "시스템 설정을 보여줘"),
        ("admin001", "백업을 실행해줘"),
    ]

    for user_id, question in test_cases:
        user_info = USERS_DB[user_id]
        permission = user_info["permission"]

        print(f"\n{'=' * 70}")
        print(f"👤 사용자: {user_info['name']} ({user_id})")
        print(f"🔑 권한: {permission.value}")
        print(f"💬 질문: {question}")
        print("=" * 70)

        agent = create_permission_based_agent(user_id, permission)

        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        final_message = result["messages"][-1]
        print(f"\n🤖 Agent 응답:\n{final_message.content}\n")


def test_permission_escalation():
    """권한 상승 시나리오 테스트"""
    print("\n" + "=" * 70)
    print("📈 권한 상승 시나리오 테스트")
    print("=" * 70)

    # User -> Manager -> Admin 순으로 권한 상승
    user_id = "user001"
    question = "팀원들에게 업무를 할당해줘"

    for permission in [PermissionLevel.USER, PermissionLevel.MANAGER, PermissionLevel.ADMIN]:
        print(f"\n{'=' * 70}")
        print(f"🔑 테스트 권한: {permission.value}")
        print(f"💬 질문: {question}")
        print("=" * 70)

        agent = create_permission_based_agent(user_id, permission)

        result = agent.invoke({"messages": [HumanMessage(content=question)]})

        final_message = result["messages"][-1]
        print(f"\n🤖 응답:\n{final_message.content}")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 권한 기반 Agent - 대화형 모드")
    print("=" * 70)

    # 사용자 선택
    print("\n사용 가능한 사용자:")
    for user_id, info in USERS_DB.items():
        print(f"  - {user_id}: {info['name']} ({info['permission'].value})")

    user_id = input("\n사용자 ID를 입력하세요: ").strip()

    if user_id not in USERS_DB:
        print("❌ 존재하지 않는 사용자입니다.")
        return

    user_info = USERS_DB[user_id]
    permission = user_info["permission"]

    print(f"\n✅ 로그인: {user_info['name']}")
    print(f"🔑 권한: {permission.value}")
    print(f"📝 {get_permission_description(permission)}")

    agent = create_permission_based_agent(user_id, permission)

    # 사용 가능한 도구 표시
    tools = get_tools_for_permission(permission)
    print(f"\n사용 가능한 기능 ({len(tools)}개):")
    for tool_func in tools:
        print(f"  - {tool_func.name}")

    print("\n명령어:")
    print("  /permissions - 권한 정보")
    print("  /tools - 사용 가능한 도구")
    print("  /switch - 사용자 전환")
    print("  /quit - 종료")
    print("=" * 70)

    while True:
        try:
            user_input = input(f"\n[{user_info['name']}] 👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("👋 안녕히 가세요!")
                break

            elif user_input == "/permissions":
                print(f"\n🔑 권한 정보:")
                print(f"  사용자: {user_info['name']}")
                print(f"  권한: {permission.value}")
                print(f"  설명: {get_permission_description(permission)}")
                continue

            elif user_input == "/tools":
                print(f"\n🔧 사용 가능한 도구:")
                for tool_func in tools:
                    print(f"  - {tool_func.name}: {tool_func.description}")
                continue

            elif user_input == "/switch":
                print("\n사용자를 전환합니다...")
                return interactive_mode()

            # 일반 질문
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]})

            final_message = result["messages"][-1]
            print(f"\n🤖 {final_message.content}")

        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ 오류: {e}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🔐 Part 6: 권한 기반 Agent - 실습 과제 2 해답")
    print("=" * 70)

    try:
        # 테스트 1: 권한 레벨별 테스트
        test_permission_levels()

        # 테스트 2: 권한 상승 시나리오
        test_permission_escalation()

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
    print("  1. 권한 기반 접근 제어 (RBAC)")
    print("  2. Context로 권한 정보 전달")
    print("  3. 권한별 도구 동적 선택")
    print("  4. 보안을 고려한 Agent 설계")
    print("\n💡 추가 학습:")
    print("  1. JWT 토큰 기반 인증")
    print("  2. 역할 상속 구조")
    print("  3. 세밀한 권한 제어 (Operation 레벨)")
    print("  4. 감사 로그 자동 기록")
    print("=" * 70)


if __name__ == "__main__":
    main()
