"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context Engineering
================================================================================

파일명: 03_dynamic_tools.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - wrap_model_call을 사용한 도구 동적 추가/제거
  - 사용자 권한 레벨에 따른 도구 필터링
  - 시간대별 도구 변경 (업무시간/비업무시간)
  - 요금제 티어별 도구 제한
  - 컨텍스트 기반 도구 조합 최적화

📖 공식 문서:
  • Runtime: /official/18-runtime.md
  • Context Engineering: /official/19-context-engineering.md

📄 교안 문서:
  • Part 6: /docs/part06_context.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 03_dynamic_tools.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call
from langchain.agents.agent import ModelRequest, ModelResponse
from langchain.tools import tool
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import dataclass
from typing import Callable

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: wrap_model_call로 도구 동적 추가/제거
# ============================================================================

def example_1_dynamic_tool_basics():
    """wrap_model_call로 도구를 동적으로 추가하거나 제거"""
    print("=" * 70)
    print("📌 예제 1: wrap_model_call로 도구 동적 추가/제거")
    print("=" * 70)

    print("""
💡 wrap_model_call이란?
   - 모델 호출을 감싸서(wrap) 제어하는 훅
   - ModelRequest를 수정하여 도구 목록 변경 가능
   - Transient 변경 (State에 저장 안 됨)

🎯 사용 사례:
   - 권한 기반 도구 필터링
   - 상황별 도구 조합
   - 성능 최적화 (불필요한 도구 제거)
    """)

    # 도구 정의
    @tool
    def read_file(filename: str) -> str:
        """파일 읽기 (읽기 권한 필요)"""
        return f"파일 '{filename}' 내용을 읽었습니다."

    @tool
    def write_file(filename: str, content: str) -> str:
        """파일 쓰기 (쓰기 권한 필요)"""
        return f"파일 '{filename}'에 내용을 작성했습니다."

    @tool
    def delete_file(filename: str) -> str:
        """파일 삭제 (삭제 권한 필요)"""
        return f"파일 '{filename}'을 삭제했습니다."

    @tool
    def public_search(query: str) -> str:
        """공개 검색 (권한 불필요)"""
        return f"'{query}' 검색 결과입니다."

    # 동적 도구 필터링
    @wrap_model_call
    def filter_tools_by_state(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """State 기반 도구 필터링"""

        # State에서 인증 상태 확인
        is_authenticated = request.state.get("authenticated", False)

        print(f"\n🔐 인증 상태: {is_authenticated}")

        if not is_authenticated:
            # 미인증: 공개 도구만 허용
            allowed_tools = [t for t in request.tools if t.name == "public_search"]
            print(f"📋 허용된 도구: {[t.name for t in allowed_tools]}")

            # request 수정
            request = request.override(tools=allowed_tools)
        else:
            # 인증됨: 모든 도구 사용 가능
            print(f"📋 허용된 도구: {[t.name for t in request.tools]}")

        # 원래 핸들러 호출
        return handler(request)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[read_file, write_file, delete_file, public_search],
        middleware=[filter_tools_by_state],
        checkpointer=MemorySaver(),
    )

    # 미인증 사용자
    print("\n🚫 미인증 사용자:")
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": "파일을 읽어줘"}],
            "authenticated": False
        },
        config={"configurable": {"thread_id": "unauth-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    # 인증 사용자
    print("\n\n✅ 인증된 사용자:")
    response = agent.invoke(
        {
            "messages": [{"role": "user", "content": "파일을 읽어줘"}],
            "authenticated": True
        },
        config={"configurable": {"thread_id": "auth-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 2: 사용자 권한 레벨 기반 도구 필터링
# ============================================================================

def example_2_permission_based_tools():
    """사용자 역할에 따라 도구 접근 제어"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 사용자 권한 레벨 기반 도구 필터링")
    print("=" * 70)

    print("""
🎯 권한 기반 접근 제어:
   - Admin: 모든 도구 사용 가능
   - Editor: 읽기/쓰기만 가능
   - Viewer: 읽기만 가능
   - Guest: 공개 도구만 가능
    """)

    # Context 정의
    @dataclass
    class UserContext:
        user_id: str
        user_role: str  # "admin", "editor", "viewer", "guest"

    # 도구 정의
    @tool
    def admin_dashboard(command: str) -> str:
        """관리자 대시보드 (Admin 전용)"""
        return f"관리자 명령 '{command}' 실행됨."

    @tool
    def read_data(query: str) -> str:
        """데이터 읽기 (Viewer 이상)"""
        return f"'{query}' 데이터를 조회했습니다."

    @tool
    def write_data(data: str) -> str:
        """데이터 쓰기 (Editor 이상)"""
        return f"데이터 '{data}'를 저장했습니다."

    @tool
    def delete_data(data_id: str) -> str:
        """데이터 삭제 (Admin 전용)"""
        return f"데이터 ID '{data_id}'를 삭제했습니다."

    @tool
    def public_info(topic: str) -> str:
        """공개 정보 조회 (모두)"""
        return f"'{topic}' 공개 정보입니다."

    # 권한 기반 필터링
    @wrap_model_call
    def permission_filter(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """역할 기반 도구 필터링"""

        ctx = request.runtime.context
        role = ctx.user_role

        # 역할별 허용 도구
        role_tools = {
            "admin": ["admin_dashboard", "read_data", "write_data", "delete_data", "public_info"],
            "editor": ["read_data", "write_data", "public_info"],
            "viewer": ["read_data", "public_info"],
            "guest": ["public_info"]
        }

        allowed_names = role_tools.get(role, ["public_info"])
        allowed_tools = [t for t in request.tools if t.name in allowed_names]

        print(f"\n👤 사용자: {ctx.user_id} ({role})")
        print(f"📋 허용된 도구: {[t.name for t in allowed_tools]}")

        request = request.override(tools=allowed_tools)
        return handler(request)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[admin_dashboard, read_data, write_data, delete_data, public_info],
        middleware=[permission_filter],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 역할 테스트
    roles = [
        ("admin_001", "admin"),
        ("editor_001", "editor"),
        ("viewer_001", "viewer"),
        ("guest_001", "guest"),
    ]

    for user_id, role in roles:
        print(f"\n{'='*60}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "데이터를 삭제해줘"}]},
            context=UserContext(user_id=user_id, user_role=role),
            config={"configurable": {"thread_id": f"role-{role}"}}
        )
        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 3: 시간대별 도구 변경 (업무시간/비업무시간)
# ============================================================================

def example_3_time_based_tools():
    """업무 시간 여부에 따라 사용 가능한 도구 변경"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 시간대별 도구 변경 (업무시간/비업무시간)")
    print("=" * 70)

    print("""
⏰ 시간대별 도구 전략:
   - 업무 시간 (09:00-18:00): 모든 업무 도구 활성화
   - 비업무 시간: 긴급 도구 + 읽기 전용
   - 주말: 읽기 전용
    """)

    # 도구 정의
    @tool
    def send_email(to: str, subject: str) -> str:
        """이메일 발송 (업무시간)"""
        return f"'{to}'에게 '{subject}' 이메일을 발송했습니다."

    @tool
    def create_report(title: str) -> str:
        """보고서 생성 (업무시간)"""
        return f"'{title}' 보고서를 생성했습니다."

    @tool
    def view_dashboard(metric: str) -> str:
        """대시보드 조회 (항상 가능)"""
        return f"'{metric}' 지표를 조회했습니다."

    @tool
    def emergency_alert(message: str) -> str:
        """긴급 알림 (항상 가능)"""
        return f"긴급 알림 발송: {message}"

    # 시간대별 도구 필터링
    @wrap_model_call
    def time_based_filter(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """시간대별 도구 필터링"""

        now = datetime.datetime.now()
        hour = now.hour
        weekday = now.weekday()  # 0=월요일, 6=일요일

        # 업무 시간 판단
        is_weekend = weekday >= 5
        is_business_hours = 9 <= hour < 18 and not is_weekend

        print(f"\n⏰ 현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📅 요일: {['월','화','수','목','금','토','일'][weekday]}요일")
        print(f"💼 업무 시간: {is_business_hours}")

        if is_business_hours:
            # 업무 시간: 모든 도구 사용 가능
            allowed_tools = request.tools
            print("📋 모든 도구 활성화")
        else:
            # 비업무 시간: 조회 + 긴급 도구만
            allowed_tools = [
                t for t in request.tools
                if t.name in ["view_dashboard", "emergency_alert"]
            ]
            print("📋 제한된 도구만 활성화 (조회 + 긴급)")

        request = request.override(tools=allowed_tools)
        return handler(request)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[send_email, create_report, view_dashboard, emergency_alert],
        middleware=[time_based_filter],
        checkpointer=MemorySaver(),
    )

    # 테스트
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "보고서를 생성해줘"}]},
        config={"configurable": {"thread_id": "time-test-001"}}
    )
    print(f"\n💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 4: 요금제 티어별 도구 제한
# ============================================================================

def example_4_tier_based_tools():
    """구독 요금제에 따라 사용 가능한 도구 제한"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 요금제 티어별 도구 제한")
    print("=" * 70)

    print("""
💎 요금제별 기능:
   - Free: 기본 도구만
   - Pro: 고급 도구 추가
   - Enterprise: 모든 도구 + 전용 지원
    """)

    # Context 정의
    @dataclass
    class SubscriptionContext:
        user_id: str
        tier: str  # "free", "pro", "enterprise"
        quota: int

    # 도구 정의
    @tool
    def basic_search(query: str) -> str:
        """기본 검색 (Free+)"""
        return f"'{query}' 기본 검색 결과입니다."

    @tool
    def advanced_analytics(data: str) -> str:
        """고급 분석 (Pro+)"""
        return f"'{data}' 고급 분석 결과입니다."

    @tool
    def ai_recommendations(context: str) -> str:
        """AI 추천 (Pro+)"""
        return f"'{context}' 기반 AI 추천입니다."

    @tool
    def dedicated_support(issue: str) -> str:
        """전담 지원 (Enterprise)"""
        return f"전담 팀이 '{issue}' 이슈를 처리합니다."

    @tool
    def custom_integration(service: str) -> str:
        """맞춤 통합 (Enterprise)"""
        return f"'{service}' 커스텀 통합을 설정했습니다."

    # 티어별 도구 필터링
    @wrap_model_call
    def tier_filter(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """요금제 티어별 도구 필터링"""

        ctx = request.runtime.context
        tier = ctx.tier
        quota = ctx.quota

        # 티어별 허용 도구
        tier_tools = {
            "free": ["basic_search"],
            "pro": ["basic_search", "advanced_analytics", "ai_recommendations"],
            "enterprise": [
                "basic_search", "advanced_analytics", "ai_recommendations",
                "dedicated_support", "custom_integration"
            ]
        }

        allowed_names = tier_tools.get(tier, ["basic_search"])
        allowed_tools = [t for t in request.tools if t.name in allowed_names]

        print(f"\n💎 요금제: {tier.upper()}")
        print(f"📊 사용 가능 횟수: {quota}")
        print(f"📋 허용된 도구: {[t.name for t in allowed_tools]}")

        request = request.override(tools=allowed_tools)
        return handler(request)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[
            basic_search, advanced_analytics, ai_recommendations,
            dedicated_support, custom_integration
        ],
        middleware=[tier_filter],
        context_schema=SubscriptionContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 티어 테스트
    tiers = [
        ("user_free", "free", 10),
        ("user_pro", "pro", 100),
        ("user_enterprise", "enterprise", 9999),
    ]

    for user_id, tier, quota in tiers:
        print(f"\n{'='*60}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "데이터를 분석해줘"}]},
            context=SubscriptionContext(user_id=user_id, tier=tier, quota=quota),
            config={"configurable": {"thread_id": f"tier-{tier}"}}
        )
        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 5: 컨텍스트 기반 도구 조합 최적화
# ============================================================================

def example_5_context_optimized_tools():
    """대화 컨텍스트를 분석하여 최적의 도구 조합 제공"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 컨텍스트 기반 도구 조합 최적화")
    print("=" * 70)

    print("""
🧠 지능형 도구 선택:
   - 사용자 의도 분석
   - 관련 도구만 활성화
   - 성능 최적화 (모델 혼란 방지)
    """)

    # 도구 정의 (카테고리별)
    # 검색 도구
    @tool
    def web_search(query: str) -> str:
        """웹 검색"""
        return f"'{query}' 웹 검색 결과입니다."

    @tool
    def document_search(keyword: str) -> str:
        """문서 검색"""
        return f"'{keyword}' 문서 검색 결과입니다."

    # 분석 도구
    @tool
    def data_analysis(dataset: str) -> str:
        """데이터 분석"""
        return f"'{dataset}' 데이터 분석 완료."

    @tool
    def trend_analysis(metric: str) -> str:
        """트렌드 분석"""
        return f"'{metric}' 트렌드 분석 완료."

    # 생성 도구
    @tool
    def generate_report(topic: str) -> str:
        """보고서 생성"""
        return f"'{topic}' 보고서를 생성했습니다."

    @tool
    def create_chart(data: str) -> str:
        """차트 생성"""
        return f"'{data}' 차트를 생성했습니다."

    # 컨텍스트 기반 도구 선택
    @wrap_model_call
    def smart_tool_selection(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """대화 내용 분석하여 관련 도구만 활성화"""

        # 최근 메시지 분석
        if request.messages:
            last_msg = request.messages[-1].content.lower()
        else:
            last_msg = ""

        print(f"\n💬 사용자 메시지: {last_msg}")

        # 의도 파악
        if any(keyword in last_msg for keyword in ["검색", "찾", "조회"]):
            # 검색 의도
            category = "검색"
            allowed_tools = [t for t in request.tools if "search" in t.name]
        elif any(keyword in last_msg for keyword in ["분석", "트렌드", "통계"]):
            # 분석 의도
            category = "분석"
            allowed_tools = [t for t in request.tools if "analysis" in t.name]
        elif any(keyword in last_msg for keyword in ["생성", "만들", "작성"]):
            # 생성 의도
            category = "생성"
            allowed_tools = [t for t in request.tools if any(
                x in t.name for x in ["generate", "create"]
            )]
        else:
            # 불명확: 모든 도구
            category = "전체"
            allowed_tools = request.tools

        print(f"🎯 감지된 의도: {category}")
        print(f"📋 활성화된 도구: {[t.name for t in allowed_tools]}")

        request = request.override(tools=allowed_tools)
        return handler(request)

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[
            web_search, document_search,
            data_analysis, trend_analysis,
            generate_report, create_chart
        ],
        middleware=[smart_tool_selection],
        checkpointer=MemorySaver(),
    )

    # 다양한 의도 테스트
    test_queries = [
        "파이썬에 대해 검색해줘",
        "매출 데이터를 분석해줘",
        "보고서를 생성해줘",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": "context-opt-001"}}
        )
        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 6: Context Engineering - Dynamic Tools")
    print("\n")

    try:
        # 예제 1: 도구 동적 추가/제거
        example_1_dynamic_tool_basics()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 2: 권한 기반 필터링
        example_2_permission_based_tools()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 3: 시간대별 도구
        example_3_time_based_tools()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 4: 티어별 도구
        example_4_tier_based_tools()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 5: 컨텍스트 최적화
        example_5_context_optimized_tools()

        # 마무리
        print("\n" + "=" * 70)
        print("🎉 Part 6 - Dynamic Tools 완료!")
        print("=" * 70)
        print("\n💡 배운 내용:")
        print("  ✅ wrap_model_call로 도구 동적 제어")
        print("  ✅ 권한 기반 도구 필터링")
        print("  ✅ 시간대별 도구 변경")
        print("  ✅ 요금제 티어별 제한")
        print("  ✅ 컨텍스트 기반 최적화")
        print("\n📚 다음 단계:")
        print("  ➡️ 04_dynamic_model.py - 동적 모델")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자가 프로그램을 중단했습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. wrap_model_call의 특징:
#    - Transient 변경 (State에 저장 안 됨)
#    - ModelRequest/ModelResponse 수정
#    - 모델 호출 전후 제어 가능
#
# 2. 동적 도구의 장점:
#    - 보안 강화 (권한 제어)
#    - 성능 향상 (불필요한 도구 제거)
#    - 사용자 경험 개선 (관련 도구만 제공)
#
# 3. 실전 적용:
#    - 멀티테넌트 SaaS
#    - B2B 플랫폼
#    - 역할 기반 접근 제어
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "도구가 필터링되지 않음"
# 해결: wrap_model_call이 middleware에 등록되었는지 확인
#
# 문제: "request.override()가 작동하지 않음"
# 해결: 반드시 수정된 request를 handler에 전달해야 함
#
# 문제: "너무 많은 도구로 모델이 혼란스러워함"
# 해결: 컨텍스트 기반으로 관련 도구만 활성화
#
# ============================================================================
