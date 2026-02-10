"""
================================================================================
LangChain AI Agent 마스터 교안
Part 6: Context Engineering
================================================================================

파일명: 06_context_injection.py
난이도: ⭐⭐⭐⭐ (고급)
예상 시간: 30분

📚 학습 목표:
  - dataclass로 Context 스키마 정의
  - create_agent()에 context_schema 전달
  - Tool에서 runtime.context 접근 및 사용
  - Middleware에서 context 활용
  - 실전: 멀티테넌트 SaaS 시스템

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
  python 06_context_injection.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model, wrap_model_call
from langchain.agents.agent import ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from dataclasses import dataclass
from typing import Callable, Any

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: dataclass로 Context 스키마 정의
# ============================================================================

def example_1_context_schema():
    """dataclass로 타입 안전한 Context 스키마 정의"""
    print("=" * 70)
    print("📌 예제 1: dataclass로 Context 스키마 정의")
    print("=" * 70)

    print("""
💡 Context 스키마란?
   - dataclass로 Context 구조 정의
   - 타입 안전성 제공
   - IDE 자동완성 지원
   - 명확한 문서화

🎯 장점:
   - 실수 방지 (타입 체크)
   - 코드 가독성 향상
   - 리팩토링 용이
    """)

    # Context 스키마 정의
    @dataclass
    class UserContext:
        """사용자 컨텍스트"""
        user_id: str
        user_name: str
        email: str
        role: str  # "admin", "user", "guest"
        subscription: str  # "free", "pro", "enterprise"

    # Context를 사용하는 도구
    @tool
    def get_profile(runtime: ToolRuntime[UserContext]) -> str:
        """사용자 프로필 조회"""
        ctx = runtime.context

        profile = f"""
👤 사용자 프로필:
  - ID: {ctx.user_id}
  - 이름: {ctx.user_name}
  - 이메일: {ctx.email}
  - 역할: {ctx.role}
  - 구독: {ctx.subscription}
        """.strip()

        print(f"\n{profile}")
        return profile

    # Agent 생성 (context_schema 지정)
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_profile],
        context_schema=UserContext,  # 스키마 지정
        checkpointer=MemorySaver(),
    )

    # Context 인스턴스 생성 및 전달
    user_context = UserContext(
        user_id="user_001",
        user_name="김철수",
        email="kim@example.com",
        role="admin",
        subscription="enterprise"
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "내 프로필을 보여줘"}]},
        context=user_context,
        config={"configurable": {"thread_id": "schema-001"}}
    )

    print(f"\n💬 응답: {response['messages'][-1].content}")
    print("\n✅ dataclass로 타입 안전한 Context를 정의했습니다!")


# ============================================================================
# 예제 2: create_agent()에 context_schema 전달
# ============================================================================

def example_2_agent_with_context():
    """Agent 생성 시 context_schema 전달 및 활용"""
    print("\n" + "=" * 70)
    print("📌 예제 2: create_agent()에 context_schema 전달")
    print("=" * 70)

    print("""
🔧 context_schema 사용법:
   1. dataclass로 Context 정의
   2. create_agent(context_schema=...)로 지정
   3. invoke(context=...)로 인스턴스 전달
   4. Tool과 Middleware에서 접근
    """)

    @dataclass
    class AppContext:
        """애플리케이션 컨텍스트"""
        app_name: str
        version: str
        environment: str  # "dev", "staging", "prod"
        api_key: str

    @tool
    def system_info(runtime: ToolRuntime[AppContext]) -> str:
        """시스템 정보 조회"""
        ctx = runtime.context

        info = f"""
🖥️ 시스템 정보:
  - 앱: {ctx.app_name}
  - 버전: {ctx.version}
  - 환경: {ctx.environment}
  - API 키: {ctx.api_key[:10]}***
        """.strip()

        print(f"\n{info}")
        return info

    # before_model에서도 context 접근
    @before_model
    def inject_environment_info(
        state: AgentState,
        runtime: Runtime[AppContext]
    ) -> dict[str, Any] | None:
        """환경 정보 주입"""
        ctx = runtime.context

        env_prompt = f"""
당신은 {ctx.app_name} ({ctx.version})의 AI 어시스턴트입니다.
현재 환경: {ctx.environment}
        """.strip()

        print(f"\n📝 환경 정보 프롬프트 주입")

        return {
            "messages": [
                {"role": "system", "content": env_prompt}
            ] + state["messages"]
        }

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[system_info],
        middleware=[inject_environment_info],
        context_schema=AppContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 환경에서 테스트
    environments = [
        ("dev", "개발"),
        ("staging", "스테이징"),
        ("prod", "운영"),
    ]

    for env, env_name in environments:
        print(f"\n{'='*60}")
        print(f"🌍 {env_name} 환경:")

        ctx = AppContext(
            app_name="MyApp",
            version="2.0.0",
            environment=env,
            api_key="sk-1234567890abcdef"
        )

        response = agent.invoke(
            {"messages": [{"role": "user", "content": "시스템 정보 알려줘"}]},
            context=ctx,
            config={"configurable": {"thread_id": f"env-{env}"}}
        )

        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 3: Tool에서 runtime.context 접근 및 사용
# ============================================================================

def example_3_context_in_tools():
    """Tool에서 Context를 활용한 동적 동작"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Tool에서 runtime.context 접근 및 사용")
    print("=" * 70)

    print("""
🛠️ Tool에서 Context 활용:
   - 사용자별 다른 동작
   - 권한 체크
   - 개인화된 결과
    """)

    @dataclass
    class UserContext:
        user_id: str
        user_name: str
        permission_level: int  # 1-10

    @tool
    def execute_command(
        command: str,
        runtime: ToolRuntime[UserContext]
    ) -> str:
        """권한 기반 명령 실행"""
        ctx = runtime.context

        # 명령별 필요 권한
        command_permissions = {
            "read": 1,
            "write": 5,
            "delete": 8,
            "admin": 10
        }

        required = command_permissions.get(command, 10)

        print(f"\n🔐 권한 체크:")
        print(f"  - 사용자: {ctx.user_name}")
        print(f"  - 현재 권한: {ctx.permission_level}")
        print(f"  - 필요 권한: {required}")

        if ctx.permission_level >= required:
            result = f"✅ '{command}' 명령을 실행했습니다."
        else:
            result = f"❌ 권한 부족: '{command}' 명령은 레벨 {required} 이상 필요합니다."

        return result

    @tool
    def personalized_greeting(runtime: ToolRuntime[UserContext]) -> str:
        """권한 레벨별 인사"""
        ctx = runtime.context

        if ctx.permission_level >= 8:
            greeting = f"환영합니다, {ctx.user_name} 관리자님!"
        elif ctx.permission_level >= 5:
            greeting = f"안녕하세요, {ctx.user_name} 편집자님!"
        else:
            greeting = f"안녕하세요, {ctx.user_name}님!"

        print(f"\n👋 {greeting}")
        return greeting

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[execute_command, personalized_greeting],
        context_schema=UserContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 권한 레벨 테스트
    users = [
        ("user_admin", "관리자", 10),
        ("user_editor", "편집자", 5),
        ("user_viewer", "뷰어", 1),
    ]

    for user_id, name, level in users:
        print(f"\n{'='*60}")
        print(f"👤 {name} (레벨 {level}):")

        ctx = UserContext(
            user_id=user_id,
            user_name=name,
            permission_level=level
        )

        response = agent.invoke(
            {"messages": [{"role": "user", "content": "delete 명령을 실행해줘"}]},
            context=ctx,
            config={"configurable": {"thread_id": f"perm-{user_id}"}}
        )

        print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 예제 4: Middleware에서 context 활용
# ============================================================================

def example_4_context_in_middleware():
    """Middleware에서 Context를 활용한 동적 제어"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Middleware에서 context 활용")
    print("=" * 70)

    print("""
⚙️ Middleware에서 Context 활용:
   - 프롬프트 동적 생성
   - 도구 필터링
   - 모델 선택
    """)

    @dataclass
    class BusinessContext:
        company_name: str
        industry: str
        tier: str  # "starter", "business", "enterprise"

    # Middleware: 프롬프트 커스터마이징
    @before_model
    def customize_prompt(
        state: AgentState,
        runtime: Runtime[BusinessContext]
    ) -> dict[str, Any] | None:
        """업종별 맞춤 프롬프트"""
        ctx = runtime.context

        industry_prompts = {
            "tech": "당신은 기술 산업에 특화된 AI 비서입니다.",
            "finance": "당신은 금융 산업에 특화된 AI 비서입니다.",
            "healthcare": "당신은 의료 산업에 특화된 AI 비서입니다.",
            "retail": "당신은 소매 산업에 특화된 AI 비서입니다."
        }

        prompt = industry_prompts.get(ctx.industry, "당신은 비즈니스 AI 비서입니다.")
        full_prompt = f"{prompt}\n회사: {ctx.company_name}"

        print(f"\n📝 업종별 프롬프트: {ctx.industry}")

        return {
            "messages": [
                {"role": "system", "content": full_prompt}
            ] + state["messages"]
        }

    # Middleware: 티어별 도구 필터링
    @wrap_model_call
    def tier_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """요금제 티어별 도구 제한"""
        ctx = request.runtime.context

        tier_limits = {
            "starter": ["basic_analytics"],
            "business": ["basic_analytics", "advanced_analytics"],
            "enterprise": ["basic_analytics", "advanced_analytics", "custom_reports"]
        }

        allowed = tier_limits.get(ctx.tier, ["basic_analytics"])
        tools = [t for t in request.tools if t.name in allowed]

        print(f"\n💎 티어 '{ctx.tier}': {len(tools)}개 도구 활성화")

        request = request.override(tools=tools)
        return handler(request)

    @tool
    def basic_analytics(metric: str) -> str:
        """기본 분석"""
        return f"'{metric}' 기본 분석 결과입니다."

    @tool
    def advanced_analytics(data: str) -> str:
        """고급 분석"""
        return f"'{data}' 고급 분석 결과입니다."

    @tool
    def custom_reports(template: str) -> str:
        """맞춤 보고서"""
        return f"'{template}' 맞춤 보고서를 생성했습니다."

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[basic_analytics, advanced_analytics, custom_reports],
        middleware=[customize_prompt, tier_based_tools],
        context_schema=BusinessContext,
        checkpointer=MemorySaver(),
    )

    # 다양한 티어 테스트
    companies = [
        ("StartupCo", "tech", "starter"),
        ("MidCorp", "finance", "business"),
        ("BigCorp", "healthcare", "enterprise"),
    ]

    for company, industry, tier in companies:
        print(f"\n{'='*60}")
        print(f"🏢 {company} ({tier}):")

        ctx = BusinessContext(
            company_name=company,
            industry=industry,
            tier=tier
        )

        response = agent.invoke(
            {"messages": [{"role": "user", "content": "매출 데이터를 분석해줘"}]},
            context=ctx,
            config={"configurable": {"thread_id": f"biz-{tier}"}}
        )

        print(f"💬 응답: {response['messages'][-1].content[:100]}...")


# ============================================================================
# 예제 5: 실전 - 멀티테넌트 SaaS 시스템
# ============================================================================

def example_5_multitenant_saas():
    """실전 예제: 멀티테넌트 SaaS 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 - 멀티테넌트 SaaS 시스템")
    print("=" * 70)

    print("""
🏢 멀티테넌트 SaaS:
   - 테넌트별 독립적인 데이터
   - 테넌트별 설정 및 브랜딩
   - 구독 플랜별 기능 제한
   - 사용량 추적 및 제한
    """)

    @dataclass
    class TenantContext:
        """테넌트 컨텍스트"""
        tenant_id: str
        tenant_name: str
        plan: str  # "free", "pro", "enterprise"
        max_users: int
        custom_domain: str
        branding_color: str
        features: list[str]

    # 테넌트별 데이터 조회
    @tool
    def get_tenant_data(
        query: str,
        runtime: ToolRuntime[TenantContext]
    ) -> str:
        """테넌트 데이터 조회"""
        ctx = runtime.context
        store = runtime.store

        # 테넌트별 namespace
        namespace = ("data", ctx.tenant_id)
        item = store.get(namespace, query)

        if item:
            data = item.value
            result = f"[{ctx.tenant_name}] {query}: {data}"
        else:
            result = f"[{ctx.tenant_name}] '{query}' 데이터가 없습니다."

        print(f"\n📊 {result}")
        return result

    # 테넌트 설정 조회
    @tool
    def get_tenant_config(runtime: ToolRuntime[TenantContext]) -> str:
        """테넌트 설정 조회"""
        ctx = runtime.context

        config = f"""
🏢 테넌트 설정:
  - 이름: {ctx.tenant_name}
  - 플랜: {ctx.plan}
  - 최대 사용자: {ctx.max_users}명
  - 도메인: {ctx.custom_domain}
  - 브랜딩: {ctx.branding_color}
  - 기능: {', '.join(ctx.features)}
        """.strip()

        print(f"\n{config}")
        return config

    # 기능 사용 가능 여부 체크
    @tool
    def use_feature(
        feature_name: str,
        runtime: ToolRuntime[TenantContext]
    ) -> str:
        """기능 사용"""
        ctx = runtime.context

        if feature_name in ctx.features:
            result = f"✅ '{feature_name}' 기능을 사용합니다."
        else:
            result = f"❌ '{feature_name}' 기능은 {ctx.plan} 플랜에서 사용할 수 없습니다."

        print(f"\n{result}")
        return result

    # Middleware: 테넌트별 브랜딩
    @before_model
    def apply_branding(
        state: AgentState,
        runtime: Runtime[TenantContext]
    ) -> dict[str, Any] | None:
        """테넌트 브랜딩 적용"""
        ctx = runtime.context

        branded_prompt = f"""
당신은 {ctx.tenant_name}의 AI 어시스턴트입니다.
브랜드 컬러: {ctx.branding_color}
도메인: {ctx.custom_domain}

{ctx.tenant_name} 스타일로 친절하게 답변하세요.
        """.strip()

        return {
            "messages": [
                {"role": "system", "content": branded_prompt}
            ] + state["messages"]
        }

    # Store 초기화 및 샘플 데이터
    store = InMemoryStore()

    # 테넌트별 샘플 데이터 저장
    store.put(("data", "tenant_001"), "sales", "2024년 매출: 1억원")
    store.put(("data", "tenant_002"), "sales", "2024년 매출: 5억원")

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_tenant_data, get_tenant_config, use_feature],
        middleware=[apply_branding],
        context_schema=TenantContext,
        checkpointer=MemorySaver(),
        store=store,
    )

    # 테넌트 1: Free 플랜
    tenant1 = TenantContext(
        tenant_id="tenant_001",
        tenant_name="스타트업A",
        plan="free",
        max_users=5,
        custom_domain="startupa.com",
        branding_color="#3B82F6",
        features=["basic_dashboard"]
    )

    print("\n🏢 테넌트 1 (Free):")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "설정을 보여줘"}]},
        context=tenant1,
        config={"configurable": {"thread_id": "tenant1-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    # 테넌트 2: Enterprise 플랜
    tenant2 = TenantContext(
        tenant_id="tenant_002",
        tenant_name="대기업B",
        plan="enterprise",
        max_users=1000,
        custom_domain="bigcorp.com",
        branding_color="#10B981",
        features=["basic_dashboard", "advanced_analytics", "custom_reports", "api_access"]
    )

    print("\n\n🏢 테넌트 2 (Enterprise):")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "고급 분석 기능을 사용하고 싶어"}]},
        context=tenant2,
        config={"configurable": {"thread_id": "tenant2-001"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")

    # 테넌트 1에서 Enterprise 기능 시도
    print("\n\n🏢 테넌트 1 - Enterprise 기능 시도:")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "API 접근 기능을 사용하고 싶어"}]},
        context=tenant1,
        config={"configurable": {"thread_id": "tenant1-002"}}
    )
    print(f"💬 응답: {response['messages'][-1].content}")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 6: Context Engineering - Context Injection")
    print("\n")

    try:
        # 예제 1: Context 스키마
        example_1_context_schema()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 2: Agent에 Context 전달
        example_2_agent_with_context()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 3: Tool에서 Context 사용
        example_3_context_in_tools()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 4: Middleware에서 Context
        example_4_context_in_middleware()
        input("\n⏎ 계속하려면 Enter를 누르세요...")

        # 예제 5: 멀티테넌트 SaaS
        example_5_multitenant_saas()

        # 마무리
        print("\n" + "=" * 70)
        print("🎉 Part 6 - Context Injection 완료!")
        print("=" * 70)
        print("\n💡 배운 내용:")
        print("  ✅ dataclass로 Context 스키마 정의")
        print("  ✅ context_schema를 Agent에 전달")
        print("  ✅ Tool에서 runtime.context 활용")
        print("  ✅ Middleware에서 context 활용")
        print("  ✅ 멀티테넌트 SaaS 시스템 구현")
        print("\n" + "=" * 70)
        print("🎓 Part 6: Context Engineering 전체 완료!")
        print("=" * 70)
        print("\n📚 다음 파트:")
        print("  ➡️ Part 7: Multi-Agent Systems")
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
# 1. Context 설계 원칙:
#    - 불변 데이터만 저장
#    - 대화 범위의 정보
#    - 비즈니스 로직과 분리
#
# 2. 멀티테넌트 패턴:
#    - 테넌트 ID로 데이터 격리
#    - Namespace 활용
#    - 플랜별 기능 제어
#
# 3. 실전 활용:
#    - SaaS 애플리케이션
#    - B2B 플랫폼
#    - 화이트라벨 솔루션
#
# 4. 보안 고려사항:
#    - Context는 서버에서만 생성
#    - 클라이언트 입력 검증
#    - 권한 체크 필수
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "context_schema 타입 에러"
# 해결: dataclass로 정의했는지 확인, @dataclass 데코레이터 필수
#
# 문제: "Context 속성에 접근할 수 없음"
# 해결: runtime.context로 접근, ToolRuntime[ContextType] 타입 힌트 추가
#
# 문제: "여러 테넌트 데이터가 섞임"
# 해결: Store namespace에 tenant_id 포함 필수
#
# ============================================================================
