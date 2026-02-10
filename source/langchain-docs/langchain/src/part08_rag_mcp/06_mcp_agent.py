"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP
================================================================================

파일명: 06_mcp_agent.py
난이도: ⭐⭐⭐⭐ (중상급)
예상 시간: 35분

📚 학습 목표:
  - MCP tools를 Agent에 통합
  - 여러 MCP 서버 동시 사용
  - MCP error handling
  - RAG + MCP 조합
  - 실전 MCP 기반 전문가 Agent

📖 공식 문서:
  • MCP: https://modelcontextprotocol.io/
  • LangChain MCP: /official/20-model-context-protocol.md

📄 교안 문서:
  • Part 8: /docs/part08_rag_mcp.md

🔧 필요한 패키지:
  pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv

🚀 실행 방법:
  python 06_mcp_agent.py

================================================================================
"""

import os
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

def example_1_mcp_agent_basics():
    """MCP Tools를 Agent에 통합하는 기본 패턴"""
    print("=" * 70)
    print("📌 예제 1: MCP Tools와 Agent 통합 기초")
    print("=" * 70)
    
    print("""
💡 MCP Agent 통합:
   - MCP 서버의 Tools를 Agent에 제공
   - Agent가 필요에 따라 Tool 선택
   - 여러 MCP 서버 도구 동시 사용 가능

통합 프로세스:

1. MCP 클라이언트 초기화
   client = MultiServerMCPClient({...})

2. Tools 가져오기
   tools = await client.get_tools()

3. Agent 생성
   agent = create_agent("gpt-4o-mini", tools)

4. Agent 실행
   response = agent.invoke({...})

예시 코드:

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# MCP 서버 설정
client = MultiServerMCPClient({
    "filesystem": {
        "transport": "stdio",
        "command": "python",
        "args": ["fs_server.py"]
    },
    "database": {
        "transport": "http",
        "url": "http://localhost:8000/mcp"
    }
})

# Tools 가져오기
tools = await client.get_tools()

# Agent 생성
agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    system_prompt="당신은 파일과 DB에 접근할 수 있는 Assistant입니다."
)

# 사용
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "DB에서 사용자 조회 후 파일에 저장해주세요"}]
})
    """)
    
    # 시뮬레이션: MCP Tools를 사용하는 Agent
    print("\n🧪 시뮬레이션: MCP Agent")
    print("-" * 70)
    
    # MCP 스타일 도구들
    @tool
    def mcp_fs_read(path: str) -> str:
        """[FileSystem MCP] 파일 읽기"""
        files = {
            "/config.json": '{"app": "demo", "version": "1.0"}',
            "/data.txt": "Sample data content"
        }
        return files.get(path, "File not found")
    
    @tool
    def mcp_fs_write(path: str, content: str) -> str:
        """[FileSystem MCP] 파일 쓰기"""
        return f"✅ File written: {path} ({len(content)} bytes)"
    
    @tool
    def mcp_db_query(query: str) -> str:
        """[Database MCP] SQL 쿼리 실행"""
        if "SELECT" in query.upper():
            return json.dumps([
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ])
        return "Error: Only SELECT allowed"
    
    # Agent 생성
    mcp_tools = [mcp_fs_read, mcp_fs_write, mcp_db_query]
    
    agent = create_agent(
        model="gpt-4o-mini",
        tools=mcp_tools,
        system_prompt="""당신은 MCP 도구를 사용할 수 있는 Assistant입니다.

사용 가능한 MCP 서버:
- FileSystem: 파일 읽기/쓰기 (mcp_fs_*)
- Database: DB 쿼리 (mcp_db_*)

작업에 맞는 도구를 선택하여 사용하세요."""
    )
    
    print(f"✅ Agent 초기화 완료 ({len(mcp_tools)}개 MCP Tools)")
    
    # 테스트 작업
    tasks = [
        "config.json 파일을 읽어주세요",
        "DB에서 모든 사용자를 조회해주세요 (SELECT * FROM users)",
        "result.txt 파일에 'Processing complete' 내용을 저장해주세요"
    ]
    
    print("\n🧪 Agent 작업 테스트:")
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. 작업: {task}")
        print("-" * 70)
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        answer = response['messages'][-1].content
        print(f"응답: {answer[:150]}...")
    
    print("\n" + "=" * 70)

def example_2_multiple_mcp_servers():
    """여러 MCP 서버를 동시에 사용하는 Agent"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 여러 MCP 서버 통합 Agent")
    print("=" * 70)
    
    print("""
💡 다중 MCP 서버 패턴:
   - 각 서버가 특정 도메인 담당
   - Agent가 작업에 맞는 서버 선택
   - 서버 간 독립적 동작

서버 구성 예시:

{
    "github": {
        "transport": "http",
        "url": "http://localhost:8001/mcp"
    },
    "database": {
        "transport": "stdio",
        "command": "python",
        "args": ["db_server.py"]
    },
    "slack": {
        "transport": "http",
        "url": "http://localhost:8003/mcp"
    }
}

Agent는 작업에 따라:
- 코드 관련 → GitHub 서버 도구
- 데이터 관련 → Database 서버 도구
- 알림 관련 → Slack 서버 도구
    """)
    
    print("\n🧪 시뮬레이션: 3개 MCP 서버 통합")
    print("-" * 70)
    
    # GitHub MCP 도구
    @tool
    def github_create_issue(title: str, body: str) -> str:
        """[GitHub MCP] 이슈 생성"""
        return f"✅ Issue created: '{title}' (#{123})"
    
    @tool
    def github_search_code(query: str) -> str:
        """[GitHub MCP] 코드 검색"""
        return f"Found 5 results for '{query}'"
    
    # Database MCP 도구
    @tool
    def db_get_users(limit: int = 10) -> str:
        """[Database MCP] 사용자 목록 조회"""
        users = [f"User {i}" for i in range(1, min(limit+1, 6))]
        return json.dumps(users)
    
    @tool
    def db_get_stats(table: str) -> str:
        """[Database MCP] 테이블 통계"""
        return json.dumps({"table": table, "rows": 1234, "size": "2.4MB"})
    
    # Slack MCP 도구
    @tool
    def slack_send_message(channel: str, message: str) -> str:
        """[Slack MCP] 메시지 전송"""
        return f"✅ Message sent to #{channel}"
    
    @tool
    def slack_get_channels() -> str:
        """[Slack MCP] 채널 목록"""
        return json.dumps(["general", "dev", "alerts"])
    
    # 모든 MCP 도구 통합
    all_mcp_tools = [
        github_create_issue, github_search_code,
        db_get_users, db_get_stats,
        slack_send_message, slack_get_channels
    ]
    
    print(f"📦 통합된 MCP Tools: {len(all_mcp_tools)}개")
    print("\nMCP 서버별 도구:")
    print("  • GitHub: github_create_issue, github_search_code")
    print("  • Database: db_get_users, db_get_stats")
    print("  • Slack: slack_send_message, slack_get_channels")
    
    # 통합 Agent
    agent = create_agent(
        model="gpt-4o-mini",
        tools=all_mcp_tools,
        system_prompt="""당신은 여러 MCP 서버에 접근할 수 있는 통합 Assistant입니다.

사용 가능한 MCP 서버:
1. GitHub (코드, 이슈 관리)
2. Database (데이터 조회)
3. Slack (알림, 커뮤니케이션)

작업의 도메인에 맞는 서버의 도구를 선택하세요."""
    )
    
    # 통합 작업 테스트
    tasks = [
        "DB에서 사용자 수를 조회하고, 그 정보를 #general 채널에 알려주세요",
        "GitHub에서 'authentication' 관련 코드를 검색하고, 발견된 내용으로 이슈를 만들어주세요",
        "모든 Slack 채널 목록을 가져와주세요"
    ]
    
    print("\n" + "=" * 70)
    print("🧪 통합 작업 테스트")
    print("=" * 70)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. 작업: {task}")
        print("-" * 70)
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        answer = response['messages'][-1].content
        print(f"결과: {answer[:200]}...\n")
    
    print("=" * 70)

def example_3_mcp_error_handling():
    """MCP Tool 호출 시 에러 처리"""
    print("\n" + "=" * 70)
    print("📌 예제 3: MCP Error Handling")
    print("=" * 70)
    
    print("""
💡 MCP 에러 처리:
   - 서버 연결 실패
   - Tool 실행 오류
   - 타임아웃
   - 잘못된 매개변수

처리 전략:

1. Tool 레벨 에러 처리:
   @mcp.tool()
   def my_tool(param: str) -> str:
       try:
           # 작업 수행
           return result
       except Exception as e:
           return f"Error: {str(e)}"

2. 재시도 로직:
   async def call_with_retry(tool, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await tool()
           except:
               if attempt == max_retries - 1:
                   raise
               await asyncio.sleep(2 ** attempt)

3. Fallback 도구:
   if primary_tool_fails:
       use_fallback_tool()
    """)
    
    print("\n🧪 시뮬레이션: 에러 처리")
    print("-" * 70)
    
    # 에러가 발생할 수 있는 도구들
    @tool
    def safe_divide(a: float, b: float) -> str:
        """안전한 나눗셈 (에러 처리 포함)"""
        try:
            if b == 0:
                return "Error: Division by zero"
            result = a / b
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def safe_file_read(path: str) -> str:
        """안전한 파일 읽기 (에러 처리 포함)"""
        try:
            valid_files = ["/data.txt", "/config.json"]
            if path not in valid_files:
                return f"Error: File not found: {path}"
            return f"File content from {path}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def safe_api_call(endpoint: str) -> str:
        """안전한 API 호출 (에러 처리 포함)"""
        try:
            valid_endpoints = ["/users", "/posts", "/comments"]
            if endpoint not in valid_endpoints:
                return f"Error: Invalid endpoint: {endpoint}"
            return f"API response from {endpoint}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Agent 생성
    safe_tools = [safe_divide, safe_file_read, safe_api_call]
    
    agent = create_agent(
        model="gpt-4o-mini",
        tools=safe_tools,
        system_prompt="""당신은 에러를 잘 처리하는 안전한 Assistant입니다.

도구 실행 결과가 "Error:"로 시작하면:
1. 에러 내용을 사용자에게 알림
2. 가능하면 대안 제시
3. 다른 방법으로 작업 수행 시도"""
    )
    
    # 정상/에러 케이스 테스트
    test_cases = [
        ("10을 5로 나눠주세요", "정상"),
        ("10을 0으로 나눠주세요", "에러: 0으로 나누기"),
        ("/data.txt 파일을 읽어주세요", "정상"),
        ("/invalid.txt 파일을 읽어주세요", "에러: 파일 없음"),
        ("/users API를 호출해주세요", "정상"),
        ("/invalid API를 호출해주세요", "에러: 잘못된 endpoint")
    ]
    
    print("🧪 에러 처리 테스트:")
    for task, expected in test_cases:
        print(f"\n작업: {task} ({expected})")
        print("-" * 70)
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        answer = response['messages'][-1].content
        print(f"응답: {answer[:150]}...")
    
    print("\n" + "=" * 70)

def example_4_rag_mcp_combination():
    """RAG와 MCP를 함께 사용하는 Agent"""
    print("\n" + "=" * 70)
    print("📌 예제 4: RAG + MCP 통합 Agent")
    print("=" * 70)
    
    print("""
💡 RAG + MCP 조합:
   - RAG: 지식 베이스 검색
   - MCP: 실행 가능한 도구
   - 함께 사용하여 강력한 Agent 구축

사용 패턴:

1. 지식 검색 후 실행
   RAG → 정책 검색 → MCP → 알림 발송

2. 실행 후 지식 참조
   MCP → 데이터 조회 → RAG → 관련 문서 검색

3. 병렬 사용
   RAG + MCP → 동시에 정보 수집
    """)
    
    print("\n🧪 시뮬레이션: RAG + MCP Agent")
    print("-" * 70)
    
    # RAG: 지식 베이스 구축
    knowledge_docs = [
        Document(
            page_content="회사 휴가 정책: 연차는 입사 1년 후 15일 부여",
            metadata={"type": "policy", "category": "vacation"}
        ),
        Document(
            page_content="재택근무 정책: 주 2회까지 가능, 팀장 승인 필요",
            metadata={"type": "policy", "category": "remote"}
        ),
        Document(
            page_content="비용 처리 절차: 영수증 제출 후 5영업일 내 승인",
            metadata={"type": "policy", "category": "expense"}
        ),
        Document(
            page_content="기술 스택: Python, React, PostgreSQL, Docker 사용",
            metadata={"type": "tech", "category": "stack"}
        )
    ]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(knowledge_docs, embeddings)
    
    # RAG 도구
    @tool
    def search_knowledge(query: str) -> str:
        """회사 지식 베이스 검색 (정책, 기술 문서 등)"""
        docs = vectorstore.similarity_search(query, k=2)
        results = "\n\n".join([f"• {d.page_content}" for d in docs])
        return f"검색 결과:\n{results}"
    
    # MCP 도구들
    @tool
    def mcp_send_email(to: str, subject: str, body: str) -> str:
        """[MCP] 이메일 발송"""
        return f"✅ Email sent to {to}: '{subject}'"
    
    @tool
    def mcp_create_ticket(title: str, description: str) -> str:
        """[MCP] 티켓 생성"""
        return f"✅ Ticket created: '{title}' (#T-123)"
    
    @tool
    def mcp_schedule_meeting(date: str, attendees: str) -> str:
        """[MCP] 회의 일정 등록"""
        return f"✅ Meeting scheduled for {date} with {attendees}"
    
    # RAG + MCP 통합 Agent
    combined_tools = [
        search_knowledge,
        mcp_send_email,
        mcp_create_ticket,
        mcp_schedule_meeting
    ]
    
    agent = create_agent(
        model="gpt-4o-mini",
        tools=combined_tools,
        system_prompt="""당신은 RAG와 MCP를 모두 사용하는 통합 Assistant입니다.

작업 패턴:
1. 정보 필요 → search_knowledge로 지식 베이스 검색
2. 실행 필요 → MCP 도구 사용
3. 복잡한 작업 → RAG 검색 후 MCP 실행

예시:
- "재택근무 정책을 알려주고 팀장에게 승인 요청 이메일 보내줘"
  → RAG 검색 → 정책 확인 → MCP 이메일 발송"""
    )
    
    print("✅ RAG + MCP Agent 초기화 완료")
    print(f"  • RAG 지식 베이스: {len(knowledge_docs)}개 문서")
    print(f"  • MCP Tools: {len([t for t in combined_tools if 'mcp' in t.name])}개")
    
    # 통합 작업 테스트
    tasks = [
        "재택근무 정책이 어떻게 되나요?",
        "재택근무 신청을 위해 팀장에게 승인 요청 이메일을 보내주세요",
        "비용 처리 절차를 확인하고, 관련 티켓을 생성해주세요"
    ]
    
    print("\n" + "=" * 70)
    print("🧪 RAG + MCP 통합 작업 테스트")
    print("=" * 70)
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{i}. 작업: {task}")
        print("=" * 70)
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        answer = response['messages'][-1].content
        print(f"\n응답:\n{answer}\n")
    
    print("=" * 70)

def example_5_expert_mcp_agent():
    """실전 MCP 기반 전문가 Agent 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 MCP 전문가 Agent")
    print("=" * 70)
    
    print("""
💡 실전 시나리오:
   - 회사 통합 업무 Assistant
   - 정보 검색 + 작업 실행 + 분석
   - 다양한 MCP 서버 통합
   - 지능적인 작업 계획 및 실행

시스템 구성:
   1. Knowledge Base (RAG)
      - 회사 정책, 기술 문서
   
   2. User Management (MCP)
      - 사용자 조회, 권한 확인
   
   3. Communication (MCP)
      - 이메일, Slack 알림
   
   4. Analytics (MCP)
      - 통계 조회, 리포트 생성
   
   5. Workflow (MCP)
      - 승인 요청, 티켓 생성
    """)
    
    print("\n🏗️ 시스템 초기화")
    print("-" * 70)
    
    # 1. RAG 지식 베이스
    kb_docs = [
        Document(
            page_content="신규 프로젝트 시작 절차: 1. 제안서 작성 2. 팀장 승인 3. PM 배정 4. 킥오프 미팅",
            metadata={"category": "process", "priority": "high"}
        ),
        Document(
            page_content="코드 리뷰 가이드라인: 모든 PR은 2명 이상의 승인 필요, 테스트 커버리지 80% 이상",
            metadata={"category": "development", "priority": "high"}
        ),
        Document(
            page_content="장애 대응 프로세스: 1. Slack #alerts 알림 2. 담당자 확인 3. 이슈 생성 4. 사후 리포트",
            metadata={"category": "operations", "priority": "critical"}
        )
    ]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    kb_vectorstore = FAISS.from_documents(kb_docs, embeddings)
    
    @tool
    def search_procedures(topic: str) -> str:
        """업무 절차 및 가이드라인 검색"""
        docs = kb_vectorstore.similarity_search(topic, k=2)
        return "\n\n".join([f"📋 {d.page_content}" for d in docs])
    
    # 2. User Management Tools
    @tool
    def get_user_info(username: str) -> str:
        """사용자 정보 조회"""
        users = {
            "alice": {"role": "Engineer", "team": "Backend", "manager": "Bob"},
            "bob": {"role": "Manager", "team": "Engineering", "manager": "CEO"}
        }
        info = users.get(username.lower(), {"error": "User not found"})
        return json.dumps(info, ensure_ascii=False)
    
    @tool
    def check_permission(username: str, action: str) -> str:
        """권한 확인"""
        # 시뮬레이션
        return f"✅ User {username} has permission for {action}"
    
    # 3. Communication Tools
    @tool
    def send_notification(recipient: str, message: str, channel: str = "email") -> str:
        """알림 발송 (email/slack)"""
        return f"✅ {channel.upper()} sent to {recipient}"
    
    @tool
    def create_slack_thread(channel: str, title: str, participants: str) -> str:
        """Slack 스레드 생성"""
        return f"✅ Thread created in #{channel}: '{title}' with {participants}"
    
    # 4. Analytics Tools
    @tool
    def get_team_stats(team: str) -> str:
        """팀 통계 조회"""
        stats = {
            "Backend": {"members": 8, "active_projects": 3, "velocity": 42},
            "Frontend": {"members": 6, "active_projects": 2, "velocity": 38}
        }
        return json.dumps(stats.get(team, {}), ensure_ascii=False)
    
    @tool
    def generate_report(report_type: str, period: str) -> str:
        """리포트 생성"""
        return f"✅ {report_type} report generated for {period}"
    
    # 5. Workflow Tools
    @tool
    def create_approval_request(title: str, approver: str, details: str) -> str:
        """승인 요청 생성"""
        return f"✅ Approval request created: '{title}' → {approver}"
    
    @tool
    def create_task(title: str, assignee: str, priority: str) -> str:
        """작업 생성"""
        return f"✅ Task created: '{title}' assigned to {assignee} (priority: {priority})"
    
    # 모든 도구 통합
    expert_tools = [
        search_procedures,
        get_user_info, check_permission,
        send_notification, create_slack_thread,
        get_team_stats, generate_report,
        create_approval_request, create_task
    ]
    
    print(f"✅ {len(expert_tools)}개 도구 로드 완료")
    print("\n도구 카테고리:")
    print("  • Knowledge: search_procedures")
    print("  • User Mgmt: get_user_info, check_permission")
    print("  • Communication: send_notification, create_slack_thread")
    print("  • Analytics: get_team_stats, generate_report")
    print("  • Workflow: create_approval_request, create_task")
    
    # 전문가 Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=expert_tools,
        system_prompt="""당신은 회사의 통합 업무 전문가 Assistant입니다.

작업 처리 원칙:
1. 먼저 관련 절차/가이드라인 확인 (search_procedures)
2. 필요한 정보 수집 (get_user_info, get_team_stats)
3. 권한 확인 (check_permission)
4. 작업 실행 (create_*, send_*)
5. 관련자에게 알림 (send_notification)

복잡한 작업은 단계적으로 수행하고, 각 단계의 결과를 사용자에게 알려주세요."""
    )
    
    print("\n✅ 전문가 Agent 초기화 완료")
    
    # 복잡한 실전 작업
    complex_tasks = [
        "새 프로젝트를 시작하려고 합니다. 절차를 확인하고 필요한 단계들을 수행해주세요",
        "Backend 팀의 현재 상황을 파악하고, 주간 리포트를 생성해주세요",
        "Alice의 권한을 확인하고, 코드 배포 승인을 Bob 매니저에게 요청해주세요"
    ]
    
    print("\n" + "=" * 70)
    print("🧪 복잡한 실전 작업 처리")
    print("=" * 70)
    
    for i, task in enumerate(complex_tasks, 1):
        print(f"\n{'=' * 70}")
        print(f"{i}. 작업: {task}")
        print("=" * 70)
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })
        
        answer = response['messages'][-1].content
        print(f"\n📊 처리 결과:\n{answer}\n")
    
    # 사용자 입력
    print("=" * 70)
    print("💬 직접 작업을 요청해보세요 (종료: 'quit' 입력)")
    print("=" * 70)
    
    user_task = input("\n📋 작업: ").strip()
    
    if user_task and user_task.lower() != 'quit':
        print("\n🔄 처리 중...\n")
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": user_task}]
        })
        
        answer = response['messages'][-1].content
        print(f"📊 처리 결과:\n{answer}\n")
    
    print("=" * 70)
    print("✅ MCP 전문가 Agent 시스템 완료!")

def main():
    """메인 함수"""
    print("\n")
    print("=" * 70)
    print("Part 8: MCP Agent 통합 (06_mcp_agent.py)")
    print("=" * 70)
    
    while True:
        print("\n📚 실행할 예제를 선택하세요:")
        print("  1. MCP Tools와 Agent 통합 기초")
        print("  2. 여러 MCP 서버 통합")
        print("  3. MCP Error Handling")
        print("  4. RAG + MCP 통합")
        print("  5. 실전 MCP 전문가 Agent ⭐")
        print("  0. 종료")
        
        choice = input("\n선택 (0-5): ").strip()
        
        if choice == "1":
            example_1_mcp_agent_basics()
        elif choice == "2":
            example_2_multiple_mcp_servers()
        elif choice == "3":
            example_3_mcp_error_handling()
        elif choice == "4":
            example_4_rag_mcp_combination()
        elif choice == "5":
            example_5_expert_mcp_agent()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main()
