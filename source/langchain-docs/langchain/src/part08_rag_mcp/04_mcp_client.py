"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP
================================================================================

파일명: 04_mcp_client.py
난이도: ⭐⭐⭐ (중급)
예상 시간: 30분

📚 학습 목표:
  - MCP 클라이언트 기본 개념
  - 로컬 MCP 서버 연결 (stdio)
  - Tool 목록 가져오기 및 검사
  - MCP tool 호출
  - 실전 여러 MCP 서버 통합 사용

📖 공식 문서:
  • MCP: https://modelcontextprotocol.io/
  • LangChain MCP: /official/20-model-context-protocol.md

📄 교안 문서:
  • Part 8: /docs/part08_rag_mcp.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 04_mcp_client.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 .env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# ============================================================================
# 예제 1: MCP 기본 개념
# ============================================================================

def example_1_mcp_basics():
    """MCP의 기본 개념과 아키텍처"""
    print("=" * 70)
    print("📌 예제 1: MCP (Model Context Protocol) 기본 개념")
    print("=" * 70)

    print("""
💡 MCP란?
   - LLM 애플리케이션이 외부 도구와 연결하는 표준 프로토콜
   - 서버-클라이언트 구조
   - 도구, 리소스, 프롬프트를 표준화된 방식으로 제공

🔧 핵심 개념:

1. Server (서버)
   - 도구와 리소스를 제공하는 프로세스
   - 예: GitHub API 서버, DB 쿼리 서버, 파일 시스템 서버

2. Client (클라이언트)
   - 서버에 연결하여 도구를 사용
   - Agent가 MCP 클라이언트 역할

3. Transport (전송 방식)
   - stdio: 표준 입출력 (로컬 프로세스)
   - HTTP: HTTP 요청 (원격 서버)

4. Tool (도구)
   - 실행 가능한 함수
   - 예: 파일 읽기, DB 쿼리, API 호출

5. Resource (리소스)
   - 읽기 가능한 데이터
   - 예: 설정 파일, 로그 파일

6. Prompt (프롬프트)
   - 재사용 가능한 프롬프트 템플릿
   - 예: 코드 리뷰 프롬프트, 요약 프롬프트

📊 아키텍처:

┌─────────────────┐
│  Agent (Client) │
└────────┬────────┘
         │
         │ MCP Protocol
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│Server1│ │Server2│
│GitHub │ │  DB   │
└───┬───┘ └──┬────┘
    │        │
┌───▼───┐ ┌──▼────┐
│Tools: │ │Tools: │
│create │ │query  │
│search │ │insert │
└───────┘ └───────┘

🎯 장점:

1. 표준화
   - 모든 외부 도구를 동일한 방식으로 사용
   - 서버 추가/변경이 쉬움

2. 확장성
   - 새로운 서버를 쉽게 추가
   - 여러 서버 동시 사용 가능

3. 재사용성
   - 한 번 만든 서버를 여러 Agent에서 사용
   - 도구 메타데이터 자동 제공

4. 분리
   - Agent 로직과 도구 구현 분리
   - 도구 업데이트가 Agent에 영향 없음

💻 사용 예시:

```python
# 클라이언트 초기화
client = MultiServerMCPClient({
    "github": {
        "transport": "http",
        "url": "http://localhost:8001/mcp"
    },
    "database": {
        "transport": "stdio",
        "command": "python",
        "args": ["db_server.py"]
    }
})

# 모든 서버의 도구 가져오기
tools = await client.get_tools()

# Agent에서 사용
agent = create_agent("gpt-4o-mini", tools)
```

📝 이번 파트에서 배울 내용:
   1. MCP 클라이언트 사용법
   2. 로컬 서버 연결
   3. Tool 검사 및 호출
   4. 여러 서버 통합
   5. MCP 서버 구현 (다음 파일)
    """)

    print("\n" + "=" * 70)


# ============================================================================
# 예제 2: 시뮬레이션된 MCP Tool 사용
# ============================================================================

def example_2_simulated_mcp_tools():
    """MCP Tool을 시뮬레이션하여 개념 이해"""
    print("\n" + "=" * 70)
    print("📌 예제 2: MCP Tool 시뮬레이션")
    print("=" * 70)

    print("""
💡 실제 MCP 서버 없이 개념 이해하기:
   - MCP Tool처럼 동작하는 일반 Tool 생성
   - Tool 메타데이터 구조 이해
   - Tool 호출 패턴 학습
    """)

    # MCP Tool을 시뮬레이션하는 도구들
    @tool
    def mcp_file_read(file_path: str) -> str:
        """[MCP Tool] 파일을 읽습니다
        
        이 도구는 MCP 파일 시스템 서버의 read_file tool을 시뮬레이션합니다.
        
        Args:
            file_path: 읽을 파일 경로
            
        Returns:
            파일 내용
        """
        # 실제로는 MCP 서버가 파일을 읽지만, 여기서는 시뮬레이션
        simulated_files = {
            "/config.json": '{"app": "demo", "version": "1.0"}',
            "/data.txt": "This is sample data from MCP server",
            "/log.txt": "[INFO] Application started\n[INFO] Processing request"
        }
        
        if file_path in simulated_files:
            return f"[MCP Server Response]\nFile: {file_path}\nContent:\n{simulated_files[file_path]}"
        else:
            return f"[MCP Server Error] File not found: {file_path}"

    @tool
    def mcp_db_query(sql: str) -> str:
        """[MCP Tool] 데이터베이스 쿼리를 실행합니다
        
        이 도구는 MCP 데이터베이스 서버의 query tool을 시뮬레이션합니다.
        
        Args:
            sql: 실행할 SQL 쿼리 (SELECT만 허용)
            
        Returns:
            쿼리 결과 (JSON 형식)
        """
        # 실제로는 MCP 서버가 DB에 쿼리하지만, 여기서는 시뮬레이션
        if not sql.strip().upper().startswith("SELECT"):
            return "[MCP Server Error] Only SELECT queries allowed"
        
        # 시뮬레이션된 결과
        simulated_result = {
            "columns": ["id", "name", "role"],
            "rows": [
                {"id": 1, "name": "Alice", "role": "Engineer"},
                {"id": 2, "name": "Bob", "role": "Designer"},
                {"id": 3, "name": "Charlie", "role": "Manager"}
            ],
            "row_count": 3
        }
        
        return f"[MCP Server Response]\nQuery: {sql}\nResult:\n{json.dumps(simulated_result, indent=2, ensure_ascii=False)}"

    @tool
    def mcp_calculate(expression: str) -> str:
        """[MCP Tool] 수학 계산을 수행합니다
        
        이 도구는 MCP 계산 서버의 calculate tool을 시뮬레이션합니다.
        
        Args:
            expression: 계산할 수식
            
        Returns:
            계산 결과
        """
        try:
            # 안전한 계산 (보안상 eval은 실제로는 위험)
            # 실제 MCP 서버는 안전한 파서를 사용
            result = eval(expression, {"__builtins__": {}}, {})
            return f"[MCP Server Response]\nExpression: {expression}\nResult: {result}"
        except Exception as e:
            return f"[MCP Server Error] Calculation failed: {str(e)}"

    # Tool 메타데이터 표시
    print("\n📦 사용 가능한 MCP Tools:")
    print("-" * 70)
    
    tools = [mcp_file_read, mcp_db_query, mcp_calculate]
    
    for i, t in enumerate(tools, 1):
        print(f"\n{i}. {t.name}")
        print(f"   설명: {t.description.split(chr(10))[0]}")
        print(f"   매개변수: {list(t.args.keys())}")

    # Agent 생성
    print("\n🤖 Agent 생성")
    print("-" * 70)

    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""당신은 MCP Tools를 사용할 수 있는 Assistant입니다.

사용 가능한 도구:
- mcp_file_read: 파일 읽기
- mcp_db_query: 데이터베이스 쿼리
- mcp_calculate: 계산

각 도구는 [MCP Server Response] 형식으로 결과를 반환합니다."""
    )

    print("✅ Agent 초기화 완료")

    # 테스트 작업
    test_tasks = [
        "config.json 파일의 내용을 읽어주세요",
        "데이터베이스에서 모든 사용자를 조회해주세요 (SELECT * FROM users)",
        "15 * 23 + 100을 계산해주세요"
    ]

    print("\n" + "=" * 70)
    print("🧪 MCP Tool 사용 테스트")
    print("=" * 70)

    for task in test_tasks:
        print(f"\n📋 작업: {task}")
        print("-" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 응답:\n{answer}\n")

    print("=" * 70)


# ============================================================================
# 예제 3: Tool 메타데이터 검사
# ============================================================================

def example_3_tool_metadata():
    """Tool의 메타데이터 구조 이해"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Tool 메타데이터 검사")
    print("=" * 70)

    print("""
💡 Tool 메타데이터란?
   - Tool의 이름, 설명, 매개변수 정보
   - Agent가 Tool을 선택하는 기준
   - 명확한 메타데이터 = 정확한 Tool 선택

구조:
   - name: Tool 이름 (함수명)
   - description: Tool 설명 (Agent가 읽음)
   - args: 매개변수 스키마 (Pydantic)
    """)

    # 다양한 Tool 정의
    @tool
    def simple_tool(text: str) -> str:
        """간단한 텍스트 처리 도구"""
        return text.upper()

    @tool
    def complex_tool(
        query: str,
        limit: int = 10,
        include_metadata: bool = False
    ) -> str:
        """복잡한 검색 도구
        
        Args:
            query: 검색 쿼리
            limit: 결과 개수 제한 (기본값: 10)
            include_metadata: 메타데이터 포함 여부 (기본값: False)
            
        Returns:
            검색 결과
        """
        return f"Searched for '{query}' (limit={limit}, metadata={include_metadata})"

    @tool
    def typed_tool(number: int, is_active: bool) -> str:
        """타입 힌트가 있는 도구
        
        Args:
            number: 숫자 (정수만 허용)
            is_active: 활성화 여부 (True/False)
        """
        return f"Number: {number}, Active: {is_active}"

    tools = [simple_tool, complex_tool, typed_tool]

    print("\n🔍 Tool 메타데이터 분석")
    print("=" * 70)

    for tool in tools:
        print(f"\n📦 Tool: {tool.name}")
        print("-" * 70)
        print(f"설명: {tool.description}")
        print(f"\n매개변수:")
        
        if hasattr(tool, 'args') and tool.args:
            for param_name, param_info in tool.args.items():
                param_type = param_info.get('type', 'any')
                required = param_info.get('required', False)
                default = param_info.get('default', None)
                
                status = "필수" if required else f"선택 (기본값: {default})"
                print(f"  • {param_name}: {param_type} ({status})")
        else:
            print("  (매개변수 없음)")

    # Tool 메타데이터가 Agent 선택에 미치는 영향
    print("\n" + "=" * 70)
    print("💡 명확한 설명의 중요성")
    print("=" * 70)

    print("""
❌ 나쁜 예:
   @tool
   def do_something(x: str) -> str:
       \"\"\"뭔가를 함\"\"\"  # 너무 모호!
       return x

✅ 좋은 예:
   @tool
   def search_company_policy(query: str) -> str:
       \"\"\"회사 정책 문서를 검색합니다.
       
       휴가, 복지, 근무 규정 등의 정보를 찾을 때 사용하세요.
       
       Args:
           query: 검색할 정책 주제 (예: '연차', '재택근무')
           
       Returns:
           관련 정책 내용
       \"\"\"
       # ...

🎯 좋은 Tool 설명 작성법:
   1. 무엇을 하는지 명확히 기술
   2. 언제 사용해야 하는지 예시 제공
   3. 매개변수 의미 설명
   4. 반환값 형식 명시
    """)

    print("\n" + "=" * 70)


# ============================================================================
# 예제 4: 여러 Tool 조합 사용
# ============================================================================

def example_4_multiple_tools():
    """여러 Tool을 조합하여 복잡한 작업 수행"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 여러 Tool 조합 사용")
    print("=" * 70)

    print("""
💡 Tool 조합 패턴:
   - 단순 Tool들을 조합하여 복잡한 작업 수행
   - Agent가 필요에 따라 여러 Tool을 순차 호출
   - 각 Tool은 단일 책임 원칙 준수

예시:
   작업: "사용자 정보 조회 후 이메일 발송"
   → 1. search_user (사용자 검색)
   → 2. get_user_email (이메일 가져오기)
   → 3. send_email (이메일 발송)
    """)

    # 시뮬레이션된 데이터베이스
    users_db = {
        "alice": {"id": 1, "name": "Alice", "email": "alice@company.com", "role": "Engineer"},
        "bob": {"id": 2, "name": "Bob", "email": "bob@company.com", "role": "Designer"},
        "charlie": {"id": 3, "name": "Charlie", "email": "charlie@company.com", "role": "Manager"}
    }

    # Tool 1: 사용자 검색
    @tool
    def search_user(username: str) -> str:
        """사용자 정보를 검색합니다
        
        Args:
            username: 검색할 사용자 이름 (소문자)
            
        Returns:
            사용자 정보 (JSON)
        """
        user = users_db.get(username.lower())
        if user:
            return json.dumps(user, ensure_ascii=False)
        else:
            return json.dumps({"error": "User not found"})

    # Tool 2: 이메일 주소 가져오기
    @tool
    def get_user_email(user_json: str) -> str:
        """사용자 JSON에서 이메일 주소를 추출합니다
        
        Args:
            user_json: 사용자 정보 JSON 문자열
            
        Returns:
            이메일 주소
        """
        try:
            user = json.loads(user_json)
            if "error" in user:
                return "Error: User data invalid"
            return user.get("email", "Email not found")
        except:
            return "Error: Invalid JSON"

    # Tool 3: 이메일 발송 (시뮬레이션)
    @tool
    def send_email(to: str, subject: str, body: str) -> str:
        """이메일을 발송합니다 (시뮬레이션)
        
        Args:
            to: 수신자 이메일
            subject: 제목
            body: 본문
            
        Returns:
            발송 결과
        """
        if "@" not in to:
            return "Error: Invalid email address"
        
        return f"""✅ 이메일 발송 완료
수신자: {to}
제목: {subject}
본문: {body[:50]}..."""

    # Tool 4: 사용자 역할 확인
    @tool
    def get_user_role(user_json: str) -> str:
        """사용자 JSON에서 역할을 추출합니다
        
        Args:
            user_json: 사용자 정보 JSON 문자열
            
        Returns:
            사용자 역할
        """
        try:
            user = json.loads(user_json)
            if "error" in user:
                return "Error: User data invalid"
            return user.get("role", "Role not found")
        except:
            return "Error: Invalid JSON"

    tools = [search_user, get_user_email, send_email, get_user_role]

    print(f"\n📦 사용 가능한 Tools: {len(tools)}개")
    for t in tools:
        print(f"  • {t.name}")

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=tools,
        system_prompt="""당신은 사용자 관리 및 이메일 발송을 담당하는 Assistant입니다.

작업을 수행할 때:
1. 먼저 search_user로 사용자 정보 조회
2. 필요한 정보 추출 (이메일, 역할 등)
3. 추출한 정보로 작업 수행 (이메일 발송 등)

여러 Tool을 조합하여 작업을 완료하세요."""
    )

    # 복잡한 작업 테스트
    complex_tasks = [
        "Alice에게 '회의 일정 안내' 제목으로 '내일 오전 10시 회의입니다'라는 내용의 이메일을 보내주세요",
        "Bob의 역할이 무엇인지 알려주세요",
        "Charlie의 이메일 주소를 알려주세요"
    ]

    print("\n" + "=" * 70)
    print("🧪 복잡한 작업 테스트")
    print("=" * 70)

    for task in complex_tasks:
        print(f"\n📋 작업: {task}")
        print("-" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 결과:\n{answer}\n")

    print("=" * 70)


# ============================================================================
# 예제 5: 실전 - MCP 스타일 통합 시스템
# ============================================================================

def example_5_integrated_system():
    """여러 'MCP 서버'를 시뮬레이션하는 통합 시스템"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 MCP 스타일 통합 시스템")
    print("=" * 70)

    print("""
💡 실전 시나리오:
   - 여러 'MCP 서버'를 시뮬레이션
   - 각 서버는 특정 도메인 담당
   - Agent가 적절한 서버의 Tool 선택

서버 구성:
   1. FileSystem Server: 파일 관리
   2. Database Server: 데이터 조회
   3. Notification Server: 알림 발송
   4. Analytics Server: 통계 분석
    """)

    # === FileSystem Server Tools ===
    files_storage = {
        "/reports/sales_2024.txt": "2024년 매출: $5.2M, 전년 대비 15% 증가",
        "/reports/users_2024.txt": "총 사용자 수: 1,250명, 신규 가입: 340명",
        "/config/app.json": '{"version": "2.0", "env": "production"}'
    }

    @tool
    def fs_list_files(directory: str) -> str:
        """[FileSystem Server] 디렉토리의 파일 목록을 조회합니다"""
        files = [f for f in files_storage.keys() if f.startswith(directory)]
        return "\n".join(files) if files else "No files found"

    @tool
    def fs_read_file(path: str) -> str:
        """[FileSystem Server] 파일 내용을 읽습니다"""
        return files_storage.get(path, "File not found")

    # === Database Server Tools ===
    @tool
    def db_get_stats(table: str) -> str:
        """[Database Server] 테이블 통계를 조회합니다"""
        stats = {
            "users": {"total": 1250, "active": 980, "inactive": 270},
            "orders": {"total": 5420, "completed": 5100, "pending": 320},
            "products": {"total": 180, "in_stock": 165, "out_of_stock": 15}
        }
        return json.dumps(stats.get(table, {}), ensure_ascii=False)

    @tool
    def db_search(table: str, query: str) -> str:
        """[Database Server] 테이블에서 검색합니다"""
        return f"Searching '{query}' in {table} table... (simulated)"

    # === Notification Server Tools ===
    @tool
    def notif_send(recipient: str, message: str, channel: str = "email") -> str:
        """[Notification Server] 알림을 발송합니다
        
        Args:
            recipient: 수신자
            message: 메시지 내용
            channel: 채널 (email, slack, sms)
        """
        return f"✅ {channel} 알림 발송 완료: {recipient}에게 '{message[:30]}...' 전송"

    # === Analytics Server Tools ===
    @tool
    def analytics_summary(period: str) -> str:
        """[Analytics Server] 기간별 요약 통계를 제공합니다
        
        Args:
            period: 기간 (daily, weekly, monthly)
        """
        summaries = {
            "daily": "오늘 방문자: 1,240명, 신규 가입: 12명, 주문: 45건",
            "weekly": "주간 방문자: 8,500명, 신규 가입: 85명, 주문: 320건",
            "monthly": "월간 방문자: 35,000명, 신규 가입: 340명, 주문: 1,420건"
        }
        return summaries.get(period, "Invalid period")

    @tool
    def analytics_compare(metric: str, period1: str, period2: str) -> str:
        """[Analytics Server] 두 기간의 지표를 비교합니다
        
        Args:
            metric: 비교할 지표 (sales, users, orders)
            period1: 첫 번째 기간
            period2: 두 번째 기간
        """
        return f"{metric} 비교: {period1} vs {period2} → +15% 증가 (시뮬레이션)"

    # 모든 Tools 수집
    all_tools = [
        fs_list_files, fs_read_file,
        db_get_stats, db_search,
        notif_send,
        analytics_summary, analytics_compare
    ]

    print(f"\n📦 총 {len(all_tools)}개 Tools (4개 서버)")
    print("-" * 70)

    servers = {
        "FileSystem": [fs_list_files, fs_read_file],
        "Database": [db_get_stats, db_search],
        "Notification": [notif_send],
        "Analytics": [analytics_summary, analytics_compare]
    }

    for server_name, tools in servers.items():
        print(f"\n🖥️  {server_name} Server:")
        for t in tools:
            print(f"    • {t.name}")

    # Agent 생성
    agent = create_agent(
        model="gpt-4o-mini",
        tools=all_tools,
        system_prompt="""당신은 여러 MCP 서버에 접근할 수 있는 통합 Assistant입니다.

사용 가능한 서버:
1. FileSystem: 파일 관리 (fs_*)
2. Database: 데이터 조회 (db_*)
3. Notification: 알림 발송 (notif_*)
4. Analytics: 통계 분석 (analytics_*)

작업에 맞는 서버의 Tools를 선택하여 사용하세요."""
    )

    # 복잡한 통합 작업
    integrated_tasks = [
        "/reports 디렉토리에 어떤 파일들이 있나요?",
        "users 테이블의 통계를 알려주세요",
        "월간 통계 요약을 보여주세요",
        "sales_2024.txt 파일의 내용을 읽고, 그 정보를 관리자에게 이메일로 보내주세요"
    ]

    print("\n" + "=" * 70)
    print("🧪 통합 시스템 테스트")
    print("=" * 70)

    for task in integrated_tasks:
        print(f"\n📋 작업: {task}")
        print("=" * 70)

        response = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })

        answer = response['messages'][-1].content
        print(f"\n🤖 결과:\n{answer}\n")

    # 사용자 입력
    print("=" * 70)
    print("💬 직접 작업을 요청해보세요 (종료: 'quit' 입력)")
    print("=" * 70)

    user_task = input("\n📋 작업: ").strip()

    if user_task and user_task.lower() != 'quit':
        print("\n🔍 처리 중...\n")

        response = agent.invoke({
            "messages": [{"role": "user", "content": user_task}]
        })

        answer = response['messages'][-1].content
        print(f"🤖 결과:\n{answer}\n")

    print("=" * 70)
    print("✅ MCP 스타일 통합 시스템 완료!")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 함수"""
    print("\n")
    print("=" * 70)
    print("Part 8: MCP 클라이언트 (04_mcp_client.py)")
    print("=" * 70)

    while True:
        print("\n📚 실행할 예제를 선택하세요:")
        print("  1. MCP 기본 개념")
        print("  2. MCP Tool 시뮬레이션")
        print("  3. Tool 메타데이터 검사")
        print("  4. 여러 Tool 조합 사용")
        print("  5. 실전 MCP 스타일 통합 시스템 ⭐")
        print("  0. 종료")

        choice = input("\n선택 (0-5): ").strip()

        if choice == "1":
            example_1_mcp_basics()
        elif choice == "2":
            example_2_simulated_mcp_tools()
        elif choice == "3":
            example_3_tool_metadata()
        elif choice == "4":
            example_4_multiple_tools()
        elif choice == "5":
            example_5_integrated_system()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")


if __name__ == "__main__":
    main()
