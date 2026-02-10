"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP
================================================================================

파일명: 05_mcp_server.py
난이도: ⭐⭐⭐⭐ (중상급)
예상 시간: 35분

📚 학습 목표:
  - FastMCP로 서버 기본 구조 이해
  - @mcp.tool()로 tool 제공
  - @mcp.resource()로 resource 제공
  - 서버 실행 및 테스트
  - 실전 커스텀 도구 MCP 서버

📖 공식 문서:
  • FastMCP: https://github.com/jlowin/fastmcp
  • MCP: https://modelcontextprotocol.io/

📄 교안 문서:
  • Part 8: /docs/part08_rag_mcp.md

🔧 필요한 패키지:
  pip install langchain langchain-openai python-dotenv

🚀 실행 방법:
  python 05_mcp_server.py

================================================================================
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)

def example_1_server_basics():
    """MCP 서버 기본 개념"""
    print("=" * 70)
    print("📌 예제 1: MCP 서버 기본 구조")
    print("=" * 70)
    
    print("""
💡 MCP 서버란?
   - Tools, Resources, Prompts를 제공하는 프로세스
   - FastMCP 라이브러리로 쉽게 구현
   - stdio 또는 HTTP transport 지원

기본 구조:

from fastmcp import FastMCP

# 서버 초기화
mcp = FastMCP("MyServer", description="나만의 MCP 서버")

# Tool 정의
@mcp.tool()
def my_tool(param: str) -> str:
    \"\"\"도구 설명\"\"\"
    return f"처리 결과: {param}"

# 서버 실행
if __name__ == "__main__":
    mcp.run(transport="stdio")  # 또는 "http"

📦 제공 요소:

1. Tools (도구)
   - 실행 가능한 함수
   - @mcp.tool() 데코레이터
   - 명확한 타입 힌트 필요

2. Resources (리소스)
   - 읽기 가능한 데이터
   - @mcp.resource() 데코레이터
   - URI 패턴 지원

3. Prompts (프롬프트)
   - 재사용 가능한 템플릿
   - @mcp.prompt() 데코레이터

🔧 Transport 방식:

1. stdio (표준 입출력)
   - 로컬 프로세스 간 통신
   - 간단하고 빠름
   - mcp.run(transport="stdio")

2. HTTP
   - 원격 서버 통신
   - 네트워크를 통한 접근
   - mcp.run(transport="http", port=8000)

📝 서버 생명주기:

1. 초기화: FastMCP 객체 생성
2. 등록: @mcp.tool(), @mcp.resource() 데코레이터로 등록
3. 실행: mcp.run()으로 서버 시작
4. 대기: 클라이언트 요청 수신
5. 처리: Tool 실행 및 결과 반환
6. 종료: Ctrl+C 또는 종료 시그널
    """)
    print("\n" + "=" * 70)

def example_2_tool_server():
    """Tool을 제공하는 MCP 서버 시뮬레이션"""
    print("\n" + "=" * 70)
    print("📌 예제 2: Tool 제공 MCP 서버")
    print("=" * 70)
    
    print("""
💡 Tool 제공 서버:
   - 여러 도구를 하나의 서버에서 제공
   - 각 도구는 명확한 책임 분리
   - 타입 힌트로 자동 검증

예시 코드:

from fastmcp import FastMCP

mcp = FastMCP("MathServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"두 숫자를 더합니다\"\"\"
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    \"\"\"두 숫자를 곱합니다\"\"\"
    return a * b

@mcp.tool()
def calculate_stats(numbers: list[float]) -> dict:
    \"\"\"숫자 리스트의 통계를 계산합니다\"\"\"
    return {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers)
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
    """)
    
    # 시뮬레이션된 Tool 서버 동작
    print("\n🧪 시뮬레이션: Tool 서버 동작")
    print("-" * 70)
    
    class MockMCPServer:
        def __init__(self, name):
            self.name = name
            self.tools = {}
        
        def tool(self):
            def decorator(func):
                self.tools[func.__name__] = {
                    "name": func.__name__,
                    "description": func.__doc__ or "",
                    "function": func
                }
                return func
            return decorator
        
        def list_tools(self):
            return [{"name": name, "description": info["description"]} 
                    for name, info in self.tools.items()]
        
        def call_tool(self, name, **kwargs):
            if name in self.tools:
                return self.tools[name]["function"](**kwargs)
            raise ValueError(f"Tool {name} not found")
    
    # 서버 생성
    server = MockMCPServer("CalculatorServer")
    
    @server.tool()
    def add(a: int, b: int) -> int:
        """두 숫자를 더합니다"""
        return a + b
    
    @server.tool()
    def multiply(a: int, b: int) -> int:
        """두 숫자를 곱합니다"""
        return a * b
    
    @server.tool()
    def power(base: int, exponent: int) -> int:
        """거듭제곱을 계산합니다"""
        return base ** exponent
    
    print(f"✅ 서버 '{server.name}' 초기화 완료")
    print(f"\n📦 제공 Tools ({len(server.tools)}개):")
    for tool in server.list_tools():
        print(f"  • {tool['name']}: {tool['description']}")
    
    # Tool 호출 테스트
    print("\n🧪 Tool 호출 테스트:")
    test_calls = [
        ("add", {"a": 5, "b": 3}),
        ("multiply", {"a": 4, "b": 7}),
        ("power", {"base": 2, "exponent": 8})
    ]
    
    for tool_name, params in test_calls:
        result = server.call_tool(tool_name, **params)
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {tool_name}({params_str}) = {result}")
    
    print("\n" + "=" * 70)

def example_3_resource_server():
    """Resource를 제공하는 MCP 서버"""
    print("\n" + "=" * 70)
    print("📌 예제 3: Resource 제공 MCP 서버")
    print("=" * 70)
    
    print("""
💡 Resource란?
   - 읽기 가능한 데이터
   - URI 패턴으로 접근
   - 동적/정적 리소스 모두 지원

예시 코드:

from fastmcp import FastMCP

mcp = FastMCP("DataServer")

# 정적 리소스
@mcp.resource("config://app")
def get_app_config():
    \"\"\"애플리케이션 설정 반환\"\"\"
    return {
        "version": "1.0",
        "env": "production"
    }

# 동적 리소스 (URI 패턴)
@mcp.resource("file://{path}")
async def read_file(path: str) -> str:
    \"\"\"파일 읽기\"\"\"
    async with aiofiles.open(path, 'r') as f:
        return await f.read()

# 리소스 목록
@mcp.resource("list://files")
def list_files() -> list[str]:
    \"\"\"파일 목록 반환\"\"\"
    import os
    return os.listdir("./data")
    """)
    
    # Resource 서버 시뮬레이션
    print("\n🧪 시뮬레이션: Resource 서버")
    print("-" * 70)
    
    class MockResourceServer:
        def __init__(self):
            self.resources = {
                "config://app": {
                    "version": "2.0",
                    "env": "development",
                    "debug": True
                },
                "config://database": {
                    "host": "localhost",
                    "port": 5432,
                    "name": "app_db"
                },
                "data://users": [
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"}
                ],
                "stats://today": {
                    "visitors": 1240,
                    "orders": 45,
                    "revenue": 12500
                }
            }
        
        def get_resource(self, uri: str):
            return self.resources.get(uri, {"error": "Resource not found"})
        
        def list_resources(self):
            return list(self.resources.keys())
    
    server = MockResourceServer()
    
    print("📚 사용 가능한 Resources:")
    for uri in server.list_resources():
        print(f"  • {uri}")
    
    print("\n🔍 Resource 조회 테스트:")
    test_uris = ["config://app", "config://database", "stats://today"]
    
    for uri in test_uris:
        data = server.get_resource(uri)
        print(f"\n  {uri}:")
        print(f"  {json.dumps(data, indent=4, ensure_ascii=False)}")
    
    print("\n" + "=" * 70)

def example_4_combined_server():
    """Tool과 Resource를 모두 제공하는 서버"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Tool + Resource 통합 서버")
    print("=" * 70)
    
    print("""
💡 통합 서버:
   - Tools와 Resources를 함께 제공
   - 각각 독립적으로 동작
   - 클라이언트는 필요에 따라 선택

예시 코드:

from fastmcp import FastMCP

mcp = FastMCP("IntegratedServer")

# Tools
@mcp.tool()
def process_data(data: str) -> str:
    \"\"\"데이터 처리\"\"\"
    return data.upper()

# Resources
@mcp.resource("config://settings")
def get_settings():
    \"\"\"설정 반환\"\"\"
    return {"theme": "dark", "language": "ko"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
    """)
    
    print("\n🧪 시뮬레이션: 통합 서버")
    print("-" * 70)
    
    class IntegratedServer:
        def __init__(self, name):
            self.name = name
            self.tools = {}
            self.resources = {}
        
        def add_tool(self, name, func, description):
            self.tools[name] = {"func": func, "description": description}
        
        def add_resource(self, uri, data):
            self.resources[uri] = data
        
        def call_tool(self, name, **kwargs):
            if name in self.tools:
                return self.tools[name]["func"](**kwargs)
            raise ValueError(f"Tool {name} not found")
        
        def get_resource(self, uri):
            return self.resources.get(uri, {"error": "Not found"})
        
        def list_all(self):
            return {
                "tools": list(self.tools.keys()),
                "resources": list(self.resources.keys())
            }
    
    # 서버 초기화
    server = IntegratedServer("CompanyServer")
    
    # Tools 추가
    server.add_tool(
        "calculate_salary",
        lambda base, bonus: base + bonus,
        "급여 계산"
    )
    server.add_tool(
        "format_name",
        lambda first, last: f"{last}, {first}",
        "이름 형식화"
    )
    
    # Resources 추가
    server.add_resource("policy://vacation", {
        "annual_days": 15,
        "sick_days": 10
    })
    server.add_resource("policy://remote", {
        "max_days_per_week": 2,
        "approval_required": True
    })
    
    print(f"✅ '{server.name}' 초기화 완료\n")
    
    available = server.list_all()
    print("📦 Tools:")
    for tool in available["tools"]:
        print(f"  • {tool}")
    
    print("\n📚 Resources:")
    for resource in available["resources"]:
        print(f"  • {resource}")
    
    print("\n🧪 Tool 호출:")
    salary = server.call_tool("calculate_salary", base=5000000, bonus=1000000)
    print(f"  calculate_salary(5000000, 1000000) = {salary:,}원")
    
    name = server.call_tool("format_name", first="철수", last="김")
    print(f"  format_name('철수', '김') = {name}")
    
    print("\n🔍 Resource 조회:")
    vacation = server.get_resource("policy://vacation")
    print(f"  policy://vacation = {json.dumps(vacation, ensure_ascii=False)}")
    
    print("\n" + "=" * 70)

def example_5_production_server():
    """실전 MCP 서버 디자인"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 실전 프로덕션 서버 설계")
    print("=" * 70)
    
    print("""
💡 실전 시나리오:
   - 회사 내부 시스템 MCP 서버
   - 사용자 관리, 문서 검색, 통계 제공
   - 에러 처리 및 로깅 포함

서버 구성:
   1. User Management Tools
   2. Document Search Tools
   3. Analytics Tools
   4. Configuration Resources
   5. System Status Resources

예시 코드:

from fastmcp import FastMCP
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "CompanyServer",
    description="회사 내부 시스템 통합 서버"
)

# ===== User Management Tools =====

@mcp.tool()
def search_user(username: str) -> dict:
    \"\"\"사용자 정보 검색
    
    Args:
        username: 검색할 사용자 이름
        
    Returns:
        사용자 정보 딕셔너리
    \"\"\"
    logger.info(f"Searching user: {username}")
    
    # 실제로는 DB 조회
    users_db = {
        "alice": {"id": 1, "name": "Alice", "dept": "Engineering"},
        "bob": {"id": 2, "name": "Bob", "dept": "Design"}
    }
    
    return users_db.get(username.lower(), {"error": "User not found"})

@mcp.tool()
def send_notification(
    user_id: int,
    message: str,
    channel: str = "email"
) -> str:
    \"\"\"사용자에게 알림 발송
    
    Args:
        user_id: 대상 사용자 ID
        message: 알림 메시지
        channel: 발송 채널 (email, slack, sms)
    \"\"\"
    logger.info(f"Sending {channel} notification to user {user_id}")
    
    # 실제로는 알림 시스템 연동
    return f"Notification sent to user {user_id} via {channel}"

# ===== Document Search Tools =====

@mcp.tool()
def search_documents(query: str, limit: int = 10) -> list[dict]:
    \"\"\"문서 검색
    
    Args:
        query: 검색 쿼리
        limit: 결과 개수 제한
    \"\"\"
    logger.info(f"Searching documents: {query}")
    
    # 실제로는 Vector Store 검색
    return [
        {"id": 1, "title": "Q1 Report", "score": 0.95},
        {"id": 2, "title": "User Guide", "score": 0.87}
    ][:limit]

# ===== Analytics Tools =====

@mcp.tool()
def get_analytics(metric: str, period: str = "today") -> dict:
    \"\"\"분석 데이터 조회
    
    Args:
        metric: 지표명 (visitors, orders, revenue)
        period: 기간 (today, week, month)
    \"\"\"
    logger.info(f"Getting analytics: {metric} for {period}")
    
    # 실제로는 분석 DB 조회
    return {
        "metric": metric,
        "period": period,
        "value": 1240,
        "change": "+15%"
    }

# ===== Configuration Resources =====

@mcp.resource("config://database")
def get_db_config():
    \"\"\"데이터베이스 설정\"\"\"
    return {
        "host": "localhost",
        "port": 5432,
        "database": "company_db"
    }

@mcp.resource("config://features")
def get_feature_flags():
    \"\"\"기능 플래그\"\"\"
    return {
        "new_ui": True,
        "beta_features": False,
        "analytics": True
    }

# ===== System Status Resources =====

@mcp.resource("status://health")
def get_health_status():
    \"\"\"시스템 헬스 체크\"\"\"
    return {
        "status": "healthy",
        "uptime": "99.9%",
        "last_check": datetime.now().isoformat()
    }

# ===== 서버 실행 =====

if __name__ == "__main__":
    logger.info("Starting Company MCP Server...")
    
    # stdio transport (로컬)
    mcp.run(transport="stdio")
    
    # 또는 HTTP transport (원격)
    # mcp.run(transport="http", port=8000)

💡 프로덕션 체크리스트:

✅ 필수 요소:
  □ 명확한 Tool 설명 (docstring)
  □ 타입 힌트 (type hints)
  □ 에러 처리 (try-except)
  □ 로깅 (logging)
  □ 입력 검증 (validation)
  □ 문서화 (documentation)
  
✅ 보안:
  □ 인증/인가 (authentication)
  □ Rate limiting
  □ 입력 sanitization
  □ 민감 정보 보호
  
✅ 모니터링:
  □ 헬스 체크 endpoint
  □ 메트릭 수집
  □ 에러 추적
    """)
    
    print("\n" + "=" * 70)

def main():
    """메인 함수"""
    print("\n")
    print("=" * 70)
    print("Part 8: MCP 서버 구현 (05_mcp_server.py)")
    print("=" * 70)
    
    while True:
        print("\n📚 실행할 예제를 선택하세요:")
        print("  1. MCP 서버 기본 개념")
        print("  2. Tool 제공 서버")
        print("  3. Resource 제공 서버")
        print("  4. Tool + Resource 통합 서버")
        print("  5. 실전 프로덕션 서버 설계 ⭐")
        print("  0. 종료")
        
        choice = input("\n선택 (0-5): ").strip()
        
        if choice == "1":
            example_1_server_basics()
        elif choice == "2":
            example_2_tool_server()
        elif choice == "3":
            example_3_resource_server()
        elif choice == "4":
            example_4_combined_server()
        elif choice == "5":
            example_5_production_server()
        elif choice == "0":
            print("\n👋 프로그램을 종료합니다.")
            break
        else:
            print("\n❌ 잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main()
