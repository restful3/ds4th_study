"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP - 실습 과제 3 해답
================================================================================

과제: MCP 기반 통합 시스템
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. MCP(Model Context Protocol) 개념 이해
2. 외부 도구 통합 시뮬레이션
3. 통합 Agent 시스템 구축

학습 목표:
- MCP 패턴 이해
- 외부 시스템 통합
- 확장 가능한 아키텍처

================================================================================
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from datetime import datetime
import json

# ============================================================================
# MCP 도구들 (시뮬레이션)
# ============================================================================

@tool
def filesystem_read(path: str) -> str:
    """파일 시스템에서 파일을 읽습니다.
    
    Args:
        path: 파일 경로
    """
    # 시뮬레이션
    mock_files = {
        "/data/users.json": '{"users": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]}',
        "/data/config.yaml": "database:\n  host: localhost\n  port: 5432",
        "/logs/app.log": "2024-01-22 10:00:00 INFO: Application started\n2024-01-22 10:01:00 INFO: User logged in",
    }
    
    if path in mock_files:
        return f"File content of {path}:\n\n{mock_files[path]}"
    else:
        return f"File not found: {path}"

@tool
def database_query(sql: str) -> str:
    """데이터베이스 쿼리를 실행합니다.
    
    Args:
        sql: SQL 쿼리
    """
    # 시뮬레이션
    return f"""Query executed: {sql}

Results:
| id | name    | email              |
|----|---------|-------------------|
| 1  | Alice   | alice@example.com |
| 2  | Bob     | bob@example.com   |
| 3  | Charlie | charlie@example.com |

(3 rows returned)
"""

@tool
def api_call(endpoint: str, method: str = "GET") -> str:
    """외부 API를 호출합니다.
    
    Args:
        endpoint: API 엔드포인트
        method: HTTP 메서드
    """
    # 시뮬레이션
    responses = {
        "/api/weather": '{"location": "Seoul", "temperature": 15, "condition": "Sunny"}',
        "/api/stock": '{"symbol": "AAPL", "price": 182.31, "change": +1.52}',
        "/api/news": '{"headlines": ["AI Breakthrough", "Tech Giants Merge"]}',
    }
    
    response = responses.get(endpoint, '{"error": "Endpoint not found"}')
    return f"API Response [{method} {endpoint}]:\n\n{response}"

@tool
def send_notification(recipient: str, message: str) -> str:
    """알림을 전송합니다.
    
    Args:
        recipient: 수신자
        message: 메시지 내용
    """
    return f"""✅ 알림 전송 완료

To: {recipient}
Message: {message}
Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: Delivered
"""

@tool
def schedule_task(task: str, time: str) -> str:
    """작업을 예약합니다.
    
    Args:
        task: 작업 내용
        time: 실행 시간
    """
    return f"""📅 작업 예약 완료

Task: {task}
Scheduled for: {time}
Task ID: TASK-{datetime.now().strftime('%Y%m%d-%H%M%S')}
Status: Scheduled
"""

@tool
def analyze_data(data_source: str, analysis_type: str = "summary") -> str:
    """데이터를 분석합니다.
    
    Args:
        data_source: 데이터 소스
        analysis_type: 분석 유형 (summary, trend, anomaly)
    """
    return f"""📊 데이터 분석 결과

Source: {data_source}
Analysis Type: {analysis_type}

Summary:
- Total Records: 1,234
- Average Value: 45.6
- Trend: Increasing (+12%)
- Anomalies: 3 detected

Recommendations:
1. Monitor anomaly at timestamp 2024-01-22 10:45
2. Consider scaling up resources
3. Review data quality for outliers
"""

# ============================================================================
# MCP 통합 Agent
# ============================================================================

def create_mcp_agent():
    """MCP 기반 통합 Agent 생성"""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    tools = [
        filesystem_read,
        database_query,
        api_call,
        send_notification,
        schedule_task,
        analyze_data,
    ]
    
    system_prompt = """당신은 MCP(Model Context Protocol)를 통해 다양한 시스템과 통합된 AI Agent입니다.

사용 가능한 도구:
- filesystem_read: 파일 시스템 접근
- database_query: 데이터베이스 쿼리
- api_call: 외부 API 호출
- send_notification: 알림 전송
- schedule_task: 작업 예약
- analyze_data: 데이터 분석

작업 수행 시:
1. 필요한 정보를 해당 도구로 수집
2. 여러 도구를 조합하여 복잡한 작업 수행
3. 결과를 명확하게 요약

사용자의 요청을 이해하고 적절한 도구를 선택하세요."""
    
    agent = create_react_agent(model, tools, state_modifier=system_prompt)
    return agent

# ============================================================================
# 테스트
# ============================================================================

def test_mcp_agent():
    """MCP Agent 테스트"""
    print("=" * 70)
    print("🔌 MCP 기반 통합 시스템 테스트")
    print("=" * 70)
    
    agent = create_mcp_agent()
    
    scenarios = [
        "사용자 데이터 파일을 읽고 분석해줘",
        "데이터베이스에서 사용자 목록을 조회하고, 첫 번째 사용자에게 환영 알림을 보내줘",
        "날씨 API를 호출하고, 그 결과를 기반으로 내일 오전 9시에 알림을 예약해줘",
        "애플리케이션 로그를 읽고 이상 징후를 분석해줘",
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'=' * 70}")
        print(f"📋 시나리오 {i}: {scenario}")
        print("=" * 70)
        
        result = agent.invoke({"messages": [HumanMessage(content=scenario)]})
        
        final_message = result["messages"][-1]
        print(f"\n✅ 완료:\n{final_message.content}\n")

def explain_mcp():
    """MCP 개념 설명"""
    print("\n" + "=" * 70)
    print("📚 MCP (Model Context Protocol) 이해하기")
    print("=" * 70)
    
    print("""
MCP란?
- AI 모델이 외부 시스템과 상호작용하기 위한 프로토콜
- 도구(Tools)를 통해 다양한 시스템 통합
- 확장 가능하고 표준화된 인터페이스

주요 구성요소:
1. Tools: 외부 시스템과의 인터페이스
2. Resources: 파일, 데이터베이스 등의 리소스
3. Prompts: 시스템별 프롬프트 템플릿

장점:
- 통합 관리: 하나의 Agent로 여러 시스템 제어
- 확장성: 새로운 도구 쉽게 추가 가능
- 재사용성: 도구를 여러 Agent에서 공유
- 표준화: 일관된 인터페이스

실제 사용 사례:
- 파일 시스템 접근
- 데이터베이스 쿼리
- API 호출
- 알림 전송
- 작업 자동화
- 데이터 분석

주의사항:
- 보안: 권한 관리 철저히
- 에러 처리: 외부 시스템 장애 대응
- 성능: 네트워크 지연 고려
- 로깅: 모든 작업 기록
    """)

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🔌 Part 8: MCP 기반 통합 시스템 - 실습 과제 3 해답")
    print("=" * 70)
    
    try:
        test_mcp_agent()
        explain_mcp()
        
        print("\n💡 학습 포인트:")
        print("  1. MCP(Model Context Protocol) 개념")
        print("  2. 외부 시스템 통합 패턴")
        print("  3. 다양한 도구 조합")
        print("  4. 확장 가능한 아키텍처")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
