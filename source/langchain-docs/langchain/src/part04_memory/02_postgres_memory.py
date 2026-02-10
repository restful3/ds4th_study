"""
================================================================================
LangChain AI Agent 마스터 교안
Part 4: Memory System - PostgreSQL Memory
================================================================================

파일명: 02_postgres_memory.py
난이도: ⭐⭐⭐☆☆ (중급)
예상 시간: 25분

📚 학습 목표:
  - PostgresSaver를 사용한 영구 메모리 저장
  - 데이터베이스 기반 Checkpointer 설정
  - Production 환경을 위한 메모리 관리
  - Thread 관리 및 정리

📖 공식 문서:
  • Short-term Memory: /official/10-short-term-memory.md

📄 교안 문서:
  • Part 4 메모리: /docs/part04_memory.md (Section 2)

🔧 필요한 패키지:
  pip install langchain langchain-openai langgraph langgraph-checkpoint-postgres psycopg2-binary python-dotenv

🔑 필요한 환경변수:
  - OPENAI_API_KEY
  - DATABASE_URL (선택사항, 기본값 제공)

⚠️  주의사항:
  - PostgreSQL 데이터베이스가 필요합니다
  - Docker로 간단히 설정 가능:
    docker run -d --name postgres-langchain \\
      -e POSTGRES_USER=langchain \\
      -e POSTGRES_PASSWORD=langchain \\
      -e POSTGRES_DB=langchain_memory \\
      -p 5432:5432 postgres:15

🚀 실행 방법:
  python 02_postgres_memory.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

# ============================================================================
# 환경 설정
# ============================================================================

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    print("📝 src/.env 파일을 확인하고 API 키를 설정하세요.")
    exit(1)

# PostgreSQL 연결 문자열
# 형식: postgresql://user:password@host:port/database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://langchain:langchain@localhost:5432/langchain_memory"
)

# ============================================================================
# PostgreSQL 사용 가능 여부 확인
# ============================================================================

def check_postgres_availability():
    """PostgreSQL 사용 가능 여부 확인"""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        import psycopg2

        # 연결 테스트
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        return True
    except ImportError:
        print("⚠️  langgraph-checkpoint-postgres 패키지가 설치되지 않았습니다.")
        print("📦 설치 명령: pip install langgraph-checkpoint-postgres psycopg2-binary")
        return False
    except Exception as e:
        print(f"⚠️  PostgreSQL 연결 실패: {e}")
        print("\n💡 Docker로 PostgreSQL 시작:")
        print("   docker run -d --name postgres-langchain \\")
        print("     -e POSTGRES_USER=langchain \\")
        print("     -e POSTGRES_PASSWORD=langchain \\")
        print("     -e POSTGRES_DB=langchain_memory \\")
        print("     -p 5432:5432 postgres:15")
        return False

# ============================================================================
# 예제용 Tools
# ============================================================================

@tool
def save_note(note: str) -> str:
    """중요한 메모를 저장합니다.

    Args:
        note: 저장할 메모 내용
    """
    return f"✅ 메모 저장됨: '{note}'"

@tool
def get_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================================================================
# 예제 1: PostgresSaver 설정
# ============================================================================

def example_1_postgres_setup():
    """PostgresSaver 설정 및 초기화"""
    print("=" * 70)
    print("📌 예제 1: PostgresSaver 설정")
    print("=" * 70)
    print("\n💡 PostgresSaver는 데이터베이스에 대화 이력을 영구 저장합니다.\n")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        print(f"📦 데이터베이스 URL: {DATABASE_URL}\n")

        # PostgresSaver 생성 및 테이블 설정
        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            print("🔧 데이터베이스 연결 성공!")

            # 필요한 테이블 자동 생성
            checkpointer.setup()
            print("✅ 체크포인트 테이블 생성/확인 완료!")

            # Agent 생성
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[save_note, get_time],
                checkpointer=checkpointer,
            )

            print("✅ Agent 생성 완료!\n")

            # 간단한 대화 테스트
            config: RunnableConfig = {"configurable": {"thread_id": "setup-test"}}

            print("🔹 테스트 대화:")
            print("👤 사용자: 안녕하세요!")

            result = agent.invoke(
                {"messages": [{"role": "user", "content": "안녕하세요!"}]},
                config
            )

            print(f"🤖 AI: {result['messages'][-1].content}\n")
            print("✅ PostgresSaver가 정상적으로 작동합니다!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n💡 문제 해결:")
        print("  1. PostgreSQL이 실행 중인지 확인")
        print("  2. DATABASE_URL이 올바른지 확인")
        print("  3. 필요한 패키지가 설치되어 있는지 확인")


# ============================================================================
# 예제 2: 데이터베이스에 저장 및 복원
# ============================================================================

def example_2_save_and_load():
    """데이터베이스에 저장하고 나중에 복원"""
    print("\n" + "=" * 70)
    print("📌 예제 2: 데이터베이스 저장 및 복원")
    print("=" * 70)
    print("\n💡 프로그램을 종료해도 대화 이력이 유지됩니다.\n")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            checkpointer.setup()

            agent = create_agent(
                model="gpt-4o-mini",
                tools=[save_note],
                checkpointer=checkpointer,
            )

            thread_id = "persistent-user"
            config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

            print(f"🧵 Thread ID: {thread_id}\n")

            # 첫 번째 세션: 정보 저장
            print("=" * 50)
            print("📝 세션 1: 정보 저장")
            print("=" * 50)

            messages = [
                "제 이름은 박지민입니다.",
                "저는 파이썬 개발자예요.",
                "'프로젝트 마감: 금요일'이라고 메모해주세요."
            ]

            for i, msg in enumerate(messages, 1):
                print(f"\n대화 {i}:")
                print(f"👤 사용자: {msg}")
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": msg}]},
                    config
                )
                print(f"🤖 AI: {result['messages'][-1].content}")

            print("\n💾 모든 대화가 데이터베이스에 저장되었습니다!")

        # 새로운 연결로 복원 (프로그램 재시작 시뮬레이션)
        print("\n" + "=" * 50)
        print("🔄 세션 2: 이전 대화 복원 (새 연결)")
        print("=" * 50)

        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            agent = create_agent(
                model="gpt-4o-mini",
                tools=[save_note],
                checkpointer=checkpointer,
            )

            # 같은 Thread ID로 이전 대화 복원
            print(f"\n🧵 Thread ID: {thread_id}")
            print("👤 사용자: 제 이름과 직업이 뭐라고 했죠?")

            result = agent.invoke(
                {"messages": [{"role": "user", "content": "제 이름과 직업이 뭐라고 했죠?"}]},
                config
            )

            print(f"🤖 AI: {result['messages'][-1].content}\n")
            print("✅ 데이터베이스에서 이전 대화를 성공적으로 복원했습니다!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# ============================================================================
# 예제 3: 여러 사용자/Thread 관리
# ============================================================================

def example_3_multiple_users():
    """여러 사용자의 Thread를 데이터베이스에서 관리"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 다중 사용자 관리")
    print("=" * 70)
    print("\n💡 PostgresSaver는 수천 개의 Thread를 효율적으로 관리할 수 있습니다.\n")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            checkpointer.setup()

            agent = create_agent(
                model="gpt-4o-mini",
                tools=[],
                checkpointer=checkpointer,
            )

            # 여러 사용자 시뮬레이션
            users = [
                ("user-001", "김철수", "서울"),
                ("user-002", "이영희", "부산"),
                ("user-003", "박민수", "대구"),
            ]

            print("=" * 50)
            print("📝 여러 사용자의 대화 저장")
            print("=" * 50)

            for user_id, name, city in users:
                config: RunnableConfig = {"configurable": {"thread_id": user_id}}

                print(f"\n🧵 {user_id}:")
                print(f"👤 {name}: 제 이름은 {name}이고 {city}에 살아요.")

                result = agent.invoke(
                    {"messages": [{"role": "user", "content": f"제 이름은 {name}이고 {city}에 살아요."}]},
                    config
                )

                print(f"🤖 AI: {result['messages'][-1].content}")

            # 각 사용자의 정보 확인
            print("\n" + "=" * 50)
            print("🔍 각 사용자의 정보 확인")
            print("=" * 50)

            for user_id, name, _ in users:
                config: RunnableConfig = {"configurable": {"thread_id": user_id}}

                print(f"\n🧵 {user_id}:")
                print(f"👤 질문: 제 이름과 사는 곳이 어디죠?")

                result = agent.invoke(
                    {"messages": [{"role": "user", "content": "제 이름과 사는 곳이 어디죠?"}]},
                    config
                )

                print(f"🤖 AI: {result['messages'][-1].content}")

            print("\n✅ 모든 사용자의 Thread가 독립적으로 관리되고 있습니다!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# ============================================================================
# 예제 4: Thread 정리 (Cleanup)
# ============================================================================

def example_4_cleanup():
    """오래된 Thread 정리"""
    print("\n" + "=" * 70)
    print("📌 예제 4: Thread 정리")
    print("=" * 70)
    print("\n💡 불필요한 Thread를 정리하여 데이터베이스를 관리할 수 있습니다.\n")

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from datetime import datetime

        with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            checkpointer.setup()

            # 테스트용 Thread 생성
            print("=" * 50)
            print("📝 테스트 Thread 생성")
            print("=" * 50)

            agent = create_agent(
                model="gpt-4o-mini",
                tools=[],
                checkpointer=checkpointer,
            )

            test_threads = ["cleanup-test-1", "cleanup-test-2", "cleanup-test-3"]

            for thread_id in test_threads:
                config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
                agent.invoke(
                    {"messages": [{"role": "user", "content": "테스트 메시지"}]},
                    config
                )
                print(f"✅ Thread 생성: {thread_id}")

            # Thread 목록 조회 방법 안내
            print("\n" + "=" * 50)
            print("🗑️  Thread 정리 방법")
            print("=" * 50)

            print("\n💡 PostgresSaver는 SQL을 통해 Thread를 관리할 수 있습니다:")
            print("""
-- 모든 Thread 조회
SELECT DISTINCT thread_id, MAX(created_at) as last_update
FROM checkpoints
GROUP BY thread_id
ORDER BY last_update DESC;

-- 특정 Thread 삭제
DELETE FROM checkpoints WHERE thread_id = 'thread-to-delete';

-- 30일 이상 오래된 Thread 삭제
DELETE FROM checkpoints
WHERE created_at < NOW() - INTERVAL '30 days';
            """)

            print("⚠️  주의: Thread 삭제는 신중하게 수행하세요!")
            print("💡 GDPR 등 개인정보 보호 규정을 준수하세요.\n")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# ============================================================================
# 예제 5: Production 베스트 프랙티스
# ============================================================================

def example_5_production_tips():
    """Production 환경을 위한 팁과 패턴"""
    print("\n" + "=" * 70)
    print("📌 예제 5: Production 베스트 프랙티스")
    print("=" * 70)

    print("""
💡 PostgresSaver Production 가이드:

1️⃣  연결 풀 설정
   - 많은 요청을 처리하려면 연결 풀 사용
   - SQLAlchemy Engine 사용 권장:

   from sqlalchemy import create_engine
   from sqlalchemy.pool import QueuePool

   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=10,
       max_overflow=20,
   )

   checkpointer = PostgresSaver(engine)

2️⃣  Thread ID 설계
   - 일관된 네이밍 규칙 사용
   - 예: {user_id}-{session_id}
   - 예: user-{uuid}
   - 중복 방지를 위해 UUID 사용 고려

3️⃣  정기적인 정리
   - 오래된 Thread 자동 삭제
   - 로그 분석으로 비활성 Thread 식별
   - 백업 후 삭제 권장

4️⃣  모니터링
   - 데이터베이스 크기 모니터링
   - Thread 생성 속도 추적
   - 평균 대화 길이 분석

5️⃣  보안
   - DATABASE_URL을 환경변수로 관리
   - SSL 연결 사용 (sslmode=require)
   - 최소 권한 원칙 적용

6️⃣  백업
   - 정기적인 데이터베이스 백업
   - 복구 절차 테스트
   - 중요한 Thread는 별도 보관

7️⃣  성능 최적화
   - 인덱스 적절히 설정
   - 파티셔닝 고려 (대규모)
   - 읽기 전용 복제본 활용

8️⃣  에러 처리
   - 연결 실패 시 재시도 로직
   - Fallback 메커니즘 (InMemorySaver)
   - 상세한 로깅

예제 코드:

```python
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def get_checkpointer(db_url: str):
    '''안전한 Checkpointer 사용'''
    checkpointer = None
    try:
        checkpointer = PostgresSaver.from_conn_string(db_url)
        checkpointer.setup()
        yield checkpointer
    except Exception as e:
        logger.error(f"Checkpointer 오류: {e}")
        # Fallback to InMemorySaver
        from langgraph.checkpoint.memory import InMemorySaver
        logger.warning("InMemorySaver로 대체")
        yield InMemorySaver()
    finally:
        if checkpointer:
            checkpointer.close()
```

9️⃣  Thread 아카이빙
   - 오래된 Thread를 별도 테이블로 이동
   - 압축 저장 고려
   - 필요 시 복원 가능하도록 유지

🔟 비용 최적화
   - 불필요한 메시지 정기 삭제
   - Message Trim/Summarization 활용
   - 저장 용량 모니터링
    """)


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n")
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 4: Memory System - PostgreSQL Memory")
    print("\n")

    # PostgreSQL 사용 가능 확인
    if not check_postgres_availability():
        print("\n❌ PostgreSQL을 사용할 수 없습니다.")
        print("💡 InMemorySaver를 먼저 학습하거나 PostgreSQL을 설정하세요.\n")
        return

    # 예제 1: 설정
    example_1_postgres_setup()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 2: 저장 및 로드
    example_2_save_and_load()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 3: 다중 사용자
    example_3_multiple_users()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 4: 정리
    example_4_cleanup()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    # 예제 5: Production 팁
    example_5_production_tips()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 4-2 예제를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 03_message_trim.py - Message Management")
    print("  2. 04_summarization.py - Message Summarization")
    print("  3. 05_custom_state.py - Custom State")
    print("\n📚 핵심 개념 복습:")
    print("  • PostgresSaver: 데이터베이스 기반 영구 저장")
    print("  • Production-ready: 재시작 후에도 데이터 유지")
    print("  • Scalable: 수천 개의 Thread 관리 가능")
    print("  • Thread Cleanup: 정기적인 데이터 정리 필요")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 다른 데이터베이스:
#    - SQLiteSaver: 파일 기반, 단일 인스턴스
#    - MongoDBSaver: NoSQL 옵션
#    - Custom Checkpointer: 직접 구현 가능
#
# 2. 연결 문자열 형식:
#    postgresql://[user[:password]@][host][:port][/database]
#    예: postgresql://user:pass@localhost:5432/mydb
#
# 3. SSL 연결:
#    postgresql://user:pass@host:5432/db?sslmode=require
#
# 4. 환경별 설정:
#    - Development: InMemorySaver 또는 SQLite
#    - Staging: PostgreSQL (테스트 DB)
#    - Production: PostgreSQL (복제, 백업 구성)
#
# ============================================================================
# 🐛 자주 발생하는 문제
# ============================================================================
#
# 문제: "connection refused"
# 해결: PostgreSQL이 실행 중인지 확인
#       docker ps로 컨테이너 상태 확인
#
# 문제: "database does not exist"
# 해결: 데이터베이스 생성
#       createdb langchain_memory
#
# 문제: "permission denied"
# 해결: 사용자 권한 확인
#       GRANT ALL ON DATABASE langchain_memory TO user;
#
# 문제: "too many connections"
# 해결: 연결 풀 크기 조정
#       max_connections 설정 확인
#
# ============================================================================
