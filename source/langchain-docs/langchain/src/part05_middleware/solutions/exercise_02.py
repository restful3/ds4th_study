"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: Middleware - 실습 과제 2 해답
================================================================================

과제: 캐싱 Middleware
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. 동일 질문에 대한 캐시 저장
2. 캐시 히트 시 LLM 호출 없이 즉시 응답
3. 캐시 만료 시간 및 최대 크기 관리

학습 목표:
- 캐싱 로직 구현
- 해시 기반 키 생성
- 캐시 히트율 측정

================================================================================
"""

from typing import Optional, Any
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import hashlib
import json
from collections import OrderedDict

# ============================================================================
# 캐시 엔트리
# ============================================================================

class CacheEntry:
    """캐시 항목"""

    def __init__(self, key: str, value: Any, ttl_seconds: int = 3600):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        self.hit_count = 0
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """만료 여부"""
        return datetime.now() > self.expires_at

    def access(self):
        """접근 기록"""
        self.hit_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed.isoformat(),
        }


# ============================================================================
# LRU 캐시
# ============================================================================

class LRUCache:
    """LRU (Least Recently Used) 캐시"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def _generate_key(self, messages: list) -> str:
        """메시지에서 캐시 키 생성"""
        # 메시지 내용을 정규화하여 해시 생성
        content = []
        for msg in messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                content.append(f"{msg.__class__.__name__}:{msg.content}")

        combined = "|".join(content)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, messages: list) -> Optional[str]:
        """캐시에서 조회"""
        self.stats["total_requests"] += 1

        key = self._generate_key(messages)

        # 캐시에 존재하는지 확인
        if key in self.cache:
            entry = self.cache[key]

            # 만료 확인
            if entry.is_expired():
                self.stats["expirations"] += 1
                del self.cache[key]
                self.stats["cache_misses"] += 1
                return None

            # 캐시 히트
            entry.access()
            self.cache.move_to_end(key)  # LRU 업데이트
            self.stats["cache_hits"] += 1

            print(f"✅ [Cache] 캐시 히트! (히트율: {self.get_hit_rate():.1%})")
            return entry.value

        # 캐시 미스
        self.stats["cache_misses"] += 1
        return None

    def put(self, messages: list, response: str):
        """캐시에 저장"""
        key = self._generate_key(messages)

        # 이미 존재하면 업데이트
        if key in self.cache:
            del self.cache[key]

        # 최대 크기 확인 및 LRU 제거
        if len(self.cache) >= self.max_size:
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
            print(f"⚠️  [Cache] LRU 제거: {oldest_key[:8]}... (히트 수: {oldest_entry.hit_count})")

        # 새 항목 추가
        entry = CacheEntry(key, response, self.ttl_seconds)
        self.cache[key] = entry

        print(f"💾 [Cache] 저장 완료 (크기: {len(self.cache)}/{self.max_size})")

    def clear(self):
        """캐시 초기화"""
        self.cache.clear()

    def get_hit_rate(self) -> float:
        """캐시 히트율"""
        if self.stats["total_requests"] == 0:
            return 0.0
        return self.stats["cache_hits"] / self.stats["total_requests"]

    def get_stats(self) -> dict:
        """통계 조회"""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "hit_rate": self.get_hit_rate(),
        }

    def print_report(self):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("📦 캐시 리포트")
        print("=" * 70)

        print(f"\n📊 통계:")
        print(f"  총 요청: {self.stats['total_requests']}")
        print(f"  캐시 히트: {self.stats['cache_hits']}")
        print(f"  캐시 미스: {self.stats['cache_misses']}")
        print(f"  히트율: {self.get_hit_rate():.1%}")
        print(f"  제거됨: {self.stats['evictions']}")
        print(f"  만료됨: {self.stats['expirations']}")

        print(f"\n💾 캐시 상태:")
        print(f"  현재 크기: {len(self.cache)}/{self.max_size}")

        if self.cache:
            print(f"\n📜 캐시 항목 (최대 5개):")
            for i, (key, entry) in enumerate(list(self.cache.items())[-5:], 1):
                age = (datetime.now() - entry.created_at).total_seconds()
                print(f"\n  {i}. {key[:16]}...")
                print(f"     생성: {int(age)}초 전")
                print(f"     히트 수: {entry.hit_count}")
                print(f"     응답: {entry.value[:50]}...")

        print("\n" + "=" * 70)


# ============================================================================
# 캐싱 Middleware
# ============================================================================

class CachingMiddleware:
    """캐싱 미들웨어"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size, ttl_seconds)

    def try_cache(self, messages: list) -> Optional[str]:
        """캐시 조회 시도"""
        return self.cache.get(messages)

    def save_to_cache(self, messages: list, response: str):
        """캐시에 저장"""
        self.cache.put(messages, response)

    def get_stats(self) -> dict:
        """통계 조회"""
        return self.cache.get_stats()

    def print_report(self):
        """리포트 출력"""
        self.cache.print_report()

    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()


# ============================================================================
# 캐싱 챗봇 구축
# ============================================================================

def create_caching_chatbot(
    model_name: str = "gpt-4o-mini",
    cache_size: int = 100,
    cache_ttl: int = 3600
):
    """캐싱 챗봇 생성"""

    model = ChatOpenAI(model=model_name, temperature=0)  # temperature=0으로 일관된 응답
    middleware = CachingMiddleware(max_size=cache_size, ttl_seconds=cache_ttl)

    # 챗봇 노드 (캐싱 통합)
    def chatbot_node(state: MessagesState) -> dict:
        """대화 응답 with 캐싱"""
        messages = state["messages"]

        # 1. 캐시 조회
        cached_response = middleware.try_cache(messages)

        if cached_response:
            # 캐시 히트 - LLM 호출 없이 반환
            return {"messages": [AIMessage(content=cached_response)]}

        # 2. 캐시 미스 - LLM 호출
        print(f"❌ [Cache] 캐시 미스 - LLM 호출")
        response = model.invoke(messages)

        # 3. 캐시에 저장
        middleware.save_to_cache(messages, response.content)

        return {"messages": [response]}

    # 그래프 구축
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph, middleware


# ============================================================================
# 테스트 함수
# ============================================================================

def test_basic_caching():
    """기본 캐싱 테스트"""
    print("=" * 70)
    print("📦 캐싱 Middleware 테스트")
    print("=" * 70)

    chatbot, middleware = create_caching_chatbot(cache_ttl=60)
    config = {"configurable": {"thread_id": "test_cache"}}

    # 테스트 질문들
    questions = [
        "파이썬이란 무엇인가요?",
        "자바스크립트란 무엇인가요?",
        "파이썬이란 무엇인가요?",  # 중복 - 캐시 히트
        "Go 언어란 무엇인가요?",
        "파이썬이란 무엇인가요?",  # 중복 - 캐시 히트
        "자바스크립트란 무엇인가요?",  # 중복 - 캐시 히트
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"👤 질문 {i}: {question}")
        print("=" * 70)

        # 시간 측정
        import time
        start_time = time.time()

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

        elapsed = time.time() - start_time

        ai_response = result["messages"][-1].content
        print(f"\n🤖 AI: {ai_response[:200]}...")
        print(f"⏱️  응답 시간: {elapsed:.2f}초")

    # 리포트 출력
    middleware.print_report()


def test_cache_expiration():
    """캐시 만료 테스트"""
    print("\n" + "=" * 70)
    print("⏰ 캐시 만료 테스트")
    print("=" * 70)

    # TTL을 짧게 설정 (5초)
    chatbot, middleware = create_caching_chatbot(cache_ttl=5)
    config = {"configurable": {"thread_id": "test_expiration"}}

    question = "테스트 질문입니다"

    # 첫 번째 요청
    print("\n1️⃣ 첫 번째 요청 (캐시 미스 예상)")
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )
    print(f"🤖 {result['messages'][-1].content[:100]}...")

    # 즉시 재요청
    print("\n2️⃣ 즉시 재요청 (캐시 히트 예상)")
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )
    print(f"🤖 {result['messages'][-1].content[:100]}...")

    # 6초 대기
    print("\n⏳ 6초 대기 중...")
    import time
    time.sleep(6)

    # 만료 후 재요청
    print("\n3️⃣ 만료 후 재요청 (캐시 미스 예상)")
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )
    print(f"🤖 {result['messages'][-1].content[:100]}...")

    middleware.print_report()


def test_lru_eviction():
    """LRU 제거 테스트"""
    print("\n" + "=" * 70)
    print("🔄 LRU 제거 테스트")
    print("=" * 70)

    # 작은 캐시 크기 설정
    chatbot, middleware = create_caching_chatbot(cache_size=3)
    config = {"configurable": {"thread_id": "test_lru"}}

    questions = [
        "질문 1입니다",
        "질문 2입니다",
        "질문 3입니다",
        "질문 4입니다",  # 질문 1 제거됨
        "질문 1입니다",  # 캐시 미스 (제거됨)
        "질문 2입니다",  # 캐시 히트
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"👤 요청 {i}: {question}")
        print("=" * 70)

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )

        print(f"🤖 {result['messages'][-1].content[:100]}...")

    middleware.print_report()


def test_performance_comparison():
    """성능 비교 테스트"""
    print("\n" + "=" * 70)
    print("⚡ 성능 비교 테스트")
    print("=" * 70)

    # 캐싱 O
    print("\n1️⃣ 캐싱 활성화")
    chatbot_cached, middleware = create_caching_chatbot()
    config = {"configurable": {"thread_id": "perf_cached"}}

    question = "인공지능이란 무엇인가요?"

    import time

    # 첫 번째 (캐시 미스)
    start = time.time()
    chatbot_cached.invoke({"messages": [HumanMessage(content=question)]}, config)
    first_time = time.time() - start

    # 두 번째 (캐시 히트)
    start = time.time()
    chatbot_cached.invoke({"messages": [HumanMessage(content=question)]}, config)
    cached_time = time.time() - start

    print(f"\n📊 결과:")
    print(f"  첫 번째 요청: {first_time:.3f}초")
    print(f"  캐시된 요청: {cached_time:.3f}초")
    print(f"  속도 향상: {first_time / cached_time:.1f}x")

    stats = middleware.get_stats()
    print(f"  히트율: {stats['hit_rate']:.1%}")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 캐싱 챗봇 - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /stats - 캐시 통계")
    print("  /report - 상세 리포트")
    print("  /clear - 캐시 초기화")
    print("  /quit - 종료")
    print("=" * 70)

    chatbot, middleware = create_caching_chatbot(cache_size=50, cache_ttl=300)
    config = {"configurable": {"thread_id": "interactive"}}

    while True:
        try:
            user_input = input("\n👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                middleware.print_report()
                print("\n👋 종료합니다.")
                break

            elif user_input == "/stats":
                stats = middleware.get_stats()
                print(f"\n📊 캐시 통계:")
                print(f"  총 요청: {stats['total_requests']}")
                print(f"  캐시 히트: {stats['cache_hits']}")
                print(f"  히트율: {stats['hit_rate']:.1%}")
                print(f"  캐시 크기: {stats['cache_size']}")
                continue

            elif user_input == "/report":
                middleware.print_report()
                continue

            elif user_input == "/clear":
                middleware.clear_cache()
                print("✅ 캐시가 초기화되었습니다.")
                continue

            # 일반 메시지
            import time
            start = time.time()

            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            elapsed = time.time() - start

            ai_response = result["messages"][-1].content
            print(f"\n🤖 {ai_response}")
            print(f"⏱️  {elapsed:.2f}초")

        except KeyboardInterrupt:
            middleware.print_report()
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
    print("📦 Part 5: 캐싱 Middleware - 실습 과제 2 해답")
    print("=" * 70)

    try:
        # 테스트 1: 기본 캐싱
        test_basic_caching()

        # 테스트 2: 캐시 만료
        print("\n캐시 만료 테스트를 실행하시겠습니까? (y/n): ", end="")
        if input().strip().lower() in ['y', 'yes', '예']:
            test_cache_expiration()

        # 테스트 3: LRU 제거
        test_lru_eviction()

        # 테스트 4: 성능 비교
        test_performance_comparison()

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
    print("  1. LRU 캐시 알고리즘 구현")
    print("  2. 해시 기반 캐시 키 생성")
    print("  3. TTL 기반 만료 처리")
    print("  4. 캐시 히트율 측정")
    print("\n💡 추가 학습:")
    print("  1. Redis 통합 (분산 캐싱)")
    print("  2. Semantic 캐싱 (유사 질문 매칭)")
    print("  3. 캐시 워밍 (사전 로딩)")
    print("  4. 캐시 무효화 전략")
    print("=" * 70)


if __name__ == "__main__":
    main()
