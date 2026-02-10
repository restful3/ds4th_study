"""
================================================================================
LangChain AI Agent 마스터 교안
Part 5: Middleware - 실습 과제 1 해답
================================================================================

과제: 비용 추적 Middleware
난이도: ⭐⭐☆☆☆ (초급)

요구사항:
1. 토큰 사용량 추적 (입력/출력)
2. 모델별 비용 계산
3. 누적 통계 및 리포트

학습 목표:
- Middleware 구조 이해
- 토큰 사용량 측정
- 비용 계산 로직 구현

================================================================================
"""

from typing import Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import tiktoken

# ============================================================================
# 비용 계산기
# ============================================================================

class CostCalculator:
    """모델별 비용 계산"""

    # 모델별 비용 (1K 토큰당 USD)
    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """비용 계산"""
        if model not in cls.PRICING:
            # 기본값 사용
            model = "gpt-4o-mini"

        pricing = cls.PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    @classmethod
    def get_pricing_info(cls, model: str) -> dict:
        """모델 가격 정보"""
        return cls.PRICING.get(model, cls.PRICING["gpt-4o-mini"])


# ============================================================================
# 토큰 카운터
# ============================================================================

class TokenCounter:
    """토큰 수 계산"""

    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # 기본 인코딩 사용
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))

    def count_messages_tokens(self, messages: Sequence[BaseMessage]) -> int:
        """메시지 리스트의 토큰 수 계산"""
        total = 0
        for message in messages:
            # 메시지 구조 오버헤드 (약 4 토큰)
            total += 4
            total += self.count_tokens(str(message.content))
        # 응답 프라이밍
        total += 2
        return total


# ============================================================================
# 비용 추적 Middleware
# ============================================================================

class CostTrackingMiddleware:
    """비용 추적 미들웨어"""

    def __init__(self):
        self.stats = {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "requests": []
        }
        self.current_request = None

    def before_call(self, state: MessagesState, config: RunnableConfig):
        """LLM 호출 전"""
        model = config.get("metadata", {}).get("model", "gpt-4o-mini")

        # 토큰 카운터 초기화
        counter = TokenCounter(model)

        # 입력 토큰 계산
        input_tokens = counter.count_messages_tokens(state["messages"])

        # 현재 요청 정보 저장
        self.current_request = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "cost": 0.0,
            "user_message": state["messages"][-1].content if state["messages"] else "",
        }

        print(f"📊 [Cost Tracker] 입력 토큰: {input_tokens}")

    def after_call(self, result: dict, config: RunnableConfig):
        """LLM 호출 후"""
        if not self.current_request:
            return

        model = self.current_request["model"]
        counter = TokenCounter(model)

        # 출력 메시지의 토큰 수 계산
        output_messages = [
            msg for msg in result["messages"]
            if isinstance(msg, AIMessage)
        ]

        if output_messages:
            last_output = output_messages[-1]
            output_tokens = counter.count_tokens(last_output.content)
            self.current_request["output_tokens"] = output_tokens
            self.current_request["ai_response"] = last_output.content[:100] + "..."

        # 비용 계산
        cost = CostCalculator.calculate_cost(
            model,
            self.current_request["input_tokens"],
            self.current_request["output_tokens"]
        )
        self.current_request["cost"] = cost

        # 통계 업데이트
        self.stats["total_requests"] += 1
        self.stats["total_input_tokens"] += self.current_request["input_tokens"]
        self.stats["total_output_tokens"] += self.current_request["output_tokens"]
        self.stats["total_cost"] += cost
        self.stats["requests"].append(self.current_request.copy())

        print(f"📊 [Cost Tracker] 출력 토큰: {output_tokens}")
        print(f"💰 [Cost Tracker] 비용: ${cost:.6f}")
        print(f"📈 [Cost Tracker] 누적 비용: ${self.stats['total_cost']:.6f}")

        self.current_request = None

    def get_stats(self) -> dict:
        """통계 조회"""
        return self.stats.copy()

    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "requests": []
        }

    def print_report(self):
        """상세 리포트 출력"""
        print("\n" + "=" * 70)
        print("💰 비용 추적 리포트")
        print("=" * 70)

        print(f"\n📊 전체 통계:")
        print(f"  총 요청 수: {self.stats['total_requests']}")
        print(f"  총 입력 토큰: {self.stats['total_input_tokens']:,}")
        print(f"  총 출력 토큰: {self.stats['total_output_tokens']:,}")
        print(f"  총 토큰: {self.stats['total_input_tokens'] + self.stats['total_output_tokens']:,}")
        print(f"  총 비용: ${self.stats['total_cost']:.6f}")

        if self.stats['total_requests'] > 0:
            avg_input = self.stats['total_input_tokens'] / self.stats['total_requests']
            avg_output = self.stats['total_output_tokens'] / self.stats['total_requests']
            avg_cost = self.stats['total_cost'] / self.stats['total_requests']

            print(f"\n📈 평균:")
            print(f"  요청당 입력 토큰: {avg_input:.1f}")
            print(f"  요청당 출력 토큰: {avg_output:.1f}")
            print(f"  요청당 비용: ${avg_cost:.6f}")

        # 최근 5개 요청
        if self.stats['requests']:
            print(f"\n📜 최근 요청 (최대 5개):")
            for i, req in enumerate(self.stats['requests'][-5:], 1):
                print(f"\n  {i}. {req['timestamp']}")
                print(f"     모델: {req['model']}")
                print(f"     토큰: {req['input_tokens']} → {req['output_tokens']}")
                print(f"     비용: ${req['cost']:.6f}")
                print(f"     질문: {req['user_message'][:50]}...")

        print("\n" + "=" * 70)


# ============================================================================
# 비용 추적 챗봇 구축
# ============================================================================

def create_cost_tracking_chatbot(model_name: str = "gpt-4o-mini"):
    """비용 추적 챗봇 생성"""

    model = ChatOpenAI(model=model_name, temperature=0.7)
    middleware = CostTrackingMiddleware()

    # 챗봇 노드 (미들웨어 통합)
    def chatbot_node(state: MessagesState, config: RunnableConfig) -> dict:
        """대화 응답 with 비용 추적"""
        # Before call
        middleware.before_call(state, config)

        # LLM 호출
        response = model.invoke(state["messages"])
        result = {"messages": [response]}

        # After call
        middleware.after_call(result, config)

        return result

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

def test_cost_tracking():
    """비용 추적 테스트"""
    print("=" * 70)
    print("💰 비용 추적 Middleware 테스트")
    print("=" * 70)

    # 여러 모델로 테스트
    models = ["gpt-4o-mini", "gpt-4o"]

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"🔍 모델: {model_name}")
        print("=" * 70)

        chatbot, middleware = create_cost_tracking_chatbot(model_name)
        config = {
            "configurable": {"thread_id": f"test_{model_name}"},
            "metadata": {"model": model_name}
        }

        # 가격 정보
        pricing = CostCalculator.get_pricing_info(model_name)
        print(f"\n💵 가격 정보 (1K 토큰):")
        print(f"  입력: ${pricing['input']}")
        print(f"  출력: ${pricing['output']}")

        # 테스트 대화
        test_messages = [
            "안녕하세요!",
            "Python에서 리스트와 튜플의 차이점을 설명해주세요.",
            "딕셔너리는 어떻게 사용하나요?",
        ]

        for msg in test_messages:
            print(f"\n{'=' * 70}")
            print(f"👤 사용자: {msg}")
            print("=" * 70)

            result = chatbot.invoke(
                {"messages": [HumanMessage(content=msg)]},
                config
            )

            ai_response = result["messages"][-1].content
            print(f"\n🤖 AI: {ai_response[:200]}...")

        # 리포트 출력
        middleware.print_report()


def test_cost_comparison():
    """모델별 비용 비교"""
    print("\n" + "=" * 70)
    print("📊 모델별 비용 비교")
    print("=" * 70)

    test_message = "인공지능의 역사와 발전 과정에 대해 자세히 설명해주세요."
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]

    results = []

    for model_name in models:
        chatbot, middleware = create_cost_tracking_chatbot(model_name)
        config = {
            "configurable": {"thread_id": f"compare_{model_name}"},
            "metadata": {"model": model_name}
        }

        print(f"\n🔍 {model_name} 테스트 중...")

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=test_message)]},
            config
        )

        stats = middleware.get_stats()
        results.append({
            "model": model_name,
            "stats": stats
        })

    # 비교 표 출력
    print("\n" + "=" * 70)
    print("📊 비교 결과")
    print("=" * 70)
    print(f"{'모델':<20} {'입력 토큰':>12} {'출력 토큰':>12} {'비용':>12}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['model']:<20} "
            f"{r['stats']['total_input_tokens']:>12} "
            f"{r['stats']['total_output_tokens']:>12} "
            f"${r['stats']['total_cost']:>11.6f}"
        )

    print("=" * 70)

    # 가장 저렴한 모델
    cheapest = min(results, key=lambda x: x['stats']['total_cost'])
    print(f"\n💡 가장 저렴한 모델: {cheapest['model']}")
    print(f"   비용: ${cheapest['stats']['total_cost']:.6f}")


def interactive_mode():
    """대화형 모드"""
    print("\n" + "=" * 70)
    print("🎮 비용 추적 챗봇 - 대화형 모드")
    print("=" * 70)
    print("명령어:")
    print("  /stats - 현재 통계")
    print("  /report - 상세 리포트")
    print("  /reset - 통계 초기화")
    print("  /model <name> - 모델 변경")
    print("  /quit - 종료")
    print("=" * 70)

    current_model = "gpt-4o-mini"
    chatbot, middleware = create_cost_tracking_chatbot(current_model)
    config = {
        "configurable": {"thread_id": "interactive"},
        "metadata": {"model": current_model}
    }

    print(f"\n현재 모델: {current_model}")
    pricing = CostCalculator.get_pricing_info(current_model)
    print(f"가격 (1K 토큰): 입력 ${pricing['input']}, 출력 ${pricing['output']}")

    while True:
        try:
            user_input = input(f"\n[{current_model}] 👤 : ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                middleware.print_report()
                print("\n👋 종료합니다.")
                break

            elif user_input == "/stats":
                stats = middleware.get_stats()
                print(f"\n📊 현재 통계:")
                print(f"  요청 수: {stats['total_requests']}")
                print(f"  총 토큰: {stats['total_input_tokens'] + stats['total_output_tokens']:,}")
                print(f"  누적 비용: ${stats['total_cost']:.6f}")
                continue

            elif user_input == "/report":
                middleware.print_report()
                continue

            elif user_input == "/reset":
                middleware.reset_stats()
                print("✅ 통계가 초기화되었습니다.")
                continue

            elif user_input.startswith("/model"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("❌ 사용법: /model <model_name>")
                    continue

                new_model = parts[1]
                current_model = new_model
                chatbot, middleware = create_cost_tracking_chatbot(current_model)
                config["metadata"]["model"] = current_model

                pricing = CostCalculator.get_pricing_info(current_model)
                print(f"✅ 모델 변경: {current_model}")
                print(f"   가격: 입력 ${pricing['input']}, 출력 ${pricing['output']}")
                continue

            # 일반 메시지
            result = chatbot.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            ai_response = result["messages"][-1].content
            print(f"\n🤖 {ai_response}")

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
    print("💰 Part 5: 비용 추적 Middleware - 실습 과제 1 해답")
    print("=" * 70)

    try:
        # 테스트 1: 기본 비용 추적
        test_cost_tracking()

        # 테스트 2: 모델별 비교
        test_cost_comparison()

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
    print("  1. before_call/after_call 패턴")
    print("  2. tiktoken으로 토큰 수 계산")
    print("  3. 모델별 가격 정보 관리")
    print("  4. 누적 통계 및 리포트")
    print("\n💡 추가 학습:")
    print("  1. 예산 한도 설정 및 알림")
    print("  2. 비용 데이터베이스 저장")
    print("  3. 사용자별 비용 추적")
    print("  4. 대시보드 시각화")
    print("=" * 70)


if __name__ == "__main__":
    main()
