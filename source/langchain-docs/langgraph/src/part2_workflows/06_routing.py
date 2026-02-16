"""
[Chapter 6] 조건부 라우팅

📝 설명:
    조건부 라우팅은 현재 State에 따라 다음에 실행할 노드를 동적으로
    결정하는 패턴입니다. if-else 분기와 유사하지만 그래프 수준에서 동작합니다.

🎯 학습 목표:
    - add_conditional_edges 사용법 익히기
    - 라우팅 함수 작성 방법
    - Structured Output을 활용한 라우팅
    - Command 객체를 활용한 제어 흐름

📚 관련 문서:
    - docs/Part2-Workflows/06-conditional-routing.md
    - 공식 문서: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#routing

💻 실행 방법:
    python -m src.part2_workflows.06_routing

📦 필요한 패키지:
    - langgraph>=0.2.0
    - langchain-anthropic>=0.3.0
"""

import os
from typing import TypedDict, Literal, Optional
from enum import Enum
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# =============================================================================
# 1. 기본 조건부 라우팅
# =============================================================================

class BasicRoutingState(TypedDict):
    """기본 라우팅을 위한 State"""
    input_type: str  # "text", "number", "other"
    input_value: str
    result: str


def process_text(state: BasicRoutingState) -> BasicRoutingState:
    """텍스트 처리"""
    return {"result": f"[TEXT] 처리됨: {state['input_value'].upper()}"}


def process_number(state: BasicRoutingState) -> BasicRoutingState:
    """숫자 처리"""
    try:
        num = float(state["input_value"])
        return {"result": f"[NUMBER] 처리됨: {num * 2}"}
    except ValueError:
        return {"result": "[NUMBER] 유효하지 않은 숫자"}


def process_other(state: BasicRoutingState) -> BasicRoutingState:
    """기타 처리"""
    return {"result": f"[OTHER] 처리됨: {state['input_value']}"}


def route_by_type(state: BasicRoutingState) -> str:
    """
    입력 타입에 따라 라우팅하는 함수

    Returns:
        다음에 실행할 노드 이름
    """
    input_type = state["input_type"]

    if input_type == "text":
        return "process_text"
    elif input_type == "number":
        return "process_number"
    else:
        return "process_other"


def create_basic_routing_graph():
    """기본 라우팅 그래프 생성"""
    graph = StateGraph(BasicRoutingState)

    # 노드 추가
    graph.add_node("process_text", process_text)
    graph.add_node("process_number", process_number)
    graph.add_node("process_other", process_other)

    # START에서 조건부 라우팅
    graph.add_conditional_edges(
        START,  # 시작점
        route_by_type,  # 라우팅 함수
        {
            # 라우팅 함수 반환값: 실제 노드 이름 매핑
            "process_text": "process_text",
            "process_number": "process_number",
            "process_other": "process_other"
        }
    )

    # 모든 처리 노드에서 END로
    graph.add_edge("process_text", END)
    graph.add_edge("process_number", END)
    graph.add_edge("process_other", END)

    return graph.compile()


def run_basic_routing_example():
    """기본 라우팅 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 1: 기본 조건부 라우팅")
    print("=" * 60)

    app = create_basic_routing_graph()

    test_cases = [
        {"input_type": "text", "input_value": "hello", "result": ""},
        {"input_type": "number", "input_value": "42", "result": ""},
        {"input_type": "other", "input_value": "???", "result": ""},
    ]

    for case in test_cases:
        result = app.invoke(case)
        print(f"\n   타입: {case['input_type']}, 값: {case['input_value']}")
        print(f"   결과: {result['result']}")


# =============================================================================
# 2. Literal 타입을 사용한 안전한 라우팅
# =============================================================================

class SafeRoutingState(TypedDict):
    """Literal 타입을 사용한 State"""
    category: Literal["urgent", "normal", "low"]
    message: str
    priority_result: str


def handle_urgent(state: SafeRoutingState) -> SafeRoutingState:
    """긴급 처리"""
    return {"priority_result": f"🔴 [긴급] {state['message']}"}


def handle_normal(state: SafeRoutingState) -> SafeRoutingState:
    """일반 처리"""
    return {"priority_result": f"🟡 [일반] {state['message']}"}


def handle_low(state: SafeRoutingState) -> SafeRoutingState:
    """낮은 우선순위 처리"""
    return {"priority_result": f"🟢 [낮음] {state['message']}"}


def route_by_priority(state: SafeRoutingState) -> Literal["urgent", "normal", "low"]:
    """
    우선순위에 따른 라우팅 (Literal 타입 반환)

    Returns:
        다음 노드 이름 (타입 안전)
    """
    return state["category"]


def create_safe_routing_graph():
    """안전한 라우팅 그래프 생성"""
    graph = StateGraph(SafeRoutingState)

    graph.add_node("urgent", handle_urgent)
    graph.add_node("normal", handle_normal)
    graph.add_node("low", handle_low)

    # Literal 타입 덕분에 path_map 없이도 사용 가능
    graph.add_conditional_edges(
        START,
        route_by_priority,
        # path_map을 생략하면 라우팅 함수의 반환값이 직접 노드 이름으로 사용됨
    )

    graph.add_edge("urgent", END)
    graph.add_edge("normal", END)
    graph.add_edge("low", END)

    return graph.compile()


def run_safe_routing_example():
    """안전한 라우팅 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 2: Literal 타입 라우팅 (타입 안전)")
    print("=" * 60)

    app = create_safe_routing_graph()

    test_cases = [
        {"category": "urgent", "message": "서버 다운!", "priority_result": ""},
        {"category": "normal", "message": "정기 점검", "priority_result": ""},
        {"category": "low", "message": "UI 개선 요청", "priority_result": ""},
    ]

    for case in test_cases:
        result = app.invoke(case)
        print(f"\n   {result['priority_result']}")


# =============================================================================
# 3. LLM을 사용한 의미 기반 라우팅
# =============================================================================

class LLMRoutingState(TypedDict):
    """LLM 라우팅을 위한 State"""
    user_query: str
    category: str
    response: str


def create_llm_routing_graph():
    """LLM 기반 라우팅 그래프 생성"""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return None

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

    def classify_query(state: LLMRoutingState) -> LLMRoutingState:
        """LLM을 사용하여 쿼리 분류"""
        messages = [
            SystemMessage(content="""사용자의 질문을 다음 카테고리 중 하나로 분류하세요:
- "tech": 기술/프로그래밍 관련
- "general": 일반 질문
- "creative": 창작/아이디어 관련

카테고리 이름만 소문자로 출력하세요."""),
            HumanMessage(content=state["user_query"])
        ]
        response = llm.invoke(messages)
        category = response.content.strip().lower()

        # 유효한 카테고리인지 확인
        if category not in ["tech", "general", "creative"]:
            category = "general"

        return {"category": category}

    def handle_tech(state: LLMRoutingState) -> LLMRoutingState:
        """기술 질문 처리"""
        messages = [
            SystemMessage(content="당신은 기술 전문가입니다. 기술적으로 정확하게 답변하세요."),
            HumanMessage(content=state["user_query"])
        ]
        response = llm.invoke(messages)
        return {"response": f"[Tech] {response.content}"}

    def handle_general(state: LLMRoutingState) -> LLMRoutingState:
        """일반 질문 처리"""
        messages = [
            SystemMessage(content="당신은 친절한 도우미입니다. 쉽게 설명해주세요."),
            HumanMessage(content=state["user_query"])
        ]
        response = llm.invoke(messages)
        return {"response": f"[General] {response.content}"}

    def handle_creative(state: LLMRoutingState) -> LLMRoutingState:
        """창작 질문 처리"""
        messages = [
            SystemMessage(content="당신은 창의적인 작가입니다. 상상력을 발휘해주세요."),
            HumanMessage(content=state["user_query"])
        ]
        response = llm.invoke(messages)
        return {"response": f"[Creative] {response.content}"}

    def route_by_category(state: LLMRoutingState) -> str:
        """분류된 카테고리에 따라 라우팅"""
        return state["category"]

    graph = StateGraph(LLMRoutingState)

    graph.add_node("classify", classify_query)
    graph.add_node("tech", handle_tech)
    graph.add_node("general", handle_general)
    graph.add_node("creative", handle_creative)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "tech": "tech",
            "general": "general",
            "creative": "creative"
        }
    )

    graph.add_edge("tech", END)
    graph.add_edge("general", END)
    graph.add_edge("creative", END)

    return graph.compile()


def run_llm_routing_example():
    """LLM 라우팅 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 3: LLM 기반 의미 라우팅")
    print("=" * 60)

    load_dotenv()
    app = create_llm_routing_graph()

    if app is None:
        print("\n⚠️  LLM을 사용할 수 없습니다.")
        return

    queries = [
        "Python에서 리스트와 튜플의 차이점은?",
        "오늘 날씨가 어때요?",
        "마법 세계를 배경으로 한 이야기를 들려주세요"
    ]

    for query in queries:
        result = app.invoke({
            "user_query": query,
            "category": "",
            "response": ""
        })
        print(f"\n📝 질문: {query}")
        print(f"📂 카테고리: {result['category']}")
        print(f"💬 응답: {result['response'][:100]}...")


# =============================================================================
# 4. Command 객체를 사용한 라우팅
# =============================================================================

class CommandRoutingState(TypedDict):
    """Command 라우팅을 위한 State"""
    score: int
    feedback: str


def evaluate_score(state: CommandRoutingState) -> Command[Literal["pass", "fail"]]:
    """
    점수를 평가하고 Command로 라우팅

    Command는 다음 노드와 함께 State 업데이트도 가능
    """
    score = state["score"]

    if score >= 60:
        return Command(
            goto="pass",
            update={"feedback": "합격입니다!"}
        )
    else:
        return Command(
            goto="fail",
            update={"feedback": "불합격입니다. 더 노력하세요."}
        )


def handle_pass(state: CommandRoutingState) -> CommandRoutingState:
    """합격 처리"""
    return {"feedback": f"🎉 {state['feedback']} 점수: {state['score']}"}


def handle_fail(state: CommandRoutingState) -> CommandRoutingState:
    """불합격 처리"""
    return {"feedback": f"😢 {state['feedback']} 점수: {state['score']}"}


def create_command_routing_graph():
    """Command 기반 라우팅 그래프 생성"""
    graph = StateGraph(CommandRoutingState)

    graph.add_node("evaluate", evaluate_score)
    graph.add_node("pass", handle_pass)
    graph.add_node("fail", handle_fail)

    graph.add_edge(START, "evaluate")
    # Command를 반환하는 노드는 별도의 edge 설정 불필요
    graph.add_edge("pass", END)
    graph.add_edge("fail", END)

    return graph.compile()


def run_command_routing_example():
    """Command 라우팅 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 4: Command 객체 라우팅")
    print("=" * 60)

    app = create_command_routing_graph()

    test_scores = [75, 45, 60]

    for score in test_scores:
        result = app.invoke({"score": score, "feedback": ""})
        print(f"\n   점수: {score}")
        print(f"   결과: {result['feedback']}")


# =============================================================================
# 5. 다중 분기 라우팅
# =============================================================================

class MultiBranchState(TypedDict):
    """다중 분기를 위한 State"""
    value: int
    path_taken: str
    result: str


def analyze_value(state: MultiBranchState) -> str:
    """값을 분석하여 5개 분기 중 하나로 라우팅"""
    value = state["value"]

    if value < 0:
        return "negative"
    elif value == 0:
        return "zero"
    elif value < 10:
        return "small"
    elif value < 100:
        return "medium"
    else:
        return "large"


def handle_negative(state: MultiBranchState) -> MultiBranchState:
    return {"path_taken": "negative", "result": f"음수: {state['value']}"}


def handle_zero(state: MultiBranchState) -> MultiBranchState:
    return {"path_taken": "zero", "result": "0입니다"}


def handle_small(state: MultiBranchState) -> MultiBranchState:
    return {"path_taken": "small", "result": f"작은 수: {state['value']}"}


def handle_medium(state: MultiBranchState) -> MultiBranchState:
    return {"path_taken": "medium", "result": f"중간 수: {state['value']}"}


def handle_large(state: MultiBranchState) -> MultiBranchState:
    return {"path_taken": "large", "result": f"큰 수: {state['value']}"}


def create_multi_branch_graph():
    """다중 분기 그래프 생성"""
    graph = StateGraph(MultiBranchState)

    graph.add_node("negative", handle_negative)
    graph.add_node("zero", handle_zero)
    graph.add_node("small", handle_small)
    graph.add_node("medium", handle_medium)
    graph.add_node("large", handle_large)

    graph.add_conditional_edges(
        START,
        analyze_value,
        {
            "negative": "negative",
            "zero": "zero",
            "small": "small",
            "medium": "medium",
            "large": "large"
        }
    )

    # 모든 노드에서 END로
    for node in ["negative", "zero", "small", "medium", "large"]:
        graph.add_edge(node, END)

    return graph.compile()


def run_multi_branch_example():
    """다중 분기 예제 실행"""
    print("\n" + "=" * 60)
    print("예제 5: 다중 분기 라우팅")
    print("=" * 60)

    app = create_multi_branch_graph()

    test_values = [-5, 0, 7, 42, 1000]

    for value in test_values:
        result = app.invoke({"value": value, "path_taken": "", "result": ""})
        print(f"\n   값: {value:5} → 경로: {result['path_taken']:8} → {result['result']}")


# =============================================================================
# 6. 라우팅 패턴 정리
# =============================================================================

def explain_routing_patterns():
    """라우팅 패턴 설명"""
    print("\n" + "=" * 60)
    print("📘 조건부 라우팅 패턴 정리")
    print("=" * 60)

    print("""
add_conditional_edges() 사용법:

graph.add_conditional_edges(
    source,      # 시작 노드 (또는 START)
    path_func,   # 라우팅 함수
    path_map     # (선택) 반환값 → 노드 매핑
)

라우팅 함수 유형:

1. 문자열 반환
   def route(state) -> str:
       return "node_name"

2. Literal 타입 반환 (타입 안전)
   def route(state) -> Literal["a", "b", "c"]:
       return "a"

3. Command 객체 반환 (State 업데이트 포함)
   def route(state) -> Command:
       return Command(goto="node", update={"key": "value"})

path_map 옵션:

1. 명시적 매핑
   {"return_value": "actual_node_name"}

2. 생략 (반환값 = 노드 이름)
   라우팅 함수가 노드 이름을 직접 반환할 때

팁:
- Literal 타입을 사용하면 IDE 지원과 타입 체크 가능
- Command는 라우팅과 State 업데이트를 한 번에
- 복잡한 분류는 LLM에게 위임 가능
""")


# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("[Chapter 6] 조건부 라우팅")
    print("=" * 60)

    load_dotenv()

    # 예제 실행
    run_basic_routing_example()
    run_safe_routing_example()
    run_llm_routing_example()
    run_command_routing_example()
    run_multi_branch_example()

    # 패턴 정리
    explain_routing_patterns()

    print("\n" + "=" * 60)
    print("✅ 모든 예제 실행 완료!")
    print("   다음 예제: 07_parallelization.py (병렬 실행)")
    print("=" * 60)


if __name__ == "__main__":
    main()
