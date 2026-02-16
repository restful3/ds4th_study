"""
Part 2: Workflows 테스트

워크플로우 패턴 (Routing, Parallelization, Orchestrator) 관련 테스트
"""

import pytest
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, START, END


class TestConditionalRouting:
    """조건부 라우팅 테스트"""

    def test_simple_routing(self):
        """간단한 조건부 라우팅"""
        class RouteState(TypedDict):
            value: int
            path: str

        def router(state: RouteState) -> Literal["high", "low"]:
            return "high" if state["value"] > 50 else "low"

        def high_node(state: RouteState) -> RouteState:
            return {"path": "high_path"}

        def low_node(state: RouteState) -> RouteState:
            return {"path": "low_path"}

        graph = StateGraph(RouteState)
        graph.add_node("high", high_node)
        graph.add_node("low", low_node)

        graph.add_conditional_edges(
            START,
            router,
            {"high": "high", "low": "low"}
        )
        graph.add_edge("high", END)
        graph.add_edge("low", END)

        app = graph.compile()

        # 높은 값 테스트
        result_high = app.invoke({"value": 75, "path": ""})
        assert result_high["path"] == "high_path"

        # 낮은 값 테스트
        result_low = app.invoke({"value": 25, "path": ""})
        assert result_low["path"] == "low_path"

    def test_multi_path_routing(self):
        """다중 경로 라우팅"""
        class CategoryState(TypedDict):
            category: str
            result: str

        def route_category(state: CategoryState) -> str:
            return state["category"]

        def handle_a(state: CategoryState) -> CategoryState:
            return {"result": "A 처리"}

        def handle_b(state: CategoryState) -> CategoryState:
            return {"result": "B 처리"}

        def handle_default(state: CategoryState) -> CategoryState:
            return {"result": "기본 처리"}

        graph = StateGraph(CategoryState)
        graph.add_node("a", handle_a)
        graph.add_node("b", handle_b)
        graph.add_node("default", handle_default)

        graph.add_conditional_edges(
            START,
            route_category,
            {"a": "a", "b": "b", "c": "default"}
        )
        graph.add_edge("a", END)
        graph.add_edge("b", END)
        graph.add_edge("default", END)

        app = graph.compile()

        assert app.invoke({"category": "a", "result": ""})["result"] == "A 처리"
        assert app.invoke({"category": "b", "result": ""})["result"] == "B 처리"
        assert app.invoke({"category": "c", "result": ""})["result"] == "기본 처리"


class TestParallelExecution:
    """병렬 실행 테스트"""

    def test_fan_out_fan_in(self):
        """Fan-out/Fan-in 패턴"""
        class ParallelState(TypedDict):
            input: str
            results: Annotated[list, operator.add]

        def process_a(state: ParallelState) -> ParallelState:
            return {"results": [f"A: {state['input']}"]}

        def process_b(state: ParallelState) -> ParallelState:
            return {"results": [f"B: {state['input']}"]}

        def combine(state: ParallelState) -> ParallelState:
            return {"results": [f"Combined: {len(state['results'])} items"]}

        graph = StateGraph(ParallelState)
        graph.add_node("a", process_a)
        graph.add_node("b", process_b)
        graph.add_node("combine", combine)

        # Fan-out: START에서 a, b로 동시에
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")

        # Fan-in: a, b에서 combine으로
        graph.add_edge("a", "combine")
        graph.add_edge("b", "combine")
        graph.add_edge("combine", END)

        app = graph.compile()
        result = app.invoke({"input": "test", "results": []})

        # a와 b의 결과 + combine 결과
        assert len(result["results"]) == 3
        assert any("A:" in r for r in result["results"])
        assert any("B:" in r for r in result["results"])


class TestSequentialChaining:
    """순차적 체이닝 테스트"""

    def test_prompt_chaining(self):
        """프롬프트 체이닝 패턴"""
        class ChainState(TypedDict):
            text: str
            steps: Annotated[list, operator.add]

        def step1(state: ChainState) -> ChainState:
            return {
                "text": state["text"].upper(),
                "steps": ["uppercase"]
            }

        def step2(state: ChainState) -> ChainState:
            return {
                "text": state["text"].replace(" ", "_"),
                "steps": ["replace_spaces"]
            }

        def step3(state: ChainState) -> ChainState:
            return {
                "text": f"[{state['text']}]",
                "steps": ["wrap_brackets"]
            }

        graph = StateGraph(ChainState)
        graph.add_node("s1", step1)
        graph.add_node("s2", step2)
        graph.add_node("s3", step3)

        graph.add_edge(START, "s1")
        graph.add_edge("s1", "s2")
        graph.add_edge("s2", "s3")
        graph.add_edge("s3", END)

        app = graph.compile()
        result = app.invoke({"text": "hello world", "steps": []})

        assert result["text"] == "[HELLO_WORLD]"
        assert result["steps"] == ["uppercase", "replace_spaces", "wrap_brackets"]


class TestOrchestratorWorker:
    """Orchestrator-Worker 패턴 테스트"""

    def test_simple_orchestrator(self):
        """간단한 오케스트레이터"""
        class OrchestratorState(TypedDict):
            task: str
            subtasks: list
            results: Annotated[list, operator.add]
            completed: bool

        def orchestrator(state: OrchestratorState) -> OrchestratorState:
            if not state.get("subtasks"):
                # 작업 분해
                return {"subtasks": ["task_1", "task_2", "task_3"]}
            elif len(state.get("results", [])) >= 3:
                # 모든 작업 완료
                return {"completed": True}
            return {}

        def worker(state: OrchestratorState) -> OrchestratorState:
            subtasks = state.get("subtasks", [])
            results = state.get("results", [])

            if subtasks and len(results) < len(subtasks):
                current_task = subtasks[len(results)]
                return {"results": [f"completed_{current_task}"]}
            return {}

        def should_continue(state: OrchestratorState) -> str:
            if state.get("completed"):
                return "end"
            return "worker"

        graph = StateGraph(OrchestratorState)
        graph.add_node("orchestrator", orchestrator)
        graph.add_node("worker", worker)

        graph.add_edge(START, "orchestrator")
        graph.add_conditional_edges(
            "orchestrator",
            should_continue,
            {"worker": "worker", "end": END}
        )
        graph.add_edge("worker", "orchestrator")

        app = graph.compile()
        result = app.invoke({
            "task": "main_task",
            "subtasks": [],
            "results": [],
            "completed": False
        })

        assert result["completed"] == True
        assert len(result["results"]) == 3
