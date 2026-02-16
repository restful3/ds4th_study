"""
Code Assistant - 코드 분석 및 생성 에이전트

이 예제는 LangGraph를 활용하여 코드 분석, 생성, 수정을 수행하는
Code Assistant를 구현합니다. Human-in-the-Loop 패턴을 통해
코드 변경 전 사용자 승인을 받습니다.

기능:
- 코드 분석 및 리뷰
- 코드 생성
- 코드 수정 제안
- Human-in-the-Loop 승인

실행 방법:
    python -m examples.05_code_assistant.main
"""

import os
import ast
from typing import TypedDict, Annotated, List, Optional, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import operator


# =============================================================================
# 환경 설정
# =============================================================================

load_dotenv()


# =============================================================================
# State 정의
# =============================================================================

class CodeAssistantState(TypedDict):
    """Code Assistant State"""
    user_request: str
    code_input: str
    task_type: str
    analysis: str
    plan: List[str]
    plan_approved: bool
    generated_code: str
    validation_result: str
    history: Annotated[List[str], operator.add]


# =============================================================================
# 유틸리티 함수
# =============================================================================

def validate_python_syntax(code: str) -> tuple:
    """Python 코드 문법 검증"""
    try:
        ast.parse(code)
        return True, "✅ 문법 오류 없음"
    except SyntaxError as e:
        return False, f"❌ 문법 오류: {e.msg} (라인 {e.lineno})"


def analyze_code_complexity(code: str) -> dict:
    """코드 복잡도 분석"""
    lines = code.strip().split('\n')
    return {
        "total_lines": len(lines),
        "blank_lines": sum(1 for line in lines if not line.strip()),
        "comment_lines": sum(1 for line in lines if line.strip().startswith('#')),
        "function_count": code.count('def '),
        "class_count": code.count('class '),
        "import_count": code.count('import ')
    }


# =============================================================================
# Code Assistant 구현
# =============================================================================

def create_code_assistant():
    """Code Assistant 그래프 생성"""

    def analyze_request(state: CodeAssistantState) -> CodeAssistantState:
        """사용자 요청 분석"""
        request = state["user_request"].lower()

        if any(kw in request for kw in ["분석", "analyze", "복잡도"]):
            task_type = "analyze"
        elif any(kw in request for kw in ["생성", "generate", "만들어", "작성"]):
            task_type = "generate"
        elif any(kw in request for kw in ["수정", "modify", "변경", "고쳐"]):
            task_type = "modify"
        elif any(kw in request for kw in ["리뷰", "review", "검토"]):
            task_type = "review"
        elif any(kw in request for kw in ["설명", "explain", "뭐하는"]):
            task_type = "explain"
        else:
            task_type = "analyze"

        analysis = f"요청 분석 완료: 작업 유형 = {task_type}"

        return {
            "task_type": task_type,
            "analysis": analysis,
            "history": [f"[분석] {analysis}"]
        }

    def plan_changes(state: CodeAssistantState) -> CodeAssistantState:
        """코드 변경 계획 수립"""
        task_type = state["task_type"]

        plans = {
            "analyze": ["1. 코드 문법 검증", "2. 복잡도 분석", "3. 결과 보고서 생성"],
            "generate": ["1. 요구사항 파악", "2. 코드 구조 설계", "3. 코드 생성", "4. 문법 검증"],
            "modify": ["1. 기존 코드 분석", "2. 수정 사항 파악", "3. 코드 수정", "4. 변경 검증"],
            "review": ["1. 코드 품질 분석", "2. 잠재적 문제 식별", "3. 개선 제안 작성"],
            "explain": ["1. 코드 구조 파악", "2. 주요 로직 분석", "3. 설명 문서 생성"]
        }

        plan = plans.get(task_type, plans["analyze"])

        return {
            "plan": plan,
            "history": [f"[계획] {len(plan)}단계 계획 수립"]
        }

    def request_approval(state: CodeAssistantState) -> CodeAssistantState:
        """Human-in-the-Loop: 계획 승인 요청"""
        plan = state["plan"]
        task_type = state["task_type"]
        plan_text = '\n'.join(plan)

        approval = interrupt({
            "message": f"다음 계획을 실행하시겠습니까?\n\n작업: {task_type}\n\n계획:\n{plan_text}",
            "options": ["승인", "거부"]
        })

        plan_approved = approval == "승인"

        return {
            "plan_approved": plan_approved,
            "history": [f"[승인] {'승인됨' if plan_approved else '거부됨'}"]
        }

    def execute_task(state: CodeAssistantState) -> CodeAssistantState:
        """작업 실행"""
        task_type = state["task_type"]
        code_input = state.get("code_input", "")

        if task_type == "analyze" and code_input:
            is_valid, syntax_msg = validate_python_syntax(code_input)
            metrics = analyze_code_complexity(code_input)
            generated_code = f"""# 코드 분석 결과
## 문법 검증: {syntax_msg}
## 메트릭: 총 {metrics['total_lines']}줄, 함수 {metrics['function_count']}개, 클래스 {metrics['class_count']}개"""
            validation_result = "분석 완료"

        elif task_type == "generate":
            generated_code = '''def example_function(param1: str, param2: int = 0) -> str:
    """예제 함수입니다."""
    return f"{param1}_{param2}"

if __name__ == "__main__":
    print(example_function("test", 42))'''
            is_valid, syntax_msg = validate_python_syntax(generated_code)
            validation_result = syntax_msg

        elif task_type == "review" and code_input:
            metrics = analyze_code_complexity(code_input)
            is_valid, syntax_msg = validate_python_syntax(code_input)
            issues = []
            if metrics['function_count'] == 0:
                issues.append("- 함수가 정의되지 않음")
            if metrics['comment_lines'] < 2:
                issues.append("- 주석 부족")
            if not is_valid:
                issues.append(f"- {syntax_msg}")
            generated_code = f"# 코드 리뷰 결과\n" + ('\n'.join(issues) if issues else "- 특별한 문제 없음")
            validation_result = "리뷰 완료"

        else:
            generated_code = "# 작업 완료"
            validation_result = "완료"

        return {
            "generated_code": generated_code,
            "validation_result": validation_result,
            "history": [f"[실행] {task_type} 완료"]
        }

    def route_by_approval(state: CodeAssistantState) -> str:
        return "execute" if state.get("plan_approved") else "rejected"

    def handle_rejection(state: CodeAssistantState) -> CodeAssistantState:
        return {
            "generated_code": "# 계획이 거부되었습니다.",
            "validation_result": "거부됨",
            "history": ["[거부] 작업 취소"]
        }

    graph = StateGraph(CodeAssistantState)
    graph.add_node("analyze", analyze_request)
    graph.add_node("plan", plan_changes)
    graph.add_node("approval", request_approval)
    graph.add_node("execute", execute_task)
    graph.add_node("rejected", handle_rejection)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "approval")
    graph.add_conditional_edges("approval", route_by_approval, {"execute": "execute", "rejected": "rejected"})
    graph.add_edge("execute", END)
    graph.add_edge("rejected", END)

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# 데모 실행
# =============================================================================

def run_demo():
    """데모 실행"""
    print("=" * 60)
    print("💻 Code Assistant Demo")
    print("=" * 60)

    assistant = create_code_assistant()

    sample_code = '''def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
'''

    test_cases = [
        {"request": "이 코드를 분석해주세요", "code": sample_code},
        {"request": "간단한 함수를 생성해주세요", "code": ""},
        {"request": "이 코드를 리뷰해주세요", "code": sample_code}
    ]

    for i, test in enumerate(test_cases):
        config = {"configurable": {"thread_id": f"demo_{i}"}}
        print(f"\n{'='*50}")
        print(f"📝 요청: {test['request']}")

        result = assistant.invoke({
            "user_request": test['request'],
            "code_input": test['code'],
            "task_type": "",
            "analysis": "",
            "plan": [],
            "plan_approved": False,
            "generated_code": "",
            "validation_result": "",
            "history": []
        }, config=config)

        if "__interrupt__" in str(result):
            print(f"⏸️ 승인 대기 중... (자동 승인)")
            result = assistant.invoke(Command(resume="승인"), config=config)

        print(f"✅ 작업: {result.get('task_type')}")
        print(f"   검증: {result.get('validation_result')}")
        code = result.get('generated_code', '')[:200]
        if code:
            print(f"   코드: {code}...")


def main():
    """메인 함수"""
    import sys
    run_demo()
    print("\n" + "=" * 60)
    print("✅ 완료!")


if __name__ == "__main__":
    main()
