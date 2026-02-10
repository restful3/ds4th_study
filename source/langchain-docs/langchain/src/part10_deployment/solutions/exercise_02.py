"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: Deployment - 실습 과제 2 해답
================================================================================

과제: 평가 시스템
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. Agent 성능 자동 평가
2. 벤치마크 데이터셋
3. 메트릭 수집 및 분석

학습 목표:
- Agent 평가 방법론
- 메트릭 정의 및 측정
- 지속적 개선

================================================================================
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import json

# ============================================================================
# 평가 데이터셋
# ============================================================================

@dataclass
class EvaluationCase:
    """평가 케이스"""
    id: str
    question: str
    expected_answer: str
    category: str
    difficulty: str  # easy, medium, hard

# 평가 데이터셋
EVALUATION_DATASET = [
    EvaluationCase(
        id="calc_01",
        question="2 + 2는 얼마인가요?",
        expected_answer="4",
        category="계산",
        difficulty="easy"
    ),
    EvaluationCase(
        id="calc_02",
        question="(10 + 5) * 2를 계산해주세요",
        expected_answer="30",
        category="계산",
        difficulty="medium"
    ),
    EvaluationCase(
        id="info_01",
        question="Python의 창시자는 누구인가요?",
        expected_answer="Guido van Rossum",
        category="지식",
        difficulty="easy"
    ),
    EvaluationCase(
        id="reason_01",
        question="리스트와 튜플의 주요 차이점은 무엇인가요?",
        expected_answer="변경 가능성 (mutable vs immutable)",
        category="추론",
        difficulty="medium"
    ),
]

# ============================================================================
# 평가 메트릭
# ============================================================================

@dataclass
class EvaluationMetrics:
    """평가 메트릭"""
    accuracy: float = 0.0
    avg_response_time: float = 0.0
    success_rate: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    difficulty_scores: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0

@dataclass
class EvaluationResult:
    """개별 평가 결과"""
    case_id: str
    question: str
    expected: str
    actual: str
    passed: bool
    response_time: float
    error: str = ""

# ============================================================================
# 평가자 (Evaluator)
# ============================================================================

class AgentEvaluator:
    """Agent 평가 시스템"""
    
    def __init__(self, agent, dataset: List[EvaluationCase]):
        self.agent = agent
        self.dataset = dataset
        self.results: List[EvaluationResult] = []
    
    def evaluate_single(self, case: EvaluationCase) -> EvaluationResult:
        """단일 케이스 평가"""
        import time
        from langchain_core.messages import HumanMessage
        
        print(f"  평가 중: {case.id}...")
        
        try:
            start = time.time()
            result = self.agent.invoke({
                "messages": [HumanMessage(content=case.question)]
            })
            elapsed = time.time() - start
            
            actual_answer = result["messages"][-1].content
            
            # 정답 확인 (간단한 키워드 매칭)
            passed = self._check_answer(case.expected_answer, actual_answer)
            
            return EvaluationResult(
                case_id=case.id,
                question=case.question,
                expected=case.expected_answer,
                actual=actual_answer,
                passed=passed,
                response_time=elapsed
            )
            
        except Exception as e:
            return EvaluationResult(
                case_id=case.id,
                question=case.question,
                expected=case.expected_answer,
                actual="",
                passed=False,
                response_time=0.0,
                error=str(e)
            )
    
    def _check_answer(self, expected: str, actual: str) -> bool:
        """답변 확인 (간단한 키워드 매칭)"""
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        # 숫자는 정확히 매칭
        if expected.strip().isdigit():
            return expected.strip() in actual_lower
        
        # 문자열은 키워드 포함 여부
        keywords = expected_lower.split()
        return any(kw in actual_lower for kw in keywords)
    
    def evaluate_all(self) -> EvaluationMetrics:
        """전체 평가 실행"""
        print("\n" + "=" * 70)
        print("📊 평가 시작")
        print("=" * 70)
        
        self.results = []
        
        for case in self.dataset:
            result = self.evaluate_single(case)
            self.results.append(result)
        
        # 메트릭 계산
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _calculate_metrics(self) -> EvaluationMetrics:
        """메트릭 계산"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        
        metrics = EvaluationMetrics(
            accuracy=passed / total if total > 0 else 0,
            avg_response_time=statistics.mean([r.response_time for r in self.results]),
            success_rate=passed / total if total > 0 else 0
        )
        
        # 카테고리별 점수
        categories = {}
        for case in self.dataset:
            if case.category not in categories:
                categories[case.category] = {"total": 0, "passed": 0}
            
            categories[case.category]["total"] += 1
            
            result = next((r for r in self.results if r.case_id == case.id), None)
            if result and result.passed:
                categories[case.category]["passed"] += 1
        
        metrics.category_scores = {
            cat: data["passed"] / data["total"]
            for cat, data in categories.items()
        }
        
        # 난이도별 점수
        difficulties = {}
        for case in self.dataset:
            if case.difficulty not in difficulties:
                difficulties[case.difficulty] = {"total": 0, "passed": 0}
            
            difficulties[case.difficulty]["total"] += 1
            
            result = next((r for r in self.results if r.case_id == case.id), None)
            if result and result.passed:
                difficulties[case.difficulty]["passed"] += 1
        
        metrics.difficulty_scores = {
            diff: data["passed"] / data["total"]
            for diff, data in difficulties.items()
        }
        
        return metrics
    
    def print_report(self, metrics: EvaluationMetrics):
        """평가 리포트 출력"""
        print("\n" + "=" * 70)
        print("📈 평가 리포트")
        print("=" * 70)
        
        print(f"\n전체 성능:")
        print(f"  정확도: {metrics.accuracy:.1%}")
        print(f"  성공률: {metrics.success_rate:.1%}")
        print(f"  평균 응답 시간: {metrics.avg_response_time:.2f}초")
        
        print(f"\n카테고리별 점수:")
        for category, score in metrics.category_scores.items():
            print(f"  {category}: {score:.1%}")
        
        print(f"\n난이도별 점수:")
        for difficulty, score in metrics.difficulty_scores.items():
            print(f"  {difficulty}: {score:.1%}")
        
        print(f"\n개별 결과:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.case_id}: {result.response_time:.2f}s")
            if not result.passed and result.error:
                print(f"     오류: {result.error}")
        
        print("\n" + "=" * 70)
    
    def export_results(self, filename: str = "evaluation_results.json"):
        """결과 내보내기"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_cases": len(self.results),
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "response_time": r.response_time,
                    "question": r.question,
                    "expected": r.expected,
                    "actual": r.actual[:200],  # 처음 200자만
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 결과 저장: {filename}")

# ============================================================================
# 벤치마크 실행
# ============================================================================

def run_benchmark():
    """벤치마크 실행"""
    print("=" * 70)
    print("🏁 Agent 벤치마크")
    print("=" * 70)
    
    # 테스트용 Agent (실제로는 제대로 된 Agent 사용)
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    
    @tool
    def calculator(expr: str) -> str:
        """계산"""
        try:
            return str(eval(expr))
        except:
            return "오류"
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_react_agent(model, [calculator])
    
    # 평가 실행
    evaluator = AgentEvaluator(agent, EVALUATION_DATASET)
    metrics = evaluator.evaluate_all()
    
    # 리포트 출력
    evaluator.print_report(metrics)
    
    # 결과 저장
    evaluator.export_results()
    
    # 성능 기준 확인
    print("\n" + "=" * 70)
    print("🎯 성능 기준")
    print("=" * 70)
    
    thresholds = {
        "정확도": (metrics.accuracy, 0.8),
        "응답 시간": (metrics.avg_response_time, 3.0),
    }
    
    all_passed = True
    for metric_name, (value, threshold) in thresholds.items():
        if metric_name == "응답 시간":
            passed = value <= threshold
            symbol = "✅" if passed else "❌"
            print(f"  {symbol} {metric_name}: {value:.2f} (기준: ≤{threshold})")
        else:
            passed = value >= threshold
            symbol = "✅" if passed else "❌"
            print(f"  {symbol} {metric_name}: {value:.1%} (기준: ≥{threshold:.1%})")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 모든 성능 기준 통과!")
    else:
        print("❌ 일부 성능 기준 미달")
    print("=" * 70)

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📊 Part 10: 평가 시스템 - 실습 과제 2 해답")
    print("=" * 70)
    
    try:
        run_benchmark()
        
        print("\n💡 학습 포인트:")
        print("  1. 평가 데이터셋 구축")
        print("  2. 다양한 메트릭 정의")
        print("  3. 자동화된 평가 파이프라인")
        print("  4. 성능 기준 설정")
        
        print("\n💡 추가 개선:")
        print("  1. LLM 기반 평가 (정답 판정)")
        print("  2. 더 큰 데이터셋")
        print("  3. A/B 테스팅")
        print("  4. 시간 경과에 따른 추적")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
