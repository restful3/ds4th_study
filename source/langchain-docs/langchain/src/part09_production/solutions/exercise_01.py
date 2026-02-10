"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: Production - 실습 과제 1 해답
================================================================================

과제: 진행 상황 표시 Agent (Custom Streaming)
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. Agent 실행 과정을 실시간 스트리밍
2. 각 단계별 진행 상황 표시
3. 사용자 경험 개선

학습 목표:
- Streaming API 사용
- 진행 상황 시각화
- 실시간 피드백

================================================================================
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import time
import sys

# ============================================================================
# 도구 정의
# ============================================================================

@tool
def search_database(query: str) -> str:
    """데이터베이스에서 정보를 검색합니다."""
    time.sleep(1)  # 시뮬레이션
    return f"검색 결과: '{query}'에 대한 데이터 10개 발견"

@tool
def analyze_data(data: str) -> str:
    """데이터를 분석합니다."""
    time.sleep(1.5)  # 시뮬레이션
    return f"분석 완료: {data}에 대한 통계 및 인사이트 생성"

@tool
def generate_report(analysis: str) -> str:
    """보고서를 생성합니다."""
    time.sleep(1)  # 시뮬레이션
    return f"보고서 생성 완료: {analysis} 기반 PDF 생성"

# ============================================================================
# 스트리밍 Agent
# ============================================================================

def create_streaming_agent():
    """스트리밍 Agent 생성"""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    tools = [search_database, analyze_data, generate_report]
    
    system_prompt = """당신은 작업 수행 Agent입니다.
작업을 단계별로 수행하고 진행 상황을 알려주세요."""
    
    memory = MemorySaver()
    agent = create_react_agent(
        model, tools,
        state_modifier=system_prompt,
        checkpointer=memory
    )
    
    return agent

# ============================================================================
# 진행 상황 표시
# ============================================================================

class ProgressTracker:
    """진행 상황 추적기"""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
    
    def add_step(self, step_name: str):
        """단계 추가"""
        self.steps.append({
            "name": step_name,
            "status": "pending",
            "start_time": None,
            "end_time": None
        })
    
    def start_step(self, step_index: int):
        """단계 시작"""
        if step_index < len(self.steps):
            self.steps[step_index]["status"] = "in_progress"
            self.steps[step_index]["start_time"] = time.time()
            self.current_step = step_index
            self._print_progress()
    
    def complete_step(self, step_index: int):
        """단계 완료"""
        if step_index < len(self.steps):
            self.steps[step_index]["status"] = "completed"
            self.steps[step_index]["end_time"] = time.time()
            self._print_progress()
    
    def _print_progress(self):
        """진행 상황 출력"""
        print("\r" + " " * 100, end="")  # 이전 출력 지우기
        print("\r", end="")
        
        symbols = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅"
        }
        
        progress = []
        for i, step in enumerate(self.steps):
            symbol = symbols[step["status"]]
            progress.append(f"{symbol} {step['name']}")
        
        print(" → ".join(progress), end="", flush=True)
    
    def finish(self):
        """완료"""
        print("\n")  # 줄바꿈

def run_with_progress(agent, question: str):
    """진행 상황 표시와 함께 실행"""
    print(f"\n질문: {question}\n")
    
    # 진행 상황 추적기
    tracker = ProgressTracker()
    tracker.add_step("분석")
    tracker.add_step("검색")
    tracker.add_step("처리")
    tracker.add_step("완료")
    
    config = {"configurable": {"thread_id": "progress_demo"}}
    
    # 스트리밍 실행
    step_index = 0
    tracker.start_step(step_index)
    
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config,
        stream_mode="updates"
    ):
        # 각 업데이트마다 진행 상황 업데이트
        if chunk:
            step_index += 1
            if step_index < len(tracker.steps):
                tracker.complete_step(step_index - 1)
                tracker.start_step(step_index)
    
    # 완료
    tracker.complete_step(len(tracker.steps) - 1)
    tracker.finish()
    
    # 최종 결과
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )
    
    return result

# ============================================================================
# 실시간 토큰 스트리밍
# ============================================================================

def stream_tokens_demo():
    """토큰 단위 스트리밍 데모"""
    print("\n" + "=" * 70)
    print("🌊 토큰 스트리밍 데모")
    print("=" * 70)
    
    model = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    
    print("\n질문: Python이란 무엇인가요?\n")
    print("답변: ", end="", flush=True)
    
    for chunk in model.stream([HumanMessage(content="Python이란 무엇인가요? 간단히 설명해주세요.")]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            time.sleep(0.02)  # 효과를 위한 지연
    
    print("\n")

# ============================================================================
# 테스트
# ============================================================================

def test_streaming_agent():
    """스트리밍 Agent 테스트"""
    print("=" * 70)
    print("📊 진행 상황 표시 Agent 테스트")
    print("=" * 70)
    
    agent = create_streaming_agent()
    
    questions = [
        "데이터베이스에서 사용자 정보를 검색하고 분석해줘",
        "최근 판매 데이터를 조회하고 보고서를 생성해줘",
    ]
    
    for question in questions:
        result = run_with_progress(agent, question)
        
        print(f"\n최종 답변:\n{result['messages'][-1].content}\n")

def test_progress_bar():
    """프로그레스 바 스타일 테스트"""
    print("\n" + "=" * 70)
    print("📊 프로그레스 바 스타일")
    print("=" * 70)
    
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
        """프로그레스 바 출력"""
        filled_length = int(length * iteration // total)
        bar = '█' * filled_length + '-' * (length - filled_length)
        percent = f"{100 * (iteration / float(total)):.1f}"
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    
    print("\n작업 진행 중...\n")
    
    steps = ["초기화", "데이터 로딩", "처리", "저장", "완료"]
    total = len(steps)
    
    for i, step in enumerate(steps, 1):
        print_progress_bar(i, total, prefix=f'{step}:', suffix='완료', length=40)
        time.sleep(0.5)
    
    print("\n\n✅ 모든 작업 완료!\n")

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📊 Part 9: 진행 상황 표시 Agent - 실습 과제 1 해답")
    print("=" * 70)
    
    try:
        # 테스트 1: 스트리밍 Agent
        test_streaming_agent()
        
        # 테스트 2: 토큰 스트리밍
        stream_tokens_demo()
        
        # 테스트 3: 프로그레스 바
        test_progress_bar()
        
        print("\n💡 학습 포인트:")
        print("  1. Streaming API 활용")
        print("  2. 실시간 진행 상황 표시")
        print("  3. 사용자 경험 개선")
        print("  4. 토큰/청크 단위 스트리밍")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
