"""
RAG Agent (Retrieval-Augmented Generation Agent)

이 예제는 문서 검색을 통해 정보를 보강하는 RAG Agent를 구현합니다.
사용자 질문에 대해 관련 문서를 검색하고, 검색 결과를 바탕으로 답변합니다.

기능:
- 문서 검색 (시뮬레이션된 벡터 스토어)
- 검색 결과 기반 답변 생성
- 소스 인용
- Adaptive RAG (검색 필요 여부 판단)

실행 방법:
    python -m examples.04_rag_agent.main
"""

import os
from typing import TypedDict, Annotated, List, Optional, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator


# =============================================================================
# 환경 설정
# =============================================================================

load_dotenv()


# =============================================================================
# 시뮬레이션된 문서 저장소
# =============================================================================

# 샘플 문서들 (실제로는 벡터 DB 사용)
DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "LangGraph 소개",
        "content": """LangGraph는 LangChain에서 개발한 상태 유지 AI 에이전트 프레임워크입니다.
        그래프 기반 아키텍처를 통해 복잡한 워크플로우를 구현할 수 있습니다.
        주요 특징으로는 순환 구조 지원, Human-in-the-Loop, 상태 영속성이 있습니다.""",
        "metadata": {"category": "framework", "date": "2024-01"}
    },
    {
        "id": "doc_2",
        "title": "LangGraph State 관리",
        "content": """LangGraph에서 State는 TypedDict 또는 Pydantic 모델로 정의합니다.
        각 노드는 State를 입력받아 업데이트된 State를 반환합니다.
        Reducer를 사용하여 상태 업데이트 방식을 커스터마이징할 수 있습니다.""",
        "metadata": {"category": "concept", "date": "2024-02"}
    },
    {
        "id": "doc_3",
        "title": "LangGraph Checkpointer",
        "content": """Checkpointer는 그래프 실행 상태를 저장하는 메커니즘입니다.
        MemorySaver, SqliteSaver, PostgresSaver 등을 사용할 수 있습니다.
        thread_id를 통해 여러 대화 세션을 관리할 수 있습니다.""",
        "metadata": {"category": "feature", "date": "2024-03"}
    },
    {
        "id": "doc_4",
        "title": "ReAct Agent 패턴",
        "content": """ReAct(Reasoning + Acting)는 AI Agent의 핵심 패턴입니다.
        추론(Thought) → 행동(Action) → 관찰(Observation) 루프를 반복합니다.
        LangGraph에서는 조건부 엣지로 이 패턴을 구현합니다.""",
        "metadata": {"category": "pattern", "date": "2024-04"}
    },
    {
        "id": "doc_5",
        "title": "Multi-Agent 시스템",
        "content": """여러 에이전트가 협업하는 시스템을 구축할 수 있습니다.
        Supervisor 패턴에서는 중앙 조율자가 작업을 분배합니다.
        Handoff 패턴에서는 에이전트가 직접 다른 에이전트에게 작업을 전달합니다.""",
        "metadata": {"category": "architecture", "date": "2024-05"}
    }
]


def search_documents(query: str, top_k: int = 3) -> List[dict]:
    """
    문서 검색 (시뮬레이션)

    실제로는 벡터 유사도 검색을 사용하지만,
    여기서는 키워드 기반 간단한 검색을 시뮬레이션합니다.
    """
    query_lower = query.lower()
    results = []

    for doc in DOCUMENTS:
        # 간단한 키워드 매칭
        title_lower = doc["title"].lower()
        content_lower = doc["content"].lower()

        score = 0
        for word in query_lower.split():
            if word in title_lower:
                score += 2
            if word in content_lower:
                score += 1

        if score > 0:
            results.append({
                **doc,
                "score": score
            })

    # 점수 기준 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# State 정의
# =============================================================================

class RAGState(TypedDict):
    """RAG Agent State"""
    question: str
    retrieved_docs: List[dict]
    context: str
    answer: str
    sources: List[str]
    needs_retrieval: bool


# =============================================================================
# RAG Agent 구현
# =============================================================================

def create_rag_agent():
    """RAG Agent 그래프 생성"""

    def analyze_question(state: RAGState) -> RAGState:
        """
        질문 분석 - 검색이 필요한지 판단

        간단한 규칙 기반 분석 (실제로는 LLM 사용)
        """
        question = state["question"].lower()

        # 검색이 필요한 키워드
        retrieval_keywords = [
            "langgraph", "state", "agent", "checkpointer",
            "react", "multi-agent", "어떻게", "무엇", "설명"
        ]

        needs_retrieval = any(kw in question for kw in retrieval_keywords)

        return {"needs_retrieval": needs_retrieval}

    def retrieve_documents(state: RAGState) -> RAGState:
        """문서 검색"""
        question = state["question"]
        docs = search_documents(question, top_k=3)

        # 컨텍스트 구성
        context_parts = []
        sources = []

        for doc in docs:
            context_parts.append(f"[{doc['title']}]\n{doc['content']}")
            sources.append(f"{doc['title']} ({doc['id']})")

        context = "\n\n---\n\n".join(context_parts)

        return {
            "retrieved_docs": docs,
            "context": context,
            "sources": sources
        }

    def generate_answer(state: RAGState) -> RAGState:
        """답변 생성"""
        question = state["question"]
        context = state.get("context", "")
        sources = state.get("sources", [])

        # LLM이 있으면 사용
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic

                llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

                if context:
                    prompt = f"""다음 문서를 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변 시 참고한 문서를 명시해주세요."""
                else:
                    prompt = f"""질문에 답변해주세요.

질문: {question}"""

                response = llm.invoke(prompt)
                answer = response.content

                return {"answer": answer}

            except ImportError:
                pass

        # 시뮬레이션 응답
        if context:
            answer = f"""질문: {question}

검색된 문서를 바탕으로 답변드립니다.

{context[:300]}...

[출처: {', '.join(sources[:2])}]"""
        else:
            answer = f"'{question}'에 대해 관련 문서 없이 답변드립니다. (시뮬레이션 모드)"

        return {"answer": answer}

    def route_by_retrieval(state: RAGState) -> str:
        """검색 필요 여부에 따라 라우팅"""
        if state.get("needs_retrieval"):
            return "retrieve"
        return "generate"

    # 그래프 구성
    graph = StateGraph(RAGState)

    graph.add_node("analyze", analyze_question)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_by_retrieval,
        {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 데모 실행
# =============================================================================

def run_demo():
    """데모 실행"""

    print("=" * 60)
    print("📚 RAG Agent Demo")
    print("=" * 60)

    agent = create_rag_agent()

    # 테스트 질문들
    questions = [
        "LangGraph란 무엇인가요?",
        "State 관리는 어떻게 하나요?",
        "Checkpointer의 종류를 알려주세요",
        "ReAct 패턴에 대해 설명해주세요",
        "오늘 날씨 어때요?",  # 관련 문서 없음
    ]

    for i, question in enumerate(questions):
        config = {"configurable": {"thread_id": f"rag_demo_{i}"}}

        print(f"\n{'='*50}")
        print(f"❓ 질문: {question}")

        result = agent.invoke({
            "question": question,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "needs_retrieval": False
        }, config=config)

        print(f"\n📖 검색된 문서: {len(result.get('retrieved_docs', []))}개")
        if result.get("sources"):
            print(f"📑 출처: {', '.join(result['sources'][:2])}")

        print(f"\n💬 답변:")
        answer = result["answer"]
        # 긴 답변은 축약
        if len(answer) > 300:
            print(f"   {answer[:300]}...")
        else:
            print(f"   {answer}")


def run_interactive():
    """인터랙티브 모드"""

    print("=" * 60)
    print("📚 RAG Agent - Interactive Mode")
    print("=" * 60)
    print("\nLangGraph 관련 질문을 해보세요!")
    print("예: 'LangGraph의 장점은?', 'State 관리 방법'")
    print("종료: /quit")
    print("-" * 60)

    agent = create_rag_agent()
    session_count = 0

    while True:
        try:
            question = input("\n❓ 질문: ").strip()

            if not question:
                continue

            if question == "/quit":
                print("\n👋 종료합니다.")
                break

            session_count += 1
            config = {"configurable": {"thread_id": f"interactive_{session_count}"}}

            result = agent.invoke({
                "question": question,
                "retrieved_docs": [],
                "context": "",
                "answer": "",
                "sources": [],
                "needs_retrieval": False
            }, config=config)

            print(f"\n📖 검색된 문서: {len(result.get('retrieved_docs', []))}개")
            if result.get("sources"):
                print(f"📑 출처: {', '.join(result['sources'])}")

            print(f"\n💬 답변:")
            print(f"   {result['answer'][:500]}...")

        except KeyboardInterrupt:
            print("\n\n👋 종료합니다.")
            break


# =============================================================================
# 메인
# =============================================================================

def main():
    """메인 함수"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive()
    else:
        run_demo()

    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
