"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP - 실습 과제 2 해답
================================================================================

과제: 스마트 검색 Agent (Agentic RAG)
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. Agent가 필요시 문서 검색
2. 검색 도구와 LLM 통합
3. 자율적인 정보 탐색

학습 목표:
- Agentic RAG 패턴
- 도구로서의 검색
- Agent의 자율적 판단

================================================================================
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# ============================================================================
# 지식 베이스
# ============================================================================

KNOWLEDGE_BASE = """
# LangChain 프레임워크

## 개요
LangChain은 LLM을 활용한 애플리케이션 개발을 위한 프레임워크입니다.
2022년에 출시되어 빠르게 성장했습니다.

## 핵심 컴포넌트
1. Models: LLM 통합 (OpenAI, Anthropic 등)
2. Prompts: 프롬프트 템플릿 관리
3. Chains: 컴포넌트 연결
4. Agents: 자율적 행동
5. Memory: 대화 기록 관리

## LangGraph
상태 기반 Agent 구축을 위한 라이브러리입니다.
StateGraph를 사용하여 복잡한 워크플로우를 구성할 수 있습니다.

### 주요 기능
- 노드와 엣지로 그래프 구성
- 조건부 라우팅
- Checkpointing (상태 저장)
- 사람의 개입 (Human-in-the-Loop)

## RAG (Retrieval-Augmented Generation)
문서 검색과 생성을 결합한 패턴입니다.
1. 문서를 Vector Store에 저장
2. 질문과 유사한 문서 검색
3. 검색된 문서를 Context로 답변 생성

## Agent 패턴
- ReAct: Reasoning + Acting
- Tool Use: 도구를 사용하여 작업 수행
- Multi-Agent: 여러 Agent 협업
"""

# ============================================================================
# Vector Store 설정
# ============================================================================

def setup_vectorstore():
    """Vector Store 초기화"""
    embeddings = OpenAIEmbeddings()
    
    # 문서 분할
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    
    docs = [Document(page_content=KNOWLEDGE_BASE)]
    splits = splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# ============================================================================
# 검색 도구
# ============================================================================

# Global vectorstore
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = setup_vectorstore()
    return _vectorstore

@tool
def search_documentation(query: str) -> str:
    """LangChain 문서를 검색합니다.
    
    Args:
        query: 검색 쿼리 (예: "RAG란 무엇인가", "Agent 패턴")
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=2)
    
    if not results:
        return "관련 문서를 찾을 수 없습니다."
    
    combined = "\n\n".join([doc.page_content for doc in results])
    return f"검색 결과:\n\n{combined}"

@tool
def get_example_code(topic: str) -> str:
    """예제 코드를 제공합니다.
    
    Args:
        topic: 주제 (예: "agent", "rag", "memory")
    """
    examples = {
        "agent": """
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(model, tools=[])
result = agent.invoke({"messages": [...]})
```
        """,
        "rag": """
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
results = vectorstore.similarity_search(query)
```
        """,
        "memory": """
```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```
        """
    }
    
    topic_lower = topic.lower()
    for key, code in examples.items():
        if key in topic_lower:
            return f"예제 코드:\n{code}"
    
    return "해당 주제의 예제 코드가 없습니다."

# ============================================================================
# Agentic RAG 시스템
# ============================================================================

def create_agentic_rag():
    """Agentic RAG Agent 생성"""
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_documentation, get_example_code]
    
    system_prompt = """당신은 LangChain 전문 어시스턴트입니다.

질문에 답변할 때:
1. 먼저 문서를 검색하여 정확한 정보 확인
2. 필요시 예제 코드 제공
3. 검색 결과를 바탕으로 명확하게 설명

도구 사용 가이드:
- search_documentation: 개념, 기능, 사용법 등을 검색
- get_example_code: 코드 예제가 필요할 때

항상 검색한 정보를 바탕으로 답변하세요."""
    
    agent = create_react_agent(model, tools, state_modifier=system_prompt)
    return agent

# ============================================================================
# 테스트
# ============================================================================

def test_agentic_rag():
    """Agentic RAG 테스트"""
    print("=" * 70)
    print("🤖 스마트 검색 Agent 테스트")
    print("=" * 70)
    
    agent = create_agentic_rag()
    
    questions = [
        "LangChain이 무엇인가요?",
        "RAG 패턴에 대해 설명하고 예제 코드를 보여주세요",
        "LangGraph의 주요 기능은 무엇인가요?",
        "Agent 예제 코드를 보여주세요",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"❓ 질문 {i}: {question}")
        print("=" * 70)
        
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        
        final_message = result["messages"][-1]
        print(f"\n💡 답변:\n{final_message.content}\n")

def compare_rag_types():
    """전통적 RAG vs Agentic RAG 비교"""
    print("\n" + "=" * 70)
    print("📊 RAG 패턴 비교")
    print("=" * 70)
    
    print("""
전통적 RAG:
- 항상 문서 검색 수행
- 고정된 검색 → 생성 파이프라인
- 단순하고 예측 가능

Agentic RAG:
- Agent가 필요 여부 판단
- 유연한 도구 사용
- 복잡한 질문 처리 가능
- 여러 번 검색 가능

언제 Agentic RAG를 사용하나:
1. 복잡한 멀티스텝 질문
2. 도구 조합이 필요한 경우
3. 동적인 정보 탐색이 필요한 경우
    """)

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🤖 Part 8: 스마트 검색 Agent - 실습 과제 2 해답")
    print("=" * 70)
    
    try:
        test_agentic_rag()
        compare_rag_types()
        
        print("\n💡 학습 포인트:")
        print("  1. Agentic RAG 패턴")
        print("  2. 도구로서의 검색")
        print("  3. Agent의 자율적 판단")
        print("  4. 유연한 정보 탐색")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
