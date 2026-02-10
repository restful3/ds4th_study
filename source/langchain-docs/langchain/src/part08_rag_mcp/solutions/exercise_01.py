"""
================================================================================
LangChain AI Agent 마스터 교안
Part 8: RAG & MCP - 실습 과제 1 해답
================================================================================

과제: 기술 문서 Q&A (Vector Store RAG)
난이도: ⭐⭐⭐☆☆ (중급)

요구사항:
1. 문서를 Vector Store에 저장
2. 유사도 검색으로 관련 문서 찾기
3. 검색된 문서를 바탕으로 답변 생성

학습 목표:
- RAG(Retrieval-Augmented Generation) 패턴
- Vector Store 사용
- Embedding과 유사도 검색

================================================================================
"""

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============================================================================
# 샘플 기술 문서
# ============================================================================

SAMPLE_DOCUMENTS = [
    """
    # Python 기초
    
    Python은 1991년 Guido van Rossum이 개발한 고급 프로그래밍 언어입니다.
    간결하고 읽기 쉬운 문법으로 초보자도 쉽게 배울 수 있습니다.
    
    ## 주요 특징
    - 인터프리터 언어
    - 동적 타이핑
    - 객체지향 프로그래밍 지원
    - 풍부한 표준 라이브러리
    
    ## 기본 문법
    ```python
    # 변수 선언
    name = "Alice"
    age = 25
    
    # 함수 정의
    def greet(name):
        return f"Hello, {name}!"
    ```
    """,
    
    """
    # Python 데이터 구조
    
    ## 리스트 (List)
    - 순서가 있는 변경 가능한 컬렉션
    - 대괄호 []로 표현
    - 예: numbers = [1, 2, 3, 4, 5]
    
    ## 튜플 (Tuple)
    - 순서가 있는 변경 불가능한 컬렉션
    - 소괄호 ()로 표현
    - 예: point = (10, 20)
    
    ## 딕셔너리 (Dictionary)
    - 키-값 쌍으로 저장
    - 중괄호 {}로 표현
    - 예: person = {"name": "Alice", "age": 25}
    
    ## 집합 (Set)
    - 중복 없는 컬렉션
    - 중괄호 {}로 표현
    - 예: unique_numbers = {1, 2, 3}
    """,
    
    """
    # Python 함수
    
    ## 함수 정의
    def 키워드를 사용하여 함수를 정의합니다.
    
    ```python
    def add(a, b):
        return a + b
    
    result = add(3, 5)  # 8
    ```
    
    ## 기본 매개변수
    ```python
    def greet(name, message="Hello"):
        return f"{message}, {name}!"
    
    greet("Alice")  # "Hello, Alice!"
    greet("Bob", "Hi")  # "Hi, Bob!"
    ```
    
    ## 가변 인자
    ```python
    def sum_all(*args):
        return sum(args)
    
    sum_all(1, 2, 3, 4)  # 10
    ```
    """,
    
    """
    # Python 클래스
    
    ## 클래스 정의
    ```python
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age
        
        def introduce(self):
            return f"My name is {self.name}, {self.age} years old"
    
    # 인스턴스 생성
    alice = Person("Alice", 25)
    print(alice.introduce())
    ```
    
    ## 상속
    ```python
    class Student(Person):
        def __init__(self, name, age, student_id):
            super().__init__(name, age)
            self.student_id = student_id
        
        def study(self):
            return f"{self.name} is studying"
    ```
    """,
]

# ============================================================================
# RAG 시스템 구축
# ============================================================================

class TechDocumentRAG:
    """기술 문서 RAG 시스템"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        
    def load_documents(self, documents: list[str]):
        """문서를 Vector Store에 로드"""
        print("📚 문서 로딩 중...")
        
        # Document 객체로 변환
        docs = [Document(page_content=doc) for doc in documents]
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(docs)
        
        print(f"📄 총 {len(splits)}개 청크로 분할")
        
        # Vector Store 생성
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        print("✅ Vector Store 생성 완료")
    
    def search(self, query: str, k: int = 3) -> list[Document]:
        """유사 문서 검색"""
        if not self.vectorstore:
            raise ValueError("문서를 먼저 로드하세요")
        
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def answer_question(self, question: str) -> str:
        """질문에 답변"""
        print(f"\n🔍 검색 중: {question}")
        
        # 관련 문서 검색
        relevant_docs = self.search(question)
        
        print(f"📋 {len(relevant_docs)}개 관련 문서 발견")
        
        # Context 구성
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 프롬프트 구성
        prompt = f"""다음은 기술 문서의 내용입니다:

{context}

질문: {question}

위 문서의 내용을 바탕으로 질문에 답변해주세요.
문서에 없는 내용은 추측하지 말고, "문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요."""

        # 답변 생성
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return response.content

# ============================================================================
# 테스트
# ============================================================================

def test_rag_system():
    """RAG 시스템 테스트"""
    print("=" * 70)
    print("📖 기술 문서 Q&A 시스템 테스트")
    print("=" * 70)
    
    # RAG 시스템 초기화
    rag = TechDocumentRAG()
    rag.load_documents(SAMPLE_DOCUMENTS)
    
    # 테스트 질문들
    questions = [
        "Python에서 리스트와 튜플의 차이점은?",
        "Python 함수의 기본 매개변수는 어떻게 사용하나요?",
        "Python 클래스에서 상속은 어떻게 구현하나요?",
        "Python의 창시자는 누구인가요?",
        "Java의 특징은 무엇인가요?",  # 문서에 없는 내용
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"❓ 질문 {i}: {question}")
        print("=" * 70)
        
        answer = rag.answer_question(question)
        
        print(f"\n💡 답변:\n{answer}\n")

def test_search_quality():
    """검색 품질 테스트"""
    print("\n" + "=" * 70)
    print("🔬 검색 품질 평가")
    print("=" * 70)
    
    rag = TechDocumentRAG()
    rag.load_documents(SAMPLE_DOCUMENTS)
    
    query = "리스트"
    results = rag.search(query, k=3)
    
    print(f"\n검색어: '{query}'")
    print(f"결과 수: {len(results)}\n")
    
    for i, doc in enumerate(results, 1):
        print(f"[결과 {i}]")
        print(doc.page_content[:200] + "...")
        print()

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("📖 Part 8: 기술 문서 Q&A - 실습 과제 1 해답")
    print("=" * 70)
    
    try:
        test_rag_system()
        test_search_quality()
        
        print("\n💡 학습 포인트:")
        print("  1. RAG 패턴 구현")
        print("  2. Vector Store (FAISS) 사용")
        print("  3. Embedding과 유사도 검색")
        print("  4. Context 기반 답변 생성")
    except Exception as e:
        print(f"⚠️ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
