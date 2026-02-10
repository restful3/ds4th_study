"""
RAG Pipeline Implementation
RAG (Retrieval-Augmented Generation) 파이프라인 구현
"""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document


class RAGPipeline:
    """RAG 파이프라인 클래스"""

    def __init__(self, documents: Optional[List[Document]] = None):
        """
        초기화

        Args:
            documents: 문서 리스트 (새로 인덱싱할 경우)
        """
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore: Optional[FAISS] = None

        if documents:
            self._build_vectorstore(documents)

    def _build_vectorstore(self, documents: List[Document]):
        """
        벡터 스토어 구축

        Args:
            documents: 인덱싱할 문서 리스트
        """
        print(f"🔨 {len(documents)}개 문서로 벡터 스토어 구축 중...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("✅ 벡터 스토어 구축 완료!")

    def create_qa_chain(self, k: int = 3) -> RetrievalQA:
        """
        Q&A 체인 생성

        Args:
            k: 검색할 상위 문서 개수

        Returns:
            RetrievalQA: 질의응답 체인
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")

        # 한국어 최적화 프롬프트
        template = """당신은 문서 기반 질의응답 전문가입니다.
주어진 문맥(context)을 바탕으로 질문에 정확하게 답변하세요.

지침:
1. 문맥에 있는 정보만 사용하여 답변하세요.
2. 답변을 모르면 "주어진 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
3. 추측하거나 외부 지식을 사용하지 마세요.
4. 답변은 명확하고 구체적으로 작성하세요.
5. 가능한 한 문맥의 원문을 활용하세요.

문맥:
{context}

질문: {question}

답변:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": k}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def search_similar_documents(
        self, query: str, k: int = 3
    ) -> List[tuple[Document, float]]:
        """
        유사 문서 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수

        Returns:
            List[tuple[Document, float]]: (문서, 유사도 점수) 튜플 리스트
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def add_documents(self, documents: List[Document]):
        """
        기존 벡터 스토어에 문서 추가

        Args:
            documents: 추가할 문서 리스트
        """
        if not self.vectorstore:
            self._build_vectorstore(documents)
        else:
            self.vectorstore.add_documents(documents)
            print(f"✅ {len(documents)}개 문서가 추가되었습니다.")

    def save_index(self, path: str):
        """
        벡터 스토어 인덱스 저장

        Args:
            path: 저장 경로
        """
        if not self.vectorstore:
            raise ValueError("저장할 벡터 스토어가 없습니다.")

        self.vectorstore.save_local(path)
        print(f"✅ 인덱스가 {path}에 저장되었습니다.")

    @classmethod
    def load_index(cls, path: str) -> "RAGPipeline":
        """
        저장된 인덱스 로드

        Args:
            path: 인덱스 경로

        Returns:
            RAGPipeline: 로드된 파이프라인 인스턴스
        """
        pipeline = cls()
        pipeline.vectorstore = FAISS.load_local(
            path, pipeline.embeddings, allow_dangerous_deserialization=True
        )
        print(f"✅ 인덱스가 {path}에서 로드되었습니다.")
        return pipeline

    def get_stats(self) -> dict:
        """
        벡터 스토어 통계 정보 반환

        Returns:
            dict: 통계 정보
        """
        if not self.vectorstore:
            return {"status": "empty", "count": 0}

        # FAISS 인덱스 크기 확인
        index_size = self.vectorstore.index.ntotal

        return {
            "status": "active",
            "count": index_size,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
        }


class HybridRAGPipeline(RAGPipeline):
    """
    하이브리드 검색을 사용하는 고급 RAG 파이프라인
    (도전 과제용)
    """

    def __init__(self, documents: Optional[List[Document]] = None):
        super().__init__(documents)
        self.bm25_retriever = None

    def _build_vectorstore(self, documents: List[Document]):
        """벡터 스토어 및 BM25 인덱스 구축"""
        super()._build_vectorstore(documents)

        # BM25 retriever 구축
        try:
            from langchain.retrievers import BM25Retriever

            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 3
            print("✅ BM25 retriever 구축 완료!")
        except ImportError:
            print("⚠️  BM25Retriever를 사용할 수 없습니다. rank_bm25 설치 필요")

    def create_qa_chain(self, k: int = 3) -> RetrievalQA:
        """하이브리드 검색을 사용하는 Q&A 체인 생성"""
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")

        # BM25가 없으면 일반 RAG 체인 반환
        if not self.bm25_retriever:
            return super().create_qa_chain(k)

        # 앙상블 retriever 생성
        from langchain.retrievers import EnsembleRetriever

        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7],  # FAISS에 더 높은 가중치
        )

        template = """당신은 문서 기반 질의응답 전문가입니다.
주어진 문맥을 바탕으로 질문에 정확하게 답변하세요.

문맥:
{context}

질문: {question}

답변:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
