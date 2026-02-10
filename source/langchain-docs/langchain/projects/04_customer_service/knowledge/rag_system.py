"""
RAG System for Customer Service
고객 서비스용 RAG 시스템
"""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path


class CustomerServiceRAG:
    """고객 서비스 RAG 시스템"""

    def __init__(self, data_path: str = None, verbose: bool = False):
        """
        초기화

        Args:
            data_path: 지식 데이터 경로
            verbose: 상세 로그
        """
        self.verbose = verbose
        self.data_path = data_path or str(Path(__file__).parent / "data")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore: Optional[FAISS] = None

        # 초기화
        self._initialize()

    def _initialize(self):
        """RAG 시스템 초기화"""
        try:
            # 기존 인덱스 로드 시도
            index_path = Path(self.data_path) / "faiss_index"
            if index_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                if self.verbose:
                    print("[RAG] 기존 인덱스 로드 완료")
            else:
                # 새로 구축
                self._build_index()
        except Exception as e:
            if self.verbose:
                print(f"[RAG] 초기화 실패: {e}")
            # 빈 벡터스토어 생성
            self.vectorstore = None

    def _build_index(self):
        """지식 베이스 인덱스 구축"""
        if self.verbose:
            print("[RAG] 지식 베이스 인덱싱 중...")

        # 샘플 문서 생성 (실제로는 data/ 디렉토리에서 로드)
        sample_docs = [
            Document(
                page_content="기술 지원: 앱이 작동하지 않으면 재시작을 시도하세요.",
                metadata={"category": "support", "source": "faq.md"}
            ),
            Document(
                page_content="결제 정책: 구매 후 7일 이내 환불이 가능합니다.",
                metadata={"category": "billing", "source": "policies.md"}
            ),
            Document(
                page_content="일반 정보: 고객센터는 평일 9시-6시에 운영됩니다.",
                metadata={"category": "general", "source": "info.md"}
            ),
        ]

        self.vectorstore = FAISS.from_documents(sample_docs, self.embeddings)

        # 저장
        index_path = Path(self.data_path) / "faiss_index"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(index_path))

        if self.verbose:
            print(f"[RAG] {len(sample_docs)}개 문서 인덱싱 완료")

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 3
    ) -> List[Document]:
        """
        지식 베이스 검색

        Args:
            query: 검색 쿼리
            category: 카테고리 필터
            k: 반환할 문서 수

        Returns:
            List[Document]: 관련 문서
        """
        if not self.vectorstore:
            return []

        try:
            # 카테고리 필터링
            if category:
                results = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"category": category}
                )
            else:
                results = self.vectorstore.similarity_search(query, k=k)

            if self.verbose:
                print(f"[RAG] {len(results)}개 관련 문서 발견")

            return results

        except Exception as e:
            if self.verbose:
                print(f"[RAG] 검색 오류: {e}")
            return []
