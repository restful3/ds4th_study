"""
Document Loader Module
문서 로딩 및 전처리 모듈
"""

import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """문서 로더 클래스"""

    def __init__(
        self,
        docs_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        초기화

        Args:
            docs_path: 문서 디렉토리 경로
            chunk_size: 청크 크기 (기본: 1000)
            chunk_overlap: 청크 중첩 크기 (기본: 200)
        """
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents(self) -> List[Document]:
        """
        문서 디렉토리에서 모든 Markdown 파일 로드

        Returns:
            List[Document]: 청크로 분할된 문서 리스트
        """
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"문서 경로를 찾을 수 없습니다: {self.docs_path}")

        print(f"📂 문서 로드 중: {self.docs_path}")

        # Markdown 파일 로드
        try:
            loader = DirectoryLoader(
                self.docs_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                show_progress=True,
            )
            documents = loader.load()
        except Exception as e:
            print(f"⚠️  TextLoader 실패, UnstructuredMarkdownLoader 시도: {e}")
            documents = self._load_with_unstructured()

        if not documents:
            print("⚠️  로드된 문서가 없습니다.")
            return []

        print(f"✅ {len(documents)}개 문서 로드 완료")

        # 메타데이터 추가
        for doc in documents:
            self._enhance_metadata(doc)

        # 문서 분할
        print(f"✂️  문서 분할 중 (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ {len(chunks)}개 청크 생성 완료")

        return chunks

    def _load_with_unstructured(self) -> List[Document]:
        """UnstructuredMarkdownLoader를 사용한 문서 로드"""
        documents = []
        for file_path in Path(self.docs_path).glob("**/*.md"):
            try:
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"⚠️  {file_path} 로드 실패: {e}")
        return documents

    def _enhance_metadata(self, doc: Document):
        """
        문서 메타데이터 강화

        Args:
            doc: 메타데이터를 추가할 문서
        """
        source_path = doc.metadata.get("source", "")

        # 파일 이름 추가
        if source_path:
            doc.metadata["filename"] = Path(source_path).name
            doc.metadata["file_extension"] = Path(source_path).suffix

        # 문서 길이 추가
        doc.metadata["length"] = len(doc.page_content)

        # 문서 타입 추측
        doc.metadata["doc_type"] = self._guess_doc_type(doc.page_content)

    def _guess_doc_type(self, content: str) -> str:
        """
        문서 내용으로부터 타입 추측

        Args:
            content: 문서 내용

        Returns:
            str: 추측된 문서 타입
        """
        content_lower = content.lower()

        if "```python" in content_lower or "def " in content_lower:
            return "code_tutorial"
        elif "langchain" in content_lower:
            return "langchain_doc"
        elif "윤리" in content_lower or "ethics" in content_lower:
            return "ethics"
        else:
            return "general"

    def load_single_file(self, file_path: str) -> List[Document]:
        """
        단일 파일 로드

        Args:
            file_path: 파일 경로

        Returns:
            List[Document]: 청크로 분할된 문서 리스트
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        print(f"📄 단일 파일 로드: {file_path}")

        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        except Exception as e:
            print(f"⚠️  TextLoader 실패: {e}")
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()

        # 메타데이터 강화
        for doc in documents:
            self._enhance_metadata(doc)

        # 문서 분할
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ {len(chunks)}개 청크 생성 완료")

        return chunks

    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        문서 통계 정보 반환

        Args:
            documents: 문서 리스트

        Returns:
            dict: 통계 정보
        """
        if not documents:
            return {"total_docs": 0}

        total_length = sum(len(doc.page_content) for doc in documents)
        avg_length = total_length / len(documents) if documents else 0

        # 문서 타입별 개수
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.get("doc_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            "total_docs": len(documents),
            "total_length": total_length,
            "avg_length": round(avg_length, 2),
            "doc_types": doc_types,
        }


class CustomMarkdownLoader(DocumentLoader):
    """
    커스텀 Markdown 로더 (헤더 기반 분할)
    도전 과제용
    """

    def __init__(self, docs_path: str, **kwargs):
        super().__init__(docs_path, **kwargs)
        # 마크다운 헤더 기반 분할기
        from langchain.text_splitter import MarkdownHeaderTextSplitter

        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )

    def load_documents(self) -> List[Document]:
        """헤더 기반으로 Markdown 문서 로드"""
        documents = []

        for file_path in Path(self.docs_path).glob("**/*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 헤더 기반 분할
                md_docs = self.md_splitter.split_text(content)

                # 메타데이터 추가
                for doc in md_docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["filename"] = file_path.name
                    self._enhance_metadata(doc)

                documents.extend(md_docs)
                print(f"✅ {file_path.name}: {len(md_docs)}개 섹션")

            except Exception as e:
                print(f"⚠️  {file_path} 로드 실패: {e}")

        # 추가 청킹 (섹션이 너무 큰 경우)
        final_chunks = []
        for doc in documents:
            if len(doc.page_content) > self.chunk_size:
                chunks = self.text_splitter.split_documents([doc])
                final_chunks.extend(chunks)
            else:
                final_chunks.append(doc)

        print(f"✅ 총 {len(final_chunks)}개 청크 생성 완료")
        return final_chunks
