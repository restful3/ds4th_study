"""
Tests for Document Loader
"""

import pytest
import tempfile
import os
from pathlib import Path
from document_loader import DocumentLoader


@pytest.fixture
def temp_docs_dir():
    """임시 문서 디렉토리 생성"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 테스트 문서 생성
        doc1_path = Path(tmpdir) / "test1.md"
        doc1_path.write_text("# 테스트 문서 1\n\n이것은 테스트 문서입니다.", encoding="utf-8")

        doc2_path = Path(tmpdir) / "test2.md"
        doc2_path.write_text("# 테스트 문서 2\n\nPython은 프로그래밍 언어입니다.", encoding="utf-8")

        yield tmpdir


def test_document_loader_initialization():
    """문서 로더 초기화 테스트"""
    loader = DocumentLoader("/fake/path")

    assert loader.docs_path == "/fake/path"
    assert loader.chunk_size == 1000
    assert loader.chunk_overlap == 200
    assert loader.text_splitter is not None


def test_load_documents(temp_docs_dir):
    """문서 로드 테스트"""
    loader = DocumentLoader(temp_docs_dir)
    documents = loader.load_documents()

    assert len(documents) > 0
    assert all(hasattr(doc, "page_content") for doc in documents)
    assert all(hasattr(doc, "metadata") for doc in documents)


def test_load_documents_with_metadata(temp_docs_dir):
    """메타데이터 포함 문서 로드 테스트"""
    loader = DocumentLoader(temp_docs_dir)
    documents = loader.load_documents()

    for doc in documents:
        assert "source" in doc.metadata or "filename" in doc.metadata
        assert "length" in doc.metadata
        assert "doc_type" in doc.metadata


def test_load_nonexistent_directory():
    """존재하지 않는 디렉토리 로드 테스트"""
    loader = DocumentLoader("/nonexistent/path")

    with pytest.raises(FileNotFoundError):
        loader.load_documents()


def test_get_document_stats(temp_docs_dir):
    """문서 통계 정보 테스트"""
    loader = DocumentLoader(temp_docs_dir)
    documents = loader.load_documents()

    stats = loader.get_document_stats(documents)

    assert "total_docs" in stats
    assert "total_length" in stats
    assert "avg_length" in stats
    assert "doc_types" in stats
    assert stats["total_docs"] > 0


def test_empty_document_stats():
    """빈 문서 리스트 통계 테스트"""
    loader = DocumentLoader("/fake/path")
    stats = loader.get_document_stats([])

    assert stats["total_docs"] == 0


def test_custom_chunk_size():
    """커스텀 청크 크기 테스트"""
    loader = DocumentLoader("/fake/path", chunk_size=500, chunk_overlap=50)

    assert loader.chunk_size == 500
    assert loader.chunk_overlap == 50


def test_load_single_file(temp_docs_dir):
    """단일 파일 로드 테스트"""
    loader = DocumentLoader(temp_docs_dir)
    file_path = Path(temp_docs_dir) / "test1.md"

    documents = loader.load_single_file(str(file_path))

    assert len(documents) > 0
    assert all(doc.metadata.get("filename") == "test1.md" for doc in documents)


def test_load_nonexistent_file():
    """존재하지 않는 파일 로드 테스트"""
    loader = DocumentLoader("/fake/path")

    with pytest.raises(FileNotFoundError):
        loader.load_single_file("/nonexistent/file.md")
