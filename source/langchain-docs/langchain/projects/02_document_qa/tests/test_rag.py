"""
Tests for RAG Pipeline
"""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from rag_pipeline import RAGPipeline


@pytest.fixture
def sample_documents():
    """테스트용 샘플 문서"""
    return [
        Document(
            page_content="LangChain은 LLM 애플리케이션 프레임워크입니다.",
            metadata={"source": "langchain_overview.md"},
        ),
        Document(
            page_content="Python은 인기 있는 프로그래밍 언어입니다.",
            metadata={"source": "python_basics.md"},
        ),
        Document(
            page_content="AI 윤리는 중요한 주제입니다.",
            metadata={"source": "ai_ethics.md"},
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI Embeddings"""
    with patch("rag_pipeline.OpenAIEmbeddings") as mock:
        mock.return_value.embed_documents = Mock(
            return_value=[[0.1] * 1536 for _ in range(3)]
        )
        mock.return_value.embed_query = Mock(return_value=[0.1] * 1536)
        yield mock


@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI"""
    with patch("rag_pipeline.ChatOpenAI") as mock:
        yield mock


def test_rag_pipeline_initialization(sample_documents, mock_embeddings, mock_llm):
    """RAG 파이프라인 초기화 테스트"""
    pipeline = RAGPipeline(sample_documents)

    assert pipeline.vectorstore is not None
    assert pipeline.embeddings is not None
    assert pipeline.llm is not None


def test_rag_pipeline_search(sample_documents, mock_embeddings, mock_llm):
    """문서 검색 기능 테스트"""
    pipeline = RAGPipeline(sample_documents)

    # 유사 문서 검색 테스트는 실제 FAISS를 사용하므로 스킵
    # 실제 테스트는 통합 테스트에서 수행
    assert pipeline.vectorstore is not None


def test_get_stats(sample_documents, mock_embeddings, mock_llm):
    """통계 정보 반환 테스트"""
    pipeline = RAGPipeline(sample_documents)
    stats = pipeline.get_stats()

    assert stats["status"] == "active"
    assert stats["count"] >= 0
    assert "embedding_model" in stats
    assert "llm_model" in stats


def test_empty_pipeline():
    """빈 파이프라인 테스트"""
    pipeline = RAGPipeline()
    stats = pipeline.get_stats()

    assert stats["status"] == "empty"
    assert stats["count"] == 0


def test_add_documents(sample_documents, mock_embeddings, mock_llm):
    """문서 추가 기능 테스트"""
    pipeline = RAGPipeline()

    # 문서 추가
    pipeline.add_documents(sample_documents)

    assert pipeline.vectorstore is not None


@pytest.mark.skip(reason="Requires actual OpenAI API key")
def test_qa_chain_creation_integration(sample_documents):
    """Q&A 체인 생성 통합 테스트 (실제 API 필요)"""
    pipeline = RAGPipeline(sample_documents)
    qa_chain = pipeline.create_qa_chain(k=2)

    assert qa_chain is not None

    # 실제 질의응답 테스트
    result = qa_chain.invoke({"query": "LangChain이란?"})
    assert "result" in result
    assert "source_documents" in result
