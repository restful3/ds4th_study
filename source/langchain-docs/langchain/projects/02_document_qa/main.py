"""
Document Q&A System - Main Entry Point
문서 Q&A 시스템 메인 실행 파일
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from rag_pipeline import RAGPipeline
from document_loader import DocumentLoader
from access_control import AccessControl

# 환경 변수 로드
load_dotenv()


class DocumentQASystem:
    """문서 Q&A 시스템 메인 클래스"""

    def __init__(self, docs_path: str, index_path: str = "faiss_index"):
        """
        초기화

        Args:
            docs_path: 문서 디렉토리 경로
            index_path: FAISS 인덱스 저장 경로
        """
        self.docs_path = docs_path
        self.index_path = index_path
        self.loader = DocumentLoader(docs_path)
        self.access_control = AccessControl()
        self.rag_pipeline: Optional[RAGPipeline] = None

    def initialize(self, force_reindex: bool = False):
        """
        시스템 초기화 및 인덱스 로드/생성

        Args:
            force_reindex: 강제 재인덱싱 여부
        """
        print("🚀 문서 Q&A 시스템 초기화 중...")

        # 기존 인덱스가 있고 재인덱싱이 아닌 경우 로드
        if os.path.exists(self.index_path) and not force_reindex:
            print(f"📂 기존 인덱스 로드 중: {self.index_path}")
            try:
                self.rag_pipeline = RAGPipeline.load_index(self.index_path)
                print("✅ 인덱스 로드 완료!")
                return
            except Exception as e:
                print(f"⚠️  인덱스 로드 실패: {e}")
                print("🔄 새로 인덱싱을 시작합니다...")

        # 문서 로드 및 인덱싱
        print(f"📚 문서 로드 중: {self.docs_path}")
        documents = self.loader.load_documents()

        if not documents:
            print("❌ 로드된 문서가 없습니다. 경로를 확인해주세요.")
            sys.exit(1)

        print(f"✅ {len(documents)}개의 청크가 로드되었습니다.")

        # RAG 파이프라인 생성
        print("🔧 RAG 파이프라인 구축 중...")
        self.rag_pipeline = RAGPipeline(documents)

        # 인덱스 저장
        print(f"💾 인덱스 저장 중: {self.index_path}")
        self.rag_pipeline.save_index(self.index_path)
        print("✅ 초기화 완료!")

    def run_interactive(self):
        """대화형 Q&A 인터페이스 실행"""
        if not self.rag_pipeline:
            print("❌ RAG 파이프라인이 초기화되지 않았습니다.")
            return

        print("\n" + "=" * 60)
        print("📖 문서 Q&A 시스템에 오신 것을 환영합니다!")
        print("=" * 60)

        # 사용자 로그인
        username = input("\n👤 사용자 이름을 입력하세요 (admin/developer/guest): ").strip()
        if username not in ["admin", "developer", "guest"]:
            print("⚠️  유효하지 않은 사용자입니다. guest로 로그인합니다.")
            username = "guest"

        print(f"\n안녕하세요, {username}님!")
        print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

        # Q&A 체인 생성
        qa_chain = self.rag_pipeline.create_qa_chain()

        # 대화 루프
        while True:
            try:
                question = input("❓ 질문: ").strip()

                if not question:
                    continue

                if question.lower() in ["quit", "exit", "종료"]:
                    print("\n👋 감사합니다. 좋은 하루 되세요!")
                    break

                # 질의응답 수행
                print("\n🤔 답변 생성 중...\n")
                result = qa_chain.invoke({"query": question})

                # 소스 문서 필터링 (접근 제어)
                filtered_sources = self.access_control.filter_documents(
                    username, result.get("source_documents", [])
                )

                if not filtered_sources:
                    print("❌ 접근 권한이 없는 문서입니다.")
                    continue

                # 답변 출력
                print("💡 답변:")
                print("-" * 60)
                print(result["result"])
                print("-" * 60)

                # 출처 표시
                if filtered_sources:
                    print("\n📚 출처:")
                    for i, doc in enumerate(filtered_sources, 1):
                        source = doc.metadata.get("source", "Unknown")
                        source_name = Path(source).name
                        print(f"  {i}. {source_name}")

                print("\n")

            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("다시 시도해주세요.\n")

    def run_single_query(self, question: str, username: str = "admin"):
        """
        단일 질문 처리 (API 모드)

        Args:
            question: 질문
            username: 사용자 이름

        Returns:
            dict: 답변 및 소스 정보
        """
        if not self.rag_pipeline:
            return {"error": "RAG 파이프라인이 초기화되지 않았습니다."}

        try:
            qa_chain = self.rag_pipeline.create_qa_chain()
            result = qa_chain.invoke({"query": question})

            # 접근 제어 적용
            filtered_sources = self.access_control.filter_documents(
                username, result.get("source_documents", [])
            )

            if not filtered_sources:
                return {"error": "접근 권한이 없는 문서입니다."}

            return {
                "answer": result["result"],
                "sources": [
                    {
                        "name": Path(doc.metadata.get("source", "")).name,
                        "content": doc.page_content[:200] + "...",
                    }
                    for doc in filtered_sources
                ],
            }

        except Exception as e:
            return {"error": str(e)}


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="문서 Q&A 시스템")
    parser.add_argument(
        "--docs-path",
        type=str,
        default="/Users/restful3/Desktop/langchain/datasets/sample_documents",
        help="문서 디렉토리 경로",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index",
        help="FAISS 인덱스 저장 경로",
    )
    parser.add_argument(
        "--reindex", action="store_true", help="강제 재인덱싱"
    )
    parser.add_argument(
        "--query", type=str, help="단일 질문 (대화형 모드 대신)"
    )
    parser.add_argument(
        "--user", type=str, default="admin", help="사용자 이름 (query 모드에서 사용)"
    )

    args = parser.parse_args()

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("💡 .env 파일을 생성하고 API 키를 설정해주세요.")
        sys.exit(1)

    # 시스템 초기화
    qa_system = DocumentQASystem(args.docs_path, args.index_path)
    qa_system.initialize(force_reindex=args.reindex)

    # 실행 모드 선택
    if args.query:
        # 단일 질문 모드
        result = qa_system.run_single_query(args.query, args.user)
        print("\n💡 답변:")
        print("-" * 60)
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(result["answer"])
            print("\n📚 출처:")
            for source in result["sources"]:
                print(f"  - {source['name']}")
        print("-" * 60)
    else:
        # 대화형 모드
        qa_system.run_interactive()


if __name__ == "__main__":
    main()
