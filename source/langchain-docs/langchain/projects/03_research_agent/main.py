"""
Research Agent System - Main Entry Point
연구 에이전트 시스템 메인 실행 파일
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from multi_agent_system import ResearchAgentSystem

# 환경 변수 로드
load_dotenv()


def save_report(report: str, topic: str, output_dir: str = "reports"):
    """
    보고서를 파일로 저장

    Args:
        report: 보고서 내용
        topic: 연구 주제
        output_dir: 저장 디렉토리
    """
    # 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)

    # 파일명 생성 (안전한 파일명으로 변환)
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_topic = safe_topic.replace(' ', '_')[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.md"
    filepath = Path(output_dir) / filename

    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n💾 보고서가 '{filepath}'에 저장되었습니다.")
    return filepath


def run_interactive():
    """대화형 모드 실행"""
    print("=" * 60)
    print("🔬 연구 에이전트 시스템에 오신 것을 환영합니다!")
    print("=" * 60)
    print("\n이 시스템은 4개의 AI Agent가 협업하여 자동으로 연구 보고서를 생성합니다:")
    print("  📋 Planner: 연구 계획 수립")
    print("  🔍 Searcher: 정보 수집")
    print("  📊 Analyst: 데이터 분석")
    print("  ✍️  Writer: 보고서 작성")
    print("\n종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    # 시스템 초기화
    system = ResearchAgentSystem()

    while True:
        try:
            topic = input("\n💡 연구 주제를 입력하세요: ").strip()

            if not topic:
                continue

            if topic.lower() in ["quit", "exit", "종료"]:
                print("\n👋 감사합니다. 좋은 하루 되세요!")
                break

            print(f"\n🚀 '{topic}'에 대한 연구를 시작합니다...\n")
            print("━" * 60)

            # 연구 실행
            report = system.research(topic)

            # 결과 출력
            print("\n" + "━" * 60)
            print("📄 연구 보고서")
            print("━" * 60)
            print(report)
            print("━" * 60)

            # 저장 여부 확인
            save_choice = input("\n💾 보고서를 저장하시겠습니까? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes', '예', 'ㅇ']:
                save_report(report, topic)

        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("다시 시도해주세요.\n")


def run_single_query(topic: str, verbose: bool = False, save: bool = True):
    """
    단일 쿼리 모드 실행

    Args:
        topic: 연구 주제
        verbose: 상세 출력 여부
        save: 자동 저장 여부
    """
    print(f"\n🚀 '{topic}'에 대한 연구를 시작합니다...\n")
    print("━" * 60)

    # 시스템 초기화
    system = ResearchAgentSystem(verbose=verbose)

    try:
        # 연구 실행
        report = system.research(topic)

        # 결과 출력
        print("\n" + "━" * 60)
        print("📄 연구 보고서")
        print("━" * 60)
        print(report)
        print("━" * 60)

        # 자동 저장
        if save:
            save_report(report, topic)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent 연구 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 대화형 모드
  python main.py

  # 단일 쿼리
  python main.py --query "인공지능의 미래"

  # 상세 모드
  python main.py --query "양자 컴퓨팅" --verbose

  # 저장하지 않기
  python main.py --query "블록체인 기술" --no-save
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="연구 주제 (단일 쿼리 모드)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 출력 모드"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="보고서 자동 저장 비활성화"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="보고서 저장 디렉토리 (기본: reports)"
    )

    args = parser.parse_args()

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("💡 .env 파일을 생성하고 API 키를 설정해주세요.")
        sys.exit(1)

    # 실행 모드 선택
    if args.query:
        # 단일 쿼리 모드
        run_single_query(
            args.query,
            verbose=args.verbose,
            save=not args.no_save
        )
    else:
        # 대화형 모드
        run_interactive()


if __name__ == "__main__":
    main()
