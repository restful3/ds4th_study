"""
Customer Service Agent System - Main Entry Point
고객 서비스 에이전트 시스템 메인 실행 파일
"""

import os
import sys
import argparse
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

from agents.router import RouterAgent
from agents.support_agent import SupportAgent
from agents.billing_agent import BillingAgent
from agents.general_agent import GeneralAgent
from agents.escalation_agent import EscalationAgent
from knowledge.rag_system import CustomerServiceRAG
from middleware.hitl import HumanInTheLoop
from middleware.monitoring import Monitor
from config import Config

# 환경 변수 로드
load_dotenv()


class CustomerServiceSystem:
    """고객 서비스 Agent 시스템"""

    def __init__(self, config: Optional[Config] = None):
        """
        초기화

        Args:
            config: 시스템 설정
        """
        self.config = config or Config()

        # 컴포넌트 초기화
        self.rag = CustomerServiceRAG(verbose=self.config.verbose)
        self.hitl = HumanInTheLoop()
        self.monitor = Monitor()

        # Agent 초기화
        self.router = RouterAgent(
            "Router",
            self.config.get_llm("router"),
            verbose=self.config.verbose
        )

        self.agents = {
            "support": SupportAgent(
                "Support",
                self.config.get_llm("support"),
                self.rag,
                verbose=self.config.verbose
            ),
            "billing": BillingAgent(
                "Billing",
                self.config.get_llm("billing"),
                self.rag,
                verbose=self.config.verbose
            ),
            "general": GeneralAgent(
                "General",
                self.config.get_llm("general"),
                self.rag,
                verbose=self.config.verbose
            ),
        }

        self.escalation = EscalationAgent(
            "Escalation",
            self.config.get_llm("escalation"),
            self.hitl,
            verbose=self.config.verbose
        )

        # 세션 관리
        self.sessions = {}

    def handle_message(self, message: str, session_id: Optional[str] = None) -> dict:
        """
        고객 메시지 처리

        Args:
            message: 고객 메시지
            session_id: 세션 ID (없으면 새로 생성)

        Returns:
            dict: 응답 정보
        """
        # 세션 ID 생성 또는 조회
        if not session_id:
            session_id = str(uuid.uuid4())

        # 모니터링 시작
        self.monitor.track_request(session_id, message)

        try:
            # 1. 라우팅
            route_result = self.router.route(message)
            category = route_result["category"]
            confidence = route_result["confidence"]

            if self.config.verbose:
                print(f"\n[Router] 카테고리: {category} (신뢰도: {confidence:.0%})")

            # 2. 적절한 Agent 선택
            agent = self.agents.get(category, self.agents["general"])

            # 3. 세션 컨텍스트 조회
            context = self.sessions.get(session_id, [])

            # 4. Agent 실행
            response = agent.run({
                "message": message,
                "context": context,
                "session_id": session_id,
            })

            # 5. 에스컬레이션 필요 여부 확인
            if response.get("needs_escalation"):
                escalation_result = self.escalation.run({
                    "message": message,
                    "response": response,
                    "context": context,
                })
                if escalation_result.get("escalated"):
                    response = escalation_result

            # 6. 세션 업데이트
            self._update_session(session_id, message, response)

            # 7. 모니터링 종료
            self.monitor.track_response(
                session_id,
                response,
                category,
                confidence
            )

            return {
                "response": response.get("answer", ""),
                "agent": category,
                "confidence": confidence,
                "sources": response.get("sources", []),
                "session_id": session_id,
            }

        except Exception as e:
            self.monitor.track_error(session_id, str(e))
            return {
                "response": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "error": str(e),
                "session_id": session_id,
            }

    def _update_session(self, session_id: str, message: str, response: dict):
        """세션 업데이트"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response,
        })

    def run_interactive(self):
        """대화형 CLI 모드"""
        print("=" * 60)
        print("   고객 서비스 AI Agent 시스템")
        print("=" * 60)
        print("\n안녕하세요! 무엇을 도와드릴까요?")
        print("(종료하려면 'quit' 또는 'exit'를 입력하세요)\n")

        session_id = str(uuid.uuid4())

        while True:
            try:
                message = input("\n고객: ").strip()

                if not message:
                    continue

                if message.lower() in ["quit", "exit", "종료"]:
                    # 만족도 조사
                    self._satisfaction_survey(session_id)
                    print("\n감사합니다. 좋은 하루 되세요!")
                    break

                # 메시지 처리
                result = self.handle_message(message, session_id)

                # 응답 출력
                print(f"\n{result['agent'].title()} Agent: {result['response']}")

                # 소스 표시 (verbose 모드)
                if self.config.verbose and result.get("sources"):
                    print("\n📚 참조 문서:")
                    for source in result["sources"][:2]:
                        print(f"  - {source.metadata.get('source', 'Unknown')}")

            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")

    def _satisfaction_survey(self, session_id: str):
        """만족도 조사"""
        try:
            rating = input("\n만족도를 평가해주세요 (1-5): ").strip()
            if rating.isdigit() and 1 <= int(rating) <= 5:
                self.monitor.track_satisfaction(session_id, int(rating))
                print(f"평가해주셔서 감사합니다!")
        except:
            pass


def run_cli_mode(verbose: bool = False):
    """CLI 모드 실행"""
    config = Config(verbose=verbose)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)

    system = CustomerServiceSystem(config)
    system.run_interactive()


def run_api_mode(port: int = 8000):
    """API 모드 실행 (FastAPI)"""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
    except ImportError:
        print("❌ FastAPI를 설치해주세요: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="Customer Service API")

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 시스템 초기화
    system = CustomerServiceSystem()

    @app.post("/chat")
    async def chat(request: dict):
        """채팅 엔드포인트"""
        message = request.get("message")
        session_id = request.get("session_id")

        if not message:
            return {"error": "message is required"}

        result = system.handle_message(message, session_id)
        return result

    @app.get("/health")
    async def health():
        """헬스 체크"""
        return {"status": "healthy"}

    print(f"🚀 API 서버 시작: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="고객 서비스 Agent 시스템")
    parser.add_argument(
        "--api", action="store_true", help="API 모드로 실행"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="API 포트 (기본: 8000)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="상세 출력"
    )
    parser.add_argument(
        "--demo", action="store_true", help="데모 모드 (샘플 대화)"
    )

    args = parser.parse_args()

    if args.api:
        run_api_mode(args.port)
    elif args.demo:
        print("데모 모드는 구현 예정입니다.")
    else:
        run_cli_mode(args.verbose)


if __name__ == "__main__":
    main()
