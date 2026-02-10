"""
================================================================================
LangChain AI Agent 마스터 교안
Part 9: 프로덕션 준비 - 백엔드 서버 (FastAPI)
================================================================================

파일명: backend_server.py
설명: React 프론트엔드와 연동되는 스트리밍 Agent 서버

실행 방법:
    uvicorn backend_server:app --reload --port 8000

================================================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncIterator
import asyncio
import json
import os
import ast
import operator
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 환경 설정
load_dotenv()

# ============================================================================
# FastAPI 앱 설정
# ============================================================================

app = FastAPI(title="LangChain Agent API")

# CORS 설정 (프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite, CRA
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 안전한 수학 계산기
# ============================================================================

def safe_eval_math(expression: str) -> float:
    """
    안전하게 수학 표현식을 평가합니다.
    eval() 대신 ast를 사용하여 허용된 연산만 실행합니다.
    """
    # 허용된 연산자 매핑
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,  # 단항 마이너스
    }

    def eval_node(node):
        """AST 노드를 재귀적으로 평가합니다."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 호환성
            return node.n
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"허용되지 않은 연산자: {op_type.__name__}")
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"허용되지 않은 연산자: {op_type.__name__}")
            return allowed_operators[op_type](operand)
        else:
            raise ValueError(f"허용되지 않은 노드 타입: {type(node).__name__}")

    try:
        # 표현식을 AST로 파싱
        tree = ast.parse(expression, mode='eval')
        # AST를 평가
        return eval_node(tree.body)
    except SyntaxError as e:
        raise ValueError(f"잘못된 표현식 구문: {e}")


# ============================================================================
# 도구 정의
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 조회합니다."""
    # 실제로는 API 호출, 여기서는 모의 데이터
    weather_data = {
        "서울": "맑음, 22°C",
        "부산": "흐림, 19°C",
        "뉴욕": "비, 15°C",
        "도쿄": "맑음, 20°C",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다."""
    try:
        # 안전한 수학 표현식 평가 (ast 사용)
        result = safe_eval_math(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def search_web(query: str) -> str:
    """웹 검색을 수행합니다. (모의)"""
    return f"'{query}'에 대한 검색 결과: LangChain은 LLM 애플리케이션 개발 프레임워크입니다."


# ============================================================================
# Agent 생성
# ============================================================================

def create_streaming_agent():
    """스트리밍 Agent를 생성합니다."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,  # 스트리밍 활성화
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_web],
        system_prompt="""
당신은 유용한 AI 어시스턴트입니다.
사용자의 질문에 도구를 활용하여 정확하게 답변하세요.
        """.strip(),
    )

    return agent


# ============================================================================
# API 모델
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    stream: bool = True


class ChatResponse(BaseModel):
    response: str


# ============================================================================
# SSE 스트리밍 엔드포인트
# ============================================================================

async def event_stream(message: str) -> AsyncIterator[str]:
    """
    Server-Sent Events 형식으로 Agent 응답을 스트리밍합니다.

    Yields:
        SSE 형식의 이벤트 문자열
    """
    try:
        agent = create_streaming_agent()

        # Agent 실행 (스트리밍)
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]}
        ):
            # 메시지 청크 처리
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        # 토큰 전송
                        event = {
                            "type": "token",
                            "content": msg.content,
                        }
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)  # 약간의 지연 (시각 효과)

            # 도구 호출 감지
            if "tool_calls" in chunk:
                for tool_call in chunk["tool_calls"]:
                    event = {
                        "type": "tool_call",
                        "tool": tool_call.get("name", "unknown"),
                    }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 스트리밍 완료
        yield "data: [DONE]\n\n"

    except Exception as e:
        # 에러 전송
        error_event = {
            "type": "error",
            "error": str(e),
        }
        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"


@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    채팅 메시지를 스트리밍으로 처리합니다.

    Args:
        request: 채팅 요청 (message, stream)

    Returns:
        StreamingResponse: SSE 스트림
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")

    return StreamingResponse(
        event_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================================
# 일반 (비스트리밍) 엔드포인트
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 메시지를 처리합니다 (비스트리밍).

    Args:
        request: 채팅 요청

    Returns:
        ChatResponse: Agent 응답
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")

    try:
        agent = create_streaming_agent()

        result = agent.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })

        response_content = result["messages"][-1].content

        return ChatResponse(response=response_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 헬스 체크
# ============================================================================

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "LangChain Agent API",
        "version": "1.0.0",
        "endpoints": {
            "stream": "POST /stream - 스트리밍 채팅",
            "chat": "POST /chat - 일반 채팅",
            "health": "GET /health - 헬스 체크",
        },
    }


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🚀 LangChain Agent 서버 시작")
    print("=" * 70)
    print()
    print("📍 서버 주소: http://localhost:8000")
    print("📖 API 문서: http://localhost:8000/docs")
    print("🔧 헬스 체크: http://localhost:8000/health")
    print()
    print("⌨️  종료하려면 Ctrl+C를 누르세요")
    print("=" * 70)
    print()

    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


# ============================================================================
# 📚 학습 포인트
# ============================================================================
#
# 1. FastAPI:
#    - 현대적인 Python 웹 프레임워크
#    - 자동 API 문서 생성 (/docs)
#    - 타입 힌트 기반 검증
#
# 2. CORS:
#    - 프론트엔드와 백엔드 도메인이 다를 때 필요
#    - allow_origins에 프론트엔드 주소 추가
#
# 3. Server-Sent Events:
#    - StreamingResponse로 구현
#    - data: 프리픽스 필수
#    - 이벤트 끝에 \n\n 추가
#
# 4. 비동기 처리:
#    - async/await로 비동기 Agent 실행
#    - astream()으로 비동기 스트리밍
#
# ============================================================================
# 🎓 프로덕션 고려사항
# ============================================================================
#
# - 인증/인가 (JWT, OAuth)
# - Rate Limiting
# - 에러 로깅
# - 모니터링 (Prometheus, Grafana)
# - 배포 (Docker, Kubernetes)
#
# ============================================================================
