"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: 배포와 관측성 (Deployment & Observability)
================================================================================

파일명: 05_deployment.py
난이도: ⭐⭐⭐⭐⭐ (전문가)
예상 시간: 30분

📚 학습 목표:
  - Docker 컨테이너화
  - API 서버 구축
  - 스케일링 전략
  - 환경 설정 관리
  - 프로덕션 체크리스트

📖 공식 문서:
  • LangServe: https://python.langchain.com/docs/langserve
  • Deployment: https://python.langchain.com/docs/deployment

📄 교안 문서:
  • Part 10 개요: /docs/part10_deployment.md

🔧 필요한 패키지:
  pip install langchain langchain-openai fastapi uvicorn

🔑 필요한 환경변수:
  - OPENAI_API_KEY

🚀 실행 방법:
  python 05_deployment.py

================================================================================
"""

# ============================================================================
# Imports
# ============================================================================

import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

# ============================================================================
# 환경 설정
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
    sys.exit(1)

# ============================================================================
# 예제 1: Docker 컨테이너화
# ============================================================================

def example_1_docker():
    """Docker 컨테이너화"""
    print("=" * 70)
    print("📌 예제 1: Docker 컨테이너화")
    print("=" * 70)

    print("""
🐳 Docker란?

정의:
  애플리케이션을 컨테이너로 패키징하여
  어디서나 동일하게 실행할 수 있도록 하는 플랫폼

왜 Docker를 사용하는가?
  1️⃣ 환경 일관성
     • 개발/스테이징/프로덕션 동일 환경
     • "내 컴퓨터에서는 되는데?" 문제 해결

  2️⃣ 이식성
     • 모든 클라우드 플랫폼에서 실행
     • 로컬 -> AWS/GCP/Azure 쉬운 이동

  3️⃣ 격리성
     • 각 서비스 독립 실행
     • 의존성 충돌 방지

  4️⃣ 확장성
     • 컨테이너 복제로 쉬운 스케일링
     • 오케스트레이션 (Kubernetes) 가능

Docker 핵심 개념:
  • Image: 실행 가능한 패키지
  • Container: Image의 실행 인스턴스
  • Dockerfile: Image 빌드 명령서
  • Docker Compose: 다중 컨테이너 관리
    """)

    print("\n🔹 Dockerfile 예제:")
    print("-" * 70)

    print("""
📄 Dockerfile (Python LangChain Agent):
""")
    print('''
# 베이스 이미지
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정 (기본값)
ENV PORT=8000
ENV WORKERS=4

# 포트 노출
EXPOSE 8000

# 헬스체크 (선택)
HEALTHCHECK --interval=30s --timeout=3s \\
  CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
''')

    print("\n📄 .dockerignore:")
    print('''
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Tests
.pytest_cache/
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Secrets
.env
*.key
*.pem

# Others
*.log
tmp/
temp/
''')

    print("\n📄 requirements.txt:")
    print('''
langchain==0.1.0
langchain-openai==0.0.5
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
python-dotenv==1.0.0
langsmith==0.0.87
''')

    print("\n🔹 Docker 명령어:")
    print("-" * 70)
    print("""
# 1. 이미지 빌드
docker build -t my-langchain-agent:latest .

# 2. 이미지 크기 확인
docker images | grep my-langchain-agent

# 3. 컨테이너 실행
docker run -d \\
  --name langchain-agent \\
  -p 8000:8000 \\
  -e OPENAI_API_KEY=sk-xxx \\
  -e LANGSMITH_API_KEY=lsv2_xxx \\
  my-langchain-agent:latest

# 4. 로그 확인
docker logs -f langchain-agent

# 5. 컨테이너 내부 접속
docker exec -it langchain-agent bash

# 6. 컨테이너 중지
docker stop langchain-agent

# 7. 컨테이너 제거
docker rm langchain-agent

# 8. 이미지 푸시 (Docker Hub)
docker tag my-langchain-agent:latest username/my-langchain-agent:latest
docker push username/my-langchain-agent:latest
    """)

    print("\n📄 docker-compose.yml (다중 서비스):")
    print('''
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - agent
    restart: unless-stopped

volumes:
  redis_data:
''')

    print("\n🔹 Docker Compose 명령어:")
    print('''
# 1. 전체 스택 시작
docker-compose up -d

# 2. 로그 보기
docker-compose logs -f

# 3. 특정 서비스만 재시작
docker-compose restart agent

# 4. 전체 중지 및 제거
docker-compose down

# 5. 볼륨까지 제거
docker-compose down -v
    ''')

    print("\n💡 Docker 최적화 팁:")
    print("   • Multi-stage build로 이미지 크기 최소화")
    print("   • .dockerignore로 불필요한 파일 제외")
    print("   • 레이어 캐싱 활용 (자주 변경되는 파일은 나중에)")
    print("   • 보안: 비밀키는 환경 변수나 Secrets 사용")
    print("   • 헬스체크 추가하여 컨테이너 상태 모니터링")


# ============================================================================
# 예제 2: FastAPI 서버 구축
# ============================================================================

def example_2_api_server():
    """API 서버 구축"""
    print("\n" + "=" * 70)
    print("📌 예제 2: FastAPI 서버 구축")
    print("=" * 70)

    print("""
🚀 FastAPI란?

정의:
  고성능 Python 웹 프레임워크

장점:
  • 빠름 (Starlette 기반)
  • 자동 API 문서 (Swagger/OpenAPI)
  • 타입 힌트 지원
  • 비동기 처리
  • Pydantic 검증

LangChain + FastAPI:
  • Agent를 REST API로 제공
  • 프론트엔드와 쉽게 통합
  • 여러 클라이언트 동시 처리
    """)

    print("\n🔹 FastAPI 서버 코드 예제:")
    print("-" * 70)

    print("""
📄 main.py (FastAPI 서버):
""")
    print('''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os

# FastAPI 앱 생성
app = FastAPI(
    title="LangChain Agent API",
    description="AI Agent REST API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 구체적으로 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tools 정의
@tool
def search_docs(query: str) -> str:
    """문서를 검색합니다."""
    return f"{query}에 대한 검색 결과"

@tool
def calculate(expression: str) -> str:
    """계산을 수행합니다."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"오류: {e}"

# Agent 생성 (전역)
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_docs, calculate],
)

# Request/Response 모델
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    metadata: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    metadata: Optional[dict] = None

# 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "LangChain Agent API", "status": "running"}

@app.get("/health")
async def health():
    """헬스체크 엔드포인트"""
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 엔드포인트"""
    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })

        answer = response['messages'][-1].content

        return ChatResponse(
            response=answer,
            session_id=request.session_id,
            metadata=request.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def info():
    """API 정보"""
    return {
        "name": "LangChain Agent API",
        "version": "1.0.0",
        "tools": ["search_docs", "calculate"],
        "model": "gpt-4o-mini"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
''')

    print("\n🔹 실제 서버 시뮬레이션:")
    print("-" * 70)

    # 간단한 Agent 생성
    @tool
    def demo_search(query: str) -> str:
        """데모 검색"""
        return f"'{query}' 검색 결과"

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[demo_search],
    )

    print("\n시뮬레이션: API 요청 처리")
    print()

    # 요청 시뮬레이션
    test_requests = [
        {"message": "인공지능이란?", "session_id": "sess-001"},
        {"message": "LangChain 사용법", "session_id": "sess-002"},
    ]

    for i, req in enumerate(test_requests, 1):
        print(f"[요청 {i}]")
        print(f"  POST /chat")
        print(f"  Body: {req}")

        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": req['message']}]
            })
            answer = response['messages'][-1].content

            print(f"  응답: {{")
            print(f"    'response': '{answer[:60]}...',")
            print(f"    'session_id': '{req['session_id']}'")
            print(f"  }}")
            print(f"  상태: 200 OK\n")
        except Exception as e:
            print(f"  오류: {e}")
            print(f"  상태: 500 Internal Server Error\n")

    print("-" * 70)

    print("\n📄 클라이언트 예제 (Python):")
    print('''
import requests

# API 기본 URL
API_URL = "http://localhost:8000"

# 채팅 요청
response = requests.post(
    f"{API_URL}/chat",
    json={
        "message": "안녕하세요!",
        "session_id": "user-123",
        "metadata": {"source": "web"}
    }
)

print(response.json())
# {'response': '안녕하세요! ...', 'session_id': 'user-123', ...}
    ''')

    print("\n📄 클라이언트 예제 (JavaScript):")
    print('''
// Fetch API
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: '안녕하세요!',
    session_id: 'user-123',
    metadata: { source: 'web' }
  })
})
.then(response => response.json())
.then(data => console.log(data));
    ''')

    print("\n💡 API 서버 모범 사례:")
    print("   • 인증/인가 추가 (JWT, API Key)")
    print("   • Rate Limiting (과도한 요청 방지)")
    print("   • 에러 처리 및 상세 메시지")
    print("   • API 버저닝 (/v1/chat, /v2/chat)")
    print("   • 자동 API 문서 활용 (/docs)")
    print("   • 로깅 및 모니터링")
    print("   • CORS 정책 적절히 설정")


# ============================================================================
# 예제 3: 스케일링 전략
# ============================================================================

def example_3_scaling():
    """스케일링 전략"""
    print("\n" + "=" * 70)
    print("📌 예제 3: 스케일링 전략")
    print("=" * 70)

    print("""
📈 스케일링 (Scaling)이란?

정의:
  트래픽 증가에 대응하여 시스템 용량을 확장하는 것

스케일링 방식:

1️⃣ 수직 스케일링 (Vertical Scaling / Scale Up)
   • 정의: 서버 사양 업그레이드
   • 예: CPU 2 → 8 코어, RAM 4GB → 32GB
   • 장점: 간단, 코드 변경 불필요
   • 단점: 물리적 한계, 비용 급증, 단일 장애점

2️⃣ 수평 스케일링 (Horizontal Scaling / Scale Out)
   • 정의: 서버 개수 증가
   • 예: 서버 1대 → 5대
   • 장점: 무한 확장 가능, 고가용성
   • 단점: 복잡성, 상태 관리 필요

LangChain Agent 스케일링:
  • Stateless 설계 (세션은 외부 저장)
  • 로드 밸런서 사용
  • 컨테이너 오케스트레이션 (Kubernetes)
  • 캐싱 (Redis)
  • 비동기 처리 (Celery, RabbitMQ)

스케일링 지표:
  • CPU 사용률 > 70%
  • 메모리 사용률 > 80%
  • 응답 시간 > SLA
  • 에러율 증가
    """)

    print("\n🔹 로드 밸런서 설정 (Nginx):")
    print("-" * 70)

    print("""
📄 nginx.conf:
""")
    print('''
http {
    # Upstream 서버 그룹 정의
    upstream agent_backend {
        # 로드 밸런싱 알고리즘 (기본: round-robin)
        # least_conn;  # 연결 수가 적은 서버 우선
        # ip_hash;     # 같은 IP는 같은 서버로

        server agent-1:8000 weight=3;
        server agent-2:8000 weight=2;
        server agent-3:8000 weight=1;
        server agent-4:8000 backup;  # 백업 서버
    }

    server {
        listen 80;
        server_name api.example.com;

        # 헬스체크
        location /health {
            access_log off;
            return 200 "healthy\\n";
        }

        # API 프록시
        location / {
            proxy_pass http://agent_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # 타임아웃 설정
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
''')

    print("\n📄 Kubernetes Deployment:")
    print('''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-agent
spec:
  replicas: 3  # 초기 Pod 수
  selector:
    matchLabels:
      app: langchain-agent
  template:
    metadata:
      labels:
        app: langchain-agent
    spec:
      containers:
      - name: agent
        image: myregistry/langchain-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: langchain-agent-service
spec:
  selector:
    app: langchain-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langchain-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langchain-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
''')

    print("\n🔹 캐싱 전략 (Redis):")
    print('''
from redis import Redis
from functools import lru_cache
import hashlib
import json

# Redis 클라이언트
redis_client = Redis(host='localhost', port=6379, db=0)

def get_cache_key(query: str) -> str:
    """캐시 키 생성"""
    return f"agent:response:{hashlib.md5(query.encode()).hexdigest()}"

def get_cached_response(query: str):
    """캐시에서 응답 조회"""
    key = get_cache_key(query)
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def set_cached_response(query: str, response: str, ttl: int = 3600):
    """응답을 캐시에 저장"""
    key = get_cache_key(query)
    redis_client.setex(key, ttl, json.dumps(response))

# 사용 예
@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. 캐시 확인
    cached = get_cached_response(request.message)
    if cached:
        return ChatResponse(response=cached, from_cache=True)

    # 2. Agent 실행
    response = agent.invoke(...)
    answer = response['messages'][-1].content

    # 3. 캐시 저장
    set_cached_response(request.message, answer)

    return ChatResponse(response=answer, from_cache=False)
    ''')

    print("\n💡 스케일링 모범 사례:")
    print("   • Stateless 아키텍처 (세션은 Redis 등에 저장)")
    print("   • 자동 스케일링 (HPA) 설정")
    print("   • 헬스체크 필수 구현")
    print("   • 로드 밸런서 뒤에 여러 인스턴스")
    print("   • 캐싱으로 중복 요청 최적화")
    print("   • 비동기 작업은 큐 시스템 사용")
    print("   • 모니터링으로 스케일링 시점 파악")


# ============================================================================
# 예제 4: 환경 설정 관리
# ============================================================================

def example_4_config_management():
    """환경 설정 관리"""
    print("\n" + "=" * 70)
    print("📌 예제 4: 환경 설정 관리")
    print("=" * 70)

    print("""
⚙️ 환경 설정 관리:

왜 중요한가?
  • 환경별 설정 분리 (dev/staging/prod)
  • 보안 (비밀키 보호)
  • 유지보수성
  • 배포 유연성

환경 설정 계층:
  1️⃣ 기본 설정 (config.py)
  2️⃣ 환경 변수 (.env)
  3️⃣ 비밀 관리 (Secrets Manager)
  4️⃣ 명령줄 인자

설정 우선순위:
  명령줄 > 환경 변수 > 설정 파일 > 기본값

도구:
  • python-dotenv (로컬)
  • AWS Secrets Manager (프로덕션)
  • HashiCorp Vault (엔터프라이즈)
  • Kubernetes ConfigMap/Secrets
    """)

    print("\n🔹 설정 관리 코드:")
    print("-" * 70)

    print("""
📄 config.py (설정 클래스):
""")
    print('''
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """애플리케이션 설정"""

    # API 설정
    api_title: str = "LangChain Agent API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 환경
    environment: str = "development"  # development, staging, production
    debug: bool = False

    # LLM 설정
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000

    # LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    langsmith_tracing: bool = False

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # 보안
    api_key_enabled: bool = False
    api_keys: list[str] = []
    cors_origins: list[str] = ["*"]

    # 성능
    max_concurrent_requests: int = 100
    request_timeout: int = 60
    cache_ttl: int = 3600

    # 로깅
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# 설정 인스턴스 (싱글톤)
settings = Settings()
''')

    print("\n📄 .env.example (템플릿):")
    print('''
# API 설정
API_HOST=0.0.0.0
API_PORT=8000

# 환경
ENVIRONMENT=development
DEBUG=true

# OpenAI
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000

# LangSmith (선택)
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=my-project
LANGSMITH_TRACING=false

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# 보안
API_KEY_ENABLED=false
API_KEYS=key1,key2,key3
CORS_ORIGINS=http://localhost:3000,https://example.com

# 성능
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=60
CACHE_TTL=3600

# 로깅
LOG_LEVEL=INFO
LOG_FORMAT=json
''')

    print("\n📄 환경별 설정 파일:")
    print('''
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
API_KEY_ENABLED=true
LANGSMITH_TRACING=true
''')

    print("\n🔹 사용 예제:")
    print('''
from config import settings
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug
)

# API Key 인증
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key 검증"""
    if not settings.api_key_enabled:
        return True

    if api_key not in settings.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return True

# 설정 사용
@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    # OpenAI 설정 사용
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.openai_max_tokens
    )
    ...
''')

    print("\n🔹 현재 설정 시뮬레이션:")
    print("-" * 70)

    # 간단한 설정 시뮬레이션
    config = {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "api_host": os.getenv("API_HOST", "0.0.0.0"),
        "api_port": int(os.getenv("API_PORT", "8000")),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }

    print("\n현재 환경 설정:")
    for key, value in config.items():
        print(f"  • {key}: {value}")

    print("\n" + "-" * 70)

    print("\n💡 환경 설정 모범 사례:")
    print("   • 비밀키는 절대 Git에 커밋하지 않기")
    print("   • .env.example 제공 (템플릿)")
    print("   • 환경별 설정 파일 분리")
    print("   • Pydantic으로 설정 검증")
    print("   • 프로덕션에서는 Secrets Manager 사용")
    print("   • 설정 변경 시 재시작 불필요하게 설계")
    print("   • 민감 정보는 로그에 출력하지 않기")


# ============================================================================
# 예제 5: 프로덕션 체크리스트
# ============================================================================

def example_5_production_checklist():
    """프로덕션 체크리스트"""
    print("\n" + "=" * 70)
    print("📌 예제 5: 프로덕션 체크리스트")
    print("=" * 70)

    print("""
✅ 프로덕션 배포 체크리스트:

═══════════════════════════════════════════════════════════
🔒 보안 (Security)
═══════════════════════════════════════════════════════════
  ☐ API Key / JWT 인증 구현
  ☐ Rate Limiting 설정
  ☐ CORS 정책 적절히 제한
  ☐ HTTPS 적용 (SSL/TLS)
  ☐ 환경 변수로 비밀키 관리
  ☐ SQL Injection 방지
  ☐ XSS 방지
  ☐ CSRF 보호
  ☐ 입력 검증 (Pydantic)
  ☐ 의존성 보안 취약점 스캔

═══════════════════════════════════════════════════════════
⚡ 성능 (Performance)
═══════════════════════════════════════════════════════════
  ☐ 로드 테스트 수행
  ☐ 캐싱 전략 구현
  ☐ 데이터베이스 인덱스 최적화
  ☐ 비동기 처리 구현
  ☐ 정적 파일 CDN 사용
  ☐ 응답 압축 (gzip)
  ☐ Connection Pooling
  ☐ 불필요한 로깅 제거
  ☐ 메모리 누수 확인
  ☐ 타임아웃 설정

═══════════════════════════════════════════════════════════
🔍 관측성 (Observability)
═══════════════════════════════════════════════════════════
  ☐ 구조화된 로깅 (JSON)
  ☐ 로그 레벨 적절히 설정
  ☐ 헬스체크 엔드포인트
  ☐ 메트릭 수집 (Prometheus)
  ☐ 분산 트레이싱 (Jaeger)
  ☐ 에러 추적 (Sentry)
  ☐ LangSmith 트레이싱
  ☐ 대시보드 구축
  ☐ 알림 설정 (Slack/PagerDuty)
  ☐ APM 도구 (Datadog/New Relic)

═══════════════════════════════════════════════════════════
🧪 테스트 (Testing)
═══════════════════════════════════════════════════════════
  ☐ 유닛 테스트 (80%+ 커버리지)
  ☐ 통합 테스트
  ☐ E2E 테스트
  ☐ 로드 테스트
  ☐ 보안 테스트
  ☐ CI/CD 파이프라인
  ☐ 자동화된 테스트 실행
  ☐ 스테이징 환경 검증
  ☐ 카나리 배포
  ☐ 롤백 계획

═══════════════════════════════════════════════════════════
📦 배포 (Deployment)
═══════════════════════════════════════════════════════════
  ☐ Docker 이미지 최적화
  ☐ Kubernetes 매니페스트
  ☐ 자동 스케일링 (HPA)
  ☐ 로드 밸런서 설정
  ☐ 블루-그린 배포 or 카나리
  ☐ 헬스체크 및 Readiness Probe
  ☐ 리소스 제한 설정
  ☐ 환경별 설정 분리
  ☐ 시크릿 관리
  ☐ 백업 및 복구 계획

═══════════════════════════════════════════════════════════
📖 문서화 (Documentation)
═══════════════════════════════════════════════════════════
  ☐ API 문서 (Swagger/OpenAPI)
  ☐ README.md
  ☐ 아키텍처 다이어그램
  ☐ 배포 가이드
  ☐ 운영 매뉴얼
  ☐ 트러블슈팅 가이드
  ☐ 환경 변수 문서
  ☐ 의존성 목록
  ☐ 변경 이력 (CHANGELOG)
  ☐ 라이센스

═══════════════════════════════════════════════════════════
🛡️ 안정성 (Reliability)
═══════════════════════════════════════════════════════════
  ☐ 재시도 로직
  ☐ Circuit Breaker
  ☐ Graceful Shutdown
  ☐ 데이터 백업
  ☐ 재해 복구 계획
  ☐ 다중 AZ 배포
  ☐ 모니터링 알림
  ☐ SLA 정의
  ☐ 인시던트 대응 프로세스
  ☐ 정기 점검 계획

═══════════════════════════════════════════════════════════
💰 비용 (Cost)
═══════════════════════════════════════════════════════════
  ☐ 리소스 사용량 모니터링
  ☐ 불필요한 리소스 정리
  ☐ 예산 알림 설정
  ☐ 토큰 사용량 추적
  ☐ 캐싱으로 비용 절감
  ☐ 오토 스케일링 최적화
  ☐ Reserved Instances 고려
  ☐ Spot Instances 활용
  ☐ 비용 대시보드
  ☐ 정기 비용 리뷰

═══════════════════════════════════════════════════════════
👥 팀 협업 (Collaboration)
═══════════════════════════════════════════════════════════
  ☐ Git 브랜치 전략
  ☐ Code Review 프로세스
  ☐ 코드 스타일 가이드
  ☐ Pre-commit Hooks
  ☐ 이슈 트래킹
  ☐ 버전 관리 전략
  ☐ 배포 승인 프로세스
  ☐ On-call 로테이션
  ☐ 지식 공유 세션
  ☐ 포스트모템 문화
    """)

    print("\n🔹 프로덕션 배포 전 최종 체크:")
    print("-" * 70)

    # 간단한 헬스체크 시뮬레이션
    checks = {
        "환경 변수 설정": bool(os.getenv("OPENAI_API_KEY")),
        "Dependencies 설치": True,  # 실제로는 import 테스트
        "설정 파일 존재": os.path.exists(".env") if os.path.exists(".env") else False,
        "Debug 모드 off": os.getenv("DEBUG", "false").lower() == "false",
        "로그 레벨 적절": os.getenv("LOG_LEVEL", "INFO") in ["INFO", "WARNING", "ERROR"],
    }

    print("\n자동 체크 결과:")
    all_passed = True
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
        if not result:
            all_passed = False

    print("\n" + "-" * 70)

    if all_passed:
        print("\n🎉 모든 체크 통과! 배포 준비 완료")
        print("\n다음 단계:")
        print("  1. 스테이징 환경에 배포")
        print("  2. 스모크 테스트 실행")
        print("  3. 프로덕션 배포 (카나리 or 블루-그린)")
        print("  4. 모니터링 대시보드 확인")
        print("  5. 알림 채널 준비")
    else:
        print("\n⚠️  일부 체크 실패. 문제를 해결한 후 재시도하세요.")

    print("\n💡 프로덕션 운영 팁:")
    print("   • 작은 변경도 스테이징에서 먼저 테스트")
    print("   • 배포 시간대 고려 (트래픽 낮은 시간)")
    print("   • 롤백 계획 항상 준비")
    print("   • 모니터링 알림 즉시 대응")
    print("   • 정기적인 보안 패치")
    print("   • 인시던트 후 포스트모템 작성")
    print("   • 지속적인 성능 최적화")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🎓 LangChain AI Agent 마스터 교안")
    print("📖 Part 10: 배포와 관측성 - 배포")
    print("=" * 70 + "\n")

    # 예제 실행
    example_1_docker()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_2_api_server()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_3_scaling()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_4_config_management()
    input("\n⏎ 계속하려면 Enter를 누르세요...")

    example_5_production_checklist()

    # 마무리
    print("\n" + "=" * 70)
    print("🎉 Part 10-05: 배포를 완료했습니다!")
    print("=" * 70)
    print("\n💡 다음 단계:")
    print("  1. 06_observability.py - 관측성")
    print("\n📚 핵심 요약:")
    print("  • Docker로 애플리케이션 컨테이너화")
    print("  • FastAPI로 REST API 서버 구축")
    print("  • 수평 스케일링으로 트래픽 대응")
    print("  • 환경별 설정 분리 및 관리")
    print("  • 프로덕션 체크리스트 준수")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# 스크립트 실행
# ============================================================================

if __name__ == "__main__":
    main()


# ============================================================================
# 📚 추가 학습 포인트
# ============================================================================
#
# 1. 컨테이너 오케스트레이션:
#    - Kubernetes
#    - Docker Swarm
#    - AWS ECS/EKS
#    - Google GKE
#
# 2. CI/CD:
#    - GitHub Actions
#    - GitLab CI
#    - Jenkins
#    - ArgoCD (GitOps)
#
# 3. 클라우드 플랫폼:
#    - AWS (Lambda, ECS, API Gateway)
#    - Google Cloud (Cloud Run, GKE)
#    - Azure (Container Apps)
#    - Vercel, Fly.io
#
# 4. 서비스 메시:
#    - Istio
#    - Linkerd
#    - Consul
#
# 5. 배포 전략:
#    - Blue-Green Deployment
#    - Canary Deployment
#    - Rolling Update
#    - Feature Flags
#
# ============================================================================
