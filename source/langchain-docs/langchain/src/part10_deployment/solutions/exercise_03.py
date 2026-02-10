"""
================================================================================
LangChain AI Agent 마스터 교안
Part 10: Deployment - 실습 과제 3 해답
================================================================================

과제: 프로덕션 배포 (LangServe + Docker)
난이도: ⭐⭐⭐⭐☆ (고급)

요구사항:
1. LangServe로 API 서버 구축
2. Docker 컨테이너화
3. 프로덕션 배포 준비

학습 목표:
- LangServe 활용
- Dockerfile 작성
- 배포 best practices

================================================================================
"""

# ============================================================================
# LangServe API 서버 (server.py)
# ============================================================================

LANGSERVE_SERVER_CODE = '''
"""
LangServe API 서버
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ============================================================================
# Agent 정의
# ============================================================================

@tool
def calculate(expression: str) -> float:
    """수식을 계산합니다."""
    try:
        return eval(expression)
    except:
        return "계산 오류"

@tool
def get_info(topic: str) -> str:
    """정보를 제공합니다."""
    return f"{topic}에 대한 정보입니다. (데모)"

def create_agent():
    """Agent 생성"""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculate, get_info]
    return create_react_agent(model, tools)

# ============================================================================
# FastAPI 앱
# ============================================================================

app = FastAPI(
    title="LangChain Agent API",
    version="1.0",
    description="LangChain Agent REST API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangServe 라우트 추가
agent = create_agent()
add_routes(
    app,
    agent,
    path="/agent",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root
@app.get("/")
async def root():
    return {
        "message": "LangChain Agent API",
        "endpoints": {
            "agent": "/agent",
            "docs": "/docs",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

# ============================================================================
# Dockerfile
# ============================================================================

DOCKERFILE = '''
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# 포트 노출
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 실행
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# ============================================================================
# Docker Compose
# ============================================================================

DOCKER_COMPOSE = '''
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - agent-api
    restart: unless-stopped
'''

# ============================================================================
# Requirements.txt
# ============================================================================

REQUIREMENTS = '''
fastapi==0.109.0
uvicorn[standard]==0.27.0
langchain==0.1.0
langchain-openai==0.0.5
langgraph==0.0.20
langserve[all]==0.0.40
pydantic==2.5.3
python-dotenv==1.0.0
'''

# ============================================================================
# Nginx 설정
# ============================================================================

NGINX_CONF = '''
events {
    worker_connections 1024;
}

http {
    upstream agent_backend {
        server agent-api:8000;
    }

    server {
        listen 80;
        
        location / {
            proxy_pass http://agent_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            proxy_pass http://agent_backend/health;
            access_log off;
        }
    }
}
'''

# ============================================================================
# 배포 가이드
# ============================================================================

DEPLOYMENT_GUIDE = '''
# 프로덕션 배포 가이드

## 1. 파일 구조

project/
├── server.py           # LangServe 서버
├── Dockerfile          # Docker 이미지
├── docker-compose.yml  # Docker Compose 설정
├── requirements.txt    # Python 패키지
├── nginx.conf          # Nginx 설정
├── .env                # 환경 변수
└── logs/               # 로그 디렉토리

## 2. 환경 변수 설정

.env 파일:
```
OPENAI_API_KEY=your-api-key-here
LOG_LEVEL=info
```

## 3. 로컬 개발

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# 서버 실행
python server.py

# 테스트
curl http://localhost:8000/health
```

## 4. Docker 빌드 및 실행

```bash
# 이미지 빌드
docker build -t agent-api .

# 컨테이너 실행
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY agent-api

# Docker Compose로 전체 스택 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

## 5. API 사용 예제

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Agent 호출
response = requests.post(
    "http://localhost:8000/agent/invoke",
    json={
        "input": {
            "messages": [
                {"role": "user", "content": "2 + 2를 계산해줘"}
            ]
        }
    }
)
print(response.json())
```

## 6. 프로덕션 체크리스트

### 보안
- [ ] HTTPS 설정 (SSL 인증서)
- [ ] API 키 보안 (환경 변수, Secrets Manager)
- [ ] Rate limiting 구현
- [ ] CORS 정책 검토
- [ ] 인증/인가 추가

### 모니터링
- [ ] Health check 엔드포인트
- [ ] 로깅 (구조화된 로그)
- [ ] 메트릭 수집 (Prometheus)
- [ ] 알림 설정 (Slack, Email)
- [ ] APM 도구 (DataDog, New Relic)

### 성능
- [ ] 요청 제한 (Rate limiting)
- [ ] 캐싱 (Redis)
- [ ] 로드 밸런싱
- [ ] Auto-scaling 설정
- [ ] 리소스 제한 (CPU, Memory)

### 배포
- [ ] CI/CD 파이프라인 (GitHub Actions)
- [ ] 무중단 배포 (Blue-Green, Rolling)
- [ ] 롤백 계획
- [ ] 백업 전략
- [ ] 재해 복구 계획

### 테스트
- [ ] Unit 테스트
- [ ] Integration 테스트
- [ ] E2E 테스트
- [ ] Load 테스트
- [ ] Security 테스트

## 7. 클라우드 배포

### AWS
```bash
# ECR에 푸시
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin xxx.dkr.ecr.us-east-1.amazonaws.com
docker tag agent-api:latest xxx.dkr.ecr.us-east-1.amazonaws.com/agent-api:latest
docker push xxx.dkr.ecr.us-east-1.amazonaws.com/agent-api:latest

# ECS에 배포
aws ecs update-service --cluster my-cluster --service agent-api --force-new-deployment
```

### Google Cloud
```bash
# GCR에 푸시
docker tag agent-api gcr.io/my-project/agent-api
docker push gcr.io/my-project/agent-api

# Cloud Run에 배포
gcloud run deploy agent-api --image gcr.io/my-project/agent-api --platform managed
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    metadata:
      labels:
        app: agent-api
    spec:
      containers:
      - name: agent-api
        image: agent-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```
'''

# ============================================================================
# 데모 및 설명
# ============================================================================

def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🚀 Part 10: 프로덕션 배포 - 실습 과제 3 해답")
    print("=" * 70)
    
    print("\n📦 배포 파일 구조:")
    print("""
project/
├── server.py           # LangServe API 서버
├── Dockerfile          # Docker 이미지
├── docker-compose.yml  # Docker Compose
├── requirements.txt    # Python 패키지
├── nginx.conf          # Nginx 설정
└── .env                # 환경 변수
    """)
    
    print("\n" + "=" * 70)
    print("📄 주요 파일 내용")
    print("=" * 70)
    
    print("\n1. server.py (LangServe API):")
    print("-" * 70)
    print(LANGSERVE_SERVER_CODE[:500] + "...")
    
    print("\n\n2. Dockerfile:")
    print("-" * 70)
    print(DOCKERFILE)
    
    print("\n\n3. requirements.txt:")
    print("-" * 70)
    print(REQUIREMENTS)
    
    print("\n" + "=" * 70)
    print("🚀 배포 명령어")
    print("=" * 70)
    
    print("""
# 로컬 개발
python server.py

# Docker 빌드
docker build -t agent-api .

# Docker 실행
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY agent-api

# Docker Compose
docker-compose up -d

# 테스트
curl http://localhost:8000/health
curl http://localhost:8000/docs
    """)
    
    print("\n" + "=" * 70)
    print("💡 학습 포인트:")
    print("=" * 70)
    print("""
  1. LangServe로 REST API 서버 구축
  2. Docker 컨테이너화
  3. Docker Compose로 전체 스택 관리
  4. Nginx 리버스 프록시
  5. Health check 및 모니터링
  6. 프로덕션 배포 best practices
    """)
    
    print("\n" + "=" * 70)
    print("📚 전체 배포 가이드")
    print("=" * 70)
    print(DEPLOYMENT_GUIDE)

if __name__ == "__main__":
    main()
