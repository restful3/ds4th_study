# CUDA 12.1 기반 이미지 사용
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업그레이드
RUN pip3 install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /workspace

# 필요한 파일을 작업 디렉토리에 복사
COPY requirements.txt .
COPY llm .

# PyTorch와 관련 라이브러리 설치 (CUDA 12.1 호환 버전)
RUN pip3 install -r requirements.txt

# JupyterLab 실행
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]