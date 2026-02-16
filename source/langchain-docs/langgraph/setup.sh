#!/bin/bash

# LangGraph 교안 환경 설정 스크립트
# 실행: bash setup.sh

set -e  # 에러 발생 시 중단

echo "=========================================="
echo "🚀 LangGraph 교안 환경 설정을 시작합니다"
echo "=========================================="
echo ""

# Python 버전 확인
echo "📌 Python 버전 확인..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   현재 Python 버전: $PYTHON_VERSION"

# Python 3.10 이상 필요
REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10 이상이 필요합니다"
    echo "   현재 버전: $PYTHON_VERSION"
    exit 1
fi
echo "   ✅ Python 버전 확인 완료"
echo ""

# 가상환경 생성
echo "📦 가상환경 생성..."
if [ -d "venv" ]; then
    echo "   ⚠️  기존 venv 폴더가 있습니다"
    read -p "   삭제하고 새로 만드시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "   ✅ 가상환경 재생성 완료"
    else
        echo "   ℹ️  기존 가상환경 사용"
    fi
else
    python3 -m venv venv
    echo "   ✅ 가상환경 생성 완료"
fi
echo ""

# 가상환경 활성화
echo "🔌 가상환경 활성화..."
source venv/bin/activate
echo "   ✅ 가상환경 활성화 완료"
echo ""

# pip 업그레이드
echo "⬆️  pip 업그레이드..."
pip install --upgrade pip --quiet
echo "   ✅ pip 업그레이드 완료"
echo ""

# 패키지 설치
echo "📚 패키지 설치 중..."
echo "   (시간이 걸릴 수 있습니다)"
pip install -r requirements.txt --quiet
echo "   ✅ 패키지 설치 완료"
echo ""

# .env 파일 생성
echo "🔑 환경 변수 파일 확인..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "   ✅ .env 파일이 생성되었습니다"
    echo "   ⚠️  .env 파일을 열어 API 키를 입력하세요"
else
    echo "   ℹ️  .env 파일이 이미 존재합니다"
fi
echo ""

# graphviz 확인 (선택사항)
echo "🎨 graphviz 확인 (그래프 시각화용)..."
if command -v dot &> /dev/null; then
    echo "   ✅ graphviz가 설치되어 있습니다"
else
    echo "   ⚠️  graphviz가 설치되어 있지 않습니다"
    echo "   그래프 시각화를 사용하려면 다음 명령으로 설치하세요:"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   macOS:   brew install graphviz"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "   Ubuntu:  sudo apt-get install graphviz"
    else
        echo "   Windows: https://graphviz.org/download/"
    fi
fi
echo ""

# 환경 검증
echo "✅ 환경 검증..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from utils.env_check import check_all_llm_providers
print()
check_all_llm_providers()
"

echo "=========================================="
echo "🎉 설정 완료!"
echo "=========================================="
echo ""
echo "다음 명령으로 가상환경을 활성화하세요:"
echo "  source venv/bin/activate"
echo ""
echo "예제 실행:"
echo "  python src/part1_foundation/01_hello_langgraph.py"
echo ""
echo "⚠️  주의: .env 파일에 API 키를 먼저 설정해야 합니다!"
echo ""
