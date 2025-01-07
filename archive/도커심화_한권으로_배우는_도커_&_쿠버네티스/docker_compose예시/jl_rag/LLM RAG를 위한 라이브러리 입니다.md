# 주요 특징
- llm으로 rag를 만들기 위한 도커컴포즈입니다.

## 라이브러리 구성

- RAG 프레임워크: langchain, llama-index, chromadb 등 최신 버전 설치
- 모델 개선 도구: peft, trl 등 포함
- 문서 처리: unstructured, pdf2image, pytesseract 등 추가
- 벡터 저장소: pinecone, faiss, mongodb, redis 지원

## 시스템 설정:

PDF, 이미지 처리를 위한 시스템 라이브러리 설치
Hugging Face 캐시 볼륨 마운트로 모델 재다운로드 방지

## 사용 방법:

1. 파일 준비:

```bashCopy 
# requirements 파일 생성
requirements-py310.txt
requirements-py311.txt
```

2. 도커 빌드 및 실행:
```bashCopy
docker-compose up --build
```

3. JupyterLab에서 사용:

- 새 노트북 생성 시 Python 3.10 커널 선택
- 각 환경에 맞는 라이브러리 버전이 자동으로 사용됨

