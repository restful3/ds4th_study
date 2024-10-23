# 주요 특징

## 환경 구성:

Python 3.10과 3.11 두 가지 버전의 가상환경이 제공되는 주피터랩입니다.
각 버전별로 독립된 가상환경과 requirements 파일 사용
Python 3.11을 기본 환경으로 설정됩니다.


## 라이브러리 구성

- RAG 프레임워크: langchain, llama-index, chromadb 등 최신 버전 설치
- 모델 개선 도구: peft, autotrain-advanced, trl 등 포함
- 문서 처리: unstructured, pdf2image, pytesseract 등 추가
- 벡터 저장소: pinecone, faiss, mongodb, redis 지원


## 버전 관리:

- Python 3.10: 안정성이 검증된 버전의 라이브러리 사용
- Python 3.11: 최신 버전의 라이브러리 사용
각 환경별로 호환성이 검증된 버전으로 구성


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
- docker-compose.yml에서 볼륨마운트할 경로를 변경하세요
```bashCopy
docker-compose up --build
```

3. JupyterLab에서 사용:

- 새 노트북 생성 시 Python 3.10 또는 3.11 커널 선택
- 각 환경에 맞는 라이브러리 버전이 자동으로 사용됨

필요한 수정사항이나 추가하고 싶은 라이브러리가 있다면 말씀해 주세요.