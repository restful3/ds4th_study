# Part 1: Introduction - LangChain 시작하기

> 📚 **학습 시간**: 약 30분
> 🎯 **난이도**: ⭐☆☆☆☆ (입문)
> 📖 **공식 문서**: [01-overview.md](/official/01-overview.md), [02-install.md](/official/02-install.md)
> 📄 **교안 문서**: [part01_introduction.md](/docs/part01_introduction.md)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] LangChain의 기본 개념과 철학 이해
- [x] 개발 환경 설정 및 의존성 설치
- [x] 첫 번째 LangChain 애플리케이션 실행
- [x] 환경 변수 및 API 키 설정 확인

---

## 📚 개요

이 파트에서는 LangChain의 기초를 다지고 개발 환경을 설정합니다. AI Agent 개발을 위한 첫걸음입니다.

**왜 중요한가?**
- LangChain은 LLM 기반 애플리케이션 개발의 표준 프레임워크입니다
- 올바른 환경 설정은 원활한 학습의 기초입니다
- 공식 문서와 커뮤니티 리소스를 활용하는 방법을 배웁니다

---

## 📁 예제 파일

### 01_hello_langchain.py
**난이도**: ⭐☆☆☆☆ | **예상 시간**: 15분

LangChain의 핵심 기능을 5개 예제로 체험합니다.

**학습 내용**:
- 단순 LLM 호출과 그 한계
- `create_agent()`로 도구를 사용하는 Agent 만들기
- Agent vs 단순 LLM 호출 비교
- 다양한 LLM 프로바이더 전환
- System Prompt로 Agent 성격 바꾸기

**실행 방법**:
```bash
cd src/part01_introduction
python 01_hello_langchain.py
```

**주요 코드 개념**:
- `ChatOpenAI` 초기화 및 기본 메시지 전송
- `@tool` 데코레이터로 도구 정의
- `create_agent()`로 Agent 생성
- `system_prompt`로 Agent 행동 제어

---

### 02_environment_check.py
**난이도**: ⭐☆☆☆☆ | **예상 시간**: 15분

개발 환경이 올바르게 설정되었는지 확인합니다.

**학습 내용**:
- Python 버전 확인
- 필수 패키지 설치 확인
- API 키 설정 확인
- LLM 프로바이더 연결 테스트

**실행 방법**:
```bash
python 02_environment_check.py
```

**확인 항목**:
- ✅ Python 3.10 이상
- ✅ langchain, langchain-core 설치
- ✅ OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 설정
- ✅ LLM 프로바이더와의 연결

---

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# 프로젝트 루트에서 실행
cd source/langchain-docs/langchain
pip install -r src/requirements.txt
```

또는 `uv`를 사용하는 경우:
```bash
uv sync
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp src/.env.example src/.env

# .env 파일 편집
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. 환경 확인

```bash
python src/part01_introduction/02_environment_check.py
```

모든 체크가 ✅로 표시되면 준비 완료입니다!

---

## 💡 실전 팁

### Tip 1: API 키 관리
- `.env` 파일은 절대 Git에 커밋하지 마세요
- `.gitignore`에 `.env`가 포함되어 있는지 확인하세요
- 프로젝트별로 다른 API 키를 사용하는 것이 좋습니다

### Tip 2: LLM 프로바이더 선택
- **개발/학습**: OpenAI gpt-4.1-nano (최저가, 빠름)
- **프로덕션**: Anthropic Claude Sonnet 4.5 (정확, 안정적)
- **로컬 테스트**: Ollama (무료, 오프라인)

### Tip 3: 디버깅
- LangChain의 verbose 모드를 활성화하면 내부 동작을 볼 수 있습니다
- LangSmith를 사용하면 더 상세한 트레이싱이 가능합니다

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: "ModuleNotFoundError: No module named 'langchain'" 오류가 발생해요</summary>

**A**: 패키지가 설치되지 않았습니다. 다음을 실행하세요:
```bash
pip install langchain langchain-core langchain-openai
```
</details>

<details>
<summary>Q2: "OPENAI_API_KEY not found" 오류가 발생해요</summary>

**A**: 환경 변수가 설정되지 않았습니다:
1. `src/.env` 파일에 API 키를 추가하세요
2. 터미널에서 직접 설정하세요:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Python 코드에서 직접 설정하세요 (권장하지 않음):
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."
   ```
</details>

<details>
<summary>Q3: API 키는 어디서 발급받나요?</summary>

**A**:
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://aistudio.google.com/
</details>

---

## 🔗 관련 리소스

### 공식 문서
- [LangChain 개요](/official/01-overview.md) - LangChain 소개
- [설치 가이드](/official/02-install.md) - 상세 설치 방법
- [빠른 시작](/official/03-quickstart.md) - 첫 Agent 만들기

### 교안 문서
- [Part 1 상세 설명](/docs/part01_introduction.md) - 개념 설명 및 배경
- [환경 설정 가이드](/SETUP_GUIDE.md) - 완전한 설정 가이드
- [문제 해결](/docs/appendix/troubleshooting.md) - 일반적인 문제 해결

### 다음 단계
- [Part 2: Fundamentals](/src/part02_fundamentals/README.md) - LangChain 기본 요소
- [Part 3: First Agent](/src/part03_first_agent/README.md) - 첫 Agent 만들기

---

## ✅ 체크리스트

Part 1을 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다
- [ ] 환경 체크가 모두 통과한다
- [ ] API 키가 올바르게 설정되어 있다
- [ ] LangChain의 기본 개념을 이해했다
- [ ] `.env` 파일을 Git에 커밋하지 않도록 설정했다

---

**다음**: [Part 2 - Fundamentals로 이동](/src/part02_fundamentals/README.md) →

---

*마지막 업데이트: 2026-02-16*
