# Part 4: Memory Systems - 대화 기억하기

> 📚 **학습 시간**: 약 3-4시간
> 🎯 **난이도**: ⭐⭐⭐⭐☆ (고급)
> 📖 **공식 문서**: [10-short-term-memory.md](/official/10-short-term-memory.md), [29-long-term-memory.md](/official/29-long-term-memory.md)
> 📄 **교안 문서**: [part04_memory.md](/docs/part04_memory.md)

---

## 📋 학습 목표

이 파트를 완료하면 다음을 할 수 있습니다:

- [x] Checkpointer를 사용한 단기 메모리 구현
- [x] PostgreSQL 기반 영구 메모리 구성
- [x] 메시지 관리 (trim, delete, summarization)
- [x] Custom State로 복잡한 상태 관리
- [x] Store를 사용한 장기 메모리 구현

---

## 📚 개요

Agent가 대화를 **기억**하게 만듭니다. 메모리는 실용적인 AI 시스템의 핵심입니다.

**왜 중요한가?**
- 사용자와의 자연스러운 대화 지속
- 컨텍스트를 유지하여 더 나은 응답 생성
- 사용자 선호도 및 이력 관리

**실무 활용 사례**
- 고객 지원 챗봇 (이전 대화 기억)
- 개인화된 AI Assistant
- 세션 기반 애플리케이션

---

## 📁 예제 파일

### 01_basic_memory.py
**난이도**: ⭐⭐☆☆☆ | **예상 시간**: 30분

InMemorySaver를 사용한 기본 메모리 시스템을 학습합니다.

**학습 내용**:
- Checkpointer 개념
- `InMemorySaver` 사용법
- Thread ID로 대화 세션 관리
- 메모리 읽기 및 쓰기

**실행 방법**:
```bash
python 01_basic_memory.py
```

**주요 개념**:
- Checkpointer = Agent 상태 저장소
- Thread = 독립적인 대화 세션
- 메모리 없이는 Agent가 이전 대화를 기억하지 못함

---

### 02_postgres_memory.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 45분

PostgreSQL을 사용한 영구 메모리를 구현합니다.

**학습 내용**:
- `PostgresSaver` 설정
- 데이터베이스 스키마
- 영구 저장소의 장점
- 프로덕션 환경 준비

**실행 방법**:
```bash
# PostgreSQL 실행 필요 (Docker 추천)
docker run -d --name langchain-postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:15

python 02_postgres_memory.py
```

**주요 개념**:
- 영구 메모리 vs 휘발성 메모리
- 데이터베이스 연결 관리
- 여러 사용자/세션 지원

---

### 03_message_trim.py
**난이도**: ⭐⭐⭐☆☆ | **예상 시간**: 40분

메시지 히스토리를 관리하여 토큰 한도를 넘지 않도록 합니다.

**학습 내용**:
- 메시지 Trim 전략
- 최근 N개 메시지만 유지
- 특정 메시지 삭제
- 토큰 카운팅

**실행 방법**:
```bash
python 03_message_trim.py
```

**주요 개념**:
- LLM의 컨텍스트 윈도우 제한
- 오래된 메시지 제거
- 중요한 메시지는 보존

---

### 04_summarization.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

긴 대화를 요약하여 컨텍스트를 압축합니다.

**학습 내용**:
- 대화 요약 전략
- 요약 후 원본 메시지 제거
- 점진적 요약 (rolling summarization)
- 중요 정보 보존

**실행 방법**:
```bash
python 04_summarization.py
```

**주요 개념**:
- 요약 = 정보 압축
- 토큰 절약 + 컨텍스트 유지
- Trade-off: 세부 정보 손실 가능

---

### 05_custom_state.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 50분

메시지 외에 추가 상태를 관리하는 방법을 학습합니다.

**학습 내용**:
- Custom State 정의
- Pydantic 모델로 상태 구조화
- Reducer 함수
- 복잡한 상태 업데이트 로직

**실행 방법**:
```bash
python 05_custom_state.py
```

**주요 개념**:
- State = Messages + Custom Fields
- 사용자 정보, 플래그, 카운터 등 저장
- 상태 기반 조건부 로직

---

### 06_long_term_store.py
**난이도**: ⭐⭐⭐⭐☆ | **예상 시간**: 60분

Store를 사용한 장기 메모리(사용자 프로필, 선호도 등)를 구현합니다.

**학습 내용**:
- Store vs Checkpointer 차이
- User Namespace
- Key-Value 저장소
- 사용자별 데이터 관리

**실행 방법**:
```bash
python 06_long_term_store.py
```

**주요 개념**:
- **Checkpointer**: 대화 히스토리 (단기)
- **Store**: 사용자 정보, 선호도 (장기)
- Namespace로 데이터 조직화

---

## 🎓 실습 과제

### 과제 1: 세션 기반 챗봇 (⭐⭐⭐)

**목표**: 여러 사용자의 독립적인 대화를 관리하는 챗봇을 만드세요.

**요구사항**:
1. 각 사용자별 Thread ID 생성
2. InMemorySaver로 메모리 관리
3. 사용자 A와 B의 대화가 섞이지 않도록

**해답**: [solutions/exercise_01.py](/src/part04_memory/solutions/exercise_01.py)

---

### 과제 2: 자동 요약 시스템 (⭐⭐⭐⭐)

**목표**: 대화가 길어지면 자동으로 요약하는 시스템을 구현하세요.

**요구사항**:
1. 메시지 개수가 10개를 넘으면 자동 요약
2. 요약본 + 최근 5개 메시지만 유지
3. 사용자에게 요약 사실 알림

**해답**: [solutions/exercise_02.py](/src/part04_memory/solutions/exercise_02.py)

---

### 과제 3: 사용자 프로필 시스템 (⭐⭐⭐⭐)

**목표**: 사용자의 선호도를 기억하는 AI Assistant를 만드세요.

**요구사항**:
1. Store로 사용자 프로필 저장
2. 이름, 선호 언어, 관심사 등
3. 대화 시 프로필 정보 활용

**예시**:
```
User: "내 이름은 김철수야"
Agent: "반갑습니다, 김철수님!"
[새 세션]
User: "안녕!"
Agent: "안녕하세요, 김철수님! 오늘은 무엇을 도와드릴까요?"
```

**해답**: [solutions/exercise_03.py](/src/part04_memory/solutions/exercise_03.py)

---

## 💡 실전 팁

### Tip 1: 메모리 전략 선택

```python
# 짧은 대화 (< 10 턴)
saver = InMemorySaver()

# 중간 길이 (10-50 턴) + 요약
saver = PostgresSaver(...)
# + summarization middleware

# 매우 긴 대화 (50+ 턴)
# → 요약 + 최근 N개 메시지 + Store로 핵심 정보 추출
```

### Tip 2: Thread ID 관리

```python
import uuid

# 사용자별 고유 ID
user_id = "user_123"
thread_id = f"{user_id}_session_{uuid.uuid4()}"

# 세션 재개
existing_thread_id = "user_123_session_abc..."
```

### Tip 3: 토큰 최적화

```python
from langchain_core.messages import trim_messages

# 최근 10개 메시지만 유지 + 시스템 메시지는 항상 포함
trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",
    include_system=True
)
```

---

## ❓ 자주 묻는 질문

<details>
<summary>Q1: InMemorySaver는 언제 사용하나요?</summary>

**A**:
- **개발/테스트**: 빠른 프로토타이핑
- **단일 서버**: 재시작 시 메모리 손실 허용
- **짧은 세션**: 영구 저장 불필요

**프로덕션에서는 PostgresSaver 권장!**
</details>

<details>
<summary>Q2: 메모리가 너무 커지면 어떻게 하나요?</summary>

**A**: 3가지 전략을 조합하세요:
1. **Trim**: 오래된 메시지 삭제
2. **Summarize**: 긴 대화 요약
3. **Store**: 핵심 정보만 장기 보관

```python
# 예시: 하이브리드 전략
if len(messages) > 20:
    # 요약 생성
    summary = summarize_conversation(messages[:-10])
    # 요약 + 최근 10개만 유지
    messages = [summary] + messages[-10:]
```
</details>

<details>
<summary>Q3: Thread ID는 어떻게 생성하나요?</summary>

**A**:
```python
# 방법 1: UUID (완전히 고유)
import uuid
thread_id = str(uuid.uuid4())

# 방법 2: 사용자 + 타임스탬프
import time
thread_id = f"user_{user_id}_{int(time.time())}"

# 방법 3: 사용자 정의
thread_id = "customer_support_20250206_001"
```
</details>

---

## 🔗 심화 학습

1. **공식 문서 심화**
   - [10-short-term-memory.md](/official/10-short-term-memory.md) - 단기 메모리
   - [29-long-term-memory.md](/official/29-long-term-memory.md) - 장기 메모리
   - [LangGraph Checkpointers](https://langchain-ai.github.io/langgraph/concepts/checkpointers/)

2. **관련 논문**
   - [MemPrompt: Memory-assisted Prompt Editing](https://arxiv.org/abs/2201.06009)
   - [Memorizing Transformers](https://arxiv.org/abs/2203.08913)

3. **커뮤니티 리소스**
   - [LangChain Memory Guide](https://python.langchain.com/docs/how_to/chatbots_memory/)
   - [프로덕션 메모리 패턴](https://blog.langchain.dev/memory-patterns/)

4. **다음 단계**
   - [Part 5: Middleware](/src/part05_middleware/README.md) - Agent 동작 제어

---

## ✅ 체크리스트

Part 4를 완료하기 전에 확인하세요:

- [ ] 모든 예제 코드를 실행해봤다 (6개)
- [ ] 실습 과제를 완료했다 (3개)
- [ ] Checkpointer의 역할을 이해했다
- [ ] Thread ID로 세션을 관리할 수 있다
- [ ] 메시지 Trim과 Summarization의 차이를 안다
- [ ] Custom State를 정의할 수 있다
- [ ] Store의 사용 사례를 이해했다

---

**이전**: [← Part 3 - First Agent](/src/part03_first_agent/README.md)
**다음**: [Part 5 - Middleware로 이동](/src/part05_middleware/README.md) →

---

**학습 진도**: ▓▓▓▓░░░░░░ 40% (Part 4/10 완료)

*마지막 업데이트: 2025-02-06*
