# Project 1: 날씨 비서 Agent - 참고 솔루션

> ⚠️ **주의**: 이 솔루션은 참고용입니다. 먼저 스스로 구현해보세요!

---

## ✅ 완성된 기능

### 1. 기본 기능
- ✅ OpenWeatherMap API 통합
- ✅ 한글/영문 도시명 지원
- ✅ 실시간 날씨 조회
- ✅ 두 도시 날씨 비교

### 2. 에러 처리
- ✅ API 키 누락 처리
- ✅ 존재하지 않는 도시 처리
- ✅ 네트워크 오류 처리
- ✅ 타임아웃 처리

### 3. 사용자 경험
- ✅ 친근한 대화 인터페이스
- ✅ 이모지 사용
- ✅ 날씨별 조언 제공
- ✅ 명확한 오류 메시지

---

## 🎯 구현 완료 체크리스트

- [x] `tools.py` - 날씨 도구 구현
- [x] `main.py` - 메인 프로그램
- [x] `requirements.txt` - 의존성 정의
- [x] `tests/test_tools.py` - 단위 테스트
- [x] 한글 도시명 매핑
- [x] 에러 핸들링
- [x] 대화 루프 구현
- [x] System Prompt 최적화

---

## 📊 테스트 결과

### 기본 기능 테스트
```bash
pytest tests/test_tools.py -v
```

예상 결과:
```
tests/test_tools.py::test_city_name_map PASSED
tests/test_tools.py::test_check_api_key_present PASSED
tests/test_tools.py::test_get_weather_data_success PASSED
tests/test_tools.py::test_format_weather_response PASSED
tests/test_tools.py::test_get_weather_tool_success PASSED
```

---

## 🎨 실행 예시

```bash
$ python main.py

======================================================================
🌤️  날씨 비서 Agent에 오신 것을 환영합니다!
======================================================================

📍 전 세계 도시의 날씨 정보를 알려드립니다.

💬 예시 질문:
   • 서울 날씨 어때?
   • 뉴욕은 지금 몇 도야?
   • 부산이랑 대구 날씨 비교해줘

⌨️  '종료', 'quit', 'exit'를 입력하면 프로그램이 종료됩니다.
======================================================================

👤 You: 서울 날씨 알려줘

🤖 Agent: ☀️ 서울 날씨 정보:
━━━━━━━━━━━━━━━━━━━━━━
🌡️ 기온: 22°C (체감 21°C)
💧 습도: 65%
🌈 날씨: 맑음

날씨가 정말 좋네요! 산책하기 딱 좋은 날씨입니다. 😊

👤 You: 뉴욕이랑 비교해줘

🤖 Agent: 📊 서울 vs 뉴욕 날씨 비교:
━━━━━━━━━━━━━━━━━━━━━━
서울: 22°C, 맑음
뉴욕: 15°C, 구름조금

🌡️ 온도 차이: 7°C
더 따뜻한 곳: 서울

서울이 뉴욕보다 7도 더 따뜻하네요! 🌡️

👤 You: 종료

👋 안녕히 가세요!
```

---

## 💡 핵심 구현 포인트

### 1. 도구 정의 (`tools.py`)

```python
@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 조회합니다."""
    data = get_weather_data(city)

    if data is None:
        return f"❌ '{city}'의 날씨 정보를 가져올 수 없습니다."

    return format_weather_response(data, city)
```

**포인트:**
- `@tool` 데코레이터로 LangChain 도구로 변환
- 명확한 docstring (Agent가 읽음)
- 에러 처리 포함

### 2. Agent 생성 (`main.py`)

```python
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    tools=[get_weather, compare_weather],
    system_prompt=SYSTEM_PROMPT,
)
```

**포인트:**
- `gpt-4o-mini`: 비용 효율적
- `temperature=0.7`: 약간의 창의성
- 상세한 System Prompt

### 3. System Prompt

```python
SYSTEM_PROMPT = """
당신은 친절하고 유용한 날씨 비서입니다. 😊

**조언 가이드:**
- 15°C 이하: 겉옷을 챙기라고 조언
- 25°C 이상: 시원한 옷차림 추천
- 비 예상: 우산 챙기기 권유
"""
```

**포인트:**
- 구체적인 역할 정의
- 조언 가이드라인
- 친근한 톤 지시

---

## 🚀 도전 과제 구현 가이드

### 도전 과제 1: 5일 예보

```python
@tool
def get_forecast(city: str) -> str:
    """5일 날씨 예보를 조회합니다."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "http://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
        "cnt": 40  # 5일 * 8 (3시간 간격)
    }

    response = requests.get(url, params=params)
    data = response.json()

    # 일별로 그룹화하여 포맷팅
    # ...
```

### 도전 과제 2: GUI 인터페이스 (Streamlit)

```python
import streamlit as st

st.title("🌤️ 날씨 비서 Agent")

city = st.text_input("도시 이름을 입력하세요:")

if st.button("날씨 조회"):
    with st.spinner("날씨 정보를 가져오는 중..."):
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"{city} 날씨 알려줘"}]
        })
        st.success(result["messages"][-1].content)
```

---

## 🐛 자주 발생하는 문제

### 문제 1: "OPENWEATHER_API_KEY not found"

**해결:**
```bash
# .env 파일에 추가
OPENWEATHER_API_KEY=your-api-key-here

# 또는 환경변수로 설정
export OPENWEATHER_API_KEY=your-api-key-here
```

### 문제 2: 도시를 찾을 수 없음

**원인:** 도시 이름 오타 또는 영문 표기 문제

**해결:**
```python
# CITY_NAME_MAP에 추가
CITY_NAME_MAP = {
    "서울": "Seoul",
    "새로운도시": "New City Name",
}
```

### 문제 3: Agent가 도구를 호출하지 않음

**원인:** System Prompt 또는 도구 docstring이 불명확

**해결:**
- 도구 docstring을 더 명확하게 작성
- System Prompt에 도구 사용 지시 추가
- 질문을 더 구체적으로 변경

---

## 📈 성능 최적화

### 1. API 응답 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_weather_data_cached(city: str):
    # 5분 동안 캐시
    return get_weather_data(city)
```

### 2. 비동기 처리

```python
import asyncio
import aiohttp

async def get_weather_async(city: str):
    # 비동기로 여러 도시 동시 조회
    pass
```

---

## 🎓 학습 요약

### 배운 것들
1. ✅ LangChain 도구 정의 (`@tool`)
2. ✅ Agent 생성 (`create_agent`)
3. ✅ System Prompt 작성
4. ✅ 외부 API 통합
5. ✅ 에러 핸들링
6. ✅ 대화 인터페이스 구현

### 다음 단계
- **Part 4**: 메모리 추가 (대화 기억하기)
- **Part 5**: 미들웨어 (로깅, 모니터링)
- **Project 2**: 문서 Q&A Agent (RAG)

---

## 🔗 참고 자료

- [OpenWeatherMap API 문서](https://openweathermap.org/api)
- [LangChain Tools 가이드](https://python.langchain.com/docs/modules/agents/tools/)
- [Part 3: 첫 번째 Agent](/docs/part03_first_agent.md)

---

**축하합니다! 🎉 첫 번째 프로젝트를 완료했습니다!**

*다음 프로젝트: [Project 2 - 문서 Q&A Agent](/projects/02_document_qa/)*
