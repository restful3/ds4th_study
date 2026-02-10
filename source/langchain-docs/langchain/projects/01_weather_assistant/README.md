# Project 1: 날씨 비서 Agent

> ⭐⭐⭐ 난이도: 중급
> ⏱️ 예상 소요 시간: 2-3시간
> 📖 관련 파트: Part 3 (첫 번째 Agent)

---

## 📋 프로젝트 개요

실제 날씨 API를 사용하는 대화형 날씨 비서 Agent를 만들어봅니다.

### 학습 목표

- ✅ 외부 API 통합 (OpenWeatherMap)
- ✅ 실전 Agent 구축
- ✅ 에러 핸들링
- ✅ 사용자 친화적 대화 인터페이스

---

## 🎯 요구사항

### 기능 요구사항

1. **기본 날씨 조회**
   - 도시 이름으로 현재 날씨 조회
   - 온도, 습도, 날씨 상태 표시

2. **다양한 질문 처리**
   - "서울 날씨 어때?"
   - "내일 비 와?"
   - "뉴욕은 지금 몇 도야?"

3. **한국어 자연스러운 대화**
   - 친근한 톤
   - 맥락 이해
   - 적절한 조언 제공

4. **에러 처리**
   - 존재하지 않는 도시
   - API 오류
   - 네트워크 문제

### 기술 요구사항

- Python 3.10+
- LangChain 1.0
- OpenWeatherMap API (무료)
- python-dotenv

---

## 🚀 시작하기

### 1. API 키 발급

**OpenWeatherMap** (무료):
1. https://openweathermap.org/api 방문
2. 회원가입 (무료)
3. API Keys 섹션에서 키 복사

### 2. 환경 설정

```bash
# .env 파일에 추가
OPENAI_API_KEY=your-openai-key
OPENWEATHER_API_KEY=your-openweather-key
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 실행

```bash
python main.py
```

---

## 📖 구현 가이드

### Step 1: 날씨 도구 만들기

```python
@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 조회합니다."""
    import requests
    import os

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",  # 섭씨 온도
        "lang": "kr",       # 한국어
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"{city}: {desc}, {temp}°C"
    else:
        return f"'{city}'의 날씨 정보를 찾을 수 없습니다."
```

### Step 2: Agent 생성

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="""
당신은 친절한 날씨 비서입니다.
사용자에게 날씨 정보를 제공하고, 날씨에 맞는 조언을 해주세요.
    """
)
```

### Step 3: 대화 루프

```python
def chat():
    print("🌤️ 날씨 비서입니다. 무엇을 도와드릴까요?")

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in ["종료", "quit", "exit"]:
            print("👋 안녕히 가세요!")
            break

        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        print(f"🤖 Agent: {result['messages'][-1].content}")
```

---

## 🎨 예상 대화 예시

```
🌤️ 날씨 비서입니다. 무엇을 도와드릴까요?

👤 You: 서울 날씨 알려줘
🤖 Agent: 서울의 현재 날씨는 맑음이고 기온은 22°C입니다.
          산책하기 좋은 날씨네요!

👤 You: 뉴욕은 어때?
🤖 Agent: 뉴욕은 현재 흐림이고 15°C입니다.
          조금 쌀쌀하니 겉옷을 챙기시는 게 좋겠어요.

👤 You: 고마워!
🤖 Agent: 천만에요! 좋은 하루 되세요 😊
```

---

## 🧪 테스트 시나리오

### 기본 기능 테스트

- [ ] "서울 날씨는?" - 서울 날씨 조회
- [ ] "부산 날씨 알려줘" - 부산 날씨 조회
- [ ] "뉴욕은 몇 도야?" - 뉴욕 날씨 조회

### 에러 처리 테스트

- [ ] "아무도시 날씨는?" - 존재하지 않는 도시
- [ ] API 키 없이 실행 - 적절한 에러 메시지
- [ ] 네트워크 오프라인 - 에러 핸들링

### 대화 품질 테스트

- [ ] 자연스러운 대화 흐름
- [ ] 맥락 이해
- [ ] 적절한 조언 제공

---

## 🎓 학습 포인트

### 배울 수 있는 것

1. **외부 API 통합**
   - REST API 호출
   - API 키 관리
   - 응답 파싱

2. **Agent 개발 패턴**
   - 도구 정의
   - 에러 핸들링
   - System Prompt 작성

3. **사용자 경험**
   - 대화 인터페이스
   - 친근한 톤
   - 유용한 조언

---

## 🚧 도전 과제

### 추가 기능 구현 (선택)

1. **⭐ 5일 예보**
   ```python
   @tool
   def get_forecast(city: str) -> str:
       """5일 날씨 예보를 조회합니다."""
       # OpenWeatherMap의 5 day forecast API 사용
   ```

2. **⭐⭐ 날씨 비교**
   ```python
   @tool
   def compare_weather(city1: str, city2: str) -> str:
       """두 도시의 날씨를 비교합니다."""
   ```

3. **⭐⭐⭐ 날씨 알림**
   - 특정 조건 만족 시 알림
   - 예: 비 올 확률 70% 이상

---

## 📝 제출 방법

### 완료 체크리스트

- [ ] 모든 기본 기능 구현
- [ ] 에러 핸들링 완료
- [ ] 테스트 시나리오 통과
- [ ] README 작성
- [ ] 코드 주석 추가

### 제출 파일

```
projects/01_weather_assistant/
├── README.md (이 파일)
├── main.py (메인 코드)
├── tools.py (날씨 도구)
├── requirements.txt
└── solution/ (참고 솔루션)
```

---

## 🔗 참고 자료

- [OpenWeatherMap API 문서](https://openweathermap.org/api)
- [Part 3: 첫 번째 Agent](/docs/part03_first_agent.md)
- [Tools 가이드](/docs/part02_fundamentals.md#도구-tools)
- [Troubleshooting](/docs/appendix/troubleshooting.md)

---

## 💡 힌트

<details>
<summary>힌트 1: API 호출이 실패하면?</summary>

```python
try:
    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
except requests.exceptions.Timeout:
    return "날씨 서비스 응답이 없습니다. 잠시 후 다시 시도해주세요."
except requests.exceptions.HTTPError as e:
    return f"날씨 정보를 가져올 수 없습니다: {e}"
```
</details>

<details>
<summary>힌트 2: 한글 도시 이름 처리</summary>

```python
# 한글 → 영문 매핑
city_map = {
    "서울": "Seoul",
    "부산": "Busan",
    "뉴욕": "New York",
}

city_english = city_map.get(city, city)
```
</details>

<details>
<summary>힌트 3: 더 친근한 응답 만들기</summary>

System Prompt에 다음 추가:
```
- 온도에 따라 옷차림 조언
- 날씨에 맞는 활동 제안
- 이모지 사용 (☀️ 🌧️ ❄️)
```
</details>

---

**Good Luck! 🚀**

*프로젝트를 완료하면 Part 4 (메모리 시스템)로 진행하세요!*
