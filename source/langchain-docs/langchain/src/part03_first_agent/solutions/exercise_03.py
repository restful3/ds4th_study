"""
================================================================================
Part 3: First Agent - 실습 과제 3 해답
멀티스텝 Agent (⭐⭐⭐⭐)
================================================================================
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 조회합니다."""
    weather_data = {
        "서울": {"temp": 15, "condition": "맑음"},
        "부산": {"temp": 18, "condition": "흐림"},
        "대구": {"temp": 16, "condition": "비"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "알 수 없음"})
    return f"{city}: {data['temp']}°C, {data['condition']}"

@tool
def compare_temperatures(city1_data: str, city2_data: str) -> str:
    """두 도시의 온도를 비교합니다."""
    # 온도 추출 (간단한 파싱)
    import re
    temps = re.findall(r'(\d+)°C', city1_data + " " + city2_data)
    if len(temps) >= 2:
        if int(temps[0]) > int(temps[1]):
            return f"첫 번째 도시가 {int(temps[0]) - int(temps[1])}도 더 따뜻합니다."
        else:
            return f"두 번째 도시가 {int(temps[1]) - int(temps[0])}도 더 따뜻합니다."
    return "비교할 수 없습니다."

@tool
def recommend_city(comparison_result: str) -> str:
    """따뜻한 도시를 추천합니다."""
    if "첫 번째" in comparison_result:
        return "첫 번째 도시를 추천합니다. ☀️"
    elif "두 번째" in comparison_result:
        return "두 번째 도시를 추천합니다. ☀️"
    return "두 도시 모두 좋습니다!"

def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, compare_temperatures, recommend_city]

    system_prompt = """당신은 여행 추천 Agent입니다.

단계:
1. 두 도시의 날씨를 각각 조회
2. 온도를 비교
3. 더 따뜻한 곳을 추천

각 단계를 명확히 실행하세요."""

    agent = create_react_agent(model, tools, state_modifier=system_prompt)

    question = "서울과 부산의 날씨를 비교하고, 더 따뜻한 곳을 추천해줘"
    print(f"👤 질문: {question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\n🤖 최종 응답:\n{result['messages'][-1].content}")

if __name__ == "__main__":
    main()
