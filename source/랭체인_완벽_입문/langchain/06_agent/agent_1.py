from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,  #← temperature를 0으로 설정해 출력의 다양성을 억제
    model="gpt-3.5-turbo"
)

tools = load_tools(  #← LangChain에 준비된 Tool을 로드
    [
        "requests",  #← 특정 URL의 결과를 얻는 Tool인 requests를 로드
    ]
)

agent = initialize_agent(  #← Agent를 초기화
    tools=tools,  #← Agent가 사용할 수 있는 Tool의 배열을 설정
    llm=chat,  #← Agent가 사용할 언어 모델을 지정
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  #←ReAct 방식으로 작동하게 설정
    verbose=True  #← 실행 중 로그를 표시
)

result = agent.run("""아래 URL에 접속해 도쿄의 날시를 검색해 한국어로 답하세요.
https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json
""")

print(f"실행 결과: {result}")
