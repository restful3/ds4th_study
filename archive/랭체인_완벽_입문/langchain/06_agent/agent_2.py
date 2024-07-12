from langchain.agents import AgentType, initialize_agent, load_tools  #←load_tools 가져오기를 추가
from langchain.chat_models import ChatOpenAI
from langchain.tools.file_management import WriteFileTool  #←파일 쓰기를 할 수 있는 Tool을 가져오기

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = load_tools(
    [
        "requests_get",
        "serpapi" #←serpapi를 추가
    ],
    llm=chat
)

tools.append(WriteFileTool( #←파일 쓰기를 할 수 있는 Tool을 추가
    root_dir="./"
))

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  #←Agent 유형을 변경
    verbose=True
)

result = agent.run("경주시의 특산물을 검색해 result.txt 파일에 한국어로 저장하세요.") #←실행 결과를 파일에 저장하도록 지시

print(f"실행 결과: {result}")
