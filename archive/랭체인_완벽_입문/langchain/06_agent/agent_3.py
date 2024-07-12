import random  #←임의의 숫자를 생성하기 위해 필요한 모듈을 가져오기
from langchain.agents import AgentType, Tool, initialize_agent  #←Tool을 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.tools import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = [] #← 다른 도구는 필요 없으므로 일단 삭제

tools.append(WriteFileTool( 
    root_dir="./"
))

def min_limit_random_number(min_number): #←최솟값을 지정할 수 있는 임의의 숫자를 생성하는 함수
    return random.randint(int(min_number), 100000)


tools.append(  #←도구를 추가
    Tool(
        name="Random",  #←도구 이름
        description="특정 최솟값 이상의 임의의 숫자를 생성할 수 있습니다.",  #←도구 설명
        func=min_limit_random_number  #←도구가 실행될 때 호출되는 함수
    )
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  
    verbose=True
)

result = agent.run("10 이상의 난수를 생성해 random.txt 파일에 저장하세요.")

print(f"실행 결과: {result}")
