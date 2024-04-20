from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool  #← create_retriever_tool을 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever #←WikipediaRetriever를 가져오기
from langchain.tools import WriteFileTool

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = [] 

tools.append(WriteFileTool( 
    root_dir="./"
))

retriever = WikipediaRetriever( #←WikipediaRetriever를 초기화
    lang="ko", #←언어를 한국어로 설정
    doc_content_chars_max=500,  #←글의 최대 글자 수를 500자로 설정
    top_k_results=1 #←검색 결과 중 상위 1건을 가져옴
)

tools.append(
    create_retriever_tool(  #←Retrievers를 사용하는 Tool을 생성
        name="WikipediaRetriever",  #←Tool 이름
        description="받은 단어에 대한 Wikipedia 기사를 검색할 수 있다",  #←Tool 설명
        retriever=retriever,  #←Retrievers를 지정
    )
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  
    verbose=True
)

result = agent.run("스카치 위스키에 대해 Wikipedia에서 찾아보고 그 개요를 한국어로 result.txt 파일에 저장하세요.")

print(f"실행 결과: {result}")
