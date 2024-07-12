from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory  #←ConversationBufferMemory 가져오기
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo"
)

tools = []

# WriteFileTool을 제거

retriever = WikipediaRetriever(
    lang="ko",
    doc_content_chars_max=500,
    top_k_results=1
)

tools.append(
    create_retriever_tool(  #←Retrievers를 사용하는 Tool을 생성
        name="WikipediaRetriever",  #←Tool 이름
        description="받은 단어에 대한 Wikipedia 기사를 검색할 수 있다",  #←Tool 설명
        retriever=retriever,  #←Retrievers를 지정
    )
)

memory = ConversationBufferMemory(  #←ConversationBufferMemory를 초기화
    memory_key="chat_history",  #←메모리 키를 설정
    return_messages=True  #←메시지를 반환하도록 설정
)

agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  #←Agent의 유형을 대화형으로 변경
    memory=memory,  #←Memory를 지정
    verbose=True
)

result = agent.run("스카치 위스키에 대해 Wikipedia에서 찾아보고 그 개요를 한국어로 개요를 정리하세요.") #←Wikipedia에서 찾아보라고 지시
print(f"1차 실행 결과: {result}") #←실행 결과를 표시
result_2 = agent.run("이전 지시를 다시 한번 실행하세요.") #←이전 지시를 다시 실행하도록 지시
print(f"2차 실행 결과: {result_2}") #←실행 결과를 표시
