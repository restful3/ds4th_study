import chainlit as cl
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI( 
    temperature=0,
    # model="gpt-3.5-turbo",
    model="gpt-4o",
    api_key = api_key
)

tools=load_tools(
    [
        "serpapi"        
    ],
    serpapi_api_key =serpapi_api_key
    # llm=chat
)

agent=initialize_agent(tools=tools, llm=chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Agent 초기화 완료").send()
    
@cl.on_message
async def on_message(input_message):
    result=agent.run(
        input_message.content,            # 객체 형태로 호출하기 때문에... (교재 코드 수정 부분)
        callbacks=[
            cl.LangchainCallbackHandler()
        ]
    )
    await cl.Message(content=result).send()
