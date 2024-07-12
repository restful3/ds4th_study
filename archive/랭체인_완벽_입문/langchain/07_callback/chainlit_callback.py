import chainlit as cl
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,  
    model="gpt-3.5-turbo"
)

tools = load_tools( 
    [
        "serpapi",
    ]
)

agent = initialize_agent(tools=tools, llm=chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Agent 초기화 완료").send() 

@cl.on_message
async def on_message(input_message):
    result = agent.run( #← Agent를 실행
        input_message, #← 입력 메시지
        callbacks=[ #← 콜백을 지정
            cl.LangchainCallbackHandler() #← chainlit에 준비된 Callbacks를 지정
        ]
    )
    await cl.Message(content=result).send()
