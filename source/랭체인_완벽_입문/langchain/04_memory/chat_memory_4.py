import os
import chainlit as cl
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

@cl.on_chat_start
async def on_chat_start():
    thread_id = None
    while not thread_id: #← 스레드 ID가 입력될 때까지 반복
        res = await cl.AskUserMessage(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 스레드 ID를 입력하세요.", timeout=600).send() #← AskUserMessage를 사용해 스레드 ID 입력
        if res:
            thread_id = res['content']

    history = RedisChatMessageHistory(  #← 새로 채팅이 시작될 때마다 초기화하도록 on_chat_start로 이동
        session_id=thread_id,  #← 스레드 ID를 세션 ID로 지정
        url=os.environ.get("REDIS_URL"),
    )

    memory = ConversationBufferMemory( #← 새로 채팅이 시작될 때마다 초기화하도록 on_chat_start로 이동
        return_messages=True,
        chat_memory=history,
    )

    chain = ConversationChain( #← 새로 채팅이 시작될 때마다 초기화하도록 on_chat_start로 이동
        memory=memory,
        llm=chat,
    )

    memory_message_result = chain.memory.load_memory_variables({}) #← 메모리 내용 가져오기

    messages = memory_message_result['history']

    for message in messages:
        if isinstance(message, HumanMessage): #← 사용자가 보낸 메시지인지 판단
            await cl.Message( #← 사용자 메시지이면 authorUser를 지정해 송신
                author="User",
                content=f"{message.content}",
            ).send()
        else:
            await cl.Message( #← AI의 메시지이면 ChatBot을 지정해 송신
                author="ChatBot",
                content=f"{message.content}",
            ).send()
    cl.user_session.set("chain", chain) #← 기록을 세션에 저장

@cl.on_message
async def on_message(message: str):
    chain = cl.user_session.get("chain") #← 세션에서 기록을 가져오기

    result = chain(message)

    await cl.Message(content=result["response"]).send()
