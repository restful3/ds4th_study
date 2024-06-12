import os
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

@cl.on_chat_start
async def on_chat_start():
    thread_id = None
    while not thread_id:
        res = await cl.AskUserMessage(content="저는 대화의 맥락을 고려해 답변을 할 수 있는 채팅 봇 입니다. 스레드 ID를 입력하세요.", timeout=600).send()
        if res:
            print(res)
            thread_id = res['output']

    history = RedisChatMessageHistory(
        session_id=thread_id,
        url='redis://192.168.0.4:6379'
    )

    memory = ConversationBufferMemory(
        return_messages=True,
        chat_memory=history
    )

    chain = ConversationChain(
        memory = memory,
        llm = chat
    )

    memory_message_result = chain.memory.load_memory_variables({})
    
    messages = memory_message_result['history']

    for message in messages:
        if isinstance(message, HumanMessage):
            await cl.Message(
                author='User',
                content=f'{message.content}'
            ).send()
        else:
            await cl.Message(
                author='ChatBot',
                content=f'{message.content}'
            ).send()
    cl.user_session.set('chain', chain)


@cl.on_message
async def on_message(input_message):
    message = input_message.content
    chain = cl.user_session.get('chain')
    result = chain(message)
   
    await cl.Message(content=result['response']).send()
