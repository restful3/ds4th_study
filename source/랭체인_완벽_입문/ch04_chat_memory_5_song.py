import os
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import RedisChatMessageHistory
from langchain.memory import ConversationSummaryMemory

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

history = RedisChatMessageHistory(
    session_id="chat_history",
    url=os.environ.get('REDIS_URL'),

)

memory = ConversationSummaryMemory(
    llm=chat,
    return_messages=True,
    chat_memory=history
)

chain = ConversationChain(
    memory = memory,
    llm = chat
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변을 할 수 있는 채팅 봇 입니다. 메시지를 입력해 주세요.").send()

@cl.on_message
async def on_message(input_message):
    message = input_message.content
    messages = chain.memory.load_memory_variables({})['history']
    print(f'저장된 메시지 개수 : {len(messages)}')
    for saved_message in messages:
        print(saved_message.content)

    result = chain(message)
   
    await cl.Message(content=result['response']).send()