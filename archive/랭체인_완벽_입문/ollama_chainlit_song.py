import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama

chat = Ollama(model="llama3")

memory = ConversationBufferMemory(
    return_messages=True
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
    result = chain(message)
   
    await cl.Message(content=result['response']).send()
