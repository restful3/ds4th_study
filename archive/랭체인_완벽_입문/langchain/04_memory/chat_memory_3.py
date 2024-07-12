import os  #← 환경변수를 얻기 위해 os를 가져오기
import chainlit as cl
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import RedisChatMessageHistory  #← RedisChatMessageHistory를 추가
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

history = RedisChatMessageHistory(  #← RedisChatMessageHistory를 초기화
    session_id="chat_history",
    url=os.environ.get("REDIS_URL"),  #← 환경변수에서 Redis의 URL을 가져오기
)

memory = ConversationBufferMemory(
    return_messages=True,
    chat_memory=history,  #← 채팅 기록을 지정
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 메시지를 입력하세요.").send()

@cl.on_message
async def on_message(message: str):

    result = chain(message)

    await cl.Message(content=result["response"]).send()
