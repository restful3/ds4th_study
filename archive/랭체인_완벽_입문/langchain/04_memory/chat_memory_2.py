import chainlit as cl
from langchain.chains import ConversationChain  #← ConversationChain을 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationBufferMemory( 
    return_messages=True
)

chain = ConversationChain( #← ConversationChain을 초기화
    memory=memory,
    llm=chat,
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 메시지를 입력하세요.").send()

@cl.on_message
async def on_message(message: str):

    result = chain( #← ConversationChain을 사용해 언어 모델을 호출
        message #← 사용자 메시지를 인수로 지정
    )

    await cl.Message(content=result["response"]).send()
