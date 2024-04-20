import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory  #← ConversationBufferMemory 가져오기
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationBufferMemory( #← 메모리 초기화
    return_messages=True
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변할 수 있는 채팅봇입니다. 메시지를 입력하세요.").send()

@cl.on_message
async def on_message(message: str):
    memory_message_result = memory.load_memory_variables({}) #← 메모리 내용을 로드

    messages = memory_message_result['history'] #← 메모리 내용에서 메시지만 얻음

    messages.append(HumanMessage(content=message)) #← 사용자의 메시지를 추가

    result = chat( #← Chat models를 사용해 언어 모델을 호출
        messages
    )

    memory.save_context(  #← 메모리에 메시지를 추가
        {
            "input": message,  #← 사용자의 메시지를 input으로 저장
        },
        {
            "output": result.content,  #← AI의 메시지를 output으로 저장
        }
    )
    await cl.Message(content=result.content).send() #← AI의 메시지를 송신
