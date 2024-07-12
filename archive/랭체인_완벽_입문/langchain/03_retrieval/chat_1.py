import chainlit as cl


@cl.on_chat_start #← 채팅이 시작될 때 실행할 함수를 정의
async def on_chat_start():
    await cl.Message(content="준비되었습니다! 메시지를 입력하세요!").send() #← 초기에 표시할 메시지를 보냄

@cl.on_message #← 메시지를 보낼 때 실행할 함수를 정의
async def on_message(input_message):
    print("입력된 메시지: " + input_message)
    await cl.Message(content="안녕하세요!").send() #← 챗봇의 답변을 보냄
