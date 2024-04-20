from langchain.callbacks.base import BaseCallbackHandler #← BaseCallbackHandler 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


class LogCallbackHandler(BaseCallbackHandler): #← Callback을 생성

    def on_chat_model_start(self, serialized, messages, **kwargs): #← Chat models 실행 시작 시 호출되는 처리를 정의
        print("Chat models 실행 시작....")
        print(f"입력: {messages}")

    def on_chain_start(self, serialized, inputs, **kwargs): #← Chain 실행 시작 시 호출되는 처리를 정의
        print("Chain 실행 시작....")
        print(f"입력: {inputs}")

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    callbacks=[ #← Chat models 초기화 시 Callback을 지정
        LogCallbackHandler() #← 생성한 LogCallbackHandler를 지정
    ]
)

result = chat([
    HumanMessage(content="안녕하세요!"),
])

print(result.content)
