from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    streaming=True,  #← streaming을 True로 설정하여 스트리밍 모드로 실행
    callbacks=[
        StreamingStdOutCallbackHandler()  #← StreamingStdOutCallbackHandler를 콜백으로 설정
    ]
)
resp = chat([ #← 요청 보내기
    HumanMessage(content="맛있는 스테이크 굽는 법을 알려주세요")
])
