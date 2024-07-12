from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    AIMessage
)

chat = ChatOpenAI(  
    model="gpt-3.5-turbo",  
)

result = chat([
    HumanMessage(content="계란찜을 만드는 재료를 알려주세요"),
    AIMessage( #← 이 언어모델에 AIMessage로 응답 추가
        content="""계란찜을 만드는 재료는 다음과 같습니다:

- 계란: 3개
- 물: 1/2컵
- 다진 양파: 1/4컵
- 다진 당근: 1/4컵
- 다진 대파: 1/4컵
- 소금: 약간
- 후추: 약간
- 참기름: 약간 (선택 사항)

위의 재료로 계란찜을 만들 수 있습니다. 하지만 재료 비율이나 추가할 수 있는 다른 재료는 개인의 취향에 따라 다를 수 있습니다."""),
    HumanMessage(content="위의 답변을 영어로 번역하세요")#← 메시지를 추가해 번역시킴
])
print(result.content)
