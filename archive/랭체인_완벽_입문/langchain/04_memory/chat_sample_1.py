from langchain.chat_models import ChatOpenAI  
from langchain.schema import HumanMessage  

chat = ChatOpenAI(  
    model="gpt-3.5-turbo",  
)

result = chat( 
    [
        HumanMessage(content="계란찜을 만드는 재료를 알려주세요"),
    ]
)
print(result.content)
