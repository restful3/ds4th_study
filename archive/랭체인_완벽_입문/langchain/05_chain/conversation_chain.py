from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI()

memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain( #← ConversationChain을 초기화
    memory=memory, #← Memory 모듈을 지정
    llm=chat, #← 언어 모델을 지정
)
