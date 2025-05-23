import chainlit as cl
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory 
from langchain.memory import ConversationSummaryMemory 
from langchain.chains import ConversationChain 
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain 

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationSummaryMemory(  #← ConversationSummaryMemory를 사용하도록 변경
    llm=chat,  #← Chat models를 지정
    return_messages=True,
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
    messages = chain.memory.load_memory_variables({})["history"] # 저장된 메시지 가져오기

    print(f"저장된 메시지 개수: {len(messages)}" # 저장된 메시지 개수를 표시
          )

    for saved_message in messages: # 저장된 메시지를 1개씩 불러옴
        print(saved_message.content # 저장된 메시지를 표시
              )

    result = chain(message)

    await cl.Message(content=result["response"]).send()
