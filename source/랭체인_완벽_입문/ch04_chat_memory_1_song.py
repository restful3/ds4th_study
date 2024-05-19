import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

memory = ConversationBufferMemory(
    return_messages=True
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="저는 대화의 맥락을 고려해 답변을 할 수 있는 채팅 봇 입니다. 메시지를 입력해 주세요.").send()

@cl.on_message
async def on_message(input_message):
    message = input_message.content
    memory_message_result = memory.load_memory_variables({})
    messages = memory_message_result['history']

    messages.append(HumanMessage(content=message))

    result = chat(messages)
    memory.save_context(
        {
            "input":message,
        },
        {
            "output":result.content,
        }
    )
    await cl.Message(content=result.content).send()
