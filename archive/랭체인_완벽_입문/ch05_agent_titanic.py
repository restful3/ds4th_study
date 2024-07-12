import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd



@cl.on_chat_start
async def on_chat_start():
    chat = ChatOpenAI(
        model="gpt-4-turbo"
    )    
    df = pd.read_csv('/home/restful3/datasets_bigpond/kaggle/titanic/train.csv', low_memory=False)
    agent = create_pandas_dataframe_agent(chat, df, verbose=True)
    await cl.Message(content="저는 타이타닉 데이터를 분석 할 수 있는 봇 입니다.").send()
    cl.user_session.set("agent ", agent)


@cl.on_message
async def on_message(input_message):
    message = input_message.content
    agent = cl.user_session.get("agent ")
    res = agent.run(message)
    await cl.Message(content=res).send()


# @cl.on_message
# async def on_message(input_message):
#     message = input_message.content
#     memory_message_result = memory.load_memory_variables({})
#     messages = memory_message_result['history']

#     messages.append(HumanMessage(content=message))

#     result = chat(messages)
#     memory.save_context(
#         {
#             "input":message,
#         },
#         {
#             "output":result.content,
#         }
#     )
#     await cl.Message(content=result.content).send()
