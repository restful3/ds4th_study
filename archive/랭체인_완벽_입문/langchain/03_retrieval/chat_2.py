import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

chat = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(template="""문장을 바탕으로 질문에 답하세요.

문장: 
{document}

질문: {query}
""", input_variables=["document", "query"])

database = Chroma(
    persist_directory="./.data", 
    embedding_function=embeddings
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="준비되었습니다! 메시지를 입력하세요!").send()

@cl.on_message
async def on_message(input_message):
    print("입력된 메시지: " + input_message)
    documents = database.similarity_search(input_message) #← input_message로 변경

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ---------------------------
    {document.page_content}
    """

    result = chat([
        HumanMessage(content=prompt.format(document=documents_string,
                                           query=input_message)) #← input_message로 변경
    ])
    await cl.Message(content=result.content).send() #← 챗봇의 답변을 보냄
