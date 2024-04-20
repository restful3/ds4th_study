import os
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

chat = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate(template="""문장을 기반으로 질문에 답하세요. 

문장: 
{document}

질문: {query}
""", input_variables=["document", "query"])

text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ko_core_news_sm")

@cl.on_chat_start
async def on_chat_start():
    files = None #← 파일이 선택되어 있는지 확인하는 변수

    while files is None: #← 파일이 선택될 때까지 반복
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDF를 선택해 주세요",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files[0]

    if not os.path.exists("tmp"): #← tmp 디렉터리가 존재하는지 확인
        os.mkdir("tmp") #← 존재하지 않으면 생성
    with open(f"tmp/{file.name}", "wb") as f: #← PDF 파일을 저장
        f.write(file.content) #← 파일 내용을 작성

    documents = PyMuPDFLoader(f"tmp/{file.name}").load() #← 저장한 PDF 파일을 로드
    splitted_documents = text_splitter.split_documents(documents) #← 문서를 분할

    database = Chroma( #← 데이터베이스 초기화
        embedding_function=embeddings,
        # 이번에는 persist_directory를 지정하지 않음으로써 데이터베이스 영속화를 하지 않음
    )

    database.add_documents(splitted_documents) #← 문서를 데이터베이스에 추가

    cl.user_session.set(  #← 데이터베이스를 세션에 저장
        "database",  #← 세션에 저장할 이름
        database  #← 세션에 저장할 값
    )

    await cl.Message(content=f"`{file.name}` 로딩이 완료되었습니다. 질문을 입력하세요.").send() #← 불러오기 완료를 알림

@cl.on_message
async def on_message(input_message):
    print("입력된 메시지: " + input_message)

    database = cl.user_session.get("database") #← 세션에서 데이터베이스를 가져옴

    documents = database.similarity_search(input_message)

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
    await cl.Message(content=result.content).send()
