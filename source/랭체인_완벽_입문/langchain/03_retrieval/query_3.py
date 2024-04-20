from langchain.chains import RetrievalQA  #← RetrievalQA를 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

chat = ChatOpenAI(model="gpt-3.5-turbo")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data", 
    embedding_function=embeddings
)

retriever = database.as_retriever() #← 데이터베이스를 Retriever로 변환

qa = RetrievalQA.from_llm(  #← RetrievalQA를 초기화
    llm=chat,  #← Chat models를 지정
    retriever=retriever,  #← Retriever를 지정
    return_source_documents=True  #← 응답에 원본 문서를 포함할지를 지정
)

result = qa("비행 자동차의 최고 속도를 알려주세요")

print(result["result"]) #← 응답을 표시

print(result["source_documents"]) #← 원본 문서를 표시
