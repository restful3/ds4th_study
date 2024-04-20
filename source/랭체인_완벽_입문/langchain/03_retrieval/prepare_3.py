from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings  #← OpenAIEmbeddings 가져오기
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma  #← Chroma 가져오기

loader = PyMuPDFLoader("./sample.pdf")
documents = loader.load()

text_splitter = SpacyTextSplitter(
    chunk_size=300, 
    pipeline="ko_core_news_sm"
)
splitted_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings( #← OpenAIEmbeddings를 초기화
    model="text-embedding-ada-002" #← 모델명을 지정
)

database = Chroma(  #← Chroma를 초기화
    persist_directory="./.data",  #← 영속화 데이터 저장 위치 지정
    embedding_function=embeddings  #← 벡터화할 모델을 지정
)

database.add_documents(  #← 문서를 데이터베이스에 추가
    splitted_documents,  #← 추가할 문서 지정
)

print("데이터베이스 생성이 완료되었습니다.") #← 완료 알림
