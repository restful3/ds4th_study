from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data", 
    embedding_function=embeddings
)

documents = database.similarity_search("비행 자동차의 최고 속도는?") #← 데이터베이스에서 유사도가 높은 문서를 가져옴
print(f"문서 개수: {len(documents)}") #← 문서 개수 표시

for document in documents:
    print(f"문서 내용: {document.page_content}") #← 문서 내용을 표시
