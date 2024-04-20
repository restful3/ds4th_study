from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever

chat = ChatOpenAI()

retriever = WikipediaRetriever(  #← WikipediaRetriever를 초기화
    lang="ko",  #← Wikipedia의 언어를 지정
    doc_content_chars_max=500,  #← 검색할 텍스트의 최대 글자수를 지정
    top_k_results=2,  #← 검색 결과 중 상위 몇 건을 가져올지 지정
)

chain = RetrievalQA.from_llm( #← RetrievalQA를 초기화
    llm=chat, #← 사용할 Chat models를 지정
    retriever=retriever, #← 사용할 Retriever를 지정
    return_source_documents=True, #← 정보를 가져온 원본 문서를 반환
)

result = chain("소주란?") #← RetrievalQA를 실행

source_documents = result["source_documents"] #← 정보 출처의 문서를 가져옴

print(f"검색 결과: {len(source_documents)}건") #← 검색 결과 건수를 표시
for document in source_documents:
    print("---------------검색한 메타데이터---------------")
    print(document.metadata)
    print("---------------검색한 텍스트---------------")
    print(document.page_content[:100])
print("---------------응답---------------")
print(result["result"]) #← 응답을 표시
