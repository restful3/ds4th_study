from langchain.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(  #← WikipediaRetriever를 초기화
    lang="ko",  #← Wikipedia의 언어를 지정
)
documents = retriever.get_relevant_documents( #← Wikipedia에서 관련 문서를 가져옴
    "대형 언어 모델" #← 검색할 키워드를 지정
)

print(f"검색 결과: {len(documents)}건") #← 검색 결과 건수를 표시

for document in documents:
    print("---------------검색한 메타데이터---------------")
    print(document.metadata) #← 메타데이터를 표시
    print("---------------검색한 텍스트---------------")
    print(document.page_content[:100]) #← 텍스트의 첫 100글자를 표시
