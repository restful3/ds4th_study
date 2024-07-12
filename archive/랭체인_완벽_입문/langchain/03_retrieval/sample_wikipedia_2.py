from langchain.retrievers import WikipediaRetriever

retriever = WikipediaRetriever( 
    lang="ko", 
    doc_content_chars_max=100,
    top_k_results=1
)
documents = retriever.get_relevant_documents( 
    "나는 라면을 좋아합니다. 그런데 소주란 무엇인가요?"
)
print(documents)
