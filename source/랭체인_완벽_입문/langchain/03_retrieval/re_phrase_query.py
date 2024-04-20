from langchain.chat_models import ChatOpenAI
from langchain.retrievers import WikipediaRetriever, RePhraseQueryRetriever #← RePhraseQueryRetriever를 가져오기
from langchain import LLMChain
from langchain.prompts import PromptTemplate

retriever = WikipediaRetriever( 
    lang="ko", 
    doc_content_chars_max=500 
)

llm_chain = LLMChain( #← LLMChain을 초기화
    llm = ChatOpenAI( #← ChatOpenAI를 지정
        temperature = 0
    ), 
    prompt= PromptTemplate( #← PromptTemplate을 지정
        input_variables=["question"],
        template="""아래 질문에서 Wikipedia에서 검색할 키워드를 추출해 주세요.
질문: {question}
"""
))

re_phrase_query_retriever = RePhraseQueryRetriever( #← RePhraseQueryRetriever를 초기화
    llm_chain=llm_chain, #← LLMChain을 지정
    retriever=retriever, #← WikipediaRetriever를 지정
)

documents = re_phrase_query_retriever.get_relevant_documents("나는 라면을 좋아합니다. 그런데 소주란 무엇인가요?")

print(documents)
