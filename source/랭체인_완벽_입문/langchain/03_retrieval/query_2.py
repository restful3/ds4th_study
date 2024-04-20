from langchain.chat_models import ChatOpenAI  #← ChatOpenAI 가져오기
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate  #← PromptTemplate 가져오기
from langchain.schema import HumanMessage  #← HumanMessage 가져오기
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

database = Chroma(
    persist_directory="./.data", 
    embedding_function=embeddings
)

query = "비행 자동차의 최고 속도는?"

documents = database.similarity_search(query)

documents_string = "" #← 문서 내용을 저장할 변수를 초기화

for document in documents:
    documents_string += f"""
---------------------------
{document.page_content}
""" #← 문서 내용을 추가

prompt = PromptTemplate( #← PromptTemplate를 초기화
    template="""문장을 바탕으로 질문에 답하세요.

문장: 
{document}

질문: {query}
""",
    input_variables=["document","query"] #← 입력 변수를 지정
)

chat = ChatOpenAI( #← ChatOpenAI를 초기화
    model="gpt-3.5-turbo"
)

result = chat([
    HumanMessage(content=prompt.format(document=documents_string, query=query))
])

print(result.content)
