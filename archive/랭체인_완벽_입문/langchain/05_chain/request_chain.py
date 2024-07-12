from langchain.chains import LLMChain, LLMRequestsChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

chat = ChatOpenAI()

prompt = PromptTemplate( #← PromptTemplate을 초기화
    input_variables=["query",
                     "requests_result"],
    template="""아래 문장을 바탕으로 질문에 답해 주세요.
문장: {requests_result}
질문: {query}""",
)

llm_chain = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True,
)

chain = LLMRequestsChain(  #← LLMRequestsChain을 초기화
    llm_chain=llm_chain,  #← llm_chain에 LLMChain을 지정
)

print(chain({
    "query": "도쿄의 날씨를 알려주세요",
    "url": "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json",
}))
