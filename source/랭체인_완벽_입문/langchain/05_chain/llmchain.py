from langchain import LLMChain, PromptTemplate  #← LLMChain 가져오기
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(  
    model="gpt-3.5-turbo",  
)

prompt = PromptTemplate(  
    template="{product}는 어느 회사에서 개발한 제품인가요?",  
    input_variables=[
        "product"  
    ]
)

chain = LLMChain( #← LLMChain을 생성
    llm=chat,
    prompt=prompt,
)

result = chain.predict(product="iPhone") #← LLMChain을 실행

print(result)
