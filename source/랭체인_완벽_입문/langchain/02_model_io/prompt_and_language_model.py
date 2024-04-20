from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(  #← 클라이언트 생성 및 chat에 저장
    model="gpt-3.5-turbo",  #← 호출할 모델 지정
)

prompt = PromptTemplate(  #← PromptTemplate을 작성
    template="{product}는 어느 회사에서 개발한 제품인가요？",  #← {product}라는 변수를 포함하는 프롬프트 작성하기
    input_variables=[
        "product"  #← product에 입력할 변수 지정
    ]
)

result = chat( #← 실행
    [
        HumanMessage(content=prompt.format(product="아이폰")),
    ]
)
print(result.content)
