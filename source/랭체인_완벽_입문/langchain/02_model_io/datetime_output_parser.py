from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import DatetimeOutputParser  #← Output Parser인 DatetimeOutputParser를 가져오기
from langchain.schema import HumanMessage

output_parser = DatetimeOutputParser() #← DatetimeOutputParser를 초기화

chat = ChatOpenAI(model="gpt-3.5-turbo", )

prompt = PromptTemplate.from_template("{product}의 출시일을 알려주세요") #← 출시일 물어보기

result = chat(
    [
        HumanMessage(content=prompt.format(product="iPhone8")),  #← iPhone8의 출시일 물어보기
        HumanMessage(content=output_parser.get_format_instructions()),  #← output_parser.get_format_instructions()를 실행하여 언어모델에 지시사항 추가하기
    ]
)

output = output_parser.parse(result.content) #← 출력 결과를 분석하여 날짜 및 시간 형식으로 변환

print(output)
