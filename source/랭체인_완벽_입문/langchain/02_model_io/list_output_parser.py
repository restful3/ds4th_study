from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import \
    CommaSeparatedListOutputParser  #← Output Parser인 CommaSeparatedListOutputParser를 가져옵니다.
from langchain.schema import HumanMessage

output_parser = CommaSeparatedListOutputParser() #← CommaSeparatedListOutputParser 초기화

chat = ChatOpenAI(model="gpt-3.5-turbo", )

result = chat(
    [
        HumanMessage(content="애플이 개발한 대표적인 제품 3개를 알려주세요"),
        HumanMessage(content=output_parser.get_format_instructions()),  #← output_parser.get_format_instructions()를 실행하여 언어모델에 지시사항 추가하기
    ]
)

output = output_parser.parse(result.content) #← 출력 결과를 분석하여 목록 형식으로 변환한다.

for item in output: #← 목록을 하나씩 꺼내어 출력한다.
    print("대표 상품 => " + item)
