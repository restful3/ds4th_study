from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, validator

chat = ChatOpenAI()

class Smartphone(BaseModel): #← Pydantic의 모델을 정의한다.
    release_date: str = Field(description="스마트폰 출시일") #← Field를 사용해 설명을 추가
    screen_inches: float = Field(description="스마트폰의 화면 크기(인치)")
    os_installed: str = Field(description="스마트폰에 설치된 OS")
    model_name: str = Field(description="스마트폰 모델명")

    @validator("screen_inches") #← validator를 사용해 값을 검증
    def validate_screen_inches(cls, field): #← 검증할 필드와 값을 validator의 인수로 전달
        if field <= 0: #← screen_inches가 0 이하인 경우 에러를 반환
            raise ValueError("Screen inches must be a positive number")
        return field

parser = PydanticOutputParser(pydantic_object=Smartphone) #← PydanticOutputParser를 SmartPhone 모델로 초기화

result = chat([ #← Chat models에 HumanMessage를 전달해 문장을 생성
    HumanMessage(content="안드로이드 스마트폰 1개를 꼽아주세요"),
    HumanMessage(content=parser.get_format_instructions())
])

parsed_result = parser.parse(result.content) #← PydanticOutputParser를 사용해 문장을 파싱

print(f"모델명: {parsed_result.model_name}")
print(f"화면 크기: {parsed_result.screen_inches}인치")
print(f"OS: {parsed_result.os_installed}")
print(f"스마트폰 출시일: {parsed_result.release_date}")
