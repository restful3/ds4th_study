from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {
        "input": "충청도의 계룡산 전라도의 내장산 강원도의 설악산은 모두 국립 공원이다",  #← 입력 예
        "output": "충청도의 계룡산, 전라도의 내장산, 강원도의 설악산은 모두 국립 공원이다."  #← 출력 예
    }
]

prompt = PromptTemplate(  #← PromptTemplate 준비
    input_variables=["input", "output"],  #← input과 output을 입력 변수로 설정
    template="입력: {input}\n출력: {output}",  #← 템플릿
)

few_shot_prompt = FewShotPromptTemplate(  #← FewShotPromptTemplate 준비
    examples=examples,  #← 입력 예와 출력 예를 정의
    example_prompt=prompt,  #← FewShotPromptTemplate에 PromptTemplate를 전달
    prefix="아래 문장부호가 빠진 입력에 문장부호를 추가하세요. 추가할 수 있는 문장부호는 ',', '.'입니다. 다른 문장부호는 추가하지 마세요.",  #← 지시어 추가하기
    suffix="입력: {input_string}\n출력:",  #← 출력 예의 입력 변수를 정의
    input_variables=["input_string"],  #← FewShotPromptTemplate의 입력 변수를 설정
)
llm = OpenAI()
formatted_prompt = few_shot_prompt.format( #← FewShotPromptTemplate을 사용하여 프롬프트 작성
    input_string="집을 보러 가면 그 집이 내가 원하는 조건에 맞는지 살기에 편한지 망가진 곳은 없는지 확인해야 한다"
)
result = llm.predict(formatted_prompt)
print("formatted_prompt: ", formatted_prompt)
print("result: ", result)
