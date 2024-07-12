from langchain.prompts import PromptTemplate

prompt = PromptTemplate(template="{product}는 어느 회사에서 개발한 제품인가요？", input_variables=["product"])
prompt_json = prompt.save("prompt.json") #← PromptTemplate를 JSON으로 변환
