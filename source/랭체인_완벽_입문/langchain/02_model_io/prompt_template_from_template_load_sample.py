from langchain.prompts import load_prompt

loaded_prompt = load_prompt("prompt.json") #← JSON에서 PromptTemplate를 로드

print(loaded_prompt.format(product="iPhone")) #← PromptTemplate를 사용해 문장 생성
