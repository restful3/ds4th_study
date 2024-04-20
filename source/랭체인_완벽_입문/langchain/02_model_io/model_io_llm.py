from langchain.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct" #← 호출할 모델 지정
             )

result = llm(
    "맛있는 라면을",  #← 언어모델에 입력되는 텍스트
    stop="."  #← "." 가 출력된 시점에서 계속을 생성하지 않도록
)
print(result)
