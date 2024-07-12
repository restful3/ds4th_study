from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

chat = ChatOpenAI(model="gpt-3.5-turbo")

write_article_chain = LLMChain( #← 기사를 쓰는 LLMChain 만들기
    llm=chat,
    prompt=PromptTemplate(
        template="{input}에 관한 기사를 써주세요.",
        input_variables=["input"],
    ),
)

translate_chain = LLMChain( #← 번역하는 LLMChain을 생성
    llm=chat,
    prompt=PromptTemplate(
        template="다음 문장을 영어로 번역해 주세요.\n{input}",
        input_variables=["input"],
    ),
)

sequential_chain = SimpleSequentialChain( #← SimpleSequentialChain을 생성
    chains=[ #← 실행할 Chain을 지정
        write_article_chain,
        translate_chain,
    ]
)

result = sequential_chain.run("일렉트릭 기타 선택 방법")

print(result)
