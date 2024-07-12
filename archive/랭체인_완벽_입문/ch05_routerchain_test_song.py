import chainlit as cl
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# 모델 초기화
model = ChatOpenAI(
    temperature=0,  # 응답 다양성 설정, 0~2, 기본값은 0.7
    model='gpt-3.5-turbo'
)



# 프롬프트 템플릿 정의
restaurant_template = """
당신은 관광 가이드입니다. 관광지 주변의 맛있는 음식점들을 많이 알고 있습니다.
좋은 음식점과 추천 요리를 추천할 수 있습니다.
여기 질문이 있습니다:
{input}
"""

transport_template = """
당신은 관광 가이드입니다. 대중 교통에 대해 많은 지식을 가지고 있습니다.
관광지로 가는 대중 교통 정보를 제공할 수 있습니다.
여기 질문이 있습니다:
{input}
"""

destination_template = """
당신은 관광 가이드입니다. 많은 좋은 관광지를 알고 있습니다.
관광객들에게 좋은 관광지를 추천할 수 있습니다.
여기 질문이 있습니다:
{input}
"""



# 프롬프트 정보 리스트
prompt_infos = [
    {
        "name": "restaurants",
        "description": "관광지 주변의 맛있는 음식점을 추천하는데 좋습니다.",
        "prompt_template": restaurant_template
    },
    {
        "name": "transport",
        "description": "관광지로 가는 교통편을 안내하는데 좋습니다.",
        "prompt_template": transport_template
    },
    {
        "name": "destination",
        "description": "관광지를 추천하는데 좋습니다.",
        "prompt_template": destination_template
    }
]

# 각 체인을 생성하고 딕셔너리에 추가
destination_chains = {}
for prompt_info in prompt_infos:
    name = prompt_info["name"]
    prompt = PromptTemplate.from_template(prompt_info["prompt_template"])
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)
    destination_chains[name] = chain

# 기본 체인 설정
default_prompt = PromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=model, prompt=default_prompt)

# 라우터 프롬프트 템플릿 설정
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)




@cl.on_chat_start
async def on_chat_start():

    # Router Chain 초기화
    router_chain = LLMRouterChain.from_llm(model, router_prompt)

    # MultiPromptChain 설정
    chain = MultiPromptChain(router_chain=router_chain, 
                            destination_chains=destination_chains, 
                            default_chain=default_chain, 
                            verbose=True
                            )


    await cl.Message(content="관광 안내 봇 입니다. 음식점, 교통, 관광지에 대한 질문에 답변 할 수 있습니다.").send()
    cl.user_session.set("chain ", chain)


@cl.on_message
async def on_message(input_message):
    message = input_message.content
    chain = cl.user_session.get("chain ")
    res = chain.run(message)
    await cl.Message(content=res).send()

