### 인덱스 구축

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

### from langchain_cohere import CohereEmbeddings

# 임베딩 설정
embd = OpenAIEmbeddings()

# 인덱싱할 문서 URL
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 문서 로드
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 문서 분할
# RecursiveCharacterTextSplitter를 사용하여 문서를 작은 청크로 나눕니다.
# tiktoken 인코더를 사용하여 토큰 기반으로 분할합니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# 벡터 저장소에 추가
# Chroma를 사용하여 분할된 문서를 벡터화하고 저장합니다.
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embd,
)
retriever = vectorstore.as_retriever()

### 라우터

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# 참고: Pydantic v2와 함께 langchain-core >= 0.3을 사용해야 합니다.
from pydantic import BaseModel, Field

# 데이터 모델
class RouteQuery(BaseModel):
    """사용자 쿼리를 가장 관련성 높은 데이터 소스로 라우팅합니다."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="주어진 사용자 질문을 웹 검색 또는 벡터 저장소로 라우팅하도록 선택합니다.",
    )

# 함수 호출이 가능한 LLM 설정
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# 프롬프트 설정
system = """당신은 사용자 질문을 벡터 저장소 또는 웹 검색으로 라우팅하는 전문가입니다.
벡터 저장소에는 에이전트, 프롬프트 엔지니어링 및 적대적 공격과 관련된 문서가 포함되어 있습니다.
이러한 주제에 대한 질문에는 벡터 저장소를 사용하세요. 그 외의 경우에는 웹 검색을 사용하세요."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who will the Bears draft first in the NFL draft?"}
    )
)
print(question_router.invoke({"question": "What are the types of agent memory?"}))

### 검색 결과 평가기

# 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 체크를 위한 이진 점수."""

    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 그렇지 않으면 'no'"
    )

# 함수 호출이 가능한 LLM 설정
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 프롬프트 설정
system = """당신은 검색된 문서가 사용자 질문과 관련이 있는지 평가하는 평가자입니다. \n 
    문서에 사용자 질문과 관련된 키워드나 의미가 포함되어 있다면 관련성이 있다고 평가하세요. \n
    엄격한 테스트일 필요는 없습니다. 목표는 잘못된 검색 결과를 필터링하는 것입니다. \n
    문서가 질문과 관련이 있는지 여부를 나타내는 이진 점수 'yes' 또는 'no'를 제공하세요."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "agent memory"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### 생성

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# 프롬프트 불러오기
prompt = hub.pull("rlm/rag-prompt")

# LLM 설정
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 후처리 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 체인 구성
rag_chain = prompt | llm | StrOutputParser()

# 실행
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

### 환각 평가기

# 데이터 모델
class GradeHallucinations(BaseModel):
    """생성된 답변에 환각이 있는지 확인하는 이진 점수."""

    binary_score: str = Field(
        description="답변이 사실에 근거하면 'yes', 그렇지 않으면 'no'"
    )

# 함수 호출이 가능한 LLM 설정
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 프롬프트 설정
system = """당신은 LLM 생성이 검색된 사실 집합에 근거하거나 지원되는지 평가하는 평가자입니다. \n 
     이진 점수 'yes' 또는 'no'를 제공하세요. 'Yes'는 답변이 사실 집합에 근거하거나 지원된다는 의미입니다."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "사실 집합: \n\n {documents} \n\n LLM 생성: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})

### 답변 평가기

# 데이터 모델
class GradeAnswer(BaseModel):
    """답변이 질문을 해결하는지 평가하는 이진 점수."""

    binary_score: str = Field(
        description="답변이 질문을 해결하면 'yes', 그렇지 않으면 'no'"
    )

# 함수 호출이 가능한 LLM 설정
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 프롬프트 설정
system = """당신은 답변이 질문을 해결하거나 다루는지 평가하는 평가자입니다. \n 
     이진 점수 'yes' 또는 'no'를 제공하세요. 'Yes'는 답변이 질문을 해결한다는 의미입니다."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "사용자 질문: \n\n {question} \n\n LLM 생성: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})

### 질문 재작성기

# LLM 설정
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# 프롬프트 설정
system = """당신은 입력 질문을 벡터 저장소 검색에 최적화된 더 나은 버전으로 변환하는 질문 재작성기입니다. \n 
     입력을 보고 기본적인 의미적 의도/의미에 대해 추론하려고 노력하세요."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "다음은 초기 질문입니다: \n\n {question} \n 개선된 질문을 작성하세요.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

### 검색

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

from typing import List

from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    그래프의 상태를 나타냅니다.

    속성:
        question: 질문
        generation: LLM 생성
        documents: 문서 리스트
    """

    question: str
    generation: str
    documents: List[str]

from langchain.schema import Document

def retrieve(state):
    """
    문서를 검색합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        state (dict): 검색된 문서를 포함하는 documents 키가 추가된 새로운 상태
    """
    print("---검색---")
    question = state["question"]

    # 검색
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    답변을 생성합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        state (dict): LLM 생성을 포함하는 generation 키가 추가된 새로운 상태
    """
    print("---생성---")
    question = state["question"]
    documents = state["documents"]

    # RAG 생성
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    검색된 문서가 질문과 관련이 있는지 판단합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        state (dict): 필터링된 관련 문서만 포함하도록 documents 키를 업데이트한 상태
    """

    print("---문서의 질문 관련성 확인---")
    question = state["question"]
    documents = state["documents"]

    # 각 문서 점수 매기기
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---평가: 문서 관련 있음---")
            filtered_docs.append(d)
        else:
            print("---평가: 문서 관련 없음---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    더 나은 질문을 생성하기 위해 쿼리를 변환합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        state (dict): 재작성된 질문으로 question 키를 업데이트한 상태
    """

    print("---쿼리 변환---")
    question = state["question"]
    documents = state["documents"]

    # 질문 재작성
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    """
    재작성된 질문을 기반으로 웹 검색을 수행합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        state (dict): 웹 검색 결과가 추가된 documents 키를 업데이트한 상태
    """

    print("---웹 검색---")
    question = state["question"]

    # 웹 검색
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

### 엣지 ###

def route_question(state):
    """
    질문을 웹 검색 또는 RAG로 라우팅합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        str: 호출할 다음 노드
    """

    print("---질문 라우팅---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---질문을 웹 검색으로 라우팅---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---질문을 RAG로 라우팅---")
        return "vectorstore"

def decide_to_generate(state):
    """
    답변을 생성할지 또는 질문을 재생성할지 결정합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        str: 호출할 다음 노드에 대한 이진 결정
    """

    print("---평가된 문서 검토---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 모든 문서가 check_relevance에 의해 필터링됨
        # 새로운 쿼리를 재생성할 것입니다
        print(
            "---결정: 모든 문서가 질문과 관련이 없음, 쿼리 변환---"
        )
        return "transform_query"
    else:
        # 관련 문서가 있으므로 답변 생성
        print("---결정: 생성---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    생성된 답변이 문서에 근거하고 질문에 답하는지 판단합니다.

    인자:
        state (dict): 현재 그래프 상태

    반환:
        str: 호출할 다음 노드에 대한 결정
    """

    print("---환각 확인---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # 환각 확인
    if grade == "yes":
        print("---결정: 생성된 답변이 문서에 근거함---")
        # 질문-답변 확인
        print("---생성된 답변 vs 질문 평가---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---결정: 생성된 답변이 질문을 해결함---")
            return "useful"
        else:
            print("---결정: 생성된 답변이 질문을 해결하지 못함---")
            return "not useful"
    else:
        pprint("---결정: 생성된 답변이 문서에 근거하지 않음, 재시도---")
        return "not supported"    

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# 노드 정의
workflow.add_node("web_search", web_search)  # 웹 검색
workflow.add_node("retrieve", retrieve)  # 검색
workflow.add_node("grade_documents", grade_documents)  # 문서 평가
workflow.add_node("generate", generate)  # 생성
workflow.add_node("transform_query", transform_query)  # 쿼리 변환

# 그래프 구축
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# 컴파일
app = workflow.compile()    

from pprint import pprint

# 실행
inputs = {
    "question": "What player at the Bears expected to draft first in the 2024 NFL draft?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # 노드
        pprint(f"노드 '{key}':")
        # 선택 사항: 각 노드에서 전체 상태 출력
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# 최종 생성
pprint(value["generation"])

# 실행
inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # 노드
        pprint(f"노드 '{key}':")
        # 선택 사항: 각 노드에서 전체 상태 출력
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# 최종 생성
pprint(value["generation"])