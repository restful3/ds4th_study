from langchain.embeddings import OpenAIEmbeddings  #← OpenAIEmbeddings를 가져오기
from numpy import dot  #← 벡터의 유사도를 계산하기 위해 dot을 가져오기
from numpy.linalg import norm  #← 벡터의 유사도를 계산하기 위해 norm을 가져오기

embeddings = OpenAIEmbeddings( #← OpenAIEmbeddings를 초기화
    model="text-embedding-ada-002"
)

query_vector = embeddings.embed_query("비행 자동차의 최고 속도는?") #← 질문을 벡터화

print(f"벡터화된 질문: {query_vector[:5]}") #← 벡터의 일부를 표시

document_1_vector = embeddings.embed_query("비행 자동차의 최고 속도는 시속 150km입니다.") #← 문서 1의 벡터를 얻음
document_2_vector = embeddings.embed_query("닭고기를 적당히 양념한 후 중불로 굽다가 가끔 뒤집어 주면서 겉은 고소하고 속은 부드럽게 익힌다.") #← 문서 2의 벡터를 얻음

cos_sim_1 = dot(query_vector, document_1_vector) / (norm(query_vector) * norm(document_1_vector)) #← 벡터의 유사도를 계산
print(f"문서 1과 질문의 유사도: {cos_sim_1}")
cos_sim_2 = dot(query_vector, document_2_vector) / (norm(query_vector) * norm(document_2_vector)) #← 벡터의 유사도를 계산
print(f"문서 2와 질문의 유사도: {cos_sim_2}")
