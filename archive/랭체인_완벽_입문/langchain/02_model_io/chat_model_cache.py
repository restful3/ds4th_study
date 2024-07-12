import time  #← 실행 시간을 측정하기 위해 time 모듈 가져오기
import langchain
from langchain.cache import InMemoryCache  #← InMemoryCache 가져오기
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

langchain.llm_cache = InMemoryCache() #← llm_cache에 InMemoryCache 설정

chat = ChatOpenAI()
start = time.time() #← 실행 시작 시간 기록
result = chat([ #← 첫 번째 실행을 수행
    HumanMessage(content="안녕하세요!")
])

end = time.time() #← 실행 종료 시간 기록
print(result.content)
print(f"실행 시간: {end - start}초")

start = time.time() #← 실행 시작 시간 기록
result = chat([ #← 같은 내용으로 두 번째 실행을 함으로써 캐시가 활용되어 즉시 실행 완료됨
    HumanMessage(content="안녕하세요!")
])

end = time.time() #← 실행 종료 시간 기록
print(result.content)
print(f"실행 시간: {end - start}초")
