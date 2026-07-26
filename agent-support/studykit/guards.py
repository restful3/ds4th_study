"""LLM·DB 호출 셀 가드가 삼켜도 되는 예외.

노트북이 외부 서비스를 부르는 셀은 프록시나 DB 가 아플 때 죽지 않아야 한다. 그렇다고
`except Exception` 으로 감싸면 결정적 회귀도 출력 한 줄만 남기고 지나가고, 완성 게이트는
오류 output 을 세므로 "실행 오류 0건" 으로 통과한다.

경계는 **같은 요청을 다시 보내면 결과가 달라질 수 있는가** 다.

  * 연결 실패·시간 초과·429·5xx  -> 다시 보내면 달라진다. 삼켜도 된다
  * 400·401·403·404·422          -> 몇 번을 보내도 같다. 삼키면 안 된다
  * 임포트·스키마·템플릿·Cypher 구성 오류 -> 환경과 무관하게 재현된다. 삼키면 안 된다

`openai.APIError` 와 `httpx.HTTPError` 를 그대로 쓰면 안 되는 이유가 여기 있다. 둘 다
결정적 4xx 를 포함하는 상위 클래스다.
"""
from __future__ import annotations

import importlib

#: (모듈, 예외 이름들). 설치돼 있는 것만 모은다 — 교재마다 의존물이 다르다.
_RETRYABLE = (
    # 연결·시간 초과·429·5xx 만. APIError·APIStatusError 는 4xx 를 포함해 넣지 않는다.
    ("openai", ("APIConnectionError", "APITimeoutError", "RateLimitError",
                "InternalServerError")),
    # TransportError 는 연결·읽기·쓰기 타임아웃과 연결 실패의 상위 클래스다.
    # HTTPStatusError 는 여기 들어오지 않는다 (HTTPError 의 다른 갈래다).
    ("httpx", ("TransportError",)),
    # 드라이버가 재시도 대상으로 분류하는 것들. CypherSyntaxError 는 아니다.
    ("neo4j.exceptions", ("ServiceUnavailable", "SessionExpired", "TransientError")),
)

#: 내장 예외. OSError 계열이라 소켓·DNS 실패가 여기로도 온다.
_BUILTIN = (TimeoutError, ConnectionError)


def transient_errors() -> tuple[type[BaseException], ...]:
    """`except` 절에 그대로 넘길 수 있는 예외 튜플."""
    found: list[type[BaseException]] = list(_BUILTIN)
    for name, attributes in _RETRYABLE:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        found += [getattr(module, attribute) for attribute in attributes
                  if hasattr(module, attribute)]
    return tuple(found)
