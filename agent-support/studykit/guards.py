"""LLM·DB 호출 셀 가드가 삼켜도 되는 예외.

노트북이 외부 서비스를 부르는 셀은 프록시나 DB 가 아플 때 죽지 않아야 한다. 그렇다고
`except Exception` 으로 감싸면 결정적 회귀도 출력 한 줄만 남기고 지나가고, 완성 게이트는
오류 output 을 세므로 "실행 오류 0건" 으로 통과한다.

경계는 **같은 요청을 다시 보내면 결과가 달라질 수 있는가** 다.

  * 연결 실패·시간 초과·408·409·429·5xx  -> 다시 보내면 달라진다. 삼켜도 된다
  * 400·401·403·404·422                  -> 몇 번을 보내도 같다. 삼키면 안 된다
  * 임포트·스키마·템플릿·Cypher 구성 오류 -> 환경과 무관하게 재현된다. 삼키면 안 된다

**예외 타입만으로는 이 경계를 그을 수 없다.** `openai.APIStatusError` 하나가 400 부터
429 까지를 다 담고, `httpx.HTTPStatusError` 도 마찬가지다. 그래서 타입 튜플이 아니라
`is_transient(exc)` 술어를 쓴다. 타입 튜플로 잡던 판의 실수: `openai.APIError` 는
`BadRequestError`·`AuthenticationError` 의 상위 클래스이고, `httpx.TransportError` 는
`UnsupportedProtocol`·`LocalProtocolError` 를 포함한다.

    from studykit import guards

    try:
        result = call_the_model()
    except Exception as exc:
        if not guards.is_transient(exc):
            raise
        print(f"일시 실패 — {type(exc).__name__}")
"""
from __future__ import annotations

import importlib

#: 다시 보내면 달라질 수 있는 HTTP 상태. 408 요청 시간 초과 · 409 충돌 · 429 한도 ·
#: 5xx 서버 오류. 나머지 4xx 는 요청 자체가 잘못된 것이라 재시도해도 같다.
RETRYABLE_STATUS = (408, 409, 429)

#: 내장 예외. OSError 계열이라 소켓·DNS 실패가 여기로도 온다.
_BUILTIN = (TimeoutError, ConnectionError)

#: (모듈, 예외 이름들) — 타입만으로 판정이 끝나는 것들. 설치된 것만 모은다.
_RETRYABLE_TYPES = (
    # APIConnectionError 는 APITimeoutError 의 상위 클래스다. APIError·APIStatusError 는
    # 4xx 를 포함하므로 여기 넣지 않고 아래 상태 코드 규칙으로 처리한다.
    ("openai", ("APIConnectionError",)),
    # TransportError 전체가 아니다 — UnsupportedProtocol·LocalProtocolError 는 우리가
    # 잘못 만든 요청이라 결정적이다.
    ("httpx", ("TimeoutException", "NetworkError", "ProxyError", "RemoteProtocolError")),
    # 드라이버가 재시도 대상으로 분류하는 것들. CypherSyntaxError 는 아니다.
    ("neo4j.exceptions", ("ServiceUnavailable", "SessionExpired", "TransientError")),
)

#: 상태 코드를 봐야 하는 예외. (모듈, 예외 이름, 코드를 꺼내는 함수)
_STATUS_CARRIERS = (
    ("openai", "APIStatusError", lambda exc: getattr(exc, "status_code", None)),
    ("httpx", "HTTPStatusError",
     lambda exc: getattr(getattr(exc, "response", None), "status_code", None)),
)


def _load(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def transient_errors() -> tuple[type[BaseException], ...]:
    """타입만으로 일시적이라고 판정되는 예외 튜플.

    상태 코드를 봐야 하는 것은 여기 없다. 일반적인 판정에는 `is_transient` 를 써라.
    """
    found: list[type[BaseException]] = list(_BUILTIN)
    for name, attributes in _RETRYABLE_TYPES:
        module = _load(name)
        if module is None:
            continue
        found += [getattr(module, attribute) for attribute in attributes
                  if hasattr(module, attribute)]
    return tuple(found)


def retryable_status(status: int | None) -> bool:
    """이 상태 코드는 다시 보낼 만한가."""
    if status is None:
        return False
    return status in RETRYABLE_STATUS or status >= 500


def is_transient(error: BaseException) -> bool:
    """이 예외를 삼켜도 되는가. 아니면 호출부가 다시 올려야 한다."""
    for name, attribute, read_status in _STATUS_CARRIERS:
        module = _load(name)
        carrier = getattr(module, attribute, None) if module else None
        if carrier is not None and isinstance(error, carrier):
            return retryable_status(read_status(error))
    return isinstance(error, transient_errors())
