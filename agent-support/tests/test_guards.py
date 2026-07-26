"""LLM·DB 호출 셀 가드가 무엇을 삼키는가.

노트북이 `except Exception` 으로 감싸면 결정적 회귀도 "실행 오류 0건" 으로 게이트를
통과한다. 그래서 좁혔는데, 좁힌 목록이 여전히 넓었다 — `openai.APIError` 는 400·401·
403·404·422 의 상위 클래스이고 `httpx.HTTPError` 는 `HTTPStatusError` 를 포함한다.
둘 다 재시도해도 같은 결과가 나오는 결정적 실패다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import guards  # noqa: E402


def module(name: str):
    try:
        return __import__(name, fromlist=["_"])
    except ImportError:
        return None


class TransientErrorTests(unittest.TestCase):
    """가드가 삼켜도 되는 것과 반드시 다시 올려야 하는 것."""

    def transient(self, error: BaseException) -> bool:
        return guards.is_transient(error)

    def test_builtin_connection_failures_are_transient(self) -> None:
        for error in (TimeoutError("t"), ConnectionError("c"), ConnectionResetError("r")):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(self.transient(error))

    def test_deterministic_builtins_are_not_transient(self) -> None:
        for error in (ImportError("i"), ModuleNotFoundError("m"), KeyError("k"),
                      ValueError("v"), TypeError("t"), AttributeError("a"),
                      NameError("n"), AssertionError("as")):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(self.transient(error))

    def test_openai_retryable_statuses_are_transient(self) -> None:
        openai = module("openai")
        if openai is None:
            self.skipTest("openai 가 없다")
        self.assertTrue(self.transient(openai.APIConnectionError(request=None)))
        for status in (408, 409, 429, 500, 502, 503, 529):
            with self.subTest(status=status):
                self.assertTrue(self.transient(fake_status_error(openai, status)))

    def test_openai_client_errors_are_not_transient(self) -> None:
        """4xx 는 같은 요청을 다시 보내도 같은 답이 온다. 삼키면 회귀가 숨는다."""
        openai = module("openai")
        if openai is None:
            self.skipTest("openai 가 없다")
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(self.transient(fake_status_error(openai, status)))

    def test_httpx_transport_errors_only(self) -> None:
        httpx = module("httpx")
        if httpx is None:
            self.skipTest("httpx 가 없다")
        for error in (httpx.ConnectError("x"), httpx.ReadTimeout("x"),
                      httpx.RemoteProtocolError("x"), httpx.ProxyError("x")):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(self.transient(error))
        # 프로토콜을 못 알아듣거나 우리가 잘못 만든 요청은 몇 번을 보내도 같다.
        for error in (httpx.UnsupportedProtocol("x"), httpx.LocalProtocolError("x")):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(self.transient(error))

    def test_httpx_status_errors_follow_the_status_code(self) -> None:
        httpx = module("httpx")
        if httpx is None:
            self.skipTest("httpx 가 없다")
        request = httpx.Request("GET", "http://localhost/x")
        for status, expected in ((429, True), (503, True), (404, False), (401, False)):
            error = httpx.HTTPStatusError(
                "x", request=request, response=httpx.Response(status, request=request))
            with self.subTest(status=status):
                self.assertEqual(expected, self.transient(error))

    def test_neo4j_retryable_errors(self) -> None:
        exceptions = module("neo4j.exceptions")
        if exceptions is None:
            self.skipTest("neo4j 가 없다")
        for name in ("ServiceUnavailable", "SessionExpired", "TransientError"):
            with self.subTest(error=name):
                self.assertTrue(self.transient(getattr(exceptions, name)("x")))
        self.assertFalse(self.transient(exceptions.CypherSyntaxError("x")))

    def test_missing_packages_do_not_break_the_predicate(self) -> None:
        """설치되지 않은 패키지는 조용히 빠지고 내장 예외 판정은 남는다."""
        self.assertTrue(self.transient(TimeoutError("t")))
        self.assertFalse(self.transient(RuntimeError("r")))


def fake_status_error(openai, status: int):
    """상태 코드만 있는 APIStatusError 대역. 실제 응답 객체를 만들 필요가 없다."""
    error = openai.APIStatusError.__new__(openai.APIStatusError)
    error.status_code = status
    return error
