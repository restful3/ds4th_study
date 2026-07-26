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

    def setUp(self) -> None:
        self.transient = guards.transient_errors()

    def test_builtin_connection_failures_are_transient(self) -> None:
        for error in (TimeoutError, ConnectionError, ConnectionResetError):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, self.transient))

    def test_deterministic_builtins_are_not_transient(self) -> None:
        for error in (ImportError, ModuleNotFoundError, KeyError, ValueError,
                      TypeError, AttributeError, NameError, AssertionError):
            with self.subTest(error=error.__name__):
                self.assertFalse(issubclass(error, self.transient))

    def test_openai_retryable_statuses_are_transient(self) -> None:
        openai = module("openai")
        if openai is None:
            self.skipTest("openai 가 없다")
        for name in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                     "InternalServerError"):
            with self.subTest(error=name):
                self.assertTrue(issubclass(getattr(openai, name), self.transient))

    def test_openai_client_errors_are_not_transient(self) -> None:
        """4xx 는 같은 요청을 다시 보내도 같은 답이 온다. 삼키면 회귀가 숨는다."""
        openai = module("openai")
        if openai is None:
            self.skipTest("openai 가 없다")
        for name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError",
                     "NotFoundError", "UnprocessableEntityError"):
            with self.subTest(error=name):
                self.assertFalse(issubclass(getattr(openai, name), self.transient))

    def test_httpx_transport_errors_only(self) -> None:
        httpx = module("httpx")
        if httpx is None:
            self.skipTest("httpx 가 없다")
        self.assertTrue(issubclass(httpx.ConnectError, self.transient))
        self.assertTrue(issubclass(httpx.ReadTimeout, self.transient))
        self.assertFalse(issubclass(httpx.HTTPStatusError, self.transient))

    def test_neo4j_retryable_errors(self) -> None:
        exceptions = module("neo4j.exceptions")
        if exceptions is None:
            self.skipTest("neo4j 가 없다")
        for name in ("ServiceUnavailable", "SessionExpired", "TransientError"):
            with self.subTest(error=name):
                self.assertTrue(issubclass(getattr(exceptions, name), self.transient))
        self.assertFalse(issubclass(exceptions.CypherSyntaxError, self.transient))

    def test_missing_packages_do_not_break_the_tuple(self) -> None:
        """설치되지 않은 패키지는 조용히 빠지고 내장 예외는 남는다."""
        self.assertTrue(len(self.transient) >= 2)
        self.assertTrue(all(isinstance(e, type) for e in self.transient))
