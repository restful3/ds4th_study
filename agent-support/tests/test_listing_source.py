"""리스팅 소스 리더 테스트.

`[mapping.listings.chXX]` 는 study.toml 이 정본이라고 선언했지만, 실제로는 노트북마다
비슷한 AST 헬퍼와 폴백 표를 각자 구현해 정본이 정본이 아니게 됐다 (ch11·ch13·ch14 는
resolve_listing() 을 호출조차 하지 않았다). 이 모듈은 그 해석을 한 곳으로 모은다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import listing_source  # noqa: E402

SAMPLE = textwrap.dedent(
    '''\
    """모듈 독스트링."""
    import os


    def alpha(x):
        """첫 함수."""
        return x + 1


    class Holder:
        CONST = 3

        def method(self, y):
            return y * 2


    if __name__ == "__main__":
        print(alpha(1))
        print(Holder().method(2))
    '''
)


class SpecParsingTests(unittest.TestCase):
    """study.toml 의 인라인 테이블을 스펙으로 읽는다."""

    def test_repo_kind_keeps_file_number(self) -> None:
        spec = listing_source.parse({"repo": "2.3"})
        self.assertEqual(spec.kind, "repo")
        self.assertEqual(spec.repo_number, "2.3")

    def test_repo_file_requires_path(self) -> None:
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.parse({"source": "repo-file", "symbol": "alpha"})

    def test_explainer_reason_is_validated(self) -> None:
        spec = listing_source.parse({"source": "explainer", "reason": "book-only"})
        self.assertEqual(spec.kind, "explainer")
        self.assertEqual(spec.reason, "book-only")
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.parse({"source": "explainer", "reason": "made-up"})

    def test_symbol_and_range_are_mutually_exclusive(self) -> None:
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.parse(
                {"source": "repo-file", "path": "a.py", "symbol": "alpha", "start": 1, "end": 2}
            )

    def test_unknown_source_rejected(self) -> None:
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.parse({"source": "somewhere-else", "path": "a.py"})


class PathConfinementTests(unittest.TestCase):
    """경로는 챕터 소스 폴더를 벗어날 수 없다."""

    def test_parent_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "src" / "ch99"
            base.mkdir(parents=True)
            (Path(tmp) / "secret.py").write_text("x = 1\n", encoding="utf-8")
            spec = listing_source.parse(
                {"source": "repo-file", "path": "../secret.py", "symbol": "__main__"}
            )
            with self.assertRaises(listing_source.ListingSpecError):
                listing_source.read(spec, base)

    def test_absolute_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            spec = listing_source.parse(
                {"source": "repo-file", "path": "/etc/hostname", "symbol": "__main__"}
            )
            with self.assertRaises(listing_source.ListingSpecError):
                listing_source.read(spec, base)


class SymbolReadingTests(unittest.TestCase):
    """symbol 로 소스 조각을 잘라낸다. 줄 번호는 매번 ast 로 다시 찾는다."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.target = self.base / "sample.py"
        self.target.write_text(SAMPLE, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _read(self, **extra):
        spec = listing_source.parse({"source": "repo-file", "path": "sample.py", **extra})
        return listing_source.read(spec, self.base)

    def test_function_symbol(self) -> None:
        chunk = self._read(symbol="alpha")
        self.assertIn("def alpha(x):", chunk.text)
        self.assertIn("return x + 1", chunk.text)
        self.assertNotIn("class Holder", chunk.text)

    def test_class_symbol(self) -> None:
        chunk = self._read(symbol="Holder")
        self.assertIn("class Holder:", chunk.text)
        self.assertIn("def method", chunk.text)
        self.assertNotIn("def alpha", chunk.text)

    def test_dotted_symbol_reaches_method(self) -> None:
        """ch11 의 EvaluateEmbedding.train 같은 클래스 안 메서드."""
        chunk = self._read(symbol="Holder.method")
        self.assertIn("def method(self, y):", chunk.text)
        self.assertNotIn("CONST", chunk.text)

    def test_main_means_whole_file(self) -> None:
        chunk = self._read(symbol="__main__")
        self.assertEqual(chunk.text.rstrip("\n"), SAMPLE.rstrip("\n"))
        self.assertEqual(chunk.start, 1)

    def test_entry_means_dunder_main_block(self) -> None:
        """__entry__ 는 `if __name__ == "__main__":` 블록만."""
        chunk = self._read(symbol="__entry__")
        self.assertIn('if __name__ == "__main__":', chunk.text)
        self.assertIn("print(alpha(1))", chunk.text)
        self.assertNotIn("def alpha", chunk.text)

    def test_whole_file_flag_distinguishes_script_from_range(self) -> None:
        """symbol="__main__" 과 start/end 는 둘 다 symbol 이 없다 — 플래그로 구분해야 한다.

        ch11 은 9.10 을 파일 전체로, 9.6 을 줄 범위로 선언한다. `chunk.symbol is None`
        으로 "스크립트인가" 를 판단하면 두 경우가 같아져 오작동한다.
        """
        whole = self._read(symbol="__main__")
        self.assertTrue(whole.whole_file)
        self.assertIsNone(whole.symbol)

        spec = listing_source.parse(
            {"source": "repo-file", "path": "sample.py", "start": 5, "end": 7}
        )
        sliced = listing_source.read(spec, self.base)
        self.assertFalse(sliced.whole_file)
        self.assertIsNone(sliced.symbol)

        named = self._read(symbol="alpha")
        self.assertFalse(named.whole_file)
        self.assertEqual(named.symbol, "alpha")

    def test_origin_marks_whole_file(self) -> None:
        self.assertIn("전체", self._read(symbol="__main__").origin())

    def test_origin_keeps_subdirectory(self) -> None:
        """model/gnn_model.py 처럼 하위 폴더에 있으면 폴더까지 보여야 찾아갈 수 있다."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "model").mkdir()
            (base / "model" / "gnn.py").write_text("class A:\n    pass\n", encoding="utf-8")
            spec = listing_source.parse(
                {"source": "repo-file", "path": "model/gnn.py", "symbol": "A"}
            )
            chunk = listing_source.read(spec, base)
            self.assertIn("model/gnn.py", chunk.origin())

    def test_missing_symbol_raises(self) -> None:
        with self.assertRaises(listing_source.ListingSpecError):
            self._read(symbol="nope")

    def test_entry_missing_raises(self) -> None:
        (self.base / "noentry.py").write_text("def a():\n    return 1\n", encoding="utf-8")
        spec = listing_source.parse(
            {"source": "repo-file", "path": "noentry.py", "symbol": "__entry__"}
        )
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.read(spec, self.base)

    def test_reported_line_numbers_are_one_based_inclusive(self) -> None:
        chunk = self._read(symbol="alpha")
        lines = SAMPLE.splitlines()
        self.assertEqual(lines[chunk.start - 1].strip(), "def alpha(x):")
        self.assertEqual(lines[chunk.end - 1].strip(), "return x + 1")


class RangeReadingTests(unittest.TestCase):
    """start/end 는 1-based inclusive 다. 함수 경계가 아닌 조각에만 쓴다."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        (self.base / "sample.py").write_text(SAMPLE, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_range_slice(self) -> None:
        spec = listing_source.parse(
            {"source": "repo-file", "path": "sample.py", "start": 5, "end": 7}
        )
        chunk = listing_source.read(spec, self.base)
        self.assertEqual(chunk.text.splitlines()[0].strip(), "def alpha(x):")
        self.assertEqual(len(chunk.text.splitlines()), 3)

    def test_range_beyond_file_raises(self) -> None:
        spec = listing_source.parse(
            {"source": "repo-file", "path": "sample.py", "start": 1, "end": 9999}
        )
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.read(spec, self.base)

    def test_inverted_range_raises(self) -> None:
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.parse(
                {"source": "repo-file", "path": "sample.py", "start": 9, "end": 4}
            )


class NonPythonTests(unittest.TestCase):
    """확장자 없는 Cypher·프롬프트 파일은 통째로 읽는다 (ast 대상이 아니다)."""

    def test_plain_text_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "12.15 DWPC").write_text("MATCH (c:Compound)\nRETURN c\n", encoding="utf-8")
            spec = listing_source.parse(
                {"source": "repo-file", "path": "12.15 DWPC", "symbol": "__main__"}
            )
            chunk = listing_source.read(spec, base)
            self.assertIn("MATCH (c:Compound)", chunk.text)

    def test_symbol_on_non_python_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "12.15 DWPC").write_text("MATCH (c)\n", encoding="utf-8")
            spec = listing_source.parse(
                {"source": "repo-file", "path": "12.15 DWPC", "symbol": "compute"}
            )
            with self.assertRaises(listing_source.ListingSpecError):
                listing_source.read(spec, base)


if __name__ == "__main__":
    unittest.main()


class GateIntegrationTests(unittest.TestCase):
    """게이트가 선언을 전수 검증하고, 여분 coverage 번호를 잡아내는가.

    이전 게이트는 `set(book) - set(declared)` 만 봐서 **여분** 은 무조건 통과했다.
    오타로 적은 번호("12.99")가 조용히 살아남는다.
    """

    def setUp(self) -> None:
        from studykit import verify
        self.verify = verify

    def test_reproduced_is_a_valid_coverage_kind(self) -> None:
        """'출력 예시' 리스팅 — 코드가 아니라 결과다. 노트북이 그 출력을 재현한다."""
        self.assertIn("reproduced", self.verify.COVERAGE_KINDS)

    def test_extra_coverage_number_is_rejected(self) -> None:
        """책에도 없고 study.toml 에도 없는 번호는 실패여야 한다."""
        failures = self.verify.check_coverage_numbers(
            declared={"10.4": "executed", "12.99": "executed"},
            book={"10.4": "제목"},
            mapped={"10.4"},
            label="x.ipynb",
        )
        self.assertTrue(any("12.99" in f for f in failures), failures)

    def test_declared_mapping_number_is_allowed_as_extra(self) -> None:
        """book_listings() 가 못 잡지만 study.toml 이 선언한 번호는 허용한다 (ch12 의 10.5)."""
        failures = self.verify.check_coverage_numbers(
            declared={"10.4": "executed", "10.5": "executed"},
            book={"10.4": "제목"},
            mapped={"10.4", "10.5"},
            label="x.ipynb",
        )
        self.assertEqual(failures, [])

    def test_missing_book_listing_still_fails(self) -> None:
        failures = self.verify.check_coverage_numbers(
            declared={"10.4": "executed"},
            book={"10.4": "제목", "10.6": "제목"},
            mapped={"10.4"},
            label="x.ipynb",
        )
        self.assertTrue(any("10.6" in f for f in failures), failures)

    @unittest.skipIf(not (REPO_ROOT / "source").is_dir(), "source/ 가 없다 (sparse checkout)")
    def test_real_study_declarations_resolve(self) -> None:
        """실제 교재의 모든 repo-file 선언이 파일·심볼까지 해결되는가."""
        from studykit import config
        studies = [config.load(p.parent)
                   for p in sorted((REPO_ROOT / "source").glob(f"*/{config.CONFIG_NAME}"))]
        if not studies:
            self.skipTest("교재가 없다")
        for study in studies:
            with self.subTest(study=study.slug):
                self.assertEqual(self.verify.check_declared_listings(study), [])


class StudyFacadeTests(unittest.TestCase):
    """노트북이 쓰는 얇은 진입점. 노트북마다 표와 ast 헬퍼를 재구현하지 않게 한다."""

    @unittest.skipIf(not (REPO_ROOT / "source").is_dir(), "source/ 가 없다 (sparse checkout)")
    def setUp(self) -> None:
        from studykit import config
        paths = sorted((REPO_ROOT / "source").glob(f"*/{config.CONFIG_NAME}"))
        if not paths:
            self.skipTest("교재가 없다")
        self.study = config.load(paths[0].parent)

    def test_chunk_for_resolves_symbol_chapter(self) -> None:
        """책 번호만 주면 study.toml 을 거쳐 소스 조각이 나온다."""
        chunk = listing_source.chunk_for(self.study, "ch11", "11.2")
        self.assertIn("class MultiHeadGraphAttention", chunk.text)
        self.assertTrue(chunk.start >= 1)
        self.assertIn("GNN_all_in_one.py", chunk.origin())

    def test_chunk_for_explainer_raises_with_reason(self) -> None:
        """해설판이 정본인 리스팅은 파일에서 읽을 수 없고, 이유가 메시지에 담긴다."""
        with self.assertRaises(listing_source.ListingSpecError) as ctx:
            listing_source.chunk_for(self.study, "ch12", "12.3")
        self.assertIn("book-only", str(ctx.exception))

    def test_chunk_for_unknown_number_raises(self) -> None:
        with self.assertRaises(listing_source.ListingSpecError):
            listing_source.chunk_for(self.study, "ch11", "11.99")

    def test_cross_reference_covers_declared_numbers(self) -> None:
        """대조표는 선언된 번호를 책 순서로 담고, 각 행이 위치를 안다."""
        rows = listing_source.cross_reference(self.study, "ch10")
        numbers = [r.number for r in rows]
        self.assertEqual(numbers, sorted(numbers, key=lambda n: [int(x) for x in n.split(".")]))
        self.assertIn("10.15", numbers)
        row = next(r for r in rows if r.number == "10.4")
        self.assertEqual(row.kind, "repo-file")
        self.assertIn("12.4", row.where)

    def test_cross_reference_marks_unnumbered_repo_files(self) -> None:
        """책 번호가 없는 저장소 리스팅도 함께 보여야 스터디원이 헷갈리지 않는다."""
        extra = listing_source.unnumbered_listings(self.study, "ch10")
        names = " ".join(extra)
        self.assertIn("12.18", names)   # 책이 번호를 주지 않은 LLM 프롬프트


class SuffixTrapTests(unittest.TestCase):
    """리스팅 파일명은 번호에 점을 포함한다 — Path.suffix 를 쓰면 안 된다.

    `Path("12.15 DWPC ...").suffix` 는 `.15 DWPC ...` 다. 파이썬 파일 판정을 suffix 로
    하면 확장자 없는 Cypher 파일에서 엉뚱한 에러 메시지가 나온다.
    """

    def test_dotted_name_without_extension_is_not_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            name = "12.15 DWPC between Metformin and Type 2 Diabetes"
            (base / name).write_text("MATCH (c)\nRETURN c\n", encoding="utf-8")
            spec = listing_source.parse(
                {"source": "repo-file", "path": name, "symbol": "compute"}
            )
            with self.assertRaises(listing_source.ListingSpecError) as ctx:
                listing_source.read(spec, base)
            self.assertIn("파이썬", str(ctx.exception))

    def test_dotted_python_name_is_python(self) -> None:
        """'12.4 Computing triangle metrics.py' 는 파이썬 파일이다."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            name = "12.4 Computing triangle metrics.py"
            (base / name).write_text("def compute():\n    return 1\n", encoding="utf-8")
            spec = listing_source.parse(
                {"source": "repo-file", "path": name, "symbol": "compute"}
            )
            chunk = listing_source.read(spec, base)
            self.assertIn("def compute():", chunk.text)


class TitleTests(unittest.TestCase):
    """대조표에 책 리스팅 제목이 붙어야 노트북이 자체 표를 들지 않는다."""

    @unittest.skipIf(not (REPO_ROOT / "source").is_dir(), "source/ 가 없다")
    def setUp(self) -> None:
        from studykit import config
        paths = sorted((REPO_ROOT / "source").glob(f"*/{config.CONFIG_NAME}"))
        if not paths:
            self.skipTest("교재가 없다")
        self.study = config.load(paths[0].parent)

    def test_title_comes_from_book_listings(self) -> None:
        """원서 md 에서 캡션을 뽑을 수 있는 번호는 제목이 자동으로 붙는다."""
        rows = {r.number: r for r in listing_source.cross_reference(self.study, "ch10")}
        self.assertIn("triangle", rows["10.4"].title.lower())

    def test_title_falls_back_to_declared(self) -> None:
        """캡션이 유실된 번호는 study.toml 의 title 선언을 쓴다 (없으면 빈 문자열)."""
        rows = {r.number: r for r in listing_source.cross_reference(self.study, "ch09")}
        self.assertIn("9.5", rows)
        self.assertIsInstance(rows["9.5"].title, str)
