"""원문 md 에서 리스팅 제목을 뽑는 규칙.

`book_listings()` 는 최장 일치를 취했다. 그런데 본문이 리스팅을 **산문으로 참조** 하는
문장("Listing 9.6 uses all the components …")이 실제 캡션("#### Listing 9.6 The full node
classification process")보다 길어서 산문이 제목으로 잡혔다. 노트북이 제목을 표에 출력하기
시작하면서 드러났다. 이 교재에서만 7건이다.
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import config, listing_map  # noqa: E402


class TitleExtractionTests(unittest.TestCase):
    """캡션과 산문 참조를 구별한다."""

    def _titles(self, body: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            chapter = Path(tmp) / "chapter_09_x"
            chapter.mkdir()
            (chapter / "09_x.md").write_text(body, encoding="utf-8")
            study = config.Study.__new__(config.Study)     # 파일 접근만 쓴다
            object.__setattr__(study, "explainer_suffix", "_ko_explained.md") \
                if hasattr(study, "__setattr__") else None
            return listing_map.book_listings(_StubStudy(chapter / "09_x.md"), chapter)

    def test_heading_caption_beats_longer_prose(self) -> None:
        """`#### Listing N.M` 캡션이 더 긴 산문 참조를 이긴다."""
        body = textwrap.dedent("""\
            At this point we have all the ingredients. Listing 9.6 uses all the components
            (functions) from previous listings and prints the results. Switching is easy.

            #### Listing 9.6 The full node classification process

            some code
            """)
        titles = self._titles(body)
        self.assertEqual(titles["9.6"], "The full node classification process")

    def test_bare_caption_still_works(self) -> None:
        """헤딩 마커 없이 줄머리에 오는 캡션도 잡는다 (ch11 9.1 형태)."""
        body = "We used the following prompt.\n\nListing 9.1 Creating and drawing a karate club network\n\ncode\n"
        self.assertEqual(
            self._titles(body)["9.1"], "Creating and drawing a karate club network"
        )

    def test_bold_caption_works(self) -> None:
        body = "**Listing 9.2** Representing each node by its degree\n\ncode\n"
        self.assertEqual(self._titles(body)["9.2"], "Representing each node by its degree")

    def test_prose_only_reference_is_kept_as_fallback(self) -> None:
        """캡션이 아예 유실된 번호는 산문이라도 있는 편이 낫다 (번호 집합이 목적이다)."""
        body = "See Listing 9.9 for the results with three degrees.\n"
        self.assertIn("9.9", self._titles(body))

    def test_markdown_escapes_are_unescaped(self) -> None:
        """PDF->md 변환이 밑줄을 이스케이프한다. 제목은 사람이 읽을 형태로 준다."""
        body = "#### Listing 9.4 Creating an edge\\_index tensor using node IDs\n"
        self.assertEqual(
            self._titles(body)["9.4"], "Creating an edge_index tensor using node IDs"
        )

    def test_asterisk_and_bracket_escapes_unescaped(self) -> None:
        body = "#### Listing 9.5 Using HAS\\_PARENT\\*0.. for inference\n"
        self.assertEqual(
            self._titles(body)["9.5"], "Using HAS_PARENT*0.. for inference"
        )

    def test_trailing_pipe_and_space_stripped(self) -> None:
        body = "#### Listing 9.3 Using Node2Vec as an embedding technique   |\n"
        self.assertEqual(
            self._titles(body)["9.3"], "Using Node2Vec as an embedding technique"
        )


class RealStudyTitleTests(unittest.TestCase):
    """실제 교재에서 산문이 제목으로 잡히지 않는가."""

    @unittest.skipIf(not (REPO_ROOT / "source").is_dir(), "source/ 가 없다")
    def test_no_prose_titles_in_declared_listings(self) -> None:
        paths = sorted((REPO_ROOT / "source").glob(f"*/{config.CONFIG_NAME}"))
        if not paths:
            self.skipTest("교재가 없다")
        study = config.load(paths[0].parent)
        # 산문 참조의 특징: 소문자 동사로 시작한다 (uses / shows / contains / is …)
        prose_starts = ("uses ", "shows ", "contains ", "is ", "provides ", "presents ")
        offenders = []
        for chapter_dir, repo in study.chapter_map.items():
            titles = listing_map.book_listings(study, study.root / chapter_dir)
            for number, title in titles.items():
                if title.lower().startswith(prose_starts):
                    offenders.append(f"{repo} {number}: {title[:60]}")
        self.assertEqual(offenders, [], "산문 참조가 제목으로 잡혔다:\n" + "\n".join(offenders))


class _StubStudy:
    """`original_md_for` 만 쓰는 최소 스텁."""

    def __init__(self, md: Path) -> None:
        self._md = md

    def original_md_for(self, chapter_dir: Path):   # noqa: ARG002
        return self._md


if __name__ == "__main__":
    unittest.main()
