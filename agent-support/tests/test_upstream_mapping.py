"""`[mapping.upstream_dirs]` 검증.

소스 폴더는 책 장 번호를, 업스트림은 MEAP 번호를 쓴다. 그 대응표가 틀리면
`check_makefile_parity` 가 엉뚱한 챕터끼리 비교하고도 통과한다 — 실제로 그렇게
실패한 적이 있다. 대응표 자체를 검사하지 않으면 renumber 의 대가가 조용히 남는다.

임시 fixture 로 검사하므로 실제 교재 없이도 돈다.

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

from studykit import config, verify  # noqa: E402

MAKEFILE = "init:\n\t$(PIP) install -r requirements.txt\n"


def build_study(
    tmp: Path,
    *,
    chapters: dict[str, str],
    upstream_dirs: dict[str, str],
    upstream_chapters: tuple[str, ...],
    meap_only: tuple[str, ...] = (),
) -> config.Study:
    """임시 교재를 만든다. chapters 는 {챕터폴더: 소스폴더} 다."""
    lines = [
        "[study]",
        'slug = "probe"',
        'title = "Probe"',
        "",
        "[mapping.chapters]",
    ]
    lines += [f'"{name}" = "{src}"' for name, src in chapters.items()]
    lines += ["", "[mapping.upstream_dirs]"]
    lines += [f'"{src}" = "{up}"' for src, up in upstream_dirs.items()]
    lines += ["", "[mapping]", f"meap_only = [{', '.join(repr(m) for m in meap_only)}]"]
    (tmp / config.CONFIG_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, src in chapters.items():
        (tmp / name / "src" / src).mkdir(parents=True, exist_ok=True)
    for name in meap_only:
        (tmp / "meap-only" / name).mkdir(parents=True, exist_ok=True)
    for name in upstream_chapters:
        chapter = tmp / "code" / "chapters" / name
        chapter.mkdir(parents=True, exist_ok=True)
        (chapter / "Makefile").write_text(MAKEFILE, encoding="utf-8")
    return config.load(tmp)


class UpstreamMappingTests(unittest.TestCase):
    """대응표가 업스트림 챕터와 1:1 인가."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_bijective_mapping_passes(self) -> None:
        """선언과 폴백이 섞여 있어도 1:1 이면 통과한다."""
        study = build_study(
            self.tmp,
            chapters={"chapter_02_a": "ch02", "chapter_03_b": "ch03"},
            upstream_dirs={"ch03": "ch04"},          # ch02 는 선언 없이 같은 이름
            upstream_chapters=("ch02", "ch04"),
        )
        self.assertEqual([], verify.check_upstream_mapping(study))

    def test_duplicate_upstream_value_is_rejected(self) -> None:
        """두 소스 폴더가 같은 업스트림을 가리키면 역방향 해석이 TOML 순서에 좌우된다."""
        study = build_study(
            self.tmp,
            chapters={"chapter_05_a": "ch05", "chapter_06_b": "ch06"},
            upstream_dirs={"ch05": "ch07", "ch06": "ch07"},
            upstream_chapters=("ch07",),
        )
        failures = verify.check_upstream_mapping(study)
        self.assertTrue(any("ch07" in f for f in failures), failures)

    def test_missing_src_for_declared_key_is_rejected(self) -> None:
        """대응표에만 있고 실물이 없는 소스 폴더."""
        study = build_study(
            self.tmp,
            chapters={"chapter_05_a": "ch05"},
            upstream_dirs={"ch05": "ch07", "ch99": "ch09"},
            upstream_chapters=("ch07", "ch09"),
        )
        failures = verify.check_upstream_mapping(study)
        self.assertTrue(any("ch99" in f for f in failures), failures)

    def test_renamed_src_without_entry_is_rejected(self) -> None:
        """이름이 바뀐 소스 폴더에 선언이 없으면 폴백이 조용히 빗나간다.

        `src_dir_for_upstream()` 은 선언이 없으면 이름이 같다고 본다. 그래서 ch05 가
        업스트림 ch07 에 대응하는데 선언이 빠지면, 존재하지도 않는 업스트림 ch05 를
        찾다가 Makefile 짝짓기가 통째로 빠진다.
        """
        study = build_study(
            self.tmp,
            chapters={"chapter_05_a": "ch05"},
            upstream_dirs={},
            upstream_chapters=("ch07",),
        )
        failures = verify.check_upstream_mapping(study)
        self.assertTrue(any("ch05" in f for f in failures), failures)

    def test_upstream_makefile_without_src_is_rejected(self) -> None:
        """업스트림 챕터가 어느 소스 폴더에도 대응되지 않는다."""
        study = build_study(
            self.tmp,
            chapters={"chapter_02_a": "ch02"},
            upstream_dirs={},
            upstream_chapters=("ch02", "ch03"),
        )
        failures = verify.check_upstream_mapping(study)
        self.assertTrue(any("ch03" in f for f in failures), failures)

    def test_meap_only_name_colliding_with_chapter_is_rejected(self) -> None:
        """meap-only 가 책 챕터와 이름이 겹치면 src_dirs() 가 조용히 덮어쓴다."""
        study = build_study(
            self.tmp,
            chapters={"chapter_05_a": "ch05"},
            upstream_dirs={"ch05": "ch07"},
            upstream_chapters=("ch05", "ch07"),
            meap_only=("ch05",),
        )
        failures = verify.check_upstream_mapping(study)
        self.assertTrue(any("ch05" in f for f in failures), failures)

    def test_src_dir_for_upstream_resolves_both_ways(self) -> None:
        """선언이 있으면 그 값을, 없으면 같은 이름을 돌려준다."""
        study = build_study(
            self.tmp,
            chapters={"chapter_02_a": "ch02", "chapter_05_b": "ch05"},
            upstream_dirs={"ch05": "ch07"},
            upstream_chapters=("ch02", "ch07"),
        )
        self.assertEqual("ch05", study.src_dir_for_upstream("ch07"))
        self.assertEqual("ch02", study.src_dir_for_upstream("ch02"))


class RealStudyMappingTests(unittest.TestCase):
    """실제 교재에도 같은 검사를 적용한다. sparse checkout 에서는 건너뛴다."""

    def test_declared_mappings_are_bijective(self) -> None:
        from test_study_materials import STUDIES

        if not STUDIES:
            self.skipTest("study.toml 이 있는 교재가 없다 (sparse checkout)")
        for study in STUDIES:
            if not study.upstream.is_dir():
                continue
            with self.subTest(study=study.slug):
                failures = verify.check_upstream_mapping(study)
                self.assertEqual([], failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
