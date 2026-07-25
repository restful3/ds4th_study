"""교재 학습자료(study-materials) 검증 테스트.

CI 는 docs/ 와 agent-support/ 만 sparse checkout 하므로 source/ 가 없다.
교재를 찾지 못하면 건너뛴다. 로컬에서는 실제 교재를 대상으로 돌아간다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import config, listing_map, verify  # noqa: E402


def discover_studies() -> list[config.Study]:
    """source/ 와 archive/ 에서 study.toml 이 있는 교재를 모은다."""
    studies = []
    for base in ("source", "archive"):
        root = REPO_ROOT / base
        if not root.is_dir():
            continue
        for path in sorted(root.glob(f"*/{config.CONFIG_NAME}")):
            try:
                studies.append(config.load(path.parent))
            except config.StudyConfigError:
                continue
    return studies


STUDIES = discover_studies()
skip_without_study = unittest.skipIf(not STUDIES, "study.toml 이 있는 교재가 없다 (sparse checkout)")


class ListingOffsetTests(unittest.TestCase):
    """리스팅 오프셋 도출 로직. 챕터마다 값이 달라 추측이 통하지 않는다."""

    @skip_without_study
    def test_declared_offsets_match_derived(self) -> None:
        """study.toml 에 적힌 오프셋이 제목 매칭으로 다시 도출된 값과 같은가."""
        for study in STUDIES:
            for repo_dir, declared in study.listing_offsets.items():
                with self.subTest(study=study.slug, chapter=repo_dir):
                    derived = listing_map.derive_offset(study, repo_dir)
                    if derived is None:
                        self.skipTest(f"{repo_dir}: 리스팅이 없어 도출 불가")
                    self.assertEqual(
                        declared, derived.offset,
                        f"{repo_dir}: study.toml={declared} 인데 도출값={derived.offset}. "
                        f"신뢰도 {derived.confidence:.2f}. 오프셋은 챕터마다 다르므로 "
                        f"다른 챕터 값을 복사하면 안 된다.",
                    )

    @skip_without_study
    def test_mapped_chapters_exist(self) -> None:
        """study.toml 의 챕터 대응이 실제 디렉터리를 가리키는가."""
        for study in STUDIES:
            src_dirs = study.src_dirs()
            for chapter_name, repo_dir in study.chapter_map.items():
                with self.subTest(study=study.slug, chapter=chapter_name):
                    self.assertTrue(
                        (study.root / chapter_name).is_dir(),
                        f"챕터 폴더가 없다: {chapter_name}",
                    )
                    self.assertIn(
                        repo_dir, src_dirs,
                        f"{chapter_name} -> {repo_dir} 인데 사본이 없다",
                    )

    @skip_without_study
    def test_meap_only_dirs_are_separated(self) -> None:
        """MEAP 전용 디렉터리가 챕터 폴더 밑에 섞여 있지 않은가."""
        for study in STUDIES:
            mapped = set(study.chapter_map.values())
            for repo_dir in study.meap_only:
                with self.subTest(study=study.slug, chapter=repo_dir):
                    self.assertNotIn(
                        repo_dir, mapped,
                        f"{repo_dir} 는 MEAP 전용인데 챕터에 매핑돼 있다",
                    )
                    location = study.src_dirs().get(repo_dir)
                    if location is not None:
                        self.assertEqual(
                            location.parent, study.meap_dir,
                            f"{repo_dir} 는 meap-only/ 에 있어야 한다 (현재 {location.parent})",
                        )


class MakefileParityTests(unittest.TestCase):
    """업스트림 Makefile 을 Python 으로 옮길 때 타깃·순서가 빠지지 않았는가."""

    @skip_without_study
    def test_tasks_match_upstream_makefiles(self) -> None:
        for study in STUDIES:
            if not study.upstream.is_dir():
                continue
            failures = verify.check_makefile_parity(study)
            with self.subTest(study=study.slug):
                self.assertEqual([], failures, "\n".join(failures))


class NotebookTests(unittest.TestCase):
    """노트북 규격. 그림 내장과 상대경로 링크가 지켜지는가."""

    @skip_without_study
    def test_notebooks_pass_gate(self) -> None:
        for study in STUDIES:
            for notebook in verify.find_notebooks(study):
                with self.subTest(study=study.slug, notebook=notebook.name):
                    failures = verify.check_notebook(study, notebook, check_urls=False)
                    self.assertEqual([], failures, "\n".join(failures))


class EnvironmentTests(unittest.TestCase):
    """부트스트랩이 만들어야 하는 상태. 미구축 교재는 건너뛴다."""

    @skip_without_study
    def test_bootstrapped_studies_are_consistent(self) -> None:
        for study in STUDIES:
            if not study.venv.is_dir():
                self.skipTest(f"{study.slug}: .venv 없음 (setup_env.py 미실행)")
            with self.subTest(study=study.slug):
                failures = verify.check_environment(study)
                self.assertEqual([], failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
