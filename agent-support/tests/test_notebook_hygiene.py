"""renumber 잔재와 저장 출력 위생 검사.

소스 폴더를 책 장 번호로 renumber 하면서 169개 파일을 옮기고 약 270개 참조를
고쳤다. 게이트가 통과하는데도 `chch07` 같은 이중 치환, 로컬 폴더명을 업스트림이라
부르는 출력, 다른 챕터를 가리키는 절대경로가 노트북에 남았다. 전부 문자열 수준의
불변식이라 정적으로 잡을 수 있다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import config, notebook, verify  # noqa: E402

UPSTREAM_URL = "https://github.com/alenegro81/knowledge-graphs-and-llms-in-action"


def write_study(tmp: Path) -> config.Study:
    """소스 폴더 ch05 가 업스트림 ch07 에 대응하는 최소 교재."""
    (tmp / config.CONFIG_NAME).write_text(
        "[study]\n"
        'slug = "probe"\n'
        'title = "Probe"\n\n'
        "[upstream]\n"
        f'url = "{UPSTREAM_URL}.git"\n\n'
        "[mapping.chapters]\n"
        '"chapter_05_extracting" = "ch05"\n\n'
        "[mapping.upstream_dirs]\n"
        '"ch05" = "ch07"\n',
        encoding="utf-8",
    )
    (tmp / "chapter_05_extracting" / "src" / "ch05").mkdir(parents=True)
    return config.load(tmp)


def write_notebook(study: config.Study, *, markdown: str = "", outputs: list | None = None) -> Path:
    path = study.root / "chapter_05_extracting" / "src" / "ch05" / "05_chapter_guide.ipynb"
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [markdown]}]
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": 1,
        "source": ["print('hi')\n"],
        "outputs": outputs or [],
    })
    path.write_text(json.dumps({
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }, ensure_ascii=False), encoding="utf-8")
    return path


def stream(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": [text]}


class RenumberResidueTests(unittest.TestCase):
    """치환을 두 번 적용하거나 한 번도 적용하지 않은 흔적."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.study = write_study(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def test_doubled_prefix_is_rejected(self) -> None:
        """`f"ch{up}"` 에 up 이 이미 ch 를 달고 있으면 chch07 이 된다."""
        path = write_notebook(self.study, markdown="업스트림 저장소는 chch07 이다.")
        failures = verify.check_notebook(self.study, path, check_urls=False)
        self.assertTrue(any("chch07" in f for f in failures), failures)

    def test_clean_upstream_name_passes(self) -> None:
        path = write_notebook(self.study, markdown="업스트림 저장소는 ch07 이다.")
        self.assertEqual([], verify.check_notebook(self.study, path, check_urls=False))

    def test_chapter_path_pointing_at_other_source_is_rejected(self) -> None:
        """chapter_05_… 아래는 src/ch05 여야 한다. 치환이 두 번 돌면 어긋난다."""
        path = write_notebook(
            self.study,
            markdown="`chapter_05_extracting/src/ch07/listings/` 를 보라.",
        )
        failures = verify.check_notebook(self.study, path, check_urls=False)
        self.assertTrue(any("src/ch07" in f for f in failures), failures)

    def test_github_url_must_use_upstream_number(self) -> None:
        """저장소 안은 책 번호, GitHub 링크는 업스트림 번호다."""
        path = write_notebook(
            self.study,
            markdown=f"[업스트림]({UPSTREAM_URL}/tree/main/chapters/ch05)",
        )
        failures = verify.check_notebook(self.study, path, check_urls=False)
        self.assertTrue(any("chapters/ch05" in f for f in failures), failures)

    def test_github_url_with_correct_number_passes(self) -> None:
        path = write_notebook(
            self.study,
            markdown=f"[업스트림]({UPSTREAM_URL}/tree/main/chapters/ch07)",
        )
        self.assertEqual([], verify.check_notebook(self.study, path, check_urls=False))


class SavedOutputPathTests(unittest.TestCase):
    """저장 출력에 실행한 기계의 절대경로가 남으면 안 된다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.study = write_study(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def test_study_root_in_output_is_rejected(self) -> None:
        path = write_notebook(
            self.study,
            outputs=[stream(f"교재 루트 : {self.study.root}/dataset/pyg\n")],
        )
        failures = verify.check_notebook(self.study, path, check_urls=False)
        self.assertTrue(any("절대경로" in f for f in failures), failures)

    def test_home_directory_in_output_is_rejected(self) -> None:
        path = write_notebook(
            self.study,
            outputs=[stream("/home/someone/.venv/lib/python3.10/site-packages/tqdm/auto.py:21\n")],
        )
        failures = verify.check_notebook(self.study, path, check_urls=False)
        self.assertTrue(any("절대경로" in f for f in failures), failures)

    def test_normalized_output_passes(self) -> None:
        path = write_notebook(self.study, outputs=[stream("교재 루트 : <교재>/dataset/pyg\n")])
        self.assertEqual([], verify.check_notebook(self.study, path, check_urls=False))

    def test_system_paths_are_not_flagged(self) -> None:
        """/usr, /opt 는 기계마다 같으므로 잡지 않는다."""
        path = write_notebook(self.study, outputs=[stream("/usr/lib/x.so 를 읽었다\n")])
        self.assertEqual([], verify.check_notebook(self.study, path, check_urls=False))

    def test_source_cells_are_not_scanned(self) -> None:
        """소스에 홈 경로를 설명으로 적는 것까지 막지는 않는다 — 검사 대상은 출력이다."""
        path = write_notebook(self.study, markdown="설치 위치는 보통 /home/<사용자>/.venv 다.")
        self.assertEqual([], verify.check_notebook(self.study, path, check_urls=False))


class OutputNormalizerTests(unittest.TestCase):
    """게이트가 요구하는 상태를 만들어 주는 도구."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.study = write_study(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def test_normalize_replaces_study_root_and_home(self) -> None:
        path = write_notebook(self.study, outputs=[
            stream(f"{self.study.root}/dataset/pyg\n"),
            stream("/home/someone/tmp/x.log\n"),
        ])
        changed = notebook.normalize_outputs(path, self.study)
        self.assertTrue(changed)
        self.assertEqual([], verify.check_notebook(self.study, path, check_urls=False))

    def test_normalize_leaves_source_untouched(self) -> None:
        path = write_notebook(
            self.study,
            markdown=f"경로는 {self.study.root} 다.",
            outputs=[stream(f"{self.study.root}\n")],
        )
        notebook.normalize_outputs(path, self.study)
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(str(self.study.root), "".join(document["cells"][0]["source"]))

    def test_normalize_is_idempotent(self) -> None:
        path = write_notebook(self.study, outputs=[stream(f"{self.study.root}\n")])
        self.assertTrue(notebook.normalize_outputs(path, self.study))
        self.assertFalse(notebook.normalize_outputs(path, self.study))


class RealNotebookTests(unittest.TestCase):
    """실제 교재 노트북에도 적용한다. sparse checkout 에서는 건너뛴다."""

    def test_notebooks_have_no_renumber_residue(self) -> None:
        from test_study_materials import STUDIES

        if not STUDIES:
            self.skipTest("study.toml 이 있는 교재가 없다 (sparse checkout)")
        for study in STUDIES:
            for path in verify.find_notebooks(study):
                with self.subTest(study=study.slug, notebook=path.name):
                    failures = verify.check_notebook(study, path, check_urls=False)
                    self.assertEqual([], failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()


class ExecuteThenNormalizeTests(unittest.TestCase):
    """재실행 -> 정규화 -> 검사 순서.

    검사를 먼저 하면 그 실행이 새로 저장한 절대경로를 이번 회차가 보지 못한다.
    게이트가 옳아도 워크플로와 결합하면 false green 이 된다.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.study = write_study(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def _executor_writing_absolute_path(self, path: Path) -> list[str]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["cells"][1]["outputs"] = [stream(f"경고: {self.study.root}/.venv/x.py:1\n")]
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return []

    def test_normalization_runs_after_a_successful_execution(self) -> None:
        path = write_notebook(self.study)
        failures = verify.check_notebook_after_execute(
            self.study, path, self._executor_writing_absolute_path, check_urls=False)
        self.assertEqual([], failures, failures)
        self.assertIn("<교재>", path.read_text(encoding="utf-8"))

    def test_failed_execution_is_reported_and_nothing_is_rewritten(self) -> None:
        path = write_notebook(self.study, outputs=[stream("깨끗하다\n")])
        before = path.read_text(encoding="utf-8")
        failures = verify.check_notebook_after_execute(
            self.study, path, lambda _p: ["재실행 실패 — 커널이 죽었다"], check_urls=False)
        self.assertTrue(any("재실행 실패" in f for f in failures), failures)
        self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_executor_that_writes_then_fails_leaves_the_original(self) -> None:
        """실패한 실행의 부분 출력을 산출물로 남기면 안 된다.

        executor 는 임의의 함수이고 nbconvert 도 판마다 다르게 실패한다. 그래서
        "실패 시 원본 불변" 은 executor 의 예의가 아니라 helper 가 보장해야 한다.
        """
        path = write_notebook(self.study, outputs=[stream("이전 회차 출력\n")])
        before = path.read_text(encoding="utf-8")

        def writes_then_fails(target: Path) -> list[str]:
            document = json.loads(target.read_text(encoding="utf-8"))
            document["cells"][1]["outputs"] = [stream("절반만 돌다 죽었다\n")]
            target.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            return ["재실행 실패 — 커널이 중간에 죽었다"]

        failures = verify.check_notebook_after_execute(
            self.study, path, writes_then_fails, check_urls=False)
        self.assertTrue(any("재실행 실패" in f for f in failures), failures)
        self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_without_executor_it_only_checks(self) -> None:
        path = write_notebook(self.study, outputs=[stream(f"{self.study.root}\n")])
        failures = verify.check_notebook_after_execute(
            self.study, path, None, check_urls=False)
        self.assertTrue(any("절대경로" in f for f in failures), failures)
