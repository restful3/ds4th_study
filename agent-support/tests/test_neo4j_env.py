"""교재 루트 neo4j_env.py 의 컨테이너 전환 규칙 검사.

Knowledge Graphs and LLMs in Action 은 3장이 Community + `n10s` 를, 4장 이후가
Enterprise 를 요구한다. 두 요구는 배타적이다 — `n10s` 를 Enterprise 에 넣으면
`NoSuchMethodError: JsonProperty.isRequired()` 로 기동에 실패한다. 그래서 컨테이너를
둘로 나눠 번갈아 띄우는데, 전환을 손으로 하면 두 가지가 조용히 어긋난다.

  1. `docker run` 정의가 노트북 마크다운에 복제되어 실제 컨테이너와 드리프트한다
  2. 정지된 컨테이너를 두고 "플러그인 설정을 확인하라" 는 엉뚱한 처방을 받는다

여기서 검사하는 것은 전환 규칙의 순수 부분이다. docker 를 호출하지 않는다.

    python3 -m unittest discover -s agent-support/tests -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

SCRIPT_NAME = "neo4j_env.py"
# 컨테이너 전환이 필요한 교재. 다른 교재는 외부 컨테이너를 요구하지 않으므로
# 이 스크립트를 두지 않는다 — 있어야 하는 곳에서만 부재를 실패로 본다.
BOOK_MARKER = "chapter_03_create_your_first_knowledge_graph_from_ontologies"


def find_book() -> Path | None:
    """source/ 아래에서 전환이 필요한 교재 루트를 찾는다."""
    if not (REPO_ROOT / "source").is_dir():
        return None
    return next(iter(sorted((REPO_ROOT / "source").glob(f"*/{BOOK_MARKER}"))), None)


BOOK = find_book()
BOOK_ROOT = BOOK.parent if BOOK is not None else None
# 교재가 archive/ 로 옮겨가면 skip 하고, source/ 에 있으면 스크립트 부재를 실패로 잡는다.
SCRIPT = BOOK_ROOT / SCRIPT_NAME if BOOK_ROOT is not None else None


@unittest.skipIf(BOOK_ROOT is None, "전환이 필요한 교재가 source/ 에 없다")
class ScriptPresenceTests(unittest.TestCase):
    def test_script_exists_next_to_setup_env(self) -> None:
        self.assertTrue(
            SCRIPT.exists(),
            f"{SCRIPT.relative_to(REPO_ROOT)} 가 없다 — 컨테이너 전환 정본이 필요하다",
        )
        self.assertTrue((BOOK_ROOT / "setup_env.py").exists())


def load_module(path: Path):
    """교재 루트의 스크립트를 모듈로 읽어들인다.

    `from __future__ import annotations` 를 쓰는 모듈의 dataclass 는 애노테이션을
    풀 때 자기 모듈을 sys.modules 에서 찾으므로, exec 전에 등록해야 한다.
    """
    name = "neo4j_env_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(SCRIPT is None or not SCRIPT.exists(), f"{SCRIPT_NAME} 가 없다")
class ProfileDeclarationTests(unittest.TestCase):
    """프로파일 선언 자체의 불변식."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module(SCRIPT)

    def test_two_profiles_named_by_edition(self) -> None:
        self.assertEqual({"community", "enterprise"}, set(self.mod.PROFILES))

    def test_n10s_and_enterprise_never_coexist(self) -> None:
        """이 교재의 핵심 제약. 위반하면 컨테이너가 기동조차 못 한다."""
        for name, profile in self.mod.PROFILES.items():
            with self.subTest(profile=name):
                if "enterprise" in profile.image:
                    self.assertNotIn("n10s", profile.plugins)
                if "n10s" in profile.plugins:
                    self.assertNotIn("enterprise", profile.image)

    def test_community_carries_n10s(self) -> None:
        community = self.mod.PROFILES["community"]
        self.assertIn("n10s", community.plugins)
        self.assertIn("n10s.*", community.unrestricted)

    def test_profiles_use_distinct_names_and_volumes(self) -> None:
        """같은 볼륨을 공유하면 전환이 데이터를 덮어쓴다."""
        names = [p.container for p in self.mod.PROFILES.values()]
        volumes = [p.data_volume for p in self.mod.PROFILES.values()]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(volumes), len(set(volumes)))

    def test_profiles_share_ports_so_swapping_is_mandatory(self) -> None:
        """포트가 같다는 사실이 '둘 중 하나만' 규칙의 근거다."""
        ports = {tuple(p.ports) for p in self.mod.PROFILES.values()}
        self.assertEqual(1, len(ports), "포트가 다르면 스왑할 필요가 없다")

    def test_only_enterprise_requires_licence(self) -> None:
        self.assertTrue(self.mod.PROFILES["enterprise"].needs_licence)
        self.assertFalse(self.mod.PROFILES["community"].needs_licence)


@unittest.skipIf(SCRIPT is None or not SCRIPT.exists(), f"{SCRIPT_NAME} 가 없다")
class ProfileForTargetTests(unittest.TestCase):
    """장 번호·에디션 이름 → 프로파일."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module(SCRIPT)

    def test_chapter_three_is_the_only_community_chapter(self) -> None:
        self.assertEqual("community", self.mod.profile_for("ch03").name)

    def test_enterprise_chapters(self) -> None:
        for chapter in ("ch04", "ch06", "ch10", "ch13", "ch15"):
            with self.subTest(chapter=chapter):
                self.assertEqual("enterprise", self.mod.profile_for(chapter).name)

    def test_chapters_without_neo4j_resolve_to_none(self) -> None:
        for chapter in ("ch02", "ch05", "ch09", "ch11", "ch12"):
            with self.subTest(chapter=chapter):
                self.assertIsNone(self.mod.profile_for(chapter))

    def test_draft_chapters_raise_instead_of_guessing(self) -> None:
        """요구사항이 아직 선언되지 않은 장은 추측하지 않는다."""
        for chapter in ("ch07", "ch08"):
            with self.subTest(chapter=chapter):
                with self.assertRaises(self.mod.UndeclaredChapter):
                    self.mod.profile_for(chapter)

    def test_bare_number_and_edition_name_accepted(self) -> None:
        self.assertEqual("community", self.mod.profile_for("3").name)
        self.assertEqual("community", self.mod.profile_for("community").name)
        self.assertEqual("enterprise", self.mod.profile_for("enterprise").name)

    def test_unknown_target_raises(self) -> None:
        with self.assertRaises(self.mod.UnknownTarget):
            self.mod.profile_for("ch99")


@unittest.skipIf(SCRIPT is None or not SCRIPT.exists(), f"{SCRIPT_NAME} 가 없다")
class DockerRunArgvTests(unittest.TestCase):
    """docker run 정의의 정본 — 노트북 마크다운이 아니라 이쪽이 기준이다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module(SCRIPT)

    def argv(self, profile_name: str) -> list[str]:
        return self.mod.docker_run_argv(self.mod.PROFILES[profile_name])

    def test_detached_named_and_imaged(self) -> None:
        argv = self.argv("community")
        self.assertEqual(["docker", "run", "-d"], argv[:3])
        self.assertIn("--name", argv)
        self.assertEqual("kglm-neo4j", argv[argv.index("--name") + 1])
        self.assertEqual("neo4j:5.26", argv[-1])

    def test_ports_bound_to_loopback_only(self) -> None:
        """외부에 노출하지 않는다."""
        for name in self.mod.PROFILES:
            with self.subTest(profile=name):
                published = [a for a in self.argv(name) if a.count(":") == 2]
                self.assertTrue(published)
                for spec in published:
                    self.assertTrue(spec.startswith("127.0.0.1:"), spec)

    def test_community_plugins_and_allowlist(self) -> None:
        argv = self.argv("community")
        joined = " ".join(argv)
        self.assertIn('NEO4J_PLUGINS=["graph-data-science","n10s","apoc"]', joined)
        self.assertIn("NEO4J_dbms_security_procedures_allowlist=gds.*,apoc.*,n10s.*", joined)

    def test_community_never_accepts_enterprise_licence(self) -> None:
        self.assertNotIn("NEO4J_ACCEPT_LICENSE_AGREEMENT", " ".join(self.argv("community")))

    def test_enterprise_licence_and_seed_provider(self) -> None:
        joined = " ".join(self.argv("enterprise"))
        self.assertIn("NEO4J_ACCEPT_LICENSE_AGREEMENT=eval", joined)
        self.assertIn(
            "NEO4J_dbms_databases_seed__from__uri__providers=URLConnectionSeedProvider",
            joined,
            "4.6·15장의 CREATE DATABASE ... seedUri 가 이 설정을 요구한다",
        )

    def test_data_volume_mounted_at_data(self) -> None:
        for name, profile in self.mod.PROFILES.items():
            with self.subTest(profile=name):
                self.assertIn(f"{profile.data_volume}:/data", self.argv(name))

    def test_memory_settings_present(self) -> None:
        """실제로 검증된 컨테이너가 갖고 있던 값 — 문서 쪽이 누락해 드리프트했다."""
        for name in self.mod.PROFILES:
            with self.subTest(profile=name):
                joined = " ".join(self.argv(name))
                self.assertIn("NEO4J_server_memory_heap_max__size=2G", joined)
                self.assertIn("NEO4J_server_memory_pagecache_size=1G", joined)


@unittest.skipIf(SCRIPT is None or not SCRIPT.exists(), f"{SCRIPT_NAME} 가 없다")
class PlanTests(unittest.TestCase):
    """현재 docker 상태 + 목표 → 실행할 단계."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module(SCRIPT)

    def plan(self, target: str, state: dict, *, accept_licence: bool = False):
        return self.mod.plan(
            self.mod.PROFILES[target], state, accept_licence=accept_licence
        )

    def test_target_already_running_is_a_noop(self) -> None:
        state = {"kglm-neo4j": "running", "kglm-neo4j-ee": "exited"}
        steps = self.plan("community", state)
        self.assertEqual([], steps)

    def test_stopped_target_is_started_and_other_stopped_first(self) -> None:
        state = {"kglm-neo4j": "exited", "kglm-neo4j-ee": "running"}
        steps = self.plan("community", state)
        self.assertEqual(
            [("stop", "kglm-neo4j-ee"), ("start", "kglm-neo4j")],
            [(s.action, s.container) for s in steps],
        )

    def test_missing_target_is_created(self) -> None:
        state = {"kglm-neo4j-ee": "running"}
        steps = self.plan("community", state)
        self.assertEqual(
            [("stop", "kglm-neo4j-ee"), ("create", "kglm-neo4j")],
            [(s.action, s.container) for s in steps],
        )

    def test_creating_enterprise_requires_explicit_licence_consent(self) -> None:
        """평가 라이선스 동의는 사용자가 직접 한다."""
        with self.assertRaises(self.mod.LicenceNotAccepted):
            self.plan("enterprise", {}, accept_licence=False)

    def test_creating_enterprise_with_consent_is_allowed(self) -> None:
        steps = self.plan("enterprise", {}, accept_licence=True)
        self.assertEqual([("create", "kglm-neo4j-ee")],
                         [(s.action, s.container) for s in steps])

    def test_starting_existing_enterprise_needs_no_consent(self) -> None:
        """이미 만들어진 컨테이너는 생성 시점에 동의가 끝났다."""
        steps = self.plan("enterprise", {"kglm-neo4j-ee": "exited"}, accept_licence=False)
        self.assertEqual([("start", "kglm-neo4j-ee")],
                         [(s.action, s.container) for s in steps])

    def test_other_profile_absent_needs_no_stop(self) -> None:
        steps = self.plan("community", {"kglm-neo4j": "exited"})
        self.assertEqual([("start", "kglm-neo4j")],
                         [(s.action, s.container) for s in steps])

    def test_create_step_carries_full_run_argv(self) -> None:
        steps = self.plan("community", {})
        self.assertEqual(1, len(steps))
        self.assertEqual(self.mod.docker_run_argv(self.mod.PROFILES["community"]),
                         steps[0].argv)


if __name__ == "__main__":
    unittest.main()
