"""교재 설정(study.toml) 로드와 경로 해석.

studykit 은 저장소에 한 벌만 있고 교재는 여러 개다. 그래서 "어느 교재인가"를
매번 알아내야 한다. 노트북과 tasks.py 는 교재 폴더 안에서 실행되므로 cwd 에서
위로 올라가며 study.toml 을 찾는다.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# tomllib 은 3.11+ 에만 있다. 교재 .venv 는 업스트림이 테스트한 3.10 일 수 있어
# tomli 로 폴백한다. bootstrap 이 3.11 미만이면 tomli 를 기반 패키지에 넣는다.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 이하
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "TOML 파서가 없다. Python 3.11+ 를 쓰거나 `pip install tomli` 하라."
        ) from exc

#: 저장소 루트 (agent-support/studykit/config.py 기준)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIG_NAME = "study.toml"

# 업스트림 코드가 임포트하는 공용 자원. 부트스트랩이 code/ 에서 교재 루트로 올린다.
# util/graphdb_base.py 처럼 자기 위치의 상위에서 설정 파일을 읽는 코드가 있어서
# 디렉터리 구조를 업스트림 저장소 루트와 같게 맞춰야 한다.
DEFAULT_SHARED = ("util", "config.ini", "data")


class StudyConfigError(RuntimeError):
    """study.toml 을 찾지 못했거나 필수 키가 빠진 경우."""


@dataclass
class Study:
    """교재 하나의 설정과 경로."""

    root: Path
    slug: str
    title: str
    python: str = "3.10"
    upstream_url: str | None = None
    upstream_dir: str = "code"
    shared: tuple[str, ...] = DEFAULT_SHARED
    base_packages: tuple[str, ...] = ("ipykernel", "jupyterlab")
    kernel_name: str | None = None
    explainer_suffix: str = "_ko_explained.md"
    #: 책 챕터 폴더명 -> 이 저장소의 소스 폴더명 (`chapter_NN/src/<값>`).
    chapter_map: dict[str, str] = field(default_factory=dict)
    #: 이 저장소의 소스 폴더명 -> 업스트림 저장소의 챕터 디렉터리명.
    #: 이 저장소는 소스 폴더를 **책 장 번호** 로 두지만 업스트림은 MEAP 번호를 쓰므로
    #: 둘이 다르다. 업스트림 산출물(Makefile 등)과 짝지으려면 이 표가 필요하다.
    upstream_dirs: dict[str, str] = field(default_factory=dict)
    #: 소스 폴더명 -> 리스팅 minor 번호 오프셋.
    #: **편의용 기본값일 뿐이다.** 책이 코드 없는 리스팅(프롬프트 등)을 중간에
    #: 끼워넣으면 그 뒤가 밀려 단일 오프셋으로 표현할 수 없다 (ch04 의 4.20 이 그렇다).
    #: 정본은 listing_overrides 다.
    listing_offsets: dict[str, int] = field(default_factory=dict)
    #: 소스 폴더명 -> {책 리스팅 번호: {"repo": 파일번호} 또는 {"source": "explainer"}}
    listing_overrides: dict[str, dict[str, dict]] = field(default_factory=dict)
    #: 최종 출간본에 대응 챕터가 없는 MEAP 잔여 디렉터리 (meap-only/ 아래 이름)
    meap_only: tuple[str, ...] = ()
    #: 봇 차단으로 200 이 아니지만 유효한 URL
    url_allow_non_200: tuple[str, ...] = ()
    #: OpenAI 호환 엔드포인트 설정을 담은 파일. **저장소 밖** 경로를 참조만 한다.
    #: 비밀값을 저장소에 복사하지 않기 위한 것이다.
    llm_env_file_raw: str | None = None

    # ---------------- 경로 ----------------
    @property
    def venv(self) -> Path:
        return self.root / ".venv"

    @property
    def upstream(self) -> Path:
        return self.root / self.upstream_dir

    @property
    def dataset(self) -> Path:
        """매니페스트로 내려받는 대용량 데이터."""
        return self.root / "dataset"

    @property
    def data(self) -> Path:
        """업스트림에 동봉된 데이터."""
        return self.root / "data"

    @property
    def config_ini(self) -> Path:
        return self.root / "config.ini"

    @property
    def manifest(self) -> Path:
        return self.root / "data-manifest.toml"

    @property
    def meap_dir(self) -> Path:
        return self.root / "meap-only"

    @property
    def llm_env_file(self) -> Path | None:
        """OpenAI 호환 설정 파일의 절대경로. 상대경로는 저장소 루트 기준."""
        if not self.llm_env_file_raw:
            return None
        candidate = Path(self.llm_env_file_raw).expanduser()
        return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)

    def venv_python(self) -> Path:
        candidate = self.venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return candidate if candidate.exists() else Path(sys.executable)

    def venv_bin(self, name: str) -> Path:
        bindir = self.venv / ("Scripts" if os.name == "nt" else "bin")
        candidate = bindir / (f"{name}.exe" if os.name == "nt" else name)
        return candidate if candidate.exists() else Path(name)

    def resolve(self, relative: str) -> Path:
        """교재 루트 기준 상대경로를 절대경로로."""
        return self.root / normalize_path(relative)

    def chapter_dirs(self) -> list[Path]:
        return sorted(p for p in self.root.glob("chapter_*") if p.is_dir())

    def src_dir_for_upstream(self, upstream_name: str) -> str:
        """업스트림 챕터 디렉터리명 -> 이 저장소의 소스 폴더명.

        선언이 없으면 이름이 같다고 본다 (ch02~ch04 가 그렇다).
        """
        for ours, theirs in self.upstream_dirs.items():
            if theirs == upstream_name:
                return ours
        return upstream_name

    def src_dirs(self) -> dict[str, Path]:
        """소스 폴더명 -> 사본 경로. meap-only 도 포함한다."""
        found: dict[str, Path] = {}
        # 챕터 소스는 chNN, meap-only 는 접두사가 붙는다 (meap-chNN). 소스 폴더가 책 장
        # 번호를 쓰게 되면서 MEAP 잔여물의 업스트림 번호와 겹쳤기 때문이다 — 접두사가
        # 없으면 meap 쪽이 같은 이름의 책 챕터 항목을 덮어쓴다.
        parents = [(c / "src", "ch*") for c in self.chapter_dirs()]
        parents.append((self.meap_dir, "*ch*"))
        for parent, pattern in parents:
            if not parent.is_dir():
                continue
            for path in sorted(parent.glob(pattern)):
                if path.is_dir():
                    found[path.name] = path
        return found

    def resolve_listing(self, repo_dir: str, book_no: str) -> tuple[str, str]:
        """책 리스팅 번호 -> (kind, value).

        kind 는 "repo"(저장소 파일 번호) 또는 "explainer"(해설판 본문에서 가져와야 함).
        listing_overrides 가 정본이고, 없으면 챕터 오프셋으로 계산한다.

        오프셋만으로는 부족한 이유: 최종 교재가 코드 없는 리스팅(LLM 프롬프트 등)을
        중간에 끼워넣으면 그 뒤 번호가 전부 밀린다. ch04 는 책 4.20 이 프롬프트라
        4.21~4.22 가 업스트림 4.20~4.21 에 대응한다.
        """
        override = self.listing_overrides.get(repo_dir, {}).get(str(book_no))
        if override:
            if "repo" in override:
                return ("repo", str(override["repo"]))
            return (str(override.get("source", "explainer")), str(book_no))
        offset = self.listing_offsets.get(repo_dir, 0)
        major, minor = str(book_no).split(".")
        return ("repo", f"{major}.{int(minor) + offset}")

    def explainer_for(self, chapter_dir: Path) -> Path | None:
        """챕터 폴더의 해설판 마크다운."""
        matches = sorted(chapter_dir.glob(f"*{self.explainer_suffix}"))
        return matches[0] if matches else None

    def original_md_for(self, chapter_dir: Path) -> Path | None:
        """챕터 폴더의 원서 원문 마크다운 (해설판·번역본 제외)."""
        for path in sorted(chapter_dir.glob("*.md")):
            name = path.name
            if name.endswith(self.explainer_suffix) or name.endswith("_ko.md"):
                continue
            return path
        return None


def normalize_path(raw: str) -> str:
    """'../../dataset/x/' -> 'dataset/x'. 교재 루트 기준으로 통일한다.

    업스트림 Makefile 이 챕터 디렉터리 기준 상대경로를 쓰기 때문에 필요하다.
    """
    return re.sub(r"^(\.\./)+", "", raw).rstrip("/")


def find_study_root(start: Path | None = None) -> Path:
    """cwd(또는 start)에서 위로 올라가며 study.toml 이 있는 디렉터리를 찾는다."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    raise StudyConfigError(
        f"{CONFIG_NAME} 을 찾지 못했다 (탐색 시작: {current}).\n"
        f"교재 폴더 안에서 실행하거나, 교재 경로를 명시하라."
    )


def load(study_root: Path | str | None = None) -> Study:
    """study.toml 을 읽어 Study 를 만든다. 경로를 주지 않으면 cwd 에서 찾는다."""
    root = Path(study_root).resolve() if study_root else find_study_root()
    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        raise StudyConfigError(f"{config_path} 가 없다")

    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    study = raw.get("study", {})
    for key in ("slug", "title"):
        if key not in study:
            raise StudyConfigError(f"{config_path}: [study].{key} 가 필요하다")

    upstream = raw.get("upstream", {})
    notebook = raw.get("notebook", {})
    llm = raw.get("llm", {})
    mapping = raw.get("mapping", {})

    return Study(
        root=root,
        slug=study["slug"],
        title=study["title"],
        python=str(study.get("python", "3.10")),
        upstream_url=upstream.get("url"),
        upstream_dir=upstream.get("dir", "code"),
        shared=tuple(upstream.get("shared", DEFAULT_SHARED)),
        base_packages=tuple(study.get("base_packages", ("ipykernel", "jupyterlab"))),
        kernel_name=study.get("kernel_name") or study["slug"],
        explainer_suffix=notebook.get("explainer_suffix", "_ko_explained.md"),
        chapter_map=dict(mapping.get("chapters", {})),
        upstream_dirs=dict(mapping.get("upstream_dirs", {})),
        listing_offsets={k: int(v) for k, v in mapping.get("listing_offsets", {}).items()},
        listing_overrides={
            chapter: {str(num): dict(spec) for num, spec in entries.items()}
            for chapter, entries in (mapping.get("listings", {}) or {}).items()
        },
        meap_only=tuple(mapping.get("meap_only", ())),
        url_allow_non_200=tuple(notebook.get("url_allow_non_200", ())),
        llm_env_file_raw=llm.get("env_file"),
    )
