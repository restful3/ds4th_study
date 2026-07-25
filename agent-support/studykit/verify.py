"""검증 게이트.

하나라도 실패하면 산출물은 미완성이다. 여기 있는 검사는 모두 Chapter 3 노트북을
만들며 실제로 발생한 실수에서 나왔다.

  * 링크 응답 코드만 보고 렌더링을 판단했다 -> 그림 참조/내장 개수 대조
  * GitHub URL 을 써서 푸시 전에는 링크가 죽었다 -> 상대경로 대상 존재 확인
  * 리스팅 오프셋을 다른 챕터에서 복사했다 -> 선언값 vs 도출값 대조
  * 업스트림 Makefile 액션을 옮기며 누락 가능 -> 타깃·순서 등가 검증
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from studykit import makefile
from studykit.config import Study

NOTEBOOK_GLOBS = ("chapter_*/src/*/*.ipynb",)

#: 업스트림이 체크포인트로 만드는 것은 검사 대상이 아니다
IGNORED_NOTEBOOK_PARTS = (".ipynb_checkpoints",)


# ---------------------------------------------------------------- 환경
def check_environment(study: Study) -> list[str]:
    """부트스트랩이 만들어야 하는 상태."""
    failures: list[str] = []
    python = study.venv_python()
    if not python.is_relative_to(study.venv):
        failures.append(f".venv 인터프리터가 없다: {study.venv}")
        return failures

    for name in study.shared:
        if not (study.root / name).exists():
            failures.append(
                f"공용 자원 없음: {name} — setup_env.py 를 실행하라"
            )
    if not (study.dataset / ".gitignore").exists():
        failures.append("dataset/.gitignore 없음 — 대용량 데이터가 git 에 올라갈 수 있다")

    # .pth 는 gitignore 대상이라 .venv 를 새로 만들면 사라진다
    pth = list(study.venv.rglob("_studykit.pth"))
    if not pth:
        failures.append(
            "_studykit.pth 없음 — 어느 cwd 에서도 임포트되게 하려면 "
            "setup_env.py 를 다시 실행하라"
        )
    return failures


# ---------------------------------------------------------------- Makefile 등가성
def check_makefile_parity(study: Study) -> list[str]:
    """업스트림 Makefile 의 타깃·액션 순서가 tasks.py 와 같은가."""
    failures: list[str] = []
    src_dirs = study.src_dirs()

    for upstream_dir, path in makefile.find_makefiles(study).items():
        expected = makefile.parse(path)
        # 소스 폴더는 책 장 번호, 업스트림은 MEAP 번호라 이름이 다르다.
        repo_dir = study.src_dir_for_upstream(upstream_dir)
        where = repo_dir if repo_dir == upstream_dir else f"{repo_dir}(업스트림 {upstream_dir})"
        location = src_dirs.get(repo_dir)
        if location is None:
            failures.append(f"{where}: 사본을 찾지 못했다 — [mapping.upstream_dirs] 를 확인하라")
            continue
        tasks_py = location / "tasks.py"
        if not tasks_py.exists():
            failures.append(f"{where}: tasks.py 없음")
            continue

        module = _load_module(tasks_py, f"tasks_{repo_dir}")
        actual = {
            name: [action.signature() for action in actions]
            for name, actions in getattr(module, "TASKS", {}).items()
        }
        if set(actual) != set(expected):
            failures.append(
                f"{where}: 타깃 불일치 Makefile={sorted(expected)} tasks.py={sorted(actual)}"
            )
            continue
        for target in expected:
            if actual[target] != expected[target]:
                failures.append(
                    f"{where}:{target} 액션 불일치\n"
                    f"    Makefile: {expected[target]}\n"
                    f"    tasks.py: {actual[target]}"
                )
    return failures


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- 리스팅
def check_listings(study: Study) -> list[str]:
    """업스트림 리스팅 파일에 0바이트·중복이 있는지 보고한다.

    실패로 처리하지 않는다. 업스트림 문제이고, 노트북이 해설판 본문으로 대체하면
    된다. 다만 조용히 넘기면 빈 결과를 정상으로 오해하게 되므로 반드시 알린다.
    """
    notes: list[str] = []
    for repo_dir, location in sorted(study.src_dirs().items()):
        listings = location / "listings"
        if not listings.is_dir():
            continue
        by_content: dict[bytes, list[str]] = {}
        for path in sorted(listings.iterdir()):
            if not path.is_file():
                continue
            if path.stat().st_size == 0:
                notes.append(f"{repo_dir}: 0바이트 리스팅 — {path.name}")
                continue
            by_content.setdefault(path.read_bytes(), []).append(path.name)
        for names in by_content.values():
            if len(names) > 1:
                notes.append(f"{repo_dir}: 내용이 같은 리스팅 — {', '.join(names)}")
    return notes


# ---------------------------------------------------------------- 노트북
def find_notebooks(study: Study) -> list[Path]:
    found: list[Path] = []
    for pattern in NOTEBOOK_GLOBS:
        for path in sorted(study.root.glob(pattern)):
            if any(part in path.parts for part in IGNORED_NOTEBOOK_PARTS):
                continue
            found.append(path)
    return found


def check_notebook(study: Study, path: Path, check_urls: bool = True) -> list[str]:
    """노트북 규격 검사. 실행은 하지 않는다 (study-verify.py --execute 가 한다)."""
    failures: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path.name}: JSON 파싱 실패 — {exc}"]

    cells = notebook.get("cells", [])
    label = path.name

    # 1) 실행 오류
    errors = [
        output for cell in cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        first = errors[0]
        failures.append(
            f"{label}: 실행 오류 {len(errors)}건 (첫 건: "
            f"{first.get('ename')}: {str(first.get('evalue'))[:80]})"
        )

    markdown = "".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )

    # 2) 그림은 attachment 로 내장한다. 원격 URL 과 상대경로는 둘 다 실패한 이력이 있다.
    stored = sum(len(cell.get("attachments", {}) or {}) for cell in cells)
    referenced = len(re.findall(r"!\[[^\]]*\]\(attachment:", markdown))
    if referenced != stored:
        failures.append(
            f"{label}: 그림 참조 {referenced}개 != attachment {stored}개"
        )
    remote_img = re.findall(r"<img[^>]+src=\"https?://", markdown)
    if remote_img:
        failures.append(
            f"{label}: HTML <img> 원격 URL {len(remote_img)}개 — "
            f"태그를 걸러내는 뷰어에서 안 보인다. attachment 로 내장하라"
        )

    # 3) 상대경로 링크의 대상이 실제로 있는가
    for text, target in re.findall(r"\[([^\]]*)\]\((?!https?:|attachment:|#)([^)]+)\)", markdown):
        resolved = (path.parent / urllib.parse.unquote(target)).resolve()
        if not resolved.exists():
            failures.append(f"{label}: 링크 대상 없음 — [{text}]({target})")

    # 4) 저장소 내부를 절대 URL 로 가리키면 푸시 전에는 죽는다
    repo_urls = re.findall(r"\]\((https://github\.com/[^)]*ds4th_study[^)]*)\)", markdown)
    if repo_urls:
        failures.append(
            f"{label}: 저장소 내부를 GitHub URL 로 가리킨다 ({len(repo_urls)}개) — "
            f"상대경로로 바꿔라. 푸시 전에는 열리지 않는다"
        )

    # 5) 외부 URL 응답 (느리므로 선택)
    if check_urls:
        failures.extend(_check_external_urls(study, markdown, label))

    return failures


def _check_external_urls(study: Study, markdown: str, label: str) -> list[str]:
    failures: list[str] = []
    urls = sorted(set(re.findall(r"\]\((https?://[^)]+)\)", markdown)))
    for url in urls:
        if any(url.startswith(prefix) for prefix in study.url_allow_non_200):
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=25) as response:
                if response.status != 200:
                    failures.append(f"{label}: URL {response.status} — {url}")
        except urllib.error.HTTPError as exc:
            failures.append(
                f"{label}: URL {exc.code} — {url} "
                f"(봇 차단이면 study.toml 의 url_allow_non_200 에 추가하라)"
            )
        except Exception as exc:  # 네트워크 단절 등
            failures.append(f"{label}: URL 확인 실패 — {url} ({type(exc).__name__})")
    return failures

# ---------------------------------------------------------------- 완성 게이트
#: 책 리스팅이 노트북에서 설명되는 방식. 전부 이 중 하나로 분류돼야 한다.
COVERAGE_KINDS = ("executed", "reproduced", "substituted", "documented-only", "optional")

TODO_MARKER = re.compile(r"TODO\(agent\)")


def notebook_status(path: Path) -> str:
    """노트북 metadata 의 status. draft 면 lint 만 통과하면 된다."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unknown"
    return str(notebook.get("metadata", {}).get("studykit", {}).get("status", "complete"))


def validate_nbformat(path: Path, study: Study | None = None) -> list[str]:
    """nbformat 스키마 검증. JSON 파싱만으로는 잡히지 않는 구조 오류를 잡는다.

    게이트는 저장소 python3 로 돌지만 nbformat 은 교재 .venv 에만 있다. 그래서
    .venv 가 있으면 그쪽 인터프리터로 위임한다.
    """
    probe = (
        "import sys,nbformat;"
        "nbformat.validate(nbformat.read(sys.argv[1], as_version=4));"
        "print('ok')"
    )
    interpreters = []
    if study is not None:
        candidate = study.venv_python()
        if candidate.is_relative_to(study.venv):
            interpreters.append(candidate)
    interpreters.append(Path(sys.executable))

    for interpreter in interpreters:
        result = subprocess.run([str(interpreter), "-c", probe, str(path)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return []
        if "ModuleNotFoundError" in (result.stderr or ""):
            continue          # 이 인터프리터엔 nbformat 이 없다 → 다음 것 시도
        tail = (result.stderr or "").strip().splitlines()[-1:]
        return [f"{path.name}: nbformat 검증 실패 — {' '.join(tail)[:140]}"]
    return [f"{path.name}: nbformat 이 없어 스키마 검증을 못 했다 (교재 .venv 를 만들어라)"]


def check_todos(path: Path) -> list[str]:
    """골격 생성기가 남긴 TODO(agent) 가 남아 있으면 미완성이다."""
    text = path.read_text(encoding="utf-8")
    count = len(TODO_MARKER.findall(text))
    return [f"{path.name}: TODO(agent) {count}개 남음 — 서술이 끝나지 않았다"] if count else []


def check_listing_coverage(study: Study, path: Path, repo_dir: str) -> list[str]:
    """책 리스팅이 전부 노트북에서 다뤄졌는지.

    노트북 metadata 의 studykit.listing_coverage 를 정본으로 본다.
    빠진 번호나 분류되지 않은 항목이 있으면 실패다.
    """
    from studykit import listing_map

    chapter_dir = listing_map.chapter_dir_for(study, repo_dir)
    if chapter_dir is None:
        return []
    book = listing_map.book_listings(study, chapter_dir)
    if not book:
        return []

    notebook = json.loads(path.read_text(encoding="utf-8"))
    declared = notebook.get("metadata", {}).get("studykit", {}).get("listing_coverage", {})
    if not declared:
        return [
            f"{path.name}: metadata.studykit.listing_coverage 가 없다 — "
            f"책 리스팅 {len(book)}개가 어떻게 다뤄졌는지 선언해야 한다 "
            f"({', '.join(COVERAGE_KINDS)})"
        ]

    mapped = set(study.listing_overrides.get(repo_dir, {}))
    return check_coverage_numbers(declared, book, mapped, path.name)


def check_coverage_numbers(declared: dict, book: dict, mapped: set, label: str) -> list[str]:
    """선언된 coverage 번호 집합이 옳은지.

    기대 집합은 `book_listings() ∪ study.toml 명시 매핑` 이다. 원서 md 에서 캡션이
    유실되거나 코드 스캔 이미지로 실려 자동 추출이 놓치는 리스팅이 있어서
    (ch09 의 9.5, ch10 의 10.5) 여분 선언을 허용해야 하지만, 아무 여분이나 통과시키면
    오타("12.99")가 조용히 살아남는다. 그래서 study.toml 이 선언한 번호만 허용한다.
    """
    failures: list[str] = []
    order = lambda n: [int(x) for x in n.split(".")]  # noqa: E731

    missing = sorted(set(book) - set(declared), key=order)
    if missing:
        failures.append(f"{label}: 분류되지 않은 책 리스팅 — {', '.join(missing)}")

    unexpected = sorted(set(declared) - set(book) - mapped, key=order)
    if unexpected:
        failures.append(
            f"{label}: 책에도 study.toml 에도 없는 번호를 분류했다 — {', '.join(unexpected)}. "
            f"오타이거나, 실재하는 리스팅이면 [mapping.listings] 에 먼저 선언하라"
        )

    for number, kind in sorted(declared.items(), key=lambda kv: order(kv[0])):
        if kind not in COVERAGE_KINDS:
            failures.append(
                f"{label}: 리스팅 {number} 의 분류 '{kind}' 가 유효하지 않다 "
                f"({', '.join(COVERAGE_KINDS)} 중 하나여야 한다)"
            )
    return failures


def check_declared_listings(study: Study) -> list[str]:
    """study.toml 의 리스팅 선언이 실제로 해결되는가.

    `[mapping.listings.chXX]` 를 정본이라고 선언해 놓고 검증하지 않으면, 경로 오타나
    이름이 바뀐 심볼이 노트북 실행 시점까지 드러나지 않는다.
    """
    from studykit import listing_source

    failures: list[str] = []
    src_dirs = study.src_dirs()
    for repo_dir, entries in sorted(study.listing_overrides.items()):
        base = src_dirs.get(repo_dir)
        if base is None:
            failures.append(f"{repo_dir}: [mapping.listings] 에 있으나 src 폴더가 없다")
            continue
        for number, raw in sorted(entries.items()):
            try:
                spec = listing_source.parse(raw)
            except listing_source.ListingSpecError as exc:
                failures.append(f"{repo_dir} 리스팅 {number}: {exc}")
                continue
            if spec.kind != "repo-file":
                continue
            try:
                listing_source.read(spec, base)
            except listing_source.ListingSpecError as exc:
                failures.append(f"{repo_dir} 리스팅 {number}: {exc}")
    return failures


def check_notebook_complete(study: Study, path: Path, repo_dir: str) -> list[str]:
    """완성 게이트. lint 를 통과한 뒤 완결성까지 본다."""
    failures = validate_nbformat(path, study)
    failures += check_todos(path)
    failures += check_listing_coverage(study, path, repo_dir)
    return failures


def repo_dir_for_notebook(study: Study, path: Path) -> str | None:
    """노트북 경로에서 업스트림 챕터 디렉터리명을 알아낸다."""
    name = path.parent.name
    return name if name in study.src_dirs() else None
