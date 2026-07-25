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

    for repo_dir, path in makefile.find_makefiles(study).items():
        expected = makefile.parse(path)
        location = src_dirs.get(repo_dir)
        if location is None:
            failures.append(f"{repo_dir}: 사본을 찾지 못했다 (study-map-sources 실행 필요)")
            continue
        tasks_py = location / "tasks.py"
        if not tasks_py.exists():
            failures.append(f"{repo_dir}: tasks.py 없음")
            continue

        module = _load_module(tasks_py, f"tasks_{repo_dir}")
        actual = {
            name: [action.signature() for action in actions]
            for name, actions in getattr(module, "TASKS", {}).items()
        }
        if set(actual) != set(expected):
            failures.append(
                f"{repo_dir}: 타깃 불일치 Makefile={sorted(expected)} tasks.py={sorted(actual)}"
            )
            continue
        for target in expected:
            if actual[target] != expected[target]:
                failures.append(
                    f"{repo_dir}:{target} 액션 불일치\n"
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
