"""업스트림 Makefile과 Python tasks.py의 등가성 검증.

교재 저장소(code/chapters/chXX/Makefile)를 파싱해 타깃별 액션 순서를 추출하고,
각 챕터 src 폴더의 tasks.py가 선언한 TASKS와 대조한다.
타깃 누락, 액션 누락, 순서 변경, 경로 오타를 잡는 것이 목적이다.

실행:
    .venv/bin/python tests/test_makefile_parity.py
"""
import importlib.util
import re
import shlex
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = BOOK_ROOT / "code" / "chapters"

# 소스 폴더명 -> 사본 경로.
# 책 챕터에 대응하는 것은 chapter_*/src/ 밑에 **책 장 번호** 로 두고, 최종 출간본에
# 대응 챕터가 없는 MEAP 전용(업스트림 ch05·ch06)은 meap-only/meap-chNN 으로 둔다.
# 업스트림 디렉터리명과의 대응은 study.toml 의 [mapping.upstream_dirs] 가 정본이다.
SRC_DIRS = {
    d.name: d
    for parent in [*sorted(BOOK_ROOT.glob("chapter_*/src")), BOOK_ROOT / "meap-only"]
    for d in sorted(parent.glob("ch*"))
    if d.is_dir()
}

# Makefile 변수 치환값. 실제 값이 아니라 분류용 토큰으로 바꾼다.
VAR_TOKENS = {
    "PIP": "@pip",
    "PYTHON": "@python",
    "STREAMLIT": "@streamlit",
}


def normalize_path(raw: str) -> str:
    """'../../dataset/x/' -> 'dataset/x' (책 루트 기준 상대경로)."""
    return re.sub(r"^(\.\./)+", "", raw).rstrip("/")


def normalize_url(url: str) -> str:
    """apiKey 값은 환경마다 달라지므로 자리표시자로 통일."""
    return re.sub(r"apiKey=[^&\"']*", "apiKey=<KEY>", url)


def read_makefile_targets(makefile: Path) -> dict[str, list[tuple]]:
    """Makefile을 {타깃: [액션 시그니처, ...]} 로 파싱."""
    text = makefile.read_text(encoding="utf-8")
    # 명시적 변수 정의 수집 (OPENAI_KEY 등)
    local_vars = dict(
        re.findall(r"^([A-Z_][A-Z0-9_]*)=(.*)$", text, flags=re.MULTILINE)
    )

    # 줄 이음(\) 처리
    text = re.sub(r"\\\n\s*", " ", text)

    targets: dict[str, list[tuple]] = {}
    current = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith(".PHONY"):
            continue
        if not line.startswith(("\t", " ")) and ":" in line and "=" not in line.split(":")[0]:
            current = line.split(":")[0].strip()
            targets[current] = []
            continue
        if current is None or not line.startswith("\t"):
            continue
        action = parse_recipe_line(line.strip(), local_vars)
        if action is not None:
            targets[current].append(action)
    return targets


def parse_recipe_line(line: str, local_vars: dict[str, str]) -> tuple | None:
    """Makefile 레시피 한 줄을 액션 시그니처 튜플로 변환."""

    def subst(match: re.Match) -> str:
        name = match.group(1)
        if name in VAR_TOKENS:
            return VAR_TOKENS[name]
        return local_vars.get(name, "<VAR>")

    line = re.sub(r"\$\(([A-Z_][A-Z0-9_]*)\)", subst, line)
    tokens = shlex.split(line)
    # 앞쪽 환경변수 지정(KEY=value)은 시그니처에서 제외
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None

    head, rest = tokens[0], tokens[1:]

    if head == "@pip" and rest[:2] == ["install", "-r"]:
        return ("pip", rest[2])
    if head == "@python" and rest[:1] == ["-m"]:
        return ("py-m", " ".join(rest[1:]))
    if head == "@python":
        return ("py", rest[0])
    if head == "@streamlit" and rest[:1] == ["run"]:
        return ("streamlit", rest[1])
    if head == "mkdir":
        return ("mkdir", normalize_path(rest[-1]))
    if head == "curl":
        url = next(t for t in rest if t.startswith(("http://", "https://")))
        dest = rest[rest.index("-o") + 1]
        return ("curl", normalize_url(url), normalize_path(dest))
    if head == "unzip":
        dest = rest[rest.index("-d") + 1] if "-d" in rest else None
        archive = next(t for t in rest if t.endswith(".zip"))
        return ("unzip", normalize_path(archive), normalize_path(dest) if dest else None)
    if head == "gunzip":
        return ("gunzip", normalize_path(rest[-1]))
    if head == "mv":
        return ("mv", normalize_path(rest[-2]), normalize_path(rest[-1]))
    raise AssertionError(f"분류하지 못한 Makefile 레시피: {line!r}")


def load_tasks_module(tasks_py: Path):
    spec = importlib.util.spec_from_file_location(
        f"tasks_{tasks_py.parent.name}", tasks_py
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    makefiles = sorted(UPSTREAM.glob("ch*/Makefile"))
    assert makefiles, f"업스트림 Makefile을 찾지 못했다: {UPSTREAM}"

    failures: list[str] = []
    checked = 0

    for makefile in makefiles:
        chapter = makefile.parent.name
        expected = read_makefile_targets(makefile)
        src_dir = SRC_DIRS.get(chapter)

        if src_dir is None:
            failures.append(f"{chapter}: src 복사본을 찾지 못했다")
            continue

        tasks_py = src_dir / "tasks.py"
        if not tasks_py.exists():
            failures.append(f"{chapter}: {tasks_py.relative_to(BOOK_ROOT)} 없음")
            continue

        module = load_tasks_module(tasks_py)
        actual = {
            name: [action.signature() for action in actions]
            for name, actions in module.TASKS.items()
        }

        if set(actual) != set(expected):
            failures.append(
                f"{chapter}: 타깃 불일치 "
                f"Makefile={sorted(expected)} tasks.py={sorted(actual)}"
            )
            continue

        for target in expected:
            if actual[target] != expected[target]:
                failures.append(
                    f"{chapter}:{target} 액션 불일치\n"
                    f"    Makefile: {expected[target]}\n"
                    f"    tasks.py: {actual[target]}"
                )
        checked += 1

    print(f"검사한 챕터: {checked}/{len(makefiles)}")
    if failures:
        print(f"\n실패 {len(failures)}건:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("모든 Makefile 타깃이 tasks.py와 일치한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
