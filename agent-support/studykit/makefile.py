"""업스트림 Makefile 파싱.

교재 저장소는 챕터마다 Makefile 로 init/download/import 를 돌리는 경우가 많다.
그 타깃과 순서를 읽어 Python 태스크로 옮기고(study-map-sources), 옮긴 결과가
원본과 등가인지 검증한다(verify).

액션은 (kind, *args) 튜플로 정규화한다. studykit.actions 의 Action.signature()
와 같은 형태라서 그대로 비교할 수 있다.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path

from studykit.config import normalize_path

# Makefile 변수를 분류용 토큰으로 바꾼다. 실제 경로는 교재 구조에 따라 다르고,
# 우리는 "무엇을 실행하는가"만 비교하면 된다.
VAR_TOKENS = {
    "PIP": "@pip",
    "PYTHON": "@python",
    "STREAMLIT": "@streamlit",
}


class MakefileParseError(RuntimeError):
    """분류하지 못한 레시피 줄. 조용히 넘기면 액션이 누락된다."""


def normalize_url(url: str) -> str:
    """apiKey 값은 환경마다 다르므로 자리표시자로 통일한다."""
    return re.sub(r"apiKey=[^&\"']*", "apiKey=<KEY>", url)


def parse(makefile: Path) -> dict[str, list[tuple]]:
    """Makefile 을 {타깃: [액션 시그니처, ...]} 로 파싱한다."""
    text = makefile.read_text(encoding="utf-8")
    local_vars = dict(
        re.findall(r"^([A-Z_][A-Z0-9_]*)=(.*)$", text, flags=re.MULTILINE)
    )
    text = re.sub(r"\\\n\s*", " ", text)   # 줄 이음

    targets: dict[str, list[tuple]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line.startswith(".PHONY"):
            continue
        is_target = (
            not line.startswith(("\t", " "))
            and ":" in line
            and "=" not in line.split(":")[0]
        )
        if is_target:
            current = line.split(":")[0].strip()
            targets[current] = []
            continue
        if current is None or not line.startswith("\t"):
            continue
        action = parse_recipe(stripped, local_vars, makefile)
        if action is not None:
            targets[current].append(action)
    return targets


def parse_recipe(line: str, local_vars: dict[str, str], source: Path) -> tuple | None:
    """레시피 한 줄 -> 액션 시그니처."""

    def subst(match: re.Match) -> str:
        name = match.group(1)
        return VAR_TOKENS.get(name, local_vars.get(name, "<VAR>"))

    line = re.sub(r"\$\(([A-Z_][A-Z0-9_]*)\)", subst, line)
    tokens = shlex.split(line)
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)          # 앞쪽 환경변수 지정은 시그니처에서 제외
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
    raise MakefileParseError(f"{source}: 분류하지 못한 레시피 — {line!r}")


def find_makefiles(study) -> dict[str, Path]:
    """업스트림 챕터 디렉터리명 -> Makefile 경로."""
    found: dict[str, Path] = {}
    chapters = study.upstream / "chapters"
    if not chapters.is_dir():
        return found
    for makefile in sorted(chapters.glob("*/Makefile")):
        found[makefile.parent.name] = makefile
    return found
