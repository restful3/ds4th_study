"""책 리스팅 하나가 실제로 어디 있는지 읽는다.

`study.toml` 의 `[mapping.listings.chXX]` 가 정본이라고 선언해 놓고도, 실제 해석은
노트북마다 각자 구현돼 있었다 — 책 9·11·12장은 `config.resolve_listing()` 을
호출조차 하지 않고 자체 표를 들고 있었고, 넷 다 비슷한 `ast` 헬퍼를 중복 구현했다.
그래서 정본이 정본이 아니었다. 해석을 여기 한 곳으로 모은다.

선언 형식 세 가지.

    "2.1"  = { repo = "2.3" }                                   번호 붙은 리스팅 파일
    "11.2" = { source = "repo-file", path = "...", symbol = "MultiHeadGraphAttention" }
    "9.6"  = { source = "repo-file", path = "...", start = 96, end = 112 }
    "4.20" = { source = "explainer", reason = "book-only" }      정본이 해설판 본문

`symbol` 의 특수값 두 개와 dotted 표기를 지원한다.

    __main__    파일 전체 (스크립트, 또는 확장자 없는 Cypher·프롬프트 파일)
    __entry__   `if __name__ == "__main__":` 블록만
    A.b         클래스 A 안의 메서드 b

줄 번호는 저장하지 않고 매번 `ast` 로 다시 찾는다. 업스트림 파일이 갱신되면 줄 번호는
조용히 어긋나지만 함수·클래스명은 그렇지 않다. `start`/`end` 는 함수 경계와 무관한
조각에만 쓰고 **1-based inclusive** 다.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

#: `explainer` 로 선언할 때 왜 업스트림을 쓸 수 없는지.
EXPLAINER_REASONS = ("book-only", "upstream-missing", "upstream-wrong")

#: 파일 전체 / `if __name__` 블록.
WHOLE_FILE = "__main__"
ENTRY_BLOCK = "__entry__"

#: `ast` 로 심볼을 찾을 수 있는 확장자. 그 외는 통째로 읽는다.
#: 주의 — Path.suffix 를 쓰면 안 된다. 리스팅 파일명이 번호에 점을 포함해
#: Path("12.15 DWPC ...").suffix 가 ".15 DWPC ..." 가 된다. endswith 로 판정한다.
PYTHON_SUFFIX = ".py"


class ListingSpecError(ValueError):
    """선언이 잘못됐거나 가리키는 대상을 읽을 수 없다."""


@dataclass(frozen=True)
class ListingSpec:
    """`[mapping.listings.chXX]` 항목 하나."""

    kind: str                      # "repo" | "repo-file" | "explainer"
    repo_number: str | None = None  # kind == "repo"
    path: str | None = None         # kind == "repo-file"
    symbol: str | None = None
    start: int | None = None
    end: int | None = None
    reason: str | None = None       # kind == "explainer"


@dataclass(frozen=True)
class ListingChunk:
    """읽어낸 소스 조각. start·end 는 1-based inclusive 다."""

    text: str
    path: Path
    start: int
    end: int
    symbol: str | None = None
    #: 챕터 폴더 기준 상대경로. model/·eval/ 처럼 하위 폴더에 있는 리스팅을
    #: path.name 으로만 표기하면 어느 폴더인지 잃는다.
    rel_path: str = ""
    #: `symbol = "__main__"` 으로 파일을 통째로 읽었나.
    #: 줄 범위 선언도 symbol 이 없으므로 이 플래그 없이는 둘을 구분할 수 없다.
    #: ch09 는 9.10 을 파일 전체로, 9.6 을 줄 범위로 선언한다.
    whole_file: bool = False

    def origin(self) -> str:
        """노트북이 출력할 사람용 위치 표기."""
        where = f"{self.rel_path or self.path.name}  L{self.start}-L{self.end}"
        if self.symbol:
            return f"{where}  ({self.symbol})"
        if self.whole_file:
            return f"{where}  (파일 전체)"
        return f"{where}  (줄 범위)"


def parse(raw: dict) -> ListingSpec:
    """study.toml 의 인라인 테이블 하나를 스펙으로 읽는다."""
    if "repo" in raw:
        return ListingSpec(kind="repo", repo_number=str(raw["repo"]))

    source = str(raw.get("source", "explainer"))

    if source == "explainer":
        reason = raw.get("reason")
        if reason is not None and reason not in EXPLAINER_REASONS:
            raise ListingSpecError(
                f"reason '{reason}' 은 유효하지 않다. {', '.join(EXPLAINER_REASONS)} 중 하나여야 한다"
            )
        return ListingSpec(kind="explainer", reason=reason)

    if source != "repo-file":
        raise ListingSpecError(
            f"source '{source}' 을 모른다. repo / repo-file / explainer 중 하나여야 한다"
        )

    path = raw.get("path")
    if not path:
        raise ListingSpecError("repo-file 은 path 가 필요하다")

    symbol = raw.get("symbol")
    start, end = raw.get("start"), raw.get("end")
    if symbol and (start is not None or end is not None):
        raise ListingSpecError(
            f"{path}: symbol 과 start/end 를 함께 쓸 수 없다. 함수·클래스 경계면 symbol, "
            f"그 밖의 조각이면 start/end 하나만 쓴다"
        )
    if not symbol and (start is None or end is None):
        raise ListingSpecError(f"{path}: symbol 또는 start+end 가 필요하다")
    if start is not None:
        start, end = int(start), int(end)
        if start < 1 or end < start:
            raise ListingSpecError(f"{path}: 줄 범위 {start}-{end} 가 뒤집혔거나 1보다 작다")

    return ListingSpec(kind="repo-file", path=str(path), symbol=symbol, start=start, end=end)


def resolve_path(spec: ListingSpec, base: Path) -> Path:
    """`path` 를 챕터 소스 폴더 기준으로 풀고 폴더 밖을 가리키면 거부한다."""
    if spec.path is None:
        raise ListingSpecError("repo-file 이 아니라서 경로가 없다")
    candidate = Path(spec.path)
    if candidate.is_absolute():
        raise ListingSpecError(f"절대경로는 쓸 수 없다: {spec.path}")
    base = base.resolve()
    target = (base / candidate).resolve()
    if not target.is_relative_to(base):
        raise ListingSpecError(f"챕터 폴더를 벗어난다: {spec.path}")
    return target


def read(spec: ListingSpec, base: Path) -> ListingChunk:
    """스펙이 가리키는 소스 조각을 읽는다."""
    if spec.kind != "repo-file":
        raise ListingSpecError(
            f"kind '{spec.kind}' 은 파일에서 읽을 수 없다 — 노트북이 원문을 직접 실어야 한다"
        )
    target = resolve_path(spec, base)
    if not target.is_file():
        raise ListingSpecError(f"파일이 없다: {spec.path}")

    lines = target.read_text(encoding="utf-8").splitlines()

    if spec.start is not None:
        if spec.end > len(lines):
            raise ListingSpecError(
                f"{spec.path}: 줄 범위 {spec.start}-{spec.end} 인데 파일은 {len(lines)}줄이다"
            )
        text = "\n".join(lines[spec.start - 1:spec.end])
        return ListingChunk(text=text, path=target, start=spec.start, end=spec.end,
                            rel_path=spec.path)

    if spec.symbol == WHOLE_FILE:
        return ListingChunk(
            text="\n".join(lines), path=target, start=1, end=max(len(lines), 1),
            whole_file=True, rel_path=spec.path,
        )

    if not target.name.endswith(PYTHON_SUFFIX):
        raise ListingSpecError(
            f"{spec.path}: 파이썬 파일이 아니라 symbol '{spec.symbol}' 을 찾을 수 없다. "
            f"통째로 읽으려면 symbol = \"{WHOLE_FILE}\" 로 두라"
        )

    tree = ast.parse("\n".join(lines))
    node = _find_entry(tree) if spec.symbol == ENTRY_BLOCK else _find_symbol(tree, spec.symbol)
    if node is None:
        what = "if __name__ 블록" if spec.symbol == ENTRY_BLOCK else f"심볼 '{spec.symbol}'"
        raise ListingSpecError(f"{spec.path}: {what} 을 찾지 못했다")

    start, end = node.lineno, node.end_lineno
    return ListingChunk(
        text="\n".join(lines[start - 1:end]), path=target,
        start=start, end=end, symbol=spec.symbol, rel_path=spec.path,
    )


@dataclass(frozen=True)
class CrossReferenceRow:
    """"책 10.4 가 어디 있지?" 를 노트북 안에서 답하는 한 줄."""

    number: str
    kind: str
    where: str
    note: str = ""
    #: 책 리스팅 제목. 원서 md 캡션에서 자동으로 온다. 캡션이 유실된 번호는
    #: study.toml 의 title 선언을 쓴다 (4.8·9.5·10.5 가 그런 경우다).
    title: str = ""


def chunk_for(study, repo_dir: str, book_no: str) -> ListingChunk:
    """책 리스팅 번호 하나를 소스 조각으로. 노트북이 쓰는 진입점이다.

    노트북이 자체 대응표와 `ast` 헬퍼를 두지 않게 하려고 있다. 실제로 책 9·11·12장이
    각자 250줄쯤을 재구현해 study.toml 이 정본이라는 규칙이 이름만 남은 적이 있다.
    """
    raw = study.listing_overrides.get(repo_dir, {}).get(str(book_no))
    if raw is None:
        raise ListingSpecError(
            f"{repo_dir} 의 책 {book_no} 이 study.toml 에 선언되지 않았다 — "
            f"[mapping.listings.{repo_dir}] 에 먼저 적어라"
        )
    spec = parse(raw)
    if spec.kind == "explainer":
        raise ListingSpecError(
            f"책 {book_no} 은 저장소에 코드가 없다 (reason={spec.reason}). "
            f"노트북이 해설판 본문을 실은 마크다운 셀을 보라"
        )
    if spec.kind == "repo":
        from studykit import cypher

        base = study.src_dirs()[repo_dir]
        path = cypher.find(spec.repo_number, base / "listings")
        lines = path.read_text(encoding="utf-8").splitlines()
        return ListingChunk(
            text="\n".join(lines), path=path, start=1, end=max(len(lines), 1),
            whole_file=True,
        )
    return read(spec, study.src_dirs()[repo_dir])


def cross_reference(study, repo_dir: str) -> list[CrossReferenceRow]:
    """선언된 리스팅 전부를 책 번호 순서로. 노트북이 대조표로 출력한다.

    제목은 원서 md 의 `Listing N.M` 캡션에서 자동으로 붙는다. 캡션이 유실된 번호만
    study.toml 에 `title` 을 적으면 되므로, 노트북이 제목 표를 따로 들 필요가 없다.
    """
    from studykit import listing_map

    entries = study.listing_overrides.get(repo_dir, {})
    chapter_dir = listing_map.chapter_dir_for(study, repo_dir)
    titles = listing_map.book_listings(study, chapter_dir) if chapter_dir else {}
    rows: list[CrossReferenceRow] = []
    for number in sorted(entries, key=lambda n: [int(x) for x in n.split(".")]):
        raw = entries[number]
        try:
            spec = parse(raw)
        except ListingSpecError as exc:
            rows.append(CrossReferenceRow(number, "?", "선언 오류", str(exc),
                                         _title(titles, raw, number)))
            continue
        if spec.kind == "explainer":
            rows.append(CrossReferenceRow(number, "explainer", "해설판 본문", spec.reason or "",
                                         _title(titles, raw, number)))
            continue
        if spec.kind == "repo":
            rows.append(CrossReferenceRow(number, "repo", f"listings/{spec.repo_number} …", "",
                                         _title(titles, raw, number)))
            continue
        try:
            chunk = read(spec, study.src_dirs()[repo_dir])
            where = f"{spec.path}  L{chunk.start}-L{chunk.end}"
        except ListingSpecError as exc:
            where, spec_note = f"{spec.path}", str(exc)
            rows.append(CrossReferenceRow(number, "repo-file", where, spec_note,
                                         _title(titles, raw, number)))
            continue
        if spec.symbol == WHOLE_FILE:
            note = "파일 전체"
        elif spec.symbol:
            note = spec.symbol
        else:
            note = "줄 범위 (함수 경계 아님)"
        rows.append(CrossReferenceRow(number, "repo-file", where, note,
                                     _title(titles, raw, number)))
    return rows


def _title(titles: dict, raw: dict, number: str) -> str:
    """study.toml 의 명시 선언 우선, 없으면 원서 캡션.

    명시 선언이 휴리스틱을 이겨야 한다 — 원문 추출이 산문 참조를 캡션으로 오인하는
    사례가 있었고, 그때 사람이 적어 준 값을 덮어쓰면 고칠 방법이 없다.
    """
    return str(raw.get("title") or titles.get(number) or "")


def unnumbered_listings(study, repo_dir: str) -> list[str]:
    """`listings/` 에 있으나 어떤 책 번호도 가리키지 않는 파일.

    스터디원이 폴더를 열면 반드시 마주치므로 노트북이 함께 보여줘야 혼란이 없다.
    """
    base = study.src_dirs().get(repo_dir)
    if base is None or not (base / "listings").is_dir():
        return []
    declared = set()
    for raw in study.listing_overrides.get(repo_dir, {}).values():
        try:
            spec = parse(raw)
        except ListingSpecError:
            continue
        if spec.kind == "repo-file" and spec.path:
            declared.add(Path(spec.path).name)
        elif spec.kind == "repo":
            declared.add(spec.repo_number)
    out = []
    for path in sorted((base / "listings").iterdir()):
        if not path.is_file() or path.name.endswith((".pyc", ".ipynb")):
            continue
        if path.name in declared:
            continue
        if any(path.name.startswith(str(n)) for n in declared):
            continue
        out.append(path.name)
    return out


def _find_symbol(tree: ast.Module, symbol: str) -> ast.AST | None:
    """'alpha' 또는 'Holder.method' 로 정의 노드를 찾는다."""
    parts = symbol.split(".")
    scope: list[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in parts:
        node = next(
            (n for n in scope
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and n.name == part),
            None,
        )
        if node is None:
            return None
        scope = node.body
    return node


def _find_entry(tree: ast.Module) -> ast.AST | None:
    """`if __name__ == "__main__":` 블록. 모듈 최상위에만 있다고 본다."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        for name in ast.walk(node.test):
            if isinstance(name, ast.Name) and name.id == "__name__":
                return node
    return None
