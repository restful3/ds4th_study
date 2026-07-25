"""해설판에서 노트북 골격을 만든다.

ch03·ch04 완성본에서 추출한 공통 척추를 그대로 쓴다.

    제목 + 헤더 표 → 실행 환경 → 환경 점검(code) → 리스팅 헬퍼(code)
    → [절: MD 개념 → CODE 리스팅 실행 (+실습)] 반복 → 요약

기계적으로 채울 수 있는 것은 채운다 — 헤더 표, 절 제목·순서, 리스팅 번호, 실습 자리,
환경 점검·헬퍼 코드. 판단이 필요한 서술은 `TODO(agent)` 로 남긴다.

생성 직후 노트북은 `--lint` 는 통과하고 완성 게이트는 **실패해야** 한다
(TODO 가 남아 있고 listing_coverage 가 선언되지 않았다).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from studykit.config import Study

TODO = "TODO(agent)"

#: 본문 절이 아닌 것 (척추가 따로 처리한다)
NON_BODY = re.compile(r"요약|핵심 용어|References|참고 문헌|이 장에서 다루는|장 소개|^Part")

#: 해설판이 리스팅을 부르는 말. 12개 챕터를 조사한 결과 세 가지가 쓰인다
#: (리스팅 6장 · 리스트 4장 · Listing 6장). 한 챕터가 둘을 섞어 쓰기도 한다.
LISTING_WORD = r"(?:리스팅|리스트|Listing)"


@dataclass
class Section:
    """해설판의 한 절."""

    level: int              # 2 = "## 4.1 ...", 3 = "### 4.2.1 ..."
    number: str | None      # 절 번호 (없을 수 있다)
    title: str
    listings: list[str] = field(default_factory=list)
    exercises: int = 0


def parse_explainer(explainer: Path) -> tuple[str, list[Section]]:
    """해설판에서 (장 제목, 절 목록) 을 뽑는다."""
    chapter_title = ""
    sections: list[Section] = []

    for raw in explainer.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if not chapter_title and line.startswith("# "):
            # 해설판 제목에는 "— 쉬운 해설판" 같은 접미사가 붙는다. 노트북 제목에는
            # 장 제목만 쓴다.
            chapter_title = re.sub(r"\s*—\s*(쉬운\s*)?해설판\s*$", "",
                                   line[2:].strip())
            continue

        if re.match(r"^#{3,4}\s+(연습\s*문제|Exercise)", line):
            if sections:
                sections[-1].exercises += 1
            continue

        # "#### Listing 3.19 ..." / "### 리스팅 5.3 ..." 은 새 절이 아니라
        # 지금 절에 속한 리스팅이다. 해설판이 한글·영문 표기를 섞어 쓴다.
        listing_heading = re.match(
            rf"^#{{2,6}}\s+{LISTING_WORD}\s+(\d+\.\d+)", line)
        if listing_heading:
            _add_listing(sections, listing_heading.group(1))
            continue

        heading = re.match(r"^(#{2,3})\s+(.*)$", line)
        if heading:
            hashes, text = heading.groups()
            numbered = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.*)$", text)
            sections.append(Section(
                level=len(hashes),
                number=numbered.group(1) if numbered else None,
                title=(numbered.group(2) if numbered else text).strip(),
            ))
            continue

        for number in re.findall(rf"{LISTING_WORD}\s+(\d+\.\d+)", line):
            _add_listing(sections, number)

    return chapter_title, sections


def _add_listing(sections: list[Section], number: str) -> None:
    """리스팅 번호를 현재 절에 붙인다. 번호 없는 절(도입 등)은 건너뛴다."""
    for section in reversed(sections):
        if section.number:
            if number not in section.listings:
                section.listings.append(number)
            return


def body_sections(sections: list[Section]) -> list[Section]:
    """본문 절만 남긴다."""
    return [s for s in sections if s.number and not NON_BODY.search(s.title)]


# ------------------------------------------------------------------ 셀
def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


def header_cell(study: Study, book_chapter: int, repo_dir: str,
                chapter_title: str, explainer_name: str,
                sections: list[Section], src_dir: Path | None = None) -> dict:
    """제목 + 헤더 표. 연속된 라벨 줄은 렌더러가 한 단락으로 합치므로 표로 만든다."""
    rows = "\n".join(f"| {s.number} | {s.title} | |" for s in body_sections(sections))
    overrides = study.listing_overrides.get(repo_dir, {})
    if overrides:
        note = (f"명시적 대응표 (`study.toml` 의 `[mapping.listings.{repo_dir}]`) — "
                f"단일 오프셋으로 표현되지 않는 항목 {len(overrides)}건")
    else:
        offset = study.listing_offsets.get(repo_dir, 0)
        note = f"오프셋 **{offset:+d}** — 책 번호 = 파일 번호 − {offset}"

    upstream = study.upstream_url or ""
    tree = upstream.replace(".git", f"/tree/main/chapters/{repo_dir}") if upstream else ""

    # 없는 파일을 링크하면 검증기가 죽은 링크로 잡는다. 챕터마다 구성이 다르다
    # (Makefile 이 없으면 tasks.py 도 없다).
    candidates = ["tasks.py", "listings", "importer", "analysis", "Readme.md"]
    present = [c for c in candidates if src_dir and (src_dir / c).exists()]
    folder_links = " · ".join(f"[{c}](./{c})" for c in present) or "(이 폴더)"

    return markdown(f"""# {book_chapter}장 — {chapter_title}

| | |
| --- | --- |
| **책** | {study.title} **{book_chapter}장** |
| **해설 원본** | [{explainer_name}](../../{explainer_name}) |
| **이 폴더** | {folder_links} |
| **원서 저장소** | [{repo_dir}]({tree}) |
| **리스팅 번호** | {note} |

---

{TODO}: 이 장이 앞 장과 무엇이 다른지, 왜 이 순서인지 2\\~3문장.

| 절 | 내용 | 실습 |
| --- | --- | --- |
{rows}

> **한 문장 요약** — {TODO}
""")


def environment_cells(study: Study) -> list[dict]:
    return [
        markdown(f"""---
## 실행 환경

교재 루트의 `.venv`(Python {study.python}) 커널에서 돌아간다. 처음이면 교재 루트에서
[`setup_env.py`](../../../setup_env.py) 를 한 번 실행한다.

```bash
python3 setup_env.py
```

{TODO}: 이 장이 필요한 외부 서비스(Neo4j 에디션·플러그인·API 키 등)와 그 이유.
에디션 제약이나 플러그인 충돌이 있으면 반드시 적는다 — 없으면 스터디원이 막힌다.
오래 걸리는 단계는 실측 시간과 용량을 적는다.
"""),
        code('''"""환경 점검 — 이 셀이 통과하면 이후 모든 셀을 돌릴 수 있다."""
from pathlib import Path

from studykit import config, cypher

STUDY = config.load()     # 위로 올라가며 study.toml 을 찾는다
HERE = Path.cwd()         # 주피터는 노트북 폴더를 cwd 로 둔다

assert (HERE / "listings").is_dir(), f"리스팅 폴더가 없다. cwd 확인: {HERE}"

print("교재      :", STUDY.title)
print("config.ini:", cypher.neo4j_params())

# TODO(agent): 이 장이 요구하는 것을 검사하고, 실패 시 무엇을 하라고 알려라.
#   예) 에디션 확인, 플러그인 존재 확인, 필요한 데이터베이스 목록
'''),
    ]


def listing_helper_cell(repo_dir: str) -> dict:
    """리스팅 헬퍼. 대응은 study.toml 이 정본이라 오프셋을 하드코딩하지 않는다."""
    return code('''"""리스팅 접근 헬퍼.

리스팅 번호 대응은 study.toml 이 정본이다. 여기에 하드코딩하면 값이 두 곳에 생겨
갈라진다. 최종 교재가 코드 없는 리스팅(프롬프트 등)을 끼워넣으면 단일 오프셋으로는
표현되지 않으므로 resolve_listing() 을 쓴다.
"""
import time

REPO_DIR = "__REPO_DIR__"


def source_of(book_no: str) -> str:
    """책 리스팅 번호로 원문을 가져온다."""
    kind, value = STUDY.resolve_listing(REPO_DIR, book_no)
    if kind != "repo":
        raise KeyError(
            f"책 {book_no} 은 저장소에 코드 파일이 없다 (source={kind}). "
            f"해설판 본문을 실은 마크다운 셀을 보라."
        )
    return cypher.read(value, HERE / "listings").strip()


def show(book_no: str) -> None:
    kind, value = STUDY.resolve_listing(REPO_DIR, book_no)
    origin = f"파일 {value}" if kind == "repo" else "해설판 본문"
    print(f"── 책 Listing {book_no}  ({origin}) ──")
    print(source_of(book_no) if kind == "repo" else "(아래 마크다운 셀에 원문을 실었다)")


def statements(book_no: str) -> list:
    """주석만으로 된 조각은 버린다."""
    out = []
    for chunk in source_of(book_no).split(";"):
        lines = [l for l in chunk.strip().splitlines() if l.strip()]
        if lines and not all(l.strip().startswith("#") for l in lines):
            out.append("\\n".join(l for l in lines if not l.strip().startswith("#")).strip())
    return out


def run(book_no: str, db: str = "neo4j", limit: int = 5, **params):
    """책 리스팅을 실행하고 결과 일부를 보여준다."""
    started = time.time()
    rows = []
    with cypher.driver() as drv, drv.session(database=db) as s:
        for statement in statements(book_no):
            rows = [r.data() for r in s.run(statement, **params)]
    print(f"책 {book_no} [{db}]: {time.time() - started:.1f}초, {len(rows)}행")
    for r in rows[:limit]:
        print("   ", {k: str(v)[:58] for k, v in r.items()})
    return rows


# 책 리스팅 -> 저장소 파일 대조표
print(f"{'책':<8}저장소 파일")
for path in cypher.listings(HERE / "listings"):
    print(f"{path.name.split(' ')[0]:<8}{path.name}")
'''.replace("__REPO_DIR__", repo_dir))


def section_cells(section: Section, book_chapter: int, letters: list[str]) -> list[dict]:
    """본문 절 하나 -> 개념 MD + 리스팅 실행 CODE (+ 실습 쌍)."""
    prefix = "#" * section.level
    cells = [markdown(f"""---
{prefix} {section.number} {section.title}

{TODO}: 이 절이 무엇을 하는지, 왜 이 방법인지. 해설판의 정의·수식·표를 옮기고 앞
절과의 연결을 밝힌다. 그림이 있으면 마크다운 이미지 문법으로 `attachment:` 키
`fig{book_chapter}-N.jpg` 를 참조하고 캡션에 무엇을 보는 그림인지 적는다 —
원격 URL 이나 상대경로가 아니라 attachment 로 내장해야 뷰어에서 보인다.
""")]

    if section.listings:
        shown = section.listings[:4]
        body = "\n".join(f'show("{n}")' for n in shown)
        cells.append(code(f'''"""책 {", ".join(section.listings)} — {TODO}: 이 코드가 무엇을 하는지 한 줄.

{TODO}: 왜 이렇게 생겼는지. 관용구·제약·주의사항. 오래 걸리면 실측 시간을 적는다.
"""
{body}

# TODO(agent): 실제 실행. run("{shown[0]}") 형태로 호출하고 결과를 책의 표·그림과
#   대조해 출력하라. 책과 다르면 왜 다른지 적어라 (데이터 갱신 등).
'''))

    for _ in range(section.exercises):
        letter = letters.pop(0) if letters else "?"
        cells.append(markdown(f"""{prefix} 실습 {book_chapter}-{letter} — 해설판 {section.number} 의 연습문제

> {TODO}: 해설판 연습문제 지문을 그대로 인용한다.

{TODO}: 이 실습이 절의 내용과 어떻게 이어지는지.
"""))
        cells.append(code(f'''"""실습 {book_chapter}-{letter} 풀이 — {TODO}: 무엇을 확인하는가.

{TODO}: 책의 지문을 그대로 풀 수 없으면(데이터가 이미 그 단계를 지났다 등)
어떻게 우회했는지 밝힌다.
"""
# TODO(agent): 구현
'''))

    return cells


def summary_cell(book_chapter: int) -> dict:
    return markdown(f"""---
## 요약

{TODO}: 해설판 요약의 핵심을 옮기고, 이 노트북에서 실제로 본 것과 연결한다.

### 이 노트북에서 실측한 수치

| 항목 | 실측값 | 책 값 |
| --- | --- | --- |
| {TODO} | | |

{TODO}: 책 값과 다르면 왜 다른지 적는다 (데이터 갱신·시드 스냅샷 등).

### 핵심 용어

| 용어 | 뜻 |
| --- | --- |
| {TODO} | |

### 참고 링크

- {TODO}

### 다음 장으로

{TODO}: 다음 장이 방향을 어떻게 바꾸는지. 저장소 디렉터리 번호가 어긋나면 그 근거도.
""")


# ------------------------------------------------------------------ 조립
def build(study: Study, repo_dir: str, output: Path | None = None,
          overwrite: bool = False) -> tuple[Path, dict]:
    """해설판을 읽어 노트북 골격을 만든다. 반환값은 (경로, 통계)."""
    from studykit import listing_map, notebook as notebook_tools

    chapter_dir = listing_map.chapter_dir_for(study, repo_dir)
    if chapter_dir is None:
        raise KeyError(f"{repo_dir} 에 대응하는 책 챕터를 찾지 못했다 (study.toml 확인)")
    src_dir = study.src_dirs().get(repo_dir)
    if src_dir is None:
        raise KeyError(f"{repo_dir} 사본이 없다")
    explainer = study.explainer_for(chapter_dir)
    if explainer is None:
        raise FileNotFoundError(f"{chapter_dir.name} 에 해설판이 없다")

    book_chapter = listing_map.book_chapter_number(chapter_dir)
    target = output or src_dir / f"{book_chapter:02d}_chapter_guide.ipynb"
    if target.exists() and not overwrite:
        raise FileExistsError(f"이미 있다: {target} (덮어쓰려면 --overwrite)")

    chapter_title, sections = parse_explainer(explainer)
    body = body_sections(sections)
    letters = [chr(ord("A") + i) for i in range(sum(s.exercises for s in body))]

    cells = [header_cell(study, book_chapter, repo_dir, chapter_title,
                         explainer.name, sections, src_dir)]
    cells += environment_cells(study)
    cells.append(listing_helper_cell(repo_dir))
    for section in body:
        cells += section_cells(section, book_chapter, letters)
    cells.append(summary_cell(book_chapter))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": study.title, "name": study.kernel_name},
            "language_info": {"name": "python"},
            # draft 이므로 완성 게이트는 lint 만 본다. 서술을 채우고
            # listing_coverage 를 선언한 뒤 status 를 complete 로 바꾼다.
            "studykit": {"status": "draft", "repo_dir": repo_dir},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")

    figures = notebook_tools.figure_map_from_explainer(explainer)
    stats = {
        "cells": len(cells),
        "sections": len(body),
        "exercises": sum(s.exercises for s in body),
        "listings": sum(len(s.listings) for s in body),
        "figures_available": len(figures),
        "todos": sum("".join(c["source"]).count(TODO) for c in cells),
    }
    return target, stats
