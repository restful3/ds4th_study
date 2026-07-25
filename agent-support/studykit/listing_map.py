"""챕터↔소스 대응과 리스팅 번호 오프셋을 도출한다.

업스트림 저장소가 MEAP 번호를 쓰면 디렉터리 이름과 책 챕터가 어긋난다.
**디렉터리 개수와 챕터 수를 맞춰 짝지으면 틀린다.** 실제로 그렇게 해서
ch05(miRNA)를 책 4장에, ch06(BBC+spaCy)을 책 5장에 잘못 넣었다.

여기서는 두 가지를 기계적으로 도출한다.

1. 챕터 대응 — 소스 디렉터리의 특징 토큰이 어느 챕터 원문에 몰려 나오는지 센다.
2. 리스팅 오프셋 — 책 리스팅 **제목** 과 저장소 파일명의 토큰 유사도. 오프셋은
   챕터마다 다르다 (3장 −3, 4장 0). 한 챕터 값을 다른 챕터에 쓰면 안 된다.

도출값은 사람이 확인해야 한다. 신뢰도가 낮으면 호출자가 실패로 처리한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from studykit.config import Study

#: 오프셋 도출 신뢰도 하한. 이보다 낮으면 사람의 확인을 요구한다.
MIN_OFFSET_CONFIDENCE = 0.30

#: 챕터 대응 신뢰도 하한 (최상위 챕터의 점유율)
MIN_MAPPING_SHARE = 0.60

STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "with", "for", "on", "from",
    "this", "that", "shows", "example", "sample", "using", "code", "is", "are",
    "at", "by", "into", "its", "it", "as", "be", "we", "our",
}


@dataclass
class DerivedOffset:
    repo_dir: str
    offset: int
    confidence: float
    #: 책 리스팅 번호 -> 저장소 파일명
    entries: dict[str, str]


def tokens(text: str) -> set[str]:
    """비교용 토큰. snake_case·CamelCase·산문을 같은 평면으로 내린다."""
    text = re.sub(r"[_\-]+", " ", text.lower()).replace("\\", "")
    return {w for w in re.findall(r"[a-z0-9]+", text)
            if w not in STOPWORDS and len(w) > 2}


def similarity(book_title: str, repo_name: str) -> float:
    book, repo = tokens(book_title), tokens(repo_name)
    if not book or not repo:
        return 0.0
    overlap = len(book & repo) / len(repo)
    ratio = SequenceMatcher(None, " ".join(sorted(book)), " ".join(sorted(repo))).ratio()
    return 0.75 * overlap + 0.25 * ratio


def book_listings(study: Study, chapter_dir: Path) -> dict[str, str]:
    """챕터 원문 md 에서 {리스팅 번호: 제목} 추출."""
    original = study.original_md_for(chapter_dir)
    if original is None:
        return {}
    text = original.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for number, title in re.findall(r"Listing (\d+\.\d+)\s+([^\n]{0,90})", text):
        title = title.strip().rstrip("|").strip()
        if number not in found or len(title) > len(found[number]):
            found[number] = title
    return found


def repo_listings(src_dir: Path) -> list[tuple[str, str]]:
    """저장소 listings/ 에서 [(번호, 파일명)]. 번호 순."""
    listings = src_dir / "listings"
    if not listings.is_dir():
        return []
    out = []
    for path in sorted(listings.iterdir()):
        if not path.is_file():
            continue
        match = re.match(r"(\d+\.\d+)", path.name)
        if match:
            out.append((match.group(1), path.name))
    return sorted(out, key=lambda t: [int(x) for x in t[0].split(".")])


def chapter_dir_for(study: Study, repo_dir: str) -> Path | None:
    """업스트림 디렉터리명에 대응하는 책 챕터 폴더."""
    for chapter_name, mapped in study.chapter_map.items():
        if mapped == repo_dir:
            return study.root / chapter_name
    return None


def book_chapter_number(chapter_dir: Path) -> int | None:
    match = re.match(r"chapter_(\d+)", chapter_dir.name)
    return int(match.group(1)) if match else None


def derive_offset(study: Study, repo_dir: str) -> DerivedOffset | None:
    """리스팅 minor 번호 오프셋을 제목 매칭으로 도출한다.

    책 번호 = 파일 번호 − offset. 리스팅이나 원문이 없으면 None.
    """
    chapter_dir = chapter_dir_for(study, repo_dir)
    src_dir = study.src_dirs().get(repo_dir)
    if chapter_dir is None or src_dir is None:
        return None

    repo = repo_listings(src_dir)
    book = book_listings(study, chapter_dir)
    chapter_number = book_chapter_number(chapter_dir)
    if not repo or not book or chapter_number is None:
        return None

    scores: dict[int, list[float]] = {}
    for offset in range(-6, 7):
        totals = []
        for repo_num, filename in repo:
            minor = int(repo_num.split(".")[1]) - offset
            candidates = [
                similarity(title, filename)
                for num, title in book.items()
                if int(num.split(".")[1]) == minor
            ]
            totals.append(max(candidates) if candidates else 0.0)
        scores[offset] = totals

    best = max(scores, key=lambda o: sum(scores[o]))
    values = scores[best]
    confidence = sum(values) / len(values) if values else 0.0

    entries = {
        f"{chapter_number}.{int(repo_num.split('.')[1]) - best}": filename
        for repo_num, filename in repo
    }
    return DerivedOffset(repo_dir=repo_dir, offset=best,
                         confidence=round(confidence, 3), entries=entries)


# ------------------------------------------------------------ 챕터 대응 검증
def signature_tokens(src_dir: Path, limit: int = 40) -> list[str]:
    """소스 디렉터리의 특징 토큰. 파일명(확장자 제외)에서 뽑는다."""
    collected: dict[str, int] = {}
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        for token in tokens(path.stem):
            collected[token] = collected.get(token, 0) + 1
    return sorted(collected, key=lambda t: (-collected[t], t))[:limit]


def score_chapters(study: Study, repo_dir: str) -> list[tuple[str, float]]:
    """특징 토큰이 어느 책 챕터 원문에 몰려 나오는가. [(챕터폴더명, 점유율)] 내림차순."""
    src_dir = study.src_dirs().get(repo_dir)
    if src_dir is None:
        return []
    marks = signature_tokens(src_dir)
    if not marks:
        return []

    counts: dict[str, int] = {}
    for chapter_dir in study.chapter_dirs():
        original = study.original_md_for(chapter_dir)
        if original is None:
            continue
        text = original.read_text(encoding="utf-8").lower()
        counts[chapter_dir.name] = sum(text.count(token) for token in marks)

    total = sum(counts.values())
    if not total:
        return []
    return sorted(((name, hits / total) for name, hits in counts.items()),
                  key=lambda pair: -pair[1])


def verify_mapping(study: Study) -> list[str]:
    """study.toml 의 챕터 대응이 키워드 빈도와 어긋나는지 보고한다."""
    problems: list[str] = []
    for chapter_name, repo_dir in sorted(study.chapter_map.items()):
        ranked = score_chapters(study, repo_dir)
        if not ranked:
            continue
        top_name, top_share = ranked[0]
        if top_name != chapter_name and top_share >= MIN_MAPPING_SHARE:
            problems.append(
                f"{repo_dir}: study.toml 은 {chapter_name} 이라 하는데 "
                f"키워드는 {top_name} 에 몰린다 (점유율 {top_share:.0%}). "
                f"개수로 짝짓지 말고 내용으로 확인하라"
            )
    return problems
