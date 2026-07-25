#!/usr/bin/env python3
"""책 리스팅 번호 <-> 저장소 리스팅 파일 대조표를 제목 매칭으로 도출한다.

교재 저장소는 MEAP 번호를 쓰기 때문에 리스팅 번호가 책과 어긋난다. 어긋나는
방식이 챕터마다 다르다 (ch03은 minor 번호가 -3, ch12/ch15는 minor 그대로).
손으로 추측하면 틀리므로, 챕터 원문 md 의 리스팅 제목과 저장소 파일명을
토큰 유사도로 정렬해 오프셋을 기계적으로 찾는다.

    python3 tools/derive_listing_map.py            # 표 출력
    python3 tools/derive_listing_map.py --write    # kgbook/listing_map.py 생성

결과는 눈으로 확인해야 한다. 저장소 파일명은 'create_disease_nodes' 같은
snake_case 이고 책 제목은 산문이므로 매칭이 완벽하지 않을 수 있다.
"""
import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parent.parent

# 책 챕터 폴더 <-> 저장소 챕터 디렉터리.
# 챕터 원문 md 의 고유 키워드 빈도로 검증했다 (Neosemantics/PPI network/openai/fraud 등).
# ch05(miRNA)와 ch06(BBC+spaCy)은 최종 출간본에 대응 챕터가 없는 MEAP 전용이라 제외한다.
CHAPTER_MAP = {
    "chapter_02_intelligent_systems_a_hybrid_approach": ["ch02"],
    "chapter_03_create_your_first_knowledge_graph_from_ontologies": ["ch03"],
    "chapter_04_from_simple_networks_to_multisource_integration": ["ch04"],
    "chapter_05_extracting_domain_specific_knowledge_from_unstructured_data": ["ch07"],
    "chapter_06_building_knowledge_graphs_with_large_language_models": ["ch08"],
    "chapter_07_named_entity_disambiguation": ["ch09"],
    "chapter_08_ned_with_open_llms_and_domain_ontologies": ["ch10"],
    "chapter_09_machine_learning_on_knowledge_graphs_a_primer_approach": ["ch11"],
    "chapter_10_graph_feature_engineering_manual_and_semiautomated_approaches": ["ch12"],
    "chapter_11_graph_representation_learning_and_graph_neural_networks": ["ch13"],
    "chapter_12_node_classification_and_link_prediction_with_gnns": ["ch14"],
    "chapter_13_knowledge_graph_powered_retrieval_augmented_generation": ["ch15"],
    "chapter_15_building_a_qa_agent_with_langgraph": ["ch17"],
}

STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "with", "for", "on", "from",
    "this", "that", "shows", "example", "sample", "using", "code", "is", "are",
    "at", "by", "into", "its", "it", "as", "be", "we", "our",
}


def tokens(text: str) -> set[str]:
    """비교용 토큰 집합. snake_case·CamelCase·산문을 같은 평면으로 내린다."""
    text = re.sub(r"[_\-]+", " ", text.lower())
    text = re.sub(r"\\", "", text)
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def similarity(book_title: str, repo_name: str) -> float:
    bt, rt = tokens(book_title), tokens(repo_name)
    if not bt or not rt:
        return 0.0
    overlap = len(bt & rt) / len(rt)          # 파일명 토큰이 제목에 얼마나 담겼나
    ratio = SequenceMatcher(None, " ".join(sorted(bt)), " ".join(sorted(rt))).ratio()
    return 0.75 * overlap + 0.25 * ratio


def book_listings(chapter_dir: Path) -> dict[str, str]:
    """챕터 영문 원문 md 에서 {리스팅 번호: 제목} 추출."""
    candidates = [
        p for p in chapter_dir.glob("*.md")
        if not p.name.endswith(("_ko.md", "_ko_explained.md"))
    ]
    if not candidates:
        return {}
    text = candidates[0].read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for number, title in re.findall(r"Listing (\d+\.\d+)\s+([^\n]{0,90})", text):
        title = title.strip().rstrip("|").strip()
        # 'Listing 3.18 shows the code to...' 처럼 산문으로 이어지는 경우가 있다
        if number not in found or len(title) > len(found[number]):
            found[number] = title
    return found


def repo_listings(src_dir: Path) -> list[tuple[str, str]]:
    """저장소 listings/ 에서 [(번호, 파일명)] 추출. 번호 순."""
    listings = src_dir / "listings"
    if not listings.is_dir():
        return []
    out = []
    for path in listings.iterdir():
        if not path.is_file():
            continue
        match = re.match(r"(\d+\.\d+)", path.name)
        if match:
            out.append((match.group(1), path.name))
    return sorted(out, key=lambda t: [int(x) for x in t[0].split(".")])


def best_offset(book: dict[str, str], repo: list[tuple[str, str]]) -> tuple[int, float]:
    """저장소 minor 번호에 더해 책 minor 번호가 되는 오프셋을 찾는다."""
    if not book or not repo:
        return 0, 0.0
    scores: dict[int, list[float]] = {}
    for offset in range(-6, 7):
        total = []
        for repo_num, filename in repo:
            major, minor = repo_num.split(".")
            book_num = f"{major}.{int(minor) + offset}"
            # 책 챕터 번호가 다른 경우(ch12 -> 10장)도 minor 만 비교하면 되도록
            # 같은 minor 를 가진 모든 책 리스팅 중 최고점을 쓴다
            matches = [
                similarity(title, filename)
                for num, title in book.items()
                if num.split(".")[1] == book_num.split(".")[1]
            ]
            total.append(max(matches) if matches else 0.0)
        scores[offset] = total
    best = max(scores, key=lambda o: sum(scores[o]))
    mean = sum(scores[best]) / len(scores[best]) if scores[best] else 0.0
    return best, mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="kgbook/listing_map.py 로 저장")
    args = parser.parse_args()

    result: dict[str, dict] = {}

    for chapter_name, repo_dirs in CHAPTER_MAP.items():
        chapter_dir = BOOK_ROOT / chapter_name
        if not chapter_dir.is_dir():
            print(f"건너뜀 (챕터 폴더 없음): {chapter_name}", file=sys.stderr)
            continue
        book = book_listings(chapter_dir)
        book_chapter = int(re.match(r"chapter_(\d+)", chapter_name).group(1))

        for repo_dir_name in repo_dirs:
            src = chapter_dir / "src" / repo_dir_name
            repo = repo_listings(src)
            if not repo:
                print(f"\n### {repo_dir_name}  (책 {book_chapter}장) — listings/ 없음")
                continue
            offset, confidence = best_offset(book, repo)
            print(f"\n### {repo_dir_name}  (책 {book_chapter}장)  "
                  f"minor 오프셋 {offset:+d}  신뢰도 {confidence:.2f}")
            entries = {}
            for repo_num, filename in repo:
                major, minor = repo_num.split(".")
                book_num = f"{book_chapter}.{int(minor) + offset}"
                title = book.get(book_num, "")
                entries[book_num] = filename
                flag = " " if title else "?"
                print(f"  {flag} 책 {book_num:<6} <- {filename[:52]:<52} "
                      f"{title[:44]}")
            result[repo_dir_name] = {
                "book_chapter": book_chapter,
                "offset": offset,
                "confidence": round(confidence, 3),
                "entries": entries,
            }

    if args.write:
        target = BOOK_ROOT / "kgbook" / "listing_map.py"
        lines = [
            '"""책 리스팅 번호 -> 저장소 리스팅 파일명 대조표.',
            "",
            "tools/derive_listing_map.py 가 챕터 원문 md 의 리스팅 제목과",
            "저장소 파일명을 매칭해 생성한다. 직접 수정하지 말고 스크립트를 다시 돌려라.",
            '"""',
            "",
            "LISTING_MAP = {",
        ]
        for repo_dir, info in result.items():
            lines.append(f"    {repo_dir!r}: {{")
            lines.append(f"        'book_chapter': {info['book_chapter']},")
            lines.append(f"        'offset': {info['offset']},")
            lines.append("        'entries': {")
            for book_num, filename in info["entries"].items():
                lines.append(f"            {book_num!r}: {filename!r},")
            lines.append("        },")
            lines.append("    },")
        lines.append("}")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n작성: {target.relative_to(BOOK_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
