#!/usr/bin/env python3
"""챕터 해설 노트북의 골격을 만든다.

해설판의 절 구성·리스팅 번호·연습문제를 읽어 셀 골격을 생성하고, 판단이 필요한
서술은 TODO(agent) 로 남긴다. 에이전트가 그 자리를 채우고 그림을 내장한 뒤
metadata 를 complete 로 바꾸면 완성 게이트가 검사한다.

    python3 agent-support/scripts/study-new-notebook.py <교재경로> <업스트림디렉터리>
    python3 agent-support/scripts/study-new-notebook.py <교재경로> ch05 --dry-run
    python3 agent-support/scripts/study-new-notebook.py <교재경로> --list
    python3 agent-support/scripts/study-new-notebook.py <교재경로> ch05 --embed

--embed 는 마크다운의 attachment: 참조를 해설판 그림으로 채운다. 서술을 먼저 쓰고
그림 참조를 넣은 뒤 실행한다.

노트북은 src 에 코드가 있는 챕터에만 만든다. 코드가 없으면 실행할 것이 없다.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import config, listing_map, notebook as notebook_tools, skeleton  # noqa: E402


def list_candidates(study: config.Study) -> int:
    """노트북을 만들 수 있는 챕터를 보여준다."""
    src_dirs = study.src_dirs()
    print(f"{'저장소':<8}{'책':<6}{'노트북':<10}{'리스팅':>7}  챕터 폴더")
    for chapter_name, repo_dir in sorted(
        study.chapter_map.items(),
        key=lambda kv: listing_map.book_chapter_number(study.root / kv[0]) or 0,
    ):
        location = src_dirs.get(repo_dir)
        if location is None:
            continue
        book_chapter = listing_map.book_chapter_number(study.root / chapter_name)
        existing = sorted(location.glob("*_chapter_guide.ipynb"))
        status = "-"
        if existing:
            from studykit import verify
            status = verify.notebook_status(existing[0])
        listings = len(listing_map.repo_listings(location))
        code_files = list(location.rglob("*.py"))
        marker = "" if code_files or listings else "  (코드 없음)"
        print(f"{repo_dir:<8}{book_chapter or '?':<6}{status:<10}{listings:>7}"
              f"  {chapter_name[:44]}{marker}")
    print("\n노트북은 src 에 코드가 있는 챕터에만 만든다.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="교재 폴더 경로")
    parser.add_argument("repo_dir", nargs="?", help="업스트림 챕터 디렉터리 (예: ch05)")
    parser.add_argument("--list", action="store_true", help="대상 챕터 목록만 보여준다")
    parser.add_argument("--dry-run", action="store_true", help="쓰지 않고 구성만 보여준다")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일을 덮어쓴다")
    parser.add_argument("--embed", action="store_true",
                        help="마크다운의 attachment 참조를 해설판 그림으로 채운다")
    args = parser.parse_args()

    try:
        study = config.load(args.study)
    except config.StudyConfigError as exc:
        print(f"설정을 읽지 못했다: {exc}", file=sys.stderr)
        return 2

    if args.list or not args.repo_dir:
        return list_candidates(study)

    repo_dir = args.repo_dir
    chapter_dir = listing_map.chapter_dir_for(study, repo_dir)
    if chapter_dir is None:
        print(f"{repo_dir} 에 대응하는 책 챕터가 study.toml 에 없다.", file=sys.stderr)
        print("study-map-sources.py 로 매핑을 먼저 확정하라.", file=sys.stderr)
        return 2

    if args.embed:
        return embed_figures(study, repo_dir, chapter_dir)

    if args.dry_run:
        explainer = study.explainer_for(chapter_dir)
        if explainer is None:
            print(f"{chapter_dir.name} 에 해설판이 없다", file=sys.stderr)
            return 2
        title, sections = skeleton.parse_explainer(explainer)
        body = skeleton.body_sections(sections)
        print(f"장 제목 : {title}")
        print(f"해설판   : {explainer.name}")
        print(f"본문 절  : {len(body)}개\n")
        for section in body:
            listings = f"  리스팅 {', '.join(section.listings)}" if section.listings else ""
            exercises = f"  실습 {section.exercises}개" if section.exercises else ""
            print(f"  {'#' * section.level} {section.number} {section.title[:44]}"
                  f"{listings}{exercises}")
        return 0

    try:
        target, stats = skeleton.build(study, repo_dir, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"생성 실패: {exc}", file=sys.stderr)
        return 1

    print(f"생성: {target.relative_to(study.root)}")
    for key, value in stats.items():
        print(f"  {key:<20} {value}")
    print(f"""
다음 순서로 채운다.

  1. TODO(agent) {stats['todos']}개를 해설판 근거로 채운다
  2. 그림 참조를 넣고  study-new-notebook.py <교재> {repo_dir} --embed
  3. 실행해 오류 0 확인
  4. metadata.studykit 에 listing_coverage 를 선언하고 status 를 complete 로
  5. study-verify.py <교재> 로 완성 게이트 통과

지금은 status=draft 이므로 게이트가 lint 만 본다.""")
    return 0


def embed_figures(study: config.Study, repo_dir: str, chapter_dir: Path) -> int:
    """마크다운의 attachment: 참조를 해설판 그림으로 채운다."""
    location = study.src_dirs().get(repo_dir)
    notebooks = sorted(location.glob("*_chapter_guide.ipynb")) if location else []
    if not notebooks:
        print(f"{repo_dir} 에 노트북이 없다", file=sys.stderr)
        return 1
    explainer = study.explainer_for(chapter_dir)
    book_chapter = listing_map.book_chapter_number(chapter_dir)
    figure_map = notebook_tools.figure_map_from_explainer(explainer)
    figures = {f"fig{book_chapter}-{num.split('.', 1)[1]}.jpg": filename
               for num, filename in figure_map.items()}

    try:
        embedded = notebook_tools.embed_named_figures(
            notebooks[0], chapter_dir / "images", figures
        )
    except (KeyError, FileNotFoundError) as exc:
        print(f"내장 실패: {exc}", file=sys.stderr)
        print(f"해설판에서 확인된 그림: {', '.join(sorted(figures))}", file=sys.stderr)
        return 1

    if not embedded:
        print("새로 내장할 그림이 없다 (참조가 없거나 이미 전부 내장됨)")
        print(f"사용 가능한 키: {', '.join(sorted(figures))}")
        return 0
    print(f"내장 {len(embedded)}개:")
    for key, filename in sorted(embedded.items()):
        print(f"  {key:<16} <- {filename[:24]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
