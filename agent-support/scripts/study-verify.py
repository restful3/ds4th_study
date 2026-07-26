#!/usr/bin/env python3
"""교재 학습자료 검증 게이트.

하나라도 실패하면 산출물은 미완성이다. 에이전트는 완성을 보고하기 전에 반드시
이것을 통과시킨다.

    python3 agent-support/scripts/study-verify.py <교재경로>
    python3 agent-support/scripts/study-verify.py <교재경로> --env-only
    python3 agent-support/scripts/study-verify.py <교재경로> --no-urls
    python3 agent-support/scripts/study-verify.py <교재경로> --execute

검사 항목:
  환경        .venv, 공용 자원, dataset/.gitignore, _studykit.pth
  매핑        study.toml 의 챕터 대응이 키워드 빈도와 어긋나지 않는가
  오프셋      선언한 리스팅 오프셋이 제목 매칭 도출값과 같은가
  대응표      [mapping.upstream_dirs] 가 업스트림 챕터와 1:1 인가
  Makefile    업스트림 타깃·액션 순서가 tasks.py 와 등가인가
  노트북      실행 오류 0, 그림 attachment 내장, 상대경로 링크 대상 존재,
              renumber 잔재(chchNN·경로 불일치)와 저장 출력의 로컬 절대경로 부재
  리스팅      0바이트·중복 파일 보고 (업스트림 문제라 경고로만)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import config, listing_map, verify  # noqa: E402


def section(title: str) -> None:
    print(f"\n── {title}")


def report(failures: list[str], ok_message: str) -> int:
    if not failures:
        print(f"   ✓ {ok_message}")
        return 0
    for item in failures:
        print(f"   ✗ {item}")
    return len(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="교재 폴더 경로")
    parser.add_argument("--env-only", action="store_true", help="환경만 검사")
    parser.add_argument("--no-urls", action="store_true",
                        help="외부 URL 응답 확인을 건너뛴다 (느림)")
    parser.add_argument("--execute", action="store_true",
                        help="노트북을 재실행하고 출력을 정규화한 뒤 검사한다")
    parser.add_argument("--lint", action="store_true",
                        help="정적 검사만 한다 (완결성·리스팅 coverage 는 보지 않는다)")
    args = parser.parse_args()

    try:
        study = config.load(args.study)
    except config.StudyConfigError as exc:
        print(f"설정을 읽지 못했다: {exc}", file=sys.stderr)
        return 2

    print(f"교재: {study.title}")
    print(f"경로: {study.root}")
    failures = 0

    section("환경")
    failures += report(verify.check_environment(study), "부트스트랩 상태 정상")
    if args.env_only:
        return finish(failures)

    section("챕터 대응")
    failures += report(listing_map.verify_mapping(study),
                       f"{len(study.chapter_map)}개 챕터 대응이 키워드와 일치")

    section("리스팅 오프셋")
    offset_failures = []
    for repo_dir, declared in sorted(study.listing_offsets.items()):
        derived = listing_map.derive_offset(study, repo_dir)
        if derived is None:
            continue
        if derived.offset != declared:
            offset_failures.append(
                f"{repo_dir}: 선언 {declared} != 도출 {derived.offset} "
                f"(신뢰도 {derived.confidence:.2f})"
            )
        elif derived.confidence < listing_map.MIN_OFFSET_CONFIDENCE:
            offset_failures.append(
                f"{repo_dir}: 오프셋 {declared} 은 맞지만 신뢰도가 낮다 "
                f"({derived.confidence:.2f}) — 리스팅 제목을 눈으로 확인하라"
            )
    failures += report(offset_failures,
                       f"{len(study.listing_offsets)}개 오프셋이 도출값과 일치")

    section("리스팅 선언")
    declared_count = sum(len(v) for v in study.listing_overrides.values())
    failures += report(verify.check_declared_listings(study),
                       f"{declared_count}개 선언이 파일·심볼까지 해결됨")

    section("업스트림 대응표")
    failures += report(verify.check_upstream_mapping(study),
                       f"{len(study.upstream_dirs)}개 선언이 업스트림 챕터와 1:1")

    section("Makefile 등가성")
    failures += report(verify.check_makefile_parity(study),
                       "업스트림 타깃·액션 순서가 tasks.py 와 등가")

    section("노트북")
    notebooks = verify.find_notebooks(study)
    if not notebooks:
        print("   - 노트북이 없다 (src 에 코드가 있는 챕터에만 만든다)")
    for notebook in notebooks:
        status = verify.notebook_status(notebook)

        # 순서가 중요하다 — 실행 -> 정규화 -> 검사. 그 순서는 verify 가 지키고
        # test_notebook_hygiene.ExecuteThenNormalizeTests 가 검증한다.
        executor = (lambda path: execute_notebook(study, path)) if args.execute else None
        problems = verify.check_notebook_after_execute(
            study, notebook, executor, check_urls=not args.no_urls)

        # draft 는 lint 만, complete 는 완결성까지 본다.
        # 이 구분이 없으면 60% 노트북도 "모든 검증 통과" 가 된다.
        if not args.lint and status != "draft":
            repo_dir = verify.repo_dir_for_notebook(study, notebook)
            if repo_dir:
                problems.extend(verify.check_notebook_complete(study, notebook, repo_dir))

        label = f"{notebook.name} 규격 통과"
        if status == "draft":
            label += " (draft — lint 만)"
        elif args.lint:
            label += " (lint 만)"
        failures += report(problems, label)

    section("리스팅 상태 (참고)")
    notes = verify.check_listings(study)
    if notes:
        for note in notes:
            print(f"   ! {note}")
        print("     -> 업스트림 문제다. 노트북이 해설판 본문으로 대체했는지 확인하라")
    else:
        print("   ✓ 0바이트·중복 리스팅 없음")

    return finish(failures)


def execute_notebook(study: config.Study, notebook: Path) -> list[str]:
    """노트북을 재실행한다. 교재 .venv 의 jupyter 를 쓴다."""
    python = study.venv_python()
    result = subprocess.run(
        [str(python), "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace", notebook.name],
        cwd=str(notebook.parent), capture_output=True, text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        return [f"{notebook.name}: 재실행 실패 — {' / '.join(tail)}"]
    return []


def finish(failures: int) -> int:
    print()
    if failures:
        print(f"검증 실패 {failures}건 — 산출물은 미완성이다.")
        return 1
    print("모든 검증 통과.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
