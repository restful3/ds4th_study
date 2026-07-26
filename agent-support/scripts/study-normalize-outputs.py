#!/usr/bin/env python3
"""저장된 노트북 출력에서 실행한 기계의 절대경로를 지운다.

site-packages 경고나 PosixPath repr 처럼 실행하면 반드시 절대경로가 붙는 출력이
있어서 노트북 소스만 고쳐서는 해결되지 않는다. 재실행한 뒤 포매터처럼 한 번 돌린다.
소스 셀은 건드리지 않는다.

    python3 agent-support/scripts/study-normalize-outputs.py <교재경로>
    python3 agent-support/scripts/study-normalize-outputs.py <교재경로> --check

교재 루트는 `<교재>`, 저장소 루트는 `<저장소>`, 홈 디렉터리는 `~` 가 된다.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "agent-support"))

from studykit import config, notebook, verify  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="교재 폴더 경로")
    parser.add_argument("--check", action="store_true",
                        help="고치지 않고 대상만 보고한다 (있으면 종료 코드 1)")
    args = parser.parse_args()

    try:
        study = config.load(args.study)
    except config.StudyConfigError as exc:
        print(f"설정을 읽지 못했다: {exc}", file=sys.stderr)
        return 2

    changed = []
    for path in verify.find_notebooks(study):
        if args.check:
            if verify.check_saved_output_paths(study, path):
                changed.append(path.name)
            continue
        if notebook.normalize_outputs(path, study):
            changed.append(path.name)

    if not changed:
        print("모든 노트북 출력이 이미 정규화돼 있다.")
        return 0
    for name in changed:
        print(f"  {name}")
    if args.check:
        print(f"노트북 {len(changed)}개에 로컬 절대경로가 남아 있다.")
        return 1
    print(f"노트북 {len(changed)}개의 출력을 정규화했다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
