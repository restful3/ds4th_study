"""MEAP 전용 (업스트림 ch06, KG + NLP 문서 파이프라인) 실행 태스크.

**최종 출간본에 대응하는 챕터가 없다.** 원문 md 검증 결과 `Jaccard` 는 0회,
`BBC` 는 책 10·11·14장에 1\~2회 스치듯 언급될 뿐이다. spaCy 기반 NER 내용은
책 7·8장(개체명 중의성 해소)으로 재배치됐고, 그쪽 코드는 ch09·ch10 이다.
그래서 챕터 폴더 밑이 아니라 meap-only/ 에 둔다.

    python tasks.py --list
    python tasks.py init          # 패키지 + spaCy en_core_web_sm 모델
    python tasks.py import        # BBC 코퍼스 적재 -> 조직 추출 -> 소유관계 보강

import 는 교재 루트의 data/bbc/ (2,225개 기사)를 읽는다. 이미 동봉되어 있어
따로 내려받을 필요가 없다.
"""
from pathlib import Path

from studykit import PipInstall, RunModule, RunScript, main
from studykit.actions import run as _run

HERE = Path(__file__).resolve().parent

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
        RunModule("spacy download en_core_web_sm"),
    ],
    "import": [
        RunScript("importer/step1__import_bbc.py"),
        RunScript("importer/step2__enrich_organizations.py"),
        RunScript("importer/step3__enrich_by_ownerships.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
