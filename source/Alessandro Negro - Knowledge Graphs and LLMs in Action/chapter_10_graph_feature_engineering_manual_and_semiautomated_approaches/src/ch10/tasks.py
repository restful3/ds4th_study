"""책 10장 (업스트림 ch12, 그래프 피처 엔지니어링) 실행 태스크.

    python tasks.py --list
    python tasks.py init

업스트림 Makefile에는 init 만 있다. 리스팅 대응은 study.toml 의
[mapping.listings.ch10] 가 정본이다. 리스팅 파일 이름이 업스트림 MEAP 번호(12.x)라
책 번호와 major 부터 다르고, minor 도 한 건이 어긋난다 — 책 10.18 = listings/'12.19 ReFeX...py'
(파일 12.18 이 프롬프트라 책에서 번호를 못 받아 뒤가 밀렸다). "일괄 -2" 로 계산하면
그 한 건에서 틀린다.

cypher.listings() 는 '.py' 를 제외하므로 이 폴더에서는 12.15~12.18 네 개만 돌려준다.
파이썬 리스팅은 노트북 헬퍼가 ast 로 심볼을 잘라 보여주고 모듈로 임포트해 실행한다.
Cypher 리스팅은 hetionet 데이터베이스를 명시해야 한다 (기본 DB 가 아니다):

    from studykit import cypher
    print(cypher.read("12.15"))                          # 원문
    cypher.run("12.15", database="hetionet")             # 4장이 만든 DB

주의: 책 10.15 는 바인딩 없는 변수 v 를 써서 DWPC 가 11배 어긋난다. 원본은 고치지
않고 노트북에서 치환해 실행한다 (10.2.2 절).
"""
from pathlib import Path

from studykit import PipInstall, main
from studykit.actions import run as _run

HERE = Path(__file__).resolve().parent

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("init")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
