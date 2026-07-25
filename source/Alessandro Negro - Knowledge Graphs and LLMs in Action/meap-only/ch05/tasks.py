"""MEAP 전용 (업스트림 ch05, miRNA 구조적 소스 통합) 실행 태스크.

**최종 출간본에 대응하는 챕터가 없다.** 원문 md 검증 결과 `miRNA` 는 책 4장에
단 1회만 나온다 — MEAP 단계에서 있었다가 출간 과정에서 잘려나간 내용이다.
그래서 챕터 폴더 밑이 아니라 meap-only/ 에 둔다. 스터디 진행에는 쓰지 않되
구조적 소스 통합·개체 정규화(reconciliation) 참고 자료로 남긴다.

    python tasks.py --list
    python tasks.py init
    python tasks.py download      # 6개 외부 데이터셋을 dataset/hmdd 로
    python tasks.py import
    python tasks.py reconciliate

download 는 교재 루트의 dataset/hmdd/ 에 내려받는다. 이미 있는 파일은 건너뛴다.
"""
from pathlib import Path

from kgbook import Download, Gunzip, Mkdir, Move, PipInstall, RunScript, Unzip, main
from kgbook.actions import run as _run

HERE = Path(__file__).resolve().parent

HMDD = "dataset/hmdd"
MISIM = f"{HMDD}/misim"

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
    "download": [
        Mkdir(f"{HMDD}/miRDB"),
        Mkdir(f"{HMDD}/dbDEMC"),
        Mkdir(f"{HMDD}/miRBase"),
        Mkdir(MISIM),
        Mkdir(f"{HMDD}/miR2Disease"),
        Download(
            "https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz",
            f"{HMDD}/miRDB/miRDB_v6.0_prediction_result.txt.gz",
        ),
        Download(
            "http://www.cuilab.cn/static/hmdd3/data/alldata.txt",
            f"{HMDD}/HMDD_v3.2.txt",
        ),
        Download(
            "https://www.biosino.org/dbDEMC/download/MiRExpAll",
            f"{HMDD}/dbDEMC/miRExpAll.txt",
        ),
        Download(
            "http://watson.compbio.iupui.edu:8080/miR2Disease/download/AllEntries.txt",
            f"{HMDD}/miR2Disease/AllEntries.txt",
        ),
        Download(
            "https://mirbase.org/download/miRNA.dat",
            f"{HMDD}/miRBase/miRNA.dat",
        ),
        Download(
            "http://www.cuilab.cn/files/images/cuilab/misim.zip",
            f"{MISIM}/misim.zip",
        ),
        Gunzip(f"{HMDD}/miRDB/miRDB_v6.0_prediction_result.txt.gz"),
        Unzip(f"{MISIM}/misim.zip", MISIM),
        Move(f"{MISIM}/miRNA similarity matrix.txt", f"{MISIM}/similarityMatrix.txt"),
        Move(f"{MISIM}/microRNA name.xls", f"{MISIM}/microRNA.xls"),
    ],
    "import": [
        RunScript("importer/import_miRNA_hmdd.py"),
        RunScript("importer/import_miRNA_dbDEMC.py"),
        RunScript("importer/import_miRNA_miR2Disease.py"),
        RunScript("importer/import_miRNA_EMBL.py"),
        RunScript("importer/import_miRNA_RDB.py"),
        RunScript("importer/import_miRNA_sim.py"),
    ],
    "reconciliate": [
        RunScript("reconciliation/reconciliate_disease.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("download")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
