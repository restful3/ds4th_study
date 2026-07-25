"""책 8장 (업스트림 ch10, 오픈 LLM + 도메인 온톨로지 NED) 실행 태스크.

    python tasks.py --list
    python tasks.py init
    python tasks.py download      # SNOMED (UMLS_API_KEY 필요)
    python tasks.py import
    python tasks.py disambiguate

download 는 UMLS API 키가 필요하다 (7장과 동일):
    1. https://uts.nlm.nih.gov/uts/ 로그인
    2. "Get Your API Key" 로 키 발급
    3. export UMLS_API_KEY=xxxxxxxx

7장에서 이미 SNOMED를 내려받았다면 같은 dataset/ontology/snomed/ 를 공유하므로
download 를 건너뛸 수 있다.
"""
from pathlib import Path

from kgbook import Download, Mkdir, Move, PipInstall, RunScript, Unzip, main
from kgbook.actions import run as _run

HERE = Path(__file__).resolve().parent

SNOMED = "dataset/ontology/snomed"
RELEASE = "SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z"
TERMINOLOGY = f"{SNOMED}/{RELEASE}/Full/Terminology"

SNOMED_URL = (
    "https://uts-ws.nlm.nih.gov/download?url="
    "https://download.nlm.nih.gov/mlb/utsauth/USExt/"
    f"{RELEASE}.zip&apiKey=<KEY>"
)

SNOMED_FILES = [
    "sct2_Relationship_Full_US1000124_20220901.txt",
    "sct2_Description_Full-en_US1000124_20220901.txt",
    "sct2_TextDefinition_Full-en_US1000124_20220901.txt",
]

TASKS = {
    "init": [
        PipInstall("requirements.lock"),
    ],
    "download": [
        Mkdir(SNOMED),
        Download(SNOMED_URL, f"{SNOMED}/{RELEASE}.zip"),
        Unzip(f"{SNOMED}/{RELEASE}.zip", SNOMED),
        *[Move(f"{TERMINOLOGY}/{name}", f"{SNOMED}/{name}") for name in SNOMED_FILES],
    ],
    "import": [
        RunScript("importer/import_snomed_rels.py"),
        RunScript("importer/import_snomed_names.py"),
        RunScript("importer/propagate_snomed_categories.py"),
    ],
    "disambiguate": [
        RunScript("disambiguation/main.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
