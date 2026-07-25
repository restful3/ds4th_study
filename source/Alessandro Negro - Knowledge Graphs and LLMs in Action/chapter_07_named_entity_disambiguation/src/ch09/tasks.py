"""책 7장 (업스트림 ch09, 개체명 중의성 해소) 실행 태스크.

    python tasks.py --list
    python tasks.py init
    python tasks.py download      # SNOMED + UMLS (UMLS_API_KEY 필요)
    python tasks.py import
    python tasks.py disambiguate

download 는 UMLS 계정의 API 키가 필요하다:
    1. https://uts.nlm.nih.gov/uts/ 로그인 (Google/Microsoft 계정 사용 가능)
    2. "Get Your API Key" 로 키 발급
    3. export UMLS_API_KEY=xxxxxxxx

내려받는 SNOMED/UMLS 아카이브는 수 GB 규모다. 교재 루트 dataset/ontology/ 에
저장되며 git 추적에서 제외된다.
"""
from pathlib import Path

from studykit import Download, Mkdir, Move, PipInstall, RunScript, Unzip, main
from studykit.actions import run as _run

HERE = Path(__file__).resolve().parent

ONTOLOGY = "dataset/ontology"
SNOMED = f"{ONTOLOGY}/snomed"
UMLS = f"{ONTOLOGY}/umls"
RELEASE = "SnomedCT_USEditionRF2_PRODUCTION_20220901T120000Z"
TERMINOLOGY = f"{SNOMED}/{RELEASE}/Full/Terminology"

UTS_DOWNLOAD = "https://uts-ws.nlm.nih.gov/download?url="
SNOMED_URL = (
    f"{UTS_DOWNLOAD}https://download.nlm.nih.gov/mlb/utsauth/USExt/"
    f"{RELEASE}.zip&apiKey=<KEY>"
)
UMLS_URL = (
    f"{UTS_DOWNLOAD}https://download.nlm.nih.gov/umls/kss/2022AB/"
    f"umls-2022AB-mrconso.zip&apiKey=<KEY>"
)

# unzip 후 Full/Terminology 에서 snomed 루트로 끌어올리는 파일들
SNOMED_FILES = [
    "sct2_Relationship_Full_US1000124_20220901.txt",
    "sct2_Description_Full-en_US1000124_20220901.txt",
    "sct2_TextDefinition_Full-en_US1000124_20220901.txt",
]

TASKS = {
    "download": [
        Mkdir(SNOMED),
        Mkdir(UMLS),
        Download(SNOMED_URL, f"{SNOMED}/{RELEASE}.zip"),
        Download(UMLS_URL, f"{UMLS}/umls-2022AB-mrconso.zip"),
        Download(
            "https://lhncbc.nlm.nih.gov/semanticnetwork/download/SemGroups.txt",
            f"{UMLS}/SemGroups.txt",
        ),
        Unzip(f"{SNOMED}/{RELEASE}.zip", SNOMED),
        Unzip(f"{UMLS}/umls-2022AB-mrconso.zip", UMLS),
        *[Move(f"{TERMINOLOGY}/{name}", f"{SNOMED}/{name}") for name in SNOMED_FILES],
    ],
    "init": [
        PipInstall("requirements.lock"),
    ],
    "import": [
        RunScript("importer/import_snomed_rels.py"),
        RunScript("importer/import_snomed_names.py"),
        RunScript("importer/propagate_snomed_categories.py"),
        RunScript("importer/import_hpo.py"),
        RunScript("importer/import_umls_concept_mapping.py"),
        RunScript("importer/import_ocred_documents.py"),
    ],
    "disambiguate": [
        RunScript("disambiguation/disambiguator.py"),
        RunScript("disambiguation/ontology_linking.py"),
        RunScript("disambiguation/co_occurrence_generator.py"),
    ],
}


def run(target: str, dry_run: bool = False):
    """노트북에서 호출할 때: import tasks; tasks.run("import")"""
    return _run(TASKS, target, HERE, dry_run=dry_run)


if __name__ == "__main__":
    raise SystemExit(main(TASKS, HERE))
