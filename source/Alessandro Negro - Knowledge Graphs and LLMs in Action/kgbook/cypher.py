"""책의 Cypher 리스팅을 Python에서 읽고 실행한다.

교재 저장소의 listings/ 안에는 확장자 없는 Cypher 파일이 들어 있다
(예: '3.19 - load_hpo_ontology', '4.2 Import PPI Network').
책이 가르치는 대상이 Cypher 자체이므로 파일은 원문 그대로 두고,
노트북·스크립트에서 호출할 수 있는 얇은 러너만 제공한다.

    from kgbook import cypher
    cypher.listings()                       # 현재 폴더의 리스팅 목록
    print(cypher.read("3.19"))              # 원문 확인 (번호 접두사로 검색)
    cypher.run("3.19")                      # Neo4j에 실행
"""
import configparser
import re
from pathlib import Path

from kgbook.paths import CONFIG

# Path.suffix 는 '3.16 - create_database' 를 확장자 '.16 - create_database' 로 읽으므로
# 화이트리스트를 쓸 수 없다. Python·설정·바이너리 파일만 제외한다.
_EXCLUDED_ENDINGS = (
    ".py", ".pyc", ".ipynb", ".lock", ".md", ".yaml", ".yml", ".template",
    ".json", ".xls", ".xlsx", ".zip", ".gz", ".png", ".jpg", ".jpeg",
)


def listings(directory: str | Path = "listings") -> list[Path]:
    """Cypher·텍스트 리스팅 파일 목록을 책의 번호 순서로 반환.

    확장자 없는 파일과 .txt 를 모두 포함한다 (교재 저장소가 두 방식을 섞어 쓴다).
    Python 리스팅은 제외된다 — 그건 직접 실행하면 된다.
    """
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"리스팅 폴더가 없다: {base.resolve()}")
    files = [
        p for p in base.iterdir()
        if p.is_file() and not p.name.lower().endswith(_EXCLUDED_ENDINGS)
    ]
    return sorted(files, key=_listing_sort_key)


def _listing_sort_key(path: Path) -> tuple:
    match = re.match(r"(\d+)\.(\d+)", path.name)
    if match:
        return (int(match.group(1)), int(match.group(2)), path.name)
    return (999, 999, path.name)


def find(number: str, directory: str | Path = "listings") -> Path:
    """'3.19' 또는 파일명 일부로 리스팅 파일 하나를 찾는다."""
    candidates = [p for p in listings(directory) if p.name.startswith(str(number))]
    if not candidates:
        candidates = [
            p for p in listings(directory) if str(number).lower() in p.name.lower()
        ]
    if not candidates:
        raise FileNotFoundError(f"리스팅을 찾지 못했다: {number}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(f"'{number}' 에 여러 리스팅이 걸린다: {names}")
    return candidates[0]


def read(number: str, directory: str | Path = "listings") -> str:
    """리스팅 원문을 문자열로 반환."""
    return find(number, directory).read_text(encoding="utf-8")


def statements(number: str, directory: str | Path = "listings") -> list[str]:
    """리스팅을 세미콜론 기준으로 나눈 Cypher 문장 목록.

    주의: 문자열 리터럴 안의 세미콜론은 구분하지 못한다. 그런 리스팅은
    read() 로 원문을 확인한 뒤 직접 실행하라.
    """
    text = read(number, directory)
    parts = [s.strip() for s in text.split(";")]
    return [s for s in parts if s and not all(l.strip().startswith("//")
                                              for l in s.splitlines() if l.strip())]


def neo4j_params() -> dict:
    """config.ini 의 [neo4j] 섹션을 읽는다."""
    parser = configparser.ConfigParser()
    if not CONFIG.exists():
        raise FileNotFoundError(f"config.ini 가 없다: {CONFIG}")
    parser.read(CONFIG)
    return dict(parser["neo4j"])


def driver():
    """config.ini 기준 Neo4j 드라이버."""
    from neo4j import GraphDatabase

    params = neo4j_params()
    extra = {}
    if "encrypted" in params:
        extra["encrypted"] = bool(int(params["encrypted"]))
    return GraphDatabase.driver(
        params.get("uri", "bolt://localhost:7687"),
        auth=(params.get("user", "neo4j"), params.get("password", "password")),
        **extra,
    )


def run(number: str, directory: str | Path = "listings", database: str | None = None):
    """리스팅의 Cypher 문장을 순서대로 실행하고 마지막 결과를 반환."""
    database = database or neo4j_params().get("database", "neo4j")
    result = None
    with driver() as drv, drv.session(database=database) as session:
        for statement in statements(number, directory):
            print(f"-- 실행:\n{statement}\n")
            result = [record.data() for record in session.run(statement)]
    return result
