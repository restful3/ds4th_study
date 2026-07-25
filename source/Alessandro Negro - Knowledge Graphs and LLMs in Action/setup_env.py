#!/usr/bin/env python3
"""Knowledge Graphs and LLMs in Action 교재 실행 환경 부트스트랩.

새로 클론한 저장소에서 한 번 실행하면 챕터 src 폴더에서 바로 코드를 돌릴 수 있다.
.venv, data/, dataset/, code/ 는 모두 .gitignore 대상이라 git 으로 복구되지 않으므로
이 스크립트가 재현을 담당한다.

    python3 setup_env.py              # 없는 것만 채운다 (idempotent)
    python3 setup_env.py --recreate   # .venv 를 지우고 다시 만든다
    python3 setup_env.py --python 3.9 # 다른 파이썬 마이너 버전으로

표준 라이브러리만 쓴다. 시스템 python3 로 실행하면 된다.

하는 일:
  1. 업스트림 교재 코드 저장소를 code/ 에 클론 (없을 때만)
  2. code/ 의 util/, config.ini, data/ 를 교재 루트로 복사 (공용 자원)
  3. dataset/ 생성 + 자체 .gitignore (대용량 데이터셋 추적 제외)
  4. .venv 생성 — 업스트림이 테스트한 Python 3.10 을 uv 로 확보
  5. site-packages 에 _kgbook_root.pth 작성 → 어느 cwd 에서도 util/kgbook 임포트
  6. 기반 패키지(neo4j, tqdm) 설치

검증: python3 tests/test_env_setup.py
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parent
UPSTREAM_URL = "https://github.com/alenegro81/knowledge-graphs-and-llms-in-action.git"

# 업스트림 README 가 테스트했다고 밝힌 버전. scispacy 가 요구하는 nmslib 는
# 3.11+ 에 휠이 없어 빌드가 깨지므로 3.10 을 넘기지 않는다.
DEFAULT_PYTHON = "3.10"

# util.graphdb_base / base_importer 가 임포트하는 최소 패키지 + 챕터 해설 노트북 실행용.
# 챕터별 정확한 pin 은 각 src/chXX/tasks.py 의 init 이 담당한다.
BASE_PACKAGES = ["neo4j>=5,<6", "tqdm", "ipykernel", "jupyterlab"]

# code/ 에서 교재 루트로 올리는 공용 자원. 업스트림 코드가
# `from util...` 와 `util/../config.ini` 를 기대하기 때문에 루트에 있어야 한다.
SHARED = ["util", "config.ini", "data"]


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"    $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def clone_upstream(code_dir: Path) -> None:
    step("업스트림 교재 코드")
    if code_dir.exists():
        print(f"    이미 존재하므로 건너뜀: {code_dir.name}/")
        return
    if shutil.which("git") is None:
        sys.exit("git 이 필요하다. git 을 설치하고 다시 실행하라.")
    run(["git", "clone", "--depth", "1", UPSTREAM_URL, code_dir.name], cwd=BOOK_ROOT)


def copy_shared(code_dir: Path) -> None:
    step("공용 자원 (util/, config.ini, data/)")
    for name in SHARED:
        target = BOOK_ROOT / name
        source = code_dir / name
        if target.exists():
            print(f"    이미 존재하므로 건너뜀: {name}")
            continue
        if not source.exists():
            sys.exit(f"업스트림에 {name} 이 없다: {source}")
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        print(f"    복사: code/{name} -> {name}")


def prepare_dataset_dir() -> None:
    step("dataset/ (대용량 데이터셋 저장소)")
    dataset = BOOK_ROOT / "dataset"
    dataset.mkdir(exist_ok=True)
    gitignore = dataset / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")
        print("    dataset/.gitignore 작성 (내용물 전체 추적 제외)")
    else:
        print("    이미 존재하므로 건너뜀: dataset/.gitignore")


def create_venv(venv: Path, python_version: str, recreate: bool) -> Path:
    step(f"가상환경 .venv (Python {python_version})")
    if recreate and venv.exists():
        print(f"    기존 .venv 삭제")
        shutil.rmtree(venv)
    if venv_python(venv).exists():
        print("    이미 존재하므로 건너뜀 (--recreate 로 재생성)")
        return venv_python(venv)

    uv = shutil.which("uv")
    if uv:
        # uv 는 요청한 파이썬이 없으면 알아서 내려받는다. --seed 로 pip 포함.
        run([uv, "python", "install", python_version], cwd=BOOK_ROOT)
        run([uv, "venv", "--seed", "--python", python_version, ".venv"], cwd=BOOK_ROOT)
    else:
        interpreter = shutil.which(f"python{python_version}")
        if interpreter is None:
            sys.exit(
                f"Python {python_version} 을 찾지 못했다. 다음 중 하나를 하라:\n"
                f"  * uv 설치 (권장, 파이썬까지 알아서 받는다): "
                f"https://docs.astral.sh/uv/getting-started/installation/\n"
                f"  * python{python_version} 설치 후 다시 실행\n"
                f"  * 다른 버전으로: python3 setup_env.py --python 3.9"
            )
        run([interpreter, "-m", "venv", str(venv)])

    python = venv_python(venv)
    if not python.exists():
        sys.exit(f"가상환경 생성에 실패했다: {python}")
    return python


def write_pth(python: Path) -> None:
    step("_kgbook_root.pth (어느 cwd 에서도 util/kgbook 임포트)")
    site_packages = Path(subprocess.run(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    pth = site_packages / "_kgbook_root.pth"
    pth.write_text(f"{BOOK_ROOT}\n", encoding="utf-8")
    print(f"    작성: {pth.relative_to(BOOK_ROOT)}")
    print(f"    내용: {BOOK_ROOT}")


def install_base(python: Path) -> None:
    step(f"기반 패키지 {BASE_PACKAGES}")
    run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "--quiet", *BASE_PACKAGES])


def verify(python: Path) -> bool:
    step("검증")
    test = BOOK_ROOT / "tests" / "test_env_setup.py"
    if not test.exists():
        print(f"    테스트 파일이 없어 건너뜀: {test}")
        return True
    return subprocess.run([str(python), str(test)], cwd=str(BOOK_ROOT)).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="교재 실행 환경 부트스트랩",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--recreate", action="store_true",
                        help=".venv 를 지우고 다시 만든다")
    parser.add_argument("--python", default=DEFAULT_PYTHON,
                        help=f"파이썬 마이너 버전 (기본 {DEFAULT_PYTHON})")
    parser.add_argument("--skip-verify", action="store_true",
                        help="마지막 검증 테스트를 건너뛴다")
    args = parser.parse_args()

    print(f"교재 루트: {BOOK_ROOT}")
    code_dir = BOOK_ROOT / "code"

    clone_upstream(code_dir)
    copy_shared(code_dir)
    prepare_dataset_dir()
    python = create_venv(BOOK_ROOT / ".venv", args.python, args.recreate)
    write_pth(python)
    install_base(python)

    if not args.skip_verify and not verify(python):
        print("\n검증 실패. 위 항목을 확인하라.")
        return 1

    rel = python.relative_to(BOOK_ROOT)
    print(
        f"\n환경 준비 완료.\n\n"
        f"다음 단계:\n"
        f"  source .venv/bin/activate                 # 또는 {rel} 직접 사용\n"
        f"  cd chapter_03_*/src/ch03 && python tasks.py --list\n\n"
        f"Neo4j 가 필요하다 (부록 B). config.ini 또는 환경변수로 접속정보를 준다:\n"
        f"  export NEO4J_URI=bolt://localhost:7687\n"
        f"  export NEO4J_USER=neo4j NEO4J_PASSWORD=...\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
