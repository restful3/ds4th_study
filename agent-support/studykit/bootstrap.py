"""교재 실행 환경 부트스트랩.

새로 클론한 저장소에서 한 번 실행하면 챕터 폴더에서 바로 코드를 돌릴 수 있다.
.venv, code/, util/, config.ini, data/, dataset/ 은 모두 git 에서 제외되므로
이 모듈이 재현을 담당한다. 스터디원이 여럿이라 "한 명령으로 같은 환경"이 요건이다.

표준 라이브러리만 쓴다. 시스템 python3 로 실행되어야 하기 때문이다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from studykit import config
from studykit.config import Study

# 3.11 미만에는 tomllib 이 없어 config.py 가 tomli 로 폴백한다.
TOML_FALLBACK = "tomli"


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(cmd: list, cwd: Path | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"    $ {printable}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


def clone_upstream(study: Study) -> None:
    """업스트림 교재 코드 저장소를 code/ 로 클론한다 (없을 때만)."""
    step(f"업스트림 교재 코드 ({study.upstream_dir}/)")
    if study.upstream.exists():
        print(f"    이미 존재하므로 건너뜀: {study.upstream_dir}/")
        return
    if not study.upstream_url:
        print("    study.toml 에 [upstream].url 이 없다 — 건너뜀 (코드 없는 교재)")
        return
    if shutil.which("git") is None:
        sys.exit("git 이 필요하다.")
    run(["git", "clone", "--depth", "1", study.upstream_url, study.upstream_dir],
        cwd=study.root)


def place_shared(study: Study) -> None:
    """업스트림 코드가 기대하는 공용 자원을 교재 루트로 올린다.

    업스트림 코드가 `from util...` 로 임포트하고 util 옆의 config.ini 를 읽는 식이면
    디렉터리 구조를 업스트림 저장소 루트와 같게 맞춰야 한다.
    """
    step(f"공용 자원 {list(study.shared)}")
    if not study.upstream.exists():
        print("    업스트림 클론이 없어 건너뜀")
        return
    for name in study.shared:
        target = study.root / name
        source = study.upstream / name
        if target.exists():
            print(f"    이미 존재하므로 건너뜀: {name}")
            continue
        if not source.exists():
            print(f"    업스트림에 없어 건너뜀: {name}")
            continue
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        print(f"    복사: {study.upstream_dir}/{name} -> {name}")


def prepare_dataset_dir(study: Study) -> None:
    """매니페스트로 받는 대용량 데이터 저장소. 내용물은 git 에서 제외한다."""
    step("dataset/")
    study.dataset.mkdir(exist_ok=True)
    gitignore = study.dataset / ".gitignore"
    if gitignore.exists():
        print("    이미 존재하므로 건너뜀: dataset/.gitignore")
        return
    gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")
    print("    dataset/.gitignore 작성 (내용물 전체 추적 제외)")


def create_venv(study: Study, recreate: bool = False) -> Path:
    """교재별 .venv 를 만든다.

    챕터별 requirements 가 서로 다른 버전을 pin 하므로 교재마다 격리한다.
    uv 는 요청한 파이썬이 없으면 알아서 받는다. --seed 없이는 pip 이 안 들어간다.
    """
    step(f"가상환경 .venv (Python {study.python})")
    if recreate and study.venv.exists():
        print("    기존 .venv 삭제")
        shutil.rmtree(study.venv)
    if study.venv_python().is_relative_to(study.venv) and study.venv_python().exists():
        print("    이미 존재하므로 건너뜀 (--recreate 로 재생성)")
        return study.venv_python()

    uv = shutil.which("uv")
    if uv:
        run([uv, "python", "install", study.python], cwd=study.root)
        run([uv, "venv", "--seed", "--python", study.python, ".venv"], cwd=study.root)
    else:
        interpreter = shutil.which(f"python{study.python}")
        if interpreter is None:
            sys.exit(
                f"Python {study.python} 을 찾지 못했다. 다음 중 하나를 하라:\n"
                f"  * uv 설치 (권장, 파이썬까지 받아준다): "
                f"https://docs.astral.sh/uv/getting-started/installation/\n"
                f"  * python{study.python} 설치 후 다시 실행\n"
                f"  * study.toml 의 [study].python 을 설치된 버전으로 바꾼다"
            )
        run([interpreter, "-m", "venv", str(study.venv)])

    python = study.venv_python()
    if not python.exists():
        sys.exit(f"가상환경 생성에 실패했다: {python}")
    return python


def write_pth(study: Study, python: Path) -> None:
    """site-packages 에 교재 루트와 agent-support 를 등록한다.

    이러면 PYTHONPATH 없이 어느 cwd 에서도 업스트림 util 과 studykit 이 임포트된다.
    .pth 는 .gitignore 대상이라 .venv 를 새로 만들 때마다 다시 써야 한다.
    """
    step("_studykit.pth (어느 cwd 에서도 임포트 가능하게)")
    site_packages = Path(subprocess.run(
        [str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    pth = site_packages / "_studykit.pth"
    lines = [str(study.root), str(config.REPO_ROOT / "agent-support")]
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        print(f"    등록: {line}")


def install_base(study: Study, python: Path) -> None:
    """기반 패키지. 챕터별 정확한 pin 은 각 챕터 tasks.py 의 init 이 담당한다."""
    packages = list(study.base_packages)
    minor = int(study.python.split(".")[1]) if "." in study.python else 11
    if minor < 11:
        packages.append(TOML_FALLBACK)   # studykit.config 가 TOML 을 읽는다
    step(f"기반 패키지 {packages}")
    run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "--quiet", *packages])


def register_kernel(study: Study, python: Path) -> None:
    """노트북용 커널. 교재별로 이름을 달리해 다른 교재와 섞이지 않게 한다."""
    if "ipykernel" not in study.base_packages:
        return
    step(f"주피터 커널 등록 ({study.kernel_name})")
    run([python, "-m", "ipykernel", "install", "--user",
         "--name", study.kernel_name,
         "--display-name", f"{study.title} (.venv py{study.python})"])


def bootstrap(study: Study, recreate: bool = False) -> Path:
    """전체 부트스트랩. 이미 있는 것은 건너뛰므로 여러 번 실행해도 안전하다."""
    print(f"교재: {study.title}")
    print(f"경로: {study.root}")
    clone_upstream(study)
    place_shared(study)
    prepare_dataset_dir(study)
    python = create_venv(study, recreate=recreate)
    write_pth(study, python)
    install_base(study, python)
    register_kernel(study, python)
    return python


def next_steps(study: Study, python: Path) -> str:
    try:
        rel = python.relative_to(study.root)
    except ValueError:
        rel = python
    chapters = study.chapter_dirs()
    example = ""
    if chapters:
        src = chapters[0] / "src"
        if src.is_dir():
            example = f"  cd '{chapters[0].name}/src'/* && python tasks.py --list\n"
    return (
        f"\n환경 준비 완료.\n\n다음 단계:\n"
        f"  source .venv/bin/activate        # 또는 {rel} 직접 사용\n"
        f"{example}"
        f"  python3 {os.path.relpath(config.REPO_ROOT / 'agent-support/scripts/study-verify.py', study.root)} .\n"
    )
