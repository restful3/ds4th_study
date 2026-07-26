"""Makefile 대체용 액션과 러너.

업스트림 교재 저장소는 챕터마다 Makefile로 init/download/import 등을 돌린다.
이 모듈은 같은 동작을 순수 Python으로 제공한다. make, curl, unzip, gunzip, mv 같은
외부 명령에 의존하지 않으므로 Windows에서도 그대로 돌아간다.

각 챕터 src 폴더의 tasks.py 가 TASKS 딕셔너리를 선언하고 main()을 호출한다.
등가성은 verify.check_makefile_parity 가 보고, 테스트는
agent-support/tests/test_study_materials.py 의 MakefileParityTests 다. 짝지을
챕터를 고르는 대응표 자체는 test_upstream_mapping.py 가 검사한다.
"""
import gzip
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from studykit import config
from studykit.config import normalize_path

_STUDY: config.Study | None = None


def study() -> config.Study:
    """현재 교재 설정. 액션은 교재 폴더 안에서 실행되므로 cwd 에서 찾는다."""
    global _STUDY
    if _STUDY is None:
        _STUDY = config.load()
    return _STUDY


class Action:
    """실행 단위 하나. signature() 는 Makefile 등가성 검증에 쓰인다."""

    kind = "?"

    def signature(self) -> tuple:
        raise NotImplementedError

    def describe(self) -> str:
        return " ".join(str(x) for x in self.signature())

    def run(self, chapter_dir: Path) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------------
# 파이썬 실행
# --------------------------------------------------------------------------
class RunScript(Action):
    """Makefile의 `PYTHONPATH=../../ $(PYTHON) importer/x.py` 대체."""

    kind = "py"

    def __init__(self, script: str, env: dict[str, str] | None = None):
        self.script = script
        self.env = env or {}

    def signature(self) -> tuple:
        return ("py", self.script)

    def run(self, chapter_dir: Path) -> None:
        env = os.environ.copy()
        # .pth 로 이미 잡히지만, 다른 인터프리터로 돌릴 때를 위한 보험
        env["PYTHONPATH"] = _pythonpath(env)
        for key, default in self.env.items():
            env.setdefault(key, default)
        _check_call([str(study().venv_python()), self.script], cwd=chapter_dir, env=env)


class RunModule(Action):
    """Makefile의 `$(PYTHON) -m spacy download en_core_web_sm` 대체."""

    kind = "py-m"

    def __init__(self, args: str):
        self.args = args

    def signature(self) -> tuple:
        return ("py-m", self.args)

    def run(self, chapter_dir: Path) -> None:
        _check_call(
            [str(study().venv_python()), "-m", *self.args.split()], cwd=chapter_dir
        )


class PipInstall(Action):
    """Makefile의 `$(PIP) install -r requirements.lock` 대체."""

    kind = "pip"

    def __init__(self, requirements: str):
        self.requirements = requirements

    def signature(self) -> tuple:
        return ("pip", self.requirements)

    def run(self, chapter_dir: Path) -> None:
        _check_call(
            [str(study().venv_python()), "-m", "pip", "install", "-r", self.requirements],
            cwd=chapter_dir,
        )


class Streamlit(Action):
    """Makefile의 `$(STREAMLIT) run app.py` 대체."""

    kind = "streamlit"

    def __init__(self, script: str):
        self.script = script

    def signature(self) -> tuple:
        return ("streamlit", self.script)

    def run(self, chapter_dir: Path) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = _pythonpath(env)
        _check_call(
            [str(study().venv_bin("streamlit")), "run", self.script],
            cwd=chapter_dir,
            env=env,
        )


# --------------------------------------------------------------------------
# 데이터셋 준비 (curl / unzip / gunzip / mv / mkdir 대체)
# --------------------------------------------------------------------------
class Mkdir(Action):
    kind = "mkdir"

    def __init__(self, path: str):
        self.path = normalize_path(path)

    def signature(self) -> tuple:
        return ("mkdir", self.path)

    def run(self, chapter_dir: Path) -> None:
        study().resolve(self.path).mkdir(parents=True, exist_ok=True)


class Download(Action):
    """curl 대체. URL의 `apiKey=<KEY>` 는 실행 시 환경변수로 치환한다.

    UMLS 계열 URL은 UMLS_API_KEY 환경변수가 필요하다
    (https://uts.nlm.nih.gov/uts/ 에서 무료 발급).
    """

    kind = "curl"

    def __init__(self, url: str, dest: str, api_key_env: str = "UMLS_API_KEY"):
        self.url = url
        self.dest = normalize_path(dest)
        self.api_key_env = api_key_env

    def signature(self) -> tuple:
        return ("curl", self.url, self.dest)

    def _resolved_url(self) -> str:
        if "<KEY>" not in self.url:
            return self.url
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"이 URL은 API 키가 필요하다. 환경변수 {self.api_key_env} 를 설정하라.\n"
                f"  UMLS 키 발급: https://uts.nlm.nih.gov/uts/  (로그인 후 Get Your API Key)\n"
                f"  예: export {self.api_key_env}=xxxxxxxx"
            )
        return self.url.replace("<KEY>", key)

    def run(self, chapter_dir: Path) -> None:
        target = study().resolve(self.dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            print(f"    이미 존재하므로 건너뜀: {self.dest}")
            return
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(self._resolved_url()) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    _progress(self.dest, done, total)
        print()
        tmp.rename(target)


class Unzip(Action):
    kind = "unzip"

    def __init__(self, archive: str, dest: str | None = None):
        self.archive = normalize_path(archive)
        self.dest = normalize_path(dest) if dest else None

    def signature(self) -> tuple:
        return ("unzip", self.archive, self.dest)

    def run(self, chapter_dir: Path) -> None:
        target = study().resolve(self.dest) if self.dest else study().resolve(self.archive).parent
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(study().resolve(self.archive)) as zf:
            zf.extractall(target)


class Gunzip(Action):
    kind = "gunzip"

    def __init__(self, path: str):
        self.path = normalize_path(path)

    def signature(self) -> tuple:
        return ("gunzip", self.path)

    def run(self, chapter_dir: Path) -> None:
        source = study().resolve(self.path)
        target = source.with_suffix("")  # .gz 제거
        with gzip.open(source, "rb") as fin, open(target, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        source.unlink()


class Move(Action):
    kind = "mv"

    def __init__(self, src: str, dst: str):
        self.src = normalize_path(src)
        self.dst = normalize_path(dst)

    def signature(self) -> tuple:
        return ("mv", self.src, self.dst)

    def run(self, chapter_dir: Path) -> None:
        target = study().resolve(self.dst)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(study().resolve(self.src)), str(target))


# --------------------------------------------------------------------------
# 러너
# --------------------------------------------------------------------------
def _pythonpath(env: dict) -> str:
    """업스트림 코드가 임포트하는 교재 루트와, studykit 이 있는 agent-support 를 넣는다.

    부트스트랩이 .pth 로도 등록하지만, 다른 인터프리터로 돌릴 때를 위한 보험이다.
    """
    parts = [str(study().root), str(config.REPO_ROOT / "agent-support")]
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _check_call(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    # subprocess가 같은 fd에 직접 쓰므로, 파이프로 넘길 때 순서가 섞이지 않도록 flush
    print(f"    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _progress(name: str, done: int, total: int) -> None:
    mb = done / (1 << 20)
    if total:
        print(f"\r    {name}: {mb:.1f}MB / {total / (1 << 20):.1f}MB", end="")
    else:
        print(f"\r    {name}: {mb:.1f}MB", end="")


def run(tasks: dict[str, list[Action]], target: str, chapter_dir: Path,
        dry_run: bool = False) -> None:
    """타깃 하나를 실행한다. 노트북에서 직접 호출해도 된다."""
    if target not in tasks:
        raise KeyError(f"없는 타깃: {target}. 사용 가능: {', '.join(tasks)}")
    print(f"[{target}] {len(tasks[target])}개 액션", flush=True)
    for i, action in enumerate(tasks[target], 1):
        print(f"  ({i}/{len(tasks[target])}) {action.describe()}", flush=True)
        if not dry_run:
            action.run(chapter_dir)


def main(tasks: dict[str, list[Action]], chapter_dir: Path | str,
         argv: list[str] | None = None) -> int:
    """tasks.py의 진입점. `python tasks.py <target>` 형태로 쓴다."""
    argv = list(sys.argv[1:] if argv is None else argv)
    chapter_dir = Path(chapter_dir).resolve()
    dry_run = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]

    if not argv or argv[0] in ("--list", "-l", "--help", "-h"):
        print(f"사용법: python tasks.py <타깃> [<타깃> ...] [--dry-run]\n")
        print("사용 가능한 타깃:")
        for name, actions in tasks.items():
            print(f"  {name:<14} {len(actions)}개 액션")
        return 0

    for target in argv:
        try:
            run(tasks, target, chapter_dir, dry_run=dry_run)
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 2
    return 0
