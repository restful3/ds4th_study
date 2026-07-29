#!/usr/bin/env python3
"""이 교재가 쓰는 Neo4j 컨테이너를 확인하고 전환한다.

    python3 neo4j_env.py status        # 어느 쪽이 떠 있나 · 에디션 · 플러그인 · DB
    python3 neo4j_env.py use ch03      # 3장용 Community + n10s 로
    python3 neo4j_env.py use ch04      # 4장 이후용 Enterprise 로

3장은 Neosemantics(`n10s`)가 필요하고 4장 이후는 Enterprise 가 필요한데, 두 요구는
배타적이다. `n10s` 를 Enterprise 에 넣으면 Jackson 버전이 충돌해
`NoSuchMethodError: JsonProperty.isRequired()` 로 기동에 실패한다(4장 노트북이 플러그인을
하나씩 시험해 확인했다). 그래서 컨테이너를 둘로 나누고, 같은 포트를 쓰므로 한 번에
하나만 띄운다.

**전환은 데이터를 지우지 않는다.** 두 컨테이너가 각자의 named volume 을 쓰므로 멈춘
쪽 데이터는 그대로 남아 있고, 되돌리면 다시 보인다.

아래 PROFILES 가 `docker run` 정의의 정본이다. 노트북 마크다운의 명령은 설명용
사본이므로, 값을 바꿀 때는 이 파일을 바꾼다.

표준 라이브러리만 쓰므로 시스템 python3 로 실행하면 된다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field

PORTS = ("127.0.0.1:7474:7474", "127.0.0.1:7687:7687")


class UnknownTarget(ValueError):
    """장 번호도 에디션 이름도 아니다."""


class UndeclaredChapter(ValueError):
    """그 장의 외부 서비스 요구가 아직 노트북에 선언되지 않았다."""


class LicenceNotAccepted(RuntimeError):
    """Enterprise 컨테이너를 새로 만들려면 사용자가 평가 라이선스에 동의해야 한다."""


@dataclass(frozen=True)
class Profile:
    name: str
    container: str
    image: str
    data_volume: str
    plugins: tuple[str, ...]
    procedures: str            # unrestricted 와 allowlist 에 같은 값을 쓴다
    needs_licence: bool = False
    extra_env: tuple[str, ...] = ()
    ports: tuple[str, ...] = PORTS

    @property
    def unrestricted(self) -> str:
        return self.procedures

    @property
    def allowlist(self) -> str:
        return self.procedures


PROFILES: dict[str, Profile] = {
    "community": Profile(
        name="community",
        container="kglm-neo4j",
        image="neo4j:5.26",
        data_volume="kglm-neo4j-data",
        plugins=("graph-data-science", "n10s", "apoc"),
        procedures="gds.*,apoc.*,n10s.*",
    ),
    "enterprise": Profile(
        name="enterprise",
        container="kglm-neo4j-ee",
        image="neo4j:5.26-enterprise",
        data_volume="kglm-neo4j-ee-data",
        plugins=("graph-data-science", "apoc"),
        procedures="gds.*,apoc.*",
        needs_licence=True,
        # 4.6·15장의 CREATE DATABASE ... seedUri 가 이 provider 를 요구한다
        extra_env=("NEO4J_dbms_databases_seed__from__uri__providers=URLConnectionSeedProvider",),
    ),
}

# 장별 요구. 노트북 「실행 환경」 절이 근거이고, 선언되지 않은 장은 추측하지 않는다.
CHAPTER_PROFILES: dict[str, str | None] = {
    "ch01": None,   # 코드 없음
    "ch02": None,   # 노트북이 "3장부터 필요" 로 명시
    "ch03": "community",
    "ch04": "enterprise",
    "ch05": None,
    "ch06": "enterprise",
    "ch09": None,
    "ch10": "enterprise",
    "ch11": None,
    "ch12": None,   # 노트북이 "이 장은 Neo4j를 쓰지 않는다" 로 명시
    "ch13": "enterprise",
    "ch15": "enterprise",
}
# 노트북이 draft 이거나 아직 없어서 요구가 선언되지 않은 장
UNDECLARED = ("ch07", "ch08", "ch14")


def profile_for(target: str) -> Profile | None:
    """`ch03`·`3`·`community` 같은 표기를 프로파일로 해석한다.

    Neo4j 가 필요 없는 장은 None, 요구가 선언되지 않은 장은 UndeclaredChapter.
    """
    token = target.strip().lower().removeprefix("chapter_").removeprefix("ch")
    if target.strip().lower() in PROFILES:
        return PROFILES[target.strip().lower()]
    if not token.isdigit():
        raise UnknownTarget(f"알 수 없는 대상: {target!r} (예: ch03, 4, community, enterprise)")
    chapter = f"ch{int(token):02d}"
    if chapter in UNDECLARED:
        raise UndeclaredChapter(
            f"{chapter} 의 외부 서비스 요구가 노트북에 아직 선언되지 않았다. "
            "해당 노트북 「실행 환경」 절을 먼저 채워라."
        )
    if chapter not in CHAPTER_PROFILES:
        raise UnknownTarget(f"{chapter} 은 이 교재에 없다")
    name = CHAPTER_PROFILES[chapter]
    return PROFILES[name] if name else None


def docker_run_argv(profile: Profile) -> list[str]:
    """이 프로파일의 컨테이너를 만드는 명령. docker run 정의의 정본."""
    argv = ["docker", "run", "-d", "--name", profile.container]
    for port in profile.ports:
        argv += ["-p", port]
    env = []
    if profile.needs_licence:
        env.append("NEO4J_ACCEPT_LICENSE_AGREEMENT=eval")
    env.append("NEO4J_AUTH=neo4j/password")
    env.append("NEO4J_PLUGINS=" + json.dumps(list(profile.plugins), separators=(",", ":")))
    env.append(f"NEO4J_dbms_security_procedures_unrestricted={profile.unrestricted}")
    env.append(f"NEO4J_dbms_security_procedures_allowlist={profile.allowlist}")
    env.append("NEO4J_server_memory_heap_max__size=2G")
    env.append("NEO4J_server_memory_pagecache_size=1G")
    env.extend(profile.extra_env)
    for item in env:
        argv += ["-e", item]
    argv += ["-v", f"{profile.data_volume}:/data", profile.image]
    return argv


@dataclass(frozen=True)
class Step:
    action: str                      # stop · start · create
    container: str
    argv: list[str] = field(default_factory=list)


def plan(profile: Profile, state: dict[str, str], *, accept_licence: bool = False) -> list[Step]:
    """현재 컨테이너 상태에서 목표 프로파일로 가는 단계.

    state 는 컨테이너 이름 → docker 상태(`running`·`exited` 등). 키가 없으면 아직 만들지 않은 것.
    """
    if state.get(profile.container) == "running":
        return []

    creating = profile.container not in state
    if creating and profile.needs_licence and not accept_licence:
        raise LicenceNotAccepted(
            f"{profile.container} 를 새로 만들려면 Neo4j Enterprise 평가 라이선스에 동의해야 한다.\n"
            "평가 목적에 한해 무료다 — 조건: https://neo4j.com/terms/enterprise_us/\n"
            "동의하면 --accept-license 를 붙여 다시 실행하거나, 아래 명령을 직접 실행하라.\n\n"
            "  " + " ".join(docker_run_argv(profile))
        )

    steps = [
        Step("stop", other.container)
        for other in PROFILES.values()
        if other.container != profile.container and state.get(other.container) == "running"
    ]
    if creating:
        steps.append(Step("create", profile.container, docker_run_argv(profile)))
    else:
        steps.append(Step("start", profile.container))
    return steps


# ---------------------------------------------------------------- docker 호출


def docker(*args: str, check: bool = True) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} 실패:\n{result.stderr.strip()}")
    return result.stdout.strip()


def docker_state() -> dict[str, str]:
    """이 교재의 컨테이너만 골라 이름 → 상태."""
    ours = {p.container for p in PROFILES.values()}
    out = docker("ps", "-a", "--format", "{{.Names}}\t{{.State}}")
    state = {}
    for line in out.splitlines():
        name, _, status = line.partition("\t")
        if name in ours:
            state[name] = status
    return state


def wait_until_ready(profile: Profile, timeout: int = 120) -> bool:
    """cypher-shell 로 실제 질의가 통할 때까지 기다린다."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = subprocess.run(
            ["docker", "exec", profile.container, "cypher-shell",
             "-u", "neo4j", "-p", "password", "--format", "plain", "RETURN 1"],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            return True
        time.sleep(2)
    return False


def chapters_served(profile: Profile) -> list[str]:
    return [ch for ch, name in sorted(CHAPTER_PROFILES.items()) if name == profile.name]


def cmd_status() -> int:
    state = docker_state()
    print("컨테이너")
    for profile in PROFILES.values():
        status = state.get(profile.container, "없음")
        mark = "▶" if status == "running" else " "
        chapters = ", ".join(chapters_served(profile))
        print(f"  {mark} {profile.container:<16} {status:<10} {profile.image:<24} {chapters}")

    running = [p for p in PROFILES.values() if state.get(p.container) == "running"]
    if not running:
        print("\n떠 있는 컨테이너가 없다. `python3 neo4j_env.py use ch03` 또는 `use ch04`.")
        return 0

    profile = running[0]
    print(f"\n{profile.container} 에 접속해 확인")
    query = (
        "CALL dbms.components() YIELD versions, edition "
        "RETURN versions[0] + ' (' + edition + ')' AS neo4j;"
        "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'n10s' OR name STARTS WITH 'gds' "
        "OR name STARTS WITH 'apoc' RETURN split(name,'.')[0] AS ns, count(*) AS n ORDER BY ns;"
        "SHOW DATABASES YIELD name, currentStatus RETURN name, currentStatus;"
    )
    probe = subprocess.run(
        ["docker", "exec", profile.container, "cypher-shell",
         "-u", "neo4j", "-p", "password", "--format", "plain", query],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        print("  아직 기동 중이거나 접속할 수 없다:", probe.stderr.strip().splitlines()[-1:] or "")
        return 0
    for line in probe.stdout.strip().splitlines():
        print("  ", line)

    idle = [p for p in PROFILES.values() if p is not profile]
    if idle:
        other = idle[0]
        print(f"\n{other.container} 는 정지 상태다 — {', '.join(chapters_served(other))} 을 "
              f"돌리려면 `use {chapters_served(other)[0]}`. 데이터는 "
              f"{other.data_volume} 볼륨에 남아 있다.")
    return 0


def cmd_use(target: str, accept_license: bool) -> int:
    try:
        profile = profile_for(target)
    except (UnknownTarget, UndeclaredChapter) as exc:
        print(exc, file=sys.stderr)
        return 2

    if profile is None:
        print(f"{target} 은 Neo4j 를 쓰지 않는다. 컨테이너를 띄울 필요가 없다.")
        return 0

    try:
        steps = plan(profile, docker_state(), accept_licence=accept_license)
    except LicenceNotAccepted as exc:
        print(exc, file=sys.stderr)
        return 2

    if not steps:
        print(f"{profile.container} 가 이미 떠 있다 ({profile.name}). 할 일이 없다.")
        return 0

    for step in steps:
        if step.action == "stop":
            print(f"■ {step.container} 정지 (데이터는 볼륨에 남는다)")
            docker("stop", step.container)
        elif step.action == "start":
            print(f"▶ {step.container} 시작")
            docker("start", step.container)
        else:
            print(f"＋ {step.container} 생성")
            print("  " + " ".join(step.argv))
            subprocess.run(step.argv, check=True)

    print(f"\n{profile.container} 기동 대기 중...")
    if not wait_until_ready(profile):
        print("시간 안에 준비되지 않았다. `docker logs " + profile.container + "` 를 확인하라.",
              file=sys.stderr)
        return 1
    print(f"준비 완료 — {profile.name}, {', '.join(chapters_served(profile))} 실행 가능")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="어느 컨테이너가 떠 있는지, 에디션·플러그인·DB 확인")
    use = sub.add_parser("use", help="해당 장에 필요한 컨테이너로 전환")
    use.add_argument("target", help="ch03 · 4 · community · enterprise")
    # 플래그는 Neo4j 환경변수(NEO4J_ACCEPT_LICENSE_AGREEMENT)의 미국식 표기를 따른다
    use.add_argument("--accept-license", action="store_true",
                     help="Neo4j Enterprise 평가 라이선스에 동의한다 (신규 생성 시에만 필요)")
    args = parser.parse_args()

    if args.command == "status":
        return cmd_status()
    return cmd_use(args.target, args.accept_license)


if __name__ == "__main__":
    sys.exit(main())
