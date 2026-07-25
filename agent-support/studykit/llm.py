"""OpenAI 호환 엔드포인트 설정을 한 곳에서 읽는다.

교재 코드가 환경변수 이름을 통일해 쓰지 않는다. 조사한 결과 두 계열이 섞여 있다.

    OPENAI_KEY · OPENAI_BASE_URL · OPENAI_MODEL   ch06 importer
    OPENAI_API_KEY                                ch05·ch13 listings, langchain

그래서 한 곳에서 설정을 읽어 **두 형태 모두** 채운다. 그러지 않으면 챕터마다
다른 변수를 수동으로 export 해야 한다.

설정 출처는 study.toml 의 [llm].env_file 이다. 비밀값을 저장소에 복사하지 않고
외부 .env 를 **참조만** 한다. 그 파일은 이 저장소 밖에 있고 git 에 들어가지 않는다.

    from studykit import llm
    llm.configure()                  # 환경변수 주입 (값은 출력하지 않는다)
    print(llm.describe())            # 무엇이 설정됐는지 (키는 마스킹)
"""
from __future__ import annotations

import os
from pathlib import Path

from studykit.config import Study

#: 교재 코드가 쓰는 변수 이름들. 어느 쪽을 읽든 동작하게 전부 채운다.
KEY_ALIASES = ("OPENAI_API_KEY", "OPENAI_KEY")
BASE_ALIASES = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
MODEL_ALIASES = ("OPENAI_MODEL",)


class LLMConfigError(RuntimeError):
    """설정 파일을 찾지 못했거나 필요한 값이 없는 경우."""


def read_env_file(path: Path) -> dict[str, str]:
    """`KEY=value` 형식을 읽는다. 값은 반환하되 절대 출력하지 않는다."""
    if not path.is_file():
        raise LLMConfigError(f"설정 파일이 없다: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def configure(study: Study | None = None, model: str | None = None,
              env_file: Path | str | None = None) -> dict[str, str]:
    """OpenAI 호환 설정을 환경변수로 주입한다.

    이미 환경에 있는 값은 덮어쓰지 않는다 — 사용자가 직접 export 한 것을 존중한다.
    반환값은 마스킹된 요약이다.
    """
    if env_file is None:
        if study is None:
            raise LLMConfigError("study 또는 env_file 중 하나가 필요하다")
        env_file = study.llm_env_file
    if env_file is None:
        raise LLMConfigError(
            "study.toml 에 [llm].env_file 이 없다. OpenAI 호환 엔드포인트 설정을\n"
            "담은 파일 경로를 적어라 (그 파일은 저장소 밖에 두어 git 에 넣지 않는다)."
        )

    source = read_env_file(Path(env_file).expanduser())
    key = source.get("OPENAI_API_KEY") or source.get("OPENAI_KEY")
    base = source.get("OPENAI_BASE_URL") or source.get("OPENAI_API_BASE")
    chosen = model or os.environ.get("OPENAI_MODEL") or source.get("OPENAI_MODEL")

    if not key:
        raise LLMConfigError(f"{env_file} 에 OPENAI_API_KEY 가 없다")

    for name in KEY_ALIASES:
        os.environ.setdefault(name, key)
    if base:
        for name in BASE_ALIASES:
            os.environ.setdefault(name, base)
    if chosen:
        for name in MODEL_ALIASES:
            os.environ[name] = chosen

    return describe()


def describe() -> dict[str, str]:
    """현재 설정 요약. 키는 마스킹한다."""
    key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL", "(기본값)"),
        "model": os.environ.get("OPENAI_MODEL", "(코드 기본값)"),
        "key": f"설정됨 ({len(key)}자)" if key else "없음",
    }


def available_models(timeout: int = 25) -> list[str]:
    """엔드포인트가 제공하는 모델 목록. 설정 확인용이다."""
    import json
    import urllib.request

    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        raise LLMConfigError("configure() 를 먼저 호출하라")
    request = urllib.request.Request(
        f"{base.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return [item.get("id", "") for item in payload.get("data", [])]


def check(timeout: int = 60) -> str:
    """엔드포인트가 실제로 응답하는지 한 번 확인한다."""
    import json
    import urllib.request

    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not (base and key and model):
        raise LLMConfigError("configure() 를 먼저 호출하라")

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 16,
    }).encode()
    request = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"].strip()


def override_module_model(module, model: str | None = None,
                          attributes: tuple[str, ...] = ("OPENAI_MODEL", "MODEL")) -> str:
    """리스팅 모듈이 하드코딩한 모델 상수를 바꿔치기한다.

    교재 리스팅은 모델명을 모듈 상수로 박아둔다 (예: OPENAI_MODEL = "gpt-4o-mini").
    OpenAI 호환 프록시가 그 모델을 제공하지 않으면 그대로는 실패한다.
    **원본 파일을 수정하지 않고** 임포트한 뒤 상수만 교체한다 — 업스트림 코드를
    건드리지 않는다는 원칙을 지키면서 실행 가능하게 만드는 방법이다.

    반환값은 적용된 모델명.
    """
    chosen = model or os.environ.get("OPENAI_MODEL")
    if not chosen:
        raise LLMConfigError("모델을 정하지 못했다. configure() 를 먼저 호출하라")
    replaced = []
    for name in attributes:
        if hasattr(module, name):
            setattr(module, name, chosen)
            replaced.append(name)
    if not replaced:
        raise AttributeError(
            f"{module.__name__} 에 {attributes} 중 어느 것도 없다. "
            f"모델 상수 이름을 확인하라."
        )
    return chosen
