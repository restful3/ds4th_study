"""노트북 그림 내장과 골격 생성.

그림은 반드시 **attachment 로 내장** 한다. Chapter 3 에서 두 방식이 모두 실패했다.

  * HTML <img src="https://raw.githubusercontent.com/..."> — 링크는 200 이었지만
    태그를 걸러내는 뷰어에서 한 장도 보이지 않았다. 응답 코드로 렌더링을 판단하면 안 된다.
  * ![](../../images/x.jpg) 상대경로 — nbconvert --embed-images 가 11장 모두
    경로를 해석하지 못하고 깨진 채 남겼다.

attachment 는 순수 마크다운 문법이고 경로·네트워크에 의존하지 않는다.
키는 64자 해시 파일명이 아니라 그림 번호(fig4-1)로 둬서 읽을 수 있게 한다.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path

#: 마크다운의 그림 참조. attachment 로 바꿀 대상.
RELATIVE_REF = re.compile(r"!\[([^\]]*)\]\((?!attachment:|https?:)([^)]+)\)")

#: 이미 내장된 참조
ATTACHMENT_REF = re.compile(r"!\[([^\]]*)\]\(attachment:([^)]+)\)")


def figure_key(book_chapter: int, index: int) -> str:
    """attachment 키. 해시 파일명 대신 그림 번호로 둔다."""
    return f"fig{book_chapter}-{index}.jpg"


def embed_figures(notebook_path: Path, images_dir: Path,
                  book_chapter: int, dry_run: bool = False) -> dict[str, str]:
    """마크다운의 상대경로 그림 참조를 attachment 내장으로 바꾼다.

    참조 순서대로 fig<장>-1, fig<장>-2 … 키를 붙인다. 이미 attachment 인 참조는
    건드리지 않는다. 반환값은 {키: 원본 파일명}.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    counter = 0

    # 이미 쓰인 번호는 건너뛰어 중복 키를 만들지 않는다
    used = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown":
            used |= set(ATTACHMENT_REF.findall("".join(cell["source"])) and
                        [k for _, k in ATTACHMENT_REF.findall("".join(cell["source"]))])

    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        refs = RELATIVE_REF.findall(text)
        if not refs:
            continue
        attachments = cell.get("attachments", {})
        for alt, target in refs:
            source = (notebook_path.parent / target).resolve()
            if not source.is_file():
                source = images_dir / Path(target).name
            if not source.is_file():
                raise FileNotFoundError(f"그림 파일을 찾지 못했다: {target}")
            counter += 1
            key = figure_key(book_chapter, counter)
            while key in used:
                counter += 1
                key = figure_key(book_chapter, counter)
            used.add(key)
            mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
            attachments[key] = {
                mime: base64.b64encode(source.read_bytes()).decode("ascii")
            }
            text = text.replace(f"![{alt}]({target})", f"![{alt}](attachment:{key})")
            mapping[key] = source.name
        cell["attachments"] = attachments
        cell["source"] = text.splitlines(keepends=True)

    if not dry_run and mapping:
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return mapping


def embed_named_figures(notebook_path: Path, images_dir: Path,
                        figures: dict[str, str]) -> dict[str, str]:
    """미리 정한 {attachment 키: 이미지 파일명} 으로 내장한다.

    그림 번호와 파일의 대응을 사람이 확인한 경우에 쓴다. 마크다운에는
    ![alt](attachment:fig4-1.jpg) 처럼 키를 직접 써 두면 된다.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    embedded: dict[str, str] = {}

    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        keys = {key for _, key in ATTACHMENT_REF.findall("".join(cell["source"]))}
        if not keys:
            continue
        attachments = cell.get("attachments", {})
        for key in sorted(keys):
            if key in attachments:
                continue
            filename = figures.get(key)
            if filename is None:
                raise KeyError(f"{notebook_path.name}: {key} 에 대응하는 이미지가 지정되지 않았다")
            source = images_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"그림 파일이 없다: {source}")
            mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
            attachments[key] = {
                mime: base64.b64encode(source.read_bytes()).decode("ascii")
            }
            embedded[key] = filename
        cell["attachments"] = attachments

    if embedded:
        notebook_path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return embedded


def figure_map_from_explainer(explainer: Path) -> dict[str, str]:
    """해설판에서 {그림 번호: 이미지 파일명} 을 뽑는다.

    해설판은 ![](images/<hash>.jpg) 다음 줄에 "그림 4.7 …" 캡션을 두는 형식이다.
    한 그림이 여러 이미지로 나뉜 경우(4.7a/b/c)는 번호에 a, b, c 를 붙인다.
    """
    lines = explainer.read_text(encoding="utf-8").splitlines()
    pending: list[str] = []
    result: dict[str, str] = {}
    for line in lines:
        image = re.match(r"!\[[^\]]*\]\(images/([^)]+)\)", line.strip())
        if image:
            pending.append(image.group(1))
            continue
        caption = re.match(r"그림\s+(\d+\.\d+)", line.strip())
        if caption and pending:
            number = caption.group(1)
            if len(pending) == 1:
                result[number] = pending[0]
            else:
                for offset, filename in enumerate(pending):
                    result[f"{number}{chr(ord('a') + offset)}"] = filename
            pending = []
    return result
