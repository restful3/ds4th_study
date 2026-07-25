# Codex·Claude Code로 스터디 참여하기

참가자는 리포트·발표자료 배포 절차를 외울 필요가 없다. 저장소 루트에서 Codex 또는 Claude Code를 시작하고 담당 날짜와 챕터를 자연어로 알려주면 된다.

검증 도구 실행에는 Python 3.11 이상이 필요하다. 별도 Python 패키지는 설치하지 않는다.

이 문서의 `bash` 블록은 macOS·Linux와 Windows의 Git Bash 기준이고, `powershell` 블록은 Windows PowerShell 기준이다. Windows에서는 [Git for Windows](https://gitforwindows.org/)와 [python.org 설치본](https://www.python.org/downloads/windows/)을 권장한다.

Windows에서 Python 실행 명령은 `python3`이 아니라 `py -3`이다(`py`가 없으면 `python`). Git Bash를 쓰더라도 마찬가지다. Windows에서 `python3`을 그대로 입력하면 Python이 설치돼 있어도 앱 실행 별칭 때문에 Microsoft Store가 열릴 수 있다.

## 1. 저장소 받기

### Windows 사전 설정

Windows 사용자는 저장소를 받기 전에 다음을 한 번만 실행한다.

```powershell
git config --global core.longpaths true
git config --global core.autocrlf input
```

이 저장소에서 가장 긴 파일 경로는 215자다. Windows 기본 경로 길이 제한은 260자여서 clone 위치에 따라 checkout이 중간에 실패할 수 있으므로 `core.longpaths`가 필요하다. `core.autocrlf input`은 편집하지 않은 HTML·CSS가 줄바꿈 차이만으로 통째로 변경된 것처럼 보이는 일을 막는다.

### 내려받기

저장소가 크므로 새로 받을 때는 전체 이력을 내려받지 않는 sparse checkout을 권장한다.

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/restful3/ds4th_study.git
cd ds4th_study
git sparse-checkout set \
  docs agent-support .agents .claude .github \
  "source/Alessandro Negro - Knowledge Graphs and LLMs in Action"
```

PowerShell에서는 줄 잇기 문자가 다르므로 `git sparse-checkout set`을 한 줄로 적는다.

```powershell
git clone --filter=blob:none --sparse --depth=1 https://github.com/restful3/ds4th_study.git
cd ds4th_study
git sparse-checkout set docs agent-support .agents .claude .github "source/Alessandro Negro - Knowledge Graphs and LLMs in Action"
```

다른 스터디를 준비한다면 마지막 경로를 `agent-support/studies.toml`의 해당 `materials_path`로 바꾼다. 이미 전체 저장소가 있다면 이 단계는 생략한다.

## 2. 에이전트 시작하기

Codex 또는 Claude Code를 저장소 루트에서 시작한다. 프로젝트를 신뢰할지 묻는 경우 이 저장소와 변경사항을 검토한 뒤 결정한다.

Windows에서 작업한다면 첫 요청에 그 사실을 함께 알려준다. 저장소의 절차 문서가 검증 명령을 `python3` 기준으로 적어 두었으므로, 에이전트가 이를 `py -3`으로 바꿔 실행해야 한다.

처음에는 다음처럼 요청할 수 있다.

```text
이 저장소가 처음이야. 현재 진행 중인 스터디와 내 발표 준비 방법을 안내해줘.
```

```text
나는 2026년 7월 25일 Chapter 1~2 발표 담당이야.
먼저 자료와 일정을 확인하고 원자료를 감사한 다음 상세 HTML 리포트를 완성·검증해줘. 그 리포트의 내용·표·SVG를 적극 반영한 HTML 발표자료를 만들어줘.
```

```text
내 발표자료를 Webex 발표 기준으로 검토해줘.
깨진 이미지, 화면 넘침, 출처, 발표 시간과 GitHub Pages 경로까지 확인해줘.
```

에이전트는 기본적으로 로컬 파일만 준비하고 검증한다. 커밋, 푸시 또는 PR이 필요하면 모든 작업을 마친 뒤 명시적으로 요청한다. 요청 방법은 [4. 게시하기](#4-게시하기)에 있다.

새 회차는 별도 지시가 없으면 Chapter 1 완성본을 기준으로 `raw → study-report-v1 → study-deck-v1` 순서로 만든다. 리포트가 내용·출처·도형·브라우저·PDF 품질 게이트를 통과한 뒤에만 발표자료를 파생한다. 리포트에는 자동 목차·테마·인쇄·전체화면 이미지 줌/팬이, 발표자료에는 18–30장 기본 흐름·키보드 이동·자동 목차·진행률·전체화면 기능이 포함된다.

가장 짧게는 다음처럼 요청하면 된다.

```text
나는 <날짜> <챕터> 발표 담당이야. 프로젝트의 study-presentation 스킬과 Chapter 1 완성 기준을 사용해 원자료를 먼저 상세 리포트로 완성·검증하고, 그 리포트의 내용·표·SVG를 적극 반영한 HTML 슬라이드와 자동 목차를 만든 뒤 실제 브라우저와 PDF까지 검증해줘.
```

## 3. 결과 확인하기

에이전트가 검증을 마치면 로컬 사이트를 열 수 있다.

```bash
python3 -m http.server 8000 -d docs
```

```powershell
py -3 -m http.server 8000 -d docs
```

브라우저에서 `http://localhost:8000/`을 연다. 서버는 `Ctrl+C`로 멈춘다. 회차별 경로는 다음 형식이다.

```text
http://localhost:8000/studies/<study-slug>/presentations/<session-slug>/
```

위 주소는 발표자료이며, 상세 리포트는 같은 경로의 `report.html`이다.

### 슬라이드 고치기

마음에 들지 않는 슬라이드는 발표자료 전체를 다시 만들 필요 없이 장 단위로 고칠 수 있다. 슬라이드 번호나 제목으로 지목해 요청한다.

```text
7번 슬라이드 글자가 화면을 넘쳐. 두 장으로 나눠줘.
```

```text
"전체 흐름" 슬라이드에 리포트의 그림 2를 발표 화면에서 읽히는 크기로 넣어줘.
```

```text
마지막 요약 슬라이드를 세 문장으로 줄이고, 앞의 확인 질문 슬라이드와 순서를 바꿔줘.
```

에이전트는 이때 생성 스크립트를 다시 실행하지 않고 해당 회차 폴더만 편집한다. 목차는 각 슬라이드의 `aria-label`에서 자동 생성되므로 따로 손댈 필요가 없다.

리포트에 없는 새 주장이나 근거가 필요하다면 슬라이드에 먼저 쓰지 말고 `report.html`을 보완한 뒤 리포트 게이트를 다시 통과시켜 슬라이드로 압축해달라고 요청한다. 발표자료가 리포트보다 앞서 나가지 않게 하는 규칙이다.

## 4. 게시하기

로컬 확인과 수정까지 모두 끝났으면 마지막으로 게시를 요청한다. 에이전트는 스스로 push하지 않으므로, 이 요청을 해야 공개 사이트에 올라간다.

```text
작업 다 끝났어. 검증 다시 한 번 돌리고 커밋한 다음 main에 push해줘.
```

이 저장소의 GitHub Pages는 `main` 브랜치의 `docs/` 폴더를 그대로 게시한다. 따라서 push가 끝나면 별도 배포 조작 없이 보통 1\~2분 안에 공개 사이트에 반영된다.

- 스터디 목록: <https://restful3.github.io/ds4th_study/>
- 회차 발표자료: `https://restful3.github.io/ds4th_study/studies/<study-slug>/presentations/<session-slug>/`
- 회차 상세 리포트: 같은 경로의 `report.html`

push 후에는 GitHub Actions의 `Validate study site`가 검증을 실행하고, 인덱스가 오래됐으면 자동으로 재생성해 커밋한다. 새 회차가 기존 스터디 목록에 자동으로 나타나는 것은 이 때문이다. 이 자동 커밋이 생기면 로컬이 한 커밋 뒤처지므로, 이어서 작업할 때는 `git pull`을 먼저 한다.

게시 뒤에는 Actions 탭에서 워크플로가 통과했는지 확인하고, 공개 URL을 실제 브라우저에서 한 번 연다. 발표 직전에 다시 고쳤다면 push를 한 번 더 해야 공개본에 반영된다. 로컬에서만 고친 내용은 공개 사이트에 없다.

## 스터디가 끝난 뒤

책이 끝나면 학습자료는 `source/`에서 `archive/`로 이동하지만 발표 URL은 바뀌지 않는다. 이 이동은 운영자가 에이전트에게 명시적으로 요청해 수행하며, 참가자가 직접 디렉터리를 옮길 필요는 없다.
