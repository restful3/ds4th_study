# Codex·Claude Code로 스터디 참여하기

참가자는 리포트·발표자료 배포 절차를 외울 필요가 없다. 저장소 루트에서 Codex 또는 Claude Code를 시작하고 담당 날짜와 챕터를 자연어로 알려주면 된다.

검증 도구 실행에는 Python 3.11 이상이 필요하다. 별도 Python 패키지는 설치하지 않는다.

## 1. 저장소 받기

저장소가 크므로 새로 받을 때는 전체 이력을 내려받지 않는 sparse checkout을 권장한다.

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/restful3/ds4th_study.git
cd ds4th_study
git sparse-checkout set \
  docs agent-support .agents .claude .github \
  "source/Alessandro Negro - Knowledge Graphs and LLMs in Action"
```

다른 스터디를 준비한다면 마지막 경로를 `agent-support/studies.toml`의 해당 `materials_path`로 바꾼다. 이미 전체 저장소가 있다면 이 단계는 생략한다.

## 2. 에이전트 시작하기

Codex 또는 Claude Code를 저장소 루트에서 시작한다. 프로젝트를 신뢰할지 묻는 경우 이 저장소와 변경사항을 검토한 뒤 결정한다.

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

에이전트는 기본적으로 로컬 파일만 준비하고 검증한다. 커밋, 푸시 또는 PR이 필요하면 마지막에 명시적으로 요청한다.

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

브라우저에서 `http://localhost:8000/`을 연다. 회차별 경로는 다음 형식이다.

```text
http://localhost:8000/studies/<study-slug>/presentations/<session-slug>/
```

위 주소는 발표자료이며, 상세 리포트는 같은 경로의 `report.html`이다.

회차 폴더가 main에 반영되면 GitHub Actions가 인덱스를 검증하고 필요 시 자동 재생성하므로, 새 회차는 기존 스터디 목록에 자동으로 나타난다.

## 스터디가 끝난 뒤

책이 끝나면 학습자료는 `source/`에서 `archive/`로 이동하지만 발표 URL은 바뀌지 않는다. 이 이동은 운영자가 에이전트에게 명시적으로 요청해 수행하며, 참가자가 직접 디렉터리를 옮길 필요는 없다.
