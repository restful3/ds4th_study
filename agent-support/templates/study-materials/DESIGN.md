# study-materials 설계 규칙

교재가 바뀔 때마다 학습 코드·데이터를 확보하고, 챕터별 `src/` 를 배치하고, 해설판 기반
노트북을 만드는 반복 작업을 규격화한다. Claude Code 와 Codex 가 같은 CLI 를 호출하므로
두 에이전트의 결과가 갈리지 않는다.

- 기준 완성본: `source/Alessandro Negro - Knowledge Graphs and LLMs in Action/chapter_03_create_your_first_knowledge_graph_from_ontologies/src/ch03/03_chapter_guide.ipynb`
- 절차 정본: `agent-support/procedures/study-materials.md`
- 리포트·발표자료는 별도다. `study-presentation` 스킬을 쓴다.

## 전제

1. 교재는 자주 바뀐다. 새 교재 도입 비용이 낮아야 한다.
2. 각 교재에는 **원서 원문** 과 **해설판** 이 있다. 노트북은 해설판을 근거로 만든다.
3. 스터디는 여러 명이 운영한다. 다른 사람이 저장소를 클론해 한 명령으로 같은 환경을 만들 수 있어야 한다.
4. 대용량 데이터는 git 에 올리지 않는다. 각자 원본에서 받는다.
5. 노트북은 `src/` 에 실행할 코드가 있는 챕터에만 만든다.

## 구조

공용 코드는 저장소에 한 벌만 둔다. 교재마다 복사하면 버그 수정이 전파되지 않는다.

```text
agent-support/
├── studykit/                  공용 파이썬 패키지 (교재 무관, git 한 벌)
│   ├── config.py              study.toml 로드, 교재 루트 경로 해석
│   ├── bootstrap.py           .venv 생성, 공용 자원 배치, .pth 등록
│   ├── actions.py             Makefile 대체 액션 (pip/py/download/unzip/move)
│   ├── cypher.py              Cypher 리스팅 읽기·실행 (.py 는 제외한다)
│   ├── listing_source.py      책 리스팅 → 실제 소스 조각 (repo-file·symbol·줄범위)
│   ├── listing_map.py         챕터↔소스 매핑, 리스팅 번호 오프셋 도출
│   ├── manifest.py            data-manifest.toml 기반 데이터 획득
│   ├── notebook.py            노트북 골격 생성, 그림 attachment 내장
│   └── verify.py              검증 게이트
├── scripts/
│   ├── study-bootstrap.py     ① 교재 등록 + 환경 구축
│   ├── study-map-sources.py   ② 챕터↔소스 매핑·배치
│   ├── study-new-notebook.py  ③ 노트북 골격 생성
│   └── study-verify.py        ④ 검증 게이트
└── templates/study-materials/  study.toml · setup_env.py · DESIGN.md
```

교재 폴더에는 설정과 산출물만 둔다. 나머지는 부트스트랩이 재생성하므로 git 에서 제외한다.

```text
source/<교재>/
├── study.toml            교재 설정 (추적)
├── setup_env.py          studykit 을 호출하는 shim (추적)
├── data-manifest.toml    데이터 획득 매니페스트 (추적)
├── errata.toml           교재 오류 정본 (추적)
├── .venv/                교재별 격리 가상환경 (무시)
├── code/                 업스트림 저장소 클론 (무시)
├── util/ config.ini      업스트림 공용 자원 (무시)
├── dataset/ data/        데이터 (무시)
└── chapter_NN_*/src/chXX/
    ├── (업스트림 챕터 코드)
    ├── tasks.py          Makefile 을 옮긴 실행 태스크
    └── NN_chapter_guide.ipynb
```

`.venv` 를 교재별로 두는 이유는 챕터별 `requirements` 가 서로 다른 버전을 pin 하기
때문이다. 교재 사이에 이 충돌이 번지면 안 된다.

## 워크플로

| 단계 | CLI | 산출물 |
| --- | --- | --- |
| ① 교재 등록 | `study-bootstrap.py <교재경로>` | `studies.toml` 등록, `.venv`, `.pth`, 공용 자원 |
| ② 매핑·배치 | `study-map-sources.py <교재경로>` | `chapter_*/src/chXX/`, `tasks.py`, 리스팅 오프셋 표 |
| ③ 노트북 골격 | `study-new-notebook.py <챕터경로>` | `TODO(agent)` 가 박힌 노트북 골격 |
| ④ 검증 | `study-verify.py <교재경로>` | 게이트 결과 |

③ 이후 에이전트가 서술 셀을 채운다. 기계적으로 확인할 수 있는 것은 스크립트가 하고,
판단이 필요한 것만 에이전트가 한다. 이 경계를 지키는 이유는 실측이다 — 첫 제작에서
발생한 오류 다섯 건 중 네 건이 스크립트로 잡히는 종류였다.

## 매핑은 추측하지 않는다

업스트림 저장소는 MEAP 번호를 쓰는 경우가 있어 디렉터리 이름과 책 챕터가 어긋난다.
디렉터리 개수와 챕터 수를 맞춰 짝지으면 틀린다. 실제로 그렇게 해서 업스트림 `ch05`(miRNA)를
책 4장에, 업스트림 `ch06`(BBC+spaCy)을 책 5장에 잘못 넣었다.

**이 저장소의 소스 폴더는 책 장 번호를 쓴다.** 업스트림 이름과의 변환은 `study.toml` 의
`[mapping.upstream_dirs]` 가 정본이고, 업스트림 산출물과 짝지을 때 이 표를 쓴다.

② 는 챕터 원문 md 의 **고유 키워드 빈도** 로 대응을 검증한다. 예: `Neosemantics` 가
3장에만 7회, `PPI network` 가 4장에만 16회, `fraud` 가 10장에만 114회. 신뢰도가 기준
미달이면 스크립트가 실패하고 사람의 확인을 요구한다. 최종 출간본에 대응 챕터가 없는
디렉터리는 `meap-only/` 로 분리한다.

리스팅 번호 오프셋은 **챕터마다 다르다.** 3장은 −3(책 3.15 = 파일 `3.18 - ...`),
4장은 0 이다. 한 챕터에서 얻은 값을 다른 챕터에 적용하면 안 된다. ② 가 리스팅 제목과
파일명을 토큰 유사도로 정렬해 챕터별로 도출한다.

업스트림 리스팅 파일이 **0바이트이거나 다른 파일과 중복** 인 경우가 있다. ② 가 이를
탐지해 해설판 본문에서 가져온 대체본을 등록한다. 해설판 본문에도 실행하면 그래프를
오염시키는 코드가 있을 수 있다(바인딩 없는 변수에 `MERGE`). 골격은 이런 리스팅에 경고
주석을 달고 실행 전후 노드 수를 대조하게 한다.

## 데이터 획득

`data-manifest.toml` 이 정본이다. NAS 미러나 사내 공유에 의존하지 않는다. 참여자가
바뀌고 교재가 바뀌는 환경에서 인프라 의존은 오래 가지 않고, UMLS 처럼 라이선스상
재배포가 제한되는 데이터도 있다.

```toml
[[dataset]]
chapter = "ch07"
name = "SNOMED CT US Edition"
url = "https://uts-ws.nlm.nih.gov/download?url=...&apiKey=<KEY>"
dest = "dataset/ontology/snomed/SnomedCT_....zip"
api_key_env = "UMLS_API_KEY"
api_key_howto = "https://uts.nlm.nih.gov/uts/ 로그인 후 Get Your API Key"
approx_size = "1.2GB"
unzip_to = "dataset/ontology/snomed"
```

이미 있는 파일은 건너뛴다. 키가 필요한 항목은 발급 방법을 함께 출력한다.

## 노트북 규격

`src/` 에 실행할 코드가 있는 챕터에만 만든다. 파일명은 `<책챕터번호>_chapter_guide.ipynb`.

셀 순서는 고정한다.

1. **헤더 표** — 책·해설판·이 폴더·원서 저장소·리스팅 오프셋. 연속된 `**라벨**: 값`
   줄은 렌더러에 따라 한 단락으로 합쳐지므로 **표** 로 만든다.
2. **실행 환경** — `setup_env.py`, 필요한 외부 서비스(Docker 명령 포함), 에디션 제약.
3. **환경 점검 코드** — 이 셀이 통과하면 이후 전부 실행 가능해야 한다.
4. **리스팅 헬퍼 코드** — 책 번호로 원문을 보고 실행한다. 손상된 리스팅의 대체본을 포함한다.
5. **본문 절** — 해설판 절 구성을 따른다. 개념 설명 마크다운 + 리스팅 실행 코드를 번갈아 둔다.
6. **실습** — 해설판의 Exercise 를 구현한다. 책의 실습 번호·이름을 유지한다.
7. **요약** — 실측 수치 표(책 값과 나란히), 핵심 용어, 참고 링크, 다음 장 연결.

규칙:

- **그림은 attachment 로 내장한다.** HTML `<img src="원격URL">` 은 태그를 걸러내는
  뷰어에서 안 보이고, 상대경로는 `nbconvert` 가 해석하지 못했다. 두 방식 모두 실패했다.
  attachment 키는 해시 파일명이 아니라 `fig3-1` 처럼 그림 번호로 둔다.
- **저장소 안의 대상은 상대경로로 링크한다.** GitHub URL 은 푸시하기 전에는 열리지
  않아 "노트북을 만들려면 먼저 푸시해야 한다"는 순서 역전을 만든다. 외부 사이트만 URL 로 둔다.
- **오래 걸리는 셀은 실측 시간을 적는다.** 배치 없이 25만 건을 갱신하는 리스팅이 8분
  걸린 사례가 있다. 이미 반영된 상태면 건너뛰는 가드를 둔다.
- **데이터 드리프트를 명시한다.** 책 집필 시점과 현재 데이터가 다르다. 요약 셀에
  실측값과 책 값을 나란히 적고, 하위 순위가 다를 수 있음을 밝힌다.
- **자체 완결성.** 노트북만 읽어도 챕터를 이해할 수 있어야 한다. 코드 셀마다 앞에
  개념 설명을, docstring 에는 "왜 이 코드가 이렇게 생겼는지"를 적는다.

## 검증 게이트

하나라도 실패하면 미완성이다. `study-verify.py` 가 전수 실행한다.

```text
nbformat validate 통과
노트북 실행 오류 0
상대경로 링크 대상 전부 존재
외부 URL 응답 확인 (봇차단 403 은 화이트리스트)
그림 참조 수 = attachment 수
Makefile 타깃·액션 순서 = tasks.py 등가
리스팅 참조가 실제 파일 또는 해설판 대체본으로 해결됨
```

링크는 응답 코드만 보고 렌더링을 판단하지 않는다. 첫 제작에서 `curl` 200 만 확인하고
이미지가 표시된다고 결론 냈으나 실제로는 한 장도 보이지 않았다.

## 환경 고정

`study.toml` 이 파이썬 버전을 고정한다. 업스트림이 특정 버전만 테스트했거나 의존성에
휠이 없는 경우가 있다 — `scispacy` 가 요구하는 `nmslib` 는 3.11+ 에 휠이 없어 빌드가 깨진다.

- `uv venv` 는 기본적으로 pip 을 넣지 않으므로 `--seed` 가 필요하다.
- 업스트림 코드가 루트의 공용 패키지를 임포트하고 그 옆의 설정 파일을 읽는 구조라면,
  site-packages 에 `.pth` 로 교재 루트를 등록한다. `PYTHONPATH` 없이 어느 cwd 에서도 동작한다.
- `.pth` 는 gitignore 대상이므로 `.venv` 를 새로 만들 때마다 부트스트랩이 다시 쓴다.
- 챕터별 `requirements` 가 서로 다른 버전을 pin 하므로 챕터를 옮길 때 그 챕터의
  `init` 을 다시 돌려야 한다. 노트북에 이 안내를 남긴다.
- `subprocess` 가 같은 fd 에 직접 쓰므로 파이프로 넘길 때 출력 순서가 섞인다. 액션의
  `print` 는 `flush=True` 로 둔다.

## 금지

- 챕터↔소스 매핑을 개수나 제목 인상으로 추측하는 것
- 한 챕터의 리스팅 오프셋을 다른 챕터에 적용하는 것
- 업스트림 원본 파일을 수정하는 것 — `tasks.py` 로 대체하고 원본은 그대로 둔다
- 책이 가르치는 Cypher·프롬프트 리스팅을 파이썬으로 다시 쓰는 것 — 러너로 실행한다
- 공용 `studykit` 을 교재 폴더로 복사하는 것
- 검증 게이트를 통과하지 않은 노트북을 완성으로 보고하는 것
