# 교재 학습자료 절차

교재가 바뀔 때마다 학습 코드·데이터를 확보하고, 챕터별 `src/` 를 배치하고, 해설판 기반
노트북을 만드는 반복 작업의 정본이다. Claude Code 와 Codex 가 같은 CLI 를 호출하므로
결과가 갈리지 않는다.

- 설계 규칙: `agent-support/templates/study-materials/DESIGN.md`
- 기준 완성본: `source/Alessandro Negro - Knowledge Graphs and LLMs in Action/chapter_03_create_your_first_knowledge_graph_from_ontologies/src/ch03/03_chapter_guide.ipynb`
  그리고 같은 교재 `chapter_04_.../src/ch04/04_chapter_guide.ipynb`
- 리포트·발표자료는 이 절차가 아니다. `study-presentation` 스킬을 쓴다.

## 0. 시작 전 확인

```bash
git status --short
cat agent-support/studies.toml
cat "source/<교재>/study.toml"
python3 agent-support/scripts/study-verify.py "source/<교재>" --no-urls
```

게이트가 이미 실패 중이면 그것을 먼저 고친다. 새 작업을 얹지 않는다.

## 1. 교재 등록과 환경 구축

```bash
python3 "source/<교재>/setup_env.py"            # 없는 것만 채운다 (여러 번 실행해도 안전)
python3 "source/<교재>/setup_env.py" --recreate # .venv 를 다시 만든다
```

`setup_env.py` 는 `agent-support/templates/study-materials/setup_env.py.template` 을 복사한
shim 이고, 실제 동작은 `studykit.bootstrap` 에 있다. 하는 일은 다섯 가지다.

1. 업스트림 교재 코드 저장소를 `code/` 로 클론
2. `code/` 의 공용 자원(`util/`·`config.ini`·`data/`)을 교재 루트로 올림
3. `dataset/` 과 그 자체 `.gitignore` 생성
4. `.venv` 생성 (`study.toml` 의 `[study].python`)
5. site-packages 에 `_studykit.pth` 작성 + 주피터 커널 등록

### 환경에서 반드시 알아야 할 것

- **파이썬 버전을 올리지 마라.** 업스트림이 테스트한 버전이 `study.toml` 에 있다.
  `scispacy` 가 요구하는 `nmslib` 는 3.11+ 에 휠이 없어 빌드가 깨진다.
- `uv venv` 는 기본적으로 pip 을 넣지 않는다. `--seed` 가 필요하다.
- `.pth` 는 `.gitignore` 대상이다. `.venv` 를 새로 만들면 사라지므로 부트스트랩을 다시 돌린다.
- `tomllib` 은 3.11+ 에만 있다. 3.10 교재에서는 `tomli` 폴백이 필요하고 부트스트랩이 설치한다.
- **챕터별 `requirements.lock` 이 서로 다른 버전을 pin 한다** (neo4j 4.4 vs 5.8, openai 1.12 vs 1.69).
  챕터를 옮길 때 그 챕터의 `tasks.py init` 을 다시 돌린다. 노트북에 이 안내를 남긴다.

## 2. 챕터↔소스 매핑과 배치

```bash
python3 agent-support/scripts/study-verify.py "source/<교재>" --no-urls   # 매핑 검증 포함
```

`study.toml` 의 `[mapping.chapters]` 가 정본이다. 매핑을 새로 정할 때는 다음을 지킨다.

### 매핑을 추측하지 마라

업스트림이 MEAP 번호를 쓰면 디렉터리 이름과 책 챕터가 어긋난다. **디렉터리 개수와 챕터
수를 맞춰 짝지으면 틀린다.** 실제로 그렇게 해서 `ch05`(miRNA)를 책 4장에, `ch06`(BBC+spaCy)을
책 5장에 잘못 넣었다.

챕터 원문 md 의 **고유 키워드 빈도** 로 확인한다. 예시:

| 근거 | 결론 |
| --- | --- |
| `Neosemantics` 7회, 3장에만 | `ch03` → 책 3장 |
| `PPI network` 16회 · `Het.io` 3회, 4장에만 | `ch04` → 책 4장 |
| `openai` 14회 · `GPT output` 3회, 5장에만 | `ch07` → 책 5장 |
| `fraud` 114회, 10장에만 | `ch12` → 책 10장 |
| `miRNA` 가 책 4장에 **1회뿐** | `ch05` → 대응 챕터 없음 |

최종 출간본에 대응 챕터가 없는 디렉터리는 `[mapping].meap_only` 에 넣고 `meap-only/` 로 분리한다.

### 리스팅 번호는 챕터마다 다르게 어긋난다

`[mapping.listings.chXX]` 가 **정본** 이고 `[mapping.listing_offsets]` 는 편의용 기본값이다.
단일 오프셋으로 표현되지 않는 경우가 있다.

```toml
[mapping.listings.ch04]
"4.20" = { source = "explainer" }   # 코드 파일 없는 프롬프트 리스팅
"4.21" = { repo = "4.20" }          # 이후가 한 칸씩 밀린다
"4.22" = { repo = "4.21" }
"4.23" = { source = "explainer" }
```

최종 교재가 코드 없는 리스팅(LLM 프롬프트 등)을 중간에 끼워넣으면 그 뒤 번호가 전부
밀린다. 오프셋 도출은 장 전체 평균을 보므로 **뒷부분의 낮은 유사도가 앞부분 다수결에
묻힌다** — ch04 는 그래서 오프셋 0 / 신뢰도 0.52 로 통과했지만 실제로는 틀렸다.

3장은 −3, 4장은 0 이다. **한 챕터의 값을 다른 챕터에 적용하지 마라.**

### 업스트림 리스팅 파일이 망가진 경우

`study-verify.py` 가 0바이트·내용 중복 파일을 보고한다. 그런 리스팅은
`{ source = "explainer" }` 로 선언하고 노트북이 해설판 본문을 싣는다. 실제 사례:

- `ch03/listings/3.28 ...` 0바이트 → 책 3.25
- `ch03/listings/3.24 ...` 가 `3.25` 와 내용 완전 중복 → 책 3.21

해설판 본문에도 **실행하면 안 되는 코드** 가 있다. 책 3.21 은 바인딩 없는 변수에 `MERGE` 를
써서 노드를 새로 만든다. `MATCH` 로 바로잡고 실행 전후 노드 수를 대조해 오염이 없음을 보인다.

## 3. 노트북 골격 생성

```bash
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" --list      # 대상 확인
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" ch07 --dry-run
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" ch07
```

**노트북은 `src` 에 코드가 있는 챕터에만 만든다.** 코드가 없으면 실행할 것이 없다.

생성 직후 `metadata.studykit.status = "draft"` 이므로 게이트가 lint 만 본다. 서술을 채우고
`listing_coverage` 를 선언한 뒤 `complete` 로 바꾼다.

## 4. 서술 채우기

`TODO(agent)` 를 해설판 근거로 채운다. 절 구성·리스팅 번호·실습 자리는 생성기가 이미 넣었다.

### 셀 순서는 고정이다

1. **헤더 표** — 책·해설판·이 폴더·원서 저장소·리스팅 번호 대응
2. **실행 환경** — 필요한 외부 서비스, 에디션 제약, 용량·시간
3. **환경 점검 코드** — 이 셀이 통과하면 이후 전부 실행 가능해야 한다
4. **리스팅 헬퍼 코드** — 책 번호로 원문을 보고 실행. 손상 리스팅 대체본 포함
5. **본문 절** — 해설판 절 구성을 따라 MD 개념 → CODE 리스팅 실행
6. **실습** — 해설판 Exercise 를 구현. 책의 실습 번호·이름 유지
7. **요약** — 실측 대비표, 핵심 용어, 참고 링크, 다음 장 연결

### 반드시 지킬 것

- **그림은 attachment 로 내장한다.** 두 방식이 모두 실패한 이력이 있다. HTML
  `<img src="원격URL">` 은 응답 200 이어도 태그를 걸러내는 뷰어에서 한 장도 안 보였고,
  상대경로는 `nbconvert` 가 11장 전부 해석하지 못했다. **링크 응답 코드로 렌더링을
  판단하지 마라.**

  ```bash
  python3 agent-support/scripts/study-new-notebook.py "source/<교재>" ch07 --embed
  ```

  attachment 키는 해시 파일명이 아니라 그림 번호(`fig4-1.jpg`)로 둔다. 한 그림이 여러
  이미지로 나뉜 경우 `fig4-7a` `fig4-7b` 처럼 붙인다.

- **저장소 안의 대상은 상대경로로 링크한다.** GitHub URL 은 푸시 전에 열리지 않아
  "노트북을 만들려면 먼저 푸시해야 한다"는 순서 역전을 만든다. 외부 사이트만 URL 로 둔다.
  봇 차단으로 200 이 아닌 URL 은 `study.toml` 의 `[notebook].url_allow_non_200` 에 넣는다.

- **연속된 `**라벨**: 값` 줄은 표로 만든다.** 렌더러가 한 단락으로 합친다.

- **오래 걸리는 셀은 실측 시간을 적고 건너뛰기 가드를 둔다.** 배치 없이 25만 건을
  갱신하는 리스팅이 8분 걸린 사례가 있다.

- **데이터 드리프트를 명시한다.** 책 집필 시점과 현재 데이터가 다르다. 요약 셀에 실측값과
  책 값을 나란히 적고 왜 다른지 밝힌다. 시드 백업은 집필 시점 스냅샷이라 일치하고,
  원본에서 새로 받는 데이터는 다르다.

- **자체 완결성.** 노트북만 읽어도 챕터를 이해할 수 있어야 한다. 코드 셀 앞에 개념 설명을,
  docstring 에는 "왜 이 코드가 이렇게 생겼는지"를 적는다.

## 5. 외부 서비스

### Neo4j

챕터마다 요구가 다르다. **에디션과 플러그인이 배타적일 수 있다.**

| 챕터 유형 | 필요 | 비고 |
| --- | --- | --- |
| 온톨로지 적재 | Community + `n10s` | `CREATE DATABASE` 불가 → 기본 DB 사용 |
| 멀티 DB·백업 시드 | **Enterprise** | `seedUri`·`IS NODE KEY` 가 Enterprise 전용 |
| 그래프 알고리즘 | `graph-data-science` | |

**`n10s` 는 Enterprise 와 Jackson 버전 충돌로 죽는다** (`NoSuchMethodError:
JsonProperty.isRequired()`). 플러그인을 하나씩 시험해 특정했다. 그래서 n10s 가 필요한
챕터와 Enterprise 가 필요한 챕터는 컨테이너를 분리해 전환한다.

Enterprise 는 **평가 목적이 무료** 다. `NEO4J_ACCEPT_LICENSE_AGREEMENT=eval` 로 평가 계약에
동의한다(상용 계약이 있으면 `=yes`). 조건은 <https://neo4j.com/terms/enterprise_us/> 에
있고 비상용·비프로덕션으로 제한되므로 **사용자가 직접 읽고 동의** 해야 한다. 에이전트가
대신 수락하지 않는다.

### OpenAI 호환 엔드포인트

`study.toml` 의 `[llm].env_file` 이 **저장소 밖** 설정 파일을 참조한다. 비밀값을 저장소에
복사하지 않는다.

```python
from studykit import config, llm

STUDY = config.load()
llm.configure(STUDY, model="gpt-5.4-mini")   # 값은 출력하지 않는다
print(llm.describe())                        # 키는 마스킹
```

교재 코드가 환경변수 이름을 통일해 쓰지 않는다(`OPENAI_KEY` vs `OPENAI_API_KEY`).
`configure()` 가 양쪽을 모두 채운다. OpenAI SDK 는 `OPENAI_BASE_URL` 을 자동으로 읽으므로
`base_url` 을 넘기지 않는 리스팅도 프록시로 간다.

**리스팅이 모델명을 하드코딩한다.** 프록시가 그 모델을 제공하지 않으면 실패한다
(`unknown provider for model gpt-4o-mini`). **원본 파일을 수정하지 말고** 임포트 후 상수만
교체한다.

```python
import listing_3
llm.override_module_model(listing_3)   # OPENAI_MODEL 상수를 현재 모델로
```

### API 키가 필요한 데이터

`data-manifest.toml` 이 정본이다. UMLS 처럼 개인 키가 필요한 데이터는 발급 방법을 함께
안내한다. 키가 없으면 그 리스팅은 `documented-only` 로 분류하고 무엇이 필요한지 노트북에 적는다.

## 6. 완성 처리

```python
# 노트북 metadata
{
  "studykit": {
    "status": "complete",
    "repo_dir": "ch07",
    "listing_coverage": {"5.1": "executed", "5.2": "documented-only", ...}
  }
}
```

**책 리스팅 전부** 를 네 가지 중 하나로 분류한다. 하나라도 빠지면 게이트가 실패한다.

| 분류 | 뜻 |
| --- | --- |
| `executed` | 노트북이 실제로 실행한다 |
| `substituted` | 업스트림 파일이 망가지거나 없어 해설판 본문·직접 구현으로 대체 |
| `documented-only` | 원문·설명만 제공 (개념 예시, 실행 불가, 실행 부적절) |
| `optional` | 선택 실행 (용량·비용·키 때문). 플래그로 제어하고 실측 결과를 보존 |

실행이 **부적절한** 경우도 `documented-only` 다. 임상 판단·개인정보가 걸린 LLM 프롬프트는
원문만 싣고 왜 실행하지 않는지 밝힌다.

## 7. 검증 (완성 보고 전 필수)

```bash
python3 agent-support/scripts/study-verify.py "source/<교재>"              # 전체
python3 agent-support/scripts/study-verify.py "source/<교재>" --no-urls    # URL 확인 생략
python3 agent-support/scripts/study-verify.py "source/<교재>" --lint       # 정적 검사만
python3 agent-support/scripts/study-verify.py "source/<교재>" --execute    # 재실행 포함
python3 -m unittest discover -s agent-support/tests
```

`--lint` 와 기본(완성) 게이트가 다르다. `status = "draft"` 면 lint 만 본다. 이 구분이 없으면
60% 노트북도 "모든 검증 통과" 가 된다.

완성 게이트가 보는 것:

```text
nbformat 스키마 · 실행 오류 0 · TODO(agent) 0 · 리스팅 coverage 100%
그림 참조 수 = attachment 수 · 상대경로 링크 대상 존재 · 외부 URL 응답
Makefile 타깃·액션 순서 = tasks.py 등가 · 챕터 매핑·오프셋 도출값 일치
```

사이트 파일을 바꿨으면 추가로 실행한다.

```bash
python3 agent-support/scripts/build-index.py --check
python3 agent-support/scripts/validate-site.py --check-materials
```

## 8. 커밋

`AGENTS.md` 규칙 9 를 따른다 — 커밋·푸시·PR 은 사용자가 명시적으로 요청한 범위에서만 한다.
`.venv`·`code/`·`config.ini`·`util/`·`data/`·`dataset/` 은 부트스트랩이 재생성하므로 추적하지 않는다.

## 금지

- 챕터↔소스 매핑을 개수나 제목 인상으로 추측하는 것
- 한 챕터의 리스팅 오프셋을 다른 챕터에 적용하는 것
- 업스트림 원본 파일을 수정하는 것 — `tasks.py` 로 대체하고 모델 상수는 런타임에 교체한다
- 책이 가르치는 Cypher·프롬프트 리스팅을 파이썬으로 다시 쓰는 것 — 러너로 실행한다
- 공용 `studykit` 을 교재 폴더로 복사하는 것
- 링크 응답 코드만 보고 렌더링이 된다고 판단하는 것
- Neo4j Enterprise 라이선스를 에이전트가 대신 수락하는 것
- 검증 게이트를 통과하지 않은 노트북을 완성으로 보고하는 것
