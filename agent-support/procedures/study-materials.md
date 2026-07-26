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

### 업스트림 lock 은 출처, 재현 기준은 저장소가 따로 보증한다

한 `.venv` 로 서로 충돌하는 업스트림 lock 을 동시에 만족시킬 수 없다. 실제로 ch10 은
`networkx==3.2.1` 을, ch06·ch15 는 `neo4j==4.4.11` 을, ch07 은 `neo4j==4.4.4` 를 pin 하는데
완성된 노트북들은 `networkx 3.4.2` · `neo4j 5.28.4` 에서 실행돼 출력이 저장됐다. 여기서
버전을 내리면 **이미 완성된 챕터의 저장 출력과 어긋난다.**

그래서 규칙을 이렇게 나눈다.

| 대상 | 역할 |
| --- | --- |
| 챕터의 `requirements.lock` | **출처 기록.** 업스트림이 무엇을 pin 했는지 보존한다. 고치지 않는다 |
| 노트북 실행 환경 셀 | **재현 기준.** 그 노트북을 실행해 출력을 얻은 실제 버전을 적는다 |

"현재 환경에서 돌았다" 는 검증 기록일 뿐 재현 가능한 pin 이 아니다. 그러니 노트북에
**업스트림이 pin 한 버전과 실제로 검증한 버전을 나란히** 적고, 다르면 왜 그 선택인지 밝힌다.
숫자를 맞추려고 다운그레이드하지는 마라 — 완성된 챕터를 깨뜨리는 대가가 더 크다.

다만 남은 챕터에서 이 방침이 깨질 가능성이 있는 곳을 미리 안다. ch07 은 `scispacy` 와
S3 모델이 현재 환경에 없고, ch13 은 구버전 LangChain 임포트를, ch15 는 구버전 LangGraph API
를 쓴다. 이들은 **현재 버전에서 동작하지 않을 수 있으므로** 착수 시점에 먼저 확인하고,
정말 다운그레이드가 필요하면 그 챕터만 별도 `.venv` 를 두는 쪽을 검토한다.

## 2. 챕터↔소스 매핑과 배치

```bash
python3 agent-support/scripts/study-verify.py "source/<교재>" --no-urls   # 매핑 검증 포함
```

`study.toml` 의 `[mapping.chapters]` 가 정본이다. 매핑을 새로 정할 때는 다음을 지킨다.

### 매핑을 추측하지 마라

업스트림이 MEAP 번호를 쓰면 그 디렉터리 이름과 책 챕터가 어긋난다. **디렉터리 개수와 챕터
수를 맞춰 짝지으면 틀린다.** 실제로 그렇게 해서 업스트림 `ch05`(miRNA)를 책 4장에,
업스트림 `ch06`(BBC+spaCy)을 책 5장에 잘못 넣었다.

확정한 대응은 `study.toml` 의 `[mapping.upstream_dirs]` 에 적는다. **이 저장소의 소스 폴더는
책 장 번호를 쓰고**, 그 표가 업스트림 이름과의 변환을 담당한다.

챕터 원문 md 의 **고유 키워드 빈도** 로 확인한다 (아래 표의 `chNN` 은 업스트림 이름이다):

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

### 오프셋으로 표현할 수 없는 챕터가 대다수다

`[mapping.listing_offsets]` 로 되는 것은 확장자 없는 Cypher 리스팅을 쓰는 챕터뿐이다.
13개 챕터 중 3개(ch02·ch03·ch04)만 그렇다. 나머지는 둘 중 하나에 걸린다.

| 걸림돌 | 해당 |
| --- | --- |
| 리스팅이 `.py` 다 — `cypher.listings()` 가 의도적으로 제외한다 | ch05 · ch06 · ch10 · ch11 · ch13 |
| `listings/` 디렉터리가 아예 없다 — 코드가 `analysis/`·`model/` 등에 흩어져 있다 | ch07 · ch08 · ch09 · ch12 · ch15 |

게다가 **리스팅 파일 이름은 업스트림 MEAP 번호 그대로** 라 책 번호와 major 부터 다르고
(책 10장의 파일이 `12.x`), 파일 하나에 리스팅 여러 개가 들어 있기도 하다
(ch11 의 `GNN_all_in_one.py` 안 클래스 6개).
그래서 선언 형식이 세 가지다.

```toml
"2.1"  = { repo = "2.3" }                               # 번호 붙은 리스팅 파일
"11.2" = { source = "repo-file", path = "listings/GNN_all_in_one.py",
           symbol = "MultiHeadGraphAttention" }         # 저장소 안이지만 번호 파일이 아니다
"9.6"  = { source = "repo-file", path = "analysis/simple_classification_example.py",
           start = 96, end = 112 }                      # 함수 경계가 아닌 조각
"4.20" = { source = "explainer", reason = "book-only" }  # 정본이 해설판 본문
```

해석은 **`studykit.listing_source` 한 곳에서만** 한다. 노트북이 자체 표나 자체 `ast`
헬퍼를 두지 마라 — 실제로 책 9·11·12장이 `resolve_listing()` 을 호출조차 하지 않고
각자 구현해서 "study.toml 이 정본" 이라는 규칙이 이름만 남은 적이 있다.

| 키 | 뜻 |
| --- | --- |
| `symbol = "이름"` | 함수·클래스. `A.b` 로 클래스 안 메서드까지 |
| `symbol = "__main__"` | 파일 전체 (스크립트, 확장자 없는 Cypher·프롬프트 파일) |
| `symbol = "__entry__"` | `if __name__ == "__main__":` 블록만 |
| `start` / `end` | 1-based inclusive. 함수 경계와 무관한 조각에만 |
| `reason` | `book-only` / `upstream-missing` / `upstream-wrong` |

**줄 번호보다 `symbol` 을 앞세운다.** 업스트림 파일이 갱신되면 줄 번호는 조용히 어긋나지만
함수·클래스명은 그렇지 않다. `listing_source` 는 매번 `ast` 로 줄 번호를 다시 찾는다.
게이트가 선언 전수를 해결해 보므로 경로 오타·이름이 바뀐 심볼·챕터 폴더 이탈이 노트북
실행 전에 드러난다.

### 원서 md 는 리스팅 캡션을 자주 잃는다

`book_listings()` 는 원문의 `Listing N.M` 캡션을 정규식으로 뽑는다. 그런데 PDF 변환에서
캡션이 사라지거나 코드가 이미지로 들어가 **실재하는 리스팅이 집합에서 빠진다.** 확인된 것만
넷이다.

| 리스팅 | 왜 빠졌나 |
| --- | --- |
| 책 4.8 | 주석 콜아웃이 코드와 뒤엉키며 캡션 줄이 사라졌다 |
| 책 9.5 | 캡션 줄이 없다 (해설판이 "리스팅 9.5 평가 함수" 로 복원) |
| 책 10.5 | 코드가 **스캔 이미지** 로 실려 텍스트가 아니다 |
| 책 10.1 · 10.3 · 10.12 | 코드는 있으나 책이 번호를 주지 않았다 |

그러니 `book_listings()` 의 결과를 그 챕터 리스팅의 전부로 믿지 마라. 저장소 파일 목록과
대조해 빠진 것을 찾고, 실재하면 `[mapping.listings]` 에 선언한다. 선언하면 `listing_coverage`
에 함께 분류할 수 있다 — 게이트는 **책 집합 ∪ study.toml 선언** 만 허용하고 그 밖의 여분은
오타로 보아 실패시킨다.

### 발견한 교재 오류는 정본에 모은다

책 코드를 실제로 실행하면 오류가 계속 드러난다. 이 교재에서만 27건이다 — 인쇄된 코드가
그대로는 안 돌아가는 곳, 표와 리스팅이 다른 양을 세는 곳, Cypher 의 바인딩 없는 변수,
PDF 변환으로 유실된 캡션·수식.

노트북 안에만 두면 검색도 중복 제거도 상태 관리도 안 된다. 교재 루트의 **`errata.toml`** 이
정본이고 노트북은 그 항목을 설명·재현하는 곳이다.

```toml
[[erratum]]
chapter = 10
location = "Listing 10.15 (Cypher)"
kind = "code"          # code | value | semantics | extraction
status = "confirmed"   # confirmed | editorial | version-drift | upstream | open
severity = "high"      # 선택. 결과 해석이 뒤집히는 것에만
observation = "..."    # 무엇을 관찰했나 (실측값 포함)
correction = "..."     # 바로잡은 결과
notebook = "chapter_10_.../src/ch10/10_chapter_guide.ipynb"
```

`status` 를 구분하는 이유가 있다. **`version-drift` 는 오류가 아니다** — 집필 시점과 지금의
도구·모델이 달라진 것이므로 "책이 틀렸다" 고 적으면 그쪽이 틀린 서술이 된다. **`open` 은
불일치는 확정했으나 원인을 규명하지 못한 것** 이고, 이것을 `confirmed` 로 올리지 마라.
`[env]` 섹션에 검증 환경을 적어 두어, 도구 버전이 바뀌면 `kind = "value"` 항목부터 다시
확인하게 한다.

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
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" ch05 --dry-run
python3 agent-support/scripts/study-new-notebook.py "source/<교재>" ch05
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
  python3 agent-support/scripts/study-new-notebook.py "source/<교재>" ch05 --embed
  ```

  attachment 키는 해시 파일명이 아니라 그림 번호(`fig4-1.jpg`)로 둔다. 한 그림이 여러
  이미지로 나뉜 경우 `fig4-7a` `fig4-7b` 처럼 붙인다.

- **`--embed` 의 자동 대응을 믿지 마라. 다섯 챕터에서 다섯 번 틀렸다.**
  `figure_map_from_explainer()` 는 해설판에서 `그림 N.M` 으로 시작하는 줄을 캡션으로 보는데,
  해설판은 **서술 문장** 에서도 그 형태를 쓴다("그림 11.2는 유클리드 공간을 …"). 그러면 앞에
  쌓인 이미지 여러 장이 엉뚱한 번호로 통째로 묶인다. 실측 오배치:

  | 챕터 | 증상 |
  | --- | --- |
  | ch09 | 이미지 13장 중 6건만, 그것도 전부 `fig9-7a~f` 로 묶임 (해설판 캡션이 `**그림 9.1**` 볼드라 정규식 미스) |
  | ch11 | 20장 중 9건. 차원 증가 시퀀스의 첫 조각(점 "0")을 그림 11.2 에 오배치 |
  | ch12 | 30장 중 17건. 모델 순서가 해설판(GCN→GAT→SAGE)과 원문 등장 순서(SAGE→GCN→GAT)가 다름 |
  | ch10 | 리스팅 코드 스캔과 **저자 아바타 아이콘(1.3KB)** 까지 그림으로 잡음 |

  그러니 **원서 원문의 `Figure N.M` 캡션 바로 위 이미지** 를 근거로 삼고, 애매하면 이미지를
  직접 열어 내용으로 판정한다. 캡션이 이미지 **안에** 인쇄된 경우도 있다(ch10 의 그림 10.7).
  확정한 대응은 `{키: 파일명}` 표로 남기고 `notebook.embed_named_figures()` 에 넘긴다.
  PDF 변환으로 **수식이 이미지가 된 것** 은 그림으로 참조하지 말고 본문에 LaTeX 로 쓴다.

- **요약 표의 수치를 손으로 옮기지 마라.** 재실행하면 비결정적 값이 달라져 표와 셀 출력이
  어긋난다. ch11 에서 실제로 발생했다(표 `0.976` ↔ 출력 `0.967`, `gensim` 이 `workers=4` 로
  학습해 스레드 스케줄링에 좌우된다). 두 가지로 막는다.

  | 값의 성격 | 표에 적는 방식 |
  | --- | --- |
  | 결정적 (시드 고정·순수 계산) | 실측값 그대로. n 을 적지 않는다 |
  | 비결정적, 재관측이 싸다 | **`x\~y (n=k)`** — 실제로 k 회 돌려 얻은 범위 |
  | 비결정적, 두 관측이 같았다 | **`x\~x (n=2)`** — 단일값으로 쓰지 마라. 같은 것과 결정적인 것은 다르다 |
  | 비결정적, 재관측이 비싸다 (GPU 학습·유료 LLM) | **`(n=1)`** 을 붙이고 재현 계약 밖이라고 밝힌다. 범위를 지어내지 마라 |

  **범위를 추정하지 마라 — 실제로 관측한 것만 적는다.** k 를 늘릴 수 없으면 늘리지 않은
  채로 n 을 밝히는 것이 맞다. 그리고 요약 셀에 실측값을 **코드로 출력** 해 두면 마크다운
  표와 나란히 놓고 대조할 수 있다. 어느 항목이 비결정적인지도 그 출력에 표시한다.

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
    "repo_dir": "ch05",
    "listing_coverage": {"5.1": "executed", "5.2": "documented-only", ...}
  }
}
```

**책 리스팅 전부** 를 다섯 가지 중 하나로 분류한다. 하나라도 빠지면 게이트가 실패한다.

판정 기준은 하나다 — **깨끗한 환경에서 기본 설정으로 Run All 하면 무엇이 일어나는가.**
`show()` 로 원문을 출력하는 것은 실행이 아니다. 메서드 **본문** 이 리스팅인데 그 호출자가
서비스를 요구한다면 `optional` 이다.

한 축으로 표현되지 않는 조합이 있다. 실행은 되는데 업스트림 원문 그대로가 아닌 경우
(런타임에 의존물을 교체한 책 15장의 15.2·15.5·15.12)가 그렇다. 그럴 때만 두 축으로 적는다.

```python
"15.2": {"run": "executed", "fidelity": "substituted", "note": "왜 원문 그대로가 아닌가"}
```

`run` 은 **깨끗한 환경의 기본 Run All 이 이 코드를 실제로 돌리는가** 이고
(`executed`·`optional`·`documented-only`), `fidelity` 는 **그 리스팅 자신의 코드 경로가
업스트림 원문대로 도는가** 다 (`original`·`reproduced`·`substituted`).

경계가 헷갈리기 쉽다. 모듈이 임포트되게 하려고 자리표시자 환경변수를 채우거나 모델
객체를 바꾸는 것은 그 리스팅이 **받는 설정** 이 달라지는 것이지 코드가 달라지는 것이
아니다 — `original` 이다. 그 리스팅이 **직접 부르는** 의존물을 런타임에 갈아 끼웠을
때만 `substituted` 다 (책 15장의 15.2 가 `jinja2_formatter` 를 그렇게 한다).

한 축 문자열은 두 축의 **줄임말** 이고, 게이트의 `verify.coverage_axes()` 가 그 변환을
정의한다.

| 줄임말 | run | fidelity |
| --- | --- | --- |
| `executed` | executed | original |
| `optional` | optional | original |
| `documented-only` | documented-only | original |
| `reproduced` | executed | reproduced |
| `substituted` | executed | substituted |

이 표로 표현되지 않는 조합만 표로 바꾼다. 두 표기가 섞여 있어도 게이트는 같은 질문에
답할 수 있다.
내가 한 번 돌려 봤는지가 아니라, 저장소를 새로 클론한 스터디원이 겪을 일로 판단한다.

| 분류 | 뜻 |
| --- | --- |
| `executed` | 기본 설정 Run All 이 그 코드를 실행한다 |
| `reproduced` | 리스팅이 **코드가 아니라 출력 예시** 다. 노트북이 그 출력을 만드는 코드를 실행해 책 인쇄값과 대조한다 |
| `substituted` | 업스트림 파일이 망가지거나 없어 해설판 본문·직접 구현으로 대체 |
| `documented-only` | 원문·설명만 제공 (개념 예시, 실행 불가, 실행 부적절) |
| `optional` | 기본 Run All 이 건너뛴다 — 플래그·외부 의존 때문. 실측 결과를 노트북에 보존해야 한다 |

경계에서 흔히 틀리는 것 셋이다.

- **출력 예시를 `executed` 로 쓰지 마라.** 책 9.7·9.9, 12.3·12.16·12.19·12.20 처럼 결과 표·
  텍스트인 리스팅은 `reproduced` 다. 실행할 코드 자체가 없다.
- **가드가 걸린 것은 `optional` 이다.** 내 환경에 마침 의존물이 있어 실행됐더라도, 깨끗한
  환경에서 건너뛴다면 `optional` 이다 (ch10 의 10.15\~10.17 은 4장이 만든 `hetionet` DB 에
  의존한다).
- **`optional` 은 실측 보존을 요구한다.** 플래그만 두고 값을 남기지 않으면 스터디원이
  무엇을 기대할지 알 수 없다. ch02 는 프록시로 한 번 돌려 날짜·모델과 함께 결과를 실었다.

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

## 책 그림·표의 공개 범위

**업스트림 코드의 라이선스나 공개 승인은 책 본문·표·그림에 전이되지 않는다.** 저자 코드
저장소를 공개해도 되는지와 출판사의 책 그림을 공개해도 되는지는 서로 다른 문서가 정한다.
base64 로 노트북에 심는 것은 저장 형태를 바꿀 뿐 공개 복제라는 성격을 바꾸지 않는다.

**앞으로의 규칙 — 권리 확인 없는 책 그림·표 스캔을 public git 에 새로 추가하지 않는다.**
대신 직접 그린 변형 도식, 텍스트 해설, 출처를 밝힌 최소 인용을 쓴다.

**현재 상태 — 이 규칙을 지키지 못하는 자료가 이미 있다.** 『Knowledge Graphs and LLMs in
Action』 노트북 11개가 Manning 책 그림 131장을 attachment 로 담고 있고, 저장소는 public
이다. 이것은 "승인된 예외" 가 아니라 **사용자 결정을 기다리는 알려진 미해결 상태** 다.
처리 방침(삭제 · 자체 재도식화 · 비공개 전환)이 정해지기 전까지 그 자료를 공개 준비
완료로 판정하지 않는다. 이미 push 된 이력의 정리는 별도 결정이 필요하다.

이 절은 법률 자문이 아니라 공개 저장소의 보수적 운영 기준이다.

## 금지

- 챕터↔소스 매핑을 개수나 제목 인상으로 추측하는 것
- 한 챕터의 리스팅 오프셋을 다른 챕터에 적용하는 것
- 업스트림 원본 파일을 수정하는 것 — `tasks.py` 로 대체하고 모델 상수는 런타임에 교체한다
- 책이 가르치는 Cypher·프롬프트 리스팅을 파이썬으로 다시 쓰는 것 — 러너로 실행한다
- 공용 `studykit` 을 교재 폴더로 복사하는 것
- 링크 응답 코드만 보고 렌더링이 된다고 판단하는 것
- **노트북에 리스팅 대응표나 자체 `ast` 헬퍼를 두는 것** — 해석은 `studykit.listing_source`
  한 곳에서만 한다. 노트북마다 재구현하면 `study.toml` 이 정본이라는 규칙이 이름만 남는다
- **`--embed` 의 자동 그림 대응을 확인 없이 쓰는 것** — 다섯 챕터에서 다섯 번 틀렸다
- **요약 표에 비결정적 수치를 단일값으로 적는 것** — `x\~y (n=k)` 로 적는다.
  관측이 하나뿐이면 `(n=1)` 과 함께 재현 계약 밖이라고 밝힌다
- **관측하지 않은 범위를 적는 것** — n 을 줄일지언정 숫자를 지어내지 않는다
- Neo4j Enterprise 라이선스를 에이전트가 대신 수락하는 것
- 검증 게이트를 통과하지 않은 노트북을 완성으로 보고하는 것
