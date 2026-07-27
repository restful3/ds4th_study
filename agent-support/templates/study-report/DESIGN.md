# study-report-v1 디자인 규칙 · Chapter 1 완성본 기준

이 템플릿은 ConnectBrick ai-odyssey 출판 템플릿의 실제 theme_report.css와 report.js를 ds4th study용으로 직접 이식하고, Chapter 1 리포트에서 검증한 장문 구조·표·도형·이미지 뷰어를 공용화한 HTML 리포트다. 구조만 참고한 경량 재구현이 아니다. 워드마크·발표 내용·개인 브랜딩은 ds4th study에 맞게 교체했고 외부 폰트와 Chart.js 의존성은 제거했다.

- 완성 기준 예시: `docs/studies/knowledge-graphs-and-llms-in-action/presentations/2026-07-25-ch01/report.html`
- 공통 설계 청사진: `agent-support/templates/STUDY_SESSION_BLUEPRINT.md`

## 출처와 변경 경계

- 기준 템플릿: ConnectBrick ai-odyssey의 theme_report.css, report.js, report HTML 구조.
- 그대로 유지: A4 인쇄 규칙, continuous-mode, 표지, 섹션 계층, 표·그림 캡션, callout, 목차 드로어, 설정 패널, 다크 테마.
- ds4th 변경: 워드마크, 링크 구조, 시스템 폰트 fallback, skip link, 회차 간 이동 링크, 전체화면 이미지 뷰어.
- 금지: 템플릿을 별도 카드형 CSS로 다시 작성하거나, 슬라이드 CSS를 리포트에 재사용하거나, 리포트를 슬라이드 문장 확장본으로 만드는 것.

## 필수 산출물

한 회차는 기본적으로 다음을 함께 제공한다.

1. report.html — 발표에서 생략한 정의·근거·논리·사례·판단 기준을 보존하는 상세 자료.
2. index.html — 발표용 슬라이드.
3. assets/figs/*.svg — 책의 개념을 발표자가 재구성한 도해.
4. presentation.toml — 회차 메타데이터와 report 경로.

상세 리포트가 없는 슬라이드 단독 산출물은 완료로 간주하지 않는다.

## 표준 정보 구조

1. .report-cover — 워드마크, 챕터, 제목, 부제, 발표자·범위·일시.
2. Section 00 · Summary — 한 문장 결론, 핵심 질문, 장 전체 논리표.
3. Section 01 · Context & Concept — 배경, 정의, 구성요소와 관계.
4. Section 02 · Mechanism — 이전/이후, 입력–변환–검증–결과.
5. Section 03 · Operating Model — 역할 분담, 반복 루프, 사람의 승인 지점.
6. Section 04 · Evidence & Limits — 근거, 조건, 실패 모드와 반증 신호.
7. Section 05 · Cases — 문제–역할–결과–실패 경계로 읽는 사례.
8. Section 06 · Decision — 적합/비적합 신호, 작은 실험과 중단 기준.
9. .report-appendix — 용어, 참고문헌, 그림 출처.

교재의 논리가 더 짧으면 인접 섹션을 합칠 수 있다. 다만 `문제 → 개념 → 메커니즘 → 운영 → 근거/한계 → 사례 → 판단` 중 빠진 축이 없는지 확인한다. 자동 목차는 각 `.report-section`의 직접 자식 `h1`과 본문 `h3`에서 생성되므로 제목 계층을 임의로 건너뛰지 않는다.

본문 섹션마다 가능하면 다음 순서를 사용한다.

- .report-section__kicker와 h1
- .section-summary로 2–4개 핵심 요약
- 2개 이상의 설명 단락
- 표 또는 도형
- 필요할 때 .callout--insight 또는 .callout--hero

## 시각 체계

- 배경 #FAFAF9, 본문 #0F0F10, 액센트 Deep Navy #0F2C59 한 색을 중심으로 쓴다.
- 본문은 continuous mode에서 약 760px 너비로 유지하고 한글 행간을 넉넉히 둔다.
- 섹션 제목은 강한 검정 타이포와 진청 구분선을 쓴다.
- 화면에서는 양쪽 플로팅 컨트롤, 자동 숨김, 드로어 목차를 사용한다.
- 인쇄에서는 컨트롤을 숨기고 A4 페이지 분리, 표·그림의 break-inside: avoid를 유지한다.
- 외부 CDN, 외부 폰트, 프레임워크에 의존하지 않는다.

## 표와 그림

### 표

모든 표는 table.cmp-table과 아래 캡션 구조를 사용한다.

~~~html
<caption class="cmp-table__caption asset-caption asset-caption--table">
  <div class="asset-caption__inner">
    <span class="asset-caption__chip">표 1</span>
    <span class="asset-caption__title">제목</span>
  </div>
</caption>
~~~

표는 정의·비교·역할 분담·판단 질문처럼 반복되는 필드를 정리할 때만 쓴다. 한 칸 안에 긴 에세이를 넣지 않는다.
발표자료가 정확히 추적할 수 있도록 핵심 표에는 `id="table-..."` 형식의 회차 내 고유 ID를 둔다.

### 그림

모든 도형은 figure.report-figure와 그림 번호·제목·근거 메모를 포함한다.

~~~html
<figure class="report-figure" id="fig-example" data-deck-use="required">
  <figcaption class="asset-caption asset-caption--figure">
    <span class="asset-caption__chip">그림 1</span>
    <span class="asset-caption__title">제목</span>
  </figcaption>
  <img src="assets/figs/example.svg" alt="도형이 전달하는 관계를 설명">
  <small class="asset-note">재구성 범위와 출처.</small>
</figure>
~~~

SVG는 차트·관계도·플로우·타임라인에만 사용한다. 사람·장식 아이콘·일러스트를 그리지 않는다. 도형 안에는 설명 문단을 넣지 말고 구조를 읽는 데 필요한 짧은 라벨만 둔다. 책의 그림을 그대로 베끼지 말고 논리를 새로 배치한 발표용 도해를 만든다. 외부 SVG의 기본 한글 font stack은 `report.css`의 `--font-sans-ko`와 같은 순서로 직접 선언한다.

같은 의미 수준의 박스는 폭·행·열·간격을 공통 그리드에 맞춘다. 화살표는 눈에 보이는 shaft를 확보하고 의도한 도형 면의 중앙 경계에 끝이 닿게 한다. 텍스트나 다른 도형을 관통하지 않게 하며, 한 그림에서 결함을 찾으면 모든 형제 그림에서 같은 유형을 다시 검사한다. 좌표만 보지 말고 실제 최종 표시 크기의 렌더를 확인한다.

모든 `.report-section`, 핵심 표와 그림에는 안정적인 `id`를 둔다. 그중 발표자료가 반드시 재사용하거나 충실히 재배치해야 할 그림은 `data-deck-use="required"`로 표시한다. 이 표식은 장식 우선순위가 아니라 주장 전달에 필요한 시각 근거라는 계약이다.

화면에서 `.report-figure`의 이미지와 SVG는 자동으로 확대 가능한 버튼이 된다. 클릭 또는 Enter로 전체 화면 뷰어를 열고, `+`/`−`·휠·더블클릭·핀치로 1–6배 확대하며 드래그·방향키로 이동하고 Esc로 닫는다. 이 동작을 유지하기 위해 `assets/report.js`의 `setupImageLightbox()`와 관련 CSS를 제거하거나 이미지 위에 링크를 씌우지 않는다. 캡션 제목은 뷰어의 접근 가능한 이름으로 쓰이므로 “그림 1”만 적지 말고 관계가 드러나는 제목을 쓴다.

## 품질 기준

- 본문은 책의 목차를 빠짐없이 따라가되 원문을 길게 복제하지 않는다.
- 각 핵심 주장에는 정의, 작동 방식, 한계 또는 판단 기준 중 최소 두 가지가 따라야 한다.
- 한 챕터 리포트에는 원칙적으로 6–10개의 의미 있는 도형/표를 제공하고 각 주요 논리 전환에 최소 하나의 시각적 근거를 둔다. 자료가 시각화에 적합하지 않으면 억지로 채우지 말고 이유를 기록한다.
- 그림과 표 번호는 각 종류별로 1부터 중복·누락 없이 증가시킨다.
- LLM 생성 결과, 추론 결과, 검증된 사실을 문장과 도형에서 구분한다.
- 외부 수치·이미지·주장에는 근접한 출처를 표시한다.
- 데스크톱, 390px 모바일, A4/PDF에서 겹침·잘림·과도한 빈 페이지가 없어야 한다.
- 목차, 테마, Print/PDF, Report/Slides/Index 링크와 이미지 확대·줌·닫기가 실제로 동작해야 한다.
- 리포트 게이트를 통과하기 전에는 회차별 발표자료 본문 작성을 시작하지 않는다. 발표자료에서 새 근거가 필요하면 리포트를 먼저 보완한다.

## 아카이브 규칙

책이 끝나 archive/로 이동해도 회차의 상대 경로만으로 리포트와 슬라이드가 함께 보존되어야 한다. 회차 파일에서 source/의 이미지나 루트의 임시 파일을 직접 참조하지 않는다. 필요한 도형과 정적 자산은 반드시 회차 assets/ 아래에 복사하거나 새로 작성한다.
