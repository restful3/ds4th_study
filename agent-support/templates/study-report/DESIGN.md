# study-report-v1 디자인 규칙 · ConnectBrick parity

이 템플릿은 ConnectBrick ai-odyssey 출판 템플릿의 실제 theme_report.css와 report.js를 ds4th study용으로 직접 이식한 장문 HTML 리포트다. 구조만 참고한 경량 재구현이 아니다. 워드마크·발표 내용·개인 브랜딩은 ds4th study에 맞게 교체했고 외부 폰트와 Chart.js 의존성은 제거했다.

## 출처와 변경 경계

- 기준 템플릿: ConnectBrick ai-odyssey의 theme_report.css, report.js, report HTML 구조.
- 그대로 유지: A4 인쇄 규칙, continuous-mode, 표지, 섹션 계층, 표·그림 캡션, callout, 목차 드로어, 설정 패널, 다크 테마.
- ds4th 변경: 워드마크, 링크 구조, 시스템 폰트 fallback, skip link, 회차 간 이동 링크.
- 금지: 템플릿을 별도 카드형 CSS로 다시 작성하거나, 슬라이드 CSS를 리포트에 재사용하거나, 리포트를 슬라이드 문장 확장본으로 만드는 것.

## 필수 산출물

한 회차는 기본적으로 다음을 함께 제공한다.

1. report.html — 발표에서 생략한 정의·근거·논리·사례·판단 기준을 보존하는 상세 자료.
2. index.html — 발표용 슬라이드.
3. assets/figs/*.svg — 책의 개념을 발표자가 재구성한 도해.
4. presentation.toml — 회차 메타데이터와 report 경로.

상세 리포트가 없는 슬라이드 단독 산출물은 완료로 간주하지 않는다.

## 정보 구조

1. .report-cover — 워드마크, 챕터, 제목, 부제, 발표자·범위·일시.
2. Section 00 — 한 문장 결론, 핵심 질문, 장 전체 논리표.
3. 본문 .report-section — 교재의 논리 순서에 따른 상세 설명.
4. 사례·도입 판단 — 기술 설명을 실제 문제와 의사결정으로 번역.
5. .report-appendix — 용어, 참고문헌, 그림 출처.

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

### 그림

모든 도형은 figure.report-figure와 그림 번호·제목·근거 메모를 포함한다.

~~~html
<figure class="report-figure">
  <figcaption class="asset-caption asset-caption--figure">
    <span class="asset-caption__chip">그림 1</span>
    <span class="asset-caption__title">제목</span>
  </figcaption>
  <img src="assets/figs/example.svg" alt="도형이 전달하는 관계를 설명">
  <small class="asset-note">재구성 범위와 출처.</small>
</figure>
~~~

SVG는 차트·관계도·플로우·타임라인에만 사용한다. 사람·장식 아이콘·일러스트를 그리지 않는다. 도형 안에는 설명 문단을 넣지 말고 구조를 읽는 데 필요한 짧은 라벨만 둔다. 책의 그림을 그대로 베끼지 말고 논리를 새로 배치한 발표용 도해를 만든다.

## 품질 기준

- 본문은 책의 목차를 빠짐없이 따라가되 원문을 길게 복제하지 않는다.
- 각 핵심 주장에는 정의, 작동 방식, 한계 또는 판단 기준 중 최소 두 가지가 따라야 한다.
- 한 챕터 리포트에는 원칙적으로 5개 이상의 의미 있는 도형/표를 제공한다. 자료가 시각화에 적합하지 않으면 억지로 채우지 말고 이유를 기록한다.
- LLM 생성 결과, 추론 결과, 검증된 사실을 문장과 도형에서 구분한다.
- 외부 수치·이미지·주장에는 근접한 출처를 표시한다.
- 데스크톱, 390px 모바일, A4/PDF에서 겹침·잘림·과도한 빈 페이지가 없어야 한다.
- 목차, 테마, Print/PDF, Report/Slides/Index 링크가 실제로 동작해야 한다.

## 아카이브 규칙

책이 끝나 archive/로 이동해도 회차의 상대 경로만으로 리포트와 슬라이드가 함께 보존되어야 한다. 회차 파일에서 source/의 이미지나 루트의 임시 파일을 직접 참조하지 않는다. 필요한 도형과 정적 자산은 반드시 회차 assets/ 아래에 복사하거나 새로 작성한다.
