---
lang: ko
format:
  html:
    toc: true
    toc-location: left
    toc-depth: 3
    theme: cosmo
    embed-resources: true
    code-fold: true
    code-tools: true
    smooth-scroll: true
    css: |
      body {
        margin-top: 0 !important;
        padding-top: 0 !important;
      }
      #quarto-header {
        display: none !important;
      }
      .quarto-title-block {
        display: none !important;
      }
      /* Center content with equal padding */
      body, #quarto-content, .content, #quarto-document-content, main, .main {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-left: 1em !important;
        padding-right: 1em !important;
        box-sizing: border-box !important;
      }
      .container, .container-fluid, article {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 auto !important;
        padding-left: 1em !important;
        padding-right: 1em !important;
        box-sizing: border-box !important;
      }
---

# LLM은 이미 데이터베이스 인터페이스로 사용될 수 있는가? 대규모 데이터베이스 기반 Text-to-SQL을 위한 대형 벤치마크

Jinyang Li<sup>1, 4, ‡</sup> Binyuan Hui<sup>2, 4, †</sup>, Ge Qu<sup>1, 4, †</sup>, Jiaxi Yang<sup>2</sup>, Binhua Li<sup>2</sup>, Bowen Li<sup>6</sup>, Bailin Wang<sup>5</sup>, Bowen Qin<sup>2</sup>, Ruiying Geng<sup>2</sup>, Nan Huo<sup>1</sup>, Xuanhe Zhou<sup>3</sup>, Chenhao Ma<sup>6</sup>, Guoliang Li<sup>3</sup>, Kevin C.C. Chang<sup>7</sup>, Fei Huang<sup>2</sup>, Reynold Cheng<sup>1</sup>, Yongbin Li<sup>2</sup>

<sup>1</sup> The University of Hong Kong <sup>2</sup> DAMO Academy, Alibaba Group

<sup>3</sup> Tsinghua University <sup>4</sup> Shanghai AI Laboratory <sup>5</sup> MIT CSAIL

<sup>6</sup> The Chinese University of Hong Kong, Shenzhen

<sup>7</sup> University of Illinois at Urbana-Champaign

{j10725, quge}@connect.hku.hk, ckcheng@cs.hku.hk

binyuan.hby@alibaba-inc.com

#### **초록 (Abstract)**

자연어 질문을 실행 가능한 SQL로 변환하는 것을 목표로 하는 Text-to-SQL 파싱(parsing)은 최근 몇 년간 점점 더 많은 관심을 받고 있습니다. 특히 GPT-4와 Claude-2는 이 작업에서 인상적인 결과를 보여주었습니다. 그러나 Spider와 WikiSQL과 같은 대부분의 널리 사용되는 벤치마크는 데이터베이스 스키마에 초점을 맞추고 있으며 데이터베이스 값의 행 수가 적어 학술 연구와 실제 응용 프로그램 사이에 격차를 남기고 있습니다. 이러한 격차를 완화하기 위해, 우리는 12,751개의 text-to-SQL 쌍과 총 33.4 GB 크기의 95개 데이터베이스를 포함하며 37개의 전문 도메인에 걸쳐 있는 대규모 데이터베이스 기반 text-to-SQL 작업을 위한 대형 벤치마크인 BIRD를 제시합니다. 데이터베이스 값에 대한 우리의 강조는 더럽고 노이즈가 많은 데이터베이스 값, 자연어 질문과 데이터베이스 값 간의 외부 지식 그라운딩(external knowledge grounding), 그리고 특히 대규모 데이터베이스의 맥락에서 SQL 효율성이라는 새로운 도전 과제를 강조합니다. 이러한 문제를 해결하기 위해 text-to-SQL 모델은 의미론적 파싱(semantic parsing) 외에도 데이터베이스 값 이해 기능을 갖추어야 합니다. 실험 결과는 대규모 데이터베이스에 대한 정확한 text-to-SQL을 생성하는 데 있어 데이터베이스 값의 중요성을 보여줍니다. 더 나아가, 가장 효과적인 text-to-SQL 모델인 GPT-4조차도 실행 정확도(execution accuracy)에서 54.89%만 달성했으며, 이는 여전히 92.96%의 인간 결과와는 거리가 멀어 도전 과제가 여전히 남아 있음을 입증합니다. 또한 산업에 유익한 text-to-efficient-SQL 생성에 대한 통찰력을 제공하기 위해 효율성 분석을 제공합니다. 우리는 BIRD가 text-to-SQL 연구의 실제 응용을 발전시키는 데 기여할 것이라고 믿습니다. 리더보드와 소스 코드는 https://bird-bench.github.io/ 에서 이용 가능합니다.

#### 1 서론 (Introduction)

자연어 질문을 SQL 쿼리로 변환하는 데 초점을 맞춘 Text-to-SQL 파싱 [55, 50, 51, 3, 52, 37]은 학계와 산업계 모두로부터 상당한 연구 관심을 받아왔습니다. 이러한 관심은 비전문가 데이터 분석가가 자연어를 사용하여 유비쿼터스한 관계형 데이터베이스에서 원하는 정보를 자동으로 추출할 수 있도록 하는 잠재력에서 비롯됩니다. 대규모 언어 모델(LLM, Large Language Model)을 기반으로 한 신경망 모델의 최근 발전은 SPIDER [53] 및 WikiSQL [58]과 같은 기존 벤치마크에서 인상적인 성능을 이끌어냈습니다. 예를 들어, SPIDER 리더보드에서 최고 성능 모델의 실행 정확도는 지난 3년 동안 53.5% [59]에서 85.3% [35]로 증가했습니다. SPIDER의 최신 최첨단(SOTA, State-Of-The-Art) 파서 [35]는 대규모 언어 모델(LLM)의 강력한 이해 및 코딩 능력의 혜택을 받으며, 이러한 뛰어난 성능은 우리에게 다음과 같은 질문을 하게 합니다: *LLM은 이미 데이터베이스 인터페이스로 사용될 수 있는가?*

<sup>†</sup> 동등 기여(Equal contribution).

<sup>‡</sup> Alibaba DAMO Academy에서 인턴 기간 중 수행한 작업.

![](_page_1_Figure_0.jpeg)

<span id="page-1-0"></span>그림 1: 우리의 **BIRD** 벤치마크에서의 도전 과제 예시. 1) 데이터베이스는 노이즈가 많은 데이터 타입의 값을 포함합니다 [14, 23, 19, 31]. 왼쪽 예시에서 평균 급여는 특수 토큰인 "US\$"와 ", "를 삭제한 후 데이터 타입을 문자열(SQLite에서 TEXT)에서 실수(SQLite에서 REAL)로 처리하여 가져올 수 있습니다. 2) 외부 지식과 추론이 필요합니다. 가운데 예시에서 모델은 "OWNER" 계정만 대출 자격이 있다는 것을 처리해야 합니다. 3) 쿼리 실행 효율성을 고려해야 합니다. 오른쪽 예시에서 더 효율적인 SQL 쿼리를 채택하면 속도에서 상당한 이득을 얻을 수 있으며, 이는 산업에서 큰 가치가 있습니다.

답은 아니오입니다. 이전 벤치마크들은 데이터베이스 값의 행 수가 적은 데이터베이스 스키마에 초점을 맞추어 학술 연구와 실제 세계 사이의 격차를 남기고 있기 때문입니다. 그림 1에서 보듯이, 첫째, 현재의 최첨단 모델들은 여전히 대규모 데이터베이스 크기와 노이즈가 많은 값으로 특징지어지는 더 현실적인 상황으로 일반화하는 데 어려움을 겪고 있음을 발견했습니다. 둘째, 데이터베이스 크기의 증가는 종종 많은 컨텍스트 압축을 초래하여 전체 컨텍스트를 드러내기 어렵게 만듭니다 [1]. 따라서 포괄적인 이해를 위해서는 외부 지식 추론이 필요합니다. 셋째, 기존 벤치마크는 실제 응용 프로그램에서, 특히 대규모 데이터베이스의 경우 상당한 실용적 중요성을 갖는 SQL 실행 효율성을 고려하지 않습니다. 이러한 관찰에 동기를 받아, 우리는 실제 시나리오를 더 잘 나타내고 실험적 설정과 실용적 설정 사이의 격차를 좁히는 새로운 text-to-SQL 벤치마크를 개발하는 것을 목표로 합니다.

이 연구에서 우리는 실제 응용 프로그램을 위한 대규모 데이터베이스 기반 Text-to-SQL을 위한 대형 벤치마크인 BIRD를 제안합니다. BIRD는 37개의 전문 도메인에 걸쳐 총 33.4 GB 크기의 95개 대규모 데이터베이스에 대한 정보 쿼리의 복잡한 12,751개 예제를 포함합니다. 훈련 및 개발을 위해, 우리는 실제 분석 플랫폼(Kaggle, Relational.fit)에서 80개의 오픈 소스 관계형 데이터베이스를 수집하고 수정했습니다. 데이터 유출을 더욱 방지하기 위해, 우리는 숨겨진 테스트 세트를 위해 15개의 추가 관계형 데이터베이스를 큐레이션했습니다. 이러한 데이터베이스가 주어지면, 우리는 크라우드소싱에 의존하여 자연어 질문과 해당 SQL을 수집합니다. 또한, 생성된 SQL의 효율성을 평가하기 위해 새로운 평가 메트릭인 유효 효율성 점수(VES, Valid Efficiency Score)를 제안합니다. 우리가 아는 한, BIRD는 효율성을 통합하여 대규모이고 노이즈가 많은 데이터베이스 값의 맥락에서 더 효율적인 쿼리 방법을 촉진하는 최초의 text-to-SQL 벤치마크입니다.

우리는 두 가지 인기 있는 방법론을 사용하여 최첨단 text-to-SQL 파서의 성능을 평가합니다: T5 [38]를 사용한 파인튜닝(FT, Fine-Tuning)과 ChatGPT [33] (gpt-3.5-turbo), Claude-2 [2] (claude-2.0), GPT-4 [32] (gpt-4-32k)와 같은 고급 대규모 언어 모델을 사용한 인컨텍스트 학습(ICL, In-Context Learning)입니다. 우리의 실험 결과는 현재 모델들이 BIRD에서 잘 일반화하는 데 어려움을 겪고 있음을 보여줍니다. 구체적으로, GPT-4조차도 실행 정확도에서 54.89%만 달성했습니다. 이에 비해, 성능은 여전히 92.96%의 인간 성능에 훨씬 뒤떨어져 있으며, 도전 과제가 여전히 남아 있음을 입증합니다. 또한, 우리는 통찰력과 방향을 제공하기 위해 포괄적인 분석을 수행합니다. 우리는 이 벤치마크에서 제시된 더 현실적인 설정을 공동으로 다루기 위해 NLP와 데이터베이스 커뮤니티의 추가 연구를 권장합니다.

### 2 작업 정식화 및 주석 (Task Formulation & Annotations)

Text-to-SQL은 자연어 질문 $\mathcal{Q}$를 데이터베이스에서 관련 데이터를 검색할 수 있는 SQL 쿼리 $\mathbf{Y}$로 변환하는 프로세스를 말합니다. 데이터베이스는 $\mathcal{D} = \langle \mathcal{C}, \mathcal{T} \rangle$로 표현될 수 있으며,

![](_page_2_Figure_0.jpeg)

<span id="page-2-2"></span>그림 2: (a)의 BIRD 주석 워크플로우 개요. 이 그림은 4단계 절차를 묘사합니다. (1) 워크플로우는 전문가가 데이터베이스와 설명 파일을 조립하고 생성하는 것으로 시작합니다. (2) 그 다음 전문가는 크라우드소싱 인력을 교육하고 평가하여 평가를 통과한 사람만 유지합니다. (3) 질문 주석자는 데이터베이스와 해당 설명 파일을 사용하여 질문 말뭉치를 만듭니다. (4) SQL 주석자는 데이터베이스, 설명 및 질문을 갖춘 상태에서 SQL 파일을 생성합니다. (b)와 (c)는 또한 이중 맹검 주석 절차와 데이터베이스 설명의 예를 묘사합니다.

여기서 C와 T는 각각 열(column)과 테이블(table)입니다. BIRD와 같은 복잡한 데이터베이스 값을 다룰 때, 모델의 데이터베이스 값 이해를 향상시키기 위해 K로 표시되는 외부 지식 증거(external knowledge evidence)를 통합하는 것이 중요합니다. 최종적으로 text-to-SQL은 다음과 같이 공식화될 수 있습니다:

$$\mathbf{Y} = f(\mathcal{Q}, \mathcal{D}, \mathcal{K} \mid \boldsymbol{\theta}),\tag{1}$$

여기서 함수 $f(\cdot \mid \theta)$는 매개변수 $\theta$를 가진 모델 또는 신경망을 나타낼 수 있습니다.

### <span id="page-2-4"></span>3 데이터셋 구축 (Dataset Construction)

#### 3.1 주석 입문 (Annotation Entrance)

고품질 벤치마크를 제공하기 위해, 우리는 모든 지원자에게 철저한 시험을 실시하고 이러한 엄격한 테스트를 통과한 사람만 고용합니다. 자세한 정보는 부록 [B.2](#page-20-0)에서 확인할 수 있습니다.

#### <span id="page-2-3"></span>3.2 데이터베이스 소스 (Database Source)

개인 정보 보호로 인해 복잡한 스키마와 충분한 값을 가진 데이터베이스를 수집하는 것은 어렵습니다. 이전 연구들 [[45,](#page-13-2) [53]](#page-14-4)은 데이터베이스 스키마와 값 생성을 자체 설계하는 것을 선택했습니다. 그럼에도 불구하고, 이러한 방식으로는 값 분포와 스키마가 실제 시나리오와 다를 수 있습니다. 우리의 연구에서는 실제 속성을 풍부하게 하기 위해 세 가지 다른 출처에서 데이터베이스를 얻고 처리합니다. 우리 데이터베이스의 32%는 어렵고 노이즈가 많은 값과 스키마를 가진 데이터 과학 경진대회를 개최하는 것으로 유명한 플랫폼인 Kaggle[\*](#page-2-0)에서 가져왔습니다. 또 다른 48%는 다중 관계형 데이터를 사용한 기계 학습 연구를 위한 개방형 플랫폼인 CTU Prague Relational Learning Repository[†](#page-2-1)에서 가져왔습니다. 나머지 20%는 개방형 테이블을 획득하고, 스키마를 합성 및 표준화하며, 데이터베이스 제약을 생성하여 구축되었습니다. 이러한 모든 데이터베이스는 실제적이고 대규모 값 분포를 포함하며 적절한 라이선스로 쉽게 접근할 수 있습니다. 최종적으로 우리는 훈련, 개발 및 테스트를 위해 각각 69개, 11개, 15개의 데이터베이스로 구성된 총 95개의 데이터베이스를 제시합니다. 우리의 데이터베이스는 블록체인, 스포츠, 의료, 정치 등을 포함한 37개의 전문 도메인을 다룹니다. 우리는 이것이 대규모 데이터베이스 값을 가진 의미론적 파싱 작업에서 도메인 일반화를 탐구하기 위한 연구자들에게 중요한 자원이 될 것으로 기대합니다.

<span id="page-2-0"></span><sup>\*</sup><https://www.kaggle.com/>

<span id="page-2-1"></span><sup>†</sup><https://relational.fit.cvut.cz/>

#### <span id="page-3-1"></span>3.3 질문 주석 (Question Annotation)

**데이터베이스 설명 파일 (Database Description File).** 데이터베이스 설명 파일은 주석자가 데이터베이스 값을 이해하는 데 도움을 주어 통찰력 있는 질문을 할 수 있도록 설계된 중요한 자원입니다. 이는 데이터베이스에 관한 두 가지 주요 정보를 제공합니다. (1) 전체 스키마 이름: 데이터베이스 테이블과 열 이름은 종종 약어로 표현되어 이해하기 어렵습니다. (2) 값 설명: 이 측면은 질문의 구문이나 토큰이 데이터베이스의 값과 직접 일치하지 않을 때 특히 유용합니다.

<span id="page-3-0"></span>**외부 지식 증거 (External Knowledge Evidence).** 전문 데이터 분석에 대한 우리의 연구에서, 우리는 자연어 지시문을 대응하는 데이터베이스 값으로 매핑하기 위해 외부 지식 증거가 필요함을 발견했습니다. 따라서 우리는 이러한 증거를 네 가지 범주로 수집하고 분류합니다: (1) 수치 추론 지식 (Numeric Reasoning Knowledge): 이 범주는 특정 SQL 연산에 필요한 수학적 계산을 나타냅니다. 우리의 벤치마크에서, 우리는 [[7]](#page-10-3)처럼 4가지 복잡한 연산을 포함하는 8가지 기본 수학 연산을 제시합니다: MINUS(빼기), ADDITION(더하기), DIVISION(나누기), MULTIPLY(곱하기). BIRD는 또한 백분율, 공식 등과 같은 기본 연산에 대한 합성 연산을 포함합니다. (2) 도메인 지식 (Domain Knowledge): 이 범주는 SQL 연산을 생성하는 데 사용되는 도메인별 지식으로 구성됩니다 [[10,](#page-11-2) [57]](#page-14-7). 예를 들어, 은행 업무의 비즈니스 분석가는 효과적인 SQL 쿼리를 생성하기 위해 투자 수익률 및 순이익과 같은 재무 지표에 대한 지식이 필요합니다. (3) 동의어 지식 (Synonym Knowledge): 이 범주는 다르게 표현되더라도 동일하거나 유사한 의미를 가진 단어나 표현을 포함합니다 [[11]](#page-11-3). (4) 값 설명 (Value Illustration): 이 범주는 값 타입, 값 범주, 엔티티에 해당하는 열과 값의 매핑 조합을 포함하여 데이터베이스 값에 대한 자세한 설명을 나타냅니다. 예를 들어: "center"는 데이터베이스 professional_basketball에서 "pos = C"로 표현될 수 있습니다.

### 3.4 SQL 주석 (SQL Annotation)

<span id="page-3-2"></span>**이중 맹검 주석 (Double-Blind Annotation).** 그림 [2](#page-2-2) (b)에서 보듯이, 우리는 SQL 주석을 위해 이중 맹검 접근법 [[42]](#page-13-3)을 사용합니다. 이 접근법은 두 명의 독립적인 SQL 주석자가 논의 없이 동일한 질문에 대해 SQL을 생성하는 것을 포함합니다. 주석이 달린 SQL은 데이터베이스에서 실행되고, 동일한 결과를 산출하는 것들이 수집됩니다. 그렇지 않으면, SQL은 합의에 도달할 때까지 전문가와 함께 확인됩니다. 이중 맹검 절차는 SQL 주석 오류율을 극적으로 줄일 수 있습니다. 왜냐하면 데이터베이스에 큰 값이 있을 때 두 명의 숙련된 주석자가 동일한 잘못된 결과를 생성할 확률이 작기 때문입니다. 각 질문에 대해 전문가가 선택한 더 의미적으로 동등하고 효율적인 SQL이 BIRD에서 정답 SQL로 선택되며, 사용된 경우 각 SQL에 대한 외부 지식 증거 문장이 기록됩니다.

**검사 (Examination).** 전문가는 데이터의 최고 품질을 보장하기 위해 각 text-to-SQL 쌍을 평가합니다. 평가 프로세스는 두 가지 차원을 포함합니다: SQL 유효성과 텍스트-지식-SQL 정렬. 첫째, 각 SQL이 실행 가능하고 데이터베이스에서 유효한 결과를 반환할 수 있음을 확인하는 SQL 유효성이 확인됩니다. "유효한 결과"는 "NULL"이 아닌 결과 세트를 나타냅니다. 실행된 결과 세트가 "NULL"이면, 전문가는 관련 SQL이 유효한 결과 세트를 제공할 수 있을 때까지 질문의 조건을 약간 변경합니다. 둘째, 텍스트-지식-SQL 정렬이 포함되어 주어진 텍스트와 지식 증거로 각 SQL이 생성될 수 있도록 보장합니다. 증거가 SQL을 생성하기에 불충분하거나 오류를 포함하는 경우, 전문가가 이를 수정합니다.

# 4 데이터 통계 (Data Statistics)

**전체 통계 (Overall Statistics)** 표 [1](#page-4-0)은 BIRD와 다른 크로스 도메인 text-to-SQL 벤치마크 간의 개요 비교를 제시합니다. 통계가 보여주듯이, BIRD는 복잡한 SQL 함수, 지식 추론 및 효율성 평가를 다루는 대규모 크로스 도메인 벤치마크입니다.

**질문 통계 (Question Statistics)** 데이터베이스 값은 text-to-SQL에 더 많은 도전 과제를 가져옵니다. 이를 강조하기 위해, 우리는 질문을 두 가지 매크로 범주로 분류합니다: 기본 유형(Fundamental Type)과 추론 유형(Reasoning Type)이며, 각각은 세부적으로 4-5개의 마이크로 범주를 포함합니다. 기본 유형의 질문은 데이터베이스 값 이해 없이 답변할 수 있는 것들을 나타냅니다. 이는 매치 기반(Match-based) (83.9%), 순위(Ranking) (20.3%), 비교(Comparison) (16.7%), 계수(Counting) (30.4%), 집계(Aggregation) (15.7%)를 포함합니다.

<span id="page-4-0"></span>표 1: BIRD와 다른 크로스 도메인 text-to-SQL 벤치마크 간의 개요 비교. SQL에서 Function은 SQL 함수를 나타냅니다(부록 B.11). Knowledge는 이 데이터셋이 모델로부터 외부 지식 추론을 필요로 하는지 여부를 나타냅니다. Efficiency는 이 데이터셋이 실행 효율성을 고려하는지 여부를 나타냅니다.

| Dataset         | # Example | # DB   | # Table/DB | # Row/DB | Function | Knowledge | Efficiency |
|-----------------|-----------|--------|------------|----------|----------|-----------|------------|
| WikiSQL [58]    | 80,654    | 26,521 | 1          | 17       | ×        | ×         | X          |
| SPIDER [53]     | 10,181    | 200    | 5.1        | 2K       | ×        | ×         | ×          |
| KaggleDBQA [24] | 272       | 8      | 2.3        | 280K     | ×        | ✓         | ×          |
| BIRD            | 12,751    | 95     | 7.3        | 549K     | <b>✓</b> | ✓         | <b>✓</b>   |

![](_page_4_Figure_2.jpeg)

<span id="page-4-1"></span>그림 3: 이것은 BIRD의 포괄적인 데이터베이스 분포입니다. a)는 각 데이터베이스의 도메인 및 크기 분포를 보여줍니다. 그리고 b)는 데이터베이스의 데이터 타입 분포를 보여줍니다.

추론 유형(Reasoning Type)은 BIRD에만 있는 값에 대한 외부 지식 그라운딩을 요구하는 질문을 수반합니다. 구체적으로, 도메인 지식(Domain Knowledge) (23.6%), 수치 계산(Numeric Computing) (24.5%), 동의어(Synonym) (7.2%), 값 설명(Value Illustration) (70.1%)에 관한 질문이 BIRD에 포함되어 있습니다. 부록 B.3에는 풍부한 예제가 있습니다. 또한, 우리는 70.1%의 질문이 값 설명을 필요로 한다는 것을 관찰합니다. 이는 text-to-SQL 애플리케이션에서 더 많은 실제 질문이 데이터베이스 값에 대한 철저한 이해를 요구한다는 것을 나타내며, 이는 BIRD 벤치마크를 만드는 우리의 동기와 일치합니다.

**데이터베이스 통계 (Database Statistics)** BIRD에서, 우리는 데이터베이스 도메인, 데이터베이스 크기 및 값 타입의 분포를 조사합니다. 그림 3 (a)는 훈련 및 개발 세트 모두에 대해 선버스트 다이어그램에서 도메인과 그에 대응하는 데이터베이스의 세부 분포를 제시합니다. 각 반원의 면적은 이 데이터베이스의 text-to-SQL 쌍 수에 해당합니다. 그림 3 (a)는 또한 데이터베이스의 크기 분포를 보여줍니다. 더 어두운 색은 데이터베이스의 크기가 더 크다는 것을 의미하며, 그 반대도 마찬가지입니다. 예를 들어, 데이터베이스 Donor는 이 데이터셋에서 4.5 GB로 가장 큰 데이터베이스입니다. 더 나아가, 우리는 그림 3 (b)에서 BIRD의 데이터 중 상당 부분이 날짜 관련 값으로 구성되어 있음을 관찰합니다. 실제 응용 프로그램이 종종 시간에 민감한 데이터에 의존한다는 점을 고려할 때 [25], 이러한 질문의 보급은 실용적 목적을 강조합니다.

**SQL 통계 (SQL Statistics)** 우리는 BIRD에서 SQL의 복잡성과 다양성을 제공합니다. 그림 4에 나타난 바와 같이, 우리는 네 가지 차원에 걸쳐 SQL의 포괄적인 분포 분석을 제시합니다. No. Toks / SQL과 No. JOINs / SQL은 BIRD의 SQL의 복잡성을 보여줍니다. No. of Keywords와 No. n-grams / SQL (n=3)은 우리가 상황을 더 현실적으로 만들기 위해 질문과 SQL 주석 절차를 분리했기 때문에 SQL의 다양한 패턴에 대한 지원 역할을 합니다 [6].

#### **5 평가 메트릭 (Evaluation Metrics)**

실용적 데이터 분석의 맥락에서, text-to-SQL 모델은 예상 결과를 정확하고 효율적으로 제공하는 것이 우선시됩니다. 따라서 우리는 BIRD에서 두 가지 메트릭, 실행 정확도(EX, Execution Accuracy)와

![](_page_5_Figure_0.jpeg)

<span id="page-5-0"></span>그림 4: BIRD 데이터셋과 다른 크로스 도메인 text-to-SQL 벤치마크의 SQL 쿼리에 대한 비교 통계 분석.

유효 효율성 점수(VES, Valid Efficiency Score)를 제공하여 대규모 실제 데이터베이스 값에 직면한 text-to-SQL 파서를 평가합니다.

**실행 정확도 (EX, Execution Accuracy)** EX는 예측된 SQL과 정답 SQL 모두의 실행 결과가 동일한 평가 세트의 예제 비율을 전체 SQL 수에 대한 상대적 비율로 정의됩니다 [37]. $n$번째 정답 SQL $Y_n$에 의해 실행된 결과 세트를 $V_n$으로, 예측된 SQL $\hat{Y}_n$에 의해 실행된 결과 세트를 $\hat{V}_n$으로 고려하면, EX는 다음과 같이 계산될 수 있습니다:

$$EX = \frac{\sum_{n=1}^{N} \mathbb{1}(V_n, \hat{V}_n)}{N}, \tag{2}$$

여기서 $\mathbb{1}(\cdot)$는 다음과 같이 표현될 수 있는 지시 함수(indicator function)입니다:

$$\mathbb{1}(V,\hat{V}) = \begin{cases} 1, & V = \hat{V} \\ 0, & V \neq \hat{V} \end{cases}$$
(3)

**유효 효율성 점수 (VES, Valid Efficiency Score)** VES는 모델이 생성한 유효한 SQL의 효율성을 측정하도록 설계되었습니다. "유효한 SQL"이라는 용어는 결과 세트가 정답 SQL의 결과 세트와 일치하는 예측된 SQL 쿼리를 나타낸다는 점을 주목할 가치가 있습니다. 올바른 값을 가져오지 못하는 모든 SQL 쿼리는 효율성과 관계없이 사용자 요청을 충족할 수 없으면 완전히 쓸모가 없기 때문에 무효로 선언됩니다. 이 경우, VES 메트릭은 실행 결과의 효율성과 정확성을 모두 고려하여 모델 성능에 대한 포괄적인 평가를 제공합니다. 공식적으로, VES는 다음과 같이 표현될 수 있습니다:

VES =
$$\frac{\sum_{n=1}^{N} \mathbb{1}(V_n, \hat{V}_n) \cdot \mathbf{R}(Y_n, \hat{Y}_n)}{N}, \quad \mathbf{R}(Y_n, \hat{Y}_n) = \sqrt{\frac{\mathbf{E}(Y_n)}{\mathbf{E}(\hat{Y}_n)}}$$
(4)

여기서 $\mathbf{R}(\cdot)$는 정답 SQL과 비교한 예측 SQL의 상대적 실행 효율성을 나타내며, 기계 상태 관련 불확실성을 허용합니다. $\mathbf{E}(\cdot)$는 주어진 환경에서 각 SQL의 절대 실행 효율성을 측정하는 함수입니다. 더 나아가, 우리는 정답 SQL보다 비정상적으로 빠르거나 느린 무작위 인스턴스를 최소화하기 위해 제곱근 함수를 통합합니다. 여기서 효율성은 실행 시간, 처리량, 메모리 비용 또는 병합된 메트릭을 나타낼 수 있습니다. BIRD에서, 우리는 현재 주로 실행 시간을 고려합니다. 부록 B.8은 VES에 대한 자세한 설명을 제공합니다.

### 6 실험 (Experiments)

#### 6.1 기준 모델 (Baseline Models)

우리는 BIRD에서 두 가지 유형의 기준 모델의 성능을 제시합니다. 첫 번째 유형의 모델은 파인튜닝(FT, Fine-Tuning) 기법에 기반하며, 주석이 달린 훈련 세트를 학습하기 위해 언어 모델의 모든 매개변수를 조정하여 SQL을 출력합니다. 반면에, 두 번째 유형의 모델은

<span id="page-5-1"></span><sup>‡</sup>BIRD 평가에서, 우리는 동일한 CPU에서 각 SQL을 100번 실행하고 이상치를 제거한 후 평균 결과를 평가합니다.

![](_page_6_Figure_0.jpeg)

<span id="page-6-1"></span>그림 5: 막대 차트는 BIRD에서 고급 모델의 성능을 명확하게 시각화합니다.

표 2: BIRD에서 고급 text-to-SQL 모델의 실행 정확도(EX). 인간 성능도 제공됩니다.

<span id="page-6-0"></span>

| Models            | Development Data |                | Testing Data  |                |  |
|-------------------|------------------|----------------|---------------|----------------|--|
|                   | w/o knowledge    | w/ knowledge   | w/o knowledge | w/ knowledge   |  |
|                   |                  | FT-based       |               |                |  |
| T5-Base           | 6.32             | 11.54 (+5.22)  | 7.06          | 12.89 (+5.83)  |  |
| T5-Large          | 9.71             | 19.75 (+10.04) | 10.38         | 20.94 (+10.56) |  |
| T5-3B             | 10.37            | 23.34 (+12.97) | 11.17         | 24.05 (+12.88) |  |
|                   |                  | ICL-based      |               |                |  |
| Palm-2            | 18.77            | 27.38 (+8.61)  | 24.71         | 33.04 (+8.33)  |  |
| Codex             | 25.42            | 34.35 (+8.93)  | 24.86         | 36.47 (+11.61) |  |
| ChatGPT           | 24.05            | 37.22 (+13.17) | 26.77         | 39.30 (+12.53) |  |
| ChatGPT + COT     | 25.88            | 36.64 (+10.76) | 28.95         | 40.08 (+11.24) |  |
| Claude-2          | 28.29            | 42.70 (+14.41) | 34.60         | 49.02 (+14.42) |  |
| GPT-4             | 30.90            | 46.35 (+15.45) | 34.88         | 54.89 (+20.01) |  |
| GPT-4 + DIN-SQL   | -                | 50.72          | -             | 55.90          |  |
| Human Performance | -                | -              | 72.37         | 92.96 (+20.59) |  |

인컨텍스트 학습(ICL, In-Context Learning)을 기반으로 하며, 추가 훈련 없이 결과를 생성할 수 있습니다. FT 모델에서, 우리는 T5 계열 [[38]](#page-13-1)을 주요 기준 모델로 선택합니다. ICL 기반 모델의 경우, Codex (code-davinci-002), ChatGPT (gpt-3.5-turbo), GPT-4 (gpt-4-32k), Claude-2 (claude-2.0), Palm-2 (text-bison-001)의 제로샷 결과를 제공합니다. 또한, 우리는 BIRD 데이터셋이 제안하는 도전 과제를 평가하기 위해 SPIDER의 최첨단(SOTA, State-Of-The-Art) 모델인 DIN-SQL [[35]](#page-12-2)도 구현합니다. 표 [2,](#page-6-0) 표 [3](#page-7-0) 및 그림 [5](#page-6-1)는 BIRD에서 고급 언어 모델의 전체 결과를 제시합니다.

#### 6.2 실행 정확도 분석 (Execution Accuracy Analysis)

표 [2](#page-6-0)와 그림 [5](#page-6-1)는 BIRD에서 다양한 모델의 계층화된 성능을 제시합니다. GPT-4는 모든 기준 언어 모델을 능가합니다. Claude-2가 그 뒤를 바짝 따르며, 의미론적 파싱과 지식 추론에서 뛰어난 능력을 보여줍니다. 더 나아가, [[35]](#page-12-2)의 전용 추론 프롬프트를 통합함으로써, DIN-SQL + GPT-4는 BIRD에서 새로운 최첨단 결과를 달성할 수 있습니다. 이는 값 샘플링, 퓨샷 데모 및 자기 수정을 포함합니다. 언어 모델 학습(LLM)과 프롬프트 지능의 상당한 발전에도 불구하고, 이러한 모델의 성능은 인간 능력에 분명히 뒤처집니다. 이러한 격차는 BIRD의 복잡한 특성을 강조할 뿐만 아니라 실제 text-to-SQL 시나리오에 적용 가능한 더 유능한 모델이나 고급 추론 프롬프트 방법을 발견할 기회를 제시합니다.

표 3: BIRD에서 고급 text-to-SQL 모델의 유효 효율성 점수(VES). 인간 성능도 제시됩니다.

<span id="page-7-0"></span>

| Models            | Development Data |                | Testing Data  |                |  |  |  |  |
|-------------------|------------------|----------------|---------------|----------------|--|--|--|--|
|                   | w/o knowledge    | w/ knowledge   | w/o knowledge | w/ knowledge   |  |  |  |  |
| FT-based          |                  |                |               |                |  |  |  |  |
| T5-Base           | 7.78             | 12.90 (+5.12)  | 8.97          | 14.71 (+5.74)  |  |  |  |  |
| T5-Large          | 9.90             | 22.74 (+12.84) | 12.25         | 25.00 (+12.75) |  |  |  |  |
| T5-3B             | 13.62            | 25.57 (+11.95) | 15.17         | 27.80 (+12.63) |  |  |  |  |
| ICL-based         |                  |                |               |                |  |  |  |  |
| Palm-2            | 20.82            | 28.64 (+7.82)  | 31.32         | 38.41 (+7.09)  |  |  |  |  |
| Codex             | 33.37            | 43.41 (+10.04) | 35.40         | 41.60 (+6.20)  |  |  |  |  |
| ChatGPT           | 27.97            | 43.81 (+15.84) | 36.68         | 51.40 (+14.72) |  |  |  |  |
| ChatGPT + COT     | 32.33            | 42.30 (+9.97)  | 49.69         | 56.56 (+6.87)  |  |  |  |  |
| Claude-2          | 32.75            | 45.28 (+12.53) | 39.32         | 55.77 (+16.45) |  |  |  |  |
| GPT-4             | 34.60            | 49.77 (+15.17) | 40.20         | 60.77 (+20.57) |  |  |  |  |
| GPT-4 + DIN-SQL   | -                | 58.79          | -             | 59.44          |  |  |  |  |
| Human Performance | -                | -              | 70.36         | 90.27 (+19.91) |  |  |  |  |

#### 6.3 Spider에서의 기준 성능 (Baseline Performance on Spider)

![](_page_7_Figure_3.jpeg)

<span id="page-7-1"></span>그림 6: SPIDER와 BIRD 개발 세트에서 동일한 기준 모델의 EX 결과.

SPIDER [[53]](#page-14-4)는 가장 널리 사용되고 복잡한 크로스 도메인 text-to-SQL 벤치마크입니다. 이는 주로 스키마 관련 의미론적 파싱 능력을 평가하는 데 초점을 맞춥니다. 복잡한 데이터베이스 스키마와 값으로 인한 BIRD 데이터셋의 증가하는 어려움을 입증하기 위해, 우리는 BIRD와 SPIDER 데이터셋 모두에서 동일한 기준 모델의 실행 정확도를 시각화합니다. 공정한 평가를 보장하기 위해, 모든 모델에는 값에 대한 지식이 제공되며, 두 데이터셋에 걸쳐 언어 모델(LM)에 대해 동일한 프로그래밍 프롬프트가 구현됩니다. 그림 [6](#page-7-1)은 데이터베이스 값에 대한 집중이 BIRD를 가장 도전적인 text-to-SQL 벤치마크로 만든다는 것을 보여줍니다. 각 모델의 성능 차이는 복잡한 데이터베이스 스키마와 값을 처리할 수 있는 모델의 추가 연구 및 개발의 필요성을 보여줍니다.

#### 6.4 효율성 분석 (Efficiency Analysis)

표 [3](#page-7-0)에 따르면, 더 높은 EX를 가진 모델이 더 높은 VES를 달성할 가능성이 더 높다는 것을 관찰할 수 있습니다. 이는 text-to-SQL 모델이 실용적인 목적을 달성하는 더 높은 VES를 얻기 위해서는 결과를 정확하게 예측해야 한다는 전제 조건으로 설명할 수 있습니다.

**2단계 최적화 (Two-Stage Optimization).** 직관적으로, text-to-efficient-SQL 변환의 목표는 두 개의 하위 단계로 분해될 수 있습니다. 이전 text-to-SQL 작업을 따라, 첫 번째 하위 단계인 의미론적 파싱은 질문을 SQL 쿼리로 정확하게 변환하는 데 집중합니다. 두 번째 하위 단계는 동일한 결과를 유지하면서 더 효율적이 되도록 SQL 쿼리를 재작성하여 최적화하는 것을 포함합니다 [[61]](#page-14-8). 이 접근법의 효과를 입증하기 위해, 우리는 ChatGPT가 정확하게 결과를 예측한 개발 세트에서 10개의 무작위 예제를 선택했습니다. 그런 다음 우리의 전문가는 확립된 쿼리 최적화 규칙 [[28,](#page-12-7) [34,](#page-12-8) [62]](#page-15-0)을 기반으로 이러한 쿼리를 최적화합니다. 우리는 2단계 최적화가 동일한 결과를 유지하면서 평균 77.75%의 시간 절감을 가져온다는 것을 관찰합니다.

**데이터베이스와의 대화 (Chat w/ Database).** BIRD는 모델이 데이터베이스와 상호 작용하는 전역 SQL 쿼리를 생성하여 데이터 타입과 분포를 인식할 수 있게 하는 "데이터베이스와의 대화"라는 새로운 모드를 도입합니다. 이 접근법은 더 효과적이고 효율적인 SQL 쿼리 개발의 기초를 마련합니다. 실험에서 관찰된 바와 같이, SQL 쿼리의 시간 절감 비율은

![](_page_8_Figure_0.jpeg)

<span id="page-8-0"></span>그림 7: BIRD에서 고급 대규모 언어 모델의 세분화된 범주별 평가.

데이터베이스 내에 인덱스를 구성함으로써 87.3%에 도달할 수 있습니다. 자세한 효율성 분석은 부록 B.5에 제시되어 있습니다.

#### 6.5 지식 증거 분석 (Knowledge Evidence Analysis)

우리는 두 가지 시나리오에 대해 각 기준 모델을 구현합니다. 첫 번째는 각 샘플에 대해 정답 외부 지식 증거 문장을 제공하지 **않는** 것입니다(w/o knowledge). 다른 테스트 베드는 이러한 증거를 제공하고(w/ knowledge) text-to-SQL 모델이 스스로 지식 그라운딩을 수행하도록 하는 것입니다. 섹션 3.3에서 논의한 바와 같이, 외부 지식 증거 문장에 대한 전문가 주석이 모델의 데이터베이스 값 이해를 향상시키기 위해 사용됩니다.

데이터베이스 값에 대한 외부 지식 증거를 쉽게 제공받은 후, 모든 모델은 표 2와 표 4에서 보듯이 다양한 난이도 수준에 걸쳐 명확한 개선을 보입니다. *이는 BIRD의 외부 지식 증거가 모델이 데이터베이스 값을 더 잘 이해하는 데 효과적이고 유익하다는 것을 나타냅니다.* 또한 이는 더 실제적인 데이터베이스에 직면했을 때 데이터베이스 값이 text-to-SQL 모델에 매우 중요하다는 것을 보여줍니다. 또한, ICL 기반 접근법은 5B 매개변수 미만의 FT 소규모 모델보다 더 나은 자체 지식 그라운딩 능력과 사전 훈련된 SQL 지식을 가지고 있습니다. COT를 갖춘 ChatGPT는 지식과 데이터가 저자원일 때 다단계 추론이 유익하기 때문에 더 나은 성능을 보일 수 있습니다. 이에도 불구하고, 우리는 COT 버전에 대해 ChatGPT + 외부 지식 증거의 성능 감소 또는 제한된 개선을 관찰합니다. 우리는 LLM의 내부 다단계 지식 추론이 이 상황에서 외부 지식(증거)의 방식과 호환되지 않는다고 가설을 세웁니다. 따라서 LLM의 강력한 다단계 자체 추론 능력을 외부 지식 추론과 일관되게 효과적으로 결합하는 방법의 개발은 유망한 미래 방향을 제시합니다 [29].

#### 6.6 추가 분석 (More Analysis)

**세분화된 범주 분석 (Fine-grained Category Analysis).** 그림 7은 BIRD에서 고급 LLM의 하위 능력의 다양한 차원에 대한 자세한 비교를 제공합니다. 결과는 GPT-4가 모든 영역에서 ChatGPT와 Claude-2에 대해 우수한 성능을 보인다는 것을 나타냅니다. 그럼에도 불구하고, 모든 모델 간에 순위와 수치 계산(수학)의 성능에서 주목할 만한 차이가 있습니다. 이러한 제한은 이러한 작업이 항상 모호한 사용자 쿼리의 맥락 내에서 수학적 계산과 순위를 통합하기 때문에 심층 데이터 과학 작업에 대한 현대 LLM의 부적절함을 시사할 수 있습니다. 반대로, 이러한 모델은 도메인 지식, 동의어 감지 및 값 설명에서 상대적으로 더 나은 성능을 보여주며, 이는 사전 훈련 단계 동안 충분한 언어 훈련과 추론 능력에 기인할 수 있습니다.

**인간 성능 (Human Performance).** 실제 시나리오에서 응용 수준의 성능을 달성하기 위한 text-to-SQL 연구의 노력을 활성화하기 위해, 우리는 BIRD에서 인간 성능을 제공합니다. 표 2, 표 3은 최첨단 text-to-SQL 모델과 인간 성능 사이에 여전히 큰 격차가 있음을 보여줍니다. 절차에 대한 철저한 소개는 부록 B.9에 있습니다.

**오류 분석 (Error Analysis).** ChatGPT는 현재 가장 널리 사용되고 비용 효율적인 LLM입니다. 따라서 ChatGPT의 성능이 이 오류 분석에 집중됩니다. 자세한 분석은 부록 B.6에 있습니다. 우리는 500개의 무작위로 샘플링된 오류 사례를 관찰하여 다음 범주에서 심층 평가를 제공합니다.

**잘못된 스키마 링킹 (Wrong Schema Linking) (41.6%)** 은 ChatGPT가 데이터베이스의 구조를 정확하게 이해할 수 있지만 부적절한 열과 테이블과 잘못 연관시키는 시나리오를 나타냅니다. 이는 스키마 링킹 [[43,](#page-13-4) [57]](#page-14-7) 작업이 복잡하고 실용적인 상황에서도 모델에게 여전히 중요한 장애물로 남아 있음을 보여줍니다. **데이터베이스 콘텐츠 오해 (Misunderstanding Database Content) (40.8%)** 는 특히 데이터베이스가 매우 클 때 ChatGPT가 올바른 데이터베이스 구조를 기억하지 못하거나(예: rtype이 satscores 테이블에 속하지 않음) 가짜 스키마 항목을 생성하는(예: lap_records가 formula_1 데이터베이스에 나타나지 않으며 많은 값이 잘못 예측됨) 경우에 발생합니다. 이 경우, ChatGPT가 데이터베이스 구조와 값을 실제로 이해하도록 만드는 방법 [[27]](#page-12-10)은 여전히 LLM의 난제입니다. **지식 증거 오해 (Misunderstanding Knowledge Evidence) (17.6%)** 는 모델이 인간이 주석을 단 증거를 정확하게 해석하지 못하는 경우를 나타냅니다. 한 예는 ChatGPT가 공식 DIVIDE(SUM(spent), COUNT(spent))를 직접 복사하는 것입니다. 이 발견은 ChatGPT가 익숙하지 않은 프롬프트나 지식에 대한 응답에서 견고성이 부족하여 SQL 구문을 고려하지 않고 공식을 직접 복제한다는 것을 보여줍니다 [[15]](#page-11-4). 우리는 또한 ChatGPT가 때때로 잘못된 키워드를 사용하거나(예: SQLite 함수 STRFTIME() 대신 MySQL Year() 함수를 잘못 사용) 디코딩 오류를 보인다는 것을 관찰합니다.

## 7 관련 연구 (Related Work)

고품질 데이터셋은 text-to-SQL을 포함한 다양한 자연어 처리 작업을 발전시키는 데 중요합니다. GeoQuery [[55]](#page-14-0), ATIS [[9]](#page-11-5), Restaurant [[20]](#page-11-6)와 같은 초기 단일 도메인 text-to-SQL 데이터셋은 특정 정보 검색 작업을 대상으로 했으며, WikiSQL [[58]](#page-14-5) 및 SPIDER [[53]](#page-14-4)와 같은 최근 데이터셋은 도메인 일반화를 요구하는 크로스 도메인 데이터셋을 제안합니다. 그러나 대부분의 크로스 도메인 text-to-SQL 데이터셋은 여전히 값보다는 데이터베이스 스키마를 강조하여 실제 시나리오와 다릅니다. KaggleDBQA [[24]](#page-12-5)는 Kaggle의 8개 데이터베이스에서 272개의 text-to-SQL 쌍을 구성하여 이를 해결했으며, EHRSQL [[25]](#page-12-6), SEDE [[13]](#page-11-7), MIMICSQL [[46]](#page-13-5)과 같은 다른 데이터셋은 더 전문적인 SQL 쿼리와 함께 다양하고 대규모 값 데이터베이스를 수집했습니다. 이러한 발전에도 불구하고, 이러한 데이터셋은 단일 도메인에 초점을 맞추고 있습니다. 최근 연구는 지식 집약적 text-to-SQL 벤치마크를 탐구했으며 [[10,](#page-11-2) [57]](#page-14-7), 지식 그라운딩을 통해 전문가의 실제 분석을 지원합니다. BIRD는 데이터베이스 값을 강조하면서 이러한 실제 기능을 통합하는 최초의 대규모 벤치마크입니다.

# 8 한계 및 향후 연구 (Limitation and Future work)

이중 맹검 주석에 의해 생성된 SQL 주석의 높은 품질에도 불구하고, 이 절차는 자원 집약적입니다. 향후 연구는 데이터 품질을 유지하면서 인간의 노력을 줄이기 위해 주석 업무의 일부를 담당하는 GPT-4와 같은 고급 AI 시스템을 통합하는 인간-컴퓨터 상호작용(HCI, Human-Computer Interaction) 기반 접근법을 탐구할 수 있습니다. 또한, SQLite는 사용자 친화적이기 때문에 이전 text-to-SQL 벤치마크와 이 연구의 주요 SQL 코드베이스로 선택되었습니다. 그러나 이는 정확한 효율성 계산을 위한 쿼리 실행 계획(QEP, Query Execution Plan)을 가져오고 다양한 SQL 구문에 적응하는 데 어려움을 제시합니다. 향후 작업은 이러한 제한을 해결하고 NLP와 DB 전문가 모두에게 더 견고한 연구 환경을 제공하기 위해 BIRD의 PostgreSQL 및 MySQL 버전을 포함할 것입니다.

### 9 결론 (Conclusion)

이 논문에서 우리는 대규모 데이터베이스 값에 특히 초점을 맞춘 대규모 크로스 도메인 text-to-SQL 벤치마크인 BIRD를 소개합니다. BIRD는 1) 대규모이고 더러운 데이터베이스 값 처리, 2) 외부 지식 증거, 3) SQL 실행 효율성 최적화라는 세 가지 추가 도전 과제를 탐구하여 text-to-SQL 연구와 실제 응용 프로그램 간의 격차를 완화합니다. 우리의 실험 결과는 가장 인기 있고 강력한 LLM인 ChatGPT조차도 인간 성능에 훨씬 못 미치기 때문에 BIRD가 기존 벤치마크에 비해 더 도전적인 과제를 제시한다는 것을 보여줍니다. 이는 text-to-SQL 작업에서 개선과 혁신의 여지를 많이 남깁니다. 더욱이, 우리의 철저한 효율성 및 오류 분석은 향후 연구를 위한 귀중한 통찰력과 방향을 제공하여 실제 시나리오에서 더 발전되고 실용적인 text-to-SQL 솔루션 개발의 길을 열어줍니다.

# 감사의 글 (Acknowledgement)

익명의 심사자들의 모든 건설적인 의견에 감사드립니다. Reynold Cheng, Jinyang Li, Ge Qu 및 Nan Huo는 홍콩 경마회 자선 신탁(프로젝트 260920140)과 홍콩대학교(프로젝트 104006830)의 지원을 받았습니다. Chenhao Ma는 NSFC 보조금 62302421, 광둥성 기초 및 응용 기초 연구 기금 보조금 2023A1515011280, 선전 과학 기술 프로그램 ZDSYS20211021111415025의 지원을 받았습니다. Jinyang Li와 Ge Qu는 HKU Presidential PhD Scholar Programme의 지원을 받았습니다. Ge Qu는 또한 홍콩 PhD Fellowship Scheme의 지원을 받았습니다. 이 연구는 Alibaba Research Intern Program을 통해 Alibaba Group의 지원을 받았습니다.

# 참고문헌 (References)

- <span id="page-10-1"></span>[1] Peter Alsberg. Space and time savings through large data base compression and dynamic restructuring. *Proceedings of the IEEE*, 63:1114–1122, 1975.
- <span id="page-10-2"></span>[2] Anthropic. Introducing Claude. 2023. URL [https://www.anthropic.com/index/](https://www.anthropic.com/index/introducing-claude) [introducing-claude](https://www.anthropic.com/index/introducing-claude).
- <span id="page-10-0"></span>[3] Ruichu Cai, Boyan Xu, Zhenjie Zhang, Xiaoyan Yang, Zijian Li, and Zhihao Liang. An encoder-decoder framework translating natural language to database queries. In *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI-18)*, page 3977–3983, 2018.
- <span id="page-10-6"></span>[4] Zefeng Cai, Xiangyu Li, Binyuan Hui, Min Yang, Bowen Li, Binhua Li, Zheng Cao, Weijie Li, Fei Huang, Luo Si, and Yongbin Li. STAR: SQL guided pre-training for context-dependent text-to-SQL parsing. In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 1235–1247, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.
- <span id="page-10-5"></span>[5] Ruisheng Cao, Lu Chen, Zhi Chen, Yanbin Zhao, Su Zhu, and Kai Yu. LGESQL: Line graph enhanced text-to-SQL model with mixed local and non-local relations. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, pages 2541–2555, Online, August 2021. Association for Computational Linguistics.

