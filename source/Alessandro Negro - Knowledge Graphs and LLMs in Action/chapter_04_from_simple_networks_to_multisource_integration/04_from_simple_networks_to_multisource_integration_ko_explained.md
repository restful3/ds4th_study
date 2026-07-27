---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 단순한 네트워크에서 다중 소스 통합으로 — 쉬운 해설판

> 이 글은 Alessandro Negro의 『Knowledge Graphs and LLMs in Action』 4장을 한국어로 풀어 쓴 해설판입니다. 원문의 모든 문단, 그림, 표, 수식, 코드, 인용을 빠짐없이 다루되, 번역을 넘어 "왜 이렇게 하는지"를 대화하듯 설명합니다. 생명의학(biomedical) 데이터를 예로 들지만, 여기서 배우는 통합·분석 기법은 다른 어떤 분야에도 그대로 옮겨 쓸 수 있습니다.

이 장을 관통하는 두 배우를 미리 소개하겠습니다. 앞 장들에서 정한 비유를 이어가자면, **LLM(대규모 언어 모델)** 은 세상만사를 다 아는 것 같지만 가끔 그럴듯하게 지어내는 달변가이고, **지식 그래프(Knowledge Graph, KG)** — 사실을 노드와 관계로 표현한 검증된 지식 저장소 — 는 꼼꼼하게 검증된 사실 대장입니다. 이 장에서는 특히 사실 대장을 어떻게 여러 출처로부터 크고 정교하게 짓느냐가 주인공이고, 달변가는 그 대장을 사람이 이해하도록 풀어 읽어 주는 보조 역할로 등장합니다.

---

## 이 장에서 다루는 것 — 큰 그림 미리 보기

원문은 이 장의 목표를 네 가지로 요약합니다. 하나씩 우리말로 풀어 보겠습니다.

- **복잡한 지식 그래프를 짓고 통합하기.** 여러 데이터 소스를 하나의 그래프로 합치는 일입니다.
- **지식 그래프의 실제 사례 살펴보기.** 생명의학 분야의 대표적인 그래프들을 직접 만져 봅니다.
- **분석·질의 기법 이해하기.** 그래프에서 원하는 정보를 끄집어내는 방법입니다.
- **LLM으로 지식 그래프 결과 해석하기.** 숫자로 나온 분석 결과를 사람이 읽을 수 있는 이야기로 바꾸는 단계입니다.

이 장은 점점 더 크고 복잡한 지식 그래프를 어떻게 만들고, 이를 이용해 **지능형 자문 시스템(Intelligent Advisor System, IAS)** — 사람의 의사결정을 돕는 똑똑한 조언 시스템 — 을 어떻게 개발하는지에 대한 이해를 넓혀 줍니다. 3장에서는 온톨로지(ontology) 형태의 지식 베이스 하나만 다뤘다면, 지금부터는 그래프로 다루기 좋은 형식으로 이미 제공되는 여러 구조화 데이터 소스로부터 지식 그래프를 만들 것입니다. 이렇게 접근하면 그래프를 어떻게 모델링할지, 여러 소스를 어떻게 통합할지, 어떤 방법으로 분석할지에 집중할 수 있습니다.

> **참고** 책 웹사이트의 부록 C는 복잡한 여러 소스에서 원시 데이터를 가져와 변환하는 방법을 폭넓게 안내합니다.

다음 절들의 예제는 다음과 같은 작업들을 다룹니다. 구조화·반구조화된 스키마와 데이터 형식을 하나의 동질적인(homogeneous) 그래프로 변환하기, 이름과 식별자를 서로 맞춰(reconcile) 짝짓기, 엔티티와 관계를 병합하는 후처리 기법, 그리고 만들어진 지식 그래프를 분석해 필요한 정보를 찾아내기입니다. 예제는 생명의학 데이터 소스를 사용하지만, 여기서 쓰는 기법과 패턴은 다른 도메인에도 곧바로 옮겨 쓸 수 있습니다.

여기서 짚어 둘 점이 있습니다. 이 단계에서 **LLM은 보조적이지만 제한된 역할**만 합니다. 이 장의 데이터 소스는 CSV 파일, 관계형 데이터베이스, API처럼 구조가 이미 잡혀 있습니다. 그래서 전통적인 데이터 통합 기법이 주된 방법이 되고, LLM은 구축 파이프라인의 핵심 부품이 아니라 곁다리 도구로 쓰입니다. 데이터가 이미 정돈돼 있으면 굳이 지어낼 위험이 있는 달변가에게 통합 작업을 맡길 필요가 없다는 뜻입니다.

---

## 4.1 생명의학 지식 그래프와 그 활용 — 어떤 문제를 풀 수 있나

여러분이 다음 상황 중 하나를 맡았다고 상상해 보겠습니다.

- 질병과 단백질 사이에 이미 알려진 관계에서 출발해, **새로운 연결**을 발견할 수 있을까요?
- 값비싼 시험관(in vitro) 실험 없이도 마이크로 RNA와 질병 사이의 의미 있는 관계를 찾아낼 수 있을까요?
- 셀리악병(celiac disease) 같은 특정 질병에 관여하는 핵심 과정은 무엇일까요?
- 수년에 걸친 연구 없이 기존 약물의 **새로운 용도(repurposing)** 를 찾아낼 수 있을까요?
- 환자 개개인의 정보를 활용하는 **정밀 의료(precision medicine)** 를 어떻게 뒷받침할 수 있을까요?

이 모든 과제는 기존 생명의학 지식을 그래프 형태로 조직하고, 그 그래프가 복잡한 질문에 답할 만한 지식을 담고 있는지 판단함으로써 풀 수 있습니다. 즉, 흩어진 사실들을 노드와 관계로 엮어 두면, 겉으로 드러나지 않던 연결을 그래프 위에서 추론할 수 있게 됩니다.

먼저 예제 도메인의 범위를 정해 두겠습니다. 생명의학(biomedical science)은 인체의 장기와 시스템을 다루며, 주로 질병, 유전자 발현, 단백질, 약물 및 관련 주제에 초점을 맞춥니다. Nicholson과 Green [1]이 보고한 대로, 지식 그래프는 연구자가 다음과 같은 생명의학 문제를 다루는 데 도움이 됩니다. 기존 약물의 새로운 용도 찾기 [2], 환자 진단 [3], 질병과 생체분자 사이의 연관성 식별 [4], 단백질의 기능 규명 [5], 암 유전자 우선순위화 [6], 그리고 환자에게 더 안전한 약물 추천 [7, 8]입니다. 각 응용은 서로 다른 비즈니스 목표를 가지며, **CRISP-DM(데이터 마이닝 표준 절차)** 모델에 따라 서로 다른 데이터 소스로 구축됩니다.

원문은 응용 유형마다 사례 연구를 하나씩 제시하고, 소스 데이터베이스를 가져와 병합한 뒤 결과 그래프를 질의·분석하는 코드까지 함께 보여 줍니다. 이 연습을 통해 여러분은 지식 그래프에 넣을 데이터 소스를 어떻게 고르는지, 그리고 그 정보가 원하는 작업을 해내기에 충분한지를 어떻게 판단하는지 배우게 됩니다. 그림 4.1은 다양한 응용과, 노드·관계로 저장되는 가장 중요한 정보들을 요약합니다.

![](images/ko/figure-4-1-ko.png)

그림 4.1 비즈니스 목표별로 묶은 지식 그래프의 주요 생명의학 응용 유형. 이들은 공통으로 쓰는 데이터 소스가 많습니다.

---

## 4.2 지식 그래프의 다중 오믹스 응용 — 유전체에서 단백질까지

**다중 오믹스(multi-omic)** 란 유전체(genome), 단백질체(proteome), 전사체(transcriptome)처럼 여러 "오믹스(omics)" 데이터셋을 함께 사용하는 생물학적 분석 접근을 말합니다(그림 4.2 참고). 분자생물학에서 접미사 **-ome** 은 "전체 총합"을 뜻합니다. 예를 들어 genome(유전체)은 한 생물의 유전 정보 전체를 가리킵니다.

![](images/ko/figure-4-2-ko.png)

그림 4.2 생명의학 지식 그래프에서 쓰이는 세 가지 주요 '오믹스' 데이터 유형. 유전체(DNA), 전사체(RNA), 단백질체(단백질) 데이터는 전사(transcription)와 번역(translation)을 통해 생물학적으로 연결됩니다.

### 유전체, 전사체, 단백질체 — 세 용어 정리

이 장 전체에서 다음 용어와 관련 개념이 반복해서 등장하므로 미리 정리합니다.

- **유전체(genome)** — 살아 있는 생물의 유전 정보 전체입니다. 대부분의 유전체는 DNA(디옥시리보핵산)로 이루어져 있지만, 일부 바이러스는 RNA(리보핵산) 유전체를 가집니다. DNA와 RNA는 뉴클레오타이드(nucleotide)라는 단량체 소단위가 사슬처럼 이어진 중합체 분자입니다.
- **전사체(transcriptome)** — 단백질체의 합성을 지시하는 RNA 분자들의 집합입니다. 전사체는 전사(transcription)라는 과정으로 만들어지는데, 이 과정에서 개별 유전자가 RNA 분자로 복사됩니다.
- **단백질체(proteome)** — 유전체 발현의 최종 산물로, 살아 있는 세포가 합성한 기능성 단백질 전체를 아우릅니다. 이는 유전체 발현의 정점이자, 세포 생명을 이루는 생화학 활동의 출발점이기도 합니다.

많은 다중 오믹스 응용은 지식 그래프를 이용해 유전체를 연구하고, 유전자가 전사체에서 어떻게 발현되는지, 그리고 그 전사 산물이 단백질체에서 어떻게 상호작용하는지를 연구합니다. 대표적인 예로 miRNA-질병 연관성 탐지 [4], 유전자–증상 우선순위화 [9], 단백질–단백질 상호작용 예측 [10, 11, 12]이 있습니다.

예를 들어 Yang 등 [9]은 주어진 증상과 연관된 후보 유전자를 찾아내는 지식 그래프 모델을 제안했습니다. 연구진은 여러 이질적인 데이터 소스를 병합했습니다. 질병 용어를 통일하고 통합하기 위해, 서로 다른 데이터베이스의 질병 식별자를 **UMLS(Unified Medical Language System)** — 미국 국립의학도서관이 관리하는 통합 의학 용어 체계(https://www.nlm.nih.gov/research/umls/index.html) — 코드로 매핑했습니다. 그림 4.3이 그 과정을 보여 줍니다.

![](images/ko/figure-4-3-ko.png)

그림 4.3 Yang 등 [9]의 그림에서 발췌한 것으로, 연구진이 여러 데이터 소스를 결합해 하나의 통합적(holistic) 지식 그래프를 만든 방식을 보여 줍니다.

또 다른 사례에서는 **단백질–단백질 상호작용(Protein–Protein Interaction, PPI)** 네트워크와 단백질–질병 연관성이 질병 경로(특정 질병과 연관된 단백질 무리)를 계산으로 발견하는 데 성공적으로 쓰였습니다 [12]. 이것은 지식 그래프가 핵심 역할을 하는 지능형 자문 시스템을 개발하기에 딱 맞는 사례입니다. 왜냐하면 각 질병 단백질을 따로 떼어 이해하려 해서는 대부분의 인간 질병을 온전히 설명할 수 없기 때문입니다. 여러 데이터 소스를 병합해야 하는 더 복잡한 시나리오로 넘어가기 전에, 먼저 이 비교적 단순한 유형의 지식 그래프를 어떻게 만들고 분석하는지 살펴보겠습니다.

---

### 4.2.1 PPI와 단백질–질병 네트워크로 지식 그래프 만들기

여기서의 목표는 이미 알려진 경로에서 출발해 새로운 **질병 경로(disease pathway)** 를 발견하는 것입니다. Agrawal 등 [12]이 제안한 접근은 그림 4.4처럼 지식 그래프에서 시작합니다. 질병은 그것과 연관된, 이미 알려진 단백질들과 연결되고, 그 단백질들은 다시 PPI 네트워크 안에서 서로 연결됩니다. 예를 들어 그림 4.5는 셀리악병과 관련 유전자들 사이의 연결을 보여 줍니다.

![](images/ko/figure-4-4-ko.png)

그림 4.4 질병 경로는 그 질병과 연관된 단백질 집합으로 정의되는, PPI 네트워크의 부분 그래프(subgraph)입니다.

![](images/ko/figure-4-5-ko.png)

그림 4.5 Agrawal [12]이 구축한 지식 그래프의 작은 일부. 셀리악병에서 출발해 연관된 유전자들을 찾은 모습입니다.

이제 이 단순한 지식 그래프로 무엇을 할지, 그 발견 과정(discovery process)을 살펴보겠습니다. 그림 4.6은 출발점과 결과를 보여 줍니다. 우리에게는 이미 알려진 경로들의 집합이 있고, 지능형 자문 시스템은 해당 질병과 연관될 가능성이 있는 단백질과 관련 경로의 집합을 예측해 보고해야 합니다. 이 단백질들은 기존 경로의 일부일 수도 있고, 새로운 경로를 이룰 수도 있습니다. 결과로 만들어지는 지식 그래프는 **단일 분할 그래프(monopartite graph)** — 한 종류의 노드끼리 연결된 그래프인 PPI 네트워크 — 와 **이분 그래프(bipartite graph)** — 두 종류의 노드가 서로 연결된 그래프인 질병–단백질 연관 네트워크 — 로 구성됩니다.

이 경우 우리에게 필요한 데이터 소스가 이미 있습니다. 스탠퍼드 네트워크 분석 프로젝트(Stanford Network Analytics Project, SNAP; http://snap.stanford.edu/pathways/)가 제공하는 "인간 인터랙톰의 질병 경로(Disease Pathways in the Human Interactome)" 데이터를 쓸 수 있습니다. Agrawal [12]이 더 복잡한 데이터 소스에서 출발해 이 비교적 단순한 네트워크를 만들어 두었고, 우리는 이를 가져와 탐색할 수 있습니다. 그런 다음 또 다른 데이터셋과 결합해, 만들어진 지식 그래프를 사람이 더 읽기 쉽게 만들 것입니다.

![](images/ko/figure-4-6-ko.png)

그림 4.6 질병과 관련된 단백질을 찾아내는 발견 과정

이제 데이터를 가져올 차례입니다. 다음의 노드 키 제약 조건(node key constraint)은 특정 레이블을 가진 모든 노드가 ID 값을 반드시 가지도록, 그리고 그 값이 유일하도록 보장합니다.

```sql
Listing 4.1 Creating the constraints
CREATE CONSTRAINT protein_key IF NOT EXISTS FOR
If you aren’t sure a constraint exists,
(n:Protein) REQUIRE (n.id) IS NODE KEY;
add IF NOT EXISTS to ensure that it
CREATE CONSTRAINT disease_key IF NOT EXISTS FOR is created only if it doesn’t exist.
(n:Disease) REQUIRE (n.id) IS NODE KEY;
```

여기서 `IF NOT EXISTS` 는 제약이 이미 존재하는지 확신이 서지 않을 때 붙입니다. 그러면 아직 없을 때만 제약이 만들어지므로, 중복 생성 오류를 피할 수 있습니다.

우리 데이터셋에 처음 가져올 파일은 Menche 등 [13]과 Chatr-Aryamontri 등 [14]이 정리한 인간 PPI 네트워크입니다. 이 결과 그래프는 인간의 단백질 21,559개 사이에서 실험으로 문서화된 상호작용 342,354개를 담고 있습니다. Listing 4.2\~4.4는 파일들이 SNAP에서 내려받아 압축 해제된 뒤, Neo4j의 import 디렉터리 안 PPI 디렉터리로 옮겨졌다고 가정합니다.

```cypher
Listing 4.2 Importing the PPI network
:auto LOAD CSV FROM 'file:///PPI/bio-pathways-network.csv' AS line
CALL {
WITH line
MERGE (f:Protein {id: trim(line[0])})
MERGE (s:Protein {id: trim(line[1])})
MERGE (f)-[:INTERACTS_WITH]->(s)
} IN TRANSACTIONS OF 100 ROWS
```

이 질의는 CSV의 각 줄을 읽어, 첫 번째와 두 번째 열의 단백질 노드를 `MERGE`(있으면 재사용, 없으면 생성)한 뒤 둘 사이에 `INTERACTS_WITH`(상호작용) 관계를 만듭니다. `IN TRANSACTIONS OF 100 ROWS` 는 100줄씩 끊어 트랜잭션으로 처리해 메모리 부담을 줄이는 방식입니다.

#### 연습 문제

단백질의 수와 그들 사이의 연결 수를 확인하는 질의를 직접 실행해 보세요. 다음 단계가 새로운 단백질을 추가하므로, 반드시 그 전에 확인해 두어야 합니다.

다음으로는 단백질–질병 연관성을 튜플 (u, d) 형태로 가져옵니다. 여기서 단백질 u의 변형이 질병 d와 연결된다는 뜻입니다. 이 연관성은 질병에 관한 지식을 한데 모아 놓은 플랫폼인 **DisGeNET(www.disgenet.org)** 에서 옵니다. 이 데이터에는 21,000개가 넘는 단백질–질병 연관성이 담겨 있고, 각각 최소 10개 이상의 질병 단백질을 가진 519개 질병으로 나뉘어 있습니다.

```cypher
Listing 4.3 Importing pathways
:auto LOAD CSV WITH HEADERS
FROM 'file:///PPI/bio-pathways-associations.csv' AS line
CALL {
WITH line
WITH trim(line["Associated Gene IDs"]) AS proteins,
trim(line["Disease Name"]) AS diseaseName,
trim(line["Disease ID"]) AS diseaseId
MERGE (d:Disease {id: diseaseId, name: diseaseName})
WITH d, proteins
UNWIND split(proteins, ",") AS protein
WITH d, protein
MERGE (p:Protein {id: trim(protein)})
MERGE (d)-[:ASSOCIATED_WITH]->(p)
} IN TRANSACTIONS OF 100 ROWS
```

이 질의는 질병 노드를 만든 다음, 쉼표로 이어진 연관 유전자 ID 목록을 `split`으로 나누고 `UNWIND`로 한 줄씩 펼쳐, 각 단백질과 질병 사이에 `ASSOCIATED_WITH`(연관됨) 관계를 맺어 줍니다.

#### 연습 문제

몇 가지 확인 질의를 실행해 수치를 검증해 보세요. 단백질 수가 달라진 것을 알아챌 수 있을 것입니다. 새로 생긴 단백질을 찾아낼 수 있겠습니까? 힌트: `NOT EXISTS` 절을 사용하세요.

SNAP 데이터셋에서 내려받을 마지막 파일은 질병 분류(category)를 담고 있습니다. 질병은 **질병 온톨로지(Disease Ontology, https://disease-ontology.org/)** 를 이용해 범주와 하위 범주로 나뉘고, UMLS 코드에도 매핑됩니다. DisGeNET에서 가져온 519개 질병 중 290개가 온톨로지의 코드로 이어지는 UMLS 코드를 가지고 있습니다. 이 데이터셋은 온톨로지의 두 번째 계층을 사용하는데, 여기에는 암(68개 질병), 신경계 질환(44개), 심혈관계 질환(33개), 면역계 질환(21개)을 포함한 10개 범주가 있습니다.

```cypher
Listing 4.4 Importing disease classes
:auto LOAD CSV WITH HEADERS
➥FROM 'file:///PPI/bio-pathways-diseaseclasses.csv' AS line
CALL {
WITH line
WITH line["Disease ID"] as diseaseId, line["Disease Class"] as class
MATCH (d:Disease {id:diseaseId})
SET d.class = class
} IN TRANSACTIONS OF 100 ROWS
```

이 질의는 이미 존재하는 질병 노드를 `MATCH`로 찾아, 그 노드에 `class`(질병 범주) 속성을 `SET`으로 채워 넣습니다.

#### 연습 문제

가져오기가 끝난 뒤, 각 질병의 범주별 수치와, 범주가 없는 질병 목록을 확인해 보세요.

지금까지 SNAP 데이터셋의 데이터를 넣었지만, 이 데이터는 단백질을 코드로만 식별합니다. 가독성을 높이기 위해, 미국 국립보건원(NIH)에서 제공하는 유전자 정보(https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz)를 가져올 수 있습니다. 파일을 내려받은 뒤 압축을 풀고, PPI와 같은 디렉터리로 옮기면 됩니다.

```cypher
Listing 4.5 Importing gene information
:auto LOAD CSV WITH HEADERS FROM 'file:///PPI/gene_info' AS line
FIELDTERMINATOR '\t'
CALL {
WITH line
WITH trim(line["GeneID"]) AS proteinId, trim(line["Symbol"]) AS symbol,
trim(line["description"]) AS description
WITH proteinId, symbol, description
MATCH (p:Protein {id:proteinId})
SET p.name = symbol, p.description = description
} IN TRANSACTIONS OF 100 ROWS
```

이 질의는 유전자 ID로 기존 단백질 노드를 찾아, 사람이 읽기 좋은 심볼(name)과 설명(description)을 붙여 줍니다. 이렇게 하면 코드 대신 실제 유전자 이름으로 그래프를 탐색할 수 있어 훨씬 이해하기 쉬워집니다.

---

### 4.2.2 만들어진 지식 그래프의 고수준 분석 — 전체 구조 살펴보기

다음 단계는 지식 그래프를 점검해 그래프의 품질을 평가하고 데이터베이스를 탐색하는 것입니다. 만약 여기까지 직접 따라 하지 않았더라도 걱정 없습니다. https://mng.bz/5v7O 에서 Neo4j 백업을 내려받을 수 있습니다. 다음 Listing은 Neo4j 5.x에서 이 데이터베이스를 가져오는 방법을 보여 줍니다.

```text
Listing 4.6 Creating the PPI database from a Neo4j backup
Add the following line to the neo4j.conf file
dbms.databases.seed_from_uri_providers=URLConnectionSeedProvider
then run the following command
CREATE DATABASE ppi OPTIONS { existingData: "use",
➥seedUri: "https://mng.bz/5v7O"}
```

이 방법은 `neo4j.conf` 파일에 시드 제공자(seed provider) 설정을 한 줄 추가한 뒤, 원격 URI에서 데이터를 끌어와 `ppi` 데이터베이스를 만드는 절차입니다.

그래프 분석은 PPI 네트워크에 대한 일반적인 평가부터 시작합니다. 이런 종류의 평가에서 우리가 선호하는 알고리즘은 **약하게 연결된 컴포넌트(Weakly Connected Component, WCC)** — 그래프 안에서 서로 끊긴 부분 그래프들을 찾아내는 커뮤니티 탐지 알고리즘 — 입니다. 이 분석을 실행하기 위해 Neo4j에서 제공하는 **GDS(Graph Data Science) 라이브러리** 를 사용합니다(설치 방법은 온라인 부록 B 참고).

먼저 `INTERACTS_WITH` 관계를 이용해 서로 연결된 단백질에 표시를 남깁니다.

```cypher
Listing 4.7 Marking proteins in the PPI network
MATCH (p:Protein)-[:INTERACTS_WITH]-()
SET p:PPIProtein
```

이 질의는 상호작용 관계를 하나라도 가진 단백질에 `PPIProtein` 이라는 레이블을 붙여, 이후 분석 대상을 명확히 구분합니다.

이제 그래프를 메모리 안에 표현한 **프로젝션(projection)** — 분석용으로 메모리에 올린 그래프의 사본 — 을 만들어야 합니다.

```text
call gds.graph.project(  Name of the  List of node types we would like to
'ppi-graph',  ← in-memory graph  include. In this example, there is
'PPIProtein',  ← only one, but it can be a list.
{
INTERACTS_WITH: {  ← Here we have the list of relationships we
orientation: 'UNDIRECTED'  would like to include in the analysis.
```

이 호출(Listing 4.8)에서 `'ppi-graph'` 는 메모리에 올릴 그래프의 이름이고, `'PPIProtein'` 은 포함할 노드 유형의 목록입니다(여기서는 하나뿐이지만 목록이 될 수도 있습니다). 그리고 `INTERACTS_WITH` 는 분석에 포함할 관계이며, `orientation: 'UNDIRECTED'` 로 방향을 무시하고 무방향으로 다룹니다.

부분 그래프가 메모리에 만들어지면, 다음 질의로 WCC 알고리즘을 실행할 수 있습니다.

```cypher
Listing 4.9 Running the WCC on the PPI network
CALL gds.wcc.write('ppi-graph', { writeProperty: 'componentId' }) YIELD nodePropertiesWritten, componentCount, componentDistribution;
```

이 질의는 각 단백질이 어느 컴포넌트에 속하는지를 `componentId` 속성에 기록하고, 몇 개의 컴포넌트가 있는지와 그 분포를 돌려줍니다. 질의 결과는 표 4.1에 나와 있습니다. 백분위수(percentile)는 데이터가 특정 비율만큼 그 값 아래에 놓이는 지점을 뜻합니다. 예를 들어 `p99: 21521` 은 연결된 컴포넌트의 99%가 21,521개보다 적은 단백질을 가진다는 의미입니다.

표 4.1 Listing 4.9 질의의 요약 결과
<table><tr><td rowspan=1 colspan=1>nodePropertiesWritten</td><td rowspan=1 colspan=4>componentCount</td><td rowspan=1 colspan=1>componentDistribution</td></tr><tr><td rowspan=8 colspan=1>21559</td><td rowspan=1 colspan=4>27</td><td rowspan=1 colspan=1>y</td></tr><tr><td rowspan=2 colspan=4></td><td rowspan=1 colspan=1>&quot;p99&quot;: 21521,</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>&quot;min&quot;: 1,</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=4 colspan=3></td><td rowspan=1 colspan=1>&quot;max&quot;: 21521,&quot;mean&quot;: 798.481,</td></tr><tr><td></td><td rowspan=3 colspan=1>&quot;p90&quot;: 3,&quot;p50&quot;: 1,&quot;p999&quot;: 21521,&quot;p95&quot;: 4,</td></tr><tr><td rowspan=1 colspan=2></td></tr><tr><td rowspan=2 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>&quot;p75&quot;: 2}</td></tr></table>

알고리즘의 출력은 PPI 네트워크의 단백질들이 매우 촘촘하게 연결돼 있음을 보여 줍니다. 그래프의 단백질 21,559개는 서로 겹치지 않는 27개의 부분 그래프로 묶입니다. 그중 하나는 21,521개 단백질을 담은 아주 큰 덩어리이고, 나머지 부분 그래프들은 단일 노드이거나 최대 4개짜리 작은 섬(island)들입니다. 다시 말해, 거의 모든 단백질이 하나의 거대한 연결 덩어리에 속해 있습니다.

WCC는 그저 연결돼 있다는 이유만으로 단백질을 같은 그룹에 넣습니다. 그런데 이 "커뮤니티" 바깥의 다른 단백질보다 서로 훨씬 더 촘촘하게 연결된 단백질 무리가 있을 수 있습니다. 이런 무리를 찾으려면 또 다른 그래프 군집화 방법인 **Louvain 모듈성(Louvain modularity) 알고리즘** [15]을 쓸 수 있습니다. 이것은 가장 빠른 모듈성 기반 알고리즘 중 하나로, 큰 그래프에서도 잘 작동합니다. 서로 다른 규모에서 커뮤니티의 계층 구조를 드러내 주어, 네트워크가 전체적으로 어떻게 작동하는지를 이해하는 데 유용합니다. 이 알고리즘은 각 커뮤니티의 **모듈성 점수(modularity score)** 를 최대화하는 방식으로 작동합니다. 모듈성 점수란 노드들이 커뮤니티로 얼마나 잘 나뉘었는지를 나타내는 값으로, 노드들이 실제로 얼마나 촘촘하게 연결됐는지를 무작위 네트워크에서 예상되는 연결 정도와 비교해 측정합니다.

다음 질의는 앞서 만든 것과 같은 메모리 그래프 위에서 Louvain의 GDS 구현을 실행합니다. 데이터베이스를 재시작했거나 앞에서 실행하지 않았다면, 이 질의를 실행하기 전에 반드시 Listing 4.8을 먼저 실행해야 합니다.

```javascript
CALL gds.louvain.write('ppi-graph',
writeProperty: 'componentLouvainId' })
YIELD communityCount, modularity, modularities, communityDistribution
```

이 알고리즘의 결과는 표 4.2에 나와 있으며, 표 4.1과는 사뭇 다른 이야기를 들려줍니다.

표 4.2 Listing 4.10 질의의 요약 결과
<table><tr><td rowspan=1 colspan=1>communityCount</td><td rowspan=1 colspan=2>modularity</td><td rowspan=1 colspan=1>communityDistribution</td></tr><tr><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=2>0.5464241018027929</td><td rowspan=1 colspan=1>{</td></tr><tr><td rowspan=6 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>&quot;p99&quot;: 3533,</td></tr><tr><td rowspan=5 colspan=2></td><td rowspan=1 colspan=1>&quot;min&quot;: 1,</td></tr><tr><td rowspan=3 colspan=1>&quot;max&quot;: 3533,&quot;mean&quot;:449.1458333333333,&quot;p90&quot;: 1817,&quot;p50&quot;: 3,&quot;p999&quot;: 3533,&quot;p95&quot;: 2336,</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>&quot;p75&quot;: 311}</td></tr></table>

이번에는 3,500개가 넘는 단백질을 가진 큰 커뮤니티도 있고, 더 작은 커뮤니티도 있습니다. 평균 크기는 커뮤니티당 약 450개 단백질입니다. 모든 커뮤니티를 고려한 모듈성 점수는 약 0.40(40%)입니다. 다음 질의로 커뮤니티의 내용을 들여다볼 수 있습니다.

```cypher
Listing 4.11 Inspecting the top 10 communities
MATCH (p:PPIProtein)
WITH p.componentLouvainId as communityId, count(p) as members
ORDER BY members desc
LIMIT 10
MATCH (p:PPIProtein)-[:INTERACTS_WITH]-(o)
WHERE p.componentLouvainId = communityId
WITH communityId, members, p.name as name, count(o) as connections
ORDER BY connections DESC
RETURN communityId, members, collect(name)[..20] as keyMembers
```

이 질의는 Louvain이 식별한 각 커뮤니티에서 가장 많이 연결된 상위 20개 요소를 보여 줍니다. 큰 클러스터에는 APP, NTRK1, GRB2, EGFR, HSP90AA1 같은 단백질이 포함되고, 또 다른 클러스터에는 ELAVL1, MOV10, NXF1, VCP, SHMT2가 들어 있습니다. (전문가가 아니므로) 검색을 조금 해 보면, 이런 그룹들이 실제로 말이 된다는 것을 알 수 있습니다.

이런 알고리즘들은 쓰기 쉽지만 일반적입니다. 즉, 모든 노드와 관계를 똑같이 취급합니다. 다음에서는 우리 도메인과 목표에 맞게 맥락화된, 그리고 각 노드와 관계의 의미를 활용하는 몇 가지 기법을 소개합니다. 책의 코드에는 이런 도구들의 라이브러리가 포함돼 있습니다.

---

### 4.2.3 PPI와 질병 지식 그래프의 도메인 특화 분석 — 질병 경로를 자로 재다

앞선 알고리즘들은 PPI 네트워크가 어떻게 구성돼 있는지를 분석했습니다. 이제는 부분 네트워크, 즉 질병 경로를 살펴보겠습니다. 이를 그래프 이론 용어로 옮기면, PPI 네트워크는 $G = \left( V , E \right)$ 로 주어집니다. 여기서 노드 V는 단백질을, 엣지 E는 단백질–단백질 상호작용을 나타냅니다. 질병 d에 대한 질병 경로는 PPI 네트워크의 무방향 부분 그래프 $H_{\mathrm{d} } = \left( V_{\mathrm{d} } , E_{\mathrm{d} } \right)$ 로, d와 연관된 단백질 집합 $V_{\mathrm{d} }$ 와 그들 사이의 단백질–단백질 상호작용 집합으로 정의됩니다.

$$
E_{d} = \{ ( u , v ) | ( u , v ) \in E a n d u , v \in V_{d} \}
$$

이 수식을 풀어 읽으면 이렇습니다. 질병 경로의 엣지 집합 $E_d$ 는, 원래 PPI 네트워크의 엣지 $(u, v)$ 중에서 양쪽 끝 단백질 $u$ 와 $v$ 가 모두 질병 d와 연관된 단백질 집합 $V_d$ 안에 들어 있는 것들만 모은 것입니다. 즉, 질병 단백질끼리 이어진 상호작용만 골라낸 셈입니다.

우리 그래프를 이용하면 간단한 질의로 이 부분 네트워크를 뽑아낼 수 있습니다.

```perl
MATCH (d:Disease {id:$id})-[:ASSOCIATED_WITH]->(p)
WITH collect(p) as proteins
UNWIND proteins as m0
UNWIND proteins as m1
OPTIONAL MATCH (m0)-[r:INTERACTS_WITH]->(m1)
RETURN DISTINCT m0, r, m1
```

이것은 단일 분할(monopartite) 부분 네트워크입니다. 즉, 단백질과 질병 사이의 연결은 포함하지 않고 단백질끼리의 연결만 담습니다. 서로 다른 질병과 관련된 질병 경로는 겹칠 수도 있습니다.

어떤 측정값을 계산할 때는 이 부분 네트워크가 PPI 네트워크의 나머지 부분과 어떻게 연결돼 있는지도 고려해야 합니다. 이를 위해 **경로 경계(pathway boundary)** $B_d$ 를 계산할 수 있습니다.

$$
B_{d} = \{ ( u , v ) | ( u , v ) \in E { \mathrm { ~ a n d ~ } } u \in V_{d} { \mathrm { ~ a n d ~ } } v \in V_{d} \backslash V \}
$$

이 수식에서 $V \backslash V_d$ 는 전체 노드 집합 V에는 있지만 $V_d$ 에는 없는 모든 노드, 즉 대상 질병과 연관되지 않은 모든 노드를 뜻합니다. 정리하면 경로 경계 $B_d$ 는 한쪽 끝($u$)은 질병 단백질이고 다른 쪽 끝($v$)은 질병 단백질이 아닌, 안과 밖을 잇는 엣지들의 집합입니다.

우리가 살펴볼 측정값들은 질병 단백질의 연결성을 특징짓습니다. 질병 경로 안쪽에서의 연결성과, 바깥쪽으로 PPI 네트워크의 다른 단백질을 향한 연결성 둘 다입니다. 다른 지표들은 경로 안에서의 거리와 밀집도를 고려합니다. 모든 측정값은 질병마다 계산해, 각 질병을 특징짓고 의미 있는 패턴을 찾는 데 씁니다. 또한 이 값들을 한데 모으면 그 통계 정보가 네트워크 전체와, 그것이 여러 질병에 걸쳐 어떻게 분포하는지에 대한 더 넓은 조망을 줍니다.

#### 가장 큰 경로 컴포넌트 (Largest pathway component)

첫 번째 측정값은 **가장 큰 경로 컴포넌트의 상대적 크기(relative size of the largest pathway component)** 입니다. 이 값은 $H_d$ 의 가장 큰 연결 컴포넌트에 놓인 질병 단백질의 비율을 계산합니다.

$$
{ \mathrm{relativeLargestCC} } ( d ) = { \frac { \left| { \mathrm{nodes} } ( \operatorname { l a r g e s t C C } ( H_{d} ) ) \right| } { \left| V_{d} \right| } }
$$

여기서 $\mathrm{nodes}(\mathrm{largestCC}(H_d))$ 는 $H_d$ 에서 가장 큰 WCC(약하게 연결된 컴포넌트)에 속한 노드들을 돌려줍니다. 분자는 그 가장 큰 컴포넌트의 노드 수, 분모 $|V_d|$ 는 질병 단백질의 총수입니다. 코드는 다음 Listing에서 networkx 함수를 사용합니다. (전체 코드는 책의 코드 저장소 `chapter/ch04/analysis/multiomic_analysis.py` 에 있습니다.)

```text
Listing 4.13 Finding the size of the largest pathway component
Python class created to analyze the PPI
network. It extends a base class that
contains the main functions for analyzing
command-line arguments and handling
class MultiOmicAnalysis(GraphDBBase):  ← connections with the Neo4j database.
def __init__(self, argv, database):
super().__init__(command=__file__, argv=argv)
self.__database = database  Loads the disease pathway,
which is a subgraph of
def load_hd(self, disease):  ← the PPI network
query = """
MATCH (d:Disease {id:$id})-[:ASSOCIATED_WITH]->(p)
WITH collect(p) as proteins
UNWIND proteins as m0  Gets a query
UNWIND proteins as m1  representing a
OPTIONAL MATCH (m0)-[r:INTERACTS_WITH]->(m1)  subgraph and
return distinct m0, r, m1  loads it as a
"""  networkx graph
param = {"id": disease}
return self.load_graph_and_get_nx_graph(query, param)  Converts
                                                        the query
def load_graph_and_get_nx_graph(self, query, param={}):  result into
data = self.get_raw_data(query, param)  a networkx
G = networkx_utility.graph_undirected_from_cypher(data)  graph
return G
                        Returns a list of nodes and
def get_raw_data(self, query, param):  ← relationships for further processing
with self._driver.session(database = self.__database) as session:
results = session.run(query, param)
return results.graph()
```

이 코드에서 `MultiOmicAnalysis` 클래스는 PPI 네트워크를 분석하기 위해 만든 파이썬 클래스로, 명령행 인자 처리와 Neo4j 연결을 담당하는 기본 클래스를 상속합니다. `load_hd` 메서드는 질병 경로(PPI 네트워크의 부분 그래프)를 불러오고, `load_graph_and_get_nx_graph` 는 질의 결과를 networkx 그래프로 변환하며, `get_raw_data` 는 이후 처리에 쓸 노드·관계 목록을 돌려줍니다.

```python
def compute_largest_components(self, networkx_graph): <
largest_cc = max(nx.connected_components(networkx_graph), key=len)
return largest_cc
Computes all connected components
and returns the largest one
if name == _main__':
analysis = MultiOmicAnalysis(argv=sys.argv[1:], database="ppi")
disease_id = 'celiac disease' < A disease to analyze (the full
networkx_graph = analysis.load_Hd(disease_id) code analyzes all diseases)
nodes_count = networkx_graph.nodes.__len __()
largest_cc = analysis.compute_largest_components(networkx_graph)
relative_size_of_largest_cc =
float(largest_cc_size.__len__())/nodes_count < Computes the relative size of the
largest component, normalized
over the total number of nodes
```

여기서 `compute_largest_components` 는 모든 연결 컴포넌트를 구한 뒤 가장 큰 것을 돌려줍니다. 예제에서는 셀리악병을 분석 대상으로 삼지만(전체 코드는 모든 질병을 분석합니다), 마지막 줄은 가장 큰 컴포넌트의 크기를 전체 노드 수로 나눠 상대적 크기를 계산합니다.

#### 밀도 (Density)

두 번째로 계산할 지표는 경로의 **밀도(density)** 입니다. 이름 그대로, 질병 경로 안에서 단백질들이 얼마나 촘촘하게 연결돼 있는지를 측정합니다.

$$
\mathrm{density} ( d ) = \frac { 2 | E_{d} | } { | V_{d} | ( | V_{d} | - 1 ) }
$$

분모는 가능한 엣지의 최대 개수를 계산하고, 분자는 실제 엣지 수를 셉니다($E_d$ 를 두 번 세는 셈이라 2를 곱합니다). 결과값은 [0, 1] 범위에 놓입니다. 밀도가 높을수록 $H_d$ 안의 노드들 사이에 가능한 모든 엣지 중 더 많은 비율이 실제로 나타난다는 뜻입니다.

```python
def compute_density(networkx_graph): 4 Computes the density for a disease
nodes_count = networkx_graph.nodes.__len__() represented by the related subgraph
edges_count = networkx_graph.edges.__len__() extracted from the full PPI network
density_pathway =
2.0 * float(edges_count) / (nodes_count * (nodes_count - 1))
if name == _main ':
analysis = MultiOmicAnalysis(argv=sys.argv[1:], database="ppi")
disease_id = 'celiac disease'
networkx_graph = analysis.load_hd(disease_id)
density_pathway = compute_density(networkx_graph)
```

이 함수는 부분 그래프에서 노드 수와 엣지 수를 세어, 위 수식을 그대로 코드로 옮긴 것입니다. 즉 실제 엣지의 두 배를 (노드 수 × (노드 수 - 1))로 나눕니다.

#### 전도도 (Conductance)

세 번째 지표는 **전도도(conductance)** [16]로, 질병 경로(부분 그래프)가 그래프의 나머지 부분으로부터 얼마나 독립적인지를 나타냅니다. 부분 그래프 안의 노드와 바깥의 노드를 잇는 엣지를 (방향과 무관하게) 사용합니다.

$$
{ \mathrm{conductance} } \left( d \right) = { \frac { | B_{d} | } { \left( | B_{d} | + 2 | E_{d} | \right) } }
$$

여기서 분자 $|B_d|$ 는 앞서 본 경로 경계, 즉 안과 밖을 잇는 엣지 수이고, 분모는 그 경계 엣지 수에 안쪽 엣지의 두 배를 더한 값입니다. 결과값은 [0, 1] 범위에 놓입니다. 전도도가 낮을수록 그 경로는 네트워크의 나머지와 분리된, 더 긴밀하게 뭉친 커뮤니티라는 뜻입니다.

```text
Listing 4.15 Computing conductance
def compute_bd(self, disease):  ← Computes Bd using
query = """  a Cypher query
MATCH (d:Disease {id:$id})-[:ASSOCIATED_WITH]->(p)
WITH collect(p) as proteins
MATCH (m0)-[r:INTERACTS_WITH]-(m1)
WHERE m0 in proteins and not m1 in proteins
RETURN count(DISTINCT r) as bd
"""  Returns a pandas dataframe with
param = {'id': disease}  columns representing the values
return self.get_data(query, param)["bd"][0]  returned; useful when a Cypher
                                              query returns values instead of
def get_data(self, query, param={}):  ← nodes and relationships
with self._driver.session(database=self.__database) as session:
results = session.run(query, param)
data = pd.DataFrame(results.values(), columns=results.keys())
return data
if __name == '_main_':
analysis = MultiOmicAnalysis(argv=sys.argv[1:], database="ppi")
disease_id = 'celiac disease'
networkx_graph = analysis.load_hd(disease_id)
bd = analysis.compute_bd(disease_id)
edges_count = networkx_graph.edges.__len__()  Computes
conductance = float(bd) / (bd + 2 * edges_count)  ← conductance
```

`compute_bd` 는 Cypher 질의로 경계 엣지 수 $B_d$ 를 계산합니다. 질의는 한쪽 끝 단백질은 질병 경로 안에 있고(`m0 in proteins`) 다른 쪽은 바깥에 있는(`not m1 in proteins`) 상호작용 관계를 세어 줍니다. `get_data` 는 질의가 노드·관계 대신 값을 돌려줄 때 유용하게, 결과를 pandas 데이터프레임으로 감싸 돌려줍니다. 마지막 줄은 앞의 전도도 수식을 그대로 코드로 옮긴 것입니다.

#### 질병 경로와 클러스터 분석하기 (Analyzing disease pathways and clusters)

이제 각 부분 그래프(질병 경로)에 대한 지표를 다 계산했으니, 이들을 전체로 놓고 살펴보겠습니다. 흔히 쓰는 방법은 결과를 여러 구간(bucket)으로 나눠 도수 분포(frequency analysis)를 그리는 것으로, 그림 4.7과 같습니다.

(a) 가장 큰 연결 컴포넌트 (Largest CC)

![](images/ko/figure-4-7a-ko.png)

(b) 밀도 (Density)

![](images/ko/figure-4-7b-ko.png)

(c) 전도도 (Conductance)

![](images/ko/figure-4-7c-ko.png)

그림 4.7 질병 경로에 대한 세 가지 핵심 측정값의 분포

이 분포에서 우리는 질병 경로가 PPI 네트워크 안에서 조각조각 나뉘어 있음을 볼 수 있습니다. 질병당 연결 컴포넌트 수의 중앙값은 16개이고, 가장 큰 경로 컴포넌트에 든 단백질은 중앙값 기준 겨우 21%에 불과합니다(그림 4.7a). 경로 중 약 10%만이 단백질의 60% 이상을 가장 큰 경로 컴포넌트에 담고 있습니다. 질병 경로는 내부적으로 잘 연결돼 있지 않은데, 밀도 중앙값이 겨우 0.07이고(전체 PPI 네트워크 밀도는 0.0015입니다), 질병의 90%가 밀도 0.17 아래입니다(그림 4.7b). 반면 질병 경로는 외부적으로는 잘 연결돼 있어, 전도도 중앙값이 0.96에 이릅니다(그림 4.7c). 요약하면, 한 질병에 관여하는 단백질들은 서로 뭉쳐 있기보다 네트워크 곳곳에 흩어져 다른 단백질들과 뒤섞여 있습니다.

질병 경로로 얻은 이런 겹치는 부분 그래프들은, WCC나 Louvain으로 군집화해 얻은 것과는 매우 다릅니다. 이 가설을 확인하기 위해, Louvain으로 계산한 클러스터에 같은 측정값을 실행해 봅시다. 결과는 그림 4.8에 있습니다.

(a) 가장 큰 연결 컴포넌트 (Largest CC)

![](images/ko/figure-4-8a-ko.png)

(b) 밀도 (Density)

![](images/ko/figure-4-8b-ko.png)

(c) 전도도 (Conductance)

![](images/ko/figure-4-8c-ko.png)

그림 4.8 Louvain 알고리즘으로 얻은 클러스터에 대한 세 가지 핵심 측정값의 분포

이 클러스터들은 다른 이야기를 들려줍니다. 예상대로 대부분의 단백질이 하나의 크게 연결된 컴포넌트 안에 자리합니다(그림 4.8a). 밀도는 네트워크의 전반적 연결성과 관련되므로, 변화는 클러스터의 서로 다른 구조에서만 비롯됩니다(그림 4.8b). 전도도는 개선됐습니다. 즉 이 클러스터들은 외부보다 내부적으로 더 잘 연결돼 있습니다(그림 4.8c). 한마디로, Louvain 클러스터는 "안이 촘촘하고 밖과는 성긴" 진짜 커뮤니티인 반면, 질병 경로는 그렇지 않다는 대비가 뚜렷합니다.

다음 절에서는 두 번째 유형의 응용(제약)과, 지식 그래프에서 정보를 뽑아내는 새로운 알고리즘을 소개합니다. 이번에는 단일 노드뿐 아니라 엣지와 경로(path)에도 초점을 맞춥니다.

---

## 4.3 지식 그래프의 제약 응용 — 약물 재활용의 지름길

새로운 치료제 하나를 개발하는 비용은 약 14억 달러로 추정됩니다 [17]. 이 과정은 첫 화합물에서 시장 출시까지 보통 15년이 걸리고 [18], 성공 확률은 놀랍도록 낮습니다 [19]. 요컨대 신약 개발은 돈도 시간도 실패 위험도 어마어마한 도박입니다.

**약물 분석과 재활용(drug repurposing)** 은 승인까지의 기간, 실패율, 비용을 극적으로 줄일 수 있습니다. 이런 분석은 이미 승인된 약물에 대한 기존 정보 — 독성 프로파일링, 전임상 모델, 임상시험, 출시 후 감시 — 를 활용합니다. 지식 그래프가 약물 상호작용을 예측하거나 [20], 약물이 상호작용할 수 있는 분자 표적을 식별하거나 [21], 기존 약물로 치료할 수 있는 새로운 질병을 찾아내는 데 [22] 쓰인 사례가 많습니다.

Dai 등 [21]은 추천 시스템, 특히 협업 필터링(collaborative filtering)을 이용해 약물–질병 연관성을 추론했습니다. 다른 연구자들은 이런 기법으로 약물–표적 상호작용 [23, 24]과 약물–질병 치료 [25, 26]를 추론했습니다. 성공 사례가 보고됐음에도, 이 접근들은 그래프에 담긴 약물과 질병에만 한정된다는 한계가 있습니다. 화학 구조, 생물학적 과정, 그 밖의 관련 지식을 함께 표현해 지식 그래프를 풍부하게 만들면, 연구자들이 아직 세상에 없던 새로운 화합물에 대해서도 예측할 수 있게 될 것입니다.

Himmelstein 등 [2]은 29개의 공개 자원에서 지식을 인코딩한 그래프를 구축해, 화합물·질병·유전자·해부 구조·경로·생물학적 과정·분자 기능·세포 구성 요소·약리학적 분류·부작용·증상을 연결했습니다. 그들은 이 그래프를 **Hetionet** 이라 불렀습니다("hetnet", 즉 heterogeneous network(이질적 네트워크)의 줄임말에서 온 이름입니다). 이 그래프 데이터베이스는 Neo4j 형식으로 공개돼 있습니다(https://het.io/). 그래서 이 예제에서는 여러 소스로부터 데이터베이스를 직접 만들 필요가 없습니다. 연구자들이 그 일을 대신해 주었기 때문입니다. 대신 우리는 일관된 스키마를 갖춘 잘 설계된 지식 그래프의 중요성과, 지식 그래프를 분석해 그 정보의 완결성을 평가하는 방법을 논의하겠습니다.

우리는 이 데이터베이스의 Neo4j 5.x 백업을 만들어 두었고, https://mng.bz/648e 에서 내려받을 수 있습니다. 다음 Listing은 이를 가져오는 방법을 보여 줍니다.

```text
Listing 4.16 Creating the Het.io database
Add the following line to the neo4j.conf file
dbms.databases.seed_from_uri_providers=URLConnectionSeedProvider
then run the following command
CREATE DATABASE hetionet OPTIONS { existingData: "use",
➥seedUri: "https://mng.bz/648e"}
```

가져온 지식 그래프는 11가지 유형의 노드 47,031개와 24가지 유형의 관계 2,250,197개로 이루어져 있습니다. 노드에는 저분자 화합물 1,552개, 복합 질병 137개를 비롯해 유전자·해부 구조·경로·생물학적 과정·분자 기능·세포 구성 요소·섭동(perturbation)·약리학적 분류·약물 부작용·질병 증상이 포함됩니다. 엣지는 이 노드들 사이의 관계를 나타내며, 지난 반세기 동안 수백만 건의 연구가 축적한 집단 지식을 담고 있습니다 [2]. 그림 4.9는 데이터셋의 전체 스키마를 보여 줍니다.

예를 들어 Compound–binds–Gene(화합물–결합–유전자) 엣지는 어떤 화합물이 유전자가 암호화하는 단백질에 결합한다는 것을 나타냅니다. Hetionet에는 이런 엣지가 11,571개 있고, 각각에 대해 참고 문헌이 관계 속성으로 저장돼 있습니다.

#### 연습 문제

가져온 그래프를 탐색하고, 노드들이 서로 다른 노드 유형에 어떻게 분포하는지 살펴보세요. 관계에 대해서도 같은 작업을 해 보세요. 다만 관계는 더 복잡할 수 있으니 조심해야 합니다. 정확한 질의를 짜려면 스키마를 참고하세요.

![](images/ko/figure-4-9-ko.png)

그림 4.9 Het.io 지식 그래프 스키마. 노드와 관계의 세부 사항은 https://mng.bz/EwgD 에서 확인할 수 있습니다.

### 메타경로와 차수 가중 경로 수 (Metapaths and degree-weighted path count, DWPC)

이어지는 경로 탐색 예제들은 관련성에 따라 순서를 매기기 위해 새로운 지표를 사용합니다. 바로 **차수 가중 경로 수(Degree-Weighted Path Count, DWPC)** 입니다. 이 지표는 Himmelstein [27]이 도입했는데, 원래 소셜 네트워크 분석을 위해 개발된 기존 방법 **PathPredict** [28]을 응용한 것입니다. DWPC는 Hetionet 안에서 메타경로가 얼마나 흔하게 나타나는지를 정량화합니다.

다음 그림의 (a) 패널이 보여 주듯, 스키마는 실제 노드와 실제 관계로 표현됩니다. 반면 **메타경로(metapath)** 는 첫 번째 유형의 노드와 마지막 유형의 노드 사이에 있을 법한 실제 경로를 묘사하는, 노드와 관계의 클래스(유형) 시퀀스를 나타냅니다. 우리는 스키마를 "질의"하듯 뒤져, 출발 유형과 도착 유형 사이의 연결 패턴을 찾을 수 있습니다. 예를 들어 (b) 패널처럼 (Gene)—a—(Disease)라는 일반 패턴에 대해 최대 길이 4까지의 가능한 메타경로 목록을 만들 수 있습니다.

![](images/ko/figure-4-10a-ko.png)

Hetionet의 메타그래프(a)와 메타경로(b) 발췌. **메타그래프(metagraph)** 는 데이터베이스의 구조를 묘사하며, 노드의 유형과 관계의 유형을 나타냅니다. 메타경로는 경로를 묘사하며, 노드와 관계의 유형을 나타냅니다.

다시 강조하지만, 이것들은 경로에 대한 "묘사"이지 경로의 실제 인스턴스가 아닙니다. 마치 "역—지하철—역"이라는 일반적 이동 패턴을 말하는 것이지, 특정한 실제 노선을 가리키는 것이 아닌 것과 같습니다.

이제 예제 지식 그래프를 탐색했으니 지표를 계산해 볼 수 있습니다. 메타경로 기반 지표 중 가장 단순한 것은 **경로 수(Path Count, PC)** 입니다. 이는 정의된 출발 노드와 목표 노드 사이에서, 지정한 메타경로에 해당하는 경로의 개수입니다. PC는 경로를 따라 그래프가 얼마나 연결돼 있는지는 보정하지 않습니다. 즉 각 경로의 값은 1입니다. 예를 들어 그림 4.10a는 특정 유전자 IRF1이 특정 질병 다발성 경화증(multiple sclerosis)과 관계 맺는 지식 그래프의 일부를 보여 줍니다. 모든 경로는 일반 패턴 (Gene)—a—(Disease)에 대한 잠재적 메타경로 중 하나에 속합니다. 그림 4.10b에서는 각 메타경로와 관련된 경로들이 묶여 있습니다. 첫 번째 그룹은 Tissue(조직) 유형의 중간 노드를 가지며 경로가 하나뿐이라 PC가 1입니다. 두 번째 그룹은 또 다른 유전자를 중간 노드로 가지며, 이 그룹에는 경로가 세 개라 PC가 3입니다.

(a) 가상 그래프 (Hypothetical graph)
(b) 경로 수 계산과 가중 (Calculating and weighting path counts)

![](images/ko/figure-4-10b-ko.png)

그림 4.10 (a) 정의된 메타경로에 기반해 경로를 추출하는 모습, 그리고 (b) 경로-차수 곱(PDP)과 DWPC를 계산하는 모습

반면 DWPC는 각 경로에 개별 값을 부여하는데, 이 값을 **경로-차수 곱(Path-Degree Product, PDP)** 이라 부릅니다. 다음 공식으로 계산합니다.

$$
\mathrm{PDP} \left( \mathrm{path} \right) = \prod_{d \in { \cal D} _ { \mathrm{path} } } d^{- w}
$$

여기서 $D_{\mathrm{path}}$ 는 경로를 따라 나타나는 차수(degree)들의 집합이고, $w$ 는 감쇠 지수(damping exponent)입니다. 이 곱을 다음 순서로 계산합니다.

1. 경로를 따라 각 메타엣지별 차수를 모두 추출합니다($D_{\mathrm{path}}$). 이때 경로의 각 엣지는 두 개의 차수를 기여합니다. 그림 4.10에서 IRF1과 IL2RA 사이의 엣지는 차수 값 4와 1을 가집니다. IRF1은 INTERACTS(상호작용) 유형의 나가는 엣지가 4개이고, IL2RA는 같은 유형의 들어오는 엣지가 1개이기 때문입니다. IRF1과 CXCR4 사이의 엣지는 차수 값 4와 2를 가지는데, CXCR4가 INTERACTS 유형의 들어오는 엣지를 2개 가지기 때문입니다.
2. 각 차수를 $-w$ 승으로 거듭제곱합니다. 여기서 $w \geq 0$ 이며 이를 **감쇠 지수(damping exponent)** 라 부릅니다. 이 값이 클수록 연결이 많은(차수가 높은) 노드를 지나는 경로의 가치를 더 깎아내립니다.
3. 거듭제곱한 차수들을 모두 곱해 PDP를 얻습니다.

그림 4.10의 경로 $( \mathrm{IRF1} ) { - } [ ] { - } ( \mathrm{CXCR4} ) { - } [ ] { - }$ (다발성 경화증)를 예로 들어 보겠습니다. $w = 0.5$ 라고 하면 계산은 다음과 같습니다.

$$
4^{- 0 . 5} * 2^{- 0 . 5} * 1^{- 0 . 5} * 4^{- 0 . 5} = 0 . 1 6 7 \cong 0 . 1 7 7
$$

그림 4.10은 다른 경로들의 값도 보여 줍니다. DWPC는 특정 메타경로에 대한 PDP들의 합과 같습니다.

$$
\mathrm{DWPC}_{m} \left( s , t \right) = \sum_{\mathrm{path} \in \mathrm{Paths}_{m} \left( s , t \right) } \mathrm { P D P ( p a t h ) }
$$

이 수식은 출발 노드 $s$ 와 목표 노드 $t$ 사이의, 메타경로 $m$ 에 해당하는 모든 경로의 PDP를 더한 것이 그 메타경로의 DWPC라는 뜻입니다. 이런 지표들은 경로가 얼마나 흔한지를 평가하면서도, 지식 그래프 분석에서 흔히 문제가 되는 "너무 유명한 노드(well-known nodes)" 를 자동으로 낮게 취급합니다. 즉 아무하고나 연결된 인기 많은 노드를 지나는 경로에는 낮은 점수를 주어, 진짜 특이하고 의미 있는 연결을 드러내 줍니다.

---

### 4.3.1 Hetionet 지식 그래프의 심층 분석 — 셀리악병을 파고들다

이 단계에서 우리는 Hetionet 그래프를 심층 분석할 재료를 모두 갖췄습니다. Cypher 질의와 DWPC 지표를 이용해, 질병과 연관된 유전자 집합에서 두드러진 **유전자 온톨로지(Gene Ontology, GO) 과정** 을 식별하겠습니다.

> **참고** 다음 분석은 Daniel Himmelstein이 시작한 Thinklab 프로젝트(https://think-lab.github.io/d/220/)에서 영감을 받았습니다.

다시 셀리악병(celiac disease, CD)을 예로 삼겠습니다. 셀리악병은 흔하고(유병률 1:100), 만성적이며, 면역 매개성 장질환으로, 유전적으로 취약한 개인에게서 글루텐 불내성으로 발생합니다. 다음 질의로 셀리악병과 연관된 48개 유전자를 볼 수 있습니다. `MATCH p = (:Disease {name: 'celiac disease'})-[rel: ASSOCIATES_DaG]-() RETURN p`.

다음으로는 셀리악병과, 셀리악 관련 유전자가 최소 두 개 이상 참여하는 각 GO 과정 사이의 DWPC를 계산하겠습니다. 결과는 참여 유전자가 최소 다섯 개 이상인 과정으로 더 좁힙니다. Cypher 질의는 다음과 같습니다.

```text
Listing 4.17 GO process enrichment for CD
MATCH path = (n0:Disease)-[:ASSOCIATES_DaG]-(n1)-[:PARTICIPATES_GpBP]-
➥(n2:BiologicalProcess)  ← Searches for
WHERE n0.name = 'celiac disease'  ← Restricts to 'celiac  DaG-GpBP paths
WITH  disease' as source
[
size([(n0)-[:ASSOCIATES_DaG]-() | n0]),
size([()-[:ASSOCIATES_DaG]-(n1) | n1]),
size([(n1)-[:PARTICIPATES_GpBP]-() | n1]),
size([()-[:PARTICIPATES_GpBP]-(n2) | n2])  Computes relationship-related
] ← degrees necessary for DWPC
AS degrees, path, n2
WITH
n2.identifier AS go_id,  Returns the GO
n2.name AS go_name,  process ID and name  Counts
count(path) AS PC,  ← paths  Computes
sum(reduce(pdp = 1.0, d in degrees| pdp * d ^ -0.4)) AS DWPC,  ← DWPC
size([(n2)-[:PARTICIPATES_GpBP]-() | n2]) AS n_genes  ← Counts the genes
WHERE n_genes >= 5 AND PC >= 2  in the GO process
RETURN
go_id, go_name, PC, DWPC, n_genes  Filters out GO processes with fewer
ORDER BY DWPC DESC  than five generic genes involved and
LIMIT 10  fewer than two celiac-related genes
```

이 질의는 질병에서 유전자로(`ASSOCIATES_DaG`), 다시 유전자에서 생물학적 과정으로(`PARTICIPATES_GpBP`) 이어지는 경로를 찾습니다. `size([...])` 로 DWPC에 필요한 차수를 계산하고, `reduce`로 각 차수를 $-0.4$ 승 해 곱한 뒤(PDP), 이를 합해 DWPC를 구합니다. 마지막으로 GO 과정에 참여하는 유전자가 다섯 개 미만이거나 셀리악 관련 유전자가 두 개 미만인 과정을 걸러 냅니다. 실행했을 때 이 질의는 표 4.3에 나열된 상위 10개 GO 과정을 돌려주었습니다.

표 4.3 Listing 4.17 질의의 결과
<table><tr><td rowspan=1 colspan=2>GO ID</td><td rowspan=1 colspan=1>GO name</td><td rowspan=1 colspan=1>PC</td><td rowspan=1 colspan=1>DWPC</td><td rowspan=1 colspan=1># genes</td></tr><tr><td rowspan=3 colspan=2>GO:0031295GO:0031294GO:0002507</td><td rowspan=3 colspan=1>T cell costimulationlymphocyte costimulationtolerance induction</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.03347</td><td rowspan=1 colspan=1>75</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.03329</td><td rowspan=1 colspan=1>76</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.03276</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=2>GO:0050870</td><td rowspan=1 colspan=1>positive regulation of T cell activation</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.02925</td><td rowspan=1 colspan=1>201</td></tr><tr><td rowspan=1 colspan=2>GO:0034112</td><td rowspan=1 colspan=1>positive regulation of homotypic cell-cell adhesion</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.02902</td><td rowspan=1 colspan=1>205</td></tr><tr><td rowspan=1 colspan=2>GO:1903039</td><td rowspan=1 colspan=1>positive regulation of leukocyte cell-cell adhesion</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.02891</td><td rowspan=1 colspan=1>207</td></tr><tr><td rowspan=3 colspan=2>GO:0051249GO:0002684</td><td rowspan=1 colspan=1>regulation of lymphocyte activation</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>0.02763</td><td rowspan=3 colspan=1>381880</td></tr><tr><td rowspan=2 colspan=1>positive regulation of the immune system process</td><td rowspan=2 colspan=1>21</td><td rowspan=2 colspan=1>0.02718</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=2>GO:0022409GO:0050863</td><td rowspan=2 colspan=1>positive regulation of cell-cell adhesion</td><td rowspan=2 colspan=1>14</td><td rowspan=2 colspan=1>0.02716</td><td rowspan=2 colspan=1>242</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>regulation of T cell activation</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.02701</td><td rowspan=1 colspan=1>290</td></tr></table>

ID가 GO:0002684인 GO 과정은 PC 값이 높고 880개 유전자와 연결돼 있습니다. PC로 정렬하면 이 과정이 맨 앞에 놓일 것입니다. 그러나 이 과정은 셀리악병뿐 아니라 다른 많은 과정에도 관여합니다. DWPC로 정렬하면 이 과정은 결과 목록의 바닥 근처로 내려갑니다. 맨 위에 오른 것은 GO:0031295, "T 세포 공동자극(T cell costimulation)" 으로, 우리가 조사하는 질병과 매우 관련이 깊습니다. 바로 이것이 DWPC의 힘입니다. 아무하고나 연결된 흔한 과정을 눌러 주고, 이 질병에 특이적으로 얽힌 과정을 위로 끌어올립니다.

단백질 상호작용 관계까지 포함한 더 복잡한 경로를 고려해 검색을 정교하게 다듬어 보겠습니다. 다음 Listing의 질의는 **전장 유전체 연관 연구(Genome-Wide Association Study, GWAS)** [29, 30]에서 유래한 질병–유전자 연관성만 고려하는데, 이런 연관성은 기존 지식에 덜 편향돼 있습니다. 또한 셀리악병 인터랙톰의 이웃에 있는 유전자를 식별하기 위해 메타경로에 단백질 상호작용을 추가합니다. 끝으로, 셀리악병에 걸린 조직에서 상향 조절(upregulated)된 유전자만 고려합니다.

```text
Listing 4.18 Tissue-specific interactomics
MATCH path = (n0:Disease)-[e1:ASSOCIATES_DaG]-(n1)-[:INTERACTS_GiG]-(n2)-
➥[:PARTICIPATES_GpBP]-(n3:BiologicalProcess)
WHERE n0.name = 'celiac disease'
AND 'GWAS Catalog' in e1.sources
AND exists((n0)-[:LOCALIZES_DlA]-()-[:UPREGULATES_AuG]-(n2))
WITH
[
size([(n0)-[:ASSOCIATES_DaG]-() | n0]),
size([()-[:ASSOCIATES_DaG]-(n1) | n1]),
size([(n1)-[:INTERACTS_GiG]-() | n1]),
size([()-[:INTERACTS_GiG]-(n2) | n2]),
size([(n2)-[:PARTICIPATES_GpBP]-() | n2]),
size([()-[:PARTICIPATES_GpBP]-(n3) | n3])
] AS degrees, path, n3 as target
WITH
target.identifier AS go_id,
target.name AS go_name,
count(path) AS PC,
sum(reduce(pdp = 1.0, d in degrees| pdp * d ^ -0.4)) AS DWPC,
size([(target)-[:PARTICIPATES_GpBP]-() | target]) AS n_genes
WHERE 5 <= n_genes <= 100 AND PC >= 2
RETURN
go_id, go_name, PC, DWPC, n_genes
ORDER BY DWPC DESC
LIMIT 10
```

이 질의는 앞의 것보다 한 단계 더 깁니다. 질병–유전자–(유전자 상호작용)–유전자–생물학적 과정으로 이어지는 경로를 찾되, GWAS 카탈로그에서 온 연관성이고, 그 질병이 자리한 조직에서 상향 조절된 유전자로 이어지는 경로만 남깁니다. 질의 결과는 표 4.4에 나와 있습니다.

표 4.4 Listing 4.18 질의의 결과
<table><tr><td>GO ID</td><td>GO name</td><td>PC</td><td>DWPC</td><td># genes</td></tr><tr><td>GO:0031295</td><td>T cell costimulation</td><td>10</td><td>0.00665</td><td>75</td></tr><tr><td>GO:0031294</td><td>lymphocyte costimulation</td><td>10</td><td>0.00662</td><td>76</td></tr><tr><td>GO:0010560</td><td>positive regulation of glycoprotein biosynthetic process</td><td>6</td><td>0.00342</td><td>17</td></tr><tr><td>GO:0033689</td><td>negative regulation of osteoblast proliferation</td><td>4</td><td>0.00341</td><td>9</td></tr><tr><td>GO:1903020</td><td>positive regulation of glycoprotein metabolic process</td><td>6</td><td>0.00327</td><td>19</td></tr><tr><td>GO:0006573</td><td>valine metabolic process</td><td>5</td><td>0.00277</td><td>8</td></tr><tr><td>GO:0070884</td><td>regulation of calcineurin-NFAT signaling cascade</td><td>2</td><td>0.00277</td><td>19</td></tr><tr><td>GO:0010559</td><td>regulation of glycoprotein biosynthetic process</td><td>7</td><td>0.00272</td><td>35</td></tr><tr><td>GO:1903018</td><td>regulation of glycoprotein metabolic process</td><td>7</td><td>0.00257</td><td>40</td></tr><tr><td>GO:0070098</td><td>chemokine-mediated signaling pathway</td><td>9</td><td>0.00256</td><td>72</td></tr></table>

결과는 한층 더 구체적입니다. 상위 두 개는 그대로지만, 단백질 상호작용을 추가하자 당단백질(glycoprotein)과 관련된 과정이 강하게 두드러지는 등 질병 특이적인 다른 측면들이 나타났습니다. 만약 "positive regulation of glycoprotein biosynthetic process(당단백질 생합성 과정의 양성 조절)" 관계에 관심이 있다면, 그 DWPC 뒤에 놓인 실제 경로들을 다음처럼 뽑아낼 수 있습니다.

```text
Listing 4.19 Finding the paths behind a DWPC
MATCH path = (n0:Disease)-[e1:ASSOCIATES_DaG]-(n1)-[:INTERACTS_GiG]-(n2)-
➥[:PARTICIPATES_GpBP]-(n3:BiologicalProcess)
WHERE n0.name = 'celiac disease'
AND n3.name = 'positive regulation of glycoprotein biosynthetic process'
AND 'GWAS Catalog' in e1.sources
AND exists((n0)-[:LOCALIZES_DlA]-()-[:UPREGULATES_AuG]-(n2))
RETURN path
```

이 질의는 앞의 것과 뼈대는 같지만, 대상 생물학적 과정을 하나로 고정하고 집계 대신 실제 경로 자체(`RETURN path`)를 돌려줍니다. 결과는 그림 4.11에 있습니다.

#### 연습 문제

질의를 조금 바꾸면 다른 질병에 대해서도 같은 분석을 할 수 있습니다. 몇 가지를 실행해 보고 조사도 해 보면서, 이 지식 그래프가 얼마나 많은 지식을 담고 있는지 평가해 보세요. Thinklab에는 탐색해 볼 만한 흥미로운 질의가 많이 있습니다.

지금까지 봤듯이, 지식 그래프는 탐색하고 활용하기 쉬운 형태로 지식을 담아낼 수 있습니다. 우리의 분석은 저장된 정보에 대한 정량적 평가를 제공했습니다.

![](images/ko/figure-4-11-ko.png)

그림 4.11 Listing 4.19 질의의 결과로, 셀리악병과 "당단백질 생합성 과정의 양성 조절" 사이의 경로들을 보여 줍니다.

---

### 4.3.2 LLM으로 경로 분석 결과 해석하기 — 숫자를 이야기로

DWPC 기반 질의는 생물학적 과정을 정량적으로 순위 매겨 주지만, 이 결과를 임상에서 실제로 행동으로 옮길 수 있는 통찰로 바꾸려면 도메인 전문성과 맥락적 이해가 필요합니다. 여기서 LLM이 똑똑한 해석자 역할을 할 수 있습니다. 복잡한 경로 분석 결과를 일관된 생물학적 이야기와 임상 권고로 종합하도록 돕는 것입니다. 바로 여기서 달변가의 재능이 빛납니다. 사실 대장이 뽑아낸 딱딱한 순위표를, 사람이 읽고 판단할 수 있는 서사로 풀어 주기 때문입니다.

셀리악병 분석에서 얻은 GO 과정 강화(enrichment) 결과로 이를 시연해 보겠습니다. 질의는 "T 세포 공동자극", "관용 유도(tolerance induction)", "당단백질 생합성 과정의 양성 조절" 같은 과정을 저마다 다른 DWPC 점수와 함께 돌려주었습니다. LLM은 이 발견들을 더 넓은 생물학적 맥락에서 해석하도록 도와줄 수 있습니다.

```text
Listing 4.20 Example LLM analysis prompt
You are a biomedical research assistant analyzing gene ontology pathway enrichment results for celiac disease. Please interpret the following results from a knowledge graph analysis and provide insights for clinical researchers:

QUERY RESULTS:
- T cell costimulation (DWPC: 0.03347, PC: 10, Genes: 75)
- lymphocyte costimulation (DWPC: 0.03329, PC: 10, Genes: 76)
- positive regulation of T cell activation (DWPC: 0.02925, PC: 14, Genes: 201)
- positive regulation of glycoprotein biosynthetic process
➥(DWPC: 0.00342, PC: 6, Genes: 17)

CONTEXT: These pathways were identified through DWPC analysis of celiac disease-associated genes in a protein-protein interaction network. DWPC scores indicate pathway relevance while accounting for node degree bias.

ANALYSIS REQUEST:
1. Interpret the biological significance of these top-ranked pathways in celiac disease
2. Explain the relationship between these processes and celiac disease pathogenesis
3. Identify potential therapeutic implications
4. Highlight any unexpected findings that warrant further investigation
5. Suggest follow-up research questions based on these results
```

이 프롬프트는 LLM에게 생명의학 연구 조수 역할을 맡기고, 위 질의 결과를 제시한 뒤 다섯 가지를 요청합니다. 상위 순위 경로의 생물학적 의미 해석, 이 과정들과 셀리악병 발병 기전의 관계 설명, 잠재적 치료 함의 식별, 추가 조사가 필요한 뜻밖의 발견 강조, 그리고 후속 연구 질문 제안입니다.

LLM은 원시 분석 결과를 종합적인 해석으로 옮겨 줍니다. T 세포 공동자극이 셀리악병의 자가면역적 성질과 어떻게 연결되는지, 왜 관용 유도가 교란된 것으로 보이는지, 당단백질 조절이 글루텐 불내성 기전과 어떻게 이어지는지를 설명합니다. 이 해석은 결과를 실행 가능한 생물학적 통찰로 바꿔 줍니다. Claude.ai Sonnet 4.0(https://claude.ai/new)으로 얻은 결과의 첫 부분을 다음 사이드바에 실었습니다.

#### 상위 순위 경로의 생물학적 의미에 대한 Claude.ai의 부분 결과

**T 세포 및 림프구 공동자극 경로(DWPC: 0.03347, 0.03329).** 이 경로들은 셀리악병의 핵심 면역학적 기능 장애를 나타냅니다. 높은 DWPC 점수와 상당한 유전자 수(75\~76개 유전자)는 조절 장애를 일으킨 공동자극 신호전달이 질병 발병 기전의 중심에 있음을 시사합니다. 셀리악병에서 항원 제시 세포는 HLA-DQ2/DQ8 분자를 통해 글리아딘 펩타이드를 T 세포에 제시하는데, 비정상적인 공동자극은 관용 대신 부적절한 T 세포 활성화로 이어집니다.

**관용 유도(DWPC: 0.03276).** 유전자 수가 더 적음(12개)에도 이 경로는 놀랍도록 높은 DWPC 점수를 보이는데, 이는 셀리악병 네트워크 안에서 이 경로가 매우 긴밀하게 얽혀 있음을 시사합니다. 이것은 셀리악병 발병 기전의 특징인, 식이 항원에 대한 경구 관용(oral tolerance)의 근본적 붕괴를 반영하는 것으로 보입니다.

**T 세포 활성화의 양성 조절(DWPC: 0.02925).** 201개 유전자가 관여하는 이 경로는 가장 광범위한 면역학적 조절 장애를 나타냅니다. 지속적인 T 세포 활성화는 셀리악병의 특징인 만성 염증 반응을 몰고 가, 융모 위축(villous atrophy)과 임상 증상으로 이어집니다.

이 접근은 LLM이 어떻게 지식 그래프 분석을 보완하는지를 보여 줍니다. 즉 정량적 결과를 정성적 이해로 바꾸는 맥락적 해석을 제공함으로써, 계산 분석과 실제 응용 사이의 간극을 메워 줍니다.

---

## 4.4 지식 그래프의 임상 응용 — 정밀 의료를 향하여

임상 응용은 아직 지식 그래프 활용의 초기 단계에 있습니다. 이 경우 장기 목표는 지식 그래프를 활용·분석해 환자 진료를, 주로 **정밀 의료(precision medicine)** 를 통해 뒷받침하는 것입니다. 정밀 의료 이니셔티브는 한 사람의 유전, 환경, 생활 습관이 질병 예방·치료의 최선책을 정하는 데 어떻게 도움이 되는지를 이해하려는 장기 연구 사업입니다. 정밀 의료를 구현하려면 단백질체·유전체·전사체 같은 오믹스 데이터를, **전자 건강 기록(Electronic Health Record, EHR)** 같은 환자 데이터가 포함된 임상 의사결정 과정에 통합해야 합니다. 생명의학 데이터의 양과 다양성, 임상적으로 중요한 지식이 여러 생명의학 데이터베이스와 논문에 흩어져 있다는 점, 그리고 프라이버시 우려는 데이터 통합에 어려움을 안깁니다 [31].

EHR은 한 환자의 진료에 관여한 모든 임상의의 정보를 담도록 설계됩니다. 그런데 이것은 해석하기 까다로울 수 있고, 상당한 주관성이 개입하며, 임상의가 무관하다고 판단한 정보는 누락되거나 추적되지 않아 정보 결손으로 이어질 수 있습니다 [32]. 그래서 임상 응용에 쓰이는 지식 그래프는 EHR을 다중 오믹스 데이터셋, 여러 온톨로지, 그 밖의 데이터 소스와 병합합니다. 핵심 요소는 환자·약물·질병을 나타내는 노드이고, 엣지는 환자가 특정 약물로 치료받는다거나 어떤 질병으로 진단받았다는 등의 관계를 인코딩합니다.

EHR과 그래프를 사용하는 또 다른 예는 **환자 여정 지도(patient journey mapping)**, 다른 말로 환자 경험 지도(patient experience mapping)입니다 [33]. 이것은 사람들이 어떻게 의료 서비스에 들어오고, 겪고, 나가는지를 더 잘 이해하기 위한 빠르게 성장하는 접근입니다. 흔히 임상 경로(clinical pathway) [34]와 비교되는데, 임상 경로는 특정 질병으로 나타난 환자의 임상 상태에 대한 표준 진료를 규정하며, 짝을 이루는 환자 여정 지도와 자주 연결됩니다. 그림 4.12는 암과 같은 복잡한 질병에 대한 한 환자의 여정 범위를 그려 보여 줍니다.

![](images/6001d287b74726925a8a75d40d03a6a3bc4958b3c1b72aacb7c1b6c19842064b.jpg)

그림 4.12 한 환자의 임상 종양학 여정(David Hughes 제공; https://www.graphable.ai/blog/patient-journey-mapping)

커뮤니티는 EHR 사용과 관련된 프라이버시·보안 우려를 풀기 위해 열심히 애쓰고 있습니다. 어떤 접근은 방대한 환자 데이터 집합을 이용해 질병·증상·치료 결과에 대한 통계 정보를 추출합니다 [32]. 이런 통계적 관계와 그 가중치는 지식 그래프에 저장하되, 실제 환자 데이터는 저장하지 않습니다. 이렇게 하면 프라이버시 보장이 강화됩니다 [35].

또 다른 접근은 비민감 데이터, 익명화된 데이터, 실험 결과, 실제 데이터에서 도출한 통계 정보를 바탕으로, 신원 식별 정보를 제거한(deidentified) 일반적 임상 지식 그래프를 구축하는 것입니다. 환자 EHR 데이터는 꼭 필요할 때만, 환자 동의를 얻어 사용합니다.

Albertos Santos 등 [31]의 **임상 지식 그래프(Clinical Knowledge Graph, CKG)** 는 지식 그래프를 핵심에 둔 플랫폼입니다. CKG는 33개 노드 레이블을 51개 관계 유형으로 연결해 데이터를 조화시키고 통합합니다(그림 4.13 참고). 이를 통해 변경된 기능에 대한 통찰을 얻거나, 조절된 단백질에 대한 약물을 제안하거나, 있을 수 있는 교란 요인(confounding factor)을 드러내는 질의가 가능해집니다.

![](images/86fd617f2f1947e03ee60befd24870f0f2b5c33c7610c578a4707402b1ce07cf.jpg)

그림 4.13 임상 지식 그래프의 스키마(https://www.nature.com/articles/s41587-021-01145-6)

CKG는 Neo4j 형식으로 https://github.com/MannLabs/CKG 에서 제공됩니다. 우리는 이 책을 쓰면서 사용 가능한 최신 버전으로 업그레이드했으며, 다음 코드를 사용해 제공된 덤프(https://mng.bz/oZQZ)를 여러분의 컴퓨터로 가져올 수 있습니다.

```text
Listing 4.21 Importing the CKG database
Add the following line to the neo4j.conf file
dbms.databases.seed_from_uri_providers=URLConnectionSeedProvider
then run the following
CREATE DATABASE ckg OPTIONS { existingData: "use",
➥seedUri: "https://mng.bz/oZQZ"}
```

CKG 같은 지식 그래프에서 가치 있는 통찰을 얼마나 간단히 뽑아낼 수 있는지 보여 주는 질의를 하나 실행해 보겠습니다. 여러분의 현재 임상 연구가 어떤 단백질 집합을 표적으로 삼고 있고, 임상시험을 계획 중이라고 가정합시다. 시험 대상으로 적합한 환자 일부가 심근병증(cardiomyopathy) 관련 질병을 앓고 있습니다. 여러분은 표적 단백질과 심근병증 관련 질병 사이에 알려진 연관성이 있는지 확인하고 싶습니다.

```text
Listing 4.22 Finding known protein–disease associations
WITH  Defines the list of proteins
['A1BG~P04217','A2M~P01023','ACACB~O00763',  we are interested in
'ACTC1~P68032','ADIPOQ~Q15848','AGT~P01019',
'AIFM2~Q9BRQ8','APOA2~V9GYM3'] as proteins,  ← Defines the minimum
3 as minScore,  ← association strength threshold
"DOID:0050700" as parentDisease  ← Defines the target
MATCH (protein:Protein)-[r]-(disease:Disease)  ← disease DOID
WHERE (
(protein.name+"~"+protein.id) IN proteins) AND  Matches any type of protein–
toFloat(r.score)> minScore AND  disease association
((disease)-[:HAS_PARENT*0..]->(:Disease {id: parentDisease})) ←
RETURN
(protein.name+"~"+protein.id) AS node1,  Matches any disease related
disease.name+" <"+disease.id+">" AS node2,  to cardiomyopathy
r.score AS weight, type(r) AS type,
r.source AS source
ORDER BY weight DESC
```

이 질의는 관심 단백질 목록과 최소 연관 강도 임계값(`minScore`), 그리고 목표 질병의 DOID를 정의한 뒤, 어떤 유형이든 단백질–질병 연관을 찾습니다. `HAS_PARENT*0..` 는 심근병증과 관련된 모든 하위 질병까지 거슬러 올라가 매칭한다는 뜻입니다. 표 4.5의 결과는 여러 유형의 내인성 심근병증과 강하게 연관된 특정 단백질을 보여 줍니다. 이 발견은 이후 조사를 이어 갈 구체적인 출발점을 제공합니다.

> **참고** DOID는 질병 온톨로지 식별자(disease ontology identifier)로, www.diseaseontology.org 에 정의돼 있습니다.

표 4.5 Listing 4.22 질의에서 나온 단백질–질병 연관성
<table><tr><td>nodel</td><td>node2</td><td>weight</td><td>type</td><td>source</td></tr><tr><td>&quot;ACTC1~P68032&quot;</td><td>&quot;intrinsic cardiomyopathy &lt;DOID:0060036&gt;&quot;</td><td>5</td><td>&quot;ASSOCIATED WITH&quot;</td><td>&quot;DISEASES&quot;</td></tr><tr><td>&quot;ACTC1~P68032&quot;</td><td>&quot;left ventricular noncompaction &lt;DOID:0060480&gt;&quot;</td><td>5</td><td>&quot;ASSOCIATED WITH&quot;</td><td>&quot;DISEASES&quot;</td></tr><tr><td>&quot;ACTC1~P68032&quot;</td><td>&quot;familial hypertrophic cardiomyopathy &lt;DOID:0080326&gt;&quot;</td><td>5</td><td>&quot;ASSOCIATED WITH&quot;</td><td>&quot;DISEASES&quot;</td></tr><tr><td>&quot;ACTC1~P68032&quot;</td><td>&quot;hypertrophic cardiomyopathy &lt;DOID:11984&gt;&quot;</td><td>5</td><td>&quot;ASSOCIATED WITH&quot;</td><td>&quot;DISEASES&quot;</td></tr><tr><td>&quot;ACTC1~P68032&quot;</td><td>&quot;dilated cardiomyopathy &lt;DOID:12930&gt;&quot;</td><td>5</td><td>&quot;ASSOCIATED WITH&quot;</td><td>&quot;DISEASES&quot;</td></tr><tr><td>&quot;ACTC1~P68032&quot;</td><td>&quot;restrictive cardiomyopathy &lt;DOID:397&gt;&quot;</td><td>5</td><td>&quot;ASSOCIATED WITH&quot;</td><td>&quot;DISEASES&quot;</td></tr></table>

---

### 4.4.1 LLM 기반 임상 의사결정 지원 분석 — 발견을 진료로 잇다

임상 지식 그래프는 환자·치료·결과 사이의 복잡한 관계를 담고 있어 신중한 해석이 필요합니다. LLM은 임상의와 연구자가 다차원 임상 데이터를 일관된 치료 권고와 연구 방향으로 종합하도록 도울 수 있습니다. ACTC1 단백질과 여러 심근병증 사이의 강한 연관성을 식별한 우리의 CKG 분석을 이용하면, LLM은 임상적 맥락과 의사결정 지원을 제공할 수 있습니다.

```text
Listing 4.23 Example LLM clinical analysis prompt
You are a clinical informatics specialist analyzing protein-disease associations from electronic health records integrated with biomedical knowledge graphs. Provide clinical decision support based on these findings:

CLINICAL SCENARIO: Research study targeting proteins in patients with cardiomyopathy-related diseases

KNOWLEDGE GRAPH FINDINGS:
- ACTC1~P68032 shows strong associations (score: 5.0) with:
* Intrinsic cardiomyopathy (DOID:0060036)
* Left ventricular noncompaction (DOID:0060480)
* Familial hypertrophic cardiomyopathy (DOID:0080326)
* Hypertrophic cardiomyopathy (DOID:11984)
* Dilated cardiomyopathy (DOID:12930)
* Restrictive cardiomyopathy (DOID:397)

TARGET PROTEIN LIST:
➥['A1BG~P04217','A2M~P01023','ACACB~O00763','ACTC1~P68032','ADIPOQ~Q15848',
➥'AGT~P01019','AIFM2~Q9BRQ8','APOA2~V9GYM3']

CLINICAL REQUEST:
1. Interpret the clinical significance of ACTC1's broad cardiomyopathy associations
2. What does this suggest about patient stratification for clinical trials?
3. Identify potential safety considerations for the research study
4. Recommend additional screening or monitoring protocols
5. Suggest companion biomarkers or genetic testing approaches
6. Propose modifications to inclusion/exclusion criteria based on these findings
```

이 프롬프트는 LLM에게 임상 정보학 전문가 역할을 맡기고, ACTC1의 광범위한 심근병증 연관성의 임상적 의미 해석, 임상시험을 위한 환자 층화(stratification) 시사점, 잠재적 안전성 고려사항, 추가 선별·모니터링 프로토콜 권고, 동반 바이오마커·유전자 검사 접근 제안, 그리고 이 발견에 기반한 포함/제외 기준 수정 제안을 요청합니다.

이 LLM 지원 분석은 지식 그래프에서 나온 계산적 발견을, 실제 임상 의사결정으로 옮기는 데 도움을 줍니다. 환자 진료와 연구 프로토콜에 정보를 제공하는 것입니다. 결과가 너무 길어 여기에 싣지는 않았으니, 이 프롬프트를 여러분이 좋아하는 LLM 도구로 직접 실습해 보길 권합니다.

> **참고** 우리는 의사와 연구자의 적절한 해석 없이 LLM의 결과를 그대로 사용하는 것을 권하지 않습니다. 우리의 목적은 LLM이 거대한 지식 그래프에서 뽑아낸 복잡한 데이터를 해석 가능한 통찰로 바꾸는 방법을 보여 주는 것입니다. 기억하세요. 우리의 목표는 지능형 자문 시스템으로 사람을 대체하는 것이 아니라, 사람의 역량을 키워 주는 것입니다!

---

## 요약

이 장의 핵심을 다시 짚어 보겠습니다.

- 구조화된 데이터 소스로 만든 지식 그래프는 이질적인 데이터셋을 체계적으로 통합해야 합니다. 이 과정에는 엔티티 해소(entity resolution), 스키마 정렬(schema alignment), 데이터 품질 검증이 포함되며, 이를 통해 일관되고 질의 가능한 지식 표현을 만들어 냅니다.
- WCC(약하게 연결된 컴포넌트)와 Louvain 같은 군집화 알고리즘은 어떤 다중 소스 지식 그래프든 그 전체 구조와 커뮤니티 조직에 대한 통찰을 제공합니다.
- 밀도, 전도도, 가장 큰 연결 컴포넌트의 상대적 크기 같은 부분 그래프 분석 측정값은 도메인을 가리지 않고 지식 그래프의 품질과 완결성을 평가하는 정량적 방법을 제공합니다.
- DWPC(차수 가중 경로 수) 같은 고급 경로 기반 지표는 단순한 경로 세기보다 훨씬 정교하게 관계 패턴과 엔티티 관련성을 분석할 수 있게 해 줍니다.
- Hetionet, PPI 네트워크, CKG 같은 포괄적 지식 그래프는 통합 기법과 분석 접근을 시연하기에 좋은 시험대(testbed)가 됩니다.
- 지식 그래프 분석 결과에 대한 LLM 기반 해석은 정량적 지표를, 실행 가능한 도메인 특화 통찰과 연구 가설로 옮기는 데 도움을 줍니다.

---

## 핵심 용어 해설

| 용어 (영문) | 뜻 |
| --- | --- |
| 지식 그래프 (Knowledge Graph, KG) | 사실을 노드와 관계로 표현한 검증된 지식 저장소 |
| 지능형 자문 시스템 (Intelligent Advisor System, IAS) | 지식 그래프 등을 활용해 사람의 의사결정을 돕는 조언 시스템 |
| CRISP-DM | 데이터 마이닝 프로젝트의 표준 단계를 정의한 절차 모델 |
| 다중 오믹스 (multi-omic) | 유전체·전사체·단백질체 등 여러 '오믹스' 데이터를 함께 쓰는 분석 접근 |
| 유전체 (genome) | 한 생물의 유전 정보 전체(대부분 DNA) |
| 전사체 (transcriptome) | 단백질체 합성을 지시하는 RNA 분자들의 집합 |
| 단백질체 (proteome) | 세포가 합성한 기능성 단백질 전체(유전체 발현의 최종 산물) |
| 단백질–단백질 상호작용 (PPI) | 단백질끼리의 물리적·기능적 상호작용, 또는 그 네트워크 |
| DisGeNET | 질병 관련 지식(단백질–질병 연관 등)을 모은 공개 플랫폼 |
| 질병 온톨로지 (Disease Ontology) | 질병을 범주·하위 범주로 체계화한 온톨로지 |
| UMLS | 미국 국립의학도서관의 통합 의학 용어 체계 |
| 단일 분할 그래프 (monopartite graph) | 한 종류의 노드끼리 연결된 그래프 |
| 이분 그래프 (bipartite graph) | 두 종류의 노드가 서로 연결된 그래프 |
| 질병 경로 (disease pathway) | 특정 질병과 연관된 단백질 집합으로 정의되는 PPI의 부분 그래프 |
| WCC (Weakly Connected Component) | 그래프 안의 끊긴 부분 그래프를 찾는 커뮤니티 탐지 알고리즘 |
| Louvain 모듈성 | 모듈성 점수를 최대화해 커뮤니티를 찾는 빠른 군집화 알고리즘 |
| 모듈성 (modularity) | 노드가 커뮤니티로 얼마나 잘 나뉘었는지를 나타내는 점수 |
| 프로젝션 (projection) | 분석을 위해 메모리에 올린 그래프의 사본 |
| GDS (Graph Data Science) | Neo4j의 그래프 분석 알고리즘 라이브러리 |
| 가장 큰 경로 컴포넌트 | 질병 경로에서 가장 큰 연결 컴포넌트가 차지하는 비율 |
| 밀도 (density) | 부분 그래프에서 가능한 엣지 대비 실제 엣지의 비율 |
| 전도도 (conductance) | 부분 그래프가 나머지 그래프로부터 얼마나 독립적인지 나타내는 값 |
| 경로 경계 (pathway boundary, $B_d$) | 질병 경로 안과 밖을 잇는 엣지들의 집합 |
| Hetionet | 29개 공개 자원을 통합한 이질적 생명의학 지식 그래프 |
| 메타그래프 (metagraph) | 노드·관계의 유형으로 데이터베이스 구조를 묘사한 것 |
| 메타경로 (metapath) | 노드·관계의 유형 시퀀스로 잠재적 경로를 묘사한 것 |
| 경로 수 (Path Count, PC) | 특정 메타경로에 해당하는 경로의 개수(각 경로 값은 1) |
| DWPC (Degree-Weighted Path Count) | 차수로 가중해 경로 관련성을 평가하는 지표 |
| PDP (Path-Degree Product) | 경로를 따른 차수들을 $-w$ 승 해 곱한 개별 경로 값 |
| 감쇠 지수 (damping exponent, $w$) | 연결 많은 노드를 지나는 경로의 가치를 깎는 정도를 정하는 값 |
| GO 과정 (Gene Ontology process) | 유전자 온톨로지에서 정의한 생물학적 과정 |
| GWAS | 전장 유전체 연관 연구, 기존 지식에 덜 편향된 질병–유전자 연관 근거 |
| 정밀 의료 (precision medicine) | 개인의 유전·환경·생활 습관을 반영해 예방·치료법을 정하는 접근 |
| EHR (Electronic Health Record) | 전자 건강 기록, 여러 임상의의 환자 진료 정보 |
| 환자 여정 지도 (patient journey mapping) | 환자가 의료 서비스에 들어오고 겪고 나가는 과정을 그린 지도 |
| CKG (Clinical Knowledge Graph) | 33개 노드·51개 관계 유형을 통합한 임상 지식 그래프 플랫폼 |
| DOID | 질병 온톨로지 식별자 |

---

## References (참고 문헌)

원문의 참고 문헌 [1]\~[35]은 본문 각 지점에 번호로 인용돼 있습니다. 상세 서지 정보는 원서의 References 절을 참조하십시오.

