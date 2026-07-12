# Graph Flashcards

## Card 1

**Front:** 서로 다른 구조화된 데이터 소스를 일관된 지식 그래프(KG)로 통합할 때, 데이터의 의미적 일관성을 유지하는 과정을 무엇이라 하는가?

**Back:** 시맨틱 통합(Semantic Integration)이라고 한다.

---

## Card 2

**Front:** 지식 그래프의 품질과 그 하위 애플리케이션의 신뢰성을 결정짓는 핵심 요소는 무엇인가?

**Back:** 기저에 깔린 지식 표현의 신뢰성과 데이터의 품질이다.

---

## Card 3

**Front:** 지식 그래프 구축 과정에서 데이터 소스 간의 로컬 스키마와 온톨로지의 참조 스키마를 연결하는 것을 무엇이라 하는가?

**Back:** 매핑(Mapping)이라고 한다.

---

## Card 4

**Front:** HPO(Human Phenotype Ontology)의 주요 목적은 무엇인가?

**Back:** 표현형 특성에 이름을 붙이기 위한 표준화된 어휘와 질병 주석 데이터를 제공하는 것이다.

---

## Card 5

**Front:** 개별적인 정보 조각들이 하나의 일관된 뷰로 융합되어 의미 있게 표현된 지식 그래프의 특징은?

**Back:** 통일되고(unified), 근거가 확실하며(well-grounded), 유의미한(meaningful) 표현이다.

---

## Card 6

**Front:** 지식 그래프 구축 시 데이터의 구문론적(Syntax) 차이의 예시는 무엇인가?

**Back:** 날짜 표기 방식이 '2022-08-09'와 '9 August 2022'처럼 서로 다른 경우이다.

---

## Card 7

**Front:** 의료 데이터 통합에서 'PE'가 신체 검사(physical examination) 또는 폐색전증(pulmonary embolism)을 의미할 수 있는 문제는 무엇인가?

**Back:** 동일한 약어가 서로 다른 개념을 정의하는 중의성 문제이다.

---

## Card 8

**Front:** 데이터 소스에서 온톨로지로의 주석(Annotation) 처리가 갖는 의미는 무엇인가?

**Back:** 서로 다른 기원의 데이터 요소들을 공통된 개념으로 묶어주는 역할을 한다.

---

## Card 9

**Front:** 표현형 이상(Phenotypic abnormalities)이란 무엇을 의미하는가?

**Back:** 유전적 변이나 환경적 영향으로 인해 전형적인 인간의 특징에서 벗어난 관찰 가능한 물리적 또는 생화학적 특성이다.

---

## Card 10

**Front:** 본 도서에서 제시하는 지식 그래프 구축 프로세스의 기반이 되는 모델은?

**Back:** CRISP-DM 모델이다.

---

## Card 11

**Front:** 지식 그래프 구축을 위한 CRISP-DM 모델의 첫 번째 단계는?

**Back:** 비즈니스 이해(Business understanding)이다.

---

## Card 12

**Front:** 데이터 이해(Data understanding) 단계에서 수행하는 핵심 활동은?

**Back:** 기존 데이터에서 정의된 목표와 관련된 데이터를 선택하고 분석하는 것이다.

---

## Card 13

**Front:** 지식 그래프 구축 파이프라인에서 '데이터 준비(Data preparation)' 단계의 역할은?

**Back:** 현재 범위 내의 관련 데이터를 가져와 다음 단계인 모델 생성을 유도하는 것이다.

---

## Card 14

**Front:** 모델링(Modeling) 단계가 머신러닝에서 의미하는 바는?

**Back:** 머신러닝 작업을 위한 알고리즘을 정의하는 것을 의미한다.

---

## Card 15

**Front:** 표현형(Phenotype)의 정의는 무엇인가?

**Back:** 한 개인이 나타내는 모든 표현형적 특징의 총합이다.

---

## Card 16

**Front:** 질병(Disease)을 규정하는 네 가지 특징은 원인, 시간적 경과, 표현형 특징, 그리고 무엇인가?

**Back:** 특정 치료에 대한 특징적인 반응이다.

---

## Card 17

**Front:** 임상 의사가 희귀 질환을 진단할 때 지식 베이스가 갖추어야 할 두 가지 주요 기능은?

**Back:** 표현형 도메인의 문맥적 설명과 표현형 이상과 질병 간의 관계 데이터이다.

---

## Card 18

**Front:** 표준화된 표현형 정보를 담고 있는 hpo.owl 파일의 주요 데이터 모델 형식은?

**Back:** RDF/XML 형식이다.

---

## Card 19

**Front:** RDF(Resource Description Framework)에서 각 문장을 구성하는 세 가지 요소는?

**Back:** 주어(Subject), 술어(Predicate), 목적어(Object)이다.

---

## Card 20

**Front:** Turtle(Terse RDF Triple Language) 구문에서 'blank node'를 나타내는 기호는?

**Back:** _: (언더스코어와 콜론)으로 시작한다.

---

## Card 21

**Front:** HPO 데이터셋 중 phenotype.hpoa 파일(TSV 형식)이 제공하는 정보는?

**Back:** 질병과 연관된 표현형적 특징들의 주석(Annotation) 데이터이다.

---

## Card 22

**Front:** HPOA 파일에서 'evidence' 필드의 'PCS'가 의미하는 바는?

**Back:** 출판된 임상 연구(published clinical study)를 의미한다.

---

## Card 23

**Front:** HPOA 파일의 'frequency' 필드에서 '30/30'이 나타내는 의미는?

**Back:** 해당 질환을 가진 환자 30명 중 30명 모두에게서 해당 표현형 이상이 발견되었다는 뜻이다.

---

## Card 24

**Front:** RDF와 LPG(Labeled Property Graph) 중 W3C에 의해 규제되는 표준 프레임워크는?

**Back:** RDF(Resource Description Framework)이다.

---

## Card 25

**Front:** LPG 모델이 데이터 저장 및 접근 효율성을 보장하는 방법은 무엇인가?

**Back:** 노드와 관계에 키-값 쌍(key-value pairs) 형태의 구조화된 정보를 연결하는 것이다.

---

## Card 26

**Front:** RDF에서 특정 관계 그룹을 하나의 엔티티로 취급하여 문맥 정보를 제공할 수 있게 하는 기능은?

**Back:** 명명된 그래프(Named Graphs)이다.

---

## Card 27

**Front:** 관계(Edge)에 직접 속성을 추가할 수 있도록 설계 중인 RDF 확장 사양은?

**Back:** RDF-star ($RDF^{\ast}$)이다.

---

## Card 28

**Front:** Neo4j에서 RDF 데이터를 사용하여 기본 추론을 실행할 수 있게 돕는 플러그인은?

**Back:** Neosemantics (n10s) 플러그인이다.

---

## Card 29

**Front:** OWL(Web Ontology Language)의 주된 목표는 무엇인가?

**Back:** RDF의 시맨틱 정보를 풍부하게 하여 표현력 있는 클래스 및 속성 정의를 지원하는 것이다.

---

## Card 30

**Front:** 임상 주석 데이터처럼 관계의 출처, 날짜 등 메타데이터가 풍부한 정보를 모델링할 때 더 적합한 기술은?

**Back:** LPG(Labeled Property Graph) 기술이다.

---

## Card 31

**Front:** RDF에서 관계(Edge)에 데이터를 모델링하기 위해 새로운 개념(예: Annotation 노드)을 만드는 접근 방식은?

**Back:** n항 관계(n-ary relations) 모델링이다.

---

## Card 32

**Front:** SPARQL 쿼리에서 명명된 그래프 내의 트리플을 검색할 때 사용하는 키워드는?

**Back:** GRAPH 키워드이다.

---

## Card 33

**Front:** Neo4j에서 HPO 데이터베이스를 생성하는 Cypher 명령어는?

**Back:** CREATE DATABASE hpo IF NOT EXISTS 이다.

---

## Card 34

**Front:** Neosemantics 설정 중 `handleVocabUris: "IGNORE"`의 역할은?

**Back:** 데이터 임포트 단계에서 네임스페이스(Namespaces)를 무시하도록 설정하는 것이다.

---

## Card 35

**Front:** Neosemantics에서 관계 유형을 대문자로 인코딩하도록 설정하는 옵션은?

**Back:** applyNeo4jNaming: True 이다.

---

## Card 36

**Front:** Neo4j로 외부 RDF 파일을 불러올 때 사용하는 Neosemantics 함수는?

**Back:** n10s.rdf.import.fetch() 이다.

---

## Card 37

**Front:** HPOA 파일(TSV)을 로드할 때 탭 구분자를 지정하는 Cypher 구문은?

**Back:** FIELDTERMINATOR '\\t' 이다.

---

## Card 38

**Front:** Cypher에서 `MERGE` 명령어의 주된 역할은 무엇인가?

**Back:** 노드나 관계가 존재하면 매치하고, 없으면 새로 생성하는 것이다.

---

## Card 39

**Front:** Cypher의 `FOREACH` 문을 사용하여 조건부로 속성을 설정하는 이유는 무엇인가?

**Back:** 데이터가 결측치(null)인 경우 속성 생성을 방지하여 스크립트의 복원력을 높이기 위해서이다.

---

## Card 40

**Front:** 지식 그래프 구축의 마지막 단계에서 불필요한 노드와 관계를 제거하는 활동은?

**Back:** 데이터 클리닝(Data cleaning)이다.

---

## Card 41

**Front:** 지식 그래프의 강력한 도구 중 하나로, 논리적 규칙을 바탕으로 암시적 정보를 도출하는 과정을 무엇이라 하는가?

**Back:** 추론(Inference) 또는 연역적 추론(Deductive reasoning)이라고 한다.

---

## Card 42

**Front:** HPO의 계층적 구조에서 'subclass' 관계를 활용하면 어떤 이점이 있는가?

**Back:** 상위 개념에 대한 쿼리를 통해 하위의 구체적인 표현형 정보를 포함한 결과를 얻을 수 있다.

---

## Card 43

**Front:** Neosemantics에서 특정 노드의 모든 하위 클래스(1~3단계 깊이)를 찾는 Cypher 패턴은?

**Back:** -[:SUBCLASSOF*1..3]-> 이다.

---

## Card 44

**Front:** 임상 의사가 지식 그래프를 통해 희귀 질환을 진단하는 일반적인 절차는?

**Back:** 환자의 표현형 특징을 입력하여 관련 가능성이 높은 질환을 검색하는 것이다.

---

## Card 45

**Front:** 지식 그래프 내에서 'Type 1 diabetes'가 질환이면서 동시에 표현형 특징으로 분류되는 이유는 무엇인가?

**Back:** 문맥에 따라 다른 의미를 가질 수 있으며, 이를 위해 서로 다른 식별자(ID)를 사용한다.

---

## Card 46

**Front:** LPG 모델에서 관계의 상세 정보를 저장하는 방식은?

**Back:** 관계(Edge) 내부에 직접 키-값 쌍의 속성으로 저장한다.

---

## Card 47

**Front:** RDF-star에서 트리플 자체를 주어로 사용하여 속성을 기술하는 구문 형식은?

**Back:** << :Subject :Predicate :Object >> :property :value . 형태이다.

---

## Card 48

**Front:** 지식 그래프 구축 시 '비즈니스 이해' 단계에서 정의해야 할 '페르소나'의 예는?

**Back:** 질병을 진단하고 치료하는 임상 의사(Clinician)이다.

---

## Card 49

**Front:** 데이터 정제 시 `DETACH DELETE n` 명령어를 사용하는 이유는?

**Back:** 노드와 그 노드에 연결된 모든 관계를 동시에 삭제하기 위해서이다.

---

## Card 50

**Front:** 온톨로지가 시맨틱 이질성(semantic heterogeneity)을 해결하는 방법은?

**Back:** 서로 다른 소스의 데이터를 표준화된 어휘와 스키마로 중개하는 역할을 한다.

---

## Card 51

**Front:** RDF 데이터 모델의 가장 큰 장점은 무엇인가?

**Back:** 지식 표현에 집중하며 온톨로지 구축에 매우 적합하다는 점이다.

---

## Card 52

**Front:** LPG 접근 방식의 가장 큰 장점은 무엇인가?

**Back:** 빠른 그래프 데이터 탐색과 경로 분석이 가능하며 저장 효율성이 높다는 점이다.

---

## Card 53

**Front:** 데이터 소스 간의 구문(Syntax) 차이와 의미(Meaning) 차이 중 무엇이 더 해결하기 어려운가?

**Back:** 데이터의 의미(Meaning) 차이를 해결하는 것이 더 복잡한 도전 과제이다.

---

## Card 54

**Front:** HPO 어휘를 Neo4j에 로드한 후 수행하는 '노드 강화(Enriching nodes)' 단계의 목적은?

**Back:** 각 노드에 사람이 읽기 쉬운 레이블이나 ID 속성을 추가하여 탐색을 용이하게 하는 것이다.

---

## Card 55

**Front:** SPARQL-star 쿼리에서 관계 속성을 검색할 때 사용하는 특수 기호는?

**Back:** | (파이프 기호)와 변수 할당 구문을 사용한다.

---

## Card 56

**Front:** CRISP-DM 모델을 지식 그래프에 적용했을 때, 모델링(Modeling) 단계의 구체적 대상은?

**Back:** 지식 그래프 모델의 생성 및 업데이트(KG model creation/update)이다.

---

## Card 57

**Front:** HPO 주석 데이터 로드 시 `MERGE (dis:Resource:HpoDisease {id: row[0]})` 구문의 의미는?

**Back:** ID를 기반으로 질환 노드를 생성하거나 이미 존재하면 가져오는 것이다.

---

## Card 58

**Front:** 질환과 표현형 관계 설정 시 `MERGE (dis)-[:HAS_PHENOTYPIC_FEATURE]->(phe)`는 어떤 작업을 수행하는가?

**Back:** 질환 노드와 표현형 노드 사이에 명시된 관계를 생성하거나 확인한다.

---

## Card 59

**Front:** 데이터 클리닝 쿼리에서 `NOT 'HpoPhenotype' in labels(n)` 조건의 목적은?

**Back:** 사용자가 정의한 핵심 도메인 노드(표현형 노드)가 아닌 리소스만 골라 삭제하기 위함이다.

---

## Card 60

**Front:** 임상 의사가 진단 중 겪는 '그레이 영역(gray areas)'을 지식 그래프가 어떻게 도울 수 있는가?

**Back:** 불확실한 증상들의 연계 정보를 제공하여 알려지지 않은 희귀 질환을 유추할 수 있게 한다.

---

## Card 61

**Front:** RDF의 '재구체화(Reification)' 기술이 실무에서 덜 사용되는 이유는?

**Back:** 명명된 그래프나 n항 관계에 비해 확장성과 유지보수성이 떨어지기 때문이다.

---

## Card 62

**Front:** Neo4j에서 대량의 데이터를 주기적으로 반복 처리할 때 사용하는 라이브러리 함수는?

**Back:** apoc.periodic.iterate() 이다.

---

## Card 63

**Front:** HPOA 파일 형식에서 'modifier' 필드가 제공하는 정보의 예는?

**Back:** 발병 연령(age of onset)과 같은 추가적인 맥락 정보이다.

---

## Card 64

**Front:** 온톨로지 기반 추론을 통해 얻은 결과가 갖는 시맨틱적 가치는?

**Back:** 데이터에 직접 명시되지 않은 숨겨진 의미적 연결을 발견할 수 있게 한다.

---

## Card 65

**Front:** 시맨틱 통합 전략에서 온톨로지는 어떤 역할을 수행하는가?

**Back:** 들어오는 데이터의 참조 스키마이자 어휘(Vocabulary) 역할을 한다.

---

## Card 66

**Front:** 지식 그래프를 LLM과 결합하기 전에 반드시 선행되어야 하는 과정은?

**Back:** 이질적인 데이터 소스로부터 일관된 지식 그래프를 구축하는 것이다.

---

## Card 67

**Front:** 데이터 정합성(Integrity) 확인과 정확한 엔티티 매칭이 중요한 이유는 무엇인가?

**Back:** 다운스트림 애플리케이션의 품질이 기본 지식 표현의 신뢰성에 의존하기 때문이다.

---

## Card 68

**Front:** Cypher 쿼리에서 `collect(phe.label)` 함수는 무엇을 수행하는가?

**Back:** 일치하는 모든 표현형 레이블을 하나의 리스트로 묶어 반환한다.

---

## Card 69

**Front:** 임상 현장에서 환자의 의료 기록(EHR)과 지식 그래프를 연동할 때의 이점은?

**Back:** 환자의 증상을 HPO와 같은 표준 용어로 매핑하여 정밀 진단 지원이 가능하다.

---

## Card 70

**Front:** 온톨로지 내에서 '내분비계 이상'의 하위 클래스를 검색하는 행위의 궁극적인 목적은?

**Back:** 갑상선 등 특정 기관과 관련된 보다 구체적인 질환 주석 정보를 찾기 위함이다.

---

## Card 71

**Front:** 지식 그래프 기술 선택 시 '정답이 하나가 아니다'라는 말의 의미는 무엇인가?

**Back:** 사용 사례의 목표와 가용 데이터의 특성에 따라 가장 적합한 기술이 달라질 수 있음을 의미한다.

---

## Card 72

**Front:** RDF-star의 도입이 지식 그래프 기술 생태계에 주는 영향은?

**Back:** RDF와 LPG 기술 간의 간극을 좁히고 두 모델의 장점을 결합할 수 있는 가능성을 열어준다.

---

## Card 73

**Front:** HPOA 파일의 'biocuration' 필드는 어떤 정보를 포함하는가?

**Back:** 주석을 작성한 연구 센터나 사용자 정보 및 작성 날짜를 포함한다.

---

## Card 74

**Front:** 지식 그래프 구축 파이프라인에서 '비즈니스 목표'는 어떻게 정의되는가?

**Back:** 해당 지식 그래프가 해결하고자 하는 현실 세계의 구체적인 문제로 정의된다.

---

## Card 75

**Front:** 의료 지식 그래프에서 '근거 수준(level of evidence)' 정보가 중요한 이유는?

**Back:** 임상 의사가 진단 시 해당 정보의 신뢰도를 판단하는 기준이 되기 때문이다.

---

## Card 76

**Front:** 본 도서의 3장에서 구축한 지식 그래프가 최종적으로 지원하는 활동은?

**Back:** 임상 의사의 희귀 질환 진단 및 치료 결정을 지원하는 활동이다.

---
